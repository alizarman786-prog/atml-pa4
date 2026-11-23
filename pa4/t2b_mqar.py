"""Task 2B: MQAR with dense / sliding-window / local+dilated / ELU+1 linear / hybrid attention.
python -m pa4.t2b_mqar --runs all            (train + evaluate every required config x seeds)
python -m pa4.t2b_mqar --sweep               (forward-only inference sweep)
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import OUT, DATA_REPO, DATA, read_jsonl, write_jsonl, save_json, load_json, env_info, summarize_ms

R = OUT / "task2" / "B"
CACHE = Path(os.environ.get("PA4_CACHE", "/tmp/pa4cache"))
T = 256; VOCAB = 256; DM = 128; NH = 4; HD = 32; W = 64
SEEDS = (6304, 6305, 6306)
UPDATES = 4500; BATCH = 32; PEAK_LR = 8e-4; WD = 0.01; WARM = 200; FINAL_FRAC = 0.15
CKPTS = (500, 1000, 1500, 2000, 3000, 4500)
if os.environ.get("PA4_QUICK"): UPDATES, CKPTS = 100, (50, 100)
BINS = ((0, 32), (32, 64), (64, 128), (128, 256))

# (name, attention, train K, train regime)
CONFIGS = [
    ("dense_K4_mixed", "dense", 4, "mixed"), ("dense_K8_mixed", "dense", 8, "mixed"),
    ("dense_K16_mixed", "dense", 16, "mixed"), ("dense_K16_near", "dense", 16, "near"),
    ("window_K4_near", "window", 4, "near"), ("window_K8_near", "window", 8, "near"),
    ("window_K16_near", "window", 16, "near"), ("window_K16_mixed", "window", 16, "mixed"),
    ("linear_K4_mixed", "linear", 4, "mixed"), ("linear_K8_mixed", "linear", 8, "mixed"),
    ("linear_K16_mixed", "linear", 16, "mixed"),
    ("dilated_K4_mixed", "dilated", 4, "mixed"), ("dilated_K16_mixed", "dilated", 16, "mixed"),
    # B3 intervention (default): 2 linear heads + 2 local softmax heads (window 16) with matched state bytes
    ("hybrid_K16_mixed", "hybrid", 16, "mixed"),
]


def _gen():
    sys.path.insert(0, str(DATA_REPO / "scripts"))
    import mqar_generator
    return mqar_generator


# ---------------- data -----------------
def to_arrays(exs, K):
    n = len(exs)
    tok = np.array([e["tokens"] for e in exs], np.int64)
    qp = np.array([e["query_positions"] for e in exs], np.int64)
    ans = np.array([e["answer_ids"] for e in exs], np.int64)
    src = np.array([e["source_positions"] for e in exs], np.int64)
    return dict(tokens=tok, qpos=qp, ans=ans, src=src, dist=qp - src, K=K)


def _mk(args):
    split, k, regime, lo, hi = args
    g = _gen()
    return [g.make_example(split, k, regime, i) for i in range(lo, hi)]


def train_stream(K, regime, n):
    f = CACHE / f"mqar_train_K{K}_{regime}_{n}.npz"
    if f.exists():
        d = dict(np.load(f)); d["K"] = K; return d
    CACHE.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    chunks = [("train", K, regime, s, min(s + 4000, n)) for s in range(0, n, 4000)]
    with Pool(min(4, os.cpu_count() or 1)) as p:
        exs = [e for part in p.map(_mk, chunks) for e in part]
    d = to_arrays(exs, K)
    np.savez(f, **{k: v for k, v in d.items() if k != "K"})
    return d


def frozen(split):
    f = DATA / "mqar" / f"mqar_{'test' if split == 'test' else 'development'}.jsonl.gz"
    rows = read_jsonl(f); g = _gen()
    for r in rows[:50]: g.validate(r)
    out = {}
    for K in (4, 8, 16):
        for reg in ("near", "far", "mixed"):
            ex = [r for r in rows if r["K"] == K and r["regime"] == reg]
            out[(K, reg)] = to_arrays(ex, K)
    return out


# ---------------- model -----------------
def sinusoid(T_, d):
    pos = torch.arange(T_)[:, None].float(); i = torch.arange(0, d, 2).float()
    ang = pos / (10000 ** (i / d)); pe = torch.zeros(T_, d)
    pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang); return pe


def band_mask(T_, window, device):
    i = torch.arange(T_, device=device)[:, None]; j = torch.arange(T_, device=device)[None]
    return (j <= i) & (j > i - window)


def dilated_mask(T_, device, n=64, step=4):
    i = torch.arange(T_, device=device)[:, None]; j = torch.arange(T_, device=device)[None]
    off = i - j
    return (off >= 0) & (off % step == 0) & (off < n * step)


def phi(x): return F.elu(x) + 1


def linear_attention(q, k, v, eps=1e-6, chunk=64):
    """Causal ELU+1 linear attention, chunked: intra-chunk masked quadratic + inter-chunk recurrent state
    S (d_f x d_v) and z (d_f). Exactly equals the per-token recurrence; persistent state = S,z only."""
    Bq, Hq, L, d = q.shape
    q, k = phi(q.float()), phi(k.float()); v = v.float()
    S = torch.zeros(Bq, Hq, d, v.shape[-1], device=q.device); z = torch.zeros(Bq, Hq, d, device=q.device)
    out = []
    for s in range(0, L, chunk):
        qc, kc, vc = q[:, :, s:s + chunk], k[:, :, s:s + chunk], v[:, :, s:s + chunk]
        n = qc.shape[2]
        a = (qc @ kc.transpose(-1, -2)) * torch.tril(torch.ones(n, n, device=q.device))
        num = qc @ S + a @ vc
        den = (qc * z[:, :, None]).sum(-1, keepdim=True) + a.sum(-1, keepdim=True)
        out.append(num / (den + eps))
        S = S + kc.transpose(-1, -2) @ vc; z = z + kc.sum(2)
    return torch.cat(out, 2)


def linear_attention_recurrent(q, k, v, eps=1e-6):
    """Reference token-by-token recurrence (testing only)."""
    q, k = phi(q.float()), phi(k.float()); v = v.float()
    S = torch.zeros(*q.shape[:2], q.shape[-1], v.shape[-1], device=q.device); z = torch.zeros(*q.shape[:2], q.shape[-1], device=q.device)
    out = []
    for t in range(q.shape[2]):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]; z = z + k[:, :, t]
        out.append((q[:, :, t, None, :] @ S)[:, :, 0] / ((q[:, :, t] * z).sum(-1, keepdim=True) + eps))
    return torch.stack(out, 2)


class Attn(nn.Module):
    def __init__(self, kind, hybrid_window=16, topk=None):
        super().__init__()
        self.kind = kind; self.qkv = nn.Linear(DM, 3 * DM); self.o = nn.Linear(DM, DM)
        self.hybrid_window = hybrid_window; self.test_mask = None; self.topk = topk

    def forward(self, x):
        Bx, L, _ = x.shape
        q, k, v = self.qkv(x).view(Bx, L, 3, NH, HD).permute(2, 0, 3, 1, 4)
        kind = self.test_mask or self.kind
        if kind == "dense" and self.topk is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif kind == "dense":  # post-hoc content top-k (dense scores computed first)
            s = (q @ k.transpose(-1, -2)) / math.sqrt(HD)
            s = s.masked_fill(~band_mask(L, L, x.device), float("-inf"))
            kth = s.topk(min(self.topk, L), -1).values[..., -1:]
            s = s.masked_fill(s < kth, float("-inf")); y = torch.softmax(s, -1) @ v
        elif kind == "window":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=band_mask(L, W, x.device))
        elif kind == "dilated":
            m = torch.stack([band_mask(L, W, x.device)] * 2 + [dilated_mask(L, x.device)] * 2)[None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        elif kind == "linear":
            y = linear_attention(q, k, v).to(q.dtype)
        elif kind == "hybrid":
            yl = F.scaled_dot_product_attention(q[:, :2], k[:, :2], v[:, :2], attn_mask=band_mask(L, self.hybrid_window, x.device))
            yg = linear_attention(q[:, 2:], k[:, 2:], v[:, 2:]).to(q.dtype)
            y = torch.cat([yl, yg], 1)
        else:
            raise ValueError(kind)
        return self.o(y.transpose(1, 2).reshape(Bx, L, DM))


class Block(nn.Module):
    def __init__(self, kind, **kw):
        super().__init__()
        self.n1 = nn.LayerNorm(DM); self.n2 = nn.LayerNorm(DM); self.attn = Attn(kind, **kw)
        self.mlp = nn.Sequential(nn.Linear(DM, 4 * DM), nn.GELU(), nn.Linear(4 * DM, DM))

    def forward(self, x):
        x = x + self.attn(self.n1(x)); return x + self.mlp(self.n2(x))


class MQARModel(nn.Module):
    def __init__(self, kind, **kw):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM); self.register_buffer("pe", sinusoid(8192, DM), persistent=False)
        self.blocks = nn.ModuleList([Block(kind, **kw) for _ in range(2)])
        self.nf = nn.LayerNorm(DM); self.head = nn.Linear(DM, VOCAB)

    def forward(self, tok, qpos=None):
        x = self.emb(tok) + self.pe[:tok.shape[1]]
        for b in self.blocks: x = b(x)
        x = self.nf(x)
        if qpos is not None: x = x.gather(1, qpos[..., None].expand(-1, -1, DM))
        return self.head(x)

    def set_test_mask(self, kind):
        for b in self.blocks: b.attn.test_mask = kind

    def set_topk(self, k):
        for b in self.blocks: b.attn.topk = k


def lr_at(step):
    if step < WARM: return PEAK_LR * (step + 1) / WARM
    frac = (step - WARM) / max(1, UPDATES - WARM)
    return PEAK_LR * (1 - (1 - FINAL_FRAC) * frac)


# ---------------- evaluation -----------------
@torch.no_grad()
def evaluate(model, sets, bs=256, device="cuda"):
    model.eval(); res = {}; per = {}
    for (K, reg), d in sets.items():
        correct, dists, losses = [], [], []
        for s in range(0, len(d["tokens"]), bs):
            tok = torch.as_tensor(d["tokens"][s:s + bs], device=device)
            qp = torch.as_tensor(d["qpos"][s:s + bs], device=device); ans = torch.as_tensor(d["ans"][s:s + bs], device=device)
            lg = model(tok, qp).float()
            losses.append(F.cross_entropy(lg.reshape(-1, VOCAB), ans.reshape(-1), reduction="none").cpu())
            correct.append((lg.argmax(-1) == ans).cpu())
        c = torch.cat(correct).numpy().reshape(-1); dist = d["dist"].reshape(-1)
        r = dict(acc=float(c.mean()), loss=float(torch.cat(losses).mean()), n_queries=int(c.size),
                 chance_label=1 / 64, chance_present=1 / K)
        for lo, hi in BINS:
            msk = (dist >= lo) & (dist < hi)
            r[f"acc_d{lo}_{hi}"] = float(c[msk].mean()) if msk.any() else None
            r[f"n_d{lo}_{hi}"] = int(msk.sum())
        res[f"K{K}_{reg}"] = r
        # fine distance profile for heatmap (16-wide bins)
        per[f"K{K}_{reg}"] = [float(c[(dist >= a) & (dist < a + 16)].mean()) if ((dist >= a) & (dist < a + 16)).any() else None
                              for a in range(0, 256, 16)]
    model.train()
    return res, per


def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_one(name, kind, K, regime, seed, test_sets, dev_sets, device="cuda", hybrid_window=16):
    out = R / "runs" / f"{name}_seed{seed}"
    if (out / "final.json").exists(): return load_json(out / "final.json")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = MQARModel(kind, hybrid_window=hybrid_window).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WD)
    data = train_stream(K, regime, UPDATES * BATCH)
    rng = np.random.default_rng(seed); order = rng.permutation(UPDATES * BATCH)
    curve = []; ck = []; t0 = time.time(); run_loss = 0.0
    for step in range(UPDATES):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        idx = order[step * BATCH:(step + 1) * BATCH]
        tok = torch.as_tensor(data["tokens"][idx], device=device); qp = torch.as_tensor(data["qpos"][idx], device=device)
        ans = torch.as_tensor(data["ans"][idx], device=device)
        lg = model(tok, qp)
        loss = F.cross_entropy(lg.reshape(-1, VOCAB).float(), ans.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        run_loss += loss.item()
        if (step + 1) % 50 == 0:
            curve.append(dict(step=step + 1, train_loss=run_loss / 50, lr=lr_at(step), grad_norm=float(gn))); run_loss = 0.0
        if (step + 1) in CKPTS:
            dev, _ = evaluate(model, dev_sets, device=device)
            ck.append(dict(step=step + 1, dev=dev, elapsed_s=time.time() - t0))
            print(name, seed, step + 1, {k: round(v["acc"], 3) for k, v in dev.items() if k.startswith(f"K{K}")}, flush=True)
    train_s = time.time() - t0
    torch.save(model.state_dict(), out / "final.pt")
    test, prof = evaluate(model, test_sets, device=device)
    res = dict(name=name, kind=kind, train_K=K, train_regime=regime, seed=seed, params=n_params(model),
               train_seconds=train_s, curve=curve, checkpoints=ck, test=test, test_profile16=prof,
               hybrid_window=hybrid_window if kind == "hybrid" else None)
    if kind == "dense":  # train-vs-test mask controls
        model.set_test_mask("window"); res["test_with_window64_mask"], _ = evaluate(model, test_sets, device=device)
        model.set_test_mask(None); model.set_topk(64); res["test_with_topk64"], _ = evaluate(model, test_sets, device=device)
        model.set_topk(None)
    save_json(out / "final.json", res)
    return res


# ---------------- theory + sweep -----------------
def theory(T_=T):
    i = np.arange(T_)
    dense = int((i + 1).sum()); window = int(np.minimum(i + 1, W).sum())
    dil = int(np.minimum(i // 4 + 1, 64).sum())
    per_layer_fp32 = lambda floats: floats * 4
    return dict(T=T_, attended_pairs_per_head=dict(dense=dense, window64=window, dilated64x4=dil,
                                                   local_dilated_model_avg=(2 * window + 2 * dil) / 4,
                                                   hybrid_local16=int(np.minimum(i + 1, 16).sum())),
                max_reachable_distance_2layers=dict(dense=T_ - 1, window64=2 * (W - 1), dilated=2 * 63 * 4,
                                                    linear="unbounded (compressed)"),
                decode_state_floats_per_layer=dict(
                    dense_kv_cache=f"2*{NH}*{HD}*t (grows with t)", window64_kv=2 * NH * HD * W,
                    dilated_kv="local: 2*2*32*64; dilated: needs last 253 positions -> 2*2*32*253",
                    linear=NH * (HD * HD + HD), hybrid=2 * (HD * HD + HD) + 2 * 16 * 2 * HD),
                note="state in floats; x4 bytes FP32")


@torch.no_grad()
def sweep(device="cuda", reps=10):
    rows = []
    g = torch.Generator(device=device).manual_seed(6304)
    for Tt in (256, 1024, 4096, 8192):
        for kind in ("dense", "window", "dilated", "linear", "hybrid"):
            a = Attn(kind).to(device).half().eval()
            x = torch.randn(8, Tt, DM, device=device, generator=g).half()
            r = dict(T=Tt, kind=kind, B=8, H=NH, d=HD,
                     implementation={"dense": "SDPA is_causal", "window": "SDPA + dense bool LxL mask",
                                     "dilated": "SDPA + dense bool HxLxL mask", "linear": "chunked (64) FP32 state",
                                     "hybrid": "SDPA band mask (2 heads) + chunked linear (2 heads)"}[kind])
            try:
                for _ in range(3): a(x)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
                ts = []
                for _ in range(reps):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); a(x); torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1e3)
                r.update(summarize_ms(ts), status="ok", peak_extra_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20)
            except torch.cuda.OutOfMemoryError:
                r["status"] = "OOM"
            torch.cuda.empty_cache(); rows.append(r); print(r, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all", help="comma list of config names or 'all'")
    ap.add_argument("--seeds", default="6304,6305,6306")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--hybrid-window", type=int, default=16)
    a = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    g = _gen()
    for regime in ("near", "far", "mixed"):
        for k in (4, 8, 16):
            for idx in range(64): g.make_example("test", k, regime, idx)
    save_json(R / "env.json", env_info(dict(theory=theory())))
    if a.sweep:
        write_jsonl(R / "inference_sweep.jsonl", sweep()); return
    test_sets, dev_sets = frozen("test"), frozen("development")
    names = [c[0] for c in CONFIGS] if a.runs == "all" else a.runs.split(",")
    for name in names:
        _, kind, K, reg = next(c for c in CONFIGS if c[0] == name)
        for s in map(int, a.seeds.split(",")):
            train_one(name, kind, K, reg, s, test_sets, dev_sets, hybrid_window=a.hybrid_window)


if __name__ == "__main__":
    main()
