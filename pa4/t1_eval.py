"""Task 1 evaluation: perplexity on fixed windows, zero-shot HellaSwag, resources."""
from __future__ import annotations
import math, re, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import DATA, read_jsonl, cuda_time, summarize_ms, peak_mem

PUB = DATA / "public_locked"
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def eligible(model):
    """Ordered list of (block_idx, proj_name, nn.Linear)."""
    out = []
    for i, layer in enumerate(model.model.layers):
        for p in PROJS:
            parent = layer.self_attn if p in ATTN else layer.mlp
            out.append((i, p, getattr(parent, p)))
    return out


def check_enumeration(model):
    el = eligible(model)
    n_layers = len(model.model.layers)
    assert len(el) == 7 * n_layers, len(el)
    ids = {id(m.weight) for _, _, m in el}
    assert len(ids) == len(el), "duplicate eligible weights"
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    seen, total, elig = set(), 0, 0
    elig_ptrs = {m.weight.data_ptr() for _, _, m in el}
    for n, p in model.named_parameters(remove_duplicate=False):
        if p.data_ptr() in seen: continue
        seen.add(p.data_ptr()); total += p.numel()
    for _, _, m in el: elig += m.weight.numel()
    # make sure no eligible weight aliases embeddings / head
    assert model.model.embed_tokens.weight.data_ptr() not in elig_ptrs
    return dict(n_eligible_matrices=len(el), eligible_params=elig, unique_params=total,
                non_eligible_params=total - elig, tied_embeddings=bool(tied),
                n_layers=n_layers, biases=[n for n, _ in model.named_parameters() if n.endswith("bias")][:3])


# ---------------- data -----------------
def load_windows(name):
    f = {"wikitext2": "wikitext2_test_windows_locked.jsonl.gz", "c4val": "c4_validation_windows_locked.jsonl.gz"}[name]
    rows = read_jsonl(PUB / f)
    assert sum(r["scored_tokens"] for r in rows) == 50000
    return rows


def load_c4(which="calibration"):
    return read_jsonl(PUB / f"c4_{which}_locked.jsonl.gz")


def load_hellaswag():
    rows = read_jsonl(PUB / "hellaswag_val_locked.jsonl.gz")
    rows.sort(key=lambda r: r["order"])
    assert len(rows) == 2000
    return rows


# ---------------- perplexity -----------------
@torch.inference_mode()
def perplexity(model, windows, bs=8, device="cuda"):
    """Each window independent; target offsets [a,b) scored once. Returns pooled stats + per-window sums."""
    order = sorted(range(len(windows)), key=lambda i: -len(windows[i]["token_ids"]))
    nll = np.zeros(len(windows)); cnt = np.zeros(len(windows), dtype=np.int64)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(windows[i]["token_ids"]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long); att = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            t = windows[i]["token_ids"]; ids[j, :len(t)] = torch.tensor(t); att[j, :len(t)] = 1
        ids, att = ids.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits  # FP16
        for j, i in enumerate(idx):
            a, b = windows[i]["score_target_offsets"]
            # logits at t-1 predict token t; FP32 log-softmax only over the scored rows of this window
            v = torch.nn.functional.cross_entropy(logits[j, a - 1:b - 1].float(), ids[j, a:b], reduction="none")
            nll[i] = v.sum().item(); cnt[i] = v.numel()
        del logits
    N = int(cnt.sum()); mean = float(nll.sum() / N)
    return dict(N=N, mean_nll=mean, ppl=math.exp(mean)), nll, cnt


@torch.inference_mode()
def dev_nll(model, seqs, bs=8, device="cuda"):
    """Mean NLL over 512-token C4 sequences (first token excluded)."""
    tot, n = 0.0, 0
    for s in range(0, len(seqs), bs):
        ids = torch.tensor([r["token_ids"] for r in seqs[s:s + bs]], device=device)
        logits = model(input_ids=ids, use_cache=False).logits
        for j in range(ids.shape[0]):
            tot += F.cross_entropy(logits[j, :-1].float(), ids[j, 1:], reduction="sum").item()
        n += ids[:, 1:].numel(); del logits
    return tot / n


# ---------------- HellaSwag (lm-eval-harness v1.0 conventions) -----------------
def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def hellaswag_requests(rows, tok):
    """Return per-example (ctx_ids, [cont_ids x4], [char_len x4], gold)."""
    out = []
    for r in rows:
        ex = r["example"]
        ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        query = _hs_pre(ex["activity_label"] + ": " + ctx)
        choices = [_hs_pre(e) for e in ex["endings"]]
        conts = []
        ctx_ids = tok(query, add_special_tokens=False)["input_ids"]
        for c in choices:
            # harness _encode_pair: encode whole, continuation = whole[len(context_enc):]
            whole = tok(query + " " + c, add_special_tokens=False)["input_ids"]
            conts.append(whole[len(ctx_ids):])
        out.append(dict(order=r["order"], ctx=ctx_ids, conts=conts, char_len=[len(c) for c in choices],
                        gold=int(ex["label"]), query=query, choices=choices))
    return out


@torch.inference_mode()
def hellaswag_eval(model, reqs, bs=32, device="cuda", pad_id=0):
    flat = [(i, j, r["ctx"] + c, len(c)) for i, r in enumerate(reqs) for j, c in enumerate(r["conts"])]
    flat.sort(key=lambda x: -len(x[2]))
    ll = np.zeros((len(reqs), 4))
    for s in range(0, len(flat), bs):
        b = flat[s:s + bs]; L = max(len(x[2]) for x in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long); att = torch.zeros(len(b), L, dtype=torch.long)
        for k, x in enumerate(b):
            ids[k, :len(x[2])] = torch.tensor(x[2]); att[k, :len(x[2])] = 1
        ids, att = ids.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for k, (i, j, seq, n) in enumerate(b):
            T = len(seq)
            tgt = torch.tensor(seq[T - n:], device=device)
            lp = torch.log_softmax(logits[k, T - n - 1:T - 1].float(), -1)
            ll[i, j] = lp.gather(-1, tgt[:, None]).sum().item()
        del logits
    gold = np.array([r["gold"] for r in reqs]); clen = np.array([r["char_len"] for r in reqs], float)
    acc = (ll.argmax(1) == gold); accn = ((ll / clen).argmax(1) == gold)
    return dict(n=len(reqs), acc=float(acc.mean()), acc_norm=float(accn.mean())), ll, acc, accn


# ---------------- resources -----------------
@torch.inference_mode()
def latency(model, vocab, device="cuda", reps=5, warmup=2, seed=6304):
    g = torch.Generator().manual_seed(seed)
    res = {}
    for L in (512, 2048):
        ids = torch.randint(100, vocab - 100, (1, L), generator=g).to(device)
        with peak_mem() as pm:
            ts = cuda_time(lambda: model(input_ids=ids, use_cache=True), reps=reps, warmup=warmup)
        res[f"prefill_{L}"] = dict(**summarize_ms(ts), **pm)
    ids = torch.randint(100, vocab - 100, (1, 512), generator=g).to(device)

    def decode():
        out = model(input_ids=ids, use_cache=True)
        pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(64):
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); return (time.perf_counter() - t0) * 1e3
    for _ in range(warmup): decode()
    with peak_mem() as pm:
        ts = [decode() for _ in range(reps)]
    res["decode_64_after_512"] = dict(**summarize_ms(ts), per_token_ms=float(np.mean(ts) / 64), **pm)
    res["unit"] = "ms, batch 1, FP16 dense execution, CUDA-synchronized"
    return res
