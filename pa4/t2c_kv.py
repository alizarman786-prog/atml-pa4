"""Task 2C: KV-cache eviction / quantization / hybrid on Qwen2.5-1.5B-Instruct at a matched INT4 byte budget.
python -m pa4.t2c_kv --stage validate|eval|bench [--limit N]"""
from __future__ import annotations
import argparse, math, re, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .common import OUT, DATA, SEED, read_jsonl, write_jsonl, save_json, load_json, env_info, revisions, summarize_ms

R = OUT / "task2" / "C"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
G = 64            # quantization group size
N_SINK, N_RECENT = 8, 64
IDX_BYTES = 2     # int16 position index (L < 32768)
POLICIES = ("full", "sink_recent", "qaware", "quant_token", "quant_kivi", "hybrid")
MAX_NEW = 16
DEV = "cuda"


# ============ quantization primitives (asymmetric, FP16 scale + FP16 offset(min), round-half-even) ============
def _sync():
    if DEV == "cuda": torch.cuda.synchronize()


def quant_groups(x, bits, dim):
    """Quantize along `dim` which must have size G (a group). Returns codes(uint8), scale, offset (FP16)."""
    qmax = 2 ** bits - 1
    xf = x.float()
    mn = xf.amin(dim, keepdim=True); mx = xf.amax(dim, keepdim=True)
    scale = ((mx - mn) / qmax).clamp(min=1e-6).half(); off = mn.half()
    q = torch.clamp(torch.round((xf - off.float()) / scale.float()), 0, qmax).to(torch.uint8)
    return q, scale, off


def dequant(q, scale, off, dtype=torch.float16):
    return (q.float() * scale.float() + off.float()).to(dtype)


def fake_quant_channelwise(K, bits):
    """K: (B,H,Lq,d) with Lq % G == 0; groups = 64 consecutive tokens per channel."""
    B, H, L, d = K.shape
    Kg = K.reshape(B, H, L // G, G, d)
    q, s, o = quant_groups(Kg, bits, dim=3)
    return dequant(q, s, o, K.dtype).reshape(B, H, L, d)


def fake_quant_tokenwise(V, bits):
    """V: (B,H,L,d) with d % G == 0; groups = 64 consecutive features per token."""
    B, H, L, d = V.shape
    Vg = V.reshape(B, H, L, d // G, G)
    q, s, o = quant_groups(Vg, bits, dim=4)
    return dequant(q, s, o, V.dtype).reshape(B, H, L, d)


# ---- real packing (C5 + byte verification) ----
def pack_codes(q, bits):
    per = 8 // bits; q = q.reshape(-1)
    pad = (-q.numel()) % per
    if pad: q = torch.cat([q, q.new_zeros(pad)])
    q = q.view(-1, per).to(torch.int32)
    out = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    for i in range(per): out |= q[:, i] << (bits * i)
    return out.to(torch.uint8)


def unpack_codes(p, bits, n):
    per = 8 // bits; p = p.to(torch.int32)
    mask = (1 << bits) - 1
    q = torch.stack([(p >> (bits * i)) & mask for i in range(per)], 1).reshape(-1)[:n]
    return q.to(torch.uint8)


def pack_layer(K, V, bits, layout):
    """Pack one layer's (B,H,L,d) K/V. layout: 'kivi' (K channel-wise, V token-wise) or 'token' (both token-wise).
    Trailing L mod 64 tokens stay FP16 for both. Returns dict of tensors (the serialized object)."""
    B, H, L, d = K.shape; Lq = (L // G) * G
    out = dict(shape=(B, H, L, d), bits=bits, layout=layout, K_res=K[:, :, Lq:].contiguous(), V_res=V[:, :, Lq:].contiguous())
    if layout == "kivi":
        qk, sk, ok = quant_groups(K[:, :, :Lq].reshape(B, H, Lq // G, G, d), bits, 3)
    else:
        qk, sk, ok = quant_groups(K[:, :, :Lq].reshape(B, H, Lq, d // G, G), bits, 4)
    qv, sv, ov = quant_groups(V[:, :, :Lq].reshape(B, H, Lq, d // G, G), bits, 4)
    out.update(Kp=pack_codes(qk, bits), Ks=sk, Ko=ok, Vp=pack_codes(qv, bits), Vs=sv, Vo=ov, nK=qk.numel(), nV=qv.numel())
    return out


def unpack_layer(P):
    B, H, L, d = P["shape"]; Lq = (L // G) * G; b = P["bits"]
    qk = unpack_codes(P["Kp"], b, P["nK"]); qv = unpack_codes(P["Vp"], b, P["nV"])
    if P["layout"] == "kivi":
        K = dequant(qk.view(B, H, Lq // G, G, d), P["Ks"], P["Ko"]).reshape(B, H, Lq, d)
    else:
        K = dequant(qk.view(B, H, Lq, d // G, G), P["Ks"], P["Ko"]).reshape(B, H, Lq, d)
    V = dequant(qv.view(B, H, Lq, d // G, G), P["Vs"], P["Vo"]).reshape(B, H, Lq, d)
    return torch.cat([K, P["K_res"]], 2), torch.cat([V, P["V_res"]], 2)


def serialized_bytes(P):
    return sum(t.numel() * t.element_size() for k, t in P.items() if isinstance(t, torch.Tensor))


# ============ byte accounting ============
class Geo:
    def __init__(self, cfg):
        self.nl = cfg.num_hidden_layers; self.hkv = cfg.num_key_value_heads
        self.dh = cfg.hidden_size // cfg.num_attention_heads
        self.ch = 2 * self.nl * self.hkv * self.dh  # stored KV channels

    def M16(self, L): return 4 * self.nl * self.hkv * self.dh * L

    def Mb(self, L, b): return self.ch * ((L // G) * (G * b // 8 + 4) + 2 * (L % G))

    def evict_bytes(self, n, indexed):
        return self.M16(n) + (self.nl * n * IDX_BYTES if indexed else 8)

    def evict_n(self, budget, indexed):
        n = budget // (4 * self.nl * self.hkv * self.dh)
        while self.evict_bytes(n, indexed) > budget: n -= 1
        return n

    def hybrid_bytes(self, L, n_hp, b=2):
        nq = L - n_hp; g = nq // G; r = nq % G
        return self.ch * (2 * n_hp + g * (G * b // 8 + 4) + 2 * r) + self.nl * n_hp * IDX_BYTES

    def hybrid_n(self, L, budget, b=2):
        best = None
        for n in range(N_SINK + N_RECENT, L + 1):
            if self.hybrid_bytes(L, n, b) <= budget: best = n
        return best


# ============ cache helpers ============
def cache_from(kv):
    c = DynamicCache()
    for l, (k, v) in enumerate(kv): c.update(k, v, l)
    return c


def cache_kv(cache):
    return [(layer.keys, layer.values) for layer in cache.layers]


def clone_cache(cache):
    return cache_from([(k.clone(), v.clone()) for k, v in cache_kv(cache)])


class QuantTail:
    """Keeps quantized layouts consistent for appended tokens: once the FP16 tail reaches 64 tokens,
    the oldest complete 64-token group is fake-quantized (stored dequantized for ordinary SDPA)."""
    def __init__(self, layout, bits, start):
        self.layout, self.bits, self.qend = layout, bits, start

    def maintain(self, cache):
        L = cache.get_seq_length()
        while L - self.qend >= G:
            a, b = self.qend, self.qend + G
            for layer in cache.layers:
                k, v = layer.keys, layer.values
                k[:, :, a:b] = (fake_quant_channelwise if self.layout == "kivi" else fake_quant_tokenwise)(k[:, :, a:b], self.bits)
                v[:, :, a:b] = fake_quant_tokenwise(v[:, :, a:b], self.bits)
            self.qend = b


# ============ prompt construction ============
def build_case(tok, case):
    msgs = [{"role": "user", "content": case["user_text"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    off = text.index(case["user_text"])
    qchar = off + case["user_text"].index("QUESTION_BEGIN")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]; offs = enc["offset_mapping"]
    split = next(i for i, (a, b) in enumerate(offs) if a >= qchar)
    assert offs[split - 1][1] <= qchar, "question boundary inside a token"
    es, ee = off + case["evidence_char_start"], off + case["evidence_char_end"]
    ev = [i for i, (a, b) in enumerate(offs) if a < ee and b > es and i < split]
    ans_ids = tok(case["code"], add_special_tokens=False)["input_ids"]
    return dict(id=case["id"], ids=ids, split=split, evidence=ev, code=case["code"], ans_ids=ans_ids,
                ctx_len=split, total_len=len(ids), target_tokens=case["target_tokens"],
                position=case["position"], distractors=case["distractors"])


def extract(text):
    m = re.search(r"(?<!\d)(\d{5})(?!\d)", text)
    return m.group(1) if m else None


# ============ model calls ============
@torch.inference_mode()
def fwd(model, ids, cache, abs_start, phys_start, logits_all=False):
    ids = torch.as_tensor([ids], device=DEV)
    n = ids.shape[1]
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                position_ids=torch.arange(abs_start, abs_start + n, device=DEV)[None],
                cache_position=torch.arange(phys_start, phys_start + n, device=DEV),
                logits_to_keep=0 if logits_all else 1)
    return out.logits[0].float()


@torch.inference_mode()
def greedy(model, tok, cache, suffix, abs_start, maint=None, max_new=MAX_NEW):
    """Process the question suffix then greedy-decode. Returns tokens, first logits, timings."""
    phys = cache.get_seq_length()
    _sync(); t0 = time.perf_counter()
    lg = fwd(model, suffix, cache, abs_start, phys)[-1]
    if maint: maint.maintain(cache)
    _sync(); t_suffix = time.perf_counter() - t0
    first = lg.clone(); out = []; a = abs_start + len(suffix); p = phys + len(suffix)
    t0 = time.perf_counter()
    for _ in range(max_new):
        nxt = int(lg.argmax())
        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
        out.append(nxt)
        lg = fwd(model, [nxt], cache, a, p)[-1]; a += 1; p += 1
        if maint: maint.maintain(cache)
    _sync(); t_dec = time.perf_counter() - t0
    return out, first, dict(suffix_s=t_suffix, decode_s=t_dec, decode_tokens=len(out))


@torch.inference_mode()
def teacher_forced(model, cache, suffix, ans_ids, abs_start):
    """Logits at the last suffix position and over the reference answer (teacher forcing)."""
    phys = cache.get_seq_length()
    lg = fwd(model, suffix + ans_ids, cache, abs_start, phys, logits_all=True)
    pos = lg[len(suffix) - 1:len(suffix) - 1 + len(ans_ids)]  # predicts each answer token
    lp = torch.log_softmax(pos, -1)
    nll = -lp.gather(-1, torch.as_tensor(ans_ids, device=DEV)[:, None])[:, 0]
    return pos, nll


@torch.inference_mode()
def question_scores(model, cache, suffix, L, pool=7):
    """SnapKV-inspired fixed policy: per layer, FP32 masked softmax attention of every suffix query (all heads,
    RoPE applied, against the existing rotary-positioned cached keys + causal suffix keys); importance of each
    context position = attention mass summed over suffix queries and heads; 1-D max-pool (kernel 7).
    Selection shared by the 2 KV heads within a layer, independent across layers. Cache is cropped back to L."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    cfg = model.config; nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = cfg.hidden_size // nh
    qs = {}
    hooks = [layer.self_attn.q_proj.register_forward_hook(lambda m, i, o, l=l: qs.__setitem__(l, o))
             for l, layer in enumerate(model.model.layers)]
    try:
        fwd(model, suffix, cache, L, L)
    finally:
        for h in hooks: h.remove()
    nq = len(suffix)
    pos = torch.arange(L, L + nq, device=DEV)[None]
    dummy = torch.zeros(1, nq, hd, device=DEV, dtype=torch.float16)
    cos, sin = model.model.rotary_emb(dummy, pos)
    scores = []
    causal = torch.ones(nq, L + nq, dtype=torch.bool, device=DEV)
    causal[:, L:] = torch.tril(torch.ones(nq, nq, dtype=torch.bool, device=DEV))
    for l, layer in enumerate(cache.layers):
        q = qs[l].view(1, nq, nh, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = repeat_kv(layer.keys, nh // nkv)  # (1, nh, L+nq, hd)
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(~causal, float("-inf"))
        a = torch.softmax(s, -1)[..., :L].sum((0, 1, 2))  # (L,)
        if pool > 1: a = F.max_pool1d(a[None, None], pool, 1, pool // 2)[0, 0]
        scores.append(a)
    cache.crop(L)
    return torch.stack(scores)  # (nl, L)


# ============ policies ============
def select_positions(scores_l, L, n, rule):
    """Return sorted LongTensor of n retained positions (always includes 8 sinks + 64 recent)."""
    base = set(range(min(N_SINK, L))) | set(range(max(0, L - N_RECENT), L))
    if rule == "sink_recent":
        keep = list(range(N_SINK)) + list(range(L - (n - N_SINK), L))
        return torch.tensor(sorted(set(keep)), device=DEV)
    s = scores_l.clone(); idx = torch.tensor(sorted(base), device=DEV); s[idx] = float("inf")
    top = torch.topk(s, n).indices
    return torch.sort(top).values


def apply_policy(policy, kv_full, L, geo, scores=None, n=None):
    """Return compressed kv list (physical), index sets per layer (for FP16-retained positions), maintainer."""
    if policy == "full": return kv_full, None, None
    if policy in ("sink_recent", "qaware"):
        out, sel = [], []
        for l, (k, v) in enumerate(kv_full):
            idx = select_positions(None if scores is None else scores[l], L, n, policy)
            out.append((k[:, :, idx].contiguous(), v[:, :, idx].contiguous())); sel.append(idx)
        return out, sel, None
    if policy in ("quant_token", "quant_kivi"):
        layout = "kivi" if policy == "quant_kivi" else "token"; Lq = (L // G) * G; out = []
        for k, v in kv_full:
            k2, v2 = k.clone(), v.clone()
            k2[:, :, :Lq] = (fake_quant_channelwise if layout == "kivi" else fake_quant_tokenwise)(k[:, :, :Lq], 4)
            v2[:, :, :Lq] = fake_quant_tokenwise(v[:, :, :Lq], 4)
            out.append((k2, v2))
        return out, None, QuantTail(layout, 4, Lq)
    if policy == "hybrid":
        out, sel = [], []
        for l, (k, v) in enumerate(kv_full):
            idx = select_positions(scores[l], L, n, "qaware")
            mask = torch.zeros(L, dtype=torch.bool, device=DEV); mask[idx] = True
            low = torch.nonzero(~mask)[:, 0]; nq = (len(low) // G) * G; lowq = low[:nq]  # residual stays FP16
            k2, v2 = k.clone(), v.clone()
            if nq:
                k2[:, :, lowq] = fake_quant_channelwise(k[:, :, lowq], 2)
                v2[:, :, lowq] = fake_quant_tokenwise(v[:, :, lowq], 2)
            out.append((k2, v2)); sel.append(idx)
        return out, sel, None
    raise ValueError(policy)


def realized_bytes(policy, L, geo, n=None, kv_full=None):
    if policy == "full": return geo.M16(L)
    if policy == "sink_recent": return geo.evict_bytes(n, indexed=False)
    if policy == "qaware": return geo.evict_bytes(n, indexed=True)
    if policy in ("quant_token", "quant_kivi"):
        if kv_full is not None:  # actual packed bytes
            lay = "kivi" if policy == "quant_kivi" else "token"
            return sum(serialized_bytes(pack_layer(k, v, 4, lay)) for k, v in kv_full)
        return geo.Mb(L, 4)
    if policy == "hybrid": return geo.hybrid_bytes(L, n, 2)


def kl(p_logits, q_logits):
    lp = torch.log_softmax(p_logits.float(), -1); lq = torch.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


# ============ stages ============
def load():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rev = revisions().get("model_target")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    return tok, model, rev


@torch.inference_mode()
def prefill(model, ids):
    cache = DynamicCache()
    _sync(); t0 = time.perf_counter()
    model(input_ids=torch.as_tensor([ids], device=DEV), past_key_values=cache, use_cache=True, logits_to_keep=1)
    _sync()
    return cache, time.perf_counter() - t0


@torch.inference_mode()
def stage_validate(tok, model, cases):
    out = dict(checks=[])
    for case in cases[:6]:
        c = build_case(tok, case); L = c["split"]; ids = c["ids"]
        full = model(input_ids=torch.as_tensor([ids], device=DEV)).logits[0].float()
        cache, _ = prefill(model, ids[:L])
        split_lg = fwd(model, ids[L:], cache, L, L, logits_all=True)
        d1 = float((split_lg - full[L:]).abs().max())
        # 100% retention through the eviction (gather) path with explicit absolute positions
        cache2, _ = prefill(model, ids[:L])
        kv = [(k[:, :, torch.arange(L, device=DEV)], v[:, :, torch.arange(L, device=DEV)]) for k, v in cache_kv(cache2)]
        toks, first, _ = greedy(model, tok, cache_from(kv), ids[L:], L)
        ref = model.generate(torch.as_tensor([ids], device=DEV), max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
        ref = [t for t in ref if t not in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))][:len(toks)]
        # eviction with original absolute positions vs explicit 4-D mask on a full forward
        sel = torch.tensor(sorted(set(range(8)) | set(range(L - 64, L)) | set(range(100, L, 3))), device=DEV)
        cache3, _ = prefill(model, ids[:L])
        kv3 = [(k[:, :, sel].contiguous(), v[:, :, sel].contiguous()) for k, v in cache_kv(cache3)]
        ev_lg = fwd(model, ids[L:], cache_from(kv3), L, len(sel), logits_all=True)
        T_ = len(ids); m = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=DEV))
        drop = torch.ones(L, dtype=torch.bool, device=DEV); drop[sel] = False
        m[L:, :L] &= ~drop[None]
        try:
            mask4 = torch.zeros(1, 1, T_, T_, device=DEV, dtype=torch.float16).masked_fill(~m, torch.finfo(torch.float16).min)
            ref3 = model(input_ids=torch.as_tensor([ids], device=DEV), attention_mask=mask4).logits[0, L:].float()
            d3 = float((ev_lg - ref3).abs().max())
        except Exception as e:
            d3 = f"4D-mask reference unavailable: {str(e)[:120]}"
        out["checks"].append(dict(id=c["id"], ctx_len=L, total_len=len(ids), split_vs_full_max_abs_logit=d1,
                                  full_retention_tokens_equal_generate=toks == ref, gen=tok.decode(toks),
                                  evict_vs_4dmask_max_abs_logit=d3, all_finite=bool(torch.isfinite(split_lg).all())))
        print(out["checks"][-1], flush=True)
    save_json(R / "c1_validation.json", out)


@torch.inference_mode()
def stage_eval(tok, model, cases, limit=None):
    geo = Geo(model.config); rows = []
    done = set()
    fout = R / "c5_per_prompt.jsonl"
    if fout.exists(): rows = read_jsonl(fout); done = {(r["id"], r["policy"]) for r in rows}
    for ci, case in enumerate(cases[:limit] if limit else cases):
        c = build_case(tok, case); L = c["split"]; ids = c["ids"]; suffix = ids[L:]
        if all((c["id"], p) in done for p in POLICIES): continue
        budget = geo.Mb(L, 4)
        _sync(); torch.cuda.reset_peak_memory_stats()
        cache0, t_prefill = prefill(model, ids[:L]); prefill_peak = torch.cuda.max_memory_allocated() / 2**20
        kv_full = cache_kv(cache0)
        _sync(); t0 = time.perf_counter()
        scores = question_scores(model, clone_cache(cache0), suffix, L)
        _sync(); t_score = time.perf_counter() - t0
        ref_pos, ref_nll = teacher_forced(model, clone_cache(cache0), suffix, c["ans_ids"], L)
        ev = torch.as_tensor(c["evidence"], device=DEV)
        for pol in POLICIES:
            if (c["id"], pol) in done: continue
            n = None
            if pol == "sink_recent": n = geo.evict_n(budget, indexed=False)
            if pol == "qaware": n = geo.evict_n(budget, indexed=True)
            if pol == "hybrid": n = geo.hybrid_n(L, budget)
            _sync(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            kv, sel, maint = apply_policy(pol, kv_full, L, geo, scores, n)
            _sync(); t_comp = time.perf_counter() - t0
            byts = realized_bytes(pol, L, geo, n, kv_full if pol.startswith("quant") else None)
            pos, nll = teacher_forced(model, cache_from([(k.clone(), v.clone()) for k, v in kv]), suffix, c["ans_ids"], L)
            cache = cache_from([(k.clone(), v.clone()) for k, v in kv])
            toks, first, tm = greedy(model, tok, cache, suffix, L, maint)
            text = tok.decode(toks); pred = extract(text)
            phys_after = cache.get_seq_length()
            r = dict(id=c["id"], policy=pol, ctx_len=L, total_len=len(ids), target_tokens=c["target_tokens"],
                     position=c["position"], distractors=c["distractors"], code=c["code"], output=text, pred=pred,
                     correct=pred == c["code"], ref_answer_nll_mean=float(nll.mean()), ref_answer_nll_sum=float(nll.sum()),
                     kl_first=float(kl(ref_pos[0], pos[0])), kl_tf_mean=float(kl(ref_pos, pos).mean()),
                     budget_bytes=budget, realized_bytes=int(byts), fp16_full_bytes=geo.M16(L),
                     realized_frac_fp16=byts / geo.M16(L), within_budget=bool(byts <= budget or pol == "full"),
                     n_fp16_positions=(int(n) if n is not None else (L if pol == "full" else L % G)),
                     suffix_and_generated_fp16_bytes=geo.M16(len(suffix) + len(toks)),
                     prefill_s=t_prefill, prefill_peak_MiB=prefill_peak,
                     score_s=t_score if pol in ("qaware", "hybrid") else 0.0, compress_s=t_comp,
                     peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20, physical_len_after=phys_after, **tm)
            if sel is not None:
                r["evidence_retained_frac"] = float(np.mean([torch.isin(ev, s).float().mean().item() for s in sel])) if len(ev) else None
                if ci < 12: r["selected_positions_layer0"] = sel[0].tolist(); r["selected_positions_layer14"] = sel[14].tolist()
            if ci < 12 and pol == "full": r["evidence_token_idx"] = c["evidence"]
            rows.append(r); write_jsonl(fout, rows) if len(rows) % 6 == 0 else None
            del kv, cache
        print(ci, c["id"], {r["policy"]: r["correct"] for r in rows if r["id"] == c["id"]}, flush=True)
        del cache0, kv_full; torch.cuda.empty_cache()
    write_jsonl(fout, rows)


@torch.inference_mode()
def stage_bench(tok, model, reps=10, warmup=3):
    rows = []
    g = torch.Generator().manual_seed(SEED)
    for B in (1, 8):
        for L in (2048, 4096):
            ids = torch.randint(1000, 100000, (B, L), generator=g).to(DEV)
            cache = DynamicCache()
            for c0 in range(0, L, 512):  # chunked prefill keeps activation memory small at B=8
                model(input_ids=ids[:, c0:c0 + 512], past_key_values=cache, use_cache=True, logits_to_keep=1)
            kv = cache_kv(cache); fp16_bytes = sum(k.numel() * 2 + v.numel() * 2 for k, v in kv)
            for layout in ("kivi", "token"):
                def do_pack(): return [pack_layer(k, v, 4, layout) for k, v in kv]
                for _ in range(warmup): P = do_pack()
                ts_p = []
                for _ in range(reps):
                    _sync(); t0 = time.perf_counter(); P = do_pack(); _sync(); ts_p.append((time.perf_counter() - t0) * 1e3)
                ser = sum(serialized_bytes(p) for p in P)
                def do_unpack(): return [unpack_layer(p) for p in P]
                for _ in range(warmup): U = do_unpack()
                ts_u = []
                for _ in range(reps):
                    _sync(); t0 = time.perf_counter(); U = do_unpack(); _sync(); ts_u.append((time.perf_counter() - t0) * 1e3)
                err_k = max(float((u[0] - k).abs().max()) for u, (k, v) in zip(U, kv))
                nxt = torch.randint(1000, 100000, (B, 1), generator=g).to(DEV)
                # decode step: FP16 reference vs unpack-then-step (cache rebuilt outside timing for FP16)
                def fp16_step():
                    c = cache_from(kv_clone)
                    _sync(); t0 = time.perf_counter()
                    model(input_ids=nxt, past_key_values=c, use_cache=True, position_ids=torch.full((B, 1), L, device=DEV))
                    _sync(); return (time.perf_counter() - t0) * 1e3
                def packed_step():
                    _sync(); t0 = time.perf_counter()
                    c = cache_from([unpack_layer(p) for p in P])
                    model(input_ids=nxt, past_key_values=c, use_cache=True, position_ids=torch.full((B, 1), L, device=DEV))
                    _sync(); return (time.perf_counter() - t0) * 1e3
                kv_clone = [(k.clone(), v.clone()) for k, v in kv]
                for _ in range(warmup): fp16_step(); packed_step()
                torch.cuda.reset_peak_memory_stats(); tf = [fp16_step() for _ in range(reps)]; pk_f = torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.reset_peak_memory_stats(); tq = [packed_step() for _ in range(reps)]; pk_q = torch.cuda.max_memory_allocated() / 2**20
                r = dict(B=B, L=L, layout=layout, bits=4, fp16_cache_bytes=fp16_bytes, serialized_packed_bytes=ser,
                         formula_bytes=B * Geo(model.config).Mb(L, 4), pack=summarize_ms(ts_p), unpack=summarize_ms(ts_u),
                         decode_step_fp16=summarize_ms(tf), decode_step_unpack_plus_step=summarize_ms(tq),
                         peak_alloc_fp16_step_MiB=pk_f, peak_alloc_packed_step_MiB=pk_q, max_abs_K_recon_err=err_k,
                         note="packed path dequantizes the whole cache to FP16 then runs ordinary SDPA")
                rows.append(r); print(r, flush=True); write_jsonl(R / "c5_pack_bench.jsonl", rows)
                del P, U, kv_clone; torch.cuda.empty_cache()
            del cache, kv; torch.cuda.empty_cache()
    write_jsonl(R / "c5_pack_bench.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="validate,eval,bench")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    tok, model, rev = load()
    cases = read_jsonl(DATA / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz")
    lens = [len(build_case(tok, c)["ids"]) for c in cases]
    save_json(R / "env.json", env_info(dict(model=MODEL, revision=rev, n_cases=len(cases), group=G,
                                            encoded_len_min=min(lens), encoded_len_max=max(lens),
                                            policies=POLICIES, max_new_tokens=MAX_NEW, n_sink=N_SINK, n_recent=N_RECENT)))
    for s in a.stage.split(","):
        if s == "validate": stage_validate(tok, model, cases)
        if s == "eval": stage_eval(tok, model, cases, a.limit)
        if s == "bench": stage_bench(tok, model)


if __name__ == "__main__":
    main()
