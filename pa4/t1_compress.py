"""Task 1 compression primitives: calibration stats, pruning masks, group quantizer, AWQ-style scaling, storage."""
from __future__ import annotations
import math
import numpy as np
import torch
from .t1_eval import eligible, PROJS

GROUP = 128
INPUT_KEY = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
             "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}


# ---------------- calibration -----------------
@torch.inference_mode()
def collect_calibration(model, seqs, n_sub_tokens=2048, seed=6304, bs=4, device="cuda"):
    """One pass over calibration sequences. Per block & input type:
       sumsq[j]  = sum_t x_tj^2  (Wanda ||X_j||^2)
       absmean[j]= mean_t |x_tj| (AWQ activation statistic)
       X_sub     = fixed random subset of token rows (FP16, CPU) for output-error / AWQ loss."""
    n_layers = len(model.model.layers)
    total_tokens = len(seqs) * len(seqs[0]["token_ids"])
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total_tokens, size=n_sub_tokens, replace=False))
    stats = {(i, k): dict(sumsq=None, abssum=None, sub=[]) for i in range(n_layers) for k in set(INPUT_KEY.values())}
    offset = [0]
    hooks = []

    def mk(i, k):
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            st = stats[(i, k)]
            ss = (x * x).sum(0); ab = x.abs().sum(0)
            st["sumsq"] = ss if st["sumsq"] is None else st["sumsq"] + ss
            st["abssum"] = ab if st["abssum"] is None else st["abssum"] + ab
            lo = offset[0]; hi = lo + x.shape[0]
            sel = pick[(pick >= lo) & (pick < hi)] - lo
            if len(sel): st["sub"].append(x[torch.as_tensor(sel, device=x.device)].half().cpu())
        return hook
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(mk(i, "attn_in")))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(i, "o_in")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(mk(i, "mlp_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk(i, "down_in")))
    try:
        for s in range(0, len(seqs), bs):
            ids = torch.tensor([r["token_ids"] for r in seqs[s:s + bs]], device=device)
            model(input_ids=ids, use_cache=False)
            offset[0] += ids.numel()
    finally:
        for h in hooks: h.remove()
    out = {}
    for key, st in stats.items():
        out[key] = dict(norm=st["sumsq"].sqrt().cpu(), absmean=(st["abssum"] / total_tokens).cpu(),
                        X=torch.cat(st["sub"], 0))
    return out, dict(total_tokens=total_tokens, n_sub_tokens=n_sub_tokens, sub_token_indices=pick.tolist()[:50])


# ---------------- original weights snapshot -----------------
def snapshot(model):
    """CPU FP16 copy of every parameter that any method may change (eligible weights, norms, biases)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if "layers." in n}


@torch.no_grad()
def restore(model, snap):
    for n, p in model.named_parameters():
        if n in snap: p.copy_(snap[n].to(p.device))


# ---------------- pruning -----------------
def row_prune_mask(score, k_rows):
    """score: (dout, din); k_rows: int or LongTensor(dout) number of zeros per row. Returns bool keep-mask."""
    dout, din = score.shape
    if isinstance(k_rows, int):
        if k_rows == 0: return torch.ones_like(score, dtype=torch.bool)
        idx = torch.topk(score, k_rows, dim=1, largest=False).indices
        keep = torch.ones_like(score, dtype=torch.bool); keep.scatter_(1, idx, False); return keep
    order = torch.argsort(score, dim=1)  # ascending
    ranks = torch.empty_like(order); ranks.scatter_(1, order, torch.arange(din, device=score.device).expand(dout, din))
    return ranks >= k_rows[:, None].to(score.device)


def nm_mask(score, n=2, m=4):
    dout, din = score.shape
    s = score.reshape(dout, din // m, m)
    idx = torch.topk(s, n, dim=-1).indices
    keep = torch.zeros_like(s, dtype=torch.bool); keep.scatter_(-1, idx, True)
    return keep.reshape(dout, din)


def prune_score(W, method, norm=None):
    W = W.float().abs()
    if method == "magnitude": return W
    if method == "wanda": return W * norm.to(W.device).float()[None, :]
    raise ValueError(method)


@torch.no_grad()
def apply_pruning(model, calib, method, sparsity=None, k_alloc=None, nm=False):
    """sparsity: uniform float; k_alloc: dict (block,proj)->LongTensor zeros-per-row; nm: 2:4."""
    zeros = 0; total = 0
    for i, p, lin in eligible(model):
        W = lin.weight
        sc = prune_score(W, method, calib[(i, INPUT_KEY[p])]["norm"] if method == "wanda" else None)
        if nm: keep = nm_mask(sc)
        elif k_alloc is not None: keep = row_prune_mask(sc, k_alloc[(i, p)])
        else: keep = row_prune_mask(sc, int(round(sparsity * W.shape[1])))
        W.mul_(keep.to(W.dtype))
        zeros += int((~keep).sum()); total += keep.numel()
    return dict(zeros=zeros, eligible=total, achieved_sparsity=zeros / total)


def count_nonzeros(model):
    nz = tot = 0
    for _, _, lin in eligible(model):
        nz += int((lin.weight != 0).sum()); tot += lin.weight.numel()
    return dict(nonzeros=nz, eligible=tot, nonzero_fraction=nz / tot, sparsity=1 - nz / tot)


# ---------------- quantizer (fixed across Q1/Q2/Q3) -----------------
def quantize_groups(W, bits, group=GROUP, return_codes=False):
    """Asymmetric uniform per-group (along input columns) quantization.
    scale = (max-min)/(2^b-1) stored FP16 (computed FP32 then cast); zero = round(-min/scale) clamped to [0,2^b-1]
    stored as b-bit int; q = clamp(round(w/scale)+zero, 0, 2^b-1) with round-half-to-even; w_hat = (q-zero)*scale.
    din must be a multiple of group (true for Qwen2.5-1.5B: 1536, 8960); otherwise zero-pad."""
    dout, din = W.shape
    pad = (-din) % group
    Wf = W.float()
    if pad: Wf = torch.nn.functional.pad(Wf, (0, pad))
    G = Wf.reshape(dout, -1, group)
    qmax = 2 ** bits - 1
    wmin = G.amin(-1, keepdim=True)
    wmax = G.amax(-1, keepdim=True)
    scale = ((wmax - wmin) / qmax).clamp(min=1e-5).half().float()
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(G / scale) + zero, 0, qmax)
    deq = ((q - zero) * scale).reshape(dout, -1)[:, :din]
    if return_codes: return deq, q.to(torch.uint8), scale.half(), zero.to(torch.uint8)
    return deq


def packed_bytes_matrix(dout, din, bits, group=GROUP):
    """Payload: ceil(dout*din_padded*b/8) bytes (bit-stream packing, row-major).
    Metadata per group: FP16 scale (16 bits) + b-bit zero-point, bit-packed: ceil(n_groups*(16+b)/8)."""
    din_p = din + ((-din) % group); n_groups = dout * din_p // group
    payload = math.ceil(dout * din_p * bits / 8)
    meta = math.ceil(n_groups * (16 + bits) / 8)
    return payload, meta


def pack_bits(codes: torch.Tensor, bits: int) -> bytes:
    """Actual bit-stream packing of uint8 codes (LSB-first per code) used to verify the byte formula."""
    a = codes.flatten().cpu().numpy().astype(np.uint8)
    bitplanes = ((a[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitplanes, bitorder="little").tobytes()


def model_storage(model, bits_map=None, mode="fp16"):
    """bits_map: dict (block,proj)->bits for quantized; mode 'fp16' | 'quant' | 'sparse16'.
    Non-eligible params (tied counted once) stay FP16."""
    seen = set(); non_elig = 0
    elig_ptrs = {lin.weight.data_ptr() for _, _, lin in eligible(model)}
    for n, p in model.named_parameters(remove_duplicate=False):
        if p.data_ptr() in seen: continue
        seen.add(p.data_ptr())
        if p.data_ptr() not in elig_ptrs: non_elig += p.numel()
    non_elig_bytes = 2 * non_elig
    out = dict(non_eligible_fp16_bytes=non_elig_bytes)
    n_el = sum(lin.weight.numel() for _, _, lin in eligible(model))
    if mode == "fp16":
        out.update(eligible_bytes=2 * n_el)
    elif mode == "sparse16":
        nnz = sum(int((lin.weight != 0).sum()) for _, _, lin in eligible(model))
        mask = sum(math.ceil(lin.weight.numel() / 8) for _, _, lin in eligible(model))  # per-matrix byte-aligned
        out.update(eligible_bytes=mask + 2 * nnz, mask_bytes=mask, value_bytes=2 * nnz, nnz=nnz)
    elif mode == "quant":
        pay = meta = 0; bits_sum = 0
        for i, p, lin in eligible(model):
            b = bits_map[(i, p)]; pl, mt = packed_bytes_matrix(*lin.weight.shape, b)
            pay += pl; meta += mt; bits_sum += b * lin.weight.numel()
        out.update(eligible_bytes=pay + meta, payload_bytes=pay, metadata_bytes=meta,
                   nominal_bits_per_weight=bits_sum / n_el, effective_bits_per_weight=8 * (pay + meta) / n_el)
    out["total_bytes"] = out["eligible_bytes"] + non_elig_bytes
    out["eligible_params"] = n_el
    return out


# ---------------- AWQ-style activation-aware scaling -----------------
ALPHA_GRID = [round(0.1 * i, 2) for i in range(11)]  # recorded candidate grid


def _group_specs(layer, cfg):
    """(name, input key, list of (proj, linear), fold target description)."""
    a, m = layer.self_attn, layer.mlp
    return [("qkv", "attn_in", [("q_proj", a.q_proj), ("k_proj", a.k_proj), ("v_proj", a.v_proj)]),
            ("o", "o_in", [("o_proj", a.o_proj)]),
            ("gateup", "mlp_in", [("gate_proj", m.gate_proj), ("up_proj", m.up_proj)]),
            ("down", "down_in", [("down_proj", m.down_proj)])]


def _gqa_expand(s_kv, n_heads, n_kv, hd):
    """Expand a scale over v_proj outputs (n_kv*hd) to o_proj inputs (n_heads*hd): head h uses kv head h//(n_heads/n_kv)."""
    rep = n_heads // n_kv
    return s_kv.view(n_kv, 1, hd).expand(n_kv, rep, hd).reshape(n_heads * hd)


def _gqa_reduce(x, n_heads, n_kv, hd):
    rep = n_heads // n_kv
    return x.view(n_kv, rep, hd).mean(1).reshape(n_kv * hd)


@torch.no_grad()
def awq_search_layer(layer, i, calib, bits, cfg, device="cuda"):
    """Grid-search alpha per scaling group: s = absmean^alpha / sqrt(max*min); minimize
    ||Q(W diag s) diag(1/s) X^T - W X^T||^2 on the fixed calibration token subset (original FP16 inputs)."""
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_heads
    res = {}
    for gname, key, lins in _group_specs(layer, cfg):
        X = calib[(i, key)]["X"].to(device).float()
        xm = calib[(i, key)]["absmean"].to(device).float()
        if gname == "o": xm = _gqa_expand(_gqa_reduce(xm, n_heads, n_kv, hd), n_heads, n_kv, hd)
        Ws = [l.weight.float() for _, l in lins]
        ref = [X @ W.t() for W in Ws]
        losses = []
        for a in ALPHA_GRID:
            s = xm.clamp(min=1e-4).pow(a)
            s = s / (s.max() * s.min()).sqrt()
            loss = 0.0
            for W, r in zip(Ws, ref):
                Wq = quantize_groups(W * s[None, :], bits) / s[None, :]
                loss += ((X @ Wq.t() - r) ** 2).sum().item()
            losses.append(loss / sum(r.numel() for r in ref))
        best = int(np.argmin(losses)); a = ALPHA_GRID[best]
        s = xm.clamp(min=1e-4).pow(a); s = s / (s.max() * s.min()).sqrt()
        if gname == "o": s = _gqa_reduce(s, n_heads, n_kv, hd)  # store kv-sized (exact since expanded)
        res[gname] = dict(alpha=a, losses=losses, loss_alpha0=losses[0], scale=s.cpu())
    return res


@torch.no_grad()
def apply_awq_scales(layer, scales, cfg):
    """Fold scales algebraically (no quantization yet). Returns per-proj (in_s, out_s) tracking vectors."""
    a, m = layer.self_attn, layer.mlp
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = cfg.hidden_size // n_heads
    dev = a.q_proj.weight.device
    s = scales["qkv"]["scale"].to(dev).float()
    for l in (a.q_proj, a.k_proj, a.v_proj): l.weight.copy_((l.weight.float() * s[None]).half())
    layer.input_layernorm.weight.copy_((layer.input_layernorm.weight.float() / s).half())
    track = {p: dict(in_s=s.clone()) for p in ("q_proj", "k_proj", "v_proj")}
    so = scales["o"]["scale"].to(dev).float()  # kv-sized
    a.v_proj.weight.copy_((a.v_proj.weight.float() / so[:, None]).half())
    if a.v_proj.bias is not None: a.v_proj.bias.copy_((a.v_proj.bias.float() / so).half())
    so_full = _gqa_expand(so, n_heads, n_kv, hd)
    a.o_proj.weight.copy_((a.o_proj.weight.float() * so_full[None]).half())
    track["v_proj"]["out_s"] = so.clone(); track["o_proj"] = dict(in_s=so_full.clone())
    s = scales["gateup"]["scale"].to(dev).float()
    for l in (m.gate_proj, m.up_proj): l.weight.copy_((l.weight.float() * s[None]).half())
    layer.post_attention_layernorm.weight.copy_((layer.post_attention_layernorm.weight.float() / s).half())
    track["gate_proj"] = dict(in_s=s.clone()); track["up_proj"] = dict(in_s=s.clone())
    sd = scales["down"]["scale"].to(dev).float()
    m.up_proj.weight.copy_((m.up_proj.weight.float() / sd[:, None]).half())
    m.down_proj.weight.copy_((m.down_proj.weight.float() * sd[None]).half())
    track["up_proj"]["out_s"] = sd.clone(); track["down_proj"] = dict(in_s=sd.clone())
    return track


GROUP_PROJS = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gateup": ("gate_proj", "up_proj"),
               "down": ("down_proj",)}


@torch.no_grad()
def quantize_model(model, bits_map, cfg=None, awq=None, blocks=None, only=None):
    """bits_map (block,proj)->bits (must be equal within an AWQ scaling group); awq: dict bits->per-layer scales or
    None (Q1). Each group's scale is the one searched at that group's bit width. All four (function-preserving)
    scalings are always folded; only projections in `only` (set of group names) are quantized if given.
    Returns tracking dict for error computation."""
    track = {}
    for i, layer in enumerate(model.model.layers):
        if blocks is not None and i not in blocks: continue
        if awq is not None:
            sc = {}
            for g, ps in GROUP_PROJS.items():
                bs = {bits_map[(i, p)] for p in ps}; assert len(bs) == 1, f"bits differ inside AWQ group {g}"
                sc[g] = awq[bs.pop()][i][g]
            t = apply_awq_scales(layer, sc, cfg)
            for p, v in t.items(): track[(i, p)] = v
        for g, ps in GROUP_PROJS.items():
            if only is not None and g not in only: continue
            for p in ps:
                lin = getattr(layer.self_attn if p in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp, p)
                lin.weight.copy_(quantize_groups(lin.weight, bits_map[(i, p)]).half())
    return track


@torch.no_grad()
def output_errors(model, orig_snap, calib, track=None, device="cuda"):
    """Relative output error per projection on fixed calibration token subset with original inputs:
    ||W_eff X - W X||^2 / ||W X||^2 where W_eff = diag(out_s) W_model diag(1/in_s)."""
    errs = {}
    for i, p, lin in eligible(model):
        X = calib[(i, INPUT_KEY[p])]["X"].to(device).float()
        W0 = orig_snap[f"model.layers.{i}.{'self_attn' if p in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'}.{p}.weight"].to(device).float()
        W = lin.weight.float()
        t = (track or {}).get((i, p), {})
        if "in_s" in t: W = W / t["in_s"].to(device)[None]
        if "out_s" in t: W = W * t["out_s"].to(device)[:, None]
        Y0 = X @ W0.t(); Y = X @ W.t()
        errs[(i, p)] = float(((Y - Y0) ** 2).sum() / (Y0 ** 2).sum())
    return errs
