"""Task 2A: dense causal attention, Python online-softmax tiled attention, PyTorch SDPA.
python -m pa4.t2a_attention"""
from __future__ import annotations
import math, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import OUT, SEED, env_info, save_json, summarize_ms, write_jsonl

R = OUT / "task2" / "A"
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
B, H, D = 2, 4, 64


def dense_attention(q, k, v, mask=None, causal=True):
    """Explicit QK^T / sqrt(d), causal masking (-inf), max-subtracted softmax in FP32, then @V.
    Fully masked rows return 0 (guarded) instead of NaN."""
    d = q.shape[-1]
    s = (q @ k.transpose(-1, -2)).float() / math.sqrt(d)
    L, S = s.shape[-2:]
    allowed = torch.ones(L, S, dtype=torch.bool, device=q.device)
    if causal: allowed = torch.tril(allowed, diagonal=S - L)
    if mask is not None: allowed = allowed & mask
    s = s.masked_fill(~allowed, float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(s - m)
    den = e.sum(-1, keepdim=True)
    p = e / den.clamp(min=torch.finfo(torch.float32).tiny)
    return (p.to(v.dtype) @ v)


def tiled_attention(q, k, v, bq=128, bk=128):
    """Exact causal attention via online softmax over key tiles; never materializes the LxL matrix.
    Running max m, normalizer l and accumulator o are FP32. Backward uses autograd through the Python loop
    (stores per-tile intermediates; NOT a fused FlashAttention kernel)."""
    Bb, Hh, L, d = q.shape
    scale = 1.0 / math.sqrt(d)
    outs = []
    for qs in range(0, L, bq):
        qe = min(qs + bq, L)
        qt = q[:, :, qs:qe].float() * scale
        m = torch.full((Bb, Hh, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((Bb, Hh, qe - qs, 1), device=q.device)
        o = torch.zeros((Bb, Hh, qe - qs, d), device=q.device)
        for ks in range(0, qe, bk):  # key tiles fully in the future are skipped
            ke = min(ks + bk, L)
            s = qt @ k[:, :, ks:ke].float().transpose(-1, -2)
            if ke > qs:  # tile intersects the diagonal -> apply causal mask
                qi = torch.arange(qs, qe, device=q.device)[:, None]; kj = torch.arange(ks, ke, device=q.device)[None]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new)
            l = alpha * l + p.sum(-1, keepdim=True)
            o = alpha * o + p @ v[:, :, ks:ke].float()
            m = m_new
        outs.append((o / l).to(q.dtype))
    return torch.cat(outs, 2)


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def detect_sdpa_backend(q, k, v):
    """Which backends accept these inputs (forced), and what default dispatch ran (via profiler kernel names)."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    ok = {}
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"):
        be = getattr(SDPBackend, name, None)
        if be is None: ok[name] = "not in this torch"; continue
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.cuda.synchronize()
            ok[name] = "supported"
        except Exception as e:
            ok[name] = f"unsupported: {str(e).splitlines()[0][:160]}"
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages()]
    kern = [n for n in names if any(t in n.lower() for t in ("flash", "fmha", "efficient", "attention", "cutlass", "softmax", "gemm"))]
    low = " ".join(kern).lower()
    dispatched = ("flash" if "flash" in low else "efficient(mem-efficient/cutlass)" if ("fmha" in low or "efficient" in low or "cutlass" in low)
                  else "math (unfused)" if ("softmax" in low or "gemm" in low) else "unknown")
    return dict(forced=ok, default_dispatch=dispatched, kernels=kern[:12])


def rel_err(a, b):
    return dict(max_abs=float((a.float() - b.float()).abs().max()),
                rel_fro=float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)))


def make_qkv(L, dtype, seed=SEED, req=True):
    g = torch.Generator(device="cuda").manual_seed(seed + L)
    t = [torch.randn(B, H, L, D, device="cuda", generator=g, dtype=torch.float32) for _ in range(3)]
    do = torch.randn(B, H, L, D, device="cuda", generator=g, dtype=torch.float32)
    return [x.to(dtype).requires_grad_(req) for x in t], do.to(dtype)


def correctness():
    res = []
    for L in (256, 512, 1024, 2048):
        (q32, k32, v32), do32 = make_qkv(L, torch.float32)
        ref = dense_attention(q32, k32, v32); ref.backward(do32)
        rg = [x.grad.clone() for x in (q32, k32, v32)]
        for dtype in (torch.float32, torch.float16):
            for name, fn in (("dense", dense_attention), ("tiled_128", tiled_attention), ("sdpa", sdpa),
                             ("tiled_32", lambda a, b, c: tiled_attention(a, b, c, 32, 32)),
                             ("tiled_64x256", lambda a, b, c: tiled_attention(a, b, c, 64, 256))):
                if name in ("tiled_32",) and L > 1024: continue
                q, k, v = [x.detach().to(dtype).requires_grad_(True) for x in (q32, k32, v32)]
                out = fn(q, k, v); out.backward(do32.to(dtype))
                r = dict(L=L, dtype=str(dtype).split(".")[-1], method=name, out=rel_err(out, ref),
                         finite=bool(torch.isfinite(out).all()))
                for gname, g, gr in zip("qkv", (q.grad, k.grad, v.grad), rg): r[f"grad_{gname}"] = rel_err(g, gr)
                res.append(r)
        del q32, k32, v32
    return res


def causality_and_edge_cases():
    out = {}
    (q, k, v), _ = make_qkv(512, torch.float32, req=False)
    i = 300
    k2, v2 = k.clone(), v.clone(); k2[:, :, i + 1:] += 5 * torch.randn_like(k2[:, :, i + 1:]); v2[:, :, i + 1:] += 5
    for name, fn in (("dense", dense_attention), ("tiled", tiled_attention), ("sdpa", sdpa)):
        a = fn(q, k, v)[:, :, :i + 1]; b = fn(q, k2, v2)[:, :, :i + 1]
        out[f"causality_{name}_max_change_upto_{i}"] = float((a - b).abs().max())
    # extreme scores: q scaled by 1e3 -> naive exp would overflow; stable versions must stay finite
    qx = q * 1e3
    for name, fn in (("dense", dense_attention), ("tiled", tiled_attention), ("sdpa", sdpa)):
        o = fn(qx, k, v); out[f"extreme_scores_{name}_finite"] = bool(torch.isfinite(o).all())
    out["extreme_dense_vs_tiled"] = rel_err(tiled_attention(qx, k, v), dense_attention(qx, k, v))
    out["naive_fp16_exp_overflow_example"] = bool(torch.isinf(torch.exp(((qx.half() @ k.half().transpose(-1, -2)) / 8).float().half())).any())
    # all-masked row edge case (custom mask with an empty row): dense guard returns zeros
    m = torch.ones(512, 512, dtype=torch.bool, device="cuda"); m[10] = False
    o = dense_attention(q, k, v, mask=m)
    out["all_masked_row_dense_output_abs_max"] = float(o[:, :, 10].abs().max())
    try:
        o2 = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tril(m))
        out["all_masked_row_sdpa_nan"] = bool(torch.isnan(o2[:, :, 10]).any())
    except Exception as e:
        out["all_masked_row_sdpa"] = str(e)[:200]
    return out


def bench(reps=10, warmup=3):
    rows = []
    for dtype in (torch.float16, torch.float32):
        for L in LENGTHS:
            for name, fn in (("dense", dense_attention), ("tiled_128", tiled_attention), ("sdpa", sdpa)):
                for mode in ("fwd", "fwd_bwd"):
                    r = dict(L=L, dtype=str(dtype).split(".")[-1], method=name, mode=mode)
                    try:
                        (q, k, v), do = make_qkv(L, dtype, req=(mode == "fwd_bwd"))
                        def run():
                            if mode == "fwd":
                                with torch.no_grad(): fn(q, k, v)
                            else:
                                for x in (q, k, v): x.grad = None
                                fn(q, k, v).backward(do)
                        for _ in range(warmup): run()
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        ts = []
                        n_rep = reps if not (name == "tiled_128" and L >= 4096 and mode == "fwd_bwd") else max(3, reps // 3)
                        for _ in range(n_rep):
                            torch.cuda.synchronize(); t0 = time.perf_counter(); run(); torch.cuda.synchronize()
                            ts.append((time.perf_counter() - t0) * 1e3)
                        r.update(summarize_ms(ts), status="ok",
                                 peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20,
                                 peak_over_inputs_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20,
                                 peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
                    except torch.cuda.OutOfMemoryError as e:
                        r.update(status="OOM")
                    except Exception as e:
                        r.update(status=f"error: {str(e)[:200]}")
                    finally:
                        q = k = v = do = None
                        torch.cuda.empty_cache()
                    rows.append(r); print(r.get("method"), r["dtype"], L, mode, r["status"], round(r.get("mean_ms", -1), 3), flush=True)
    return rows


def tiled_tile_sweep():
    rows = []
    (q, k, v), _ = make_qkv(2048, torch.float16, req=False)
    ref = dense_attention(q.float(), k.float(), v.float())
    for bq, bk in ((32, 32), (64, 64), (128, 128), (256, 256), (64, 256), (256, 64)):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        with torch.no_grad():
            o = tiled_attention(q, k, v, bq, bk)
        rows.append(dict(bq=bq, bk=bk, err=rel_err(o, ref),
                         extra_peak_MiB=(torch.cuda.max_memory_allocated() - base) / 2**20,
                         score_tile_bytes=B * H * bq * bk * 4, dense_score_bytes=B * H * 2048 * 2048 * 4))
    return rows


def main():
    R.mkdir(parents=True, exist_ok=True)
    save_json(R / "env.json", env_info(dict(B=B, H=H, D=D, lengths=LENGTHS)))
    be = {}
    for dtype in (torch.float16, torch.float32):
        (q, k, v), _ = make_qkv(1024, dtype, req=False)
        be[str(dtype)] = detect_sdpa_backend(q, k, v)
    save_json(R / "sdpa_backends.json", be); print(be)
    save_json(R / "causality_edge.json", causality_and_edge_cases())
    write_jsonl(R / "correctness.jsonl", correctness())
    write_jsonl(R / "tile_sweep.jsonl", tiled_tile_sweep())
    write_jsonl(R / "bench.jsonl", bench())


if __name__ == "__main__":
    main()
