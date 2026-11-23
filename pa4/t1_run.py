"""Task 1 driver.  python -m pa4.t1_run --stages baseline,calib,prune,p3,quant,q3 [--quick]
Every condition starts from the CPU snapshot of the unchanged FP16 weights; results are cached per condition."""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path
import numpy as np
import torch
from .common import OUT, SEED, seed_all, env_info, save_json, load_json, revisions, Timer, peak_mem
from . import t1_eval as E
from . import t1_compress as C
from .t1_eval import PROJS

R = OUT / "task1"; COND = R / "conditions"; ARR = R / "arrays"
CACHE = Path(os.environ.get("PA4_CACHE", "/tmp/pa4cache"))
MODEL = "Qwen/Qwen2.5-1.5B"
FULL_HS = {"fp16", "P1_s0.50", "P2_s0.50", "P3_s0.50", "Q1_b4", "Q2_b4", "Q3_b4", "Q1_b3", "Q2_b3", "Q3_b3"}
PROBE_N = 16  # first 16 calibration sequences (same IDs/tokenization) for isolated-block probes


class Ctx:
    pass


def setup(quick=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    seed_all(SEED)
    X = Ctx(); X.quick = quick
    rev = revisions().get("model_base")
    X.tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    X.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, torch_dtype=torch.float16,
                                                   attn_implementation="sdpa").cuda().eval()
    X.cfg = X.model.config
    X.snap = C.snapshot(X.model)
    X.wins = {k: E.load_windows(k) for k in ("wikitext2", "c4val")}
    hs = E.load_hellaswag()
    X.hs = E.hellaswag_requests(hs, X.tok)
    X.calib_seqs = E.load_c4("calibration"); X.dev_seqs = E.load_c4("development")
    assert len(X.calib_seqs) == 128 and len(X.dev_seqs) == 32
    assert not ({r["source_row"] for r in X.calib_seqs} & {r["source_row"] for r in X.dev_seqs})
    if quick:
        X.wins = {k: v[:6] for k, v in X.wins.items()}; X.hs = X.hs[:40]
    R.mkdir(parents=True, exist_ok=True); COND.mkdir(exist_ok=True); ARR.mkdir(exist_ok=True)
    save_json(R / "env.json", env_info(dict(model=MODEL, model_revision=rev, attn="sdpa", dtype="float16",
                                            calib_rows=[r["source_row"] for r in X.calib_seqs],
                                            dev_rows=[r["source_row"] for r in X.dev_seqs],
                                            hellaswag_convention="lm-eval-harness hellaswag v1.0 preprocessing; "
                                            "acc_norm = loglik / len(choice chars)",
                                            quick=quick)))
    X.calib = None
    return X


def restore(X):
    C.restore(X.model, X.snap)


def restore_block(X, i):
    with torch.no_grad():
        for n, p in X.model.model.layers[i].named_parameters():
            p.copy_(X.snap[f"model.layers.{i}.{n}"].to(p.device))


def get_calib(X):
    if X.calib is not None: return X.calib
    f = CACHE / "t1_calib.pt"
    if f.exists():
        X.calib = torch.load(f, weights_only=False)
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        restore(X); t0 = time.time()
        calib, meta = C.collect_calibration(X.model, X.calib_seqs)
        meta["seconds"] = time.time() - t0
        X.calib = calib; torch.save(calib, f); save_json(R / "calibration_meta.json", meta)
    return X.calib


def evaluate(X, name, info, track=None, storage_mode="fp16", bits_map=None, errors=True):
    """Full evaluation of the model currently loaded in X.model."""
    out = COND / f"{name}.json"
    if out.exists(): return load_json(out)
    full = name in FULL_HS
    res = dict(name=name, **info)
    res["nonzeros"] = C.count_nonzeros(X.model)
    res["storage_fp16_dense"] = C.model_storage(X.model, mode="fp16")
    if storage_mode == "sparse16": res["storage_sparse16"] = C.model_storage(X.model, mode="sparse16")
    if storage_mode == "quant": res["storage_packed"] = C.model_storage(X.model, bits_map=bits_map, mode="quant")
    if errors and X.calib is not None:
        errs = C.output_errors(X.model, X.snap, X.calib, track)
        res["calib_output_rel_err_mean"] = float(np.mean(list(errs.values())))
        res["calib_output_rel_err"] = {f"{i}.{p}": v for (i, p), v in errs.items()}
    with peak_mem() as pm:
        arrays = {}
        for k, w in X.wins.items():
            s, nll, cnt = E.perplexity(X.model, w)
            res[f"ppl_{k}"] = s; arrays[f"{k}_nll"] = nll; arrays[f"{k}_cnt"] = cnt
        reqs = X.hs if full else X.hs[:512]
        s, ll, acc, accn = E.hellaswag_eval(X.model, reqs, pad_id=X.tok.pad_token_id or 0)
        res["hellaswag_full" if full else "hellaswag_diag512"] = s
        if full: res["hellaswag_diag512_from_full"] = dict(acc=float(acc[:512].mean()), acc_norm=float(accn[:512].mean()))
        arrays["hs_ll"] = ll; arrays["hs_order"] = np.array([r["order"] for r in reqs])
    res["eval_peak_mem"] = pm
    res["latency"] = E.latency(X.model, X.cfg.vocab_size, reps=5 if not X.quick else 2)
    res["dev_nll_c4dev32"] = E.dev_nll(X.model, X.dev_seqs)
    np.savez_compressed(ARR / f"{name}.npz", **arrays)
    save_json(out, res)
    print(f"[{name}] wiki={res['ppl_wikitext2']['ppl']:.3f} c4={res['ppl_c4val']['ppl']:.3f} "
          f"hs={res.get('hellaswag_full', res.get('hellaswag_diag512'))}", flush=True)
    return res


# ---------------- stages -----------------
def stage_baseline(X):
    restore(X)
    enum = E.check_enumeration(X.model); save_json(R / "enumeration.json", enum); print(enum)
    # tokenization sanity: HellaSwag context/continuation boundary agreement
    evaluate(X, "fp16", dict(method="fp16"), errors=False)


def stage_calib(X):
    get_calib(X)


def stage_prune(X):
    calib = get_calib(X); T = Timer()
    for method, tag in (("magnitude", "P1"), ("wanda", "P2")):
        for s in (0.25, 0.50, 0.75):
            name = f"{tag}_s{s:.2f}"
            if (COND / f"{name}.json").exists(): continue
            restore(X)
            with T(name): st = C.apply_pruning(X.model, calib, method, sparsity=s)
            evaluate(X, name, dict(method=method, target_sparsity=s, prune=st, compress_seconds=T.t[name]),
                     storage_mode="sparse16")
    name = "P2_2of4_s0.50"
    if not (COND / f"{name}.json").exists():
        restore(X)
        with T(name): st = C.apply_pruning(X.model, calib, "wanda", nm=True)
        evaluate(X, name, dict(method="wanda_2:4", target_sparsity=0.5, prune=st, compress_seconds=T.t[name]),
                 storage_mode="sparse16")
    # per-projection Wanda / magnitude score distributions (for RQ1 and P3 justification)
    f = R / "score_distributions.json"
    if not f.exists():
        restore(X); d = {}
        qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        for i, p, lin in E.eligible(X.model):
            W = lin.weight.float().abs(); nrm = calib[(i, C.INPUT_KEY[p])]["norm"].cuda().float()
            sw = (W * nrm[None]).flatten(); sm = W.flatten()
            sub = torch.randperm(sw.numel(), generator=torch.Generator().manual_seed(SEED))[:200000].cuda()
            d[f"{i}.{p}"] = dict(
                wanda_q=torch.quantile(sw[sub], torch.tensor(qs, device="cuda")).tolist(),
                mag_q=torch.quantile(sm[sub], torch.tensor(qs, device="cuda")).tolist(),
                wanda_outlier_ratio_M5=float((sw > 5 * sw.mean()).float().mean()),
                act_norm_max_over_median=float(nrm.max() / nrm.median()),
                # overlap of pruned sets at 50%: fraction of magnitude-pruned weights also Wanda-pruned
                overlap50=_overlap(W, W * nrm[None], 0.5), overlap75=_overlap(W, W * nrm[None], 0.75))
        d["quantiles"] = qs
        save_json(f, d)


def _overlap(sm, sw, s):
    k = int(round(s * sm.shape[1]))
    a = C.row_prune_mask(sm, k); b = C.row_prune_mask(sw, k)
    return float(((~a) & (~b)).sum() / (~a).sum())


# ---- P3 -----
def owl_lod(X, M=5.0):
    calib = get_calib(X); restore(X); out = []
    for i, layer in enumerate(X.model.model.layers):
        sc = []
        for p in PROJS:
            lin = getattr(layer.self_attn if p in E.ATTN else layer.mlp, p)
            sc.append((lin.weight.float().abs() * calib[(i, C.INPUT_KEY[p])]["norm"].cuda().float()[None]).flatten())
        sc = torch.cat(sc); out.append(float((sc > M * sc.mean()).float().mean()))
    return np.array(out)


@torch.no_grad()
def isolated_prune_dnll(X, s):
    calib = get_calib(X); restore(X)
    probe = X.calib_seqs[:PROBE_N]; base = E.dev_nll(X.model, probe); out = []
    for i, layer in enumerate(X.model.model.layers):
        for p in PROJS:
            lin = getattr(layer.self_attn if p in E.ATTN else layer.mlp, p)
            sc = C.prune_score(lin.weight, "wanda", calib[(i, C.INPUT_KEY[p])]["norm"])
            lin.weight.mul_(C.row_prune_mask(sc, int(round(s * lin.weight.shape[1]))).half())
        out.append(E.dev_nll(X.model, probe) - base); restore_block(X, i)
    return np.array(out), base


def allocate(sens, S, lam, lo=0.10, hi=0.85):
    """Linear rank-free rule: s_l = S + lam*(1 - 2 r_l), r = minmax(sens) (more sensitive -> sparser less);
    then clip to [lo,hi] and shift by a common constant (bisection) so the mean equals S."""
    sens = np.asarray(sens, float)
    r = (sens - sens.min()) / max(sens.max() - sens.min(), 1e-12)
    base = S + lam * (1 - 2 * r)
    a, b = -1.0, 1.0
    for _ in range(100):
        c = (a + b) / 2
        if np.clip(base + c, lo, hi).mean() > S: b = c
        else: a = c
    return np.clip(base + (a + b) / 2, lo, hi)


def k_alloc_exact(X, s_blocks, S):
    """Per-row zero counts: floor(s_l*din) then distribute remaining zeros (+1 per row, round-robin over
    projections) so total zeros equal the uniform per-row-rounded P1/P2 budget exactly."""
    el = E.eligible(X.model)
    target = sum(int(round(S * l.weight.shape[1])) * l.weight.shape[0] for _, _, l in el)
    k = {(i, p): torch.full((l.weight.shape[0],), int(math.floor(s_blocks[i] * l.weight.shape[1])), dtype=torch.long)
         for i, p, l in el}
    R_ = target - sum(int(v.sum()) for v in k.values())
    keys = list(k.keys()); ptr = {kk: 0 for kk in keys}; j = 0
    while R_ > 0:
        kk = keys[j % len(keys)]
        if ptr[kk] < len(k[kk]):
            k[kk][ptr[kk]] += 1; ptr[kk] += 1; R_ -= 1
        j += 1
    assert R_ == 0
    return k, target


def stage_p3(X):
    calib = get_calib(X)
    f = R / "p3_selection.json"
    if f.exists(): sel = load_json(f)
    else:
        T = Timer()
        with T("owl"): lod = owl_lod(X)
        sens = {"owl_lod_M5": lod.tolist()}
        for S in (0.5, 0.75):
            with T(f"dnll_{S}"): d, base = isolated_prune_dnll(X, S)
            sens[f"isolated_dnll_s{S}"] = d.tolist()
        cands = []
        for S in (0.5, 0.75):
            for meas in ("owl_lod_M5", f"isolated_dnll_s{S}"):
                for lam in (0.0, 0.04, 0.08, 0.12, 0.16):
                    if lam == 0.0 and meas != "owl_lod_M5": continue
                    s_bl = allocate(sens[meas], S, lam)
                    restore(X)
                    k, target = k_alloc_exact(X, s_bl, S)
                    st = C.apply_pruning(X.model, calib, "wanda", k_alloc=k)
                    assert st["zeros"] == target
                    dn = E.dev_nll(X.model, X.dev_seqs)
                    cands.append(dict(S=S, measure=meas if lam > 0 else "uniform", lam=lam, dev_nll=dn,
                                      s_blocks=s_bl.tolist(), zeros=st["zeros"]))
                    print("P3 cand", S, meas, lam, round(dn, 4), flush=True)
        best = {str(S): min([c for c in cands if c["S"] == S and c["lam"] > 0], key=lambda c: c["dev_nll"])
                for S in (0.5, 0.75)}
        sel = dict(sensitivity=sens, candidates=cands, selected=best, timing_s=T.t,
                   rule="s_l = S + lam(1-2*minmax(sens_l)); clip [0.10,0.85]; common shift to mean S; "
                        "unit = Transformer block (all 7 projections); selection = min C4-dev(32) NLL; lam grid {.04,.08,.12,.16}",
                   probe=f"first {PROBE_N} calibration sequences")
        save_json(f, sel)
    for S in (0.5, 0.75):
        name = f"P3_s{S:.2f}"
        if (COND / f"{name}.json").exists(): continue
        b = sel["selected"][str(S)]; restore(X); t0 = time.time()
        k, target = k_alloc_exact(X, np.array(b["s_blocks"]), S)
        st = C.apply_pruning(X.model, calib, "wanda", k_alloc=k)
        evaluate(X, name, dict(method="wanda_nonuniform", target_sparsity=S, prune=st, allocation=b,
                               compress_seconds=time.time() - t0), storage_mode="sparse16")


# ---- quantization -----
def awq_scales(X, bits):
    f = CACHE / f"t1_awq_b{bits}.pt"
    if f.exists(): return torch.load(f, weights_only=False)
    calib = get_calib(X); restore(X); t0 = time.time()
    sc = [C.awq_search_layer(layer, i, calib, bits, X.cfg) for i, layer in enumerate(X.model.model.layers)]
    torch.save(sc, f)
    save_json(R / f"awq_search_b{bits}.json", dict(seconds=time.time() - t0, alpha_grid=C.ALPHA_GRID,
              per_layer=[{g: dict(alpha=v["alpha"], losses=v["losses"]) for g, v in l.items()} for l in sc]))
    return sc


def uniform_bits(X, b):
    return {(i, p): b for i in range(X.cfg.num_hidden_layers) for p in PROJS}


@torch.no_grad()
def verify_awq_equivalence(X, bits=4):
    f = R / "awq_equivalence_check.json"
    if f.exists(): return
    restore(X); sc = awq_scales(X, bits)
    ids = torch.tensor([r["token_ids"] for r in X.calib_seqs[:2]], device="cuda")
    ref = X.model(input_ids=ids).logits.float()
    for i, layer in enumerate(X.model.model.layers): C.apply_awq_scales(layer, sc[i], X.cfg)
    new = X.model(input_ids=ids).logits.float()
    d = (new - ref).abs()
    save_json(f, dict(bits_scales=bits, max_abs_logit_diff=float(d.max()), mean_abs_logit_diff=float(d.mean()),
                      ref_logit_abs_mean=float(ref.abs().mean()),
                      argmax_agreement=float((new.argmax(-1) == ref.argmax(-1)).float().mean()),
                      note="FP16 rescaling only (no rounding); differences come from FP16 re-rounding of folded weights"))
    restore(X)


def stage_quant(X):
    get_calib(X)
    for b in (8, 4, 3):
        name = f"Q1_b{b}"
        if (COND / f"{name}.json").exists(): continue
        restore(X); bm = uniform_bits(X, b); t0 = time.time()
        C.quantize_model(X.model, bm)
        evaluate(X, name, dict(method="rtn_group128_asym", bits=b, compress_seconds=time.time() - t0),
                 storage_mode="quant", bits_map=bm)
    for b in (2, 3, 4, 8): awq_scales(X, b)
    verify_awq_equivalence(X)
    # packing verification for each bit width on one real matrix
    f = R / "packing_check.json"
    if not f.exists():
        restore(X); W = X.model.model.layers[0].mlp.down_proj.weight; chk = {}
        for b in (2, 3, 4, 8):
            deq, q, s, z = C.quantize_groups(W, b, return_codes=True)
            pay = len(C.pack_bits(q, b)) ; meta = len(C.pack_bits(z, b)) + s.numel() * 2
            fp, fm = C.packed_bytes_matrix(*W.shape, b)
            chk[b] = dict(packed_payload=pay, formula_payload=fp, packed_meta_bytes_zero_bits_plus_fp16=meta,
                          formula_meta=fm, max_code=int(q.max()))
        save_json(f, chk)
    for b in (8, 4, 3):
        name = f"Q2_b{b}"
        if (COND / f"{name}.json").exists(): continue
        sc = awq_scales(X, b)
        restore(X); bm = uniform_bits(X, b); t0 = time.time()
        track = C.quantize_model(X.model, bm, cfg=X.cfg, awq={b: sc})
        evaluate(X, name, dict(method="awq_style_group128_asym", bits=b, compress_seconds=time.time() - t0,
                               awq_alphas=[{g: v["alpha"] for g, v in l.items()} for l in sc]),
                 track=track, storage_mode="quant", bits_map=bm)


@torch.no_grad()
def q3_sensitivity(X):
    """Isolated sensitivity of each (block, AWQ scaling group) at b in {2,3,4,8}: quantize only that group (all
    scalings folded, which is function-preserving), measure NLL increase on the probe set and the mean relative
    output error of its projections on the calibration token subset."""
    f = R / "q3_sensitivity.json"
    if f.exists(): return load_json(f)
    calib = get_calib(X); restore(X)
    probe = X.calib_seqs[:PROBE_N]; base = E.dev_nll(X.model, probe)
    sc = {b: awq_scales(X, b) for b in (2, 3, 4, 8)}
    dnll, oerr = {}, {}; t0 = time.time()
    for i in range(X.cfg.num_hidden_layers):
        for g, ps in C.GROUP_PROJS.items():
            for b in sc:
                bm = uniform_bits(X, b)
                track = C.quantize_model(X.model, bm, cfg=X.cfg, awq={b: sc[b]}, blocks={i}, only={g})
                dnll[f"{i}.{g}.{b}"] = E.dev_nll(X.model, probe) - base
                errs = []
                for p in ps:
                    lin = getattr(X.model.model.layers[i].self_attn if p in E.ATTN else X.model.model.layers[i].mlp, p)
                    Xs = calib[(i, C.INPUT_KEY[p])]["X"].cuda().float()
                    W0 = X.snap[f"model.layers.{i}.{'self_attn' if p in E.ATTN else 'mlp'}.{p}.weight"].cuda().float()
                    W = lin.weight.float(); t = track[(i, p)]
                    W = W / t["in_s"].cuda()[None]
                    if "out_s" in t: W = W * t["out_s"].cuda()[:, None]
                    Y0 = Xs @ W0.t(); errs.append(float(((Xs @ W.t() - Y0) ** 2).sum() / (Y0 ** 2).sum()))
                oerr[f"{i}.{g}.{b}"] = float(np.mean(errs))
                restore_block(X, i)
        print("q3 sens block", i, {g: round(dnll[f"{i}.{g}.2"], 4) for g in C.GROUP_PROJS}, flush=True)
    out = dict(base_nll=base, isolated_dnll=dnll, mean_rel_output_err=oerr, seconds=time.time() - t0,
               unit="(block, AWQ scaling group)", key="block.group.bits", probe=f"first {PROBE_N} calibration sequences")
    save_json(f, out); return out


@torch.no_grad()
def q3_context_sensitivity(X, b0):
    """Sensitivity around the operating point: start from uniform AWQ-style b0 quantization of the whole model,
    change one (block, group) unit to b in {2,3,4,8}, and record the probe-NLL change (captures interactions with the
    already-quantized rest of the model better than isolated probes)."""
    f = R / f"q3_context_sensitivity_b{b0}.json"
    if f.exists(): return load_json(f)
    restore(X); probe = X.calib_seqs[:PROBE_N]; t0 = time.time()
    sc = {b: awq_scales(X, b) for b in (2, 3, 4, 8)}
    base_bm = uniform_bits(X, b0)
    C.quantize_model(X.model, base_bm, cfg=X.cfg, awq=sc)
    base = E.dev_nll(X.model, probe); out = {}
    for i in range(X.cfg.num_hidden_layers):
        for g, ps in C.GROUP_PROJS.items():
            for b in (2, 3, 4, 8):
                if b == b0: out[f"{i}.{g}.{b}"] = 0.0; continue
                bm = dict(base_bm); bm.update({(i, p): b for p in ps})
                restore_block(X, i); C.quantize_model(X.model, bm, cfg=X.cfg, awq=sc, blocks={i})
                out[f"{i}.{g}.{b}"] = E.dev_nll(X.model, probe) - base
            restore_block(X, i); C.quantize_model(X.model, base_bm, cfg=X.cfg, awq=sc, blocks={i})
        print(f"q3 context b{b0} block", i, {g: round(out[f"{i}.{g}.2"], 4) for g in C.GROUP_PROJS}, flush=True)
    res = dict(b0=b0, base_nll=base, context_dnll=out, seconds=time.time() - t0, key="block.group.bits")
    save_json(f, res); restore(X); return res


def unit_bytes(X, g, b):
    layer = X.model.model.layers[0]; tot = 0
    for p in C.GROUP_PROJS[g]:
        lin = getattr(layer.self_attn if p in E.ATTN else layer.mlp, p)
        tot += sum(C.packed_bytes_matrix(*lin.weight.shape, b))
    return tot


def dp_allocate(cost, nbytes, budget, bits=(2, 3, 4, 8), quantum=4096):
    """Multiple-choice knapsack: minimize sum_u cost[u][b] s.t. sum_u bytes[u][b] <= budget.
    Exact DP over capacity measured in `quantum`-byte units with per-unit bytes rounded UP (so any returned
    assignment is feasible in exact bytes); backtracking table stored per unit."""
    U = len(cost); cap = budget // quantum
    w = [{b: -(-nbytes[u][b] // quantum) for b in bits} for u in range(U)]
    INF = 1e18
    best = np.full(cap + 1, INF); best[0] = 0.0
    choice = np.zeros((U, cap + 1), dtype=np.int8)
    for u in range(U):
        new = np.full(cap + 1, INF)
        for k, b in enumerate(bits):
            wb = w[u][b]
            if wb > cap: continue
            cand = np.full(cap + 1, INF); cand[wb:] = best[:cap + 1 - wb] + cost[u][b]
            m = cand < new; new[m] = cand[m]; choice[u][m] = k
        best = new
    c = int(np.argmin(best)); val = float(best[c]); asg = [0] * U
    for u in range(U - 1, -1, -1):
        b = bits[choice[u][c]]; asg[u] = b; c -= w[u][b]
    assert sum(nbytes[u][asg[u]] for u in range(U)) <= budget
    return asg, val


def stage_q3(X):
    sens = q3_sensitivity(X)
    nL = X.cfg.num_hidden_layers; G = list(C.GROUP_PROJS)
    units = [(i, g) for i in range(nL) for g in G]
    ub = {(g, b): unit_bytes(X, g, b) for g in G for b in (2, 3, 4, 8)}
    nbytes = [{b: ub[(g, b)] for b in (2, 3, 4, 8)} for (i, g) in units]
    sc = {b: awq_scales(X, b) for b in (2, 3, 4, 8)}
    def to_bits_map(asg):
        return {(i, p): b for (i, g), b in zip(units, asg) for p in C.GROUP_PROJS[g]}
    f = R / "q3_selection.json"
    if f.exists(): sel = load_json(f)
    else:
        cands = []
        for b0 in (4, 3):
            budget = sum(n[b0] for n in nbytes)  # = uniform Q2 eligible packed bytes (no extra metadata: scales folded)
            ctx = q3_context_sensitivity(X, b0)["context_dnll"]
            for meas in ("isolated_dnll", "mean_rel_output_err", f"context_dnll_b{b0}"):
                src = ctx if meas.startswith("context") else sens[meas]
                cost = [{b: src[f"{i}.{g}.{b}"] for b in (2, 3, 4, 8)} for (i, g) in units]
                asg, pred = dp_allocate(cost, nbytes, budget)
                restore(X); C.quantize_model(X.model, to_bits_map(asg), cfg=X.cfg, awq=sc)
                dn = E.dev_nll(X.model, X.dev_seqs)
                hist = {b: sum(1 for a in asg if a == b) for b in (2, 3, 4, 8)}
                cands.append(dict(budget_bits=b0, measure=meas, bits=asg, units=[f"{i}.{g}" for i, g in units],
                                  predicted_cost=pred, dev_nll=dn, bit_histogram=hist,
                                  eligible_bytes=sum(n[a] for n, a in zip(nbytes, asg)), budget_eligible_bytes=budget))
                print("Q3 cand", b0, meas, hist, round(dn, 4), flush=True)
        sel = dict(candidates=cands, unit_bytes={f"{g}.{b}": v for (g, b), v in ub.items()},
                   selected={str(b0): min([c for c in cands if c["budget_bits"] == b0], key=lambda c: c["dev_nll"])
                             for b0 in (4, 3)},
                   rule="unit = (block, AWQ scaling group: qkv | o | gate+up | down), 112 units; cost(u,b) = isolated "
                        "sensitivity at b (FP16 context) or context sensitivity around uniform b0; exact multiple-choice knapsack DP (4 KiB quantum, bytes rounded up) under the "
                        "packed-byte budget of uniform Q2; selection = min C4-dev(32) NLL")
        save_json(f, sel)
    for b0 in (4, 3):
        name = f"Q3_b{b0}"
        if (COND / f"{name}.json").exists(): continue
        asg = sel["selected"][str(b0)]["bits"]; bm = to_bits_map(asg)
        restore(X); t0 = time.time()
        track = C.quantize_model(X.model, bm, cfg=X.cfg, awq=sc)
        evaluate(X, name, dict(method="awq_style_mixed", budget_bits=b0, unit_bits=dict(zip(sel["selected"][str(b0)]["units"], asg)),
                               compress_seconds=time.time() - t0), track=track, storage_mode="quant", bits_map=bm)


STAGES = dict(baseline=stage_baseline, calib=stage_calib, prune=stage_prune, p3=stage_p3, quant=stage_quant, q3=stage_q3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="baseline,calib,prune,p3,quant,q3")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    X = setup(a.quick)
    for s in a.stages.split(","):
        t0 = time.time(); print(f"=== stage {s}", flush=True)
        STAGES[s](X)
        print(f"=== stage {s} done in {time.time()-t0:.1f}s", flush=True)
