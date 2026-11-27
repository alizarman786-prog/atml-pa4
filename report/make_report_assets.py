"""LaTeX tables, caption macros and PDF figures for the PA4 report, generated from the saved results (CPU only).

python report/make_report_assets.py --results results --out report_tex

Writes <out>/tables/*.tex (booktabs tabulars, \\input by main.tex), <out>/tables/notes.tex (macros used in captions),
<out>/figures/*.pdf, and copies report/main.tex and report/references.bib next to them. Every number comes from a
file under --results; a missing input only skips the affected table or figure (listed in <out>/MISSING.txt).
"""
from __future__ import annotations
import argparse, glob, gzip, json, math, shutil, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 6304
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 3.5,
                     "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
MISSING: list[str] = []
NOTES: dict[str, str] = {}
W1 = 5.5   # NeurIPS text width (in)


# ---------------------------------------------------------------- io / formatting
def jload(p): return json.loads(Path(p).read_text())


def jlines(p):
    p = Path(p); op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def num(v, nd=2, pct=False, sci_below=None):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))): return "--"
    if isinstance(v, (bool, np.bool_)): return "yes" if v else "no"
    if isinstance(v, str): return esc(v)
    v = float(v) * (100 if pct else 1)
    if sci_below is not None and v != 0 and abs(v) < sci_below:
        m, e = f"{v:.1e}".split("e"); return f"${m}{{\\times}}10^{{{int(e)}}}$"
    if abs(v) >= 1e5: m, e = f"{v:.2e}".split("e"); return f"${m}{{\\times}}10^{{{int(e)}}}$"
    return f"{v:,.{nd}f}"


def ci(lo, hi, nd=2, pct=False):
    if lo is None or hi is None or any(isinstance(x, float) and math.isnan(x) for x in (lo, hi)): return "--"
    return f"[{num(lo, nd, pct)}, {num(hi, nd, pct)}]"


def tabular(rows, header, align, out, name, midrules=(), header2=None):
    """rows: list of lists of already-formatted cells."""
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    if header2: lines.append(header2)
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        if i in midrules: lines.append(r"\midrule")
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"--- tables/{name}.tex ({len(rows)} rows)")


def boot_mean(x, n=2000, seed=SEED):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ppl_diff(a, b, n=2000, seed=SEED):
    cnt = np.asarray(a["cnt"], float)
    if not np.array_equal(cnt, np.asarray(b["cnt"], float)): return np.nan, np.nan, np.nan
    na, nb = np.asarray(a["nll"], float), np.asarray(b["nll"], float)
    idx = np.random.default_rng(seed).integers(0, len(cnt), size=(n, len(cnt)))
    d = np.exp(nb[idx].sum(1) / cnt[idx].sum(1)) - np.exp(na[idx].sum(1) / cnt[idx].sum(1))
    return (math.exp(nb.sum() / cnt.sum()) - math.exp(na.sum() / cnt.sum()),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def savefig(fig, out, name):
    try:
        if not getattr(fig, "_gs_layout", False): fig.tight_layout(w_pad=1.2)
    except Exception: pass
    fig.savefig(out / "figures" / f"{name}.pdf"); fig.savefig(out / "figures" / f"{name}.png", dpi=150)
    plt.close(fig); print(f"--- figures/{name}.pdf")


def note(key, val): NOTES[key] = str(val)


def section(fn):
    def run(R, out):
        try: fn(R, out)
        except Exception as e:
            MISSING.append(f"{fn.__name__}: {type(e).__name__}: {e}"); traceback.print_exc()
    run.__name__ = fn.__name__
    return run


def panel(ax, tag, title=""):
    ax.set_title(f"({tag}) {title}".strip(), loc="left")
    from matplotlib.ticker import NullFormatter
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log": axis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------- Task 1
T1_ORDER = ["fp16", "P1_s0.25", "P1_s0.50", "P1_s0.75", "P2_s0.25", "P2_s0.50", "P2_s0.75", "P2_2of4_s0.50",
            "P3_s0.50", "P3_s0.75", "Q1_b3", "Q1_b4", "Q1_b8", "Q2_b3", "Q2_b4", "Q2_b8", "Q3_b3", "Q3_b4"]
T1_LABEL = {"fp16": "FP16", "P2_2of4_s0.50": "P2 2:4"}


def t1_label(n):
    if n in T1_LABEL: return T1_LABEL[n]
    a, b = n.split("_", 1)
    return f"{a} {b.replace('s0.', 's=.').replace('b', '')}" + (" bit" if b.startswith("b") else "")


def t1_conditions(R):
    C = R / "task1" / "conditions"
    rows = {Path(f).stem: jload(f) for f in glob.glob(str(C / "*.json"))}
    return {k: rows[k] for k in T1_ORDER if k in rows} | {k: v for k, v in rows.items() if k not in T1_ORDER}


def t1_arrays(R, name):
    f = R / "task1" / "arrays" / f"{name}.npz"
    if not f.exists(): return None
    a = np.load(f); return {k: a[k] for k in a.files}


def storage(r):
    for k in ("storage_packed", "storage_sparse16", "storage_fp16_dense"):
        v = r.get(k)
        if isinstance(v, dict) and "total_bytes" in v: return v["total_bytes"] / 2**30, k.split("_", 1)[1], v
    return np.nan, "", {}


def hs(r):
    if r.get("hellaswag_full"): return r["hellaswag_full"], 2000
    h = r.get("hellaswag_diag512") or {}
    return h, h.get("n", 512)


def tex_text(s, n):
    """Plain-ASCII, LaTeX-escaped snippet of model or dataset text (for qualitative tables)."""
    s = " ".join(str(s).split()).encode("ascii", "replace").decode()
    s = s[:n] + ("..." if len(s) > n else "")
    s = s.replace("\\", "\x00")
    for a, b in (("{", r"\{"), ("}", r"\}"), ("$", r"\$"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_"), ("%", r"\%"),
                 ("^", r"\^{}"), ("~", r"\~{}"), ("<", r"\textless{}"), (">", r"\textgreater{}"), ("|", r"\textbar{}")):
        s = s.replace(a, b)
    return s.replace("\x00", r"\textbackslash{}")


def _hs_pre(text):
    import re
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def hs_rows():
    """Locked HellaSwag rows keyed by 'order' (needed for gold labels and ending lengths)."""
    import os
    roots = [os.environ.get("PA4_DATA", ""), "/kaggle/working/atml-pa4-datasets", "/kaggle/working"]
    for r in [r for r in roots if r]:
        hits = glob.glob(str(Path(r) / "**" / "hellaswag_val_locked.jsonl.gz"), recursive=True)
        if hits: return {x["order"]: x for x in jlines(hits[0])}
    return None


def hs_correct(arr, orders, hsd):
    """acc_norm correctness per example (lm-eval-harness convention: log-lik / character length)."""
    idx = {o: i for i, o in enumerate(arr["hs_order"].tolist())}
    ll = arr["hs_ll"][[idx[o] for o in orders]]
    clen = np.array([[len(_hs_pre(e)) for e in hsd[o]["example"]["endings"]] for o in orders], float)
    gold = np.array([int(hsd[o]["example"]["label"]) for o in orders])
    pred = (ll / clen).argmax(1)
    return pred == gold, pred, gold


@section
def task1_tables(R, out):
    rows = t1_conditions(R)
    if not rows: raise FileNotFoundError("task1/conditions")
    base = t1_arrays(R, "fp16")
    fp = rows.get("fp16", {})
    note("TOneWikiN", f"{fp.get('ppl_wikitext2', {}).get('N', '--'):,}" if isinstance(fp.get('ppl_wikitext2', {}).get('N'), int) else "--")
    note("TOneCFourN", f"{fp.get('ppl_c4val', {}).get('N', '--'):,}" if isinstance(fp.get('ppl_c4val', {}).get('N'), int) else "--")
    body, mids, last = [], [], None
    for name, r in rows.items():
        fam = name[:2] if name != "fp16" else "fp"
        if last is not None and fam != last and fam in ("P1", "P3", "Q1", "Q3"): mids.append(len(body))
        if last is not None and fam.startswith("Q") and not last.startswith("Q"): mids.append(len(body))
        last = fam
        gib, kind, st = storage(r)
        nz = r.get("nonzeros") or {}
        if r.get("target_sparsity") is not None: achieved = num(nz.get("sparsity"), 3)
        elif st.get("effective_bits_per_weight") is not None: achieved = num(st["effective_bits_per_weight"], 2) + " b/w"
        else: achieved = "--"
        w, c = r.get("ppl_wikitext2", {}), r.get("ppl_c4val", {})
        dw = dc = "--"
        arr = t1_arrays(R, name)
        if base is not None and arr is not None and name != "fp16":
            for key, holder in (("wikitext2", "dw"), ("c4val", "dc")):
                if f"{key}_nll" in arr and f"{key}_nll" in base:
                    d, lo, hi = boot_ppl_diff(dict(nll=base[f"{key}_nll"], cnt=base[f"{key}_cnt"]),
                                              dict(nll=arr[f"{key}_nll"], cnt=arr[f"{key}_cnt"]))
                    cell = f"{num(d, 2)} {ci(lo, hi, 2)}" if abs(d) < 1e4 else num(d, 0)
                    if holder == "dw": dw = cell
                    else: dc = cell
        h, n = hs(r)
        hcell = num(h.get("acc_norm"), 1, pct=True) + ("" if n == 2000 else r"$^\dagger$")
        body.append([esc(t1_label(name)), achieved, num(gib, 2), num(w.get("ppl"), 2), dw, num(c.get("ppl"), 2), dc,
                     hcell, num(r.get("dev_nll_c4dev32"), 3)])
    tabular(body, ["Condition", "Achieved", "GiB", "Wiki PPL", r"$\Delta$Wiki [95\% CI]", "C4 PPL",
                   r"$\Delta$C4 [95\% CI]", "HS acc$_n$", "Dev NLL"], "lrrrrrrrr", out, "t1_main", midrules=sorted(set(mids)))

    body = []
    for name, r in rows.items():
        lat = r.get("latency") or {}
        g = lambda k: (lat.get(k) or {}).get("median_ms")
        gib, kind, st = storage(r)
        body.append([esc(t1_label(name)), esc(kind or "--"), num(r.get("compress_seconds"), 1),
                     num(r.get("calib_output_rel_err_mean"), 4), num(g("prefill_512"), 0), num(g("prefill_2048"), 0),
                     num(g("decode_64_after_512"), 0)])
    tabular(body, ["Condition", "Storage format", "Compress (s)", "Calib.\\ rel.\\ err", "Prefill 512 (ms)",
                   "Prefill 2048 (ms)", "Decode 64 (ms)"], "llrrrrr", out, "t1_cost")


@section
def task1_selection(R, out):
    rows = t1_conditions(R)
    body = []
    p3 = jload(R / "task1" / "p3_selection.json")
    for S in ("0.5", "0.75"):
        c = [x for x in p3["candidates"] if str(x["S"]) == S]
        uni = next(x for x in c if x["lam"] == 0)
        for x in sorted(c, key=lambda x: (x["measure"] != "uniform", x["measure"], x["lam"])):
            sel = x is p3["selected"].get(S) or (x["measure"] == p3["selected"][S]["measure"] and x["lam"] == p3["selected"][S]["lam"])
            body.append([f"P3 $S$={S}", esc(x["measure"]), num(x["lam"], 2), num(min(x["s_blocks"]), 2) + "--" + num(max(x["s_blocks"]), 2),
                         num(x["dev_nll"], 4), num(x["dev_nll"] - uni["dev_nll"], 4), r"\checkmark" if sel and x["lam"] > 0 else ""])
    q3 = jload(R / "task1" / "q3_selection.json")
    for b0 in ("4", "3"):
        q2 = rows.get(f"Q2_b{b0}", {}).get("dev_nll_c4dev32")
        if q2 is not None:
            body.append([f"Q3 $b_0$={b0}", "uniform (= Q2)", "--", f"all {b0}", num(q2, 4), num(0.0, 4), ""])
        for x in [x for x in q3["candidates"] if str(x["budget_bits"]) == b0]:
            h = x["bit_histogram"]; hist = "/".join(str(h.get(str(b), h.get(b, 0))) for b in (2, 3, 4, 8))
            sel = x["measure"] == q3["selected"][b0]["measure"]
            body.append([f"Q3 $b_0$={b0}", esc(x["measure"]), "--", hist, num(x["dev_nll"], 4),
                         num(x["dev_nll"] - q2, 4) if q2 is not None else "--", r"\checkmark" if sel else ""])
    tabular(body, ["Rule", "Sensitivity measure", r"$\lambda$", r"$s_\ell$ range / units at 2/3/4/8 b", "Dev NLL",
                   r"$\Delta$ vs uniform", "Sel."], "lllcrrc", out, "t1_selection")
    note("TOneQThreeUnits", len(q3["candidates"][0]["bits"]) if q3.get("candidates") else "--")


@section
def task1_scores(R, out):
    d = jload(R / "task1" / "score_distributions.json")
    recs = []
    for k, v in d.items():
        if k == "quantiles" or not isinstance(v, dict): continue
        blk, proj = k.split(".", 1)
        recs.append(dict(block=int(blk), proj=proj, **{kk: v[kk] for kk in ("overlap50", "overlap75", "act_norm_max_over_median", "wanda_outlier_ratio_M5")}))
    df = pd.DataFrame(recs)
    order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    body = []
    for p in [p for p in order if p in set(df.proj)]:
        g = df[df.proj == p]
        body.append([esc(p), num(g.overlap50.mean(), 3), num(g.overlap50.min(), 3), num(g.overlap75.mean(), 3),
                     num(g.act_norm_max_over_median.median(), 1), num(g.act_norm_max_over_median.max(), 1),
                     num(100 * g.wanda_outlier_ratio_M5.mean(), 3)])
    tabular(body, ["Projection", "Overlap@50\\% (mean)", "(min)", "Overlap@75\\%", r"$\max/\mathrm{med}\ \|X_j\|$ (median)",
                   "(max)", "Outliers $>5\\bar{S}$ (\\%)"], "lrrrrrr", out, "t1_scores")
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.7))
    for i, p in enumerate([p for p in order if p in set(df.proj)]):
        g = df[df.proj == p].sort_values("block")
        ax[0].plot(g.block, g.overlap50, marker=MARK[i % 8], color=PAL[i % 8], label=p, markevery=3)
        ax[1].plot(g.block, g.act_norm_max_over_median, marker=MARK[i % 8], color=PAL[i % 8], label=p, markevery=3)
    ax[0].set_xlabel("block"); ax[0].set_ylabel("kept-set overlap, magnitude vs Wanda (50%)")
    ax[1].set_xlabel("block"); ax[1].set_ylabel(r"max / median input-feature norm"); ax[1].set_yscale("log")
    panel(ax[0], "a", "pruned-set overlap"); panel(ax[1], "b", "activation outliers"); ax[1].legend(ncol=2, loc="upper left")
    savefig(fig, out, "fig_task1_scores")


@section
def task1_costnotes(R, out):
    """One-time costs that the per-condition compress time excludes (calibration pass, AWQ search, sensitivity probes)."""
    T = R / "task1"
    if (T / "calibration_meta.json").exists(): note("TOneCalibS", num(jload(T / "calibration_meta.json").get("seconds"), 0))
    aw = {b: jload(T / f"awq_search_b{b}.json").get("seconds") for b in (2, 3, 4, 8) if (T / f"awq_search_b{b}.json").exists()}
    if aw: note("TOneAwqS", f"{num(min(aw.values()), 0)}--{num(max(aw.values()), 0)}")
    q = [jload(f).get("seconds", 0) for f in [T / "q3_sensitivity.json", T / "q3_context_sensitivity_b3.json", T / "q3_context_sensitivity_b4.json"] if f.exists()]
    if q: note("TOneQThreeProbeMin", num(sum(q) / 60, 0))
    if (T / "p3_selection.json").exists():
        t = jload(T / "p3_selection.json").get("timing_s") or {}
        note("TOnePThreeProbeMin", num(sum(v for v in t.values() if isinstance(v, (int, float))) / 60, 0))


T1_PAIRS = [("P2_s0.50", "P3_s0.50"), ("P2_s0.75", "P3_s0.75"), ("Q2_b3", "Q3_b3"), ("Q2_b4", "Q3_b4")]


@section
def task1_paired(R, out):
    """Paired 95% bootstrap CIs for P3-P2 and Q3-Q2 (whole windows for PPL, examples for HellaSwag)."""
    hsd, rows, conds = hs_rows(), [], t1_conditions(R)
    for a_, b_ in T1_PAIRS:
        A, B = t1_arrays(R, a_), t1_arrays(R, b_)
        if A is None or B is None: continue
        cells = [f"{esc(t1_label(b_))} $-$ {esc(t1_label(a_))}"]
        for key in ("wikitext2", "c4val"):
            d, lo, hi = boot_ppl_diff(dict(nll=A[f"{key}_nll"], cnt=A[f"{key}_cnt"]), dict(nll=B[f"{key}_nll"], cnt=B[f"{key}_cnt"]))
            cells.append(f"{num(d, 2)} {ci(lo, hi, 2)}")
        if hsd is not None and "hs_ll" in A and "hs_ll" in B:
            ob = set(B["hs_order"].tolist()); common = [o for o in A["hs_order"].tolist() if o in ob]
            ca, _, _ = hs_correct(A, common, hsd); cb, _, _ = hs_correct(B, common, hsd)
            for n_, c_ in ((a_, ca), (b_, cb)):
                h, nn = hs(conds.get(n_, {}))
                print(f"  check {n_}: recomputed acc_norm on {len(common)} = {c_.mean():.4f}; stored ({nn}) = {h.get('acc_norm')}")
            m, lo, hi = boot_mean(cb.astype(float) - ca.astype(float))
            cells += [f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", f"{len(common):,}"]
        else:
            cells += ["--", "--"]
        rows.append(cells)
    tabular(rows, ["Comparison", r"$\Delta$Wiki PPL [95\% CI]", r"$\Delta$C4 PPL [95\% CI]", r"$\Delta$HS acc$_n$ (pp) [95\% CI]",
                   "HS $n$"], "lrrrr", out, "t1_paired")


T1_QUAL = ["fp16", "P2_s0.50", "P3_s0.50", "Q2_b3", "Q3_b3"]


@section
def task1_qual(R, out):
    """Fixed selection: the first two examples in locked order, then the first two (in order) where FP16 is correct and
    P3 s=.50 or Q3 3-bit picks a different ending than FP16."""
    hsd = hs_rows()
    if hsd is None: raise FileNotFoundError("hellaswag_val_locked.jsonl.gz")
    arrs = {n: t1_arrays(R, n) for n in T1_QUAL}
    if any(a is None or "hs_ll" not in a for a in arrs.values()): raise FileNotFoundError("task1 arrays")
    orders = sorted(set.intersection(*[set(a["hs_order"].tolist()) for a in arrs.values()]))
    P = {n: hs_correct(a, orders, hsd) for n, a in arrs.items()}
    gold = P["fp16"][2]
    pick = orders[:2]
    for i, o in enumerate(orders):
        if len(pick) >= 4: break
        if o in pick: continue
        if P["fp16"][0][i] and (P["P3_s0.50"][1][i] != P["fp16"][1][i] or P["Q3_b3"][1][i] != P["fp16"][1][i]): pick.append(o)
    rows = []
    for o in pick:
        i = orders.index(o); ex = hsd[o]["example"]
        cells = [str(o), tex_text(_hs_pre(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"]), 48),
                 tex_text(_hs_pre(ex["endings"][gold[i]]), 30) + f" ({gold[i]})"]
        for n in T1_QUAL:
            cells.append(f"{P[n][1][i]}" + (r" \checkmark" if P[n][0][i] else r" $\times$"))
        rows.append(cells)
    tabular(rows, ["Order", "Context (truncated)", "Gold ending (index)"] + [esc(t1_label(n)) for n in T1_QUAL], "rll" + "c" * len(T1_QUAL),
            out, "t1_qual")


@section
def task1_figure(R, out):
    rows = t1_conditions(R)
    fp = rows.get("fp16")
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.85))
    fams = [("P1", "magnitude"), ("P2", "Wanda"), ("P3", "Wanda non-uniform")]
    for i, (pre, lab) in enumerate(fams):
        pts = sorted([(r["target_sparsity"], r["ppl_wikitext2"]["ppl"], hs(r)) for n, r in rows.items()
                      if n.startswith(pre + "_s")], key=lambda t: t[0])
        if not pts: continue
        x = [p[0] for p in pts]
        ax[0].plot(x, [p[1] for p in pts], marker=MARK[i], color=PAL[i], label=lab)
        ax[1].plot(x, [p[2][0].get("acc_norm", np.nan) * 100 for p in pts], color=PAL[i], label=lab)
        for xx, p in zip(x, pts):
            ax[1].plot(xx, p[2][0].get("acc_norm", np.nan) * 100, marker=MARK[i], color=PAL[i],
                       markerfacecolor=PAL[i] if p[2][1] == 2000 else "white")
    if "P2_2of4_s0.50" in rows:
        r = rows["P2_2of4_s0.50"]
        ax[0].plot(0.5, r["ppl_wikitext2"]["ppl"], marker=MARK[3], color=PAL[3], linestyle="none", label="Wanda 2:4")
        ax[1].plot(0.5, hs(r)[0].get("acc_norm", np.nan) * 100, marker=MARK[3], color=PAL[3], linestyle="none",
                   markerfacecolor=PAL[3] if hs(r)[1] == 2000 else "white")
    if fp:
        ax[0].axhline(fp["ppl_wikitext2"]["ppl"], color=INK2, ls="--", lw=0.8, label="FP16")
        ax[1].axhline(hs(fp)[0].get("acc_norm", np.nan) * 100, color=INK2, ls="--", lw=0.8)
        ax[1].axhline(25, color=INK2, ls=":", lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_xlabel("target sparsity"); ax[0].set_ylabel("WikiText-2 PPL (log)")
    ax[0].legend(loc="upper left")
    ax[1].set_xlabel("target sparsity"); ax[1].set_ylabel("HellaSwag acc$_{norm}$ (%)")
    qs = [("Q1", "RTN"), ("Q2", "AWQ-style"), ("Q3", "mixed")]
    for i, (pre, lab) in enumerate(qs):
        pts = sorted([(storage(r)[0], r["ppl_wikitext2"]["ppl"], n) for n, r in rows.items() if n.startswith(pre + "_b")])
        if pts: ax[2].plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[i + 4], color=PAL[i + 4], label=lab)
    if fp:
        ax[2].plot(storage(fp)[0], fp["ppl_wikitext2"]["ppl"], marker="*", color=INK, linestyle="none", markersize=6, label="FP16")
    ax[2].set_xlabel("packed model size (GiB)"); ax[2].set_ylabel("WikiText-2 PPL"); ax[2].legend()
    for a, t, ti in zip(ax, "abc", ("perplexity vs sparsity", "HellaSwag vs sparsity", "quantization: size vs PPL")): panel(a, t, ti)
    savefig(fig, out, "fig_task1")


@section
def task1_alloc_figure(R, out):
    p3 = jload(R / "task1" / "p3_selection.json"); q3 = jload(R / "task1" / "q3_selection.json")
    fig = plt.figure(figsize=(W1, 1.9)); fig._gs_layout = True
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=0.6, wspace=0.28)
    a0 = fig.add_subplot(gs[:, 0])
    for i, S in enumerate(("0.5", "0.75")):
        s = p3["selected"][S]["s_blocks"]
        a0.plot(range(len(s)), s, marker=MARK[i], color=PAL[i], markevery=3,
                label=f"S={S}: {p3['selected'][S]['measure']}, $\\lambda$={p3['selected'][S]['lam']}")
        a0.axhline(float(S), color=PAL[i], ls=":", lw=0.8)
    a0.set_xlabel("block"); a0.set_ylabel("block sparsity $s_\\ell$"); a0.legend(loc="lower left", fontsize=5); panel(a0, "a", "P3 block sparsity (selected)")
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#cfe0f6", "#7fb0ea", "#2a78d6", "#0d3b75"]); norm = BoundaryNorm([1.5, 2.5, 3.5, 6, 9], cmap.N)
    for j, b0 in enumerate(("3", "4")):
        a = fig.add_subplot(gs[j, 1])
        sel = q3["selected"][b0]; groups = sorted({u.split(".", 1)[1] for u in sel["units"]}, key=["qkv", "o", "gate_up", "down"].index
                                                    if all(g in ["qkv", "o", "gate_up", "down"] for g in {u.split(".", 1)[1] for u in sel["units"]}) else str)
        nb = max(int(u.split(".")[0]) for u in sel["units"]) + 1
        M = np.full((len(groups), nb), np.nan)
        for u, b in zip(sel["units"], sel["bits"]):
            i, g = u.split(".", 1); M[groups.index(g), int(i)] = b
        im = a.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        a.set_yticks(range(len(groups))); a.set_yticklabels(groups); a.grid(False)
        a.set_title(f"({'bc'[j]}) Q3 $b_0$={b0}: {sel['measure']}", loc="left", fontsize=6.5)
        if j == 1: a.set_xlabel("block")
        else: a.set_xticklabels([])
    cb = fig.colorbar(im, ax=fig.axes[1:], ticks=[2, 3, 4, 8], fraction=0.04, pad=0.02); cb.set_label("bits", fontsize=6)
    savefig(fig, out, "fig_task1_alloc")


# ---------------------------------------------------------------- Task 2A
def err_cell(v, key):
    if isinstance(v, dict): return v.get(key, np.nan)
    return v


@section
def task2a(R, out):
    D = R / "task2" / "A"
    c = pd.DataFrame(jlines(D / "correctness.jsonl"))
    body = []
    for dt in ("float16", "float32"):
        for m in [m for m in ("dense", "tiled_128", "tiled_32", "tiled_64x256", "sdpa") if m in set(c.method)]:
            g = c[(c.dtype == dt) & (c.method == m)]
            if g.empty: continue
            cells = [esc(m), dt.replace("float", "FP"), f"{int(g.L.min())}--{int(g.L.max())}"]
            for col in ("out", "grad_q", "grad_k", "grad_v"):
                cells.append(num(max(err_cell(v, "rel_fro") for v in g[col]), 1, sci_below=1e-2))
            cells.append(num(max(err_cell(v, "max_abs") for v in g["out"]), 1, sci_below=1e-2))
            cells.append(r"\checkmark" if g.get("finite", pd.Series([True])).all() else "no")
            body.append(cells)
    tabular(body, ["Method", "dtype", "$L$", "Out rel.\\ err", "$\\nabla Q$", "$\\nabla K$", "$\\nabla V$", "Out max-abs", "Finite"],
            "llcrrrrrc", out, "t2a_fidelity")
    b = pd.DataFrame(jlines(D / "bench.jsonl"))
    if "status" not in b: b["status"] = "ok"
    Ls = sorted(b.L.unique()); keepL = [L for L in Ls if L in (1024, 4096, 8192)] or Ls[-3:]
    body = []
    for mode in ("fwd", "fwd_bwd"):
        for m in ("dense", "tiled_128", "sdpa"):
            g = b[(b.dtype == "float16") & (b["mode"] == mode) & (b.method == m)]
            cells = [mode.replace("_", "+"), esc(m)]
            for L in keepL:
                r = g[g.L == L]
                if r.empty: cells += ["--", "--"]; continue
                r = r.iloc[0]
                if r.status != "ok": cells += [esc(str(r.status)[:12]), "--"]; continue
                cells += [num(r.median_ms, 1), num(r.get("peak_over_inputs_MiB", np.nan), 0)]
            body.append(cells)
    hdr2 = " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$L$={L:,}}}" for L in keepL) + r" \\"
    tabular(body, ["Pass", "Method"] + ["ms", "MiB"] * len(keepL), "ll" + "rr" * len(keepL), out, "t2a_bench",
            midrules=[3], header2=hdr2)
    note("TTwoABenchReps", r"10 timed runs after 3 warm-up runs (3 timed runs for tiled fwd+bwd at $L\ge 4096$)")
    env = R / "task2" / "A" / "env.json"
    if env.exists():
        e = jload(env)
        note("TTwoAShape", f"$B$={e.get('B', '?')}, $H$={e.get('H', '?')}, $d$={e.get('D', '?')}")
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.8))
    for i, m in enumerate(("dense", "tiled_128", "sdpa")):
        for j, mode in enumerate(("fwd", "fwd_bwd")):
            g = b[(b.dtype == "float16") & (b["mode"] == mode) & (b.method == m) & (b.status == "ok")].sort_values("L")
            ax[j].plot(g.L, g.median_ms, marker=MARK[i], color=PAL[i], label=m)
            if mode == "fwd_bwd" and "peak_over_inputs_MiB" in g:
                ax[2].plot(g.L, g.peak_over_inputs_MiB, marker=MARK[i], color=PAL[i], label=m)
            oom = b[(b.dtype == "float16") & (b["mode"] == mode) & (b.method == m) & (b.status != "ok")]
            for L in oom.L: ax[j].annotate("OOM", (L, ax[j].get_ylim()[1] if ax[j].lines else 1), fontsize=5, color=PAL[i], ha="center")
    for a, t, yl in zip(ax, ("forward", "forward+backward", "peak extra memory, fwd+bwd"), ("median ms", "median ms", "MiB above inputs")):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("sequence length $L$"); a.set_ylabel(yl)
    ax[0].legend()
    for a, t, ti in zip(ax, "abc", ("forward, FP16", "forward+backward, FP16", "peak memory, fwd+bwd")): panel(a, t, ti)
    savefig(fig, out, "fig_task2a")
    if (D / "tile_sweep.jsonl").exists():
        ts = jlines(D / "tile_sweep.jsonl")
        tabular([[r["bq"], r["bk"], num(err_cell(r["err"], "rel_fro"), 1, sci_below=1e-2), num(err_cell(r["err"], "max_abs"), 1, sci_below=1e-2),
                  num(r.get("extra_peak_MiB"), 1), num(r.get("score_tile_bytes", np.nan) / 1024, 0) if r.get("score_tile_bytes") else "--"] for r in ts],
                ["$B_q$", "$B_k$", "Rel.\\ err", "Max-abs", "Extra peak (MiB)", "Score tile (KiB)"], "rrrrrr", out, "t2a_tiles")


# ---------------------------------------------------------------- Task 2B
@section
def task2b(R, out):
    runs = [jload(f) for f in sorted(glob.glob(str(R / "task2" / "B" / "runs" / "*" / "final.json")))]
    if not runs: raise FileNotFoundError("task2/B/runs")
    names = sorted({r["name"] for r in runs}, key=lambda n: (["dense", "window", "dilated", "linear", "hybrid"].index(n.split("_")[0])
                                                             if n.split("_")[0] in ["dense", "window", "dilated", "linear", "hybrid"] else 9, n))
    tests = ["K4_near", "K4_far", "K8_mixed", "K16_near", "K16_mixed", "K16_far"]
    body = []
    for n in names:
        g = [r for r in runs if r["name"] == n]
        cells = [esc(n), str(len(g)), f"{g[0]['params'] / 1e3:.0f}k" if g[0].get("params") else "--"]
        for t in tests:
            a = [r["test"][t]["acc"] for r in g if t in r.get("test", {})]
            cells.append(f"{100 * np.mean(a):.1f}" + (f"$\\pm${100 * np.std(a, ddof=1):.1f}" if len(a) > 1 else "") if a else "--")
        cells.append(num(np.mean([r.get("train_seconds", np.nan) for r in g]) / 60, 1))
        body.append(cells)
    tabular(body, ["Model (train $K$, regime)", "Seeds", "Params"] + [esc(t) for t in tests] + ["Train (min)"],
            "lcr" + "r" * len(tests) + "r", out, "t2b_mqar")
    note("TTwoBSeeds", max(sum(1 for r in runs if r["name"] == n) for n in names))
    ctl = []
    for n in names:
        g = [r for r in runs if r["name"] == n and r.get("test_with_window64_mask")]
        if not g: continue
        for lab, key in (("trained (dense)", "test"), ("+ window-64 mask", "test_with_window64_mask"), ("+ top-64 keys", "test_with_topk64")):
            cells = [esc(n), lab]
            for t in ("K16_near", "K16_mixed", "K16_far"):
                a = [r[key][t]["acc"] for r in g if t in r.get(key, {})]
                cells.append(f"{100 * np.mean(a):.1f}" if a else "--")
            ctl.append(cells)
    if ctl: tabular(ctl, ["Model", "Test-time attention", "K16\\_near", "K16\\_mixed", "K16\\_far"], "llrrr", out, "t2b_controls")
    env = R / "task2" / "B" / "env.json"
    if env.exists():
        th = jload(env).get("theory") or jload(env).get("extra", {}).get("theory")
        if th:
            ap, rd, st = th["attended_pairs_per_head"], th["max_reachable_distance_2layers"], th["decode_state_floats_per_layer"]
            rows = [["dense", num(ap["dense"], 0), str(rd["dense"]), esc(st["dense_kv_cache"])],
                    ["window-64", num(ap["window64"], 0), str(rd["window64"]), num(st["window64_kv"], 0)],
                    ["dilated (64, stride 4)", num(ap["dilated64x4"], 0), str(rd["dilated"]), esc(st["dilated_kv"])],
                    ["linear", "--", esc(rd["linear"]), num(st["linear"], 0)],
                    ["hybrid (2 linear + 2 local-16)", num(ap.get("hybrid_local16", np.nan), 0) + " (local)", "--", num(st["hybrid"], 0)]]
            tabular(rows, ["Attention", f"Pairs/head ($T$={th['T']})", "Reach (2 layers)", "Decode state (floats/layer)"],
                    "lrrl", out, "t2b_theory")
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.85))
    k16 = [n for n in names if n.endswith("K16_mixed")]
    for i, n in enumerate(k16):
        g = [r for r in runs if r["name"] == n]
        bins = sorted({k for r in g for k in r["test"].get("K16_mixed", {}) if k.startswith("acc_d")}, key=lambda k: int(k.split("_")[1][1:]))
        if bins:
            y = [np.nan if all(r['test']['K16_mixed'].get(b) is None for r in g) else np.nanmean([r["test"]["K16_mixed"].get(b) if r["test"]["K16_mixed"].get(b) is not None else np.nan for r in g]) * 100 for b in bins]
            ax[0].plot(range(len(bins)), y, marker=MARK[i % 8], color=PAL[i % 8], label=n.replace("_K16_mixed", ""))
            ax[0].set_xticks(range(len(bins))); ax[0].set_xticklabels([b[5:].replace("_", "–") for b in bins], rotation=30)
        ck = g[0].get("checkpoints") or []
        if ck:
            ys = [np.mean([next((c["dev"].get("K16_mixed", {}).get("acc", np.nan) for c in r.get("checkpoints", []) if c["step"] == s), np.nan) for r in g]) * 100
                  for s in [c["step"] for c in ck]]
            ax[1].plot([c["step"] for c in ck], ys, marker=MARK[i % 8], color=PAL[i % 8], label=n.replace("_K16_mixed", ""))
    ax[0].axhline(100 / 16, color=INK2, ls=":", lw=0.8); ax[0].set_xlabel("key–query distance"); ax[0].set_ylabel("K16_mixed test acc (%)")
    ax[0].legend(fontsize=5.5)
    ax[1].axhline(100 / 16, color=INK2, ls=":", lw=0.8); ax[1].set_xlabel("update"); ax[1].set_ylabel("K16_mixed dev acc (%)")
    sw = R / "task2" / "B" / "inference_sweep.jsonl"
    if sw.exists():
        s = pd.DataFrame(jlines(sw))
        for i, k in enumerate(["dense", "window", "dilated", "linear", "hybrid"]):
            g = s[(s.kind == k) & (s.get("status", "ok") == "ok")].sort_values("T") if "status" in s else s[s.kind == k].sort_values("T")
            if not g.empty: ax[2].plot(g["T"], g.median_ms, marker=MARK[i], color=PAL[i], label=k)
        ax[2].set_xscale("log", base=2); ax[2].set_yscale("log"); ax[2].set_xlabel("sequence length $T$ (B=8)"); ax[2].set_ylabel("forward ms (one layer)")
        ax[2].legend(fontsize=5.5)
        rows = []
        for T in sorted(s["T"].unique()):
            cells = [f"{T:,}"]
            for k in ["dense", "window", "dilated", "linear", "hybrid"]:
                r = s[(s["T"] == T) & (s.kind == k)]
                if r.empty: cells.append("--"); continue
                r = r.iloc[0]
                cells.append(num(r.median_ms, 2) + (f" / {num(r.get('peak_extra_MiB'), 0)}" if r.get("status", "ok") == "ok" else "") if r.get("status", "ok") == "ok" else "OOM")
            rows.append(cells)
        tabular(rows, ["$T$", "dense", "window", "dilated", "linear", "hybrid"], "rrrrrr", out, "t2b_sweep")
    for a, t, ti in zip(ax, "abc", ("accuracy by distance", "dev accuracy in training", "inference cost")): panel(a, t, ti)
    savefig(fig, out, "fig_task2b")


# ---------------------------------------------------------------- Task 2C
POL = ["full", "sink_recent", "qaware", "quant_token", "quant_kivi", "hybrid"]


@section
def task2c(R, out):
    d = pd.DataFrame(jlines(R / "task2" / "C" / "c5_per_prompt.jsonl"))
    note("TTwoCN", d.id.nunique())
    # group by the nominal context bucket; ctx_len is each prompt's actual token count
    d["Lb"] = d.target_tokens if "target_tokens" in d and d.target_tokens.notna().all() else d.ctx_len
    body, mids = [], []
    for j, L in enumerate(sorted(d.Lb.unique())):
        if j: mids.append(len(body))
        for p in [p for p in POL if p in set(d.policy)]:
            g = d[(d.Lb == L) & (d.policy == p)]
            m, lo, hi = boot_mean(g.correct.astype(float))
            ev = g.evidence_retained_frac.mean() if "evidence_retained_frac" in g and g.evidence_retained_frac.notna().any() else np.nan
            body.append([f"{L:,}", esc(p), f"{100 * m:.1f} {ci(lo, hi, 1, pct=True)}", num(g.realized_frac_fp16.mean(), 3),
                         f"{100 * g.within_budget.mean():.0f}", num(ev, 2), num(g.kl_first.mean(), 3), num(g.ref_answer_nll_mean.mean(), 3),
                         num(1e3 * (g.compress_s + g.get("score_s", 0)).mean(), 1), num(g.peak_alloc_MiB.mean() / 1024, 2)])
    tabular(body, ["$L$", "Policy", "Acc.\\ \\% [95\\% CI]", "Bytes/FP16", "In budget \\%", "Evid.\\ kept", "KL first",
                   "Ans.\\ NLL", "Select+compress (ms)", "Peak (GiB)"], "llrrrrrrrr", out, "t2c_policies", midrules=mids)
    if "position" in d:
        pos = pd.to_numeric(d.position, errors="coerce")
        d["pb"] = pd.cut(pos, 4).astype(str) if pos.notna().all() else d.position.astype(str)
        order = [x for x in ("start", "early", "middle", "late", "end") if x in set(d.pb)] or sorted(d.pb.unique())
        rows = []
        for p in [p for p in POL if p in set(d.policy)]:
            cells = [esc(p)]
            for o in order:
                g = d[(d.policy == p) & (d.pb == o)]
                cells.append(f"{100 * g.correct.mean():.1f}" if len(g) else "--")
            rows.append(cells)
        tabular(rows, ["Policy"] + [esc(o) for o in order], "l" + "r" * len(order), out, "t2c_position")
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.85))
    Ls = sorted(d.Lb.unique())
    for i, p in enumerate([p for p in POL if p in set(d.policy)]):
        ys, los, his = [], [], []
        for L in Ls:
            m, lo, hi = boot_mean(d[(d.Lb == L) & (d.policy == p)].correct.astype(float)); ys.append(100 * m); los.append(100 * (m - lo)); his.append(100 * (hi - m))
        off = (i - 2.5) * 0.025
        ax[0].errorbar([np.log2(L) + off for L in Ls], ys, yerr=[los, his], marker=MARK[i], color=PAL[i], label=p, capsize=1.5, lw=1)
        g = d[d.policy == p]
        ax[1].scatter(g.groupby("Lb").realized_frac_fp16.mean(), g.groupby("Lb").ref_answer_nll_mean.mean(),
                      marker=MARK[i], color=PAL[i], s=14, label=p)
        if "pb" in d:
            order = [x for x in ("start", "early", "middle", "late", "end") if x in set(d.pb)] or sorted(d.pb.unique())
            ax[2].plot(range(len(order)), [100 * d[(d.policy == p) & (d.pb == o)].correct.mean() for o in order], marker=MARK[i], color=PAL[i], label=p)
            ax[2].set_xticks(range(len(order))); ax[2].set_xticklabels(order)
    ax[0].set_xticks([np.log2(L) for L in Ls]); ax[0].set_xticklabels([f"{L:,}" for L in Ls])
    ax[0].set_xlabel("context length"); ax[0].set_ylabel("retrieval accuracy (%)"); ax[0].legend(fontsize=5.5)
    ax[1].set_xlabel("stored KV bytes / FP16"); ax[1].set_ylabel("mean answer NLL (nats)")
    ax[2].set_xlabel("evidence position"); ax[2].set_ylabel("accuracy (%)")
    for a, t, ti in zip(ax, "abc", ("accuracy vs context", "answer NLL vs bytes", "accuracy by position")): panel(a, t, ti)
    savefig(fig, out, "fig_task2c")
    pk = R / "task2" / "C" / "c5_pack_bench.jsonl"
    if pk.exists():
        rows = []
        for r in jlines(pk):
            md = lambda k: (r.get(k) or {}).get("median_ms") if isinstance(r.get(k), dict) else r.get(k)
            rows.append([r["B"], f"{r['L']:,}", esc(r["layout"]), num(r["fp16_cache_bytes"] / 2**20, 0), num(r["serialized_packed_bytes"] / 2**20, 1),
                         num(r["formula_bytes"] / 2**20, 1), num(md("pack"), 1), num(md("unpack"), 1), num(md("decode_step_fp16"), 1),
                         num(md("decode_step_unpack_plus_step"), 1), num(r.get("peak_alloc_fp16_step_MiB", np.nan) / 1024, 2),
                         num(r.get("peak_alloc_packed_step_MiB", np.nan) / 1024, 2)])
        tabular(rows, ["$B$", "$L$", "Layout", "FP16 MiB", "Packed MiB", "Formula MiB", "Pack ms", "Unpack ms", "Step FP16 ms",
                       "Step packed ms", "Peak FP16 GiB", "Peak packed GiB"], "rrlrrrrrrrrr", out, "t2c_packing")


@section
def task2c_qual(R, out):
    """Fixed selection: at the longest context, the first prompt id (sorted) for each evidence position."""
    d = pd.DataFrame(jlines(R / "task2" / "C" / "c5_per_prompt.jsonl"))
    if "output" not in d: raise KeyError("output")
    d["Lb"] = d.target_tokens if "target_tokens" in d and d.target_tokens.notna().all() else d.ctx_len
    L = d.Lb.max(); dl_ = d[d.Lb == L]
    order = [x for x in ("start", "early", "middle", "late", "end") if x in set(dl_.position.astype(str))] or sorted(dl_.position.astype(str).unique())
    pols = [p for p in POL if p in set(d.policy)]
    rows, texts = [], []
    for pos in order:
        ids = sorted(dl_[dl_.position.astype(str) == pos].id.unique())
        if not ids: continue
        pid = ids[0]; g = dl_[dl_.id == pid].set_index("policy")
        code = g.code.iloc[0]
        cells = [tex_text(pid, 22), esc(pos), tex_text(code, 10)]
        for p in pols:
            if p not in g.index: cells.append("--"); continue
            r = g.loc[p]; cells.append(tex_text(r.pred if r.pred is not None else "none", 10) + (r" \checkmark" if r.correct else r" $\times$"))
        rows.append(cells)
        if pos == order[0]:
            texts = [[esc(p), tex_text(g.loc[p].output, 70)] for p in pols if p in g.index]
    tabular(rows, ["Prompt", "Evidence", "Gold"] + [esc(p) for p in pols], "lll" + "l" * len(pols), out, "t2c_qual")
    if texts: tabular(texts, ["Policy", f"Greedy output, prompt {tex_text(rows[0][0], 22)} (first 70 characters)"], "ll", out, "t2c_qual_text")


# ---------------------------------------------------------------- Task 3
DRAFT_LABEL = {"original": "original", "off_6304": "off-policy s1", "off_6305": "off-policy s2",
               "on_6304": "on-policy s1", "on_6305": "on-policy s2", "target": "target (1.5B)"}
DRAFTS = ["original", "off_6304", "off_6305", "on_6304", "on_6305"]


def dl(d): return DRAFT_LABEL.get(d, d)


def bank_label(b):
    return {"teacher": "target", "original_draft": "orig.\\ draft", "exact_spec_original_gamma4": "spec ($\\gamma$=4)"}.get(
        b, esc(b.replace("student_", "own: ").replace("_6304", " s1").replace("_6305", " s2")))


@section
def task3_train(R, out):
    T = R / "task3"
    tox = T / "sampler_toy_chi2.json"
    if tox.exists():
        tj = jload(tox); rows = []
        for k, v in tj.items():
            if not isinstance(v, dict) or "chi2" not in v: continue
            g, mode = k.replace("gamma", "").split("_", 1)
            rows.append([g, esc(mode), num(v["chi2"], 1), v.get("dof", "--"), num(v["p_value"], 3, sci_below=1e-3),
                         num(v.get("max_abs_freq_err"), 4), "reject" if v.get("rejected_at_p01") else "accept"])
        tabular(rows, ["$\\gamma$", "Sampler", "$\\chi^2$", "dof", "$p$", "Max $|\\hat f - p|$", "$H_0$ at 0.01"], "rlrrrrl", out, "t3_toy")
    ut = T / "sampler_unit_tests.json"
    if ut.exists():
        u = jload(ut); note("TThreeUnitTests", f"{sum(1 for v in u.values() if v is True)}/{sum(1 for v in u.values() if isinstance(v, bool))}")
    summ = sorted(glob.glob(str(T / "distill" / "*" / "summary.json")))
    rows = []
    for f in summ:
        s = jload(f); c = s.get("cost", {})
        run = Path(f).parent.name.replace("_seed", "_")
        rows.append([esc(dl(run)), s.get("updates"), s.get("skipped_updates"), f"{s.get('scored_tokens', 0):,}", s.get("presentations", "--"),
                     num(c.get("teacher_generation_s", 0) / 60, 1), num(c.get("student_rollout_s", 0) / 60, 1),
                     num(c.get("teacher_scoring_s", 0) / 60, 1), num(c.get("student_fwd_bwd_s", 0) / 60, 1),
                     num(s.get("wall_s", np.nan) / 60, 1), num(s.get("peak_alloc_MiB", np.nan) / 1024, 2)])
    if rows:
        tabular(rows, ["Run", "Updates", "Skipped", "Scored tok.", "Seqs", "Teacher gen.", "Rollout", "Teacher score", "Fwd/bwd",
                       "Wall", "Peak GiB"], "lrrrrrrrrrr", out, "t3_cost")
        s0 = jload(summ[0]); note("TThreeTrainable", f"{s0.get('trainable_params', 0):,}")


@section
def task3_align(R, out):
    a = pd.DataFrame(jlines(R / "task3" / "eval" / "alignment_per_prefix.jsonl.gz"))
    banks = list(dict.fromkeys(["teacher", "original_draft", "exact_spec_original_gamma4"] + sorted(b for b in a.bank.unique() if b.startswith("student"))))
    banks = [b for b in banks if b in set(a.bank)]
    note("TThreeAlignRows", f"{len(a):,}")
    note("TThreeAlignPrefixes", f"{a[a.draft == a.draft.iloc[0]].shape[0]:,}")
    for metric, name, nd in (("exp_accept", "t3_align_accept", 3), ("fkl", "t3_align_fkl", 3)):
        rows = []
        for d in [d for d in DRAFTS if d in set(a.draft)]:
            cells = [esc(dl(d))]
            for b in banks:
                g = a[(a.draft == d) & (a.bank == b)][metric]
                cells.append(num(g.mean(), nd) if len(g) else "--")
            rows.append(cells)
        tabular(rows, ["Draft"] + [bank_label(b) for b in banks], "l" + "r" * len(banks), out, name)
    rows = []
    for d in [d for d in DRAFTS if d in set(a.draft)]:
        for b in banks:
            g = a[(a.draft == d) & (a.bank == b)]
            if g.empty: continue
            cells = [esc(dl(d)), bank_label(b), len(g)]
            for k in ("fkl", "rkl", "tv", "agree", "exp_accept"):
                m, lo, hi = boot_mean(g[k]); cells.append(f"{num(m, 3)} {ci(lo, hi, 3)}" if k in ("fkl", "exp_accept") else num(m, 3))
            rows.append(cells)
    tabular(rows, ["Draft", "Prefix bank", "$n$", "FKL [CI]", "RKL", "TV", "Agree", "$1-$TV [CI]"], "llrrrrrr", out, "t3_align_full")
    dom = a.groupby(["draft", "domain"]).exp_accept.mean().unstack()
    rows = [[esc(dl(d))] + [num(dom.loc[d, c], 3) for c in dom.columns] for d in DRAFTS if d in dom.index]
    tabular(rows, ["Draft"] + [esc(c) for c in dom.columns], "l" + "r" * len(dom.columns), out, "t3_align_domain")


def spec_join(R, files):
    s = pd.concat([pd.DataFrame(jlines(f)) for f in files], ignore_index=True)
    key = ["draft", "prompt_id", "gamma", "rep"]
    sp = s[s["mode"] == "spec"].set_index(key); tg = s[s["mode"] == "target"].set_index(key)
    j = sp.join(tg[["wall_s", "tok_per_s", "tokens"] + [c for c in ("target_forwards",) if c in tg]], rsuffix="_t", how="inner").reset_index()
    j["speedup"] = j.tok_per_s / j.tok_per_s_t
    j["acc_rate"] = j.accepted / j.proposed.replace(0, np.nan)
    j["tok_per_call"] = j.tokens / j.target_forwards.replace(0, np.nan)
    return j


@section
def task3_spec(R, out):
    E = R / "task3" / "eval"
    files = sorted(set(glob.glob(str(E / "spec*_per_run.jsonl"))))
    j = spec_join(R, files)
    note("TThreeSpecPrompts", j.prompt_id.nunique()); note("TThreeSpecReps", j.rep.nunique())
    al = E / "alignment_per_prefix.jsonl.gz"
    alpha = {}
    if al.exists():
        a = pd.DataFrame(jlines(al)); alpha = a[a.bank == "teacher"].groupby("draft").exp_accept.mean().to_dict()
    rows, recs, mids = [], [], []
    for i, d in enumerate([d for d in DRAFTS if d in set(j.draft)]):
        if i: mids.append(len(rows))
        for g_ in sorted(j.gamma.unique()):
            g = j[(j.draft == d) & (j.gamma == g_)]
            if g.empty: continue
            m, lo, hi = boot_mean(np.log(g.speedup))
            al_ = alpha.get(d, np.nan)
            pred = (1 - al_ ** (g_ + 1)) / (1 - al_) if not np.isnan(al_) else np.nan
            emp = g.tokens.sum() / g.rounds.sum() if "rounds" in g else np.nan
            rows.append([esc(dl(d)), g_, len(g), num(100 * g.acc_rate.mean(), 1), num(g.accepted.sum() / g.rounds.sum(), 2),
                         num(emp, 2), num(pred, 2), num(g.tok_per_call.mean(), 2), num(g.tok_per_s.mean(), 1), num(g.tok_per_s_t.mean(), 1),
                         f"{num(math.exp(m), 2)} {ci(math.exp(lo), math.exp(hi), 2)}"])
            recs.append(dict(draft=d, gamma=g_, pred=pred, emp=emp, sp=math.exp(m), lo=math.exp(lo), hi=math.exp(hi)))
    tabular(rows, ["Draft", "$\\gamma$", "Runs", "Accept \\%", "Acc./round", "Tok./round", "Pred.", "Tok./target call",
                   "Spec tok/s", "Target tok/s", "Speedup [95\\% CI]"], "lrrrrrrrrrr", out, "t3_spec", midrules=mids)
    dom = []
    for d in [d for d in DRAFTS if d in set(j.draft)]:
        for dm_ in sorted(j.domain.unique()):
            g = j[(j.draft == d) & (j.domain == dm_) & (j.gamma == j.gamma.max())]
            if g.empty: continue
            dom.append([esc(dl(d)), esc(dm_), len(g), num(100 * g.acc_rate.mean(), 1), num(g.tokens.mean(), 1),
                        num(g.tok_per_call.mean(), 2), num(math.exp(np.log(g.speedup).mean()), 2)])
    tabular(dom, ["Draft", "Domain", "Runs", "Accept \\%", "Tokens", "Tok./target call", "Speedup"], "llrrrrr", out, "t3_spec_domain")
    note("TThreeSpecGammaMax", int(j.gamma.max()))
    fx = sorted(glob.glob(str(E / "fixed64*_per_run.jsonl")))
    if fx:
        k = spec_join(R, fx); rows = []
        for d in [d for d in DRAFTS if d in set(k.draft)]:
            for g_ in sorted(k.gamma.unique()):
                g = k[(k.draft == d) & (k.gamma == g_)]
                if g.empty: continue
                rows.append([esc(dl(d)), g_, len(g), num(100 * g.acc_rate.mean(), 1), num(g.tok_per_call.mean(), 2),
                             num(g.wall_s.mean(), 2), num(g.wall_s_t.mean(), 2), num(math.exp(np.log(g.speedup).mean()), 2)])
        tabular(rows, ["Draft", "$\\gamma$", "Runs", "Accept \\%", "Tok./target call", "Spec s", "Target s", "Speedup"], "lrrrrrrr", out, "t3_fixed64")
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.85))
    summ = sorted(glob.glob(str(R / "task3" / "distill" / "*" / "updates.jsonl")))
    for i, f in enumerate(summ):
        u = pd.DataFrame(jlines(f)); run = Path(f).parent.name.replace("_seed", "_")
        ci_ = DRAFTS.index(run) if run in DRAFTS else i
        ax[0].plot(u["update"], u.kl_per_token.rolling(10, min_periods=1).mean(), color=PAL[ci_ % 8], label=dl(run),
                   ls="-" if run.startswith("off") else "--")
    ax[0].set_xlabel("optimizer update"); ax[0].set_ylabel("forward KL / token (rolling 10)"); ax[0].legend(fontsize=5.5)
    if recs:
        r = pd.DataFrame(recs)
        for i, d in enumerate([d for d in DRAFTS if d in set(r.draft)]):
            g = r[r.draft == d]
            ax[1].plot(g.pred, g.emp, marker=MARK[i], color=PAL[i], linestyle="none", label=dl(d))
            ax[2].errorbar(g.gamma + (i - 2) * 0.08, g.sp, yerr=[g.sp - g.lo, g.hi - g.sp], marker=MARK[i], color=PAL[i], capsize=1.5, lw=1, label=dl(d))
        lim = [min(r.pred.min(), r.emp.min()) * 0.95, max(r.pred.max(), r.emp.max()) * 1.05]
        ax[1].plot(lim, lim, color=INK2, ls=":", lw=0.8)
        ax[1].set_xlabel("predicted tokens/round from $1-$TV"); ax[1].set_ylabel("measured tokens/round")
        ax[2].axhline(1, color=INK2, ls=":", lw=0.8); ax[2].set_xticks(sorted(r.gamma.unique()))
        ax[2].set_xlabel("proposal length $\\gamma$"); ax[2].set_ylabel("speedup vs target (tok/s)")
    for a, t, ti in zip(ax, "abc", ("distillation loss", "predicted vs measured", "measured speedup")): panel(a, t, ti)
    savefig(fig, out, "fig_task3")


@section
def task3_misc(R, out):
    E = R / "task3" / "eval"
    acc = sorted(glob.glob(str(E / "standalone_accuracy_*.jsonl")))
    if acc:
        s = pd.concat([pd.DataFrame(jlines(f)) for f in acc], ignore_index=True).drop_duplicates(["model", "rep", "prompt_id"])
        s["dom"] = np.where(s.domain == "math", "math", "short")
        rows = []
        models = ["target"] + [d for d in DRAFTS if d in set(s.model)] + sorted(set(s.model) - set(DRAFTS) - {"target"})
        for m in [m for m in models if m in set(s.model)]:
            cells = [esc(dl(m))]
            for dm_ in ("math", "short"):
                g = s[(s.model == m) & (s.dom == dm_)]
                mu, lo, hi = boot_mean(g.correct.astype(float))
                cells += [f"{100 * mu:.1f} {ci(lo, hi, 1, pct=True)}" if len(g) else "--", num(100 * g.truncated.mean(), 1) if len(g) else "--"]
            rows.append(cells)
        tabular(rows, ["Model", "Math acc.\\ \\% [CI]", "Trunc.\\ \\%", "Short-form acc.\\ \\% [CI]", "Trunc.\\ \\%"], "lrrrr", out, "t3_accuracy")
    dc = E / "decode_costs.jsonl"
    if dc.exists():
        d = pd.DataFrame(jlines(dc))
        piv = d.pivot_table(index=["model", "prompt_len"], columns="n_tokens", values="median_ms").reset_index()
        rows = [[esc(dl(r["model"])), r["prompt_len"]] + [num(r[c], 1) for c in piv.columns[2:]] for _, r in piv.iterrows()]
        tabular(rows, ["Model", "Prefix len"] + [f"{c} tok (ms)" for c in piv.columns[2:]], "lr" + "r" * (len(piv.columns) - 2), out, "t3_decode")
    um = E / "unmerged_vs_merged.jsonl"
    if um.exists():
        d = pd.DataFrame(jlines(um)); g = d.groupby("path").agg(wall=("wall_s", "mean"), tok=("tokens", "mean"), n=("wall_s", "size"))
        rows = [[esc(p), int(r.n), num(r.wall, 3), num(r.tok, 1), num(r.tok / r.wall, 1)] for p, r in g.iterrows()]
        tabular(rows, ["Adapter path", "Runs", "Wall (s)", "Tokens", "tok/s"], "lrrrr", out, "t3_merge")


T3_QUAL = ["target", "original", "off_6304", "on_6304"]


@section
def task3_qual(R, out):
    """Fixed selection: the first two math prompts and the first short-form prompt (sorted ids), first repetition."""
    acc = sorted(glob.glob(str(R / "task3" / "eval" / "standalone_accuracy_*.jsonl")))
    s = pd.concat([pd.DataFrame(jlines(f)) for f in acc], ignore_index=True).drop_duplicates(["model", "rep", "prompt_id"])
    s = s[s.rep == s.rep.min()]
    ids = [(p, True) for p in sorted(s[s.domain == "math"].prompt_id.unique())[:2]] + \
          [(p, False) for p in sorted(s[s.domain != "math"].prompt_id.unique())[:1]]
    rows = []
    for pid, is_math in ids:
        for m in [m for m in T3_QUAL if m in set(s.model)]:
            g = s[(s.prompt_id == pid) & (s.model == m) & ((s.domain == "math") == is_math)]
            if g.empty: continue
            r = g.iloc[0]
            rows.append([tex_text(pid, 14), esc(dl(m)), int(r.n_tokens), "yes" if r.truncated else "no",
                         r"\checkmark" if r.correct else r"$\times$", tex_text(r.text, 75)])
    tabular(rows, ["Prompt", "Model", "Tokens", "Trunc.", "Correct", "Sampled response (first 75 characters)"], "llrccl", out, "t3_qual")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="report_tex")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    for fn in (task1_tables, task1_costnotes, task1_paired, task1_qual, task1_selection, task1_scores, task1_figure, task1_alloc_figure, task2a, task2b,
               task2c, task2c_qual, task3_train, task3_align, task3_spec, task3_misc, task3_qual):
        fn(R, out)
    lines = ["% generated by report/make_report_assets.py -- numbers used inside captions"]
    for k in ("TOneWikiN", "TOneCFourN", "TOneQThreeUnits", "TTwoAShape", "TTwoBSeeds", "TTwoCN", "TThreeUnitTests", "TThreeTrainable",
              "TThreeAlignRows", "TThreeAlignPrefixes", "TThreeSpecPrompts", "TThreeSpecReps", "TThreeSpecGammaMax", "TTwoABenchReps",
              "TOneCalibS", "TOneAwqS", "TOneQThreeProbeMin", "TOnePThreeProbeMin"):
        lines.append(f"\\newcommand{{\\{k}}}{{{NOTES.get(k, '--')}}}")
    (out / "tables" / "notes.tex").write_text("\n".join(lines) + "\n")
    here = Path(__file__).resolve().parent
    for f in ("main.tex", "references.bib"):
        if (here / f).exists(): shutil.copy(here / f, out / f)
    (out / "MISSING.txt").write_text("\n".join(MISSING) + ("\n" if MISSING else ""))
    print("notes:", NOTES)
    print("MISSING:" if MISSING else "all sections generated", *MISSING, sep="\n  ")
    print("wrote", out)


if __name__ == "__main__":
    main()
