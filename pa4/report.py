"""Summary tables and figures from the saved results (CPU only).

python -m pa4.report --results /path/to/results --out report_assets

`--results` is the directory that holds task1/, task2/ and task3/ (unpack results_A.tgz and results_C.tgz, or
results_B.tgz, into the same folder). Every section is optional: missing inputs are skipped with a note.
Outputs: <out>/*.csv, <out>/*.md (the same tables in Markdown), <out>/*.png and <out>/summary.md.
"""
from __future__ import annotations
import argparse, glob, gzip, json, math, os, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 6304
NOTES: list[str] = []


# ---------------- io helpers ----------------
def jload(p):
    return json.loads(Path(p).read_text())


def jlines(p):
    p = Path(p); op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def emit(df: pd.DataFrame, out: Path, name: str, title: str, floatfmt=".4g"):
    df.to_csv(out / f"{name}.csv", index=False)
    try: body = df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:   # tabulate not installed
        cells = [[(f"{v:{floatfmt}}" if isinstance(v, float) else str(v)) for v in row] for row in df.itertuples(index=False)]
        body = "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                         + ["| " + " | ".join(c) + " |" for c in cells])
    md = f"### {title}\n\n" + body + "\n"
    (out / f"{name}.md").write_text(md)
    with open(out / "summary.md", "a") as f: f.write(md + "\n")
    print(md)


def savefig(fig, out: Path, name: str):
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=160); plt.close(fig)


def boot_mean_ci(x, n=2000, seed=SEED):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ppl_diff(nll_a, cnt_a, nll_b, cnt_b, n=2000, seed=SEED):
    """Window-level paired bootstrap of PPL_b - PPL_a (same windows, same scored tokens)."""
    nll_a, nll_b, cnt = (np.asarray(v, float) for v in (nll_a, nll_b, cnt_a))
    if not np.array_equal(np.asarray(cnt_a), np.asarray(cnt_b)): return (np.nan, np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(cnt), size=(n, len(cnt)))
    d = np.exp(nll_b[idx].sum(1) / cnt[idx].sum(1)) - np.exp(nll_a[idx].sum(1) / cnt[idx].sum(1))
    full = math.exp(nll_b.sum() / cnt.sum()) - math.exp(nll_a.sum() / cnt.sum())
    return full, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def section(fn):
    def run(R, out):
        try: fn(R, out)
        except Exception as e:
            NOTES.append(f"{fn.__name__}: skipped ({type(e).__name__}: {e})"); traceback.print_exc()
    run.__name__ = fn.__name__
    return run


# ---------------- Task 1 ----------------
def _storage_mb(r):
    for k in ("storage_packed", "storage_sparse16", "storage_fp16_dense"):
        v = r.get(k)
        if isinstance(v, dict):
            for kk in ("total_bytes", "bytes", "total"):
                if kk in v: return v[kk] / 2**20, k.replace("storage_", "")
        elif isinstance(v, (int, float)): return v / 2**20, k.replace("storage_", "")
    return np.nan, ""


def _hs(r):
    h = r.get("hellaswag_full") or r.get("hellaswag_diag512") or {}
    return h.get("acc_norm", np.nan), h.get("n", np.nan)


def _latency(r):
    lat = r.get("latency") or {}
    keys = sorted(k for k in lat if isinstance(lat[k], dict) and "median_ms" in lat[k])
    return {f"lat_{k}_ms": lat[k]["median_ms"] for k in keys}


@section
def task1(R, out):
    C = R / "task1" / "conditions"; A = R / "task1" / "arrays"
    rows = {Path(f).stem: jload(f) for f in sorted(glob.glob(str(C / "*.json")))}
    if not rows: raise FileNotFoundError(C)
    base = np.load(A / "fp16.npz") if (A / "fp16.npz").exists() else None
    tab = []
    for name, r in rows.items():
        mb, kind = _storage_mb(r); acc, n_hs = _hs(r)
        nz = r.get("nonzeros") or {}
        t = dict(condition=name, method=r.get("method"), target_sparsity=r.get("target_sparsity"),
                 bits=r.get("bits", r.get("budget_bits")), realized_sparsity=nz.get("sparsity"),
                 storage_MiB=mb, storage_kind=kind,
                 eff_bits_per_weight=(r.get("storage_packed") or {}).get("effective_bits_per_weight"),
                 wikitext2_ppl=(r.get("ppl_wikitext2") or {}).get("ppl"), c4val_ppl=(r.get("ppl_c4val") or {}).get("ppl"),
                 hellaswag_acc_norm=acc, hellaswag_n=n_hs, dev_nll=r.get("dev_nll_c4dev32"),
                 calib_rel_err=r.get("calib_output_rel_err_mean"), compress_s=r.get("compress_seconds"), **_latency(r))
        if base is not None and (A / f"{name}.npz").exists() and name != "fp16":
            arr = np.load(A / f"{name}.npz")
            for k in ("wikitext2", "c4val"):
                if f"{k}_nll" in arr and f"{k}_nll" in base:
                    d, lo, hi = boot_ppl_diff(base[f"{k}_nll"], base[f"{k}_cnt"], arr[f"{k}_nll"], arr[f"{k}_cnt"])
                    t[f"d{k}_ppl"] = d; t[f"d{k}_ci95"] = f"[{lo:.3g}, {hi:.3g}]"
        tab.append(t)
    df = pd.DataFrame(tab)
    order = ["fp16", "P1", "P2", "2:4", "P3", "Q1", "Q2", "Q3"]
    df["_o"] = df.condition.map(lambda s: next((i for i, p in enumerate(order) if s.startswith(p) or p in s), 99))
    df = df.sort_values(["_o", "condition"]).drop(columns="_o")
    emit(df, out, "task1_conditions", "Task 1 — all compressed conditions (ΔPPL vs FP16 with window-bootstrap 95% CI)")

    # sparsity sweep figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, g in df[df.target_sparsity.notna() & df.method.astype(str).str.contains("magnitude|wanda", regex=True)].groupby("method"):
        g = g.sort_values("target_sparsity")
        ax[0].plot(g.target_sparsity, g.wikitext2_ppl, "o-", label=m)
        ax[1].plot(g.target_sparsity, g.hellaswag_acc_norm, "o-", label=m)
    ax[0].set_yscale("log"); ax[0].set_xlabel("target sparsity"); ax[0].set_ylabel("WikiText-2 PPL (log)")
    ax[1].set_xlabel("target sparsity"); ax[1].set_ylabel("HellaSwag acc_norm")
    if "fp16" in rows:
        ax[0].axhline(rows["fp16"]["ppl_wikitext2"]["ppl"], ls="--", c="k", lw=0.8, label="FP16")
        ax[1].axhline(_hs(rows["fp16"])[0], ls="--", c="k", lw=0.8)
    ax[0].legend(fontsize=7); savefig(fig, out, "task1_pruning")

    q = df[df.bits.notna()]
    if len(q):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m, g in q.groupby("method"):
            ax.scatter(g.storage_MiB, g.wikitext2_ppl, label=m)
            for _, r in g.iterrows(): ax.annotate(f"{r.bits:g}b", (r.storage_MiB, r.wikitext2_ppl), fontsize=7)
        ax.set_xlabel("packed storage (MiB)"); ax.set_ylabel("WikiText-2 PPL"); ax.set_yscale("log"); ax.legend(fontsize=7)
        savefig(fig, out, "task1_quant_tradeoff")

    p3 = R / "task1" / "p3_selection.json"
    if p3.exists():
        s = jload(p3); sel = s.get("selected", {})
        rows3 = []
        for S, b in sel.items():
            alloc = b.get("allocation", b) if isinstance(b, dict) else b
            if isinstance(alloc, (list, tuple)):
                rows3.append(dict(target=S, lambda_=b.get("lambda") if isinstance(b, dict) else None,
                                  min=min(alloc), max=max(alloc), mean=float(np.mean(alloc))))
        if rows3: emit(pd.DataFrame(rows3), out, "task1_p3_allocation", "Task 1 — P3 per-block sparsity allocation")


# ---------------- Task 2A ----------------
@section
def task2a(R, out):
    D = R / "task2" / "A"
    if (D / "correctness.jsonl").exists():
        c = pd.DataFrame(jlines(D / "correctness.jsonl"))
        flat = c.copy()
        for col in ("out", "grad_q", "grad_k", "grad_v"):
            if col in flat and isinstance(flat[col].iloc[0], dict):
                flat[col] = flat[col].map(lambda d: d.get("rel_l2", d.get("max_abs", np.nan)) if isinstance(d, dict) else d)
        emit(flat, out, "task2a_correctness", "Task 2A — error vs FP32 reference (relative L2 / max-abs)")
    if (D / "bench.jsonl").exists():
        b = pd.DataFrame(jlines(D / "bench.jsonl"))
        keep = [c for c in ("dtype", "mode", "method", "L", "median_ms", "mean_ms", "std_ms", "peak_MiB", "extra_MiB", "oom", "error") if c in b]
        emit(b[keep], out, "task2a_bench", "Task 2A — latency / memory")
        for dt in b.dtype.unique():
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for i, mode in enumerate(("fwd", "fwd_bwd")):
                for m, g in b[(b.dtype == dt) & (b["mode"] == mode)].groupby("method"):
                    g = g.sort_values("L"); ax[i].plot(g.L, g.median_ms, "o-", label=m)
                ax[i].set_xscale("log", base=2); ax[i].set_yscale("log"); ax[i].set_title(f"{dt} {mode}")
                ax[i].set_xlabel("sequence length"); ax[i].set_ylabel("median ms"); ax[i].legend(fontsize=7)
            savefig(fig, out, f"task2a_latency_{dt}")
    if (D / "tile_sweep.jsonl").exists():
        emit(pd.DataFrame(jlines(D / "tile_sweep.jsonl")), out, "task2a_tile_sweep", "Task 2A — tile-size sweep")


# ---------------- Task 2B ----------------
@section
def task2b(R, out):
    runs = sorted(glob.glob(str(R / "task2" / "B" / "runs" / "*" / "final.json")))
    if not runs: raise FileNotFoundError("task2/B/runs/*/final.json")
    rows = []
    for f in runs:
        r = jload(f)
        for split, t in (r.get("test") or {}).items():
            rows.append(dict(name=r["name"], kind=r["kind"], train_K=r["train_K"], train_regime=r["train_regime"],
                             seed=r["seed"], params=r.get("params"), test=split, acc=t["acc"],
                             **{k: v for k, v in t.items() if k.startswith("acc_d")}))
    df = pd.DataFrame(rows)
    agg = (df.groupby(["name", "test"]).acc.agg(["mean", "std", "count"]).reset_index()
             .pivot(index="name", columns="test", values="mean").reset_index())
    emit(agg, out, "task2b_accuracy", "Task 2B — MQAR test accuracy (mean over seeds)")
    emit(df, out, "task2b_accuracy_per_seed", "Task 2B — per seed / split / distance bin")
    dist_cols = sorted([c for c in df if c.startswith("acc_d")], key=lambda c: int(c.split("_")[1][1:]))
    if dist_cols:
        sub = df[df.test.str.endswith("mixed") | df.test.str.endswith("far")]
        prof = sub.groupby("name")[dist_cols].mean()
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(prof) + 1.5))
        im = ax.imshow(prof.values.astype(float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(prof))); ax.set_yticklabels(prof.index, fontsize=7)
        ax.set_xticks(range(len(dist_cols))); ax.set_xticklabels([c[5:] for c in dist_cols], rotation=45, fontsize=7)
        ax.set_xlabel("key–query distance bin"); fig.colorbar(im, label="accuracy"); savefig(fig, out, "task2b_distance_heatmap")
    sw = R / "task2" / "B" / "inference_sweep.jsonl"
    if sw.exists():
        s = pd.DataFrame(jlines(sw)); emit(s, out, "task2b_inference_sweep", "Task 2B — inference cost sweep")
        if "median_ms" in s:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for k, g in s.groupby("kind"): g = g.sort_values("T"); ax.plot(g["T"], g.median_ms, "o-", label=k)
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("T"); ax.set_ylabel("median ms"); ax.legend(fontsize=7)
            savefig(fig, out, "task2b_inference_sweep")


# ---------------- Task 2C ----------------
@section
def task2c(R, out):
    f = R / "task2" / "C" / "c5_per_prompt.jsonl"
    if not f.exists(): raise FileNotFoundError(f)
    d = pd.DataFrame(jlines(f))
    rows = []
    for (pol, L), g in d.groupby(["policy", "ctx_len"]):
        m, lo, hi = boot_mean_ci(g.correct.astype(float))
        rows.append(dict(policy=pol, ctx_len=L, n=len(g), accuracy=m, acc_ci95=f"[{lo:.3f}, {hi:.3f}]",
                         realized_frac_fp16=g.realized_frac_fp16.mean(), within_budget=g.within_budget.mean(),
                         kl_first=g.kl_first.mean(), kl_tf_mean=g.kl_tf_mean.mean(),
                         ref_nll=g.ref_answer_nll_mean.mean(), compress_ms=1e3 * g.compress_s.mean()))
    t = pd.DataFrame(rows).sort_values(["ctx_len", "policy"])
    emit(t, out, "task2c_policies", "Task 2C — KV policies at the matched INT4 byte budget (bootstrap 95% CI)")
    if "position" in d:
        d["pos_bin"] = pd.cut(d.position.astype(float), 5)
        pb = d.groupby(["policy", "pos_bin"], observed=True).correct.mean().unstack()
        pb.columns = [str(c) for c in pb.columns]
        emit(pb.reset_index(), out, "task2c_by_position", "Task 2C — accuracy by evidence position")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p, g in t.groupby("policy"): ax.scatter(g.realized_frac_fp16, g.accuracy, label=p)
    ax.set_xlabel("KV bytes / FP16 bytes"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7)
    savefig(fig, out, "task2c_bytes_vs_accuracy")
    pk = R / "task2" / "C" / "c5_pack_bench.jsonl"
    if pk.exists(): emit(pd.DataFrame(jlines(pk)), out, "task2c_packing", "Task 2C — packed cache bytes and timings")


# ---------------- Task 3 ----------------
@section
def task3(R, out):
    T = R / "task3"
    tox = T / "sampler_toy_chi2.json"
    if tox.exists():
        tj = jload(tox)
        items = tj if isinstance(tj, list) else [dict(case=k, **v) if isinstance(v, dict) else dict(case=k, value=v) for k, v in tj.items()]
        df = pd.json_normalize(items)
        emit(df[[c for c in df if not c.startswith("stats.accepted_per_round")]], out, "task3_toy_chi2", "Task 3 — toy exactness test")
    summ = sorted(glob.glob(str(T / "distill" / "*" / "summary.json")))
    if summ:
        rows = []
        for f in summ:
            s = jload(f); c = s.get("cost", {})
            rows.append(dict(run=Path(f).parent.name, updates=s.get("updates"), skipped=s.get("skipped_updates"),
                             scored_tokens=s.get("scored_tokens"), wall_s=s.get("wall_s"),
                             teacher_gen_s=c.get("teacher_generation_s"), student_rollout_s=c.get("student_rollout_s"),
                             teacher_scoring_s=c.get("teacher_scoring_s"), fwd_bwd_s=c.get("student_fwd_bwd_s"),
                             peak_MiB=s.get("peak_alloc_MiB")))
        emit(pd.DataFrame(rows), out, "task3_training_cost", "Task 3 — distillation cost")
        fig, ax = plt.subplots(figsize=(6, 4))
        for f in summ:
            u = T / "distill" / Path(f).parent.name / "updates.jsonl"
            if u.exists():
                ud = pd.DataFrame(jlines(u)); ax.plot(ud["update"], ud.kl_per_token.rolling(10, min_periods=1).mean(), label=Path(f).parent.name)
        ax.set_xlabel("update"); ax.set_ylabel("FKL per token (rolling 10)"); ax.legend(fontsize=7); savefig(fig, out, "task3_training_curves")

    E = T / "eval"
    al = E / "alignment_per_prefix.jsonl.gz"
    if al.exists():
        a = pd.DataFrame(jlines(al))
        rows = []
        for (d, bk), g in a.groupby(["draft", "bank"]):
            r = dict(draft=d, bank=bk, n=len(g))
            for k in ("fkl", "rkl", "tv", "agree", "exp_accept"):
                m, lo, hi = boot_mean_ci(g[k]); r[k] = m; r[f"{k}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
            rows.append(r)
        emit(pd.DataFrame(rows), out, "task3_alignment", "Task 3 — draft/target alignment by bank (prefix bootstrap 95% CI)")
        dom = a.groupby(["draft", "domain"])[["fkl", "tv", "exp_accept"]].mean().reset_index()
        emit(dom, out, "task3_alignment_by_domain", "Task 3 — alignment by domain")
        fig, ax = plt.subplots(figsize=(7, 4))
        piv = a.groupby(["bank", "draft"]).exp_accept.mean().unstack()
        piv.plot.bar(ax=ax); ax.set_ylabel("1 − TV (expected acceptance)"); ax.legend(fontsize=7); savefig(fig, out, "task3_alignment")

    spec = sorted(set(glob.glob(str(E / "spec*_per_run.jsonl"))) - set(glob.glob(str(E / "fixed64*"))))
    if spec:
        s = pd.concat([pd.DataFrame(jlines(f)) for f in spec], ignore_index=True)
        key = ["draft", "prompt_id", "gamma", "rep"]
        sp = s[s["mode"] == "spec"].set_index(key); tg = s[s["mode"] == "target"].set_index(key)
        j = sp.join(tg[["wall_s", "tok_per_s", "tokens"]], rsuffix="_target", how="inner").reset_index()
        j["speedup"] = j.tok_per_s / j.tok_per_s_target
        j["accept_rate"] = j.accepted / j.proposed.replace(0, np.nan)
        j["tokens_per_target_fwd"] = j.tokens / j.target_forwards.replace(0, np.nan)
        rows = []
        for (d, g_), g in j.groupby(["draft", "gamma"]):
            m, lo, hi = boot_mean_ci(np.log(g.speedup))
            rows.append(dict(draft=d, gamma=g_, n=len(g), accept_rate=g.accept_rate.mean(),
                             tokens_per_target_forward=g.tokens_per_target_fwd.mean(),
                             speedup_geomean=math.exp(m), speedup_ci95=f"[{math.exp(lo):.3f}, {math.exp(hi):.3f}]",
                             spec_tok_s=g.tok_per_s.mean(), target_tok_s=g.tok_per_s_target.mean()))
        emit(pd.DataFrame(rows), out, "task3_speculative", "Task 3 — speculative decoding (paired by prompt/seed; bootstrap CI of geometric-mean speedup)")
        emit(j.groupby(["draft", "gamma", "domain"])[["accept_rate", "speedup"]].mean().reset_index(), out,
             "task3_speculative_by_domain", "Task 3 — speculative decoding by domain")
    fx = sorted(glob.glob(str(E / "fixed64*_per_run.jsonl")))
    if fx:
        s = pd.concat([pd.DataFrame(jlines(f)) for f in fx], ignore_index=True)
        emit(s.groupby(["draft", "gamma", "mode"])[["tok_per_s", "wall_s"]].mean().reset_index(), out,
             "task3_fixed64", "Task 3 — fixed 64-token timing diagnostic")
    acc = sorted(glob.glob(str(E / "standalone_accuracy_*.jsonl")))
    if acc:
        s = pd.concat([pd.DataFrame(jlines(f)) for f in acc], ignore_index=True).drop_duplicates(["model", "rep", "prompt_id"])
        s["dom"] = np.where(s.domain == "math", "math", "short")
        rows = []
        for (m, dm), g in s.groupby(["model", "dom"]):
            mu, lo, hi = boot_mean_ci(g.correct.astype(float))
            rows.append(dict(model=m, domain=dm, n=len(g), accuracy=mu, ci95=f"[{lo:.3f}, {hi:.3f}]",
                             truncated=g.truncated.mean(), mean_tokens=g.n_tokens.mean()))
        emit(pd.DataFrame(rows), out, "task3_standalone_accuracy", "Task 3 — standalone answer quality")
    for name, title in (("decode_costs.jsonl", "Task 3 — per-forward decode cost"),
                        ("unmerged_vs_merged.jsonl", "Task 3 — merged vs unmerged adapter")):
        if (E / name).exists():
            df = pd.DataFrame(jlines(E / name))
            if name.startswith("unmerged"): df = df.groupby("path")[["wall_s", "tokens"]].mean().reset_index()
            emit(df, out, Path(name).stem, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("PA4_OUT", "results"))
    ap.add_argument("--out", default="report_assets")
    a = ap.parse_args()
    R = Path(a.results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("# PA4 result tables\n\n")
    for fn in (task1, task2a, task2b, task2c, task3): fn(R, out)
    if NOTES:
        with open(out / "summary.md", "a") as f: f.write("\n## Skipped\n\n" + "\n".join(f"- {n}" for n in NOTES) + "\n")
        print("\n".join(NOTES))
    print("wrote", out)


if __name__ == "__main__":
    main()
