"""Build notebooks/PA4_kaggle.ipynb: a Kaggle runner that clones this repository and launches each phase."""
import json, pathlib
REPO = "https://github.com/alizarman786-prog/atml-pa4.git"

ROOT = pathlib.Path(__file__).parent
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))

md("""# ATML PA4 — Kaggle runner (GPU T4 x2, Internet ON)

Set `PHASE` in the next cell, then **Run All** (interactive) or **Save Version → Save & Run All** (long runs).

| PHASE | What it does | Expected time |
|---|---|---|
| `smoke` | quick end-to-end tests of every script (tiny budgets) → `/kaggle/working/smoke` | ~40 min (already passed) |
| `A` | full Task 1 on GPU0 ‖ full Task 2 (A, B, C) on GPU1 → `/kaggle/working/results` | ~3–3.5 h |
| `B` | full Task 3 (prefix banks, 4 distillation runs, all evaluations) on both GPUs | ~3–4 h |
| `R` | report only: attach the Phase A output (as a dataset) and the Phase C output, then build tables and figures | ~5 min |
| `C` | re-run only the Task 3 alignment stage on top of a finished Phase B (attach the Phase B notebook output as Input first) | ~40–50 min |

Logs stream to `/kaggle/working/logs/*.log`; every stage resumes from saved results if restarted.""")

code('PHASE = "A"   # "smoke" | "A" | "B" | "C" | "R"\n')

code("""%%bash
set -e
pip install -q transformers==4.57.1 huggingface_hub==0.36.2 "datasets>=3,<5" peft==0.17.1 2>&1 | grep -v "requires\\|incompatible" | tail -2 || true
cd /kaggle/working
[ -d atml-pa4-datasets ] || git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git
cd atml-pa4-datasets && echo "data repo commit: $(git log -1 --format=%H)"
python scripts/check_setup.py >/dev/null 2>&1 || python scripts/setup_data.py --public
python scripts/check_setup.py
cat data/local_source_revisions.json
rm -rf /kaggle/working/code && git clone -q https://github.com/alizarman786-prog/atml-pa4.git /kaggle/working/code
echo "code commit: $(git -C /kaggle/working/code log -1 --format=%H)"
mkdir -p /kaggle/working/logs
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "no GPU (CPU session)"
python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"
""")


code('''import os, subprocess, time, pathlib
BASE = {"PA4_DATA": "/kaggle/working/atml-pa4-datasets", "PYTHONPATH": "/kaggle/working/code",
        "PA4_CACHE": "/tmp/pa4cache", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
LOGS = pathlib.Path("/kaggle/working/logs"); LOGS.mkdir(exist_ok=True)

def stream(name, gpu, cmds, extra=None):
    """Run shell commands sequentially on one GPU (a failing step does not stop the next)."""
    env = {**os.environ, **BASE, "CUDA_VISIBLE_DEVICES": str(gpu), **(extra or {})}
    script = "\\n".join(f'echo "### $(date +%T) {c}"; {c} || echo "!!! FAILED: {c}"' for c in cmds)
    return subprocess.Popen(["bash", "-c", script], cwd="/kaggle/working/code", env=env,
                            stdout=open(LOGS / f"{name}.log", "w"), stderr=subprocess.STDOUT)

def wait_all(procs, every=300):
    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(every)
        print(f"--- {(time.time()-t0)/60:.0f} min", {k: ("running" if p.poll() is None else p.returncode) for k, p in procs.items()})
        for k in procs:
            lines = (LOGS / f"{k}.log").read_text().splitlines()
            print(f"[{k}]", *lines[-3:], sep="\\n  ")
    print("all done in", round((time.time() - t0) / 60, 1), "min")

def show_failures():
    for f in sorted(LOGS.glob("*.log")):
        txt = f.read_text()
        bad = [l for l in txt.splitlines() if l.startswith("!!! FAILED")]
        print(f.name, "OK" if not bad else bad)
        if bad: print("\\n".join(txt.splitlines()[-40:]))
''')

code('''if PHASE == "smoke":
    S = {"PA4_OUT": "/kaggle/working/smoke", "PA4_QUICK": "1", "PA4_CACHE": "/tmp/pa4cache_smoke"}
    procs = {
      "smoke_gpu0": stream("smoke_gpu0", 0, [
          "python -m pa4.t1_run --quick --stages baseline,calib,prune,p3,quant,q3",
          "python -m pa4.t3_spec --toy --n 100000",
      ], S),
      "smoke_gpu1": stream("smoke_gpu1", 1, [
          "python -m pa4.t2a_attention",
          "python -m pa4.t2b_mqar --runs dense_K4_mixed,window_K4_near,dilated_K4_mixed,linear_K4_mixed,hybrid_K16_mixed --seeds 6304",
          "python -m pa4.t2b_mqar --sweep",
          "python -m pa4.t2c_kv --stage validate,eval,bench --limit 3",
          "python -m pa4.t3_distill --method off --seed 6304",
          "python -m pa4.t3_distill --method on --seed 6304",
          "python -m pa4.t3_eval --stage validate,costs,specbank --drafts original",
          "python -m pa4.t3_eval --stage studentbanks,align,spec,fixed64,accuracy,unmerged --drafts original,off_6304,on_6304",
      ], S),
    }
    wait_all(procs, every=120); show_failures()
''')

code('''if PHASE == "A":
    procs = {
      "task1": stream("task1", 0, ["python -m pa4.t1_run --stages baseline,calib,prune,p3,quant,q3"]),
      "task2": stream("task2", 1, ["python -m pa4.t2a_attention",
                                    "python -m pa4.t2b_mqar --runs all",
                                    "python -m pa4.t2b_mqar --sweep",
                                    "python -m pa4.t2c_kv --stage validate,eval,bench"]),
    }
    wait_all(procs); show_failures()
''')

code('''if PHASE == "B":
    procs = {
      "t3_train0": stream("t3_train0", 0, ["python -m pa4.t3_distill --method off --seed 6304",
                                            "python -m pa4.t3_distill --method on --seed 6304"]),
      "t3_train1": stream("t3_train1", 1, ["python -m pa4.prefix_banks",
                                            "python -m pa4.t3_spec --toy --n 100000",
                                            "python -m pa4.t3_distill --method off --seed 6305",
                                            "python -m pa4.t3_distill --method on --seed 6305"]),
    }
    wait_all(procs); show_failures()
    procs = {
      "t3_eval0": stream("t3_eval0", 0, ["python -m pa4.t3_eval --stage validate,costs,specbank,studentbanks,align",
                                          "python -m pa4.t3_eval --stage spec --drafts original,off_6304,on_6304 --tag spec_g0",
                                          "python -m pa4.t3_eval --stage fixed64,unmerged --tag all"]),
      "t3_eval1": stream("t3_eval1", 1, ["python -m pa4.t3_eval --stage spec --drafts off_6305,on_6305 --tag spec_g1",
                                          "python -m pa4.t3_eval --stage accuracy"]),
    }
    wait_all(procs); show_failures()
''')

code('''if PHASE == "C":
    # Restore Phase B outputs from the attached notebook output, then recompute the alignment metrics only.
    import glob, shutil
    hits = [p for p in glob.glob("/kaggle/input/**/results/task3/distill", recursive=True)]
    assert hits, "attach the Phase B notebook output as an Input"
    src = pathlib.Path(hits[0]).parents[2]; print("restoring from", src)
    shutil.copytree(src / "results", "/kaggle/working/results", dirs_exist_ok=True)
    banks = glob.glob(str(src / "**" / "prefix_banks_locked.jsonl.gz"), recursive=True)
    dst = pathlib.Path("/kaggle/working/atml-pa4-datasets/data/prefix_banks"); dst.mkdir(parents=True, exist_ok=True)
    for b in banks: shutil.copy(b, dst)
    print("prefix banks restored:", banks)
    # teacher / original-draft banks are generated here if the Phase B run did not produce them
    procs = {"t3_align": stream("t3_align", 0, ["python -m pa4.prefix_banks",
                                                "python -m pa4.t3_eval --stage align"])}
    wait_all(procs, every=120); show_failures()
''')

code('''if PHASE == "R":
    # Report only: gather Task 1-2 results (Phase A dataset) and Task 3 results (Phase C output) into one folder.
    import glob, shutil, tarfile
    dst = pathlib.Path("/kaggle/working/results"); dst.mkdir(exist_ok=True); pathlib.Path("/kaggle/working/logs").mkdir(exist_ok=True)
    for task in ("task1", "task2", "task3"):
        hits = sorted(glob.glob(f"/kaggle/input/**/results/{task}", recursive=True))
        if hits:
            best = max(hits, key=lambda h: sum(1 for _ in pathlib.Path(h).rglob("*")))
            shutil.copytree(best, dst / task, dirs_exist_ok=True); print(task, "<-", best); continue
        for tgz in glob.glob("/kaggle/input/**/results_*.tgz", recursive=True):
            with tarfile.open(tgz) as tf:
                mem = [m for m in tf.getmembers() if m.name.startswith(f"results/{task}")]
                if mem: tf.extractall("/kaggle/working", members=mem); print(task, "<-", tgz); break
        else: print("!!! no results for", task)
    print(sorted(str(p.relative_to(dst)) for p in dst.glob("*/*"))[:60])
''')

code('''# Package results for download (Output tab) — small JSON/JSONL/NPZ files only
src = "smoke" if PHASE == "smoke" else "results"
subprocess.run(f"cd /kaggle/working/code && python -m pa4.report --results /kaggle/working/{src} --out /kaggle/working/report_{PHASE}", shell=True)
if PHASE == "R":   # LaTeX tables, caption macros and PDF figures for the report, zipped for download
    import shutil
    subprocess.run("cd /kaggle/working/code && python report/make_report_assets.py --results /kaggle/working/results "
                   "--out /kaggle/working/report_tex && cat /kaggle/working/report_tex/tables/*.tex", shell=True)
    shutil.make_archive("/kaggle/working/report_tex", "zip", "/kaggle/working", "report_tex")
subprocess.run(f"cd /kaggle/working && tar czf results_{PHASE}.tgz {src} logs report_{PHASE} && ls -la /kaggle/working", shell=True)
''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=4)
out = ROOT / "notebooks" / "PA4_kaggle.ipynb"; out.parent.mkdir(exist_ok=True)
for c in nb["cells"]:
    src = c["source"]; c["source"] = src.splitlines(keepends=True)
out.write_text(json.dumps(nb, indent=1))
print(out, len(cells), "cells", out.stat().st_size, "bytes")
