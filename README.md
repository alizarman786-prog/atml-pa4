# ATML PA4 — Model compression, efficient attention / KV memory, speculative decoding

Code for EE-5102/CS-6304 Programming Assignment 4 (Fall 2026).

* Course data: <https://github.com/AbDu11aHHH/atml-pa4-datasets> (prepared with `scripts/setup_data.py --public` and `--prefix-banks`).
* Hardware used: Kaggle, 2× NVIDIA T4 (16 GB), FP16. Pinned: `transformers==4.57.1`, `huggingface_hub==0.36.2`, `peft==0.17.1` (see `requirements.txt`).
* Seeds: 6304 (reference), 6305/6306 where the assignment requires them.

## Layout

| Path | Contents |
|---|---|
| `pa4/common.py` | paths, seeding, environment capture, CUDA timing, paired bootstrap helpers |
| `pa4/t1_eval.py` | WikiText-2 / C4 perplexity on fixed windows, zero-shot HellaSwag (acc, acc_norm), latency / memory |
| `pa4/t1_compress.py` | calibration statistics, magnitude / Wanda / 2:4 masks, group-wise asymmetric quantizer, AWQ-style scale search and folding, packed-byte accounting |
| `pa4/t1_run.py` | Task 1 driver: baseline, P1, P2, P3 (non-uniform sparsity), Q1, Q2, Q3 (mixed precision) |
| `pa4/t2a_attention.py` | Task 2A: dense, online-softmax tiled, and SDPA causal attention — correctness, gradients, timing, memory |
| `pa4/t2b_mqar.py` | Task 2B: MQAR training with dense / sliding-window / local+dilated / ELU+1 linear / hybrid attention |
| `pa4/t2c_kv.py` | Task 2C: KV-cache eviction (sink+recent, question-aware), INT4 layouts (token-wise, channel-K/token-V), INT2/FP16 hybrid at a matched byte budget; packed encode/decode benchmark |
| `pa4/t3_spec.py` | Task 3: exact speculative sampling core, toy Markov chi-square test, real-model decoder with cache rollback |
| `pa4/t3_distill.py` | Task 3: off-policy / on-policy forward-KL LoRA distillation, adapter merging |
| `pa4/t3_eval.py` | Task 3: prefix banks, distribution alignment, speculative benchmark, standalone accuracy, wrapper overhead |
| `pa4/prefix_banks.py` | runs the course `setup_data.py --prefix-banks` under the pinned transformers release |
| `pa4/report.py` | CPU-only summary tables (CSV + Markdown, bootstrap 95% CIs) and figures from the saved results |
| `tests/` | CPU unit tests on tiny random Qwen2 models (`PYTHONPATH=. python tests/test_t1.py`, ...) |
| `notebooks/PA4_kaggle.ipynb` | Kaggle runner (clones this repo; `PHASE = "smoke" | "A" | "B" | "C"`) |
| `build_notebook.py` | regenerates the runner notebook |

## Reproduction

```bash
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git
(cd atml-pa4-datasets && python scripts/setup_data.py --public)        # + --prefix-banks for Task 3 (GPU)
export PA4_DATA=$PWD/atml-pa4-datasets PA4_OUT=$PWD/results PYTHONPATH=$PWD/atml-pa4
cd atml-pa4
python -m pa4.t1_run --stages baseline,calib,prune,p3,quant,q3          # Task 1
python -m pa4.t2a_attention                                            # Task 2A
python -m pa4.t2b_mqar --runs all && python -m pa4.t2b_mqar --sweep     # Task 2B
python -m pa4.t2c_kv --stage validate,eval,bench                       # Task 2C
python -m pa4.t3_spec --toy                                            # Task 3 sampler tests
python -m pa4.t3_distill --method off --seed 6304                      # (repeat for on / 6305)
python -m pa4.t3_eval --stage validate,costs,specbank,studentbanks,align,spec,fixed64,accuracy,unmerged
```

On Kaggle, open `notebooks/PA4_kaggle.ipynb` with GPU T4 x2 and Internet on, set `PHASE`, and run. Phase `C` recomputes only the Task 3 alignment metrics from a finished Phase B (attach that version's output as an Input).
Teacher and original-draft prefix banks are built with `python -m pa4.prefix_banks`, which runs the course script `setup_data.py --prefix-banks` unchanged; the shim only adapts its `generate(generator=...)` call to the pinned transformers 4.57.1 by seeding the RNG from the same per-prompt seed.
To turn saved results into tables and figures, unpack the result archives into one folder and run
`python -m pa4.report --results results --out report_assets` (needs pandas and matplotlib; `tabulate` is optional).
The report skeleton (`report/main.tex`, `report/references.bib`) is filled by `python report/make_report_assets.py --results results --out report_tex`, which writes the LaTeX tables, caption macros and PDF figures (Phase `R` runs it on Kaggle and saves `report_tex.zip`).

Every stage writes machine-readable JSON/JSONL/NPZ files under `$PA4_OUT` and skips work that is already saved.
Setting `PA4_QUICK=1` runs a reduced smoke test.

## Implementation notes

* **Task 1.** Eligible weights: `q,k,v,o,gate,up,down` projections of all 28 blocks; embeddings (tied, counted once), norms, biases and LM head stay FP16.
  Quantizer: asymmetric, group 128 along input columns, round-half-to-even, FP16 scale + b-bit zero point (bit-packed).
  AWQ-style: per-group scale `s = mean|x|^α / sqrt(max·min)` with α ∈ {0, 0.1, …, 1.0} chosen on a fixed 2,048-token calibration subset;
  scales folded into RMSNorm weights (qkv, gate/up), into `v_proj` rows with GQA-shared scales (o), and into `up_proj` rows (down).
  The activation-aware clipping search of the published AWQ and its fused inference kernels are not reproduced; execution is dense FP16 with dequantized weights.
  P3 unit: Transformer block, rule `s_l = S + λ(1 − 2·minmax(sens_l))`, clipped to [0.10, 0.85] and re-centred; Q3 unit: (block, AWQ scaling group), exact multiple-choice knapsack under the uniform-Q2 packed-byte budget. Hyper-parameters are chosen on the 32-sequence C4 development set only.
* **Task 2C.** Cached keys are rotary-positioned; eviction keeps original absolute position ids and uses consecutive physical cache positions (validated against a 4-D-masked full forward). Question-aware scores are FP32 attention of the question suffix queries (SnapKV-inspired, simplified, shared by the two KV heads of a layer, chosen per layer).
* **Task 3.** Shared accept/residual/bonus core used by both the toy chi-square test and the real decoder; drafts are LoRA (r=8, α=16, q/k/v/o) on a frozen FP16 0.5B backbone, merged into FP16 checkpoints for timing.

## Attribution

Methods re-implemented from the papers: Wanda (Sun et al., 2024), AWQ (Lin et al., 2024), OWL outlier ratio (Yin et al., 2024), FlashAttention online softmax (Dao et al., 2022),
linear attention (Katharopoulos et al., 2020), MQAR (Arora et al., 2024), SnapKV (Li et al., 2024), KIVI (Liu et al., 2024), speculative sampling (Leviathan et al., 2023), DistillSpec / GKD (Zhou et al., 2024; Agarwal et al., 2024).
HellaSwag preprocessing and `acc_norm` follow the `hellaswag` task of EleutherAI lm-evaluation-harness. The MQAR generator and data manifests come from the course data repository.
Libraries: PyTorch, Hugging Face Transformers / Datasets / PEFT, NumPy, SciPy, pandas.
