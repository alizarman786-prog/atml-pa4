"""Task 3 evaluation.
python -m pa4.t3_eval --stage validate   sampler checks on real models + baseline cost probes
python -m pa4.t3_eval --stage specbank   exact-speculative prefix bank (original draft, gamma=4)
python -m pa4.t3_eval --stage studentbanks   own-policy rollout banks for each trained student
python -m pa4.t3_eval --stage align      KL / TV / agreement on all banks for all drafts
python -m pa4.t3_eval --stage spec [--drafts original,off_6304,...]   EOS-aware 120-prompt speculative benchmark
python -m pa4.t3_eval --stage fixed64    24 math prompts, ignore EOS (timing diagnostic)
python -m pa4.t3_eval --stage accuracy   standalone answer quality (512 math / 32 short)
python -m pa4.t3_eval --stage unmerged   wrapper overhead of an unmerged adapter"""
from __future__ import annotations
import argparse, math, os, random, re, time
from pathlib import Path
import numpy as np
import torch
from .common import OUT, DATA, SEED, read_jsonl, write_jsonl, save_json, load_json, env_info, revisions, summarize_ms
from . import t3_spec as S
from .t3_distill import TARGET, DRAFT, CACHE, load_tok, load_model, eos_ids, chat_ids

R = OUT / "task3" / "eval"
DRAFTS = ("original", "off_6304", "off_6305", "on_6304", "on_6305")
BANK_FILE = DATA / "prefix_banks" / "prefix_banks_locked.jsonl.gz"


def draft_model(name):
    if name == "original": return load_model(DRAFT, "model_draft")
    from transformers import AutoModelForCausalLM
    m, s = name.split("_")
    if not (CACHE / f"merged_{m}_seed{s}").exists():   # cache lives in /tmp: rebuild from the saved adapter
        from .t3_distill import merge
        merge(m, int(s))
    return AutoModelForCausalLM.from_pretrained(CACHE / f"merged_{m}_seed{s}", torch_dtype=torch.float16,
                                                attn_implementation="sdpa").cuda().eval()


def test_prompts():
    gsm = sorted(read_jsonl(DATA / "public_locked" / "gsm8k_test_locked.jsonl.gz"), key=lambda r: r["order"])
    short = read_jsonl(DATA / "short_form" / "short_test.jsonl")
    out = [dict(id=f"GSM8K-{r['order']:04d}", text=r["example"]["question"], domain="math",
                gold=r["example"]["answer"].split("####")[-1].strip().replace(",", "")) for r in gsm]
    out += [dict(id=r["id"], text=r["input"], domain="short_" + r["split"], gold=r["answer"]) for r in short]
    assert len(out) == 120
    if os.environ.get("PA4_QUICK"): out = out[:3] + out[60:62] + out[90:92]
    return out


# ---------------- banks ----------------
def bank_positions(pid, which, L):
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{which}"); pos = []
    for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
        cand = list(range(lo, min(hi, L) + 1)); pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


def gen_bank(tok, target, draft, which, kind, seed_off):
    """kind: 'spec' (exact speculative, gamma 4) or 'student' (draft's own sampling). Mirrors setup_data selection."""
    eos = eos_ids(tok); rows = []
    for j, p in enumerate(test_prompts()):
        ids = chat_ids(tok, p["text"]); seed = 6304 + 10007 * j + seed_off
        gen = torch.Generator(device="cuda").manual_seed(seed)
        if kind == "spec":
            seq, st, _ = S.speculative_generate(target, draft, ids, 4, 192, gen, eos)
        else:
            seq, st = S.target_generate(draft, ids, 192, gen, eos)
        for pos in bank_positions(p["id"], which, len(seq)):
            rows.append(dict(prompt_id=p["id"], domain="math" if p["domain"] == "math" else "short", bank=which,
                             position=pos, prompt=p["text"], response_prefix_token_ids=seq[:pos - 1], source_length=len(seq),
                             source_generation_seed=seed))
        if (j + 1) % 20 == 0: print(which, j + 1, flush=True)
    return rows


def load_banks():
    rows = read_jsonl(BANK_FILE) if BANK_FILE.exists() else []
    for f in sorted((R / "banks").glob("*.jsonl.gz")): rows += read_jsonl(f)
    return rows


# ---------------- distribution alignment ----------------
@torch.inference_mode()
def dist_metrics(p_logits, q_logits):
    lp = torch.log_softmax(p_logits.float(), -1); lq = torch.log_softmax(q_logits.float(), -1)
    p, q = lp.exp(), lq.exp()
    return dict(fkl=(p * (lp - lq)).sum(-1), rkl=(q * (lq - lp)).sum(-1), tv=0.5 * (p - q).abs().sum(-1),
                agree=(p.argmax(-1) == q.argmax(-1)).float())


@torch.inference_mode()
def stage_align(tok, target, drafts=DRAFTS):
    """Streams over (bank, prompt) groups: one target forward and one forward per draft on prompt + longest prefix,
    metrics computed on the GPU for the needed rows only (no full-vocab rows kept in host RAM)."""
    banks = load_banks(); short_split = {r["id"]: r["split"] for r in read_jsonl(DATA / "short_form" / "short_test.jsonl")}
    groups = {}
    for i, r in enumerate(banks): groups.setdefault((r["bank"], r["prompt_id"]), []).append(i)
    models = {}
    for d in drafts:
        try: models[d] = draft_model(d)
        except Exception as e: print("skip", d, e, flush=True)
    res = []
    for g, ((bk, pid), idx) in enumerate(groups.items()):
        recs = [banks[i] for i in idx]; longest = max(recs, key=lambda r: len(r["response_prefix_token_ids"]))
        pids = chat_ids(tok, recs[0]["prompt"]); seq = torch.as_tensor([pids + longest["response_prefix_token_ids"]], device="cuda")
        for r in recs: assert r["response_prefix_token_ids"] == longest["response_prefix_token_ids"][:len(r["response_prefix_token_ids"])]
        rows = torch.as_tensor([len(pids) + len(r["response_prefix_token_ids"]) - 1 for r in recs], device="cuda")
        t_lg = target(input_ids=seq).logits[0].index_select(0, rows)
        for d, m in models.items():
            met = dist_metrics(t_lg, m(input_ids=seq).logits[0].index_select(0, rows))
            met = {k: v.float().cpu().tolist() for k, v in met.items()}
            for j, r in enumerate(recs):
                dom = r["domain"] if r["domain"] == "math" else ("short_" + short_split.get(r["prompt_id"], "?"))
                res.append(dict(draft=d, bank=r["bank"], prompt_id=r["prompt_id"], position=r["position"], domain=dom,
                                **{k: met[k][j] for k in met}, exp_accept=1 - met["tv"][j]))
        if (g + 1) % 100 == 0: print("aligned groups", g + 1, "/", len(groups), flush=True)
    write_jsonl(R / "alignment_per_prefix.jsonl.gz", res)
    print("aligned", sorted(models), len(res), "rows", flush=True)


# ---------------- speculative benchmark ----------------
@torch.inference_mode()
def stage_spec(tok, target, drafts, cap=64, ignore_eos=False, prompts=None, tag="spec", reps=2, gammas=(2, 4)):
    reps = 1 if os.environ.get("PA4_QUICK") else reps
    eos = eos_ids(tok); prompts = prompts or test_prompts()
    f = R / f"{tag}_per_run.jsonl"
    rows = read_jsonl(f) if f.exists() else []
    done = {(r["draft"], r["prompt_id"], r["gamma"], r["rep"], r["mode"]) for r in rows}
    for d in drafts:
        dm = draft_model(d)
        for _ in range(2):  # warm-up
            S.speculative_generate(target, dm, chat_ids(tok, "warm up"), 4, 16, torch.Generator(device="cuda").manual_seed(0), eos)
            S.target_generate(target, chat_ids(tok, "warm up"), 16, torch.Generator(device="cuda").manual_seed(0), eos)
        for j, p in enumerate(prompts):
            ids = chat_ids(tok, p["text"])
            for g in gammas:
                for rep in range(reps):
                    seed = SEED + 1000 * j + 10 * rep + g
                    order = ("spec", "target") if (j + rep + g) % 2 == 0 else ("target", "spec")  # counterbalanced
                    for mode in order:
                        if (d, p["id"], g, rep, mode) in done: continue
                        gen = torch.Generator(device="cuda").manual_seed(seed)
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                        if mode == "spec": out, st, _ = S.speculative_generate(target, dm, ids, g, cap, gen, eos, ignore_eos)
                        else: out, st = S.target_generate(target, ids, cap, gen, eos, ignore_eos)
                        torch.cuda.synchronize(); wall = time.perf_counter() - t0
                        st = {k: v for k, v in st.items()}
                        rows.append(dict(draft=d, prompt_id=p["id"], domain=p["domain"], gamma=g, rep=rep, mode=mode, seed=seed,
                                         order=order.index(mode), prompt_len=len(ids), wall_s=wall, tokens=len(out),
                                         tok_per_s=len(out) / wall, peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20,
                                         text=tok.decode(out)[:300] if rep == 0 else None, **st))
            if (j + 1) % 10 == 0:
                write_jsonl(f, rows); print(tag, d, j + 1, flush=True)
        del dm; torch.cuda.empty_cache()
    write_jsonl(f, rows)


@torch.inference_mode()
def stage_costs(tok, target, drafts=DRAFTS):
    """Isolated decode-step costs (c, v_gamma) at several prompt lengths for target and each draft."""
    res = []
    g = torch.Generator().manual_seed(SEED)
    for d in ["target"] + list(drafts):
        m = target if d == "target" else draft_model(d)
        for L in (64, 256, 512):
            ids = torch.randint(1000, 100000, (1, L), generator=g).tolist()[0]
            from transformers.cache_utils import DynamicCache
            def one(k):
                c = DynamicCache(); S._step(m, ids, c, 0)
                torch.cuda.synchronize(); t0 = time.perf_counter(); S._step(m, ids[:k], c, L); torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1e3
            for k in (1, 3, 5):
                for _ in range(3): one(k)
                ts = [one(k) for _ in range(10)]
                res.append(dict(model=d, prompt_len=L, n_tokens=k, **summarize_ms(ts)))
        if d != "target": del m; torch.cuda.empty_cache()
    write_jsonl(R / "decode_costs.jsonl", res)


# ---------------- standalone accuracy ----------------
def extract_number(t):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def norm(s): return re.sub(r"[\s\.\"']+$", "", s.strip()).strip().casefold()


@torch.inference_mode()
def stage_accuracy(tok, target, drafts=DRAFTS, reps=2, bs=20):
    eos = eos_ids(tok); prompts = test_prompts(); rows = []
    for d in ["target"] + list(drafts):
        m = target if d == "target" else draft_model(d)
        for rep in range(reps):
            for dom, cap in (("math", 512), ("short", 32)):
                ps = [p for p in prompts if p["domain"].startswith(dom)]
                for s in range(0, len(ps), bs):
                    chunk = ps[s:s + bs]; enc = [chat_ids(tok, p["text"]) for p in chunk]
                    L = max(map(len, enc)); pad = tok.pad_token_id or eos[0]
                    ids = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device="cuda")
                    att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device="cuda")
                    torch.manual_seed(SEED + rep); torch.cuda.manual_seed(SEED + rep)
                    out = m.generate(ids, attention_mask=att, max_new_tokens=cap, do_sample=True, temperature=1.0, top_p=1.0,
                                     top_k=0, pad_token_id=pad)[:, L:]
                    for p, o in zip(chunk, out.tolist()):
                        hit = [i for i, t in enumerate(o) if t in eos]; o2 = o[:hit[0]] if hit else o
                        text = tok.decode(o2, skip_special_tokens=True)
                        r = dict(model=d, rep=rep, prompt_id=p["id"], domain=p["domain"], gold=p["gold"], truncated=not hit,
                                 n_tokens=len(o2), text=text[:600])
                        if dom == "math":
                            pred = extract_number(text); r.update(pred=pred, extractable=pred is not None,
                                                                  correct=pred is not None and _num_eq(pred, p["gold"]))
                        else:
                            r.update(correct=norm(text) == norm(p["gold"]), contains=norm(p["gold"]) in norm(text),
                                     format_compliant=("\n" not in text.strip()) and len(text.strip()) <= max(2 * len(p["gold"]), 12))
                        rows.append(r)
        print("accuracy", d, np.mean([r["correct"] for r in rows if r["model"] == d]), flush=True)
        if d != "target": del m; torch.cuda.empty_cache()
    write_jsonl(R / f"standalone_accuracy_{'_'.join(drafts)}.jsonl", rows)


def _num_eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-6
    except Exception: return False


@torch.inference_mode()
def stage_validate(tok, target):
    dm = draft_model("original"); eos = eos_ids(tok)
    # tokenizer / chat template agreement between target and draft
    from transformers import AutoTokenizer
    dt = AutoTokenizer.from_pretrained(DRAFT, revision=revisions().get("model_draft"))
    texts = [p["text"] for p in test_prompts()[::10]]
    tok_ok = all(chat_ids(tok, t) == dt.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True) for t in texts)
    checks = [S.check_cache_consistency(target, dm, chat_ids(tok, t), eos, gamma=g) for t in texts[:3] for g in (2, 4)]
    save_json(R / "validate.json", dict(tokenizer_chat_template_agree=tok_ok, vocab=(target.config.vocab_size, dm.config.vocab_size),
                                        eos_ids=eos, cache_checks=checks, env=env_info()))
    print(checks, flush=True)


@torch.inference_mode()
def stage_unmerged(tok, target, adapter="off_6304", n=20):
    """Wrapper overhead: same exact sampler with the PEFT (unmerged) adapter vs the merged FP16 checkpoint."""
    from peft import PeftModel
    m, s = adapter.split("_")
    pm = PeftModel.from_pretrained(load_model(DRAFT, "model_draft"),
                                   str(OUT / "task3" / "distill" / f"{m}_seed{s}" / "adapter")).eval()
    merged = draft_model(adapter); eos = eos_ids(tok); rows = []
    for name, dm in (("unmerged", pm), ("merged", merged)):
        S.speculative_generate(target, dm, chat_ids(tok, "warm up"), 4, 16, torch.Generator(device="cuda").manual_seed(0), eos)
    for j, p in enumerate(test_prompts()[:n]):
        ids = chat_ids(tok, p["text"])
        for name, dm in (("unmerged", pm), ("merged", merged)):
            gen = torch.Generator(device="cuda").manual_seed(SEED + j)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out, st, _ = S.speculative_generate(target, dm, ids, 4, 64, gen, eos); torch.cuda.synchronize()
            rows.append(dict(path=name, prompt_id=p["id"], wall_s=time.perf_counter() - t0, tokens=len(out),
                             timers=st["timers"], draft_forwards=st["draft_forwards"], accepted=st["accepted"]))
    write_jsonl(R / "unmerged_vs_merged.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--drafts", default=",".join(DRAFTS))
    ap.add_argument("--tag", default="spec")
    ap.add_argument("--no-target-acc", action="store_true")
    a = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True); (R / "banks").mkdir(exist_ok=True)
    tok = load_tok(); target = load_model(TARGET, "model_target")
    drafts = a.drafts.split(",")
    for st in a.stage.split(","):
        if st == "validate": stage_validate(tok, target)
        elif st == "costs": stage_costs(tok, target, drafts)
        elif st == "specbank":
            dm = draft_model("original")
            write_jsonl(R / "banks" / "exact_spec_original_gamma4.jsonl.gz", gen_bank(tok, target, dm, "exact_spec_original_gamma4", "spec", 777))
        elif st == "studentbanks":
            for d in [x for x in drafts if x != "original"]:
                f = R / "banks" / f"student_{d}.jsonl.gz"
                if f.exists(): continue
                dm = draft_model(d); write_jsonl(f, gen_bank(tok, target, dm, f"student_{d}", "student", 555)); del dm
        elif st == "align": stage_align(tok, target, drafts)
        elif st == "spec": stage_spec(tok, target, drafts, tag=a.tag)
        elif st == "fixed64":
            ps = [p for p in test_prompts() if p["domain"] == "math"][:24]
            stage_spec(tok, target, drafts, cap=64, ignore_eos=True, prompts=ps, tag="fixed64_" + a.tag, reps=1)
        elif st == "accuracy": stage_accuracy(tok, target, drafts, reps=1 if os.environ.get("PA4_QUICK") else 2)
        elif st == "unmerged": stage_unmerged(tok, target)


if __name__ == "__main__":
    main()
