"""Task 3: off-policy / on-policy forward-KL LoRA distillation of Qwen2.5-0.5B-Instruct toward 1.5B-Instruct.
python -m pa4.t3_distill --method off --seed 6304     (then --merge)"""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import OUT, DATA, SEED, read_jsonl, write_jsonl, save_json, load_json, env_info, revisions

R = OUT / "task3" / "distill"
CACHE = Path(os.environ.get("PA4_CACHE", "/tmp/pa4cache"))
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"; DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
QUOTA = dict(math=30000, short=10000) if not os.environ.get("PA4_QUICK") else dict(math=600, short=200); CAP = dict(math=192, short=32)
TOK_PER_UPDATE = 256; LR = 3e-4; WD = 0.01


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TARGET, revision=revisions().get("model_target"))


def load_model(name, key):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, revision=revisions().get(key), torch_dtype=torch.float16,
                                                attn_implementation="sdpa").cuda().eval()


def eos_ids(tok):
    return sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})


def chat_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def train_prompts():
    gsm = read_jsonl(DATA / "public_locked" / "gsm8k_train_locked.jsonl.gz")
    gsm.sort(key=lambda r: r["order"])
    short = read_jsonl(DATA / "short_form" / "short_train.jsonl")
    math_p = [dict(id=f"GSM8K-train-{r['order']:04d}", text=r["example"]["question"], cat="math") for r in gsm]
    short_p = [dict(id=r["id"], text=r["input"], cat="short") for r in short]
    assert len(math_p) == 500 and len(short_p) == 300
    return dict(math=math_p, short=short_p)


class Schedule:
    """Fixed seeded round-robin permutation within each category; the category whose quota is least
    consumed (fractionally) is presented next; prompts are reused after a category is exhausted."""
    def __init__(self, prompts, seed):
        rng = np.random.default_rng(seed)
        self.p = prompts; self.perm = {c: rng.permutation(len(v)).tolist() for c, v in prompts.items()}
        self.ptr = {c: 0 for c in prompts}; self.used = {c: 0 for c in prompts}

    def done(self): return all(self.used[c] >= QUOTA[c] for c in QUOTA)

    def next(self):
        c = min((c for c in QUOTA if self.used[c] < QUOTA[c]), key=lambda c: self.used[c] / QUOTA[c])
        i = self.perm[c][self.ptr[c] % len(self.perm[c])]; cycle = self.ptr[c] // len(self.perm[c]); self.ptr[c] += 1
        return c, self.p[c][i], cycle

    def take(self, c, n):
        n = min(n, QUOTA[c] - self.used[c]); self.used[c] += n; return n


@torch.inference_mode()
def sample(model, prompt_ids, cap, seed, eos):
    gen_ids = torch.as_tensor([prompt_ids], device="cuda")
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    out = model.generate(gen_ids, attention_mask=torch.ones_like(gen_ids), max_new_tokens=cap, do_sample=True,
                         temperature=1.0, top_p=1.0, top_k=0, pad_token_id=eos[0])[0, len(prompt_ids):].tolist()
    if any(t in eos for t in out):
        k = next(i for i, t in enumerate(out) if t in eos); return out[:k + 1], "eos"
    return out, "cap"


@torch.inference_mode()
def teacher_collection(teacher, tok, prompts, seed, bs=32):
    """Frozen teacher-response pool: one sampled response per training prompt (800), batched (left padding),
    one recorded seed per batch."""
    f = CACHE / f"teacher_pool_seed{seed}.jsonl"
    if f.exists(): return {r["id"]: r for r in read_jsonl(f)}, load_json(CACHE / f"teacher_pool_seed{seed}_meta.json")
    eos = eos_ids(tok); rows = []; t0 = time.time(); ntok = 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    for c in ("math", "short"):
        ps = prompts[c]
        for b in range(0, len(ps), bs):
            chunk = ps[b:b + bs]; enc = [chat_ids(tok, p["text"]) for p in chunk]; L = max(map(len, enc))
            ids = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device="cuda")
            att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device="cuda")
            bseed = seed * 100003 + (0 if c == "math" else 50000) + b
            torch.manual_seed(bseed); torch.cuda.manual_seed(bseed)
            out = teacher.generate(ids, attention_mask=att, max_new_tokens=CAP[c], do_sample=True, temperature=1.0,
                                   top_p=1.0, top_k=0, pad_token_id=pad)[:, L:].tolist()
            for p, o in zip(chunk, out):
                hit = [i for i, t in enumerate(o) if t in eos]
                resp, why = (o[:hit[0] + 1], "eos") if hit else (o, "cap")
                rows.append(dict(id=p["id"], cat=c, rollout_seed=bseed, response=resp, termination=why)); ntok += len(resp)
    meta = dict(seconds=time.time() - t0, responses=len(rows), tokens_generated=ntok, batch_size=bs)
    CACHE.mkdir(parents=True, exist_ok=True); write_jsonl(f, rows); save_json(CACHE / f"teacher_pool_seed{seed}_meta.json", meta)
    return {r["id"]: r for r in rows}, meta


def make_student(seed):
    from peft import LoraConfig, get_peft_model
    base = load_model(DRAFT, "model_draft")
    for p in base.parameters(): p.requires_grad_(False)
    torch.manual_seed(seed)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(base, cfg)
    for n, p in m.named_parameters():
        if p.requires_grad and p.dtype != torch.float32: p.data = p.data.float()
    return m


def train(method, seed):
    out = R / f"{method}_seed{seed}"
    if (out / "summary.json").exists(): return load_json(out / "summary.json")
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tok(); eos = eos_ids(tok); prompts = train_prompts()
    teacher = load_model(TARGET, "model_target")
    for p in teacher.parameters(): p.requires_grad_(False)
    cost = dict(teacher_generation_s=0.0, student_rollout_s=0.0, teacher_scoring_s=0.0, student_fwd_bwd_s=0.0, optimizer_s=0.0)
    pool = None
    if method == "off":
        pool, meta = teacher_collection(teacher, tok, prompts, seed); cost["teacher_generation_s"] = meta["seconds"]
        cost["teacher_pool_tokens"] = meta["tokens_generated"]
    torch.cuda.reset_peak_memory_stats()
    student = make_student(seed); student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params); n_total = sum(p.numel() for p in student.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    sch = Schedule(prompts, seed)
    total_quota = sum(QUOTA.values())
    n_updates_planned = math.ceil(total_quota / TOK_PER_UPDATE)
    buf = 0; upd = 0; acc_loss = 0.0; pres = []; log = []; skipped = 0; scored = 0
    t_start = time.time(); k_pres = 0

    def update_size(u):  # tokens in update u (last may be short)
        return min(TOK_PER_UPDATE, total_quota - u * TOK_PER_UPDATE)

    while not sch.done():
        cat, p, cycle = sch.next(); pid = chat_ids(tok, p["text"])
        if method == "off":
            r = pool[p["id"]]; resp, why, rs = r["response"], r["termination"], r["rollout_seed"]
        else:
            rs = seed * 100003 + 7 * k_pres + 1
            student.eval(); t0 = time.time()
            resp, why = sample(student, pid, CAP[cat], rs, eos)
            torch.cuda.synchronize(); cost["student_rollout_s"] += time.time() - t0; student.train()
        k_pres += 1
        if len(resp) == 0: continue
        n_take = sch.take(cat, len(resp))
        if n_take < len(resp): why = why + "+quota_truncated"
        resp = resp[:n_take]
        ids = torch.as_tensor([pid + resp], device="cuda"); P0 = len(pid)
        t0 = time.time()
        with torch.no_grad():
            tl = teacher(input_ids=ids).logits[0, P0 - 1:P0 - 1 + n_take].float()
            logp_t = torch.log_softmax(tl, -1); p_t = logp_t.exp()
        torch.cuda.synchronize(); cost["teacher_scoring_s"] += time.time() - t0
        pres.append(dict(k=k_pres, source_id=p["id"], cat=cat, cycle=cycle, rollout_seed=rs, response_ids=resp,
                         termination=why, teacher_scored_tokens=n_take, update_index=upd))
        # split this response across optimizer updates of exactly 256 scored tokens
        start = 0
        while start < n_take:
            need = update_size(upd) - buf; seg = slice(start, min(n_take, start + need))
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                sl = student(input_ids=ids).logits[0, P0 - 1:P0 - 1 + n_take]
            logq = torch.log_softmax(sl.float()[seg], -1)
            kl_tok = (p_t[seg] * (logp_t[seg] - logq)).sum(-1)  # forward KL(p_T || q_phi) per position
            loss = kl_tok.sum() / update_size(upd)
            scaler.scale(loss).backward()
            torch.cuda.synchronize(); cost["student_fwd_bwd_s"] += time.time() - t0
            acc_loss += float(kl_tok.sum()); buf += seg.stop - seg.start; scored += seg.stop - seg.start
            start = seg.stop
            if buf == update_size(upd):
                t0 = time.time()
                scaler.unscale_(opt); gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                s_before = scaler.get_scale(); scaler.step(opt); scaler.update()
                was_skipped = scaler.get_scale() < s_before; skipped += int(was_skipped)
                opt.zero_grad(set_to_none=True); cost["optimizer_s"] += time.time() - t0
                log.append(dict(update=upd, tokens=buf, kl_per_token=acc_loss / buf, grad_norm=gn, lr=LR,
                                skipped=bool(was_skipped), scale=scaler.get_scale(), scored_cum=scored,
                                math_cum=sch.used["math"], short_cum=sch.used["short"], elapsed_s=time.time() - t_start))
                if upd % 10 == 0: print(method, seed, log[-1], flush=True)
                upd += 1; buf = 0; acc_loss = 0.0
                # any remaining segment of this response is re-forwarded above with the updated parameters
    train_s = time.time() - t_start
    student.save_pretrained(out / "adapter")
    summ = dict(method=method, seed=seed, updates=upd, updates_planned=n_updates_planned, skipped_updates=skipped,
                scored_tokens=scored, quota=QUOTA, presentations=len(pres), trainable_params=n_train, total_params=n_total,
                adapter_dtype=str(next(p.dtype for p in params)), wall_s=train_s, cost=cost,
                peak_alloc_MiB=torch.cuda.max_memory_allocated() / 2**20,
                mean_response_len={c: float(np.mean([x["response_ids"].__len__() for x in pres if x["cat"] == c])) for c in QUOTA},
                termination_counts={w: sum(1 for x in pres if x["termination"] == w) for w in set(x["termination"] for x in pres)},
                env=env_info())
    write_jsonl(out / "presentations.jsonl.gz", pres); write_jsonl(out / "updates.jsonl", log)
    save_json(out / "summary.json", summ)
    return summ


def merge(method, seed):
    """Merge LoRA into a fresh FP16 0.5B backbone; verify merged vs unmerged logits."""
    from peft import PeftModel
    out = R / f"{method}_seed{seed}"; dst = CACHE / f"merged_{method}_seed{seed}"
    if (out / "merge.json").exists() and dst.exists(): return load_json(out / "merge.json")
    tok = load_tok()
    t0 = time.time()
    base = load_model(DRAFT, "model_draft")
    pm = PeftModel.from_pretrained(base, str(out / "adapter")).eval()
    texts = ["Natalia sold clips to 48 of her friends in April. How many in total?", "Return only the year: March 3, 1999.",
             "Explain photosynthesis briefly.", "What is 17*23?"]
    batch = [chat_ids(tok, t) for t in texts]
    with torch.no_grad():
        un = [pm(input_ids=torch.as_tensor([b], device="cuda")).logits[0].float() for b in batch]
    merged = pm.merge_and_unload(); merged.half()
    torch.cuda.synchronize(); merge_s = time.time() - t0
    with torch.no_grad():
        mg = [merged(input_ids=torch.as_tensor([b], device="cuda")).logits[0].float() for b in batch]
    err = max(float((a - b).abs().max()) for a, b in zip(un, mg))
    agree = float(np.mean([(a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(un, mg)]))
    merged.save_pretrained(dst); tok.save_pretrained(dst)
    res = dict(merge_s=merge_s, max_abs_logit_diff_merged_vs_unmerged=err, argmax_agreement=agree, path=str(dst))
    save_json(out / "merge.json", res); return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["off", "on"], required=True)
    ap.add_argument("--seed", type=int, default=6304)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    if not a.merge_only: print(train(a.method, a.seed))
    print(merge(a.method, a.seed))
