"""Task 3: exact speculative sampling (shared accept/residual/bonus core), toy chi-square tests, real-model decoder.
python -m pa4.t3_spec --toy        (CPU OK)"""
from __future__ import annotations
import argparse, itertools, time
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from .common import OUT, SEED, save_json

R = OUT / "task3"


# ====================== shared exact core ======================
def verify_round(P, Q, props, gen, eos_ids=(), residual="correct"):
    """P: (g+1, V) target probs at proposal positions + bonus position; Q: (g, V) draft probs; props: list[int].
    Returns emitted tokens (accepted prefix + residual/bonus token), n_accepted, n_examined, hit_eos.
    Acceptance a = min(1, p(x)/q(x)); on rejection sample from normalized [p-q]+ ; all accepted -> bonus ~ p_{g+1}."""
    g = len(props); emitted = []
    for i, x in enumerate(props):
        p, q = P[i], Q[i]
        u = torch.rand((), generator=gen, device=P.device)
        if u * q[x] <= p[x]:  # u < p/q  (q[x] > 0 since x ~ q)
            emitted.append(x)
            if x in eos_ids: return emitted, i + 1, i + 1, True
            continue
        if residual == "correct":
            r = (p - q).clamp(min=0); s = r.sum()
            r = r / s if s > 0 else p  # s == 0 only if p == q exactly (then rejection has prob 0)
        else:  # deliberately WRONG negative control: resample from p instead of the residual
            r = p
        y = int(torch.multinomial(r, 1, generator=gen))
        emitted.append(y)
        return emitted, i, i + 1, y in eos_ids
    y = int(torch.multinomial(P[g], 1, generator=gen))
    emitted.append(y)
    return emitted, g, g, y in eos_ids


# ====================== toy Markov chi-square test ======================
def toy_models(V=3):
    pi_p = torch.tensor([0.5, 0.3, 0.2]); A_p = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.25, 0.25, 0.5]])
    pi_q = torch.tensor([0.2, 0.3, 0.5]); A_q = torch.tensor([[0.2, 0.2, 0.6], [0.5, 0.25, 0.25], [0.1, 0.6, 0.3]])
    # second-order dependence for the third symbol makes it genuinely multi-step
    def dist(pi, A, seq, flip):
        if len(seq) == 0: return pi
        d = A[seq[-1]].clone()
        if len(seq) == 2 and seq[0] == 0: d = d.flip(0) if flip else d
        return d / d.sum()
    tp = lambda s: dist(pi_p, A_p, s, True)
    dq = lambda s: dist(pi_q, A_q, s, False)
    return tp, dq


def toy_generate(tp, dq, gamma, length, gen, residual="correct", stats=None):
    seq = []
    while len(seq) < length:
        g = min(gamma, length - len(seq))  # variable proposal length near the end
        props, Q = [], []
        for _ in range(g):
            q = dq(seq + props); x = int(torch.multinomial(q, 1, generator=gen)); props.append(x); Q.append(q)
        P = torch.stack([tp(seq + props[:i]) for i in range(g + 1)])
        emitted, n_acc, n_ex, _ = verify_round(P, torch.stack(Q), props, gen, residual=residual)
        if stats is not None:
            stats["rounds"] += 1; stats["accepted"] += n_acc; stats["examined"] += n_ex; stats["proposed"] += g
            stats["full_accept_rounds"] += int(n_acc == g); stats["one_token_rounds"] += int(len(emitted) == 1)
        seq += emitted
    return tuple(seq[:length])  # final-token truncation


def toy_test(n=100_000, length=3, seed=SEED):
    from scipy.stats import chisquare
    tp, dq = toy_models()
    seqs = list(itertools.product(range(3), repeat=length))
    def joint(fn, s):
        pr = 1.0
        for t in range(length): pr *= float(fn(list(s[:t]))[s[t]])
        return pr
    p_joint = np.array([joint(tp, s) for s in seqs]); q_joint = np.array([joint(dq, s) for s in seqs])
    assert abs(p_joint.sum() - 1) < 1e-6 and (n * p_joint).min() >= 10
    res = dict(n=n, length=length, sequences=[list(s) for s in seqs], target_joint=p_joint.tolist(), draft_joint=q_joint.tolist())
    for gamma in (1, 2, 4):
        for mode in ("correct", "wrong_residual_is_p"):
            gen = torch.Generator().manual_seed(seed + gamma)
            st = dict(rounds=0, accepted=0, examined=0, proposed=0, full_accept_rounds=0, one_token_rounds=0)
            counts = {s: 0 for s in seqs}
            t0 = time.time()
            for _ in range(n): counts[toy_generate(tp, dq, gamma, length, gen, "correct" if mode == "correct" else "p", st)] += 1
            obs = np.array([counts[s] for s in seqs]); exp = n * p_joint / p_joint.sum()
            chi = chisquare(obs, exp)
            res[f"gamma{gamma}_{mode}"] = dict(chi2=float(chi.statistic), p_value=float(chi.pvalue), dof=len(seqs) - 1,
                                               rejected_at_p01=bool(chi.pvalue < 0.01), empirical=(obs / n).tolist(),
                                               max_abs_freq_err=float(np.abs(obs / n - p_joint).max()), stats=st,
                                               seconds=time.time() - t0)
            print(gamma, mode, round(chi.statistic, 2), chi.pvalue, flush=True)
    return res


def unit_tests():
    """Deterministic edge cases of the shared core."""
    out = {}
    gen = torch.Generator().manual_seed(0)
    V = 5
    # identical p, q -> always full acceptance + bonus
    p = torch.full((3, V), 0.2); q = torch.full((2, V), 0.2)
    em, na, ne, _ = verify_round(p, q, [1, 2], gen); out["identical_full_accept"] = (na == 2 and len(em) == 3)
    # q puts mass where p is zero -> rejection, residual can never emit that token
    p = torch.tensor([[0.0, 1.0, 0, 0, 0]] * 2); q = torch.tensor([[1.0, 0, 0, 0, 0]])
    em, na, ne, _ = verify_round(p, q, [0], gen); out["forced_reject_residual_token1"] = (na == 0 and em == [1])
    # EOS accepted mid-proposal stops the round and drops later proposals
    p = torch.full((4, V), 0.2); q = torch.full((3, V), 0.2)
    em, na, ne, eos = verify_round(p, q, [3, 4, 2], gen, eos_ids=(4,)); out["eos_mid_round"] = (em == [3, 4] and eos)
    # one-token round: first proposal rejected
    p = torch.tensor([[0.0, 0, 1, 0, 0]] * 3); q = torch.tensor([[0.5, 0.5, 0, 0, 0]] * 2)
    em, na, ne, _ = verify_round(p, q, [0, 1], gen); out["one_token_round"] = (em == [2] and ne == 1)
    assert all(out.values()), out
    return out


# ====================== real-model exact speculative decoder ======================
def probs(logits, temperature=1.0):
    return torch.softmax(logits.float() / temperature, -1)


class Timers:
    def __init__(self, sync=True): self.t = dict(draft=0.0, verify=0.0, accept=0.0, rollback=0.0, prefill=0.0); self.sync = sync

    def tic(self):
        if self.sync and torch.cuda.is_available(): torch.cuda.synchronize()
        return time.perf_counter()

    def toc(self, k, t0):
        if self.sync and torch.cuda.is_available(): torch.cuda.synchronize()
        self.t[k] += time.perf_counter() - t0


@torch.inference_mode()
def _step(model, toks, cache, start):
    ids = torch.as_tensor([toks], device=model.device)
    return model(input_ids=ids, past_key_values=cache, use_cache=True,
                 cache_position=torch.arange(start, start + len(toks), device=model.device)).logits[0]


@torch.inference_mode()
def speculative_generate(target, draft, prompt, gamma, max_new, gen, eos_ids, ignore_eos=False, temperature=1.0,
                         record_p=False, residual="correct"):
    """Exact speculative sampling, batch 1. Caches always hold the committed sequence minus a pending suffix."""
    tm = Timers(); t0 = tm.tic()
    tc, dc = DynamicCache(), DynamicCache()
    _step(target, prompt[:-1], tc, 0); _step(draft, prompt[:-1], dc, 0)
    tm.toc("prefill", t0)
    t_pend, d_pend = [prompt[-1]], [prompt[-1]]
    out, st = [], dict(rounds=0, proposed=0, examined=0, accepted=0, draft_forwards=0, target_forwards=0,
                       full_accept_rounds=0, accepted_per_round=[])
    used_p = [] if record_p else None
    eos = () if ignore_eos else tuple(eos_ids)
    stop_reason = "length"
    while len(out) < max_new:
        g = min(gamma, max_new - len(out))
        # ---- draft proposals
        t0 = tm.tic()
        Ld = dc.get_seq_length()
        lg = _step(draft, d_pend, dc, Ld)[-1]; st["draft_forwards"] += 1
        props, Q = [], []
        for i in range(g):
            q = probs(lg, temperature); x = int(torch.multinomial(q, 1, generator=gen)); props.append(x); Q.append(q)
            if i < g - 1:
                lg = _step(draft, [x], dc, dc.get_seq_length())[-1]; st["draft_forwards"] += 1
        tm.toc("draft", t0)
        # ---- one target verification pass over pending + proposals
        t0 = tm.tic()
        Lt = tc.get_seq_length()
        tl = _step(target, t_pend + props, tc, Lt); st["target_forwards"] += 1
        P = probs(tl[len(t_pend) - 1:], temperature)  # g+1 rows
        tm.toc("verify", t0)
        # ---- accept / residual / bonus
        t0 = tm.tic()
        emitted, n_acc, n_ex, hit_eos = verify_round(P, torch.stack(Q), props, gen, eos, residual)
        tm.toc("accept", t0)
        if record_p:
            for j in range(len(emitted)): used_p.append(P[j].half().cpu())
        # ---- rollback / advance caches
        t0 = tm.tic()
        tc.crop(Lt + len(t_pend) + n_acc)
        keep_d = Ld + len(d_pend) + min(n_acc, g - 1)
        dc.crop(keep_d)
        d_pend = ([props[-1]] if n_acc == g else []) + [emitted[-1]]
        t_pend = [emitted[-1]]
        tm.toc("rollback", t0)
        st["rounds"] += 1; st["proposed"] += g; st["examined"] += n_ex; st["accepted"] += n_acc
        st["full_accept_rounds"] += int(n_acc == g); st["accepted_per_round"].append(n_acc)
        out += emitted
        if hit_eos:
            stop_reason = "eos"; break
    truncated = len(out) > max_new
    out = out[:max_new]
    if record_p: used_p = used_p[:len(out)]
    st.update(output_tokens=len(out), stop_reason=stop_reason, truncated_last_round=truncated, timers=tm.t)
    return out, st, used_p


@torch.inference_mode()
def target_generate(target, prompt, max_new, gen, eos_ids, ignore_eos=False, temperature=1.0):
    tm = Timers(); t0 = tm.tic(); c = DynamicCache()
    lg = _step(target, prompt, c, 0)[-1]; tm.toc("prefill", t0)
    out = []; t0 = tm.tic(); reason = "length"
    while len(out) < max_new:
        x = int(torch.multinomial(probs(lg, temperature), 1, generator=gen)); out.append(x)
        if not ignore_eos and x in eos_ids: reason = "eos"; break
        lg = _step(target, [x], c, c.get_seq_length())[-1]
    tm.toc("verify", t0)
    return out, dict(output_tokens=len(out), stop_reason=reason, target_forwards=len(out), timers=tm.t)


@torch.inference_mode()
def check_cache_consistency(target, draft, prompt, eos_ids, gamma=4, n=48, seed=SEED):
    """The target distributions used inside speculation must equal a single full forward over prompt+output."""
    gen = torch.Generator(device=target.device).manual_seed(seed)
    out, st, used = speculative_generate(target, draft, prompt, gamma, n, gen, eos_ids, record_p=True)
    dev = target.device
    full = probs(target(input_ids=torch.as_tensor([prompt + out], device=dev)).logits[0, len(prompt) - 1:len(prompt) - 1 + len(out)])
    used = torch.stack(used).float().to(dev)
    # split-prefill equivalence for the draft as well
    c = DynamicCache(); a = _step(draft, prompt[:-1], c, 0); b = _step(draft, [prompt[-1]], c, len(prompt) - 1)[-1]
    ref = draft(input_ids=torch.as_tensor([prompt], device=dev)).logits[0, -1]
    return dict(max_abs_prob_diff_target=float((used - full).abs().max()), n_tokens=len(out), stats={k: v for k, v in st.items() if k != "timers"},
                draft_split_prefill_max_abs_logit=float((b.float() - ref.float()).abs().max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", action="store_true"); ap.add_argument("--n", type=int, default=100_000)
    a = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    if a.toy:
        save_json(R / "sampler_unit_tests.json", unit_tests())
        save_json(R / "sampler_toy_chi2.json", toy_test(a.n))
