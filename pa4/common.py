"""Shared helpers for PA4: paths, seeding, environment capture, CUDA timing, bootstrap."""
from __future__ import annotations
import gzip, json, os, platform, random, time, contextlib
from pathlib import Path
import numpy as np

SEED = 6304
DATA_REPO = Path(os.environ.get("PA4_DATA", "/kaggle/working/atml-pa4-datasets"))
DATA = DATA_REPO / "data"
OUT = Path(os.environ.get("PA4_OUT", "/kaggle/working/results"))


def seed_all(seed: int = SEED):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_np_default) + "\n")


def _np_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_np_default))


def load_json(path):
    return json.loads(Path(path).read_text())


def revisions():
    p = DATA / "local_source_revisions.json"
    return load_json(p)["revisions"] if p.exists() else {}


def env_info(extra: dict | None = None):
    import torch
    info = dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import transformers, huggingface_hub
        info.update(transformers=transformers.__version__, huggingface_hub=huggingface_hub.__version__)
    except Exception:
        pass
    if torch.cuda.is_available():
        d = torch.cuda.current_device()
        info.update(gpu=torch.cuda.get_device_name(d), gpu_count=torch.cuda.device_count(),
                    gpu_capability=torch.cuda.get_device_capability(d))
    info["source_revisions"] = revisions()
    if extra: info.update(extra)
    return info


def cuda_time(fn, reps=5, warmup=2):
    """Time fn() with CUDA sync; returns list of ms."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def summarize_ms(ts):
    a = np.asarray(ts, dtype=float)
    return dict(mean_ms=float(a.mean()), std_ms=float(a.std(ddof=1) if len(a) > 1 else 0.0),
                median_ms=float(np.median(a)), min_ms=float(a.min()), reps=int(len(a)))


@contextlib.contextmanager
def peak_mem():
    import torch
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    box = {}
    yield box
    torch.cuda.synchronize()
    box["peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    box["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20


class Timer:
    def __init__(self): self.t = {}
    @contextlib.contextmanager
    def __call__(self, name, sync=True):
        import torch
        if sync and torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if sync and torch.cuda.is_available(): torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0


def paired_bootstrap(a, b, n=2000, seed=SEED, stat=np.mean):
    """95% CI of stat(b) - stat(a) for paired per-example arrays."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed); n_ex = len(a)
    idx = rng.integers(0, n_ex, size=(n, n_ex))
    d = stat(b[idx], axis=1) - stat(a[idx], axis=1) if stat is np.mean else \
        np.array([stat(b[i]) - stat(a[i]) for i in idx])
    return dict(diff=float(stat(b) - stat(a)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), n_resamples=n, n_examples=n_ex)


def paired_bootstrap_ppl(nll_a, cnt_a, nll_b, cnt_b, n=2000, seed=SEED):
    """Resample whole windows, pool token NLL, exponentiate; CI for PPL_b - PPL_a."""
    nll_a, nll_b, cnt = map(lambda x: np.asarray(x, float), (nll_a, nll_b, cnt_a))
    assert np.array_equal(np.asarray(cnt_a), np.asarray(cnt_b))
    rng = np.random.default_rng(seed); m = len(cnt)
    idx = rng.integers(0, m, size=(n, m))
    pa = np.exp(nll_a[idx].sum(1) / cnt[idx].sum(1)); pb = np.exp(nll_b[idx].sum(1) / cnt[idx].sum(1))
    d = pb - pa
    full = np.exp(nll_b.sum() / cnt.sum()) - np.exp(nll_a.sum() / cnt.sum())
    return dict(diff=float(full), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                n_resamples=n, n_windows=m)
