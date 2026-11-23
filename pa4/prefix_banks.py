"""Run the course script `setup_data.py --prefix-banks` unchanged under the pinned transformers==4.57.1.

That transformers release rejects a `generator=` keyword in `generate()` (ValueError: model_kwargs not used).
This shim drops the keyword and seeds the global RNG from that generator's seed right before each call, so every
prompt is still sampled from its fixed seed (6304 + 10007*j, +123 for the original draft) and the bank is reproducible.

python -m pa4.prefix_banks [--force]
"""
import runpy, sys
import torch
from transformers.generation.utils import GenerationMixin
from .common import DATA_REPO

_generate = GenerationMixin.generate


def _generate_with_generator(self, *args, generator=None, **kwargs):
    if generator is not None:
        torch.manual_seed(generator.initial_seed())   # seeds CPU and every CUDA device
    return _generate(self, *args, **kwargs)


GenerationMixin.generate = _generate_with_generator

if __name__ == "__main__":
    script = str(DATA_REPO / "scripts" / "setup_data.py")
    sys.argv = [script, "--prefix-banks"] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
