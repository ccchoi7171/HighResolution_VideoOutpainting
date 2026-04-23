from __future__ import annotations

from wancanvas.backbones.runtime_env import inspect_diffusers_runtime
from wancanvas.utils.logging import dump_json


if __name__ == "__main__":
    print(dump_json(inspect_diffusers_runtime().to_dict()))
