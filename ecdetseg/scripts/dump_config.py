#!/usr/bin/env python3
"""
Read-only merged-config dumper for EdgeCrafter/ecdetseg.

Reproduces exactly what `YAMLConfig.__init__` would see *before* building any
module: `load_config` (+ `-u` overrides) followed by `reset_cfg`'s resolution
rewrite. It is intentionally TORCH-FREE and NEVER touches the dataset or XPU,
so it is safe to run while a training job is live.

Usage:
    python scripts/dump_config.py -c configs/ecdet/ecdet_s.yml
    python scripts/dump_config.py -c configs/ecdet/ecdet_s_p30u.yml -u eval_spatial_size='[960,960]'
    python scripts/dump_config.py -c configs/ecdet/ecdet_s_p30u.yml -o configs_dump/E1.yml
"""
import argparse
import importlib.util
import os
import sys
import types

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))          # ecdetseg/
CORE = os.path.join(PROJECT, "engine", "core")


def _load_yaml_utils_torch_free():
    """Import engine/core/yaml_utils.py without importing the heavy engine package."""
    pkg = types.ModuleType("_ec_audit_core")
    pkg.__path__ = [CORE]
    sys.modules["_ec_audit_core"] = pkg

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"_ec_audit_core.{name}", os.path.join(CORE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_ec_audit_core.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("workspace")          # yaml_utils does `from .workspace import GLOBAL_CONFIG`
    return _load("yaml_utils")


def _reset_cfg(cfg):
    """Mirror YAMLConfig.reset_cfg (engine/core/yaml_config.py:180-211) without imports."""
    input_size = cfg["eval_spatial_size"][0]

    def glom(data, path):
        for key in path.split("."):
            data = data[key]
        return data

    for path in ("train_dataloader.dataset.transforms.ops",
                 "val_dataloader.dataset.transforms.ops"):
        for op in glom(cfg, path):
            t = op.get("type")
            if t == "Mosaic":
                op["output_size"] = input_size // 2
            elif t == "Resize":
                op["size"] = (input_size, input_size)

    stop = glom(cfg, "train_dataloader.dataset.transforms.stop_epoch")
    epochs = cfg["epochs"]
    if not 0 <= epochs - stop <= 5:
        cfg["train_dataloader"]["dataset"]["transforms"]["stop_epoch"] = epochs - 2
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-u", "--update", nargs="+", default=[])
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    yu = _load_yaml_utils_torch_free()
    # NOTE: pass a fresh {} explicitly — load_config's default `cfg=dict()` is a
    # shared mutable default and would leak state across multiple loads.
    cfg = yu.load_config(os.path.abspath(args.config), cfg={})
    if args.update:
        cfg = yu.merge_dict(cfg, yu.parse_cli(args.update))
    cfg = _reset_cfg(cfg)

    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[dump_config] wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
