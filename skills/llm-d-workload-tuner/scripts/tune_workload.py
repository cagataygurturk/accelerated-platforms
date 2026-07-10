#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import sys

import yaml

AC = {"h100": 80.0, "h200": 141.0, "rtx-pro-6000": 96.0, "v6e": 32.0}
DEFAULT_ACCEL_UTIL, MEM_MULT, BYTES_BF16 = 0.95, 1.2, 2.0
MODELS = {"google/gemma-4-31b-it": (32.0, 48, 8, 256, "gemma-4-31b-it")}


def ld_env(p):
    return (
        dict(l.strip().split("=", 1) for l in open(p) if "=" in l)
        if os.path.exists(p)
        else {}
    )


def sv_env(p, d):
    open(p, "w").writelines(f"{k}={v}\n" for k, v in d.items())


def patch_yaml(p, ks, v):
    if not os.path.exists(p):
        return
    with open(p) as f:
        d = yaml.safe_load(f)
    c = d
    for i, k in enumerate(ks[:-1]):
        if isinstance(c, list):
            c = c[k]
        elif isinstance(c, dict):
            c = c.setdefault(k, {} if isinstance(ks[i + 1], str) else [])
    c[ks[-1]] = v
    with open(p, "w") as f:
        yaml.dump(d, f, default_flow_style=False)


def get_n(d, ks, df="unknown"):
    for k in ks:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return df
    return d


def main():
    p = argparse.ArgumentParser()
    for a, k in [
        ("--config", {}),
        ("--perf-yaml", {"required": True}),
        ("--accelerator-type", {"required": True, "choices": list(AC.keys())}),
        ("--spec", {"required": True}),
        ("--model", {"default": "google/gemma-4-31b-it"}),
        ("--apply", {"action": "store_true"}),
        ("--chunked-prefill-threshold", {"type": int, "default": 8000}),
    ]:
        p.add_argument(a, **k)
    args = p.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    specs_f = f"{repo}/skills/llm-d-workload-tuner/references/model_specs.json"
    if os.path.exists(specs_f):
        MODELS.update(
            {
                k: tuple(v)
                for k, v in json.load(open(specs_f)).items()
                if k != "_comment"
            }
        )

    if args.config and not os.path.exists(args.config):
        sys.exit(f"Err: {args.config}")
    if not os.path.exists(args.perf_yaml):
        sys.exit(f"Err: {args.perf_yaml}")

    perf = yaml.safe_load(open(args.perf_yaml))
    max_out = (
        json.load(open(args.config)).get("output_sequence_length", {}).get("max", 2048)
        if args.config
        else (
            get_n(perf, ["data", "output_distribution", "max"], None)
            or get_n(
                perf,
                ["data", "conversation_replay", "output_tokens_per_turn", "max"],
                None,
            )
            or 2048
        )
    )
    max_c = max(
        (
            s.get("concurrency_level", 1)
            for s in perf.get("stages") or perf.get("load", {}).get("stages", [])
        ),
        default=1,
    )

    params, layers, kv, hd, sfx = MODELS.get(
        args.model, MODELS["google/gemma-4-31b-it"]
    )
    accel = args.accelerator_type.replace("nvidia-", "")
    vram = AC.get(accel)

    pfx = "tpu" if accel == "v6e" else "gpu"
    ovl = f"platforms/gke/base/use-cases/inference-ref-arch/kubernetes-manifests/online-inference-{pfx}/llmd-{args.spec}/vllm/{accel}-{sfx}"

    if not os.path.isdir(ovl):
        sys.exit(f"Warn: {ovl} not found.")

    env_f = f"{ovl}/runtime.env"
    env = ld_env(env_f)
    accel_util = float(env.get("GPU_MEMORY_UTILIZATION", DEFAULT_ACCEL_UTIL))
    w_size = params * BYTES_BF16 * MEM_MULT
    bpt = 4 * layers * kv * hd
    c_size = (max_c * max_out * bpt) / (1024**3)

    tp, tr = 1, w_size + c_size
    while tp * vram * accel_util < tr:
        if tp < 8:
            tp *= 2
        else:
            sys.exit("Err: Exceeds capacity with TP 8.")

    avail_c = (tp * vram * accel_util) - w_size
    max_len = max(
        min(int((avail_c * (1024**3)) / (max_c * bpt)), 32768), max_out + 1024
    )
    extra = f'--enable-chunked-prefill={"True" if max_out > args.chunked_prefill_threshold else "False"}'

    res_f, node_f = f"{ovl}/patch-resources.yaml", f"{ovl}/patch-nodeselector.yaml"
    res_d = yaml.safe_load(open(res_f)) if os.path.exists(res_f) else {}
    node_d = yaml.safe_load(open(node_f)) if os.path.exists(node_f) else {}
    l_k = "nvidia.com/gpu" if pfx == "gpu" else "google.com/tpu"
    n_k = (
        "cloud.google.com/gke-gpu-count"
        if pfx == "gpu"
        else "cloud.google.com/compute-class"
    )

    old_res = get_n(
        res_d, ["spec", "template", "spec", "containers", 0, "resources", "limits", l_k]
    )
    old_node = get_n(node_d, ["spec", "template", "spec", "nodeSelector", n_k])
    tpu_c = "tpu-v6e-2x2" if tp == 4 else ("tpu-v6e-2x4" if tp == 8 else None)
    s_val = str(tp) if pfx == "gpu" else tpu_c

    gaps = [
        ("TENSOR_PARALLEL_SIZE", env.get("TENSOR_PARALLEL_SIZE", "unknown"), tp),
        ("MAX_MODEL_LEN", env.get("MAX_MODEL_LEN", "unknown"), max_len),
        (
            "GPU_MEMORY_UTILIZATION",
            env.get("GPU_MEMORY_UTILIZATION", "unknown"),
            accel_util,
        ),
        ("QUANTIZATION", env.get("QUANTIZATION", "unknown"), "null"),
        ("EXTRA_ARGS", env.get("EXTRA_ARGS", "unknown").replace('"', ""), extra),
        ("resources limits", old_res, tp),
        ("nodeSelector", old_node, s_val),
    ]

    print(
        f"\nProfile: {args.perf_yaml}\nSpec: {args.spec}\nModel: {args.model}\nConcurrency: {max_c}\nOutput: {max_out}\nWeights: {w_size:.2f}GB\nCache: {c_size:.2f}GB"
    )
    for n, c, r in gaps:
        print(f"{n}: curr={c}, req={r}")

    needs_up = any(str(c) != str(r) for _, c, r in gaps)
    if needs_up and args.apply:
        env.update(
            {
                "TENSOR_PARALLEL_SIZE": str(tp),
                "MAX_MODEL_LEN": str(max_len),
                "GPU_MEMORY_UTILIZATION": str(accel_util),
                "QUANTIZATION": "null",
                "EXTRA_ARGS": f'"{extra}"',
            }
        )
        sv_env(env_f, env)
        patch_yaml(
            res_f,
            ["spec", "template", "spec", "containers", 0, "resources", "limits", l_k],
            tp,
        )
        if get_n(
            res_d,
            ["spec", "template", "spec", "containers", 0, "resources", "requests", l_k],
            None,
        ):
            patch_yaml(
                res_f,
                [
                    "spec",
                    "template",
                    "spec",
                    "containers",
                    0,
                    "resources",
                    "requests",
                    l_k,
                ],
                tp,
            )
        if s_val:
            patch_yaml(node_f, ["spec", "template", "spec", "nodeSelector", n_k], s_val)
        print("Updated manifests.")
    elif needs_up:
        sys.exit(2)


if __name__ == "__main__":
    main()
