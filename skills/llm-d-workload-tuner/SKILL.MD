---
name: llm-d-workload-tuner
description: This is an experimental Skill. It automatically tunes GKE vLLM inference server parameters and resources based on workload profiles specified in the benchmark configs.
version: 0.1.0
allowed-tools: python3 kubectl yaml json
mcp-servers: []
---

# LLM-D Workload Tuner Skill

Follow these instructions to run the workload optimizer and optionally apply the tuned vLLM parameters and GKE node resource requirements.

## 1. Prerequisites

- Review the target workload characteristics and configurations in the reference guide: [llm-d-workload-profiles.md](references/llm-d-workload-profiles.md).
- Ensure you know the path to your llm-d-benchmark directory. The benchmarking files `config.json` (defining sequence lengths) and `inference-perf.yaml` (defining stages and model servers) come from the specific llm-d-benchmark workload profile chosen and must exist under the target benchmarking directories.
- Know the target accelerator type (e.g., `rtx-pro-6000`, `nvidia-h100`, or `v6e` TPU) and the deployment strategy (e.g., `precise-prefix-cache-routing`, `optimized-baseline`, or `predicted-latency-routing`). You must explicitly supply the target strategy folder name using the `--strategy` flag.

## 2. Running the Tuner

To compute optimal sizing (Tensor Parallelism size, maximum model length limits, memory margins, and chunked prefill). Note that the `--strategy` parameter is required:

- **Command Format:**

  ```bash
  python3 "${ACP_REPO_DIR}/skills/llm-d-workload-tuner/scripts/tune_workload.py" \
      --config <path_to_config.json> \
      --perf-yaml <path_to_inference-perf.yaml> \
      --accelerator-type <accelerator_name> \
      --strategy <strategy_directory_name>
  ```

- **Example (Dry Run):**
  ```bash
  python3 "${ACP_REPO_DIR}/skills/llm-d-workload-tuner/scripts/tune_workload.py" \
      --perf-yaml llm-d-benchmark/workload/profiles/inference-perf/chatbot_synthetic.yaml.in \
      --accelerator-type rtx-pro-6000 \
      --strategy precise-prefix-cache-routing
  ```

## 3. Applying the Tuned Configuration

Confirm the GKE cluster name, region and reservation (if ANY) and then use the `--apply` flag to commit the calculated tuning configs directly to the GKE deployment overlays.

- **Command:**
  ```bash
  python3 "${ACP_REPO_DIR}/skills/llm-d-workload-tuner/scripts/tune_workload.py" \
      --perf-yaml llm-d-benchmark/workload/profiles/inference-perf/chatbot_synthetic.yaml.in \
      --accelerator-type rtx-pro-6000 \
      --strategy precise-prefix-cache-routing \
      --apply
  ```

When `--apply` is set, the tuner:

1. Updates `runtime.env` inside the GKE overlay directory (e.g., setting `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and `EXTRA_ARGS`).
2. Patches `patch-nodeselector.yaml` to request matching GPU / TPU counts on nodes.
3. Patches `patch-resources.yaml` to configure container GPU / TPU limit settings.
4. **Verifies and logs the parameter diff:** Always review the printed `=== Parameter Tuning Diff ===` output to see exactly which parameters were tuned from the baseline deployment in this repo.

## 4. Verification

After deploying the tuned stack, verify:

- That the baseline diff output matches the expected transitions.
- That vLLM deployment specs match the calculated values:
  ```bash
  kubectl get deployment -n $NAMESPACE -l app=vllm -o jsonpath='{.items[0].spec.template.spec.containers[0].args}'
  ```
- Check if the user want to execute benchmark then call llm-d-benchmarking skill and pass the workload profile and endpoint url to it:
  ```bash
  .${ACP_REPO_DIR}/skills/llm-d-benchmarking/scripts/run_benchmark.sh" <workload_profile_name> <endpoint_url> [namespace] [model_name]
  ```
- That GCS benchmarking uploads automatically bundle the metadata:
  - When running `run_benchmark.sh`, a file named `vllm_config.json` is generated on the PVC and uploaded directly to the GCS bucket alongside the performance metrics.
