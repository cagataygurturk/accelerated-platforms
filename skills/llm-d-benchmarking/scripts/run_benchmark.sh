#!/bin/bash
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

# Shared script to run llm-d benchmark
# Should be run from the repository root.
# Assumes llmdbenchmark CLI and other tools are available.

set -e
# Parse CLI inputs
WORKLOAD=$1
ENDPOINT_URL=$2
NAMESPACE=${3:-"default"}
MODEL_NAME=${4:-"google/gemma-4-31b-it"}
# Resolve hardware and deployment specs
ACCELERATOR_TYPE=${ACCELERATOR_TYPE:-"nvidia-h100"}
SPEC=${SPEC:-${DEPLOYMENT_STRATEGY:-"optimized-baseline"}}
MODEL_SERVER=${MODEL_SERVER:-"vllm"}
REPO_DIR="${ACP_REPO_DIR:-$(pwd)}"
# Resolve GCP Region dynamically
REGION=${REGION:-${TF_VAR_cluster_region:-$TF_VAR_platform_default_region}}
[ -z "$REGION" ] && [ -n "$ZONE" ] && REGION=$(echo "$ZONE" | sed -E 's/-[a-z0-9]+$//')
export REGION LLMDBENCH_BASE_DIR="${LLMDBENCH_BASE_DIR:-$(cd "${REPO_DIR}/.." && pwd)/llm-d-benchmark}"
# Determine GCS bucket for results
RESULTS_BUCKET=${5:-$RESULTS_BUCKET}
if [ -z "$RESULTS_BUCKET" ]; then
    TFVARS_FILE="${REPO_DIR}/platforms/gke/base/_shared_config/platform.auto.tfvars"
    PLATFORM_NAME=$(grep -oP '(?<=^platform_name = ")[^"]*' "$TFVARS_FILE" 2>/dev/null || echo "llm-d-bench")
    RESULTS_BUCKET="inf-${PLATFORM_NAME:-llm-d-bench}-bench-results"
fi
export RESULTS_BUCKET="${RESULTS_BUCKET#gs://}"

# Enforce required args and provision bucket if missing
[ -z "$WORKLOAD" ] || [ -z "$ENDPOINT_URL" ] && { echo "Usage: $0 <workload> <url> [ns] [model] [bucket]"; exit 1; }
gcloud storage buckets describe "gs://${RESULTS_BUCKET}" &>/dev/null || gcloud storage buckets create "gs://${RESULTS_BUCKET}" ${REGION:+--location=$REGION} || true

# Function to validate and propose optimal sizing configs
validate_workload_config() {
  local f="$1"
  [ ! -f "$f" ] && f=$(find . -path "./workspaces" -prune -o -name "$1" -print -quit 2>/dev/null)
  [ ! -f "$f" ] && return 0
  set +e; python3 "${REPO_DIR}/skills/llm-d-workload-tuner/scripts/tune_workload.py" --perf-yaml "$f" --accelerator-type "$ACCELERATOR_TYPE" --spec "$SPEC" --model "$MODEL_NAME"; code=$?; set -e
  if [ $code -eq 2 ]; then
    read -p "Gaps detected. Apply updates automatically? (y/N): " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] && python3 "${REPO_DIR}/skills/llm-d-workload-tuner/scripts/tune_workload.py" --perf-yaml "$f" --accelerator-type "$ACCELERATOR_TYPE" --spec "$SPEC" --model "$MODEL_NAME" --apply && exit 0
  fi
}

# Pre-flight Checks
[ -n "$CLUSTER_NAME" ] && [ -n "$ZONE" ] && { gcloud container clusters describe "$CLUSTER_NAME" --zone "$ZONE" --format="value(managedPrometheusConfig.enabled)" 2>/dev/null | grep -q "true" && echo "Managed Prometheus is enabled." || echo "Warning: Managed Prometheus may not be enabled."; }
validate_workload_config "$WORKLOAD"

# Skill Mock Evaluation Mode fallback
if [ -n "$MOCK_LOG_FILE" ]; then
    curl -s "${ENDPOINT_URL}/v1/models" >/dev/null || true
    llmdbenchmark run --base-dir "$LLMDBENCH_BASE_DIR" --spec "guides/${SPEC}" --workload "$WORKLOAD" --endpoint-url "$ENDPOINT_URL" --namespace "$NAMESPACE" --harness inference-perf --output results.json
    echo '{"mock_key": "mock_value", "results": {"request_performance": {"aggregate": {"requests": {"input_length": {"mean": 10}, "output_length": {"mean": 20}}}}}}' > report_v0.2.json
    echo '{"mock_vllm_args": ["--model", "google/gemma-4-31b-it"]}' > ./vllm_config.json
    echo '{}' > ./dcgm_metrics.json
    echo -e "treatment,source_file,ttft_mean_s\ndefault,report_v0.2.json,0.5" > output.csv
    gcloud storage cp report_v0.2.json vllm_config.json dcgm_metrics.json output.csv gs://${RESULTS_BUCKET}/
    exit 0
fi

# Configure Runtime Workspace
WORKSPACE_DIR="workspaces/run-$(date +%Y%m%d-%H%M%S)"
WORKLOAD_ARG="--workload $WORKLOAD"
[ -n "$RESULTS_BUCKET" ] && WORKLOAD_ARG="$WORKLOAD_ARG --overrides storage.google_cloud_storage.bucket=$RESULTS_BUCKET"

# Function to capture GPU DCGM metrics
collect_dcgm() {
  sed -e "s/TARGET_NAMESPACE_PLACEHOLDER/$1/g" -e "s/START_TIME_PLACEHOLDER/$2/g" -e "s/END_TIME_PLACEHOLDER/$3/g" skills/llm-d-benchmarking/scripts/helper-pods/telemetry-collector.yaml | kubectl apply -f - -n "$1"
  kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/telemetry-collector -n "$1" --timeout=120s || true
  kubectl delete pod telemetry-collector -n "$1" --grace-period=0 --force
}

# Phase 1: Setup Namespace & PVC
llmdbenchmark run --base-dir "$LLMDBENCH_BASE_DIR" --spec "guides/${SPEC}" $WORKLOAD_ARG --model "$MODEL_NAME" --endpoint-url "$ENDPOINT_URL" --namespace "$NAMESPACE" --workspace "$WORKSPACE_DIR" --dry-run
llmdbenchmark run --base-dir "$LLMDBENCH_BASE_DIR" --spec "guides/${SPEC}" $WORKLOAD_ARG --model "$MODEL_NAME" --endpoint-url "$ENDPOINT_URL" --namespace "$NAMESPACE" --workspace "$WORKSPACE_DIR" -s 0-6
kubectl delete pod access-to-harness-data-workload-pvc -n "$NAMESPACE" --ignore-not-found --grace-period=0 --force

# Phase 2: Execute Benchmark Harness
BENCHMARK_START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e; llmdbenchmark run --base-dir "$LLMDBENCH_BASE_DIR" --spec "guides/${SPEC}" $WORKLOAD_ARG --model "$MODEL_NAME" --endpoint-url "$ENDPOINT_URL" --namespace "$NAMESPACE" --harness inference-perf --workspace "$WORKSPACE_DIR" -s 7; set -e
BENCHMARK_END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
collect_dcgm "$NAMESPACE" "$BENCHMARK_START_TIME" "$BENCHMARK_END_TIME"

# Phase 3: Result Retrieval & Cleanup
kubectl apply -n "$NAMESPACE" -f skills/llm-d-benchmarking/scripts/helper-pods/data-access.yaml
kubectl wait --for=condition=Ready pod/access-to-harness-data-workload-pvc -n "$NAMESPACE" --timeout=120s
llmdbenchmark run --base-dir "$LLMDBENCH_BASE_DIR" --spec "guides/${SPEC}" -s 8-11 --model "$MODEL_NAME" --endpoint-url "$ENDPOINT_URL" --namespace "$NAMESPACE" --harness inference-perf --workspace "$WORKSPACE_DIR"

# Phase 4: Report Generation & Archival
RESULTS_DIR=$(ls -td "$WORKSPACE_DIR"/*/results/inference-perf-* 2>/dev/null | head -n 1 || true)
[ -z "$RESULTS_DIR" ] && exit 1
cp "$RESULTS_DIR/summary_lifecycle_metrics.json" ./results.json
python3 -c "from llmdbenchmark.analysis.benchmark_report.native_to_br0_2 import import_inference_perf; import_inference_perf('./results.json').export_json('./report_v0.2.json')"
python3 "${REPO_DIR}/skills/llm-d-benchmarking/scripts/extract_csv.py" --input report_v0.2.json --output output.csv
kubectl get $(kubectl get deployment -n "$NAMESPACE" -l app=vllm -o name | head -n 1) -n "$NAMESPACE" -o json > ./vllm_config.json 2>/dev/null || echo '{"error": "not found"}' > ./vllm_config.json
cp report_v0.2.json output.csv vllm_config.json "$RESULTS_DIR/"
gcloud storage cp report_v0.2.json vllm_config.json output.csv gs://${RESULTS_BUCKET}/
[ -f "$RESULTS_DIR/dcgm_metrics.json" ] && cp "$RESULTS_DIR/dcgm_metrics.json" ./dcgm_metrics.json && gcloud storage cp dcgm_metrics.json gs://${RESULTS_BUCKET}/
kubectl delete pod access-to-harness-data-workload-pvc -n "$NAMESPACE" --ignore-not-found --grace-period=0 --force
