#!/bin/bash

# Requirement: 8x H100 GPUs.


# Model: neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV 
# Query: 2048 input tokens, 11 output tokens, QPS 4, 500 requests
# Resource: 8x H100
# Approaches:
# 1. Chunked prefill: 1 vllm instance with tp=8
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pkill -f pt_main_thread
  pkill -f python3
  ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_chunked_prefill() {
  model="meta-llama/Meta-Llama-3.1-70B-Instruct"
  # disagg prefill
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
      -m vllm.entrypoints.openai.api_server \
      --model $model \
      --port 8100 \
      -tp 4 \
      --max-model-len 10000 \
      --disable-log-stats \
      --disable-log-requests \
      --enable-chunked-prefill \
      --gpu-memory-utilization 0.8 &
  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    -tp 4 \
    --max-model-len 10000 \
    --disable-log-stats \
    --disable-log-requests \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.8 &
  wait_for_server 8100
  wait_for_server 8200
  python3 round_robin_proxy.py &
  sleep 1
}


launch_disagg_prefill() {
  model="meta-llama/Meta-Llama-3.1-70B-Instruct" 
  # disagg prefill
  VLLM_PORT=12345 VLLM_DISTRIBUTED_KV_ROLE=producer CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
      -m vllm.entrypoints.openai.api_server \
      --model $model \
      --port 8100 \
      -tp 4 \
      --max-model-len 10000 \
      --disable-log-stats \
      --disable-log-requests \
      --gpu-memory-utilization 0.8 &
  VLLM_PORT=12345 VLLM_DISTRIBUTED_KV_ROLE=consumer CUDA_VISIBLE_DEVICES=4,5,6,7 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    -tp 4 \
    --max-model-len 10000 \
    --disable-log-stats \
    --disable-log-requests \
    --gpu-memory-utilization 0.8 &
  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="meta-llama/Meta-Llama-3.1-70B-Instruct"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=200
  qps=$1
  prefix_len=50
  input_len=1024
  output_len=$2
  tag=$3

  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len $output_len \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename $tag-qps-$qps.json \
          --request-rate $qps

  sleep 2

}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)

  pip install quart httpx matplotlib aiohttp

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_output_len=6

  export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  launch_chunked_prefill
  for qps in 2 4 6 8; do
  benchmark $qps $default_output_len chunked_prefill
  done
  kill_gpu_processes

  launch_disagg_prefill
  for qps in 2 4 6 8; do
  benchmark $qps $default_output_len disagg_prefill
  done
  kill_gpu_processes

  python3 visualize_benchmark_results.py

}


main "$@"
