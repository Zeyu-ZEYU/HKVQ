#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>

using Element = __half; // Q, K, V数据类型
using AccType = float;  // 累加与softmax使用FP32以提高数值精度

__device__ inline float warpReduceMax(float val)
{

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

__device__ inline float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val += other;
    }
    return val;
}

__global__ void flash_attention_decode_kernel(
    const Element *__restrict__ Q,
    const Element *__restrict__ K,
    const Element *__restrict__ V,
    Element *__restrict__ Out,
    int N, int K_dim)
{

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    __shared__ Element q_sh[64];
    if (lane_id < K_dim)
    {
        q_sh[lane_id] = Q[lane_id];
    }
    __syncthreads();

    int chunk_size = N;
    int start_idx = 0;
    int end_idx = start_idx + chunk_size;

    float score_sum = 0.0f;

    extern __shared__ float temp_storage[]; // 动态共享内存
    float *scores_sh = temp_storage;        // 大小至少N个float

    float thread_max = -INFINITY;
    for (int i = lane_id; i < N; i += 32)
    {
        float acc = 0.0f;
        for (int k = 0; k < K_dim; k++)
        {
            float q_val = __half2float(q_sh[k]);
            float k_val = __half2float(K[i * K_dim + k]);
            acc += q_val * k_val;
        }
        scores_sh[i] = acc;
        if (acc > thread_max)
            thread_max = acc;
    }

    float warp_max = warpReduceMax(thread_max);

    float thread_sum = 0.0f;
    for (int i = lane_id; i < N; i += 32)
    {
        float val = scores_sh[i];
        float exp_val = expf(val - warp_max);
        scores_sh[i] = exp_val;
        thread_sum += exp_val;
    }

    float warp_sum = warpReduceSum(thread_sum);
    float inv_sum = 1.0f / warp_sum;

    float out_acc[64]; // 存储最终输出维度的累加器
    for (int x = 0; x < K_dim; x++)
    {
        out_acc[x] = 0.0f;
    }

    for (int i = lane_id; i < N; i += 32)
    {
        float soft_val = scores_sh[i] * inv_sum;

        for (int k = 0; k < K_dim; k++)
        {
            float v_val = __half2float(V[i * K_dim + k]);
            out_acc[k] += soft_val * v_val;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        for (int k = 0; k < K_dim; k++)
        {
            float other = __shfl_xor_sync(0xffffffff, out_acc[k], offset);
            out_acc[k] += other;
        }
    }

    if (lane_id == 0)
    {
        for (int k = 0; k < K_dim; k++)
        {
            Out[k] = __float2half(out_acc[k]);
        }
    }
}

int main()
{
    int N = 1024;   // 序列长度
    int K_dim = 64; // head维度

    std::vector<__half> Q(K_dim), K(N * K_dim), V(N * K_dim), Out(K_dim);
    for (int i = 0; i < K_dim; i++)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        Q[i] = __float2half(val);
    }
    for (int i = 0; i < N * K_dim; i++)
    {
        K[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        V[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }

    __half *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ, K_dim * sizeof(__half));
    cudaMalloc(&dK, N * K_dim * sizeof(__half));
    cudaMalloc(&dV, N * K_dim * sizeof(__half));
    cudaMalloc(&dOut, K_dim * sizeof(__half));

    cudaMemcpy(dQ, Q.data(), K_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), N * K_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), N * K_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, K_dim * sizeof(__half));

    size_t smem_size = N * sizeof(float);
    flash_attention_decode_kernel<<<1, 32, smem_size>>>(dQ, dK, dV, dOut, N, K_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(Out.data(), dOut, K_dim * sizeof(__half), cudaMemcpyDeviceToHost);

    std::cout << "Out[0..9]: ";
    for (int i = 0; i < 10 && i < K_dim; i++)
    {
        std::cout << __half2float(Out[i]) << " ";
    }
    std::cout << "\n";

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOut);

    return 0;
}
