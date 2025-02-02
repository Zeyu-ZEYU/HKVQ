#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <iostream>

using Element = cutlass::half_t;
using Accumulator = float;

struct FlashAttentionEpilogue
{

    template <typename FragmentC, typename FragmentV, typename FragmentOutput>
    CUTLASS_DEVICE void operator()(
        FragmentC &frag_c, // QK^T的部分tile结果
        FragmentV &frag_v, // 对应的V的tile数据
        FragmentOutput &frag_out)
    {
        Accumulator max_val = -CUTLASS_INFINITY;
        for (int i = 0; i < frag_c.size(); ++i)
        {
            max_val = max(max_val, Accumulator(frag_c[i]));
        }

        Accumulator sum_exp = 0.f;
        for (int i = 0; i < frag_c.size(); ++i)
        {
            Accumulator val = exp(Accumulator(frag_c[i]) - max_val);
            sum_exp += val;
            frag_c[i] = Element(val);
        }

        Accumulator inv_sum = Accumulator(1) / sum_exp;
        for (int i = 0; i < frag_c.size(); ++i)
        {
            frag_c[i] = Element(Accumulator(frag_c[i]) * inv_sum);
        }

        for (int i = 0; i < frag_c.size(); ++i)
        {

            frag_out[i] = Element(Accumulator(frag_c[i]) * Accumulator(frag_v[i]));
        }
    }
};

__global__ void flash_attention_kernel(
    Element *Q,   // [M, K]
    Element *K,   // [N, K]
    Element *V,   // [N, K]
    Element *Out, // [M, K]
    int M, int N, int K_dim)
{

    FlashAttentionEpilogue epilogue_op;

    const int FragSize = 8;
    Element frag_c[FragSize];
    Element frag_v[FragSize];
    Element frag_out[FragSize];

    for (int i = 0; i < FragSize; ++i)
    {
        frag_c[i] = Element(0.1f * i);
        frag_v[i] = Element(0.05f * i);
        frag_out[i] = Element(0);
    }

    epilogue_op(frag_c, frag_v, frag_out);

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < M * K_dim && thread_id < FragSize)
    {
        Out[thread_id] = frag_out[thread_id % FragSize];
    }
}

int main()
{

    int M = 128;    // sequence length
    int N = 128;    // sequence length
    int K_dim = 64; // head dimension

    std::vector<Element> Q(M * K_dim), K(N * K_dim), V(N * K_dim), Out(M * K_dim, Element(0));

    for (int i = 0; i < M * K_dim; i++)
    {
        Q[i] = Element(float(rand()) / RAND_MAX);
    }
    for (int i = 0; i < N * K_dim; i++)
    {
        K[i] = Element(float(rand()) / RAND_MAX);
        V[i] = Element(float(rand()) / RAND_MAX);
    }

    Element *dQ, *dK, *dV, *dOut;
    cudaMalloc((void **)&dQ, M * K_dim * sizeof(Element));
    cudaMalloc((void **)&dK, N * K_dim * sizeof(Element));
    cudaMalloc((void **)&dV, N * K_dim * sizeof(Element));
    cudaMalloc((void **)&dOut, M * K_dim * sizeof(Element));

    cudaMemcpy(dQ, Q.data(), M * K_dim * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), N * K_dim * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), N * K_dim * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, M * K_dim * sizeof(Element));

    dim3 grid(1);
    dim3 block(128);
    flash_attention_kernel<<<grid, block>>>(dQ, dK, dV, dOut, M, N, K_dim);

    cudaMemcpy(Out.data(), dOut, M * K_dim * sizeof(Element), cudaMemcpyDeviceToHost);

    std::cout << "Out[0..9]: ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << float(Out[i]) << " ";
    }
    std::cout << "\n";

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOut);

    return 0;
}
