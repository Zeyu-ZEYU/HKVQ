#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/tensor_view_io.h"

// This example uses single-precision float. Change as needed.
using Element = float;

// ----------------------------
// Example GEMM configuration
// ----------------------------
// You can choose a predefined CUTLASS kernel or define your own tile sizes.
// For instance, let’s pick tile sizes:
using CutlassGemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor,    // Layout of A (e.g. Q or QK)
    Element, cutlass::layout::RowMajor,    // Layout of B (e.g. K^T or V)
    Element, cutlass::layout::RowMajor,    // Layout of C (output)
    Element,                               // Compute type (accumulator)
    cutlass::arch::OpClassTensorOp,        // Tensor cores (if available)
    cutlass::arch::Sm80,                   // Target GPU architecture (e.g., Ampere)
    cutlass::gemm::GemmShape<128, 64, 64>, // Threadblock tile size (M, N, K)
    cutlass::gemm::GemmShape<64, 32, 64>,  // Warp tile size
    cutlass::gemm::GemmShape<16, 8, 8>,    // Instruction tile size
    cutlass::epilogue::thread::LinearCombination<
        Element, 1, Element, Element>,                            // Epilogue functor (alpha * AB + beta * C)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
    2                                                             // Stages
    >;

// ----------------------------
// Helper to run a GEMM
// ----------------------------
void run_gemm(
    int M, int N, int K,
    Element const *A, Element const *B, Element *C,
    Element alpha = Element(1), Element beta = Element(0))
{

    CutlassGemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename CutlassGemm::Arguments args(
        problem_size,
        {A, K}, // A is MxK
        {B, N}, // B is KxN
        {C, N}, // C is MxN
        {C, N}, // D is output in C (alpha * A*B + beta * C)
        {alpha, beta});

    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "GEMM initialization failed.\n";
        return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "GEMM execution failed.\n";
        return;
    }
}

int main()
{
    // ----------------------------
    // Example Dimensions
    // ----------------------------
    // Let's say:
    // M = sequence length (e.g. 128)
    // N = sequence length (e.g. 128)
    // K = head dimension (e.g. 64)
    // For attention Q, K, V shapes: [Batch x M x K]
    // We do per-head GEMMs. For simplicity, assume batch=1, single-head.
    int M = 128;    // sequence length
    int N = 128;    // sequence length
    int K_dim = 64; // head dimension

    // 1) Compute QK^T: Q(MxK) * K^T(KxN) = (MxN)
    // 2) Compute (QK)*V: (MxN)*(NxK) = (MxK)

    // Allocate and initialize host tensors
    std::vector<Element> Q(M * K_dim), K(M * K_dim), V(M * K_dim);
    std::vector<Element> QK(M * N), Out(M * K_dim);

    for (int i = 0; i < M * K_dim; i++)
    {
        Q[i] = Element((float)rand() / RAND_MAX);
        K[i] = Element((float)rand() / RAND_MAX);
        V[i] = Element((float)rand() / RAND_MAX);
    }
    // Initialize outputs to zero
    std::fill(QK.begin(), QK.end(), Element(0));
    std::fill(Out.begin(), Out.end(), Element(0));

    // Allocate device memory
    Element *dQ, *dK, *dV, *dQK, *dOut;
    cudaMalloc((void **)&dQ, M * K_dim * sizeof(Element));
    cudaMalloc((void **)&dK, M * K_dim * sizeof(Element));
    cudaMalloc((void **)&dV, M * K_dim * sizeof(Element));
    cudaMalloc((void **)&dQK, M * N * sizeof(Element));
    cudaMalloc((void **)&dOut, M * K_dim * sizeof(Element));

    cudaMemcpy(dQ, Q.data(), M * K_dim * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), M * K_dim * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), M * K_dim * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemset(dQK, 0, M * N * sizeof(Element));
    cudaMemset(dOut, 0, M * K_dim * sizeof(Element));

    // ----------------------------
    // Compute QK^T
    // ----------------------------
    // Q: MxK
    // K: MxK but we need K^T which is KxM
    // We can either transpose K on-the-fly or store K transposed.
    // For simplicity, let's say K is actually stored as K^T (KxN).
    // If you need to transpose, you'd do that step beforehand.
    // Here, treat `K` as if we loaded it in column-major or transposed form.
    // For demonstration, assume K is row-major and we transpose by switching args:

    // Actually, to run QK^T, we can just interpret K as N x K and choose
    // appropriate layout. If we must stick to row-major, we need a pre-transpose.
    // Let's do a quick transpose on host for demonstration:
    std::vector<Element> K_T(M * K_dim);
    for (int r = 0; r < M; ++r)
    {
        for (int c = 0; c < K_dim; ++c)
        {
            K_T[c * M + r] = K[r * K_dim + c];
        }
    }
    cudaMemcpy(dK, K_T.data(), M * K_dim * sizeof(Element), cudaMemcpyHostToDevice);

    // Now run QK^T: (M,K) * (K,N) = (M,N)
    run_gemm(M, N, K_dim, dQ, dK, dQK);

    // ----------------------------
    // Apply softmax to QK result if needed
    // ----------------------------
    // Typically attention: A = softmax(QK^T) * V
    // Here we assume you do softmax outside of this code snippet.
    // After softmax: A(MxN) * V(NxK) = Out(MxK)

    // For demonstration, we skip softmax and just do A = QK * V.
    // run_gemm(M, K_dim, N, dQK, dV, dOut);

    // If you want to do the second GEMM:
    run_gemm(M, K_dim, N, dQK, dV, dOut);

    // Copy back result
    cudaMemcpy(Out.data(), dOut, M * K_dim * sizeof(Element), cudaMemcpyDeviceToHost);

    // Print a small portion of the output
    std::cout << "Out[0..9]: ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << Out[i] << " ";
    }
    std::cout << "\n";

    // Cleanup
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dQK);
    cudaFree(dOut);

    return 0;
}
