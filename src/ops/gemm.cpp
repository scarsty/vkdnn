#include "../core/compute_internal.h"
#include "vkdnn/vkblas.h"
#include <algorithm>
#include <new>

struct vkblasContext
{
    int device;
    vkblasPointerMode_t mode = VKBLAS_POINTER_MODE_HOST;
};

namespace
{
vkblasStatus_t status(vkdnnError_t error)
{
    switch (error)
    {
    case vkdnnSuccess: return VKBLAS_STATUS_SUCCESS;
    case vkdnnErrorMemoryAllocation: return VKBLAS_STATUS_ALLOC_FAILED;
    case vkdnnErrorInvalidValue:
    case vkdnnErrorInvalidDevice: return VKBLAS_STATUS_INVALID_VALUE;
    case vkdnnErrorNoDevice: return VKBLAS_STATUS_NOT_INITIALIZED;
    case vkdnnErrorNotSupported: return VKBLAS_STATUS_NOT_SUPPORTED;
    default: return VKBLAS_STATUS_EXECUTION_FAILED;
    }
}
bool operation(vkblasOperation_t op) { return op == VKBLAS_OP_N || op == VKBLAS_OP_T || op == VKBLAS_OP_C; }
uint64_t extent(int rows, int columns, int ld)
{
    return (uint64_t(columns - 1) * uint64_t(ld) + uint64_t(rows)) * sizeof(float);
}
}    //namespace

vkblasStatus_t vkblasCreate(vkblasHandle_t* handle)
{
    if (!handle) { return VKBLAS_STATUS_INVALID_VALUE; }
    int device;
    auto result = vkdnnGetDevice(&device);
    if (result != vkdnnSuccess) { return status(result); }
    result = vkdnnDeviceSynchronize();
    if (result != vkdnnSuccess) { return status(result); }
    auto context = new (std::nothrow) vkblasContext{ device };
    if (!context) { return VKBLAS_STATUS_ALLOC_FAILED; }
    *handle = context;
    return VKBLAS_STATUS_SUCCESS;
}

vkblasStatus_t vkblasDestroy(vkblasHandle_t handle)
{
    if (!handle) { return VKBLAS_STATUS_NOT_INITIALIZED; }
    delete handle;
    return VKBLAS_STATUS_SUCCESS;
}

vkblasStatus_t vkblasSetPointerMode(vkblasHandle_t handle, vkblasPointerMode_t mode)
{
    if (!handle) { return VKBLAS_STATUS_NOT_INITIALIZED; }
    if (mode == VKBLAS_POINTER_MODE_DEVICE) { return VKBLAS_STATUS_NOT_SUPPORTED; }
    if (mode != VKBLAS_POINTER_MODE_HOST) { return VKBLAS_STATUS_INVALID_VALUE; }
    handle->mode = mode;
    return VKBLAS_STATUS_SUCCESS;
}

vkblasStatus_t vkblasSscal(vkblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    if (!handle) { return VKBLAS_STATUS_NOT_INITIALIZED; }
    int device;
    auto result = vkdnnGetDevice(&device);
    if (result != vkdnnSuccess) { return status(result); }
    if (device != handle->device) { return VKBLAS_STATUS_INVALID_VALUE; }
    if (n <= 0 || incx <= 0) { return VKBLAS_STATUS_SUCCESS; }
    GemmParams p{};
    result = runtimeHostScalar(alpha, p.alpha);
    if (result != vkdnnSuccess) { return status(result); }
    uint64_t elements = uint64_t(n - 1) * uint64_t(incx) + 1;
    if (elements > UINT32_MAX) { return VKBLAS_STATUS_NOT_SUPPORTED; }
    // Reuse the GEMM registry, descriptors and dispatch without input buffers.
    p.transa = 2;
    p.transb = uint32_t(n);
    p.lda = uint32_t(incx);
    p.m = uint32_t(std::min(n, 65520));
    p.n = (uint32_t(n) + p.m - 1) / p.m;
    return status(runtimeGemm(device, nullptr, 0, nullptr, 0, x, size_t(elements * sizeof(float)), p));
}

vkblasStatus_t vkblasSgemm(vkblasHandle_t handle, vkblasOperation_t transa, vkblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc)
{
    if (!handle) { return VKBLAS_STATUS_NOT_INITIALIZED; }
    if (!operation(transa) || !operation(transb) || m < 0 || n < 0 || k < 0
        || lda < std::max(1, transa == VKBLAS_OP_N ? m : k)
        || ldb < std::max(1, transb == VKBLAS_OP_N ? k : n)
        || ldc < std::max(1, m)) { return VKBLAS_STATUS_INVALID_VALUE; }
    int device;
    auto result = vkdnnGetDevice(&device);
    if (result != vkdnnSuccess) { return status(result); }
    if (device != handle->device) { return VKBLAS_STATUS_INVALID_VALUE; }
    if (!m || !n) { return VKBLAS_STATUS_SUCCESS; }
    GemmParams p{};
    result = runtimeHostScalar(beta, p.beta);
    if (result != vkdnnSuccess) { return status(result); }
    if (k)
    {
        result = runtimeHostScalar(alpha, p.alpha);
        if (result != vkdnnSuccess) { return status(result); }
    }
    p.m = m;
    p.n = n;
    p.k = k;
    p.lda = lda;
    p.ldb = ldb;
    p.ldc = ldc;
    p.transa = transa != VKBLAS_OP_N;
    p.transb = transb != VKBLAS_OP_N;
    uint64_t aBytes = 0, bBytes = 0;
    if (k && p.alpha != 0)
    {
        aBytes = extent(transa == VKBLAS_OP_N ? m : k, transa == VKBLAS_OP_N ? k : m, lda);
        bBytes = extent(transb == VKBLAS_OP_N ? k : n, transb == VKBLAS_OP_N ? n : k, ldb);
    }
    else
    {
        p.k = 0;
    }
    uint64_t cBytes = extent(m, n, ldc);
    if (aBytes / 4 > UINT32_MAX || bBytes / 4 > UINT32_MAX || cBytes / 4 > UINT32_MAX) { return VKBLAS_STATUS_NOT_SUPPORTED; }
    return status(runtimeGemm(device, A, size_t(aBytes), B, size_t(bBytes), C, size_t(cBytes), p));
}
