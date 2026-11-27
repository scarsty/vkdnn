#ifndef VKDNN_VKBLAS_H
#define VKDNN_VKBLAS_H
#include "vkdnn.h"
#ifdef __cplusplus
extern "C"
{
#endif
typedef struct vkblasContext* vkblasHandle_t;
typedef enum vkblasStatus_t
{
    VKBLAS_STATUS_SUCCESS = 0,
    VKBLAS_STATUS_NOT_INITIALIZED = 1,
    VKBLAS_STATUS_ALLOC_FAILED = 3,
    VKBLAS_STATUS_INVALID_VALUE = 7,
    VKBLAS_STATUS_EXECUTION_FAILED = 13,
    VKBLAS_STATUS_INTERNAL_ERROR = 14,
    VKBLAS_STATUS_NOT_SUPPORTED = 15
} vkblasStatus_t;
typedef enum vkblasOperation_t
{
    VKBLAS_OP_N = 0,
    VKBLAS_OP_T = 1,
    VKBLAS_OP_C = 2
} vkblasOperation_t;
typedef enum vkblasPointerMode_t
{
    VKBLAS_POINTER_MODE_HOST = 0,
    VKBLAS_POINTER_MODE_DEVICE = 1
} vkblasPointerMode_t;
/* cuBLAS-style column-major matrices; scalar pointer mode defaults to HOST.
 * FP32 Vulkan GEMM; completion follows vkdnnSetExecutionMode (asynchronous by default).
 * Real OP_C is OP_T. Leading dimensions include padding.
 * DEVICE pointer mode returns NOT_SUPPORTED and leaves HOST mode unchanged.
 * Handles bind the current device. Caller serializes handle access/destruction.
 * Matrices must be runtime allocations (aligned offsets allowed); overlapping
 * input/output ranges are NOT_SUPPORTED. A/B are unused for alpha=0 or k=0;
 * beta=0 does not read C. m=0 or n=0 needs no pointers (dimensions still checked).
 * k=0 does not read alpha. Nonempty C must always be valid, even for beta=1.
 * Destroyed handles must not be reused. No batched GEMM API. */
VKDNN_API vkblasStatus_t vkblasCreate(vkblasHandle_t* handle);
VKDNN_API vkblasStatus_t vkblasDestroy(vkblasHandle_t handle);
VKDNN_API vkblasStatus_t vkblasSetPointerMode(vkblasHandle_t handle, vkblasPointerMode_t mode);
/* In-place FP32 scaling: x[i*incx] *= *alpha (0 <= i < n).
 * Completion follows vkdnnSetExecutionMode.
 * n<=0 or incx<=0 is a successful no-op without accessing alpha/x, after
 * handle/current-device validation. Otherwise alpha is HOST and x must be an
 * aligned runtime allocation/view spanning (n-1)*incx+1 floats on that device.
 * Gaps are preserved; multiplication reads x even when alpha is zero. */
VKDNN_API vkblasStatus_t vkblasSscal(vkblasHandle_t handle, int n, const float* alpha, float* x, int incx);
/* Synchronous FP32 GPU reductions with a HOST result pointer, matching CUDA 13.2
 * cuBLAS signatures. n<=0 or nonpositive increments write zero without reading
 * inputs. Positive increments and aligned runtime views are supported. Only the
 * final scalar is copied to HOST; inputs are never reduced on the CPU. */
VKDNN_API vkblasStatus_t vkblasSasum(vkblasHandle_t handle, int n, const float* x, int incx, float* result);
VKDNN_API vkblasStatus_t vkblasSdot(vkblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
VKDNN_API vkblasStatus_t vkblasSgemm(vkblasHandle_t handle, vkblasOperation_t transa, vkblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);
#ifdef __cplusplus
}
#endif
#endif
