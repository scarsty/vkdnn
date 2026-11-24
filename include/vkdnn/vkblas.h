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
 * All functions below are stubs: NOT_SUPPORTED, no outputs or resources changed. */
VKDNN_API vkblasStatus_t vkblasCreate(vkblasHandle_t* handle);
VKDNN_API vkblasStatus_t vkblasDestroy(vkblasHandle_t handle);
VKDNN_API vkblasStatus_t vkblasSetPointerMode(vkblasHandle_t handle, vkblasPointerMode_t mode);
VKDNN_API vkblasStatus_t vkblasSgemm(vkblasHandle_t handle, vkblasOperation_t transa, vkblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);
#ifdef __cplusplus
}
#endif
#endif
