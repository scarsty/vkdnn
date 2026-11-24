#include "vkdnn/vkblas.h"
#include "vkdnn/vkdnn_ops.h"
#include <stdio.h>

#define CHECK(call) \
    do \
    { \
        if ((call) != VKDNN_STATUS_NOT_SUPPORTED) \
        { \
            fprintf(stderr, "Unexpected status: line %d\n", __LINE__); \
            return 1; \
        } \
    } while (0)

int main(void)
{
    vkdnnHandle_t handle = NULL;
    vkdnnTensorDescriptor_t tensor = NULL;
    vkdnnFilterDescriptor_t filter = NULL;
    vkdnnConvolutionDescriptor_t conv = NULL;
    vkdnnActivationDescriptor_t activation = NULL;
    vkdnnPoolingDescriptor_t pool = NULL;
    vkblasHandle_t blas = NULL;
    size_t workspace = 123;
    int n = -1, c = -1, h = -1, w = -1;
    float data = 17.0f;
    CHECK(vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, filter, tensor, conv, tensor, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace));
    CHECK(vkdnnConvolutionBackwardData(handle, &data, filter, &data, tensor, &data, conv, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &data, sizeof(data), &data, tensor, &data));
    CHECK(vkdnnGetConvolutionBackwardFilterWorkspaceSize(handle, tensor, tensor, conv, filter, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &workspace));
    CHECK(vkdnnConvolutionBackwardFilter(handle, &data, tensor, &data, tensor, &data, conv, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &data, sizeof(data), &data, filter, &data));
    CHECK(vkdnnConvolutionBackwardBias(handle, &data, tensor, &data, &data, tensor, &data));
    if (vkdnnActivationBackward(handle, activation, &data, tensor, &data, tensor, &data, tensor, &data, &data, tensor, &data) != VKDNN_STATUS_NOT_INITIALIZED) { return 1; }
    CHECK(vkdnnPoolingBackward(handle, pool, &data, tensor, &data, tensor, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnSoftmaxBackward(handle, VKDNN_SOFTMAX_ACCURATE, VKDNN_SOFTMAX_MODE_CHANNEL, &data, tensor, &data, tensor, &data, &data, tensor, &data));
    if (vkblasCreate(NULL) != VKBLAS_STATUS_INVALID_VALUE
        || vkblasSetPointerMode(blas, VKBLAS_POINTER_MODE_HOST) != VKBLAS_STATUS_NOT_INITIALIZED
        || vkblasSgemm(blas, VKBLAS_OP_N, VKBLAS_OP_N, 1, 1, 1, &data, &data, 1, &data, 1, &data, &data, 1) != VKBLAS_STATUS_NOT_INITIALIZED
        || vkblasDestroy(blas) != VKBLAS_STATUS_NOT_INITIALIZED) { return 1; }
    if (handle || tensor || filter || conv || activation || pool || blas
        || workspace != 123 || n != -1 || c != -1 || h != -1 || w != -1 || data != 17.0f) { return 1; }
    puts("PASS: CNN stubs preserve outputs; BLAS C API validates null handles");
    return 0;
}
