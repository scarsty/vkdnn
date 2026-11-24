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
    CHECK(vkdnnCreate(&handle));
    CHECK(vkdnnCreateTensorDescriptor(&tensor));
    CHECK(vkdnnSetTensor4dDescriptor(tensor, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 1, 4, 4));
    CHECK(vkdnnCreateFilterDescriptor(&filter));
    CHECK(vkdnnSetFilter4dDescriptor(filter, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 1, 1, 3, 3));
    CHECK(vkdnnCreateConvolutionDescriptor(&conv));
    CHECK(vkdnnSetConvolution2dDescriptor(conv, 1, 1, 1, 1, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
    CHECK(vkdnnSetConvolutionGroupCount(conv, 1));
    CHECK(vkdnnGetConvolution2dForwardOutputDim(conv, tensor, filter, &n, &c, &h, &w));
    CHECK(vkdnnGetConvolutionForwardWorkspaceSize(handle, tensor, filter, conv, tensor, VKDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace));
    CHECK(vkdnnConvolutionForward(handle, &data, tensor, &data, filter, &data, conv, VKDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &data, sizeof(data), &data, tensor, &data));
    CHECK(vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, filter, tensor, conv, tensor, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace));
    CHECK(vkdnnConvolutionBackwardData(handle, &data, filter, &data, tensor, &data, conv, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &data, sizeof(data), &data, tensor, &data));
    CHECK(vkdnnGetConvolutionBackwardFilterWorkspaceSize(handle, tensor, tensor, conv, filter, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &workspace));
    CHECK(vkdnnConvolutionBackwardFilter(handle, &data, tensor, &data, tensor, &data, conv, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &data, sizeof(data), &data, filter, &data));
    CHECK(vkdnnConvolutionBackwardBias(handle, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnAddTensor(handle, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnCreateActivationDescriptor(&activation));
    CHECK(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_RELU, VKDNN_PROPAGATE_NAN, 0));
    CHECK(vkdnnActivationForward(handle, activation, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnActivationBackward(handle, activation, &data, tensor, &data, tensor, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnCreatePoolingDescriptor(&pool));
    CHECK(vkdnnSetPooling2dDescriptor(pool, VKDNN_POOLING_MAX, VKDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));
    CHECK(vkdnnGetPooling2dForwardOutputDim(pool, tensor, &n, &c, &h, &w));
    CHECK(vkdnnPoolingForward(handle, pool, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnPoolingBackward(handle, pool, &data, tensor, &data, tensor, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnSoftmaxForward(handle, VKDNN_SOFTMAX_ACCURATE, VKDNN_SOFTMAX_MODE_CHANNEL, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnSoftmaxBackward(handle, VKDNN_SOFTMAX_ACCURATE, VKDNN_SOFTMAX_MODE_CHANNEL, &data, tensor, &data, tensor, &data, &data, tensor, &data));
    CHECK(vkdnnDestroyPoolingDescriptor(pool));
    CHECK(vkdnnDestroyActivationDescriptor(activation));
    CHECK(vkdnnDestroyConvolutionDescriptor(conv));
    CHECK(vkdnnDestroyFilterDescriptor(filter));
    CHECK(vkdnnDestroyTensorDescriptor(tensor));
    CHECK(vkdnnDestroy(handle));
    if (vkblasCreate(&blas) != VKBLAS_STATUS_NOT_SUPPORTED
        || vkblasSetPointerMode(blas, VKBLAS_POINTER_MODE_HOST) != VKBLAS_STATUS_NOT_SUPPORTED
        || vkblasSgemm(blas, VKBLAS_OP_N, VKBLAS_OP_N, 1, 1, 1, &data, &data, 1, &data, 1, &data, &data, 1) != VKBLAS_STATUS_NOT_SUPPORTED
        || vkblasDestroy(blas) != VKBLAS_STATUS_NOT_SUPPORTED) { return 1; }
    if (handle || tensor || filter || conv || activation || pool || blas
        || workspace != 123 || n != -1 || c != -1 || h != -1 || w != -1 || data != 17.0f) { return 1; }
    puts("PASS: CNN/BLAS stubs link from C, report NOT_SUPPORTED and preserve outputs");
    return 0;
}
