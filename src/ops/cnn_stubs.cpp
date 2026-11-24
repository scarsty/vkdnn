#include "vkdnn/vkblas.h"
#include "vkdnn/vkdnn_ops.h"

vkblasStatus_t vkblasCreate(vkblasHandle_t*) { return VKBLAS_STATUS_NOT_SUPPORTED; }
vkblasStatus_t vkblasDestroy(vkblasHandle_t) { return VKBLAS_STATUS_NOT_SUPPORTED; }
vkblasStatus_t vkblasSetPointerMode(vkblasHandle_t, vkblasPointerMode_t) { return VKBLAS_STATUS_NOT_SUPPORTED; }
vkblasStatus_t vkblasSgemm(vkblasHandle_t, vkblasOperation_t, vkblasOperation_t, int, int, int,
    const float*, const float*, int, const float*, int, const float*, float*, int)
{ return VKBLAS_STATUS_NOT_SUPPORTED; }

vkdnnStatus_t vkdnnCreate(vkdnnHandle_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnDestroy(vkdnnHandle_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnCreateTensorDescriptor(vkdnnTensorDescriptor_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnDestroyTensorDescriptor(vkdnnTensorDescriptor_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSetTensor4dDescriptor(vkdnnTensorDescriptor_t, vkdnnTensorFormat_t, vkdnnDataType_t, int, int, int, int) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnCreateFilterDescriptor(vkdnnFilterDescriptor_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnDestroyFilterDescriptor(vkdnnFilterDescriptor_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSetFilter4dDescriptor(vkdnnFilterDescriptor_t, vkdnnDataType_t, vkdnnTensorFormat_t, int, int, int, int) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnCreateConvolutionDescriptor(vkdnnConvolutionDescriptor_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnDestroyConvolutionDescriptor(vkdnnConvolutionDescriptor_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSetConvolution2dDescriptor(vkdnnConvolutionDescriptor_t, int, int, int, int, int, int, vkdnnConvolutionMode_t, vkdnnDataType_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSetConvolutionGroupCount(vkdnnConvolutionDescriptor_t, int) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnGetConvolution2dForwardOutputDim(vkdnnConvolutionDescriptor_t, vkdnnTensorDescriptor_t, vkdnnFilterDescriptor_t, int*, int*, int*, int*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnGetConvolutionForwardWorkspaceSize(vkdnnHandle_t, vkdnnTensorDescriptor_t, vkdnnFilterDescriptor_t, vkdnnConvolutionDescriptor_t, vkdnnTensorDescriptor_t, vkdnnConvolutionFwdAlgo_t, size_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnConvolutionForward(vkdnnHandle_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnFilterDescriptor_t, const void*, vkdnnConvolutionDescriptor_t, vkdnnConvolutionFwdAlgo_t, void*, size_t, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnGetConvolutionBackwardDataWorkspaceSize(vkdnnHandle_t, vkdnnFilterDescriptor_t, vkdnnTensorDescriptor_t, vkdnnConvolutionDescriptor_t, vkdnnTensorDescriptor_t, vkdnnConvolutionBwdDataAlgo_t, size_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnConvolutionBackwardData(vkdnnHandle_t, const void*, vkdnnFilterDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnConvolutionDescriptor_t, vkdnnConvolutionBwdDataAlgo_t, void*, size_t, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnGetConvolutionBackwardFilterWorkspaceSize(vkdnnHandle_t, vkdnnTensorDescriptor_t, vkdnnTensorDescriptor_t, vkdnnConvolutionDescriptor_t, vkdnnFilterDescriptor_t, vkdnnConvolutionBwdFilterAlgo_t, size_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnConvolutionBackwardFilter(vkdnnHandle_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnConvolutionDescriptor_t, vkdnnConvolutionBwdFilterAlgo_t, void*, size_t, const void*, vkdnnFilterDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnConvolutionBackwardBias(vkdnnHandle_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnAddTensor(vkdnnHandle_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnCreateActivationDescriptor(vkdnnActivationDescriptor_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnDestroyActivationDescriptor(vkdnnActivationDescriptor_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSetActivationDescriptor(vkdnnActivationDescriptor_t, vkdnnActivationMode_t, vkdnnNanPropagation_t, double) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnActivationForward(vkdnnHandle_t, vkdnnActivationDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnActivationBackward(vkdnnHandle_t, vkdnnActivationDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnCreatePoolingDescriptor(vkdnnPoolingDescriptor_t*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnDestroyPoolingDescriptor(vkdnnPoolingDescriptor_t) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSetPooling2dDescriptor(vkdnnPoolingDescriptor_t, vkdnnPoolingMode_t, vkdnnNanPropagation_t, int, int, int, int, int, int) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnGetPooling2dForwardOutputDim(vkdnnPoolingDescriptor_t, vkdnnTensorDescriptor_t, int*, int*, int*, int*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnPoolingForward(vkdnnHandle_t, vkdnnPoolingDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnPoolingBackward(vkdnnHandle_t, vkdnnPoolingDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSoftmaxForward(vkdnnHandle_t, vkdnnSoftmaxAlgorithm_t, vkdnnSoftmaxMode_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
vkdnnStatus_t vkdnnSoftmaxBackward(vkdnnHandle_t, vkdnnSoftmaxAlgorithm_t, vkdnnSoftmaxMode_t, const void*, vkdnnTensorDescriptor_t, const void*, vkdnnTensorDescriptor_t, const void*, const void*, vkdnnTensorDescriptor_t, void*) { return VKDNN_STATUS_NOT_SUPPORTED; }
