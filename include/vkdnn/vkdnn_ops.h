#ifndef VKDNN_OPS_H
#define VKDNN_OPS_H
#include "vkdnn.h"
#ifdef __cplusplus
extern "C"
{
#endif

typedef struct vkdnnDnnContext* vkdnnHandle_t;
typedef struct vkdnnTensorStruct* vkdnnTensorDescriptor_t;
typedef struct vkdnnFilterStruct* vkdnnFilterDescriptor_t;
typedef struct vkdnnConvolutionStruct* vkdnnConvolutionDescriptor_t;
typedef struct vkdnnActivationStruct* vkdnnActivationDescriptor_t;
typedef struct vkdnnPoolingStruct* vkdnnPoolingDescriptor_t;
typedef enum vkdnnStatus_t
{
    VKDNN_STATUS_SUCCESS = 0,
    VKDNN_STATUS_NOT_INITIALIZED = 1,
    VKDNN_STATUS_ALLOC_FAILED = 2,
    VKDNN_STATUS_BAD_PARAM = 3,
    VKDNN_STATUS_INTERNAL_ERROR = 4,
    VKDNN_STATUS_EXECUTION_FAILED = 8,
    VKDNN_STATUS_NOT_SUPPORTED = 9
} vkdnnStatus_t;
typedef enum vkdnnDataType_t
{
    VKDNN_DATA_FLOAT = 0,
    VKDNN_DATA_DOUBLE = 1,
    VKDNN_DATA_HALF = 2
} vkdnnDataType_t;
typedef enum vkdnnTensorFormat_t
{
    VKDNN_TENSOR_NCHW = 0,
    VKDNN_TENSOR_NHWC = 1
} vkdnnTensorFormat_t;
typedef enum vkdnnConvolutionMode_t
{
    VKDNN_CONVOLUTION = 0,
    VKDNN_CROSS_CORRELATION = 1
} vkdnnConvolutionMode_t;
typedef enum vkdnnNanPropagation_t
{
    VKDNN_NOT_PROPAGATE_NAN = 0,
    VKDNN_PROPAGATE_NAN = 1
} vkdnnNanPropagation_t;
typedef enum vkdnnActivationMode_t
{
    VKDNN_ACTIVATION_SIGMOID = 0,
    VKDNN_ACTIVATION_RELU = 1,
    VKDNN_ACTIVATION_TANH = 2,
    VKDNN_ACTIVATION_CLIPPED_RELU = 3,
    VKDNN_ACTIVATION_ELU = 4,
    VKDNN_ACTIVATION_IDENTITY = 5
} vkdnnActivationMode_t;
typedef enum vkdnnPoolingMode_t
{
    VKDNN_POOLING_MAX = 0,
    VKDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    VKDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    VKDNN_POOLING_MAX_DETERMINISTIC = 3
} vkdnnPoolingMode_t;
typedef enum vkdnnSoftmaxAlgorithm_t
{
    VKDNN_SOFTMAX_FAST = 0,
    VKDNN_SOFTMAX_ACCURATE = 1,
    VKDNN_SOFTMAX_LOG = 2
} vkdnnSoftmaxAlgorithm_t;
typedef enum vkdnnSoftmaxMode_t
{
    VKDNN_SOFTMAX_MODE_INSTANCE = 0,
    VKDNN_SOFTMAX_MODE_CHANNEL = 1
} vkdnnSoftmaxMode_t;
/* Initial algorithm subset, not a promise of implemented algorithms. */
typedef enum vkdnnConvolutionFwdAlgo_t
{
    VKDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
} vkdnnConvolutionFwdAlgo_t;
typedef enum vkdnnConvolutionBwdDataAlgo_t
{
    VKDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
} vkdnnConvolutionBwdDataAlgo_t;
typedef enum vkdnnConvolutionBwdFilterAlgo_t
{
    VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
} vkdnnConvolutionBwdFilterAlgo_t;

/* All declarations below are cuDNN-style stubs. Every call returns NOT_SUPPORTED
 * without parameter validation, output writes, resource creation or GPU work.
 * Future initial implementation targets FP32; enum presence is not dtype support. */
VKDNN_API vkdnnStatus_t vkdnnCreate(vkdnnHandle_t* handle);
VKDNN_API vkdnnStatus_t vkdnnDestroy(vkdnnHandle_t handle);
VKDNN_API vkdnnStatus_t vkdnnCreateTensorDescriptor(vkdnnTensorDescriptor_t* tensorDesc);
VKDNN_API vkdnnStatus_t vkdnnDestroyTensorDescriptor(vkdnnTensorDescriptor_t tensorDesc);
VKDNN_API vkdnnStatus_t vkdnnSetTensor4dDescriptor(vkdnnTensorDescriptor_t tensorDesc,
    vkdnnTensorFormat_t format, vkdnnDataType_t dataType, int n, int c, int h, int w);
VKDNN_API vkdnnStatus_t vkdnnCreateFilterDescriptor(vkdnnFilterDescriptor_t* filterDesc);
VKDNN_API vkdnnStatus_t vkdnnDestroyFilterDescriptor(vkdnnFilterDescriptor_t filterDesc);
VKDNN_API vkdnnStatus_t vkdnnSetFilter4dDescriptor(vkdnnFilterDescriptor_t filterDesc,
    vkdnnDataType_t dataType, vkdnnTensorFormat_t format, int k, int c, int h, int w);
VKDNN_API vkdnnStatus_t vkdnnCreateConvolutionDescriptor(vkdnnConvolutionDescriptor_t* convDesc);
VKDNN_API vkdnnStatus_t vkdnnDestroyConvolutionDescriptor(vkdnnConvolutionDescriptor_t convDesc);
VKDNN_API vkdnnStatus_t vkdnnSetConvolution2dDescriptor(vkdnnConvolutionDescriptor_t convDesc,
    int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w,
    vkdnnConvolutionMode_t mode, vkdnnDataType_t computeType);
VKDNN_API vkdnnStatus_t vkdnnSetConvolutionGroupCount(vkdnnConvolutionDescriptor_t convDesc, int groupCount);
VKDNN_API vkdnnStatus_t vkdnnGetConvolution2dForwardOutputDim(vkdnnConvolutionDescriptor_t convDesc,
    vkdnnTensorDescriptor_t inputTensorDesc, vkdnnFilterDescriptor_t filterDesc, int* n, int* c, int* h, int* w);
VKDNN_API vkdnnStatus_t vkdnnGetConvolutionForwardWorkspaceSize(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t xDesc, vkdnnFilterDescriptor_t wDesc, vkdnnConvolutionDescriptor_t convDesc,
    vkdnnTensorDescriptor_t yDesc, vkdnnConvolutionFwdAlgo_t algo, size_t* sizeInBytes);
VKDNN_API vkdnnStatus_t vkdnnConvolutionForward(vkdnnHandle_t handle, const void* alpha,
    vkdnnTensorDescriptor_t xDesc, const void* x, vkdnnFilterDescriptor_t wDesc, const void* w,
    vkdnnConvolutionDescriptor_t convDesc, vkdnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes, const void* beta, vkdnnTensorDescriptor_t yDesc, void* y);
VKDNN_API vkdnnStatus_t vkdnnGetConvolutionBackwardDataWorkspaceSize(vkdnnHandle_t handle,
    vkdnnFilterDescriptor_t wDesc, vkdnnTensorDescriptor_t dyDesc, vkdnnConvolutionDescriptor_t convDesc,
    vkdnnTensorDescriptor_t dxDesc, vkdnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes);
VKDNN_API vkdnnStatus_t vkdnnConvolutionBackwardData(vkdnnHandle_t handle, const void* alpha,
    vkdnnFilterDescriptor_t wDesc, const void* w, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    vkdnnConvolutionDescriptor_t convDesc, vkdnnConvolutionBwdDataAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes, const void* beta, vkdnnTensorDescriptor_t dxDesc, void* dx);
VKDNN_API vkdnnStatus_t vkdnnGetConvolutionBackwardFilterWorkspaceSize(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t xDesc, vkdnnTensorDescriptor_t dyDesc, vkdnnConvolutionDescriptor_t convDesc,
    vkdnnFilterDescriptor_t dwDesc, vkdnnConvolutionBwdFilterAlgo_t algo, size_t* sizeInBytes);
VKDNN_API vkdnnStatus_t vkdnnConvolutionBackwardFilter(vkdnnHandle_t handle, const void* alpha,
    vkdnnTensorDescriptor_t xDesc, const void* x, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    vkdnnConvolutionDescriptor_t convDesc, vkdnnConvolutionBwdFilterAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes, const void* beta, vkdnnFilterDescriptor_t dwDesc, void* dw);
VKDNN_API vkdnnStatus_t vkdnnConvolutionBackwardBias(vkdnnHandle_t handle, const void* alpha,
    vkdnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, vkdnnTensorDescriptor_t dbDesc, void* db);
VKDNN_API vkdnnStatus_t vkdnnAddTensor(vkdnnHandle_t handle, const void* alpha,
    vkdnnTensorDescriptor_t aDesc, const void* A, const void* beta, vkdnnTensorDescriptor_t cDesc, void* C);
VKDNN_API vkdnnStatus_t vkdnnCreateActivationDescriptor(vkdnnActivationDescriptor_t* activationDesc);
VKDNN_API vkdnnStatus_t vkdnnDestroyActivationDescriptor(vkdnnActivationDescriptor_t activationDesc);
VKDNN_API vkdnnStatus_t vkdnnSetActivationDescriptor(vkdnnActivationDescriptor_t activationDesc,
    vkdnnActivationMode_t mode, vkdnnNanPropagation_t reluNanOpt, double coef);
VKDNN_API vkdnnStatus_t vkdnnActivationForward(vkdnnHandle_t handle, vkdnnActivationDescriptor_t activationDesc,
    const void* alpha, vkdnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, vkdnnTensorDescriptor_t yDesc, void* y);
VKDNN_API vkdnnStatus_t vkdnnActivationBackward(vkdnnHandle_t handle, vkdnnActivationDescriptor_t activationDesc,
    const void* alpha, vkdnnTensorDescriptor_t yDesc, const void* y, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    vkdnnTensorDescriptor_t xDesc, const void* x, const void* beta, vkdnnTensorDescriptor_t dxDesc, void* dx);
VKDNN_API vkdnnStatus_t vkdnnCreatePoolingDescriptor(vkdnnPoolingDescriptor_t* poolingDesc);
VKDNN_API vkdnnStatus_t vkdnnDestroyPoolingDescriptor(vkdnnPoolingDescriptor_t poolingDesc);
VKDNN_API vkdnnStatus_t vkdnnSetPooling2dDescriptor(vkdnnPoolingDescriptor_t poolingDesc,
    vkdnnPoolingMode_t mode, vkdnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride);
VKDNN_API vkdnnStatus_t vkdnnGetPooling2dForwardOutputDim(vkdnnPoolingDescriptor_t poolingDesc,
    vkdnnTensorDescriptor_t inputTensorDesc, int* n, int* c, int* h, int* w);
VKDNN_API vkdnnStatus_t vkdnnPoolingForward(vkdnnHandle_t handle, vkdnnPoolingDescriptor_t poolingDesc,
    const void* alpha, vkdnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, vkdnnTensorDescriptor_t yDesc, void* y);
VKDNN_API vkdnnStatus_t vkdnnPoolingBackward(vkdnnHandle_t handle, vkdnnPoolingDescriptor_t poolingDesc,
    const void* alpha, vkdnnTensorDescriptor_t yDesc, const void* y, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    vkdnnTensorDescriptor_t xDesc, const void* x, const void* beta, vkdnnTensorDescriptor_t dxDesc, void* dx);
VKDNN_API vkdnnStatus_t vkdnnSoftmaxForward(vkdnnHandle_t handle, vkdnnSoftmaxAlgorithm_t algo,
    vkdnnSoftmaxMode_t mode, const void* alpha, vkdnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, vkdnnTensorDescriptor_t yDesc, void* y);
VKDNN_API vkdnnStatus_t vkdnnSoftmaxBackward(vkdnnHandle_t handle, vkdnnSoftmaxAlgorithm_t algo,
    vkdnnSoftmaxMode_t mode, const void* alpha, vkdnnTensorDescriptor_t yDesc, const void* y,
    vkdnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, vkdnnTensorDescriptor_t dxDesc, void* dx);

#ifdef __cplusplus
}
#endif
#endif
