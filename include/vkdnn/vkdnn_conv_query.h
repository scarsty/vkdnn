#ifndef VKDNN_CONV_QUERY_H
#define VKDNN_CONV_QUERY_H
#include "vkdnn_ops.h"
#ifdef __cplusplus
extern "C"
{
#endif

typedef enum vkdnnMathType_t
{
    VKDNN_DEFAULT_MATH = 0,
    VKDNN_TENSOR_OP_MATH = 1,
    VKDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    VKDNN_FMA_MATH = 3
} vkdnnMathType_t;
typedef enum vkdnnDeterminism_t
{
    VKDNN_NON_DETERMINISTIC = 0,
    VKDNN_DETERMINISTIC = 1
} vkdnnDeterminism_t;

typedef struct vkdnnConvolutionFwdAlgoPerfStruct
{
    vkdnnConvolutionFwdAlgo_t algo;
    vkdnnStatus_t status;
    float time;
    size_t memory;
    vkdnnDeterminism_t determinism;
    vkdnnMathType_t mathType;
    int reserved[3];
} vkdnnConvolutionFwdAlgoPerf_t;
typedef struct vkdnnConvolutionBwdDataAlgoPerfStruct
{
    vkdnnConvolutionBwdDataAlgo_t algo;
    vkdnnStatus_t status;
    float time;
    size_t memory;
    vkdnnDeterminism_t determinism;
    vkdnnMathType_t mathType;
    int reserved[3];
} vkdnnConvolutionBwdDataAlgoPerf_t;
typedef struct vkdnnConvolutionBwdFilterAlgoPerfStruct
{
    vkdnnConvolutionBwdFilterAlgo_t algo;
    vkdnnStatus_t status;
    float time;
    size_t memory;
    vkdnnDeterminism_t determinism;
    vkdnnMathType_t mathType;
    int reserved[3];
} vkdnnConvolutionBwdFilterAlgoPerf_t;

/* DEFAULT is a no-op on a registered live descriptor. Invalid descriptors return BAD_PARAM;
 * other math modes return NOT_SUPPORTED without changing the convolution configuration. */
VKDNN_API vkdnnStatus_t vkdnnSetConvolutionMathType(vkdnnConvolutionDescriptor_t convDesc, vkdnnMathType_t mathType);
/* One validated local algorithm only; time is NaN (not measured), never a timing estimate.
 * Errors leave returnedAlgoCount and perfResults unchanged. Positive requestedAlgoCount required. */
VKDNN_API vkdnnStatus_t vkdnnGetConvolutionForwardAlgorithm_v7(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t xDesc, vkdnnFilterDescriptor_t wDesc, vkdnnConvolutionDescriptor_t convDesc,
    vkdnnTensorDescriptor_t yDesc, int requestedAlgoCount, int* returnedAlgoCount, vkdnnConvolutionFwdAlgoPerf_t* perfResults);
VKDNN_API vkdnnStatus_t vkdnnGetConvolutionBackwardDataAlgorithm_v7(vkdnnHandle_t handle,
    vkdnnFilterDescriptor_t wDesc, vkdnnTensorDescriptor_t dyDesc, vkdnnConvolutionDescriptor_t convDesc,
    vkdnnTensorDescriptor_t dxDesc, int requestedAlgoCount, int* returnedAlgoCount, vkdnnConvolutionBwdDataAlgoPerf_t* perfResults);
VKDNN_API vkdnnStatus_t vkdnnGetConvolutionBackwardFilterAlgorithm_v7(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t xDesc, vkdnnTensorDescriptor_t dyDesc, vkdnnConvolutionDescriptor_t convDesc,
    vkdnnFilterDescriptor_t dwDesc, int requestedAlgoCount, int* returnedAlgoCount, vkdnnConvolutionBwdFilterAlgoPerf_t* perfResults);
/* Execute the sole algorithm once with alpha=1, beta=0, overwriting the destination.
 * time is measured host milliseconds for the complete synchronous operator call, including
 * pipeline/setup/submission/wait overhead, NOT pure GPU time. No warmup or averaging.
 * Errors preserve count/results; execution failures may modify the destination.
 * Data ranges and aliasing follow the corresponding convolution operator's contract. */
VKDNN_API vkdnnStatus_t vkdnnFindConvolutionForwardAlgorithmEx(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t xDesc, const void* x, vkdnnFilterDescriptor_t wDesc, const void* w,
    vkdnnConvolutionDescriptor_t convDesc, vkdnnTensorDescriptor_t yDesc, void* y,
    int requestedAlgoCount, int* returnedAlgoCount, vkdnnConvolutionFwdAlgoPerf_t* perfResults,
    void* workSpace, size_t workSpaceSizeInBytes);
VKDNN_API vkdnnStatus_t vkdnnFindConvolutionBackwardDataAlgorithmEx(vkdnnHandle_t handle,
    vkdnnFilterDescriptor_t wDesc, const void* w, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    vkdnnConvolutionDescriptor_t convDesc, vkdnnTensorDescriptor_t dxDesc, void* dx,
    int requestedAlgoCount, int* returnedAlgoCount, vkdnnConvolutionBwdDataAlgoPerf_t* perfResults,
    void* workSpace, size_t workSpaceSizeInBytes);
VKDNN_API vkdnnStatus_t vkdnnFindConvolutionBackwardFilterAlgorithmEx(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t xDesc, const void* x, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    vkdnnConvolutionDescriptor_t convDesc, vkdnnFilterDescriptor_t dwDesc, void* dw,
    int requestedAlgoCount, int* returnedAlgoCount, vkdnnConvolutionBwdFilterAlgoPerf_t* perfResults,
    void* workSpace, size_t workSpaceSizeInBytes);
#ifdef __cplusplus
}
#endif
#endif
