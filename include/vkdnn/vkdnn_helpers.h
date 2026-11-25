#ifndef VKDNN_HELPERS_H
#define VKDNN_HELPERS_H
#include "vkdnn_ops.h"
#ifdef __cplusplus
extern "C"
{
#endif

typedef struct vkdnnTensorView
{
    int n, c, h, w;
    void* data;
    void* gradient;
} vkdnnTensorView;

typedef enum vkdnnActivationKind
{
    VKDNN_HELPER_RELU,
    VKDNN_HELPER_SIGMOID,
    VKDNN_HELPER_TANH,
    VKDNN_HELPER_SOFTMAX,
    VKDNN_HELPER_SOFTMAX_FAST,
    VKDNN_HELPER_SOFTMAX_LOG,
    VKDNN_HELPER_SOFTMAX_CHANNEL
} vkdnnActivationKind;

typedef enum vkdnnConvolutionDirection
{
    VKDNN_HELPER_FORWARD,
    VKDNN_HELPER_BACKWARD_DATA,
    VKDNN_HELPER_BACKWARD_FILTER
} vkdnnConvolutionDirection;

VKDNN_API vkdnnStatus_t vkdnnExecuteActivation(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnActivationKind activation, float alpha, float beta, int backward);
VKDNN_API vkdnnStatus_t vkdnnExecutePooling(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnPoolingMode_t mode, int windowH, int windowW, int strideH, int strideW, int padH, int padW,
    float alpha, float beta, int backward);
VKDNN_API vkdnnStatus_t vkdnnExecuteConvolution(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights,
    vkdnnTensorView output, int strideH, int strideW, int padH, int padW, float alpha, float beta,
    vkdnnConvolutionDirection direction);

#ifdef __cplusplus
}
#endif
#endif