#include "vkdnn/vkdnn_helpers.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace
{
vkdnnStatus_t runtimeStatus(vkdnnError_t status)
{
    if (status == vkdnnSuccess) { return VKDNN_STATUS_SUCCESS; }
    if (status == vkdnnErrorMemoryAllocation) { return VKDNN_STATUS_ALLOC_FAILED; }
    return VKDNN_STATUS_EXECUTION_FAILED;
}
#define VKDNN_HELPER_CHECK(expression) \
    do \
    { \
        const auto status = (expression); \
        if (status != VKDNN_STATUS_SUCCESS) { return status; } \
    } while (false)
size_t bytes(vkdnnTensorView view)
{
    size_t count = sizeof(float);
    for (int dimension : { view.n, view.c, view.h, view.w })
    {
        if (dimension <= 0 || count > (std::numeric_limits<size_t>::max)() / static_cast<size_t>(dimension))
        {
            return 0;
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}
template <class Descriptor, vkdnnStatus_t (*Create)(Descriptor*), vkdnnStatus_t (*Destroy)(Descriptor)>
struct Resource
{
    Descriptor value = nullptr;
    Resource() = default;
    vkdnnStatus_t create() { return Create(&value); }
    ~Resource()
    {
        if (value) { Destroy(value); }
    }
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
};
using Tensor = Resource<vkdnnTensorDescriptor_t, vkdnnCreateTensorDescriptor, vkdnnDestroyTensorDescriptor>;
using Activation = Resource<vkdnnActivationDescriptor_t, vkdnnCreateActivationDescriptor, vkdnnDestroyActivationDescriptor>;
using Pooling = Resource<vkdnnPoolingDescriptor_t, vkdnnCreatePoolingDescriptor, vkdnnDestroyPoolingDescriptor>;
using Filter = Resource<vkdnnFilterDescriptor_t, vkdnnCreateFilterDescriptor, vkdnnDestroyFilterDescriptor>;
using Convolution = Resource<vkdnnConvolutionDescriptor_t, vkdnnCreateConvolutionDescriptor, vkdnnDestroyConvolutionDescriptor>;
vkdnnStatus_t set(Tensor& descriptor, vkdnnTensorView view)
{
    if (!bytes(view)) { return VKDNN_STATUS_BAD_PARAM; }
    VKDNN_HELPER_CHECK(descriptor.create());
    return vkdnnSetTensor4dDescriptor(descriptor.value, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, view.n, view.c, view.h, view.w);
}
struct Output
{
    void* data = nullptr;
    void* target = nullptr;
    size_t size = 0;
    bool owned = false;
    Output() = default;
    vkdnnStatus_t initialize(vkdnnTensorView view, void* destination, float beta, bool direct = false)
    {
        target = destination;
        size = bytes(view);
        if (!target || !size) { return VKDNN_STATUS_BAD_PARAM; }
        if (direct)
        {
            data = target;
            return VKDNN_STATUS_SUCCESS;
        }
        VKDNN_HELPER_CHECK(runtimeStatus(vkdnnMalloc(&data, size)));
        owned = true;
        if (beta != 0)
        {
            auto status = vkdnnMemcpy(data, target, size, vkdnnMemcpyDeviceToDevice);
            if (status != vkdnnSuccess)
            {
                vkdnnFree(data);
                data = nullptr;
                return runtimeStatus(status);
            }
        }
        return VKDNN_STATUS_SUCCESS;
    }
    ~Output()
    {
        if (data && owned) { vkdnnFree(data); }
    }
    vkdnnStatus_t commit()
    {
        if (!owned) { return VKDNN_STATUS_SUCCESS; }
        return runtimeStatus(vkdnnMemcpy(target, data, size, vkdnnMemcpyDeviceToDevice));
    }
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;
};
bool overlaps(const void* destination, size_t destinationBytes, const void* source, size_t sourceBytes)
{
    if (!source) { return false; }
    auto destinationAddress = reinterpret_cast<uintptr_t>(destination);
    auto sourceAddress = reinterpret_cast<uintptr_t>(source);
    if (destinationAddress >= sourceAddress) { return destinationAddress - sourceAddress < sourceBytes; }
    return sourceAddress - destinationAddress < destinationBytes;
}
struct ConvolutionDescriptors
{
    Tensor inputDesc, outputDesc;
    Filter filter;
    Convolution convolution;
    vkdnnStatus_t initialize(vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output, int strideH, int strideW, int padH, int padW)
    {
        VKDNN_HELPER_CHECK(set(inputDesc, input));
        VKDNN_HELPER_CHECK(set(outputDesc, output));
        if (!bytes(weights)) { return VKDNN_STATUS_BAD_PARAM; }
        VKDNN_HELPER_CHECK(filter.create());
        VKDNN_HELPER_CHECK(convolution.create());
        VKDNN_HELPER_CHECK(vkdnnSetFilter4dDescriptor(filter.value, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, weights.n, weights.c, weights.h, weights.w));
        return vkdnnSetConvolution2dDescriptor(convolution.value, padH, padW, strideH, strideW, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT);
    }
};
}    //namespace

vkdnnStatus_t vkdnnActivationForwardEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnActivationKind activation, float alpha, float beta)
{
    if (activation < VKDNN_HELPER_RELU || activation > VKDNN_HELPER_SOFTMAX_CHANNEL) { return VKDNN_STATUS_NOT_SUPPORTED; }
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    Output temporary;
    bool direct = !overlaps(output.data, bytes(output), input.data, bytes(input));
    VKDNN_HELPER_CHECK(temporary.initialize(output, output.data, beta, direct));
    if (activation >= VKDNN_HELPER_SOFTMAX)
    {
        auto mode = activation == VKDNN_HELPER_SOFTMAX_CHANNEL ? VKDNN_SOFTMAX_MODE_CHANNEL : VKDNN_SOFTMAX_MODE_INSTANCE;
        auto algorithm = VKDNN_SOFTMAX_ACCURATE;
        if (activation == VKDNN_HELPER_SOFTMAX_FAST) { algorithm = VKDNN_SOFTMAX_FAST; }
        else if (activation == VKDNN_HELPER_SOFTMAX_LOG) { algorithm = VKDNN_SOFTMAX_LOG; }
        VKDNN_HELPER_CHECK(vkdnnSoftmaxForward(handle, algorithm, mode, &alpha, inputDesc.value, input.data, &beta, outputDesc.value, temporary.data));
    }
    else
    {
        Activation descriptor;
        VKDNN_HELPER_CHECK(descriptor.create());
        auto mode = VKDNN_ACTIVATION_RELU;
        if (activation == VKDNN_HELPER_SIGMOID) { mode = VKDNN_ACTIVATION_SIGMOID; }
        else if (activation == VKDNN_HELPER_TANH) { mode = VKDNN_ACTIVATION_TANH; }
        VKDNN_HELPER_CHECK(vkdnnSetActivationDescriptor(descriptor.value, mode, VKDNN_NOT_PROPAGATE_NAN, 0));
        VKDNN_HELPER_CHECK(vkdnnActivationForward(handle, descriptor.value, &alpha, inputDesc.value, input.data, &beta, outputDesc.value, temporary.data));
    }
    return temporary.commit();
}

vkdnnStatus_t vkdnnActivationBackwardEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnActivationKind activation, float alpha, float beta)
{
    if (activation < VKDNN_HELPER_RELU || activation > VKDNN_HELPER_SOFTMAX_CHANNEL) { return VKDNN_STATUS_NOT_SUPPORTED; }
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    Output temporary;
    bool direct = !overlaps(input.gradient, bytes(input), input.data, bytes(input))
        && !overlaps(input.gradient, bytes(input), output.data, bytes(output))
        && !overlaps(input.gradient, bytes(input), output.gradient, bytes(output));
    VKDNN_HELPER_CHECK(temporary.initialize(input, input.gradient, beta, direct));
    if (activation >= VKDNN_HELPER_SOFTMAX)
    {
        auto mode = activation == VKDNN_HELPER_SOFTMAX_CHANNEL ? VKDNN_SOFTMAX_MODE_CHANNEL : VKDNN_SOFTMAX_MODE_INSTANCE;
        auto algorithm = VKDNN_SOFTMAX_ACCURATE;
        if (activation == VKDNN_HELPER_SOFTMAX_FAST) { algorithm = VKDNN_SOFTMAX_FAST; }
        else if (activation == VKDNN_HELPER_SOFTMAX_LOG) { algorithm = VKDNN_SOFTMAX_LOG; }
        VKDNN_HELPER_CHECK(vkdnnSoftmaxBackward(handle, algorithm, mode, &alpha, outputDesc.value, output.data, outputDesc.value, output.gradient, &beta, inputDesc.value, temporary.data));
    }
    else
    {
        Activation descriptor;
        VKDNN_HELPER_CHECK(descriptor.create());
        auto mode = VKDNN_ACTIVATION_RELU;
        if (activation == VKDNN_HELPER_SIGMOID) { mode = VKDNN_ACTIVATION_SIGMOID; }
        else if (activation == VKDNN_HELPER_TANH) { mode = VKDNN_ACTIVATION_TANH; }
        VKDNN_HELPER_CHECK(vkdnnSetActivationDescriptor(descriptor.value, mode, VKDNN_NOT_PROPAGATE_NAN, 0));
        VKDNN_HELPER_CHECK(vkdnnActivationBackward(handle, descriptor.value, &alpha, outputDesc.value, output.data, outputDesc.value, output.gradient, inputDesc.value, input.data, &beta, inputDesc.value, temporary.data));
    }
    return temporary.commit();
}

vkdnnStatus_t vkdnnPoolingForwardEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnPoolingMode_t mode, int windowH, int windowW, int strideH, int strideW, int padH, int padW, float alpha, float beta)
{
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    Pooling descriptor;
    VKDNN_HELPER_CHECK(descriptor.create());
    VKDNN_HELPER_CHECK(vkdnnSetPooling2dDescriptor(descriptor.value, mode, VKDNN_NOT_PROPAGATE_NAN, windowH, windowW, padH, padW, strideH, strideW));
    Output temporary;
    bool direct = !overlaps(output.data, bytes(output), input.data, bytes(input));
    VKDNN_HELPER_CHECK(temporary.initialize(output, output.data, beta, direct));
    VKDNN_HELPER_CHECK(vkdnnPoolingForward(handle, descriptor.value, &alpha, inputDesc.value, input.data, &beta, outputDesc.value, temporary.data));
    return temporary.commit();
}

vkdnnStatus_t vkdnnPoolingBackwardEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnPoolingMode_t mode, int windowH, int windowW, int strideH, int strideW, int padH, int padW, float alpha, float beta)
{
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    Pooling descriptor;
    VKDNN_HELPER_CHECK(descriptor.create());
    VKDNN_HELPER_CHECK(vkdnnSetPooling2dDescriptor(descriptor.value, mode, VKDNN_NOT_PROPAGATE_NAN, windowH, windowW, padH, padW, strideH, strideW));
    Output temporary;
    bool direct = !overlaps(input.gradient, bytes(input), input.data, bytes(input))
        && !overlaps(input.gradient, bytes(input), output.data, bytes(output))
        && !overlaps(input.gradient, bytes(input), output.gradient, bytes(output));
    VKDNN_HELPER_CHECK(temporary.initialize(input, input.gradient, beta, direct));
    VKDNN_HELPER_CHECK(vkdnnPoolingBackward(handle, descriptor.value, &alpha, outputDesc.value, output.data, outputDesc.value, output.gradient, inputDesc.value, input.data, &beta, inputDesc.value, temporary.data));
    return temporary.commit();
}

vkdnnStatus_t vkdnnConvolutionForwardEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, float alpha, float beta, vkdnnConvolutionFwdAlgo_t algorithm)
{
    ConvolutionDescriptors descriptors;
    VKDNN_HELPER_CHECK(descriptors.initialize(input, weights, output, strideH, strideW, padH, padW));
    Output temporary;
    bool direct = !overlaps(output.data, bytes(output), input.data, bytes(input))
        && !overlaps(output.data, bytes(output), weights.data, bytes(weights));
    VKDNN_HELPER_CHECK(temporary.initialize(output, output.data, beta, direct));
    VKDNN_HELPER_CHECK(vkdnnConvolutionForward(handle, &alpha, descriptors.inputDesc.value, input.data, descriptors.filter.value, weights.data, descriptors.convolution.value, algorithm, nullptr, 0, &beta, descriptors.outputDesc.value, temporary.data));
    return temporary.commit();
}
vkdnnStatus_t vkdnnConvolutionBackwardDataEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, float alpha, float beta, vkdnnConvolutionBwdDataAlgo_t algorithm)
{
    ConvolutionDescriptors descriptors;
    VKDNN_HELPER_CHECK(descriptors.initialize(input, weights, output, strideH, strideW, padH, padW));
    Output temporary;
    bool direct = !overlaps(input.gradient, bytes(input), weights.data, bytes(weights))
        && !overlaps(input.gradient, bytes(input), output.gradient, bytes(output));
    VKDNN_HELPER_CHECK(temporary.initialize(input, input.gradient, beta, direct));
    VKDNN_HELPER_CHECK(vkdnnConvolutionBackwardData(handle, &alpha, descriptors.filter.value, weights.data, descriptors.outputDesc.value, output.gradient, descriptors.convolution.value, algorithm, nullptr, 0, &beta, descriptors.inputDesc.value, temporary.data));
    return temporary.commit();
}
vkdnnStatus_t vkdnnConvolutionBackwardFilterEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, float alpha, float beta, vkdnnConvolutionBwdFilterAlgo_t algorithm)
{
    ConvolutionDescriptors descriptors;
    VKDNN_HELPER_CHECK(descriptors.initialize(input, weights, output, strideH, strideW, padH, padW));
    Output temporary;
    bool direct = !overlaps(weights.gradient, bytes(weights), input.data, bytes(input))
        && !overlaps(weights.gradient, bytes(weights), output.gradient, bytes(output));
    VKDNN_HELPER_CHECK(temporary.initialize(weights, weights.gradient, beta, direct));
    VKDNN_HELPER_CHECK(vkdnnConvolutionBackwardFilter(handle, &alpha, descriptors.inputDesc.value, input.data, descriptors.outputDesc.value, output.gradient, descriptors.convolution.value, algorithm, nullptr, 0, &beta, descriptors.filter.value, temporary.data));
    return temporary.commit();
}

vkdnnStatus_t vkdnnFindConvolutionForwardAlgorithmViewEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, int requested, int* returned, vkdnnConvolutionFwdAlgoPerf_t* results)
{
    ConvolutionDescriptors descriptors;
    VKDNN_HELPER_CHECK(descriptors.initialize(input, weights, output, strideH, strideW, padH, padW));
    return vkdnnFindConvolutionForwardAlgorithmEx(handle, descriptors.inputDesc.value, input.data, descriptors.filter.value, weights.data,
        descriptors.convolution.value, descriptors.outputDesc.value, output.data, requested, returned, results, nullptr, 0);
}
vkdnnStatus_t vkdnnFindConvolutionBackwardDataAlgorithmViewEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, int requested, int* returned, vkdnnConvolutionBwdDataAlgoPerf_t* results)
{
    ConvolutionDescriptors descriptors;
    VKDNN_HELPER_CHECK(descriptors.initialize(input, weights, output, strideH, strideW, padH, padW));
    return vkdnnFindConvolutionBackwardDataAlgorithmEx(handle, descriptors.filter.value, weights.data, descriptors.outputDesc.value, output.gradient,
        descriptors.convolution.value, descriptors.inputDesc.value, input.gradient, requested, returned, results, nullptr, 0);
}
vkdnnStatus_t vkdnnFindConvolutionBackwardFilterAlgorithmViewEx(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, int requested, int* returned, vkdnnConvolutionBwdFilterAlgoPerf_t* results)
{
    ConvolutionDescriptors descriptors;
    VKDNN_HELPER_CHECK(descriptors.initialize(input, weights, output, strideH, strideW, padH, padW));
    return vkdnnFindConvolutionBackwardFilterAlgorithmEx(handle, descriptors.inputDesc.value, input.data, descriptors.outputDesc.value, output.gradient,
        descriptors.convolution.value, descriptors.filter.value, weights.gradient, requested, returned, results, nullptr, 0);
}