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
    vkdnnStatus_t initialize(vkdnnTensorView view, bool backward, float beta, bool direct = false)
    {
        target = backward ? view.gradient : view.data;
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
}    //namespace

vkdnnStatus_t vkdnnExecuteActivation(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnActivationKind activation, float alpha, float beta, int backward)
{
    if (backward != 0 && backward != 1) { return VKDNN_STATUS_BAD_PARAM; }
    if (activation < VKDNN_HELPER_RELU || activation > VKDNN_HELPER_SOFTMAX_CHANNEL) { return VKDNN_STATUS_NOT_SUPPORTED; }
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    Output temporary;
    auto target = backward ? input.gradient : output.data;
    auto targetBytes = bytes(backward ? input : output);
    auto overlaps = [&](const void* source, size_t sourceBytes)
    {
        auto destinationAddress = reinterpret_cast<uintptr_t>(target);
        auto sourceAddress = reinterpret_cast<uintptr_t>(source);
        if (!source) { return false; }
        if (destinationAddress >= sourceAddress) { return destinationAddress - sourceAddress < sourceBytes; }
        return sourceAddress - destinationAddress < targetBytes;
    };
    bool direct = !overlaps(input.data, bytes(input));
    if (backward)
    {
        direct = direct && !overlaps(output.data, bytes(output)) && !overlaps(output.gradient, bytes(output));
    }
    VKDNN_HELPER_CHECK(temporary.initialize(backward ? input : output, backward != 0, beta, direct));
    if (activation >= VKDNN_HELPER_SOFTMAX)
    {
        auto mode = activation == VKDNN_HELPER_SOFTMAX_CHANNEL ? VKDNN_SOFTMAX_MODE_CHANNEL : VKDNN_SOFTMAX_MODE_INSTANCE;
        auto algorithm = VKDNN_SOFTMAX_ACCURATE;
        if (activation == VKDNN_HELPER_SOFTMAX_FAST) { algorithm = VKDNN_SOFTMAX_FAST; }
        else if (activation == VKDNN_HELPER_SOFTMAX_LOG) { algorithm = VKDNN_SOFTMAX_LOG; }
        if (backward) { VKDNN_HELPER_CHECK(vkdnnSoftmaxBackward(handle, algorithm, mode, &alpha, outputDesc.value, output.data, outputDesc.value, output.gradient, &beta, inputDesc.value, temporary.data)); }
        else
        {
            VKDNN_HELPER_CHECK(vkdnnSoftmaxForward(handle, algorithm, mode, &alpha, inputDesc.value, input.data, &beta, outputDesc.value, temporary.data));
        }
    }
    else
    {
        Activation descriptor;
        VKDNN_HELPER_CHECK(descriptor.create());
        auto mode = VKDNN_ACTIVATION_RELU;
        if (activation == VKDNN_HELPER_SIGMOID) { mode = VKDNN_ACTIVATION_SIGMOID; }
        else if (activation == VKDNN_HELPER_TANH) { mode = VKDNN_ACTIVATION_TANH; }
        VKDNN_HELPER_CHECK(vkdnnSetActivationDescriptor(descriptor.value, mode, VKDNN_NOT_PROPAGATE_NAN, 0));
        if (backward) { VKDNN_HELPER_CHECK(vkdnnActivationBackward(handle, descriptor.value, &alpha, outputDesc.value, output.data, outputDesc.value, output.gradient, inputDesc.value, input.data, &beta, inputDesc.value, temporary.data)); }
        else
        {
            VKDNN_HELPER_CHECK(vkdnnActivationForward(handle, descriptor.value, &alpha, inputDesc.value, input.data, &beta, outputDesc.value, temporary.data));
        }
    }
    return temporary.commit();
}

vkdnnStatus_t vkdnnExecutePooling(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView output,
    vkdnnPoolingMode_t mode, int windowH, int windowW, int strideH, int strideW, int padH, int padW, float alpha, float beta, int backward)
{
    if (backward != 0 && backward != 1) { return VKDNN_STATUS_BAD_PARAM; }
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    Pooling descriptor;
    VKDNN_HELPER_CHECK(descriptor.create());
    VKDNN_HELPER_CHECK(vkdnnSetPooling2dDescriptor(descriptor.value, mode, VKDNN_NOT_PROPAGATE_NAN, windowH, windowW, padH, padW, strideH, strideW));
    Output temporary;
    auto target = backward ? input : output;
    auto destination = backward ? target.gradient : target.data;
    auto overlaps = [&](const void* source, size_t sourceBytes)
    {
        auto destinationAddress = reinterpret_cast<uintptr_t>(destination);
        auto sourceAddress = reinterpret_cast<uintptr_t>(source);
        if (!source) { return false; }
        if (destinationAddress >= sourceAddress) { return destinationAddress - sourceAddress < sourceBytes; }
        return sourceAddress - destinationAddress < bytes(target);
    };
    bool direct = !overlaps(input.data, bytes(input));
    if (backward) { direct = direct && !overlaps(output.data, bytes(output)) && !overlaps(output.gradient, bytes(output)); }
    VKDNN_HELPER_CHECK(temporary.initialize(target, backward != 0, beta, direct));
    if (backward) { VKDNN_HELPER_CHECK(vkdnnPoolingBackward(handle, descriptor.value, &alpha, outputDesc.value, output.data, outputDesc.value, output.gradient, inputDesc.value, input.data, &beta, inputDesc.value, temporary.data)); }
    else
    {
        VKDNN_HELPER_CHECK(vkdnnPoolingForward(handle, descriptor.value, &alpha, inputDesc.value, input.data, &beta, outputDesc.value, temporary.data));
    }
    return temporary.commit();
}

vkdnnStatus_t vkdnnExecuteConvolution(vkdnnHandle_t handle, vkdnnTensorView input, vkdnnTensorView weights, vkdnnTensorView output,
    int strideH, int strideW, int padH, int padW, float alpha, float beta, vkdnnConvolutionDirection direction)
{
    if (direction < VKDNN_HELPER_FORWARD || direction > VKDNN_HELPER_BACKWARD_FILTER) { return VKDNN_STATUS_BAD_PARAM; }
    Tensor inputDesc, outputDesc;
    VKDNN_HELPER_CHECK(set(inputDesc, input));
    VKDNN_HELPER_CHECK(set(outputDesc, output));
    if (!bytes(weights)) { return VKDNN_STATUS_BAD_PARAM; }
    Filter filter;
    VKDNN_HELPER_CHECK(filter.create());
    Convolution convolution;
    VKDNN_HELPER_CHECK(convolution.create());
    VKDNN_HELPER_CHECK(vkdnnSetFilter4dDescriptor(filter.value, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, weights.n, weights.c, weights.h, weights.w));
    VKDNN_HELPER_CHECK(vkdnnSetConvolution2dDescriptor(convolution.value, padH, padW, strideH, strideW, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
    auto target = output;
    if (direction == VKDNN_HELPER_BACKWARD_DATA) { target = input; }
    else if (direction == VKDNN_HELPER_BACKWARD_FILTER) { target = weights; }
    Output temporary;
    auto destination = direction == VKDNN_HELPER_FORWARD ? target.data : target.gradient;
    auto overlaps = [&](const void* source, size_t sourceBytes)
    {
        auto destinationAddress = reinterpret_cast<uintptr_t>(destination);
        auto sourceAddress = reinterpret_cast<uintptr_t>(source);
        if (!source) { return false; }
        if (destinationAddress >= sourceAddress) { return destinationAddress - sourceAddress < sourceBytes; }
        return sourceAddress - destinationAddress < bytes(target);
    };
    bool direct = false;
    if (direction == VKDNN_HELPER_FORWARD) { direct = !overlaps(input.data, bytes(input)) && !overlaps(weights.data, bytes(weights)); }
    else if (direction == VKDNN_HELPER_BACKWARD_DATA) { direct = !overlaps(weights.data, bytes(weights)) && !overlaps(output.gradient, bytes(output)); }
    else
    {
        direct = !overlaps(input.data, bytes(input)) && !overlaps(output.gradient, bytes(output));
    }
    VKDNN_HELPER_CHECK(temporary.initialize(target, direction != VKDNN_HELPER_FORWARD, beta, direct));
    if (direction == VKDNN_HELPER_BACKWARD_DATA) { VKDNN_HELPER_CHECK(vkdnnConvolutionBackwardData(handle, &alpha, filter.value, weights.data, outputDesc.value, output.gradient, convolution.value, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0, &beta, inputDesc.value, temporary.data)); }
    else if (direction == VKDNN_HELPER_BACKWARD_FILTER) { VKDNN_HELPER_CHECK(vkdnnConvolutionBackwardFilter(handle, &alpha, inputDesc.value, input.data, outputDesc.value, output.gradient, convolution.value, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, filter.value, temporary.data)); }
    else
    {
        VKDNN_HELPER_CHECK(vkdnnConvolutionForward(handle, &alpha, inputDesc.value, input.data, filter.value, weights.data, convolution.value, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, nullptr, 0, &beta, outputDesc.value, temporary.data));
    }
    return temporary.commit();
}