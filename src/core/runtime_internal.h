#pragma once

#include "compute_internal.h"
#include "vkdnn/vkdnn.h"

namespace vkdnnRuntime
{
using vkdnnStatus_t = vkdnnError_t;
using vkdnnHandle_t = struct vkdnnContext*;
using vkdnnMemory_t = struct vkdnnMemory*;
constexpr auto VKDNN_STATUS_SUCCESS = vkdnnSuccess;
constexpr auto VKDNN_STATUS_NOT_IMPLEMENTED = vkdnnErrorNotImplemented;
constexpr auto VKDNN_STATUS_INVALID_VALUE = vkdnnErrorInvalidValue;
constexpr auto VKDNN_STATUS_NO_DEVICE = vkdnnErrorNoDevice;
constexpr auto VKDNN_STATUS_NOT_SUPPORTED = vkdnnErrorNotSupported;
constexpr auto VKDNN_STATUS_OUT_OF_MEMORY = vkdnnErrorMemoryAllocation;
constexpr auto VKDNN_STATUS_DEVICE_LOST = vkdnnErrorDeviceLost;
constexpr auto VKDNN_STATUS_INTERNAL_ERROR = vkdnnErrorUnknown;

vkdnnStatus_t createContext(vkdnnHandle_t*, int);
vkdnnStatus_t destroyContext(vkdnnHandle_t);
vkdnnStatus_t synchronizeContext(vkdnnHandle_t);
vkdnnError_t setExecutionMode(vkdnnHandle_t, vkdnnExecutionMode);
vkdnnStatus_t allocateMemory(vkdnnHandle_t, vkdnnMemory_t*, size_t);
vkdnnStatus_t freeMemory(vkdnnHandle_t, vkdnnMemory_t);
vkdnnStatus_t copyHtoD(vkdnnHandle_t, vkdnnMemory_t, size_t, const void*, size_t);
vkdnnStatus_t copyDtoH(vkdnnHandle_t, void*, vkdnnMemory_t, size_t, size_t);
vkdnnStatus_t copyDtoD(vkdnnHandle_t, vkdnnMemory_t, size_t, vkdnnMemory_t, size_t, size_t);
vkdnnStatus_t setMemory(vkdnnHandle_t, vkdnnMemory_t, size_t, int, size_t);
vkdnnError_t computeBuffers(vkdnnHandle_t, vkdnnMemory_t, vkdnnMemory_t, const ComputeParams&);
vkdnnError_t activationBackwardBuffers(vkdnnHandle_t, vkdnnMemory_t, vkdnnMemory_t, vkdnnMemory_t, vkdnnMemory_t, const ActivationBackwardParams&);
vkdnnError_t gemmBuffers(vkdnnHandle_t, vkdnnMemory_t, vkdnnMemory_t, vkdnnMemory_t, const GemmParams&);
vkdnnError_t reduceBuffers(vkdnnHandle_t, vkdnnMemory_t, vkdnnMemory_t, float*, const GemmParams&);
vkdnnError_t convolutionBuffers(vkdnnHandle_t, vkdnnMemory_t, vkdnnMemory_t, vkdnnMemory_t, const ConvolutionParams&);
}    //namespace vkdnnRuntime
