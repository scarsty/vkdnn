#ifndef VKDNN_VKDNN_H
#define VKDNN_VKDNN_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(VKDNN_BUILDING_LIBRARY)
#define VKDNN_API __declspec(dllexport)
#else
#define VKDNN_API __declspec(dllimport)
#endif
#else
#define VKDNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/* Names and signatures follow CUDA Runtime; numeric error values are not ABI compatible. */
typedef enum vkdnnError
{
    vkdnnSuccess = 0,
    vkdnnErrorInvalidValue = 1,
    vkdnnErrorMemoryAllocation = 2,
    vkdnnErrorInitializationError = 3,
    vkdnnErrorInvalidDevice = 4,
    vkdnnErrorInvalidMemcpyDirection = 5,
    vkdnnErrorNoDevice = 6,
    vkdnnErrorNotSupported = 7,
    vkdnnErrorDeviceLost = 8,
    vkdnnErrorNotImplemented = 9,
    vkdnnErrorUnknown = 999
} vkdnnError_t;

typedef enum vkdnnMemcpyKind
{
    vkdnnMemcpyHostToHost = 0,
    vkdnnMemcpyHostToDevice = 1,
    vkdnnMemcpyDeviceToHost = 2,
    vkdnnMemcpyDeviceToDevice = 3,
    vkdnnMemcpyDefault = 4
} vkdnnMemcpyKind;

typedef enum vkdnnDeviceType_t
{
    VKDNN_DEVICE_TYPE_OTHER = 0,
    VKDNN_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VKDNN_DEVICE_TYPE_DISCRETE_GPU = 2,
    VKDNN_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VKDNN_DEVICE_TYPE_CPU = 4
} vkdnnDeviceType_t;

/* Vulkan-specific properties are retained rather than inventing CUDA hardware limits. */
typedef struct vkdnnDeviceProp
{
    char name[256];
    uint32_t vendorId;
    uint32_t deviceId;
    vkdnnDeviceType_t deviceType;
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint64_t totalGlobalMem;
    uint64_t maxStorageBufferRange;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t maxComputeWorkGroupCount[3];
} vkdnnDeviceProp;

VKDNN_API const char* vkdnnGetErrorString(vkdnnError_t error);
/* Queries include software Vulkan 1.2 compute devices and preserve outputs on error. */
VKDNN_API vkdnnError_t vkdnnGetDeviceCount(int* count);
VKDNN_API vkdnnError_t vkdnnGetDeviceProperties(struct vkdnnDeviceProp* prop, int device);
/* Current device is thread-local, initially zero. Contexts are created lazily and
 * shared across threads. Calls are internally serialized. */
VKDNN_API vkdnnError_t vkdnnSetDevice(int device);
VKDNN_API vkdnnError_t vkdnnGetDevice(int* device);
VKDNN_API vkdnnError_t vkdnnDeviceSynchronize(void);
/* Frees all allocations on the current device. Caller must coordinate other
 * threads; all old pointers become invalid. Reset before unloading the library. */
VKDNN_API vkdnnError_t vkdnnDeviceReset(void);
/* Device pointers reserve inaccessible host virtual addresses, not host memory.
 * Byte pointer arithmetic is supported; CPU dereferencing is forbidden.
 * Zero-sized malloc succeeds with NULL; free(NULL) succeeds. Free needs the base
 * pointer, and operates on its owning device without changing the current device. */
VKDNN_API vkdnnError_t vkdnnMalloc(void** devPtr, size_t size);
VKDNN_API vkdnnError_t vkdnnFree(void* devPtr);
/* Copies are synchronous. Default infers direction from registered allocations.
 * Explicit directions must match pointers. Device-to-device copies across devices
 * are unsupported; overlapping ranges are invalid. Zero count is a no-op.
 * Failed validation preserves destination; runtime failures may partially write. */
VKDNN_API vkdnnError_t vkdnnMemcpy(void* dst, const void* src, size_t count, enum vkdnnMemcpyKind kind);
VKDNN_API vkdnnError_t vkdnnMemset(void* devPtr, int value, size_t count);

#ifdef __cplusplus
}
#endif
#endif
