#include "runtime_internal.h"

#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

using namespace vkdnnRuntime;

namespace
{
struct Allocation
{
    vkdnnHandle_t context;
    vkdnnMemory_t memory;
    size_t size;
    size_t reserved;
    int device;
};

struct Runtime
{
    std::mutex mutex;
    std::map<int, vkdnnHandle_t> contexts;
    std::map<uintptr_t, Allocation> allocations;
};

Runtime& runtime()
{
    static Runtime state;
    return state;
}

thread_local int currentDevice = 0;

template <typename Function>
vkdnnError_t guarded(Function function)
{
    try
    {
        auto& state = runtime();
        std::lock_guard<std::mutex> lock(state.mutex);
        return function(state);
    } catch (const std::bad_alloc&) { return vkdnnErrorMemoryAllocation; } catch (const std::length_error&)
    {
        return vkdnnErrorMemoryAllocation;
    } catch (const std::system_error&) { return vkdnnErrorUnknown; }
}

vkdnnError_t validateDevice(int device)
{
    if (device < 0) { return vkdnnErrorInvalidDevice; }
    int count = 0;
    auto status = vkdnnGetDeviceCount(&count);
    if (status != vkdnnSuccess) { return status; }
    if (!count) { return vkdnnErrorNoDevice; }
    if (device >= count) { return vkdnnErrorInvalidDevice; }
    return vkdnnSuccess;
}

vkdnnError_t contextFor(Runtime& state, vkdnnHandle_t& context)
{
    auto found = state.contexts.find(currentDevice);
    if (found != state.contexts.end())
    {
        context = found->second;
        return vkdnnSuccess;
    }
    auto status = validateDevice(currentDevice);
    if (status != vkdnnSuccess) { return status; }
    // Insert before allocating Vulkan resources so map allocation failure cannot leak them.
    auto entry = state.contexts.emplace(currentDevice, nullptr).first;
    status = createContext(&entry->second, currentDevice);
    if (status != vkdnnSuccess)
    {
        state.contexts.erase(entry);
        return status;
    }
    context = entry->second;
    return vkdnnSuccess;
}

void* reserveAddress(size_t size)
{
#ifdef _WIN32
    return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
#else
    void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address == MAP_FAILED) { return nullptr; }
    return address;
#endif
}

bool releaseAddress(void* address, size_t size)
{
#ifdef _WIN32
    (void)size;
    return VirtualFree(address, 0, MEM_RELEASE) != 0;
#else
    return munmap(address, size) == 0;
#endif
}

struct Pointer
{
    Allocation* allocation = nullptr;
    size_t offset = 0;
};

Pointer locate(Runtime& state, const void* pointer)
{
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    auto entry = state.allocations.upper_bound(address);
    if (entry == state.allocations.begin()) { return {}; }
    --entry;
    size_t offset = address - entry->first;
    // Include the guard area so out-of-range device pointers are not treated as host pointers.
    if (offset < entry->second.reserved) { return { &entry->second, offset }; }
    return {};
}

bool fits(Pointer pointer, size_t bytes)
{
    return pointer.allocation && pointer.offset <= pointer.allocation->size
        && bytes <= pointer.allocation->size - pointer.offset;
}
}    //namespace

vkdnnError_t vkdnnSetDevice(int device)
{
    return guarded([&](Runtime&)
        {
            auto status = validateDevice(device);
            if (status == vkdnnSuccess) { currentDevice = device; }
            return status;
        });
}

vkdnnError_t runtimeCompute(int device, const void* x, size_t xBytes, void* y, size_t yBytes, ComputeParams params)
{
    return guarded([&](Runtime& state)
        {
            if (device != currentDevice) { return vkdnnErrorInvalidDevice; }
            auto source = locate(state, x);
            auto destination = locate(state, y);
            if (!xBytes) { source = destination; }
            if (!fits(source, xBytes) || !fits(destination, yBytes)) { return vkdnnErrorInvalidValue; }
            if (source.allocation->device != device || destination.allocation->device != device) { return vkdnnErrorInvalidDevice; }
            if (source.offset % 4 || destination.offset % 4) { return vkdnnErrorInvalidValue; }
            if ((source.offset + xBytes) / 4 > UINT32_MAX || (destination.offset + yBytes) / 4 > UINT32_MAX) { return vkdnnErrorNotSupported; }
            if (xBytes && source.allocation == destination.allocation)
            {
                if (source.offset < destination.offset + yBytes && destination.offset < source.offset + xBytes)
                {
                    return vkdnnErrorNotSupported;
                }
            }
            params.xOffset = static_cast<uint32_t>(source.offset / 4);
            params.yOffset = static_cast<uint32_t>(destination.offset / 4);
            return computeBuffers(source.allocation->context, source.allocation->memory, destination.allocation->memory, params);
        });
}

vkdnnError_t runtimeActivationBackward(int device, const void* x, const void* y, const void* dy,
    void* dx, size_t bytes, ActivationBackwardParams params)
{
    return guarded([&](Runtime& state)
        {
            if (device != currentDevice) { return vkdnnErrorInvalidDevice; }
            Pointer output = locate(state, dx);
            if (!fits(output, bytes) || output.offset % 4) { return vkdnnErrorInvalidValue; }
            if (output.allocation->device != device) { return vkdnnErrorInvalidDevice; }
            if ((output.offset + bytes) / 4 > UINT32_MAX) { return vkdnnErrorNotSupported; }
            const void* addresses[] = { x, y, dy };
            Pointer inputs[] = { output, output, output };
            if (params.alpha != 0)
            {
                for (int i = 0; i < 3; ++i)
                {
                    inputs[i] = locate(state, addresses[i]);
                    if (!fits(inputs[i], bytes) || inputs[i].offset % 4) { return vkdnnErrorInvalidValue; }
                    if (inputs[i].allocation->device != device) { return vkdnnErrorInvalidDevice; }
                    if ((inputs[i].offset + bytes) / 4 > UINT32_MAX) { return vkdnnErrorNotSupported; }
                    if (inputs[i].allocation == output.allocation && inputs[i].offset < output.offset + bytes
                        && output.offset < inputs[i].offset + bytes) { return vkdnnErrorNotSupported; }
                }
            }
            params.xOffset = static_cast<uint32_t>(inputs[0].offset / 4);
            params.yOffset = static_cast<uint32_t>(inputs[1].offset / 4);
            params.dyOffset = static_cast<uint32_t>(inputs[2].offset / 4);
            params.dxOffset = static_cast<uint32_t>(output.offset / 4);
            return activationBackwardBuffers(output.allocation->context, inputs[0].allocation->memory,
                inputs[1].allocation->memory, inputs[2].allocation->memory, output.allocation->memory, params);
        });
}

vkdnnError_t runtimeHostScalar(const float* pointer, float& value)
{
    return guarded([&](Runtime& state)
        {
            if (!pointer || locate(state, pointer).allocation) { return vkdnnErrorInvalidValue; }
            value = *pointer;
            return vkdnnSuccess;
        });
}

template <class Params, class Dispatch>
vkdnnError_t runtimeProduct(int device, const void* a, size_t aBytes, const void* b, size_t bBytes,
    void* c, size_t cBytes, Params params, Dispatch dispatch)
{
    return guarded([&](Runtime& state)
        {
            if (device != currentDevice) { return vkdnnErrorInvalidDevice; }
            Pointer output = locate(state, c);
            if (!fits(output, cBytes) || output.offset % 4) { return vkdnnErrorInvalidValue; }
            if (output.allocation->device != device) { return vkdnnErrorInvalidDevice; }
            Pointer inputs[2] = { output, output };
            const void* addresses[2] = { a, b };
            size_t sizes[2] = { aBytes, bBytes };
            for (int i = 0; i < 2; ++i)
            {
                if (!sizes[i]) { continue; }
                inputs[i] = locate(state, addresses[i]);
                if (!fits(inputs[i], sizes[i]) || inputs[i].offset % 4) { return vkdnnErrorInvalidValue; }
                if (inputs[i].allocation->device != device) { return vkdnnErrorInvalidDevice; }
                if (inputs[i].allocation == output.allocation && inputs[i].offset < output.offset + cBytes
                    && output.offset < inputs[i].offset + sizes[i]) { return vkdnnErrorNotSupported; }
            }
            if ((output.offset + cBytes) / 4 > UINT32_MAX
                || (inputs[0].offset + aBytes) / 4 > UINT32_MAX
                || (inputs[1].offset + bBytes) / 4 > UINT32_MAX) { return vkdnnErrorNotSupported; }
            params.aOffset = static_cast<uint32_t>(inputs[0].offset / 4);
            params.bOffset = static_cast<uint32_t>(inputs[1].offset / 4);
            params.cOffset = static_cast<uint32_t>(output.offset / 4);
            return dispatch(output.allocation->context, inputs[0].allocation->memory, inputs[1].allocation->memory,
                output.allocation->memory, params);
        });
}

vkdnnError_t runtimeGemm(int device, const void* a, size_t aBytes, const void* b, size_t bBytes,
    void* c, size_t cBytes, GemmParams params)
{
    return runtimeProduct(device, a, aBytes, b, bBytes, c, cBytes, params, gemmBuffers);
}

vkdnnError_t runtimeConvolution(int device, const void* x, size_t xBytes, const void* w, size_t wBytes,
    void* y, size_t yBytes, ConvolutionParams params)
{
    return runtimeProduct(device, x, xBytes, w, wBytes, y, yBytes, params, convolutionBuffers);
}

vkdnnError_t vkdnnGetDevice(int* device)
{
    if (!device) { return vkdnnErrorInvalidValue; }
    return guarded([&](Runtime&)
        {
            auto status = validateDevice(currentDevice);
            if (status == vkdnnSuccess) { *device = currentDevice; }
            return status;
        });
}

vkdnnError_t vkdnnDeviceSynchronize(void)
{
    return guarded([](Runtime& state)
        {
            vkdnnHandle_t context = nullptr;
            auto status = contextFor(state, context);
            if (status != vkdnnSuccess) { return status; }
            return synchronizeContext(context);
        });
}

vkdnnError_t vkdnnDeviceReset(void)
{
    return guarded([](Runtime& state)
        {
            auto status = validateDevice(currentDevice);
            if (status != vkdnnSuccess) { return status; }
            auto context = state.contexts.find(currentDevice);
            if (context == state.contexts.end()) { return vkdnnSuccess; }
            status = synchronizeContext(context->second);
            if (status != vkdnnSuccess && status != vkdnnErrorDeviceLost) { return status; }
            for (auto entry = state.allocations.begin(); entry != state.allocations.end();)
            {
                if (entry->second.device != currentDevice)
                {
                    ++entry;
                    continue;
                }
                if (!releaseAddress(reinterpret_cast<void*>(entry->first), entry->second.reserved))
                {
                    return vkdnnErrorUnknown;
                }
                auto freed = freeMemory(context->second, entry->second.memory);
                if (freed != vkdnnSuccess) { return freed; }
                entry = state.allocations.erase(entry);
            }
            auto destroyed = destroyContext(context->second);
            if (destroyed == vkdnnSuccess || destroyed == vkdnnErrorDeviceLost) { state.contexts.erase(context); }
            if (destroyed != vkdnnSuccess) { return destroyed; }
            return status;
        });
}

vkdnnError_t vkdnnMalloc(void** devPtr, size_t size)
{
    if (!devPtr || size > std::numeric_limits<size_t>::max() - 65536) { return vkdnnErrorInvalidValue; }
    if (!size)
    {
        *devPtr = nullptr;
        return vkdnnSuccess;
    }
    return guarded([&](Runtime& state)
        {
            vkdnnHandle_t context = nullptr;
            auto status = contextFor(state, context);
            if (status != vkdnnSuccess) { return status; }
            size_t reserved = size + 65536;
            void* address = reserveAddress(reserved);
            if (!address) { return vkdnnErrorMemoryAllocation; }
            // Allocate the registry node before allocating GPU memory.
            decltype(state.allocations)::iterator entry;
            try
            {
                entry = state.allocations.emplace(reinterpret_cast<uintptr_t>(address),
                                             Allocation{ context, nullptr, size, reserved, currentDevice })
                            .first;
            } catch (const std::bad_alloc&)
            {
                if (!releaseAddress(address, reserved)) { return vkdnnErrorUnknown; }
                return vkdnnErrorMemoryAllocation;
            }
            status = allocateMemory(context, &entry->second.memory, size);
            if (status != vkdnnSuccess)
            {
                state.allocations.erase(entry);
                if (!releaseAddress(address, reserved)) { return vkdnnErrorUnknown; }
                return status;
            }
            *devPtr = address;
            return vkdnnSuccess;
        });
}

vkdnnError_t vkdnnFree(void* devPtr)
{
    if (!devPtr) { return vkdnnSuccess; }
    return guarded([&](Runtime& state)
        {
            auto entry = state.allocations.find(reinterpret_cast<uintptr_t>(devPtr));
            if (entry == state.allocations.end()) { return vkdnnErrorInvalidValue; }
            if (!releaseAddress(devPtr, entry->second.reserved)) { return vkdnnErrorUnknown; }
            auto status = freeMemory(entry->second.context, entry->second.memory);
            if (status == vkdnnSuccess) { state.allocations.erase(entry); }
            return status;
        });
}

vkdnnError_t vkdnnMemcpy(void* dst, const void* src, size_t count, enum vkdnnMemcpyKind kind)
{
    if (kind < vkdnnMemcpyHostToHost || kind > vkdnnMemcpyDefault) { return vkdnnErrorInvalidMemcpyDirection; }
    if (!count) { return vkdnnSuccess; }
    if (!dst || !src) { return vkdnnErrorInvalidValue; }
    return guarded([&](Runtime& state)
        {
            auto destination = locate(state, dst);
            auto source = locate(state, src);
            bool dstDevice = destination.allocation != nullptr;
            bool srcDevice = source.allocation != nullptr;
            if ((dstDevice && !fits(destination, count)) || (srcDevice && !fits(source, count)))
            {
                return vkdnnErrorInvalidValue;
            }
            vkdnnMemcpyKind inferred = vkdnnMemcpyHostToHost;
            if (dstDevice && srcDevice) { inferred = vkdnnMemcpyDeviceToDevice; }
            else if (dstDevice) { inferred = vkdnnMemcpyHostToDevice; }
            else if (srcDevice) { inferred = vkdnnMemcpyDeviceToHost; }
            if (kind != vkdnnMemcpyDefault && kind != inferred) { return vkdnnErrorInvalidMemcpyDirection; }
            if (inferred == vkdnnMemcpyHostToHost)
            {
                uintptr_t d = reinterpret_cast<uintptr_t>(dst);
                uintptr_t s = reinterpret_cast<uintptr_t>(src);
                if (count > std::numeric_limits<uintptr_t>::max() - d || count > std::numeric_limits<uintptr_t>::max() - s)
                {
                    return vkdnnErrorInvalidValue;
                }
                if (d < s + count && s < d + count) { return vkdnnErrorInvalidValue; }
                std::memcpy(dst, src, count);
                return vkdnnSuccess;
            }
            if (inferred == vkdnnMemcpyHostToDevice)
            {
                return copyHtoD(destination.allocation->context, destination.allocation->memory, destination.offset, src, count);
            }
            if (inferred == vkdnnMemcpyDeviceToHost)
            {
                return copyDtoH(source.allocation->context, dst, source.allocation->memory, source.offset, count);
            }
            if (destination.allocation->device != source.allocation->device) { return vkdnnErrorNotSupported; }
            return copyDtoD(destination.allocation->context, destination.allocation->memory, destination.offset,
                source.allocation->memory, source.offset, count);
        });
}

vkdnnError_t vkdnnMemset(void* devPtr, int value, size_t count)
{
    if (!count) { return vkdnnSuccess; }
    if (!devPtr) { return vkdnnErrorInvalidValue; }
    return guarded([&](Runtime& state)
        {
            auto pointer = locate(state, devPtr);
            if (!fits(pointer, count)) { return vkdnnErrorInvalidValue; }
            return setMemory(pointer.allocation->context, pointer.allocation->memory, pointer.offset, value, count);
        });
}
