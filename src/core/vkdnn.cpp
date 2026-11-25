#include "runtime_internal.h"

#include <vulkan/vulkan.h>

#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

static const uint32_t forwardSpirv[] =
#include "forward_spv.inc"
    ;
static const uint32_t backwardSpirv[] =
#include "backward_spv.inc"
    ;
static const uint32_t gemmSpirv[] =
#include "gemm_spv.inc"
    ;
static const uint32_t convolutionSpirv[] =
#include "convolution_spv.inc"
    ;

namespace vkdnnRuntime
{
struct vkdnnMemory
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize capacity = 0;
};

struct vkdnnContext
{
    struct ComputeResources
    {
        VkDevice device;
        const uint32_t* code;
        size_t codeSize;
        uint32_t bufferCount, paramSize;
        VkShaderModule shader = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        ~ComputeResources()
        {
            if (pool) { vkDestroyDescriptorPool(device, pool, nullptr); }
            if (pipeline) { vkDestroyPipeline(device, pipeline, nullptr); }
            if (layout) { vkDestroyPipelineLayout(device, layout, nullptr); }
            if (setLayout) { vkDestroyDescriptorSetLayout(device, setLayout, nullptr); }
            if (shader) { vkDestroyShaderModule(device, shader, nullptr); }
        }
    };
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    int index = -1;
    std::unordered_set<vkdnnMemory_t> allocations;
    // Runtime calls serialize and wait for the fence before reusing descriptors.
    // Contexts are explicitly reset, not deleted by DLL process-detach cleanup.
    std::vector<std::unique_ptr<ComputeResources>> compute;

    ~vkdnnContext()
    {
        if (device)
        {
            compute.clear();
            if (fence) { vkDestroyFence(device, fence, nullptr); }
            if (pool) { vkDestroyCommandPool(device, pool, nullptr); }
            vkDestroyDevice(device, nullptr);
        }
        if (instance) { vkDestroyInstance(instance, nullptr); }
    }
};

}    //namespace vkdnnRuntime
using namespace vkdnnRuntime;

static vkdnnError_t dispatchBuffers(vkdnnHandle_t context, vkdnnMemory_t* memory, uint32_t bufferCount,
    const uint32_t* code, size_t codeSize, const void* params, uint32_t paramSize, uint32_t groups, uint32_t groupsY)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physical, &properties);
    if (paramSize > properties.limits.maxPushConstantsSize
        || groups > properties.limits.maxComputeWorkGroupCount[0]
        || groupsY > properties.limits.maxComputeWorkGroupCount[1])
    {
        return vkdnnErrorNotSupported;
    }
    for (uint32_t i = 0; i < bufferCount; ++i)
    {
        if (memory[i]->size > properties.limits.maxStorageBufferRange) { return vkdnnErrorNotSupported; }
    }
    auto status = [](VkResult result)
    {
        if (result == VK_SUCCESS)
        {
            return vkdnnSuccess;
        }
        if (result == VK_ERROR_DEVICE_LOST)
        {
            return vkdnnErrorDeviceLost;
        }
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY)
        {
            return vkdnnErrorMemoryAllocation;
        }
        return vkdnnErrorUnknown;
    };
    vkdnnContext::ComputeResources* cached = nullptr;
    for (const auto& entry : context->compute)
    {
        if (entry->code == code && entry->codeSize == codeSize
            && entry->bufferCount == bufferCount && entry->paramSize == paramSize)
        {
            cached = entry.get();
            break;
        }
    }
    VkResult result = VK_SUCCESS;
    if (!cached)
    {
        auto pending = std::make_unique<vkdnnContext::ComputeResources>();
        auto& r = *pending;
        r.device = context->device;
        r.code = code;
        r.codeSize = codeSize;
        r.bufferCount = bufferCount;
        r.paramSize = paramSize;
        VkShaderModuleCreateInfo shader{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        shader.codeSize = codeSize;
        shader.pCode = code;
        result = vkCreateShaderModule(r.device, &shader, nullptr, &r.shader);
        if (result != VK_SUCCESS)
        {
            return status(result);
        }
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t i = 0; i < bufferCount; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo setInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setInfo.bindingCount = bufferCount;
        setInfo.pBindings = bindings;
        result = vkCreateDescriptorSetLayout(r.device, &setInfo, nullptr, &r.setLayout);
        if (result != VK_SUCCESS)
        {
            return status(result);
        }
        VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, paramSize };
        VkPipelineLayoutCreateInfo layout{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &r.setLayout;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        result = vkCreatePipelineLayout(r.device, &layout, nullptr, &r.layout);
        if (result != VK_SUCCESS)
        {
            return status(result);
        }
        VkComputePipelineCreateInfo pipeline{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = r.shader;
        pipeline.stage.pName = "main";
        pipeline.layout = r.layout;
        result = vkCreateComputePipelines(r.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &r.pipeline);
        if (result != VK_SUCCESS)
        {
            return status(result);
        }
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bufferCount };
        VkDescriptorPoolCreateInfo pool{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool.maxSets = 1;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &poolSize;
        result = vkCreateDescriptorPool(r.device, &pool, nullptr, &r.pool);
        if (result != VK_SUCCESS)
        {
            return status(result);
        }
        VkDescriptorSetAllocateInfo allocate{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocate.descriptorPool = r.pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &r.setLayout;
        result = vkAllocateDescriptorSets(r.device, &allocate, &r.set);
        if (result != VK_SUCCESS)
        {
            return status(result);
        }
        cached = pending.get();
        context->compute.push_back(std::move(pending));
    }
    auto& r = *cached;
    VkDescriptorSet set = r.set;
    VkDescriptorBufferInfo buffers[4]{};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < bufferCount; ++i)
    {
        buffers[i] = { memory[i]->buffer, 0, memory[i]->size };
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(r.device, bufferCount, writes, 0, nullptr);
    result = vkResetCommandPool(r.device, context->pool, 0);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(context->command, &begin);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipeline);
    vkCmdBindDescriptorSets(context->command, VK_PIPELINE_BIND_POINT_COMPUTE, r.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(context->command, r.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, paramSize, params);
    vkCmdDispatch(context->command, groups, groupsY, 1);
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    result = vkEndCommandBuffer(context->command);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    result = vkResetFences(r.device, 1, &context->fence);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &context->command;
    result = vkQueueSubmit(context->queue, 1, &submit, context->fence);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    return status(vkWaitForFences(r.device, 1, &context->fence, VK_TRUE, UINT64_MAX));
}

vkdnnError_t vkdnnRuntime::activationBackwardBuffers(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t y,
    vkdnnMemory_t dy, vkdnnMemory_t dx, const ActivationBackwardParams& params)
{
    vkdnnMemory_t buffers[] = { x, y, dy, dx };
    return dispatchBuffers(context, buffers, 4, backwardSpirv, sizeof(backwardSpirv), &params, sizeof(params), (params.count + 63u) / 64u, 1);
}

vkdnnError_t vkdnnRuntime::computeBuffers(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t y, const ComputeParams& params)
{
    vkdnnMemory_t buffers[] = { x, y };
    uint32_t groups = (params.count + 63u) / 64u;
    uint32_t groupsY = 1;
    if (params.op == 0) { groups = (params.count + 255u) / 256u; }
    if (params.op == 2 || params.op == 3)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context->physical, &properties);
        groups = params.count;
        uint32_t limit = properties.limits.maxComputeWorkGroupCount[0];
        if (groups > limit)
        {
            groupsY = (groups - 1u) / limit + 1u;
            groups = limit;
        }
    }
    return dispatchBuffers(context, buffers, 2, forwardSpirv, sizeof(forwardSpirv), &params, sizeof(params), groups, groupsY);
}

vkdnnError_t vkdnnRuntime::convolutionBuffers(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t w, vkdnnMemory_t y, const ConvolutionParams& params)
{
    vkdnnMemory_t buffers[] = { x, w, y };
    return dispatchBuffers(context, buffers, 3, convolutionSpirv, sizeof(convolutionSpirv), &params, sizeof(params), (params.count + 63u) / 64u, 1);
}

vkdnnError_t vkdnnRuntime::gemmBuffers(vkdnnHandle_t context, vkdnnMemory_t a, vkdnnMemory_t b, vkdnnMemory_t c, const GemmParams& params)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physical, &properties);
    if (properties.limits.maxComputeWorkGroupInvocations < 256
        || properties.limits.maxComputeWorkGroupSize[0] < 16 || properties.limits.maxComputeWorkGroupSize[1] < 16
        || properties.limits.maxComputeSharedMemorySize < 2048)
    {
        return vkdnnErrorNotSupported;
    }
    vkdnnMemory_t buffers[] = { a, b, c };
    return dispatchBuffers(context, buffers, 3, gemmSpirv, sizeof(gemmSpirv), &params, sizeof(params), (params.m + 15u) / 16u, (params.n + 15u) / 16u);
}

namespace
{
vkdnnStatus_t resultStatus(VkResult result)
{
    switch (result)
    {
    case VK_SUCCESS: return VKDNN_STATUS_SUCCESS;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return VKDNN_STATUS_OUT_OF_MEMORY;
    case VK_ERROR_INCOMPATIBLE_DRIVER: return VKDNN_STATUS_NOT_SUPPORTED;
    case VK_ERROR_DEVICE_LOST: return VKDNN_STATUS_DEVICE_LOST;
    default: return VKDNN_STATUS_INTERNAL_ERROR;
    }
}

vkdnnDeviceType_t deviceType(VkPhysicalDeviceType type)
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return VKDNN_DEVICE_TYPE_INTEGRATED_GPU;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return VKDNN_DEVICE_TYPE_DISCRETE_GPU;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return VKDNN_DEVICE_TYPE_VIRTUAL_GPU;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return VKDNN_DEVICE_TYPE_CPU;
    default: return VKDNN_DEVICE_TYPE_OTHER;
    }
}

struct DeviceRegistry
{
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<vkdnnDeviceProp> devices;
    std::vector<std::array<uint8_t, VK_UUID_SIZE>> uuids;
    std::mutex mutex;
    bool initialized = false;

    void reset()
    {
        if (instance) { vkDestroyInstance(instance, nullptr); }
        instance = VK_NULL_HANDLE;
    }

    vkdnnStatus_t initialize()
    {
        if (initialized) { return VKDNN_STATUS_SUCCESS; }
        reset();
        // Release Vulkan before returning, never from DLL process-detach cleanup.
        struct Cleanup
        {
            DeviceRegistry& state;
            ~Cleanup() { state.reset(); }
        } cleanup{ *this };
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "vkdnn";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        VkResult result = vkCreateInstance(&info, nullptr, &instance);
        if (result != VK_SUCCESS) { return resultStatus(result); }

        std::vector<VkPhysicalDevice> physicalDevices;
        // Bound retries so a changing driver inventory cannot block indefinitely.
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            uint32_t count = 0;
            result = vkEnumeratePhysicalDevices(instance, &count, nullptr);
            if (result != VK_SUCCESS) { return resultStatus(result); }
            physicalDevices.resize(count);
            if (count == 0) { break; }
            result = vkEnumeratePhysicalDevices(instance, &count, physicalDevices.data());
            if (result == VK_INCOMPLETE) { continue; }
            if (result != VK_SUCCESS) { return resultStatus(result); }
            physicalDevices.resize(count);
            break;
        }
        if (result == VK_INCOMPLETE) { return VKDNN_STATUS_INTERNAL_ERROR; }

        std::vector<vkdnnDeviceProp> discovered;
        std::vector<std::array<uint8_t, VK_UUID_SIZE>> discoveredUuids;
        for (VkPhysicalDevice device : physicalDevices)
        {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(device, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_2) { continue; }
            uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(count);
            if (count == 0) { continue; }
            vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queues.data());
            bool compute = false;
            for (uint32_t i = 0; i < count; ++i)
            {
                if (queues[i].queueCount > 0 && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
                {
                    compute = true;
                    break;
                }
            }
            if (!compute) { continue; }

            VkPhysicalDeviceMemoryProperties memory{};
            vkGetPhysicalDeviceMemoryProperties(device, &memory);
            vkdnnDeviceProp output{};
            std::memcpy(output.name, properties.deviceName, sizeof(output.name));
            output.name[sizeof(output.name) - 1] = '\0';
            output.vendorId = properties.vendorID;
            output.deviceId = properties.deviceID;
            output.deviceType = deviceType(properties.deviceType);
            output.apiVersion = properties.apiVersion;
            output.driverVersion = properties.driverVersion;
            for (uint32_t i = 0; i < memory.memoryHeapCount; ++i)
            {
                if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                {
                    if (memory.memoryHeaps[i].size > std::numeric_limits<uint64_t>::max() - output.totalGlobalMem)
                    {
                        return VKDNN_STATUS_INTERNAL_ERROR;
                    }
                    output.totalGlobalMem += memory.memoryHeaps[i].size;
                }
            }
            output.maxStorageBufferRange = properties.limits.maxStorageBufferRange;
            output.maxComputeSharedMemorySize = properties.limits.maxComputeSharedMemorySize;
            output.maxComputeWorkGroupInvocations = properties.limits.maxComputeWorkGroupInvocations;
            for (int i = 0; i < 3; ++i)
            {
                output.maxComputeWorkGroupSize[i] = properties.limits.maxComputeWorkGroupSize[i];
                output.maxComputeWorkGroupCount[i] = properties.limits.maxComputeWorkGroupCount[i];
            }
            discovered.push_back(output);
            VkPhysicalDeviceIDProperties id{};
            id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 extended{};
            extended.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            extended.pNext = &id;
            vkGetPhysicalDeviceProperties2(device, &extended);
            std::array<uint8_t, VK_UUID_SIZE> uuid{};
            std::memcpy(uuid.data(), id.deviceUUID, uuid.size());
            discoveredUuids.push_back(uuid);
        }
        if (discovered.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return VKDNN_STATUS_INTERNAL_ERROR;
        }
        devices = std::move(discovered);
        uuids = std::move(discoveredUuids);
        initialized = true;
        return VKDNN_STATUS_SUCCESS;
    }
};

DeviceRegistry& registry()
{
    static DeviceRegistry value;
    return value;
}

void releaseBuffer(vkdnnHandle_t handle, vkdnnMemory& memory)
{
    if (memory.buffer) { vkDestroyBuffer(handle->device, memory.buffer, nullptr); }
    if (memory.allocation) { vkFreeMemory(handle->device, memory.allocation, nullptr); }
    memory = {};
}

vkdnnStatus_t allocateBuffer(vkdnnHandle_t handle, VkDeviceSize bytes,
    VkMemoryPropertyFlags flags, vkdnnMemory& memory)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) { info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; }
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(handle->device, &info, nullptr, &memory.buffer);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(handle->device, memory.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(handle->physical, &properties);
    uint32_t type = properties.memoryTypeCount;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
    {
        if ((requirements.memoryTypeBits & (1u << i))
            && (properties.memoryTypes[i].propertyFlags & flags) == flags)
        {
            type = i;
            break;
        }
    }
    if (type == properties.memoryTypeCount)
    {
        releaseBuffer(handle, memory);
        return VKDNN_STATUS_NOT_SUPPORTED;
    }
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    result = vkAllocateMemory(handle->device, &allocation, nullptr, &memory.allocation);
    if (result == VK_SUCCESS) { result = vkBindBufferMemory(handle->device, memory.buffer, memory.allocation, 0); }
    if (result != VK_SUCCESS)
    {
        releaseBuffer(handle, memory);
        return resultStatus(result);
    }
    memory.size = bytes;
    memory.capacity = requirements.size;
    return VKDNN_STATUS_SUCCESS;
}

bool validRange(vkdnnHandle_t handle, vkdnnMemory_t memory, size_t offset, size_t bytes)
{
    return handle && memory && handle->allocations.count(memory)
        && offset <= memory->size && bytes <= memory->size - offset;
}

vkdnnStatus_t copyBuffer(vkdnnHandle_t handle, VkBuffer dst, VkDeviceSize dstOffset,
    VkBuffer src, VkDeviceSize srcOffset, VkDeviceSize bytes)
{
    VkResult result = vkResetCommandPool(handle->device, handle->pool, 0);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(handle->command, &begin);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(handle->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    VkBufferCopy region{ srcOffset, dstOffset, bytes };
    vkCmdCopyBuffer(handle->command, src, dst, 1, &region);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(handle->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    result = vkEndCommandBuffer(handle->command);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    result = vkResetFences(handle->device, 1, &handle->fence);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &handle->command;
    result = vkQueueSubmit(handle->queue, 1, &submit, handle->fence);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    return resultStatus(vkWaitForFences(handle->device, 1, &handle->fence, VK_TRUE, UINT64_MAX));
}

// A padded staging range preserves neighboring bytes for unaligned transfers.
vkdnnStatus_t transfer(vkdnnHandle_t handle, vkdnnMemory_t memory, size_t offset,
    void* host, size_t bytes, bool upload)
{
    if (!bytes) { return VKDNN_STATUS_SUCCESS; }
    VkDeviceSize start = offset & ~VkDeviceSize(3);
    VkDeviceSize end = (offset + bytes + VkDeviceSize(3)) & ~VkDeviceSize(3);
    vkdnnMemory staging;
    vkdnnStatus_t status = allocateBuffer(handle, end - start, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, staging);
    if (status != VKDNN_STATUS_SUCCESS) { return status; }
    struct Cleanup
    {
        vkdnnHandle_t handle;
        vkdnnMemory& memory;
        ~Cleanup() { releaseBuffer(handle, memory); }
    } cleanup{ handle, staging };
    if (!upload || start != offset || end != offset + bytes)
    {
        status = copyBuffer(handle, staging.buffer, 0, memory->buffer, start, end - start);
        if (status != VKDNN_STATUS_SUCCESS) { return status; }
    }
    void* mapped = nullptr;
    VkResult result = vkMapMemory(handle->device, staging.allocation, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = staging.allocation;
    range.size = VK_WHOLE_SIZE;
    result = vkInvalidateMappedMemoryRanges(handle->device, 1, &range);
    if (result == VK_SUCCESS)
    {
        auto* data = static_cast<unsigned char*>(mapped) + (offset - start);
        if (upload)
        {
            std::memcpy(data, host, bytes);
            result = vkFlushMappedMemoryRanges(handle->device, 1, &range);
        }
        else
        {
            std::memcpy(host, data, bytes);
        }
    }
    vkUnmapMemory(handle->device, staging.allocation);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    if (upload) { return copyBuffer(handle, memory->buffer, start, staging.buffer, 0, end - start); }
    return VKDNN_STATUS_SUCCESS;
}
}    //namespace

const char* vkdnnGetErrorString(vkdnnStatus_t status)
{
    switch (status)
    {
    case VKDNN_STATUS_SUCCESS: return "Success";
    case VKDNN_STATUS_NOT_IMPLEMENTED: return "Not implemented";
    case VKDNN_STATUS_INVALID_VALUE: return "Invalid value";
    case VKDNN_STATUS_NO_DEVICE: return "No device";
    case VKDNN_STATUS_NOT_SUPPORTED: return "Not supported";
    case VKDNN_STATUS_OUT_OF_MEMORY: return "Out of memory";
    case VKDNN_STATUS_DEVICE_LOST: return "Device lost";
    case VKDNN_STATUS_INTERNAL_ERROR: return "Internal error";
    case vkdnnErrorInitializationError: return "Initialization error";
    case vkdnnErrorInvalidDevice: return "Invalid device";
    case vkdnnErrorInvalidMemcpyDirection: return "Invalid memcpy direction";
    default: return "Unknown status";
    }
}

vkdnnStatus_t vkdnnGetDeviceCount(int* count)
{
    if (!count) { return VKDNN_STATUS_INVALID_VALUE; }
    try
    {
        auto& state = registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        vkdnnStatus_t status = state.initialize();
        if (status != VKDNN_STATUS_SUCCESS) { return status; }
        *count = static_cast<int>(state.devices.size());
        return VKDNN_STATUS_SUCCESS;
    } catch (const std::bad_alloc&)
    {
        return VKDNN_STATUS_OUT_OF_MEMORY;
    } catch (const std::system_error&)
    {
        return VKDNN_STATUS_INTERNAL_ERROR;
    }
}

vkdnnStatus_t vkdnnGetDeviceProperties(vkdnnDeviceProp* properties, int device)
{
    if (!properties) { return VKDNN_STATUS_INVALID_VALUE; }
    if (device < 0) { return vkdnnErrorInvalidDevice; }
    try
    {
        auto& state = registry();
        std::lock_guard<std::mutex> lock(state.mutex);
        vkdnnStatus_t status = state.initialize();
        if (status != VKDNN_STATUS_SUCCESS) { return status; }
        if (static_cast<size_t>(device) >= state.devices.size()) { return vkdnnErrorInvalidDevice; }
        *properties = state.devices[device];
        return VKDNN_STATUS_SUCCESS;
    } catch (const std::bad_alloc&)
    {
        return VKDNN_STATUS_OUT_OF_MEMORY;
    } catch (const std::system_error&)
    {
        return VKDNN_STATUS_INTERNAL_ERROR;
    }
}

vkdnnStatus_t vkdnnRuntime::createContext(vkdnnHandle_t* handle, int device)
{
    if (!handle || device < 0) { return VKDNN_STATUS_INVALID_VALUE; }
    try
    {
        std::array<uint8_t, VK_UUID_SIZE> uuid;
        {
            auto& state = registry();
            std::lock_guard<std::mutex> lock(state.mutex);
            auto status = state.initialize();
            if (status != VKDNN_STATUS_SUCCESS) { return status; }
            if (static_cast<size_t>(device) >= state.devices.size()) { return VKDNN_STATUS_INVALID_VALUE; }
            uuid = state.uuids[device];
        }
        auto context = std::make_unique<vkdnnContext>();
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        VkResult result = vkCreateInstance(&info, nullptr, &context->instance);
        if (result != VK_SUCCESS) { return resultStatus(result); }
        std::vector<VkPhysicalDevice> physical;
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            uint32_t count = 0;
            result = vkEnumeratePhysicalDevices(context->instance, &count, nullptr);
            if (result != VK_SUCCESS) { return resultStatus(result); }
            physical.resize(count);
            if (!count) { break; }
            result = vkEnumeratePhysicalDevices(context->instance, &count, physical.data());
            if (result == VK_INCOMPLETE) { continue; }
            if (result != VK_SUCCESS) { return resultStatus(result); }
            physical.resize(count);
            break;
        }
        if (result == VK_INCOMPLETE) { return VKDNN_STATUS_INTERNAL_ERROR; }
        for (auto candidate : physical)
        {
            VkPhysicalDeviceIDProperties id{};
            id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties.pNext = &id;
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            if (std::memcmp(id.deviceUUID, uuid.data(), uuid.size()) == 0)
            {
                context->physical = candidate;
                break;
            }
        }
        if (!context->physical) { return VKDNN_STATUS_NO_DEVICE; }
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &count, queues.data());
        uint32_t family = count;
        for (uint32_t i = 0; i < count; ++i)
        {
            if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                family = i;
                break;
            }
        }
        if (family == count) { return VKDNN_STATUS_NOT_SUPPORTED; }
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        result = vkCreateDevice(context->physical, &deviceInfo, nullptr, &context->device);
        if (result != VK_SUCCESS) { return resultStatus(result); }
        vkGetDeviceQueue(context->device, family, 0, &context->queue);
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.queueFamilyIndex = family;
        result = vkCreateCommandPool(context->device, &pool, nullptr, &context->pool);
        if (result != VK_SUCCESS) { return resultStatus(result); }
        VkCommandBufferAllocateInfo command{};
        command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command.commandPool = context->pool;
        command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(context->device, &command, &context->command);
        if (result != VK_SUCCESS) { return resultStatus(result); }
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vkCreateFence(context->device, &fence, nullptr, &context->fence);
        if (result != VK_SUCCESS) { return resultStatus(result); }
        context->index = device;
        *handle = context.release();
        return VKDNN_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) { return VKDNN_STATUS_OUT_OF_MEMORY; } catch (const std::system_error&)
    {
        return VKDNN_STATUS_INTERNAL_ERROR;
    }
}

vkdnnStatus_t vkdnnRuntime::destroyContext(vkdnnHandle_t handle)
{
    if (!handle || !handle->allocations.empty()) { return VKDNN_STATUS_INVALID_VALUE; }
    auto status = resultStatus(vkDeviceWaitIdle(handle->device));
    if (status != VKDNN_STATUS_SUCCESS && status != VKDNN_STATUS_DEVICE_LOST) { return status; }
    delete handle;
    return status;
}

vkdnnStatus_t vkdnnRuntime::synchronizeContext(vkdnnHandle_t handle)
{
    if (!handle) { return VKDNN_STATUS_INVALID_VALUE; }
    return resultStatus(vkDeviceWaitIdle(handle->device));
}

vkdnnStatus_t vkdnnRuntime::allocateMemory(vkdnnHandle_t handle, vkdnnMemory_t* memory, size_t bytes)
{
    if (!handle || !memory || !bytes || bytes > std::numeric_limits<size_t>::max() - 3)
    {
        return VKDNN_STATUS_INVALID_VALUE;
    }
    try
    {
        auto allocation = std::make_unique<vkdnnMemory>();
        auto status = allocateBuffer(handle, (bytes + 3) & ~VkDeviceSize(3), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, *allocation);
        if (status != VKDNN_STATUS_SUCCESS) { return status; }
        allocation->size = bytes;
        try
        {
            handle->allocations.insert(allocation.get());
        } catch (const std::bad_alloc&)
        {
            releaseBuffer(handle, *allocation);
            return VKDNN_STATUS_OUT_OF_MEMORY;
        }
        *memory = allocation.release();
        return VKDNN_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) { return VKDNN_STATUS_OUT_OF_MEMORY; }
}

vkdnnStatus_t vkdnnRuntime::freeMemory(vkdnnHandle_t handle, vkdnnMemory_t memory)
{
    if (!handle || !memory || !handle->allocations.count(memory)) { return VKDNN_STATUS_INVALID_VALUE; }
    releaseBuffer(handle, *memory);
    handle->allocations.erase(memory);
    delete memory;
    return VKDNN_STATUS_SUCCESS;
}

vkdnnStatus_t vkdnnRuntime::copyHtoD(vkdnnHandle_t handle, vkdnnMemory_t dst, size_t offset, const void* src, size_t bytes)
{
    if (!src || !validRange(handle, dst, offset, bytes)) { return VKDNN_STATUS_INVALID_VALUE; }
    return transfer(handle, dst, offset, const_cast<void*>(src), bytes, true);
}

vkdnnStatus_t vkdnnRuntime::copyDtoH(vkdnnHandle_t handle, void* dst, vkdnnMemory_t src, size_t offset, size_t bytes)
{
    if (!dst || !validRange(handle, src, offset, bytes)) { return VKDNN_STATUS_INVALID_VALUE; }
    return transfer(handle, src, offset, dst, bytes, false);
}

vkdnnStatus_t vkdnnRuntime::copyDtoD(vkdnnHandle_t handle, vkdnnMemory_t dst, size_t dstOffset, vkdnnMemory_t src, size_t srcOffset, size_t bytes)
{
    if (!validRange(handle, dst, dstOffset, bytes) || !validRange(handle, src, srcOffset, bytes))
    {
        return VKDNN_STATUS_INVALID_VALUE;
    }
    if (dst == src && bytes && dstOffset < srcOffset + bytes && srcOffset < dstOffset + bytes)
    {
        return VKDNN_STATUS_INVALID_VALUE;
    }
    if (!bytes) { return VKDNN_STATUS_SUCCESS; }
    if ((dstOffset | srcOffset | bytes) % 4 == 0)
    {
        return copyBuffer(handle, dst->buffer, dstOffset, src->buffer, srcOffset, bytes);
    }
    try
    {
        std::vector<unsigned char> host(bytes);
        auto status = transfer(handle, src, srcOffset, host.data(), bytes, false);
        if (status != VKDNN_STATUS_SUCCESS) { return status; }
        return transfer(handle, dst, dstOffset, host.data(), bytes, true);
    } catch (const std::bad_alloc&) { return VKDNN_STATUS_OUT_OF_MEMORY; }
}

vkdnnStatus_t vkdnnRuntime::setMemory(vkdnnHandle_t handle, vkdnnMemory_t memory, size_t offset, int value, size_t bytes)
{
    if (!validRange(handle, memory, offset, bytes)) { return VKDNN_STATUS_INVALID_VALUE; }
    if (!bytes) { return VKDNN_STATUS_SUCCESS; }
    try
    {
        std::vector<unsigned char> host(bytes, static_cast<unsigned char>(value));
        return transfer(handle, memory, offset, host.data(), bytes, true);
    } catch (const std::bad_alloc&) { return VKDNN_STATUS_OUT_OF_MEMORY; }
}
