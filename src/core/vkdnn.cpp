#include "runtime_internal.h"

#include "profile_internal.h"
#include <vulkan/vulkan.h>

#ifdef VKDNN_PROFILE
static VkResult profileWaitForFences(VkDevice device, uint32_t count, const VkFence* fences, VkBool32 all, uint64_t timeout)
{
    VKDNN_PROFILE_SCOPE("stage.fence_wait");
    return vkWaitForFences(device, count, fences, all, timeout);
}
static VkResult profileDeviceWaitIdle(VkDevice device)
{
    VKDNN_PROFILE_SCOPE("stage.device_wait");
    return vkDeviceWaitIdle(device);
}
static VkResult profileQueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submit, VkFence fence)
{
    VKDNN_PROFILE_SCOPE("stage.queue_submit");
    return vkQueueSubmit(queue, count, submit, fence);
}
#define vkWaitForFences profileWaitForFences
#define vkDeviceWaitIdle profileDeviceWaitIdle
#define vkQueueSubmit profileQueueSubmit
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
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
#ifdef VKDNN_PROFILE
    struct Timeline
    {
        uint64_t submitted = 0, collected = 0, previousEnd = 0;
        std::map<uint64_t, std::array<uint64_t, 2>> completed;
        void add(uint64_t sequence, const uint64_t* ticks, double period, uint64_t mask)
        {
            completed.emplace(sequence, std::array<uint64_t, 2>{ ticks[0], ticks[1] });
            auto& statistics = vkdnnProfile::statistics();
            std::lock_guard<std::mutex> lock(statistics.mutex);
            while (completed.count(collected))
            {
                auto value = completed.at(collected);
                if (collected)
                {
                    auto& entry = statistics.entries["timeline.dispatch_gap"];
                    ++entry.calls;
                    entry.milliseconds += double((value[0] - previousEnd) & mask) * period / 1000000;
                }
                previousEnd = value[1];
                completed.erase(collected++);
            }
        }
    } timeline;
#endif
    struct Submission
    {
        VkDevice device = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkDescriptorPool descriptors = VK_NULL_HANDLE;
        bool pending = false;
#ifdef VKDNN_PROFILE
        VkQueryPool queries = VK_NULL_HANDLE;
        double timestampPeriod = 0;
        uint64_t timestampMask = 0;
        std::string category;
        bool measured = false;
        Timeline* timeline = nullptr;
        uint64_t sequence = 0;
        void collect()
        {
            if (!measured) { return; }
            uint64_t ticks[2]{};
            auto result = vkGetQueryPoolResults(device, queries, 0, 2, sizeof(ticks), ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (result != VK_SUCCESS)
            {
                std::fprintf(stderr, "GPU timestamp read failed: %d\n", int(result));
                return;
            }
            timeline->add(sequence, ticks, timestampPeriod, timestampMask);
            auto& statistics = vkdnnProfile::statistics();
            std::lock_guard<std::mutex> lock(statistics.mutex);
            auto& entry = statistics.entries[category];
            ++entry.calls;
            entry.milliseconds += double((ticks[1] - ticks[0]) & timestampMask) * timestampPeriod / 1000000;
            measured = false;
        }
#endif
        ~Submission()
        {
#ifdef VKDNN_PROFILE
            if (queries) { vkDestroyQueryPool(device, queries, nullptr); }
#endif
            if (descriptors) { vkDestroyDescriptorPool(device, descriptors, nullptr); }
            if (fence) { vkDestroyFence(device, fence, nullptr); }
            if (pool) { vkDestroyCommandPool(device, pool, nullptr); }
        }
    };
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
    VkPhysicalDeviceProperties properties{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t family = 0;
    bool asynchronous = false;
    size_t nextSubmission = 0;
    std::array<std::unique_ptr<Submission>, 32> submissions;
#ifdef VKDNN_AUTOTUNE
    std::map<std::array<uint32_t, 20>, uint32_t> convolutionChoices;
#endif
    vkdnnMemory staging;
    vkdnnMemory reduction;
    vkdnnMemory_t convolutionPartial = nullptr;
    int index = -1;
    std::unordered_set<vkdnnMemory_t> allocations;
    // Runtime calls serialize and wait for the fence before reusing descriptors.
    // Contexts are explicitly reset, not deleted by DLL process-detach cleanup.
    std::vector<std::unique_ptr<ComputeResources>> compute;

    ~vkdnnContext()
    {
        if (device)
        {
            for (auto& submission : submissions) { submission.reset(); }
            compute.clear();
            if (staging.buffer) { vkDestroyBuffer(device, staging.buffer, nullptr); }
            if (staging.allocation) { vkFreeMemory(device, staging.allocation, nullptr); }
            if (reduction.buffer) { vkDestroyBuffer(device, reduction.buffer, nullptr); }
            if (reduction.allocation) { vkFreeMemory(device, reduction.allocation, nullptr); }
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
    VKDNN_PROFILE_SCOPE("stage.dispatch_total");
    const auto& properties = context->properties;
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
    VkCommandPool commandPool = context->pool;
    VkCommandBuffer command = context->command;
    VkFence fence = context->fence;
    vkdnnContext::Submission* submission = nullptr;
    if (context->asynchronous)
    {
        auto& slot = context->submissions[context->nextSubmission];
        if (!slot)
        {
            auto pending = std::make_unique<vkdnnContext::Submission>();
            pending->device = r.device;
            VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            poolInfo.queueFamilyIndex = context->family;
            result = vkCreateCommandPool(r.device, &poolInfo, nullptr, &pending->pool);
            if (result != VK_SUCCESS) { return status(result); }
            VkCommandBufferAllocateInfo commandInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            commandInfo.commandPool = pending->pool;
            commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandInfo.commandBufferCount = 1;
            result = vkAllocateCommandBuffers(r.device, &commandInfo, &pending->command);
            if (result != VK_SUCCESS) { return status(result); }
            VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            result = vkCreateFence(r.device, &fenceInfo, nullptr, &pending->fence);
            if (result != VK_SUCCESS) { return status(result); }
            VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
            VkDescriptorPoolCreateInfo descriptorInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            descriptorInfo.maxSets = 1;
            descriptorInfo.poolSizeCount = 1;
            descriptorInfo.pPoolSizes = &poolSize;
            result = vkCreateDescriptorPool(r.device, &descriptorInfo, nullptr, &pending->descriptors);
            if (result != VK_SUCCESS) { return status(result); }
            slot = std::move(pending);
        }
        submission = slot.get();
        if (submission->pending)
        {
            result = vkWaitForFences(r.device, 1, &submission->fence, VK_TRUE, UINT64_MAX);
            if (result != VK_SUCCESS) { return status(result); }
            submission->pending = false;
        }
#ifdef VKDNN_PROFILE
        submission->collect();
        if (!submission->queries)
        {
            uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(context->physical, &count, families.data());
            uint32_t bits = families[context->family].timestampValidBits;
            if (!bits) { return vkdnnErrorNotSupported; }
            submission->timestampMask = UINT64_MAX >> (64 - bits);
            submission->timestampPeriod = context->properties.limits.timestampPeriod;
            VkQueryPoolCreateInfo query{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            query.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query.queryCount = 2;
            result = vkCreateQueryPool(r.device, &query, nullptr, &submission->queries);
            if (result != VK_SUCCESS) { return status(result); }
        }
        submission->category = "gpu.other";
        if (code == convolutionSpirv) { submission->category = "gpu.convolution." + std::to_string(static_cast<const ConvolutionParams*>(params)->op); }
        else if (code == gemmSpirv) { submission->category = "gpu.gemm." + std::to_string(static_cast<const GemmParams*>(params)->transa); }
        else if (code == forwardSpirv) { submission->category = "gpu.compute." + std::to_string(static_cast<const ComputeParams*>(params)->op); }
        else if (code == backwardSpirv) { submission->category = "gpu.activation_backward"; }
#endif
        result = vkResetDescriptorPool(r.device, submission->descriptors, 0);
        if (result != VK_SUCCESS) { return status(result); }
        VkDescriptorSetAllocateInfo allocate{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocate.descriptorPool = submission->descriptors;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &r.setLayout;
        result = vkAllocateDescriptorSets(r.device, &allocate, &set);
        if (result != VK_SUCCESS) { return status(result); }
        commandPool = submission->pool;
        command = submission->command;
        fence = submission->fence;
    }
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
    result = vkResetCommandPool(r.device, commandPool, 0);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, r.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(command, r.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, paramSize, params);
#ifdef VKDNN_PROFILE
    if (submission)
    {
        vkCmdResetQueryPool(command, submission->queries, 0, 2);
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, submission->queries, 0);
    }
#endif
    vkCmdDispatch(command, groups, groupsY, 1);
#ifdef VKDNN_PROFILE
    if (submission) { vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, submission->queries, 1); }
#endif
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    result = vkEndCommandBuffer(command);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    result = vkResetFences(r.device, 1, &fence);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    result = vkQueueSubmit(context->queue, 1, &submit, fence);
    if (result != VK_SUCCESS)
    {
        return status(result);
    }
    if (submission)
    {
        submission->pending = true;
#ifdef VKDNN_PROFILE
        submission->measured = true;
        submission->timeline = &context->timeline;
        submission->sequence = context->timeline.submitted++;
#endif
        context->nextSubmission = (context->nextSubmission + 1) % context->submissions.size();
        return vkdnnSuccess;
    }
    return status(vkWaitForFences(r.device, 1, &fence, VK_TRUE, UINT64_MAX));
}

vkdnnError_t vkdnnRuntime::activationBackwardBuffers(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t y,
    vkdnnMemory_t dy, vkdnnMemory_t dx, const ActivationBackwardParams& params)
{
    vkdnnMemory_t buffers[] = { x, y, dy, dx };
    return dispatchBuffers(context, buffers, 4, backwardSpirv, sizeof(backwardSpirv), &params, sizeof(params), (params.count + 255u) / 256u, 1);
}

vkdnnError_t vkdnnRuntime::computeBuffers(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t y, const ComputeParams& params)
{
    vkdnnMemory_t buffers[] = { x, y };
    uint32_t groups = (params.count + 63u) / 64u;
    uint32_t groupsY = 1;
    if (params.op == 0) { groups = (params.count + 255u) / 256u; }
    if (params.op == 2 || params.op == 3)
    {
        const auto& properties = context->properties;
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

static vkdnnError_t convolutionCandidate(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t w, vkdnnMemory_t y, const ConvolutionParams& params, uint32_t splits)
{
    vkdnnMemory_t buffers[] = { x, w, y };
    if (params.op == 1 && splits == 2)
    {
        auto specialized = params;
        specialized.op = 10;
        return dispatchBuffers(context, buffers, 3, convolutionSpirv, sizeof(convolutionSpirv), &specialized,
            sizeof(specialized), (params.count + 63u) / 64u, 1);
    }
    if (splits > 1 && params.alpha != 0)
    {
        size_t bytes = size_t(params.count) * splits * sizeof(float);
        if (!context->convolutionPartial || context->convolutionPartial->size < bytes)
        {
            if (context->convolutionPartial)
            {
                auto status = freeMemory(context, context->convolutionPartial);
                if (status != vkdnnSuccess) { return status; }
                context->convolutionPartial = nullptr;
            }
            auto status = allocateMemory(context, &context->convolutionPartial, bytes);
            if (status != vkdnnSuccess) { return status; }
        }
        auto partialParams = params;
        partialParams.op = 7;
        if (params.op == 3) { partialParams.op = 9; }
        partialParams.nan = splits;
        partialParams.cOffset = 0;
        vkdnnMemory_t partialBuffers[] = { x, w, context->convolutionPartial };
        auto status = dispatchBuffers(context, partialBuffers, 3, convolutionSpirv, sizeof(convolutionSpirv),
            &partialParams, sizeof(partialParams), params.count * splits, 1);
        if (status != vkdnnSuccess) { return status; }
        auto finalParams = params;
        finalParams.op = 8;
        finalParams.nan = splits;
        finalParams.aOffset = 0;
        vkdnnMemory_t finalBuffers[] = { context->convolutionPartial, w, y };
        return dispatchBuffers(context, finalBuffers, 3, convolutionSpirv, sizeof(convolutionSpirv),
            &finalParams, sizeof(finalParams), (params.count + 63u) / 64u, 1);
    }
    uint32_t groups = (params.count + 63u) / 64u, groupsY = 1;
    if ((params.op == 2 || params.op == 3) && params.count <= 65535) { groups = params.count; }
    if (params.op == 6)
    {
        groups = params.n;
        if (params.mode == 1) { groups *= params.h * params.w; }
        const auto& properties = context->properties;
        uint32_t limit = properties.limits.maxComputeWorkGroupCount[0];
        if (groups > limit)
        {
            groupsY = (groups - 1u) / limit + 1u;
            groups = limit;
        }
    }
    return dispatchBuffers(context, buffers, 3, convolutionSpirv, sizeof(convolutionSpirv), &params, sizeof(params), groups, groupsY);
}

vkdnnError_t vkdnnRuntime::convolutionBuffers(vkdnnHandle_t context, vkdnnMemory_t x, vkdnnMemory_t w, vkdnnMemory_t y, const ConvolutionParams& params)
{
    uint64_t length = params.n * uint64_t(params.oh) * params.ow;
    uint32_t splits = 1;
    if (params.count <= 256 && ((params.op == 2 && length >= 8192) || (params.op == 3 && length >= 32768))) { splits = 32; }
#ifdef VKDNN_AUTOTUNE
    bool specializedData = params.op == 1 && params.kh == 3 && params.kw == 3
        && params.sh == 1 && params.sw == 1 && params.dh == 1 && params.dw == 1
        && params.ph == 0 && params.pw == 0 && params.groups == 1 && params.mode == 1
        && params.h >= 3 && params.w >= 3 && params.oh == params.h - 2 && params.ow == params.w - 2;
    if ((((params.op == 2 || params.op == 3) && params.count <= 65535) || specializedData) && params.alpha != 0)
    {
        std::array<uint32_t, 20> key{ params.count, params.c, params.h, params.w, params.k, params.fc,
            params.kh, params.kw, params.oh, params.ow, params.ph, params.pw, params.sh, params.sw,
            params.dh, params.dw, params.groups, params.mode, params.op, params.n };
        auto found = context->convolutionChoices.find(key);
        if (found != context->convolutionChoices.end()) { splits = found->second; }
        else
        {
            vkdnnMemory_t temporary = nullptr;
            auto status = allocateMemory(context, &temporary, y->size);
            if (status != vkdnnSuccess) { return status; }
            auto trial = params;
            trial.alpha = 1;
            trial.beta = 0;
            std::vector<float> reference(params.count), actual(params.count);
            double fastest = std::numeric_limits<double>::max();
            std::vector<uint32_t> candidates{ 1, 8, 32, 64 };
            if (specializedData) { candidates = { 1, 2 }; }
            for (uint32_t candidate : candidates)
            {
                if (!specializedData && uint64_t(params.count) * candidate > context->properties.limits.maxComputeWorkGroupCount[0]) { continue; }
                status = convolutionCandidate(context, x, w, temporary, trial, candidate);
                if (status != vkdnnSuccess) { break; }
                status = copyDtoH(context, actual.data(), temporary, size_t(params.cOffset) * sizeof(float), actual.size() * sizeof(float));
                if (status != vkdnnSuccess) { break; }
                if (candidate == 1) { reference = actual; }
                bool valid = true;
                for (size_t index = 0; index < actual.size(); ++index)
                {
                    if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])
                        || std::abs(actual[index] - reference[index]) > 0.001f + 0.001f * std::abs(reference[index])) { valid = false; }
                }
                if (!valid)
                {
                    std::fprintf(stderr, "VKAUTOTUNE,rejected,op=%u,count=%u,splits=%u,numerical_mismatch\n", params.op, params.count, candidate);
                    continue;
                }
                double best = std::numeric_limits<double>::max();
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    auto start = std::chrono::steady_clock::now();
                    for (int iteration = 0; iteration < 4; ++iteration)
                    {
                        status = convolutionCandidate(context, x, w, temporary, trial, candidate);
                        if (status != vkdnnSuccess) { break; }
                    }
                    if (status == vkdnnSuccess) { status = synchronizeContext(context); }
                    if (status != vkdnnSuccess) { break; }
                    double microseconds = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 4;
                    if (microseconds < best) { best = microseconds; }
                }
                if (status != vkdnnSuccess) { break; }
                std::fprintf(stderr, "VKAUTOTUNE,candidate,op=%u,n=%u,c=%u,h=%u,w=%u,k=%u,count=%u,splits=%u,us=%.3f\n",
                    params.op, params.n, params.c, params.h, params.w, params.k, params.count, candidate, best);
                if (best < fastest)
                {
                    fastest = best;
                    splits = candidate;
                }
            }
            auto released = freeMemory(context, temporary);
            if (status != vkdnnSuccess) { return status; }
            if (released != vkdnnSuccess) { return released; }
            context->convolutionChoices.emplace(key, splits);
            std::fprintf(stderr, "VKAUTOTUNE,selected,op=%u,n=%u,c=%u,h=%u,w=%u,k=%u,count=%u,splits=%u,us=%.3f\n",
                params.op, params.n, params.c, params.h, params.w, params.k, params.count, splits, fastest);
        }
    }
#endif
    return convolutionCandidate(context, x, w, y, params, splits);
}

vkdnnError_t vkdnnRuntime::gemmBuffers(vkdnnHandle_t context, vkdnnMemory_t a, vkdnnMemory_t b, vkdnnMemory_t c, const GemmParams& params)
{
    const auto& properties = context->properties;
    if (properties.limits.maxComputeWorkGroupInvocations < 256
        || properties.limits.maxComputeWorkGroupSize[0] < 16 || properties.limits.maxComputeWorkGroupSize[1] < 16
        || properties.limits.maxComputeSharedMemorySize < 2048)
    {
        return vkdnnErrorNotSupported;
    }
    vkdnnMemory_t buffers[] = { a, b, c };
    if (params.transa <= 1 && params.transb <= 1 && params.m <= 16 && params.k >= 128)
    {
        return dispatchBuffers(context, buffers, 3, gemmSpirv, sizeof(gemmSpirv), &params, sizeof(params), (params.m + 3u) / 4u, params.n);
    }
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
    info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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
    if (handle->asynchronous)
    {
        auto status = synchronizeContext(handle);
        if (status != vkdnnSuccess) { return status; }
    }
    VkDeviceSize start = offset & ~VkDeviceSize(3);
    VkDeviceSize end = (offset + bytes + VkDeviceSize(3)) & ~VkDeviceSize(3);
    auto& staging = handle->staging;
    vkdnnStatus_t status = VKDNN_STATUS_SUCCESS;
    if (staging.size < end - start)
    {
        releaseBuffer(handle, staging);
        status = allocateBuffer(handle, end - start, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, staging);
        if (status != VKDNN_STATUS_SUCCESS) { return status; }
    }
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
                context->properties = properties.properties;
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
        context->family = family;
        context->index = device;
        *handle = context.release();
        return VKDNN_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) { return VKDNN_STATUS_OUT_OF_MEMORY; } catch (const std::system_error&)
    {
        return VKDNN_STATUS_INTERNAL_ERROR;
    }
}

vkdnnError_t vkdnnRuntime::reduceBuffers(vkdnnHandle_t handle, vkdnnMemory_t first, vkdnnMemory_t second,
    float* output, const GemmParams& params)
{
    auto& reduction = handle->reduction;
    if (!reduction.buffer)
    {
        auto status = allocateBuffer(handle, sizeof(float), VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, reduction);
        if (status != vkdnnSuccess) { return status; }
    }
    auto status = gemmBuffers(handle, first, second, &reduction, params);
    if (status != vkdnnSuccess) { return status; }
    if (handle->asynchronous)
    {
        size_t last = (handle->nextSubmission + handle->submissions.size() - 1) % handle->submissions.size();
        auto& submission = *handle->submissions[last];
        status = resultStatus(vkWaitForFences(handle->device, 1, &submission.fence, VK_TRUE, UINT64_MAX));
        if (status != vkdnnSuccess) { return status; }
        submission.pending = false;
    }
    void* mapped = nullptr;
    auto result = vkMapMemory(handle->device, reduction.allocation, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (result != VK_SUCCESS) { return resultStatus(result); }
    VkMappedMemoryRange range{ VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    range.memory = reduction.allocation;
    range.size = VK_WHOLE_SIZE;
    result = vkInvalidateMappedMemoryRanges(handle->device, 1, &range);
    if (result == VK_SUCCESS) { std::memcpy(output, mapped, sizeof(float)); }
    vkUnmapMemory(handle->device, reduction.allocation);
    return resultStatus(result);
}

vkdnnStatus_t vkdnnRuntime::destroyContext(vkdnnHandle_t handle)
{
    if (handle && handle->convolutionPartial)
    {
        auto status = freeMemory(handle, handle->convolutionPartial);
        if (status != vkdnnSuccess) { return status; }
        handle->convolutionPartial = nullptr;
    }
    if (!handle || !handle->allocations.empty()) { return VKDNN_STATUS_INVALID_VALUE; }
    auto status = resultStatus(vkDeviceWaitIdle(handle->device));
    if (status != VKDNN_STATUS_SUCCESS && status != VKDNN_STATUS_DEVICE_LOST) { return status; }
    delete handle;
    return status;
}

vkdnnError_t vkdnnRuntime::setExecutionMode(vkdnnHandle_t handle, vkdnnExecutionMode mode)
{
    auto status = synchronizeContext(handle);
    if (status == vkdnnSuccess) { handle->asynchronous = mode == vkdnnExecutionAsynchronous; }
    return status;
}

vkdnnStatus_t vkdnnRuntime::synchronizeContext(vkdnnHandle_t handle)
{
    if (!handle) { return VKDNN_STATUS_INVALID_VALUE; }
    auto result = vkDeviceWaitIdle(handle->device);
#ifdef VKDNN_PROFILE
    if (result == VK_SUCCESS)
    {
        for (auto& submission : handle->submissions)
        {
            if (submission) { submission->collect(); }
        }
    }
#endif
    return resultStatus(result);
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
    if (handle->asynchronous)
    {
        auto status = synchronizeContext(handle);
        if (status != vkdnnSuccess) { return status; }
    }
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
