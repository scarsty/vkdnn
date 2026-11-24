#include "vkdnn/vkdnn.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) \
    do \
    { \
        if (!(condition)) \
        { \
            fprintf(stderr, "Check failed at line %d: %s\n", __LINE__, #condition); \
            return 1; \
        } \
    } while (0)

int main(void)
{
    int count = -1;
    int repeatedCount = -1;
    vkdnnDeviceProp properties;
    vkdnnDeviceProp original;
    memset(&properties, 0x5a, sizeof(properties));
    memcpy(&original, &properties, sizeof(original));
    CHECK(vkdnnGetDeviceCount(NULL) == vkdnnErrorInvalidValue);
    CHECK(vkdnnGetDeviceProperties(NULL, 0) == vkdnnErrorInvalidValue);
    CHECK(vkdnnGetDeviceProperties(&properties, -1) == vkdnnErrorInvalidDevice);
    CHECK(memcmp(&properties, &original, sizeof(properties)) == 0);

    vkdnnError_t status = vkdnnGetDeviceCount(&count);
    if (status == vkdnnErrorNotSupported)
    {
        CHECK(count == -1);
        fprintf(stderr, "SKIP: Vulkan 1.2 driver unavailable: %s\n", vkdnnGetErrorString(status));
        return 77;
    }
    if (status != vkdnnSuccess)
    {
        fprintf(stderr, "Device enumeration failed: %s\n", vkdnnGetErrorString(status));
        return 1;
    }
    CHECK(count >= 0);
    CHECK(vkdnnGetDeviceCount(&repeatedCount) == vkdnnSuccess);
    CHECK(count == repeatedCount);
    CHECK(vkdnnGetDeviceProperties(&properties, count) == vkdnnErrorInvalidDevice);
    CHECK(memcmp(&properties, &original, sizeof(properties)) == 0);
    printf("Vulkan compute devices: %d\n", count);
    if (count == 0)
    {
        fprintf(stderr, "SKIP: no eligible Vulkan compute devices\n");
        return 77;
    }
    for (int device = 0; device < count; ++device)
    {
        vkdnnDeviceProp repeated;
        CHECK(vkdnnGetDeviceProperties(&properties, device) == vkdnnSuccess);
        CHECK(vkdnnGetDeviceProperties(&repeated, device) == vkdnnSuccess);
        CHECK(memcmp(&properties, &repeated, sizeof(properties)) == 0);
        CHECK(memchr(properties.name, '\0', sizeof(properties.name)) != NULL);
        CHECK(properties.name[0] != '\0');
        CHECK(properties.apiVersion >= ((1u << 22) | (2u << 12)));
        CHECK(properties.deviceType >= VKDNN_DEVICE_TYPE_OTHER && properties.deviceType <= VKDNN_DEVICE_TYPE_CPU);
        CHECK(properties.maxStorageBufferRange > 0);
        CHECK(properties.maxComputeSharedMemorySize > 0);
        CHECK(properties.maxComputeWorkGroupInvocations > 0);
        for (int axis = 0; axis < 3; ++axis)
        {
            CHECK(properties.maxComputeWorkGroupSize[axis] > 0);
            CHECK(properties.maxComputeWorkGroupCount[axis] > 0);
        }
        printf("[%d] %s | type=%d vendor=0x%04x device=0x%04x | device-local memory=%llu MiB\n",
            device, properties.name, (int)properties.deviceType, (unsigned)properties.vendorId,
            (unsigned)properties.deviceId, (unsigned long long)(properties.totalGlobalMem / (1024 * 1024)));
    }
    return 0;
}
