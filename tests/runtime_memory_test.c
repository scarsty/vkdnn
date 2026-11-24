#include "vkdnn/vkdnn.h"

#include <stdint.h>
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
#define OK(call) CHECK((call) == vkdnnSuccess)
#define INVALID(call) CHECK((call) == vkdnnErrorInvalidValue)

int main(void)
{
    int count = 0;
    vkdnnError_t status = vkdnnGetDeviceCount(&count);
    if (status == vkdnnErrorNotSupported || (status == vkdnnSuccess && !count)) { return 77; }
    CHECK(status == vkdnnSuccess);
    void* allocationA = NULL;
    void* allocationB = NULL;
    unsigned char input[1031];
    unsigned char output[1031];
    unsigned char expected[1031];
    for (size_t i = 0; i < sizeof(input); ++i) { input[i] = (unsigned char)(i * 37); }
    CHECK(vkdnnSetDevice(-1) == vkdnnErrorInvalidDevice);
    CHECK(vkdnnSetDevice(count) == vkdnnErrorInvalidDevice);
    OK(vkdnnSetDevice(0));
    int device = -1;
    OK(vkdnnGetDevice(&device));
    CHECK(device == 0);
    INVALID(vkdnnGetDevice(NULL));
    INVALID(vkdnnMalloc(NULL, 1));
    INVALID(vkdnnMalloc(&allocationA, SIZE_MAX));
    CHECK(allocationA == NULL);
    OK(vkdnnMalloc(&allocationA, 0));
    CHECK(allocationA == NULL);
    OK(vkdnnFree(NULL));
    OK(vkdnnMalloc(&allocationA, sizeof(input)));
    OK(vkdnnMalloc(&allocationB, sizeof(input)));
    unsigned char* a = (unsigned char*)allocationA;
    unsigned char* b = (unsigned char*)allocationB;
    OK(vkdnnMemcpy(a, input, sizeof(input), vkdnnMemcpyHostToDevice));
    OK(vkdnnMemcpy(output, a, sizeof(output), vkdnnMemcpyDeviceToHost));
    CHECK(memcmp(input, output, sizeof(input)) == 0);
    OK(vkdnnMemset(b, 0x12a, sizeof(input)));
    memset(expected, 0x2a, sizeof(expected));
    OK(vkdnnMemcpy(b + 4, a, 1024, vkdnnMemcpyDeviceToDevice));
    memcpy(expected + 4, input, 1024);
    OK(vkdnnMemcpy(b + 3, a + 1, 1019, vkdnnMemcpyDeviceToDevice));
    OK(vkdnnMemcpy(b + 3, a + 1, 1019, vkdnnMemcpyDefault));
    memcpy(expected + 3, input + 1, 1019);
    OK(vkdnnMemcpy(b + 7, input + 5, 13, vkdnnMemcpyDefault));
    memcpy(expected + 7, input + 5, 13);
    OK(vkdnnMemset(b + 1029, 0xab, 2));
    memset(expected + 1029, 0xab, 2);
    OK(vkdnnMemcpy(b + 30, b + 3, 7, vkdnnMemcpyDeviceToDevice));
    memcpy(expected + 30, expected + 3, 7);
    INVALID(vkdnnMemcpy(b + 3, b + 4, 10, vkdnnMemcpyDeviceToDevice));
    INVALID(vkdnnMemcpy(b + 1030, input, 2, vkdnnMemcpyHostToDevice));
    INVALID(vkdnnMemcpy(output, b + 1032, 1, vkdnnMemcpyDefault));
    INVALID(vkdnnMemset(input, 0, 1));
    INVALID(vkdnnFree(b + 1));
    CHECK(vkdnnMemcpy(b, input, 1, vkdnnMemcpyDeviceToHost) == vkdnnErrorInvalidMemcpyDirection);
    CHECK(vkdnnMemcpy(b, input, 1, (enum vkdnnMemcpyKind)99) == vkdnnErrorInvalidMemcpyDirection);
    INVALID(vkdnnMemcpy(b, NULL, 1, vkdnnMemcpyHostToDevice));
    OK(vkdnnMemcpy(NULL, NULL, 0, vkdnnMemcpyDefault));
    OK(vkdnnMemcpy(output, b, sizeof(output), vkdnnMemcpyDefault));
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    OK(vkdnnMemcpy(output, input, sizeof(output), vkdnnMemcpyHostToHost));
    CHECK(memcmp(output, input, sizeof(output)) == 0);
    OK(vkdnnMemcpy(output, input, sizeof(output), vkdnnMemcpyDefault));
    INVALID(vkdnnMemcpy(output + 1, output, 3, vkdnnMemcpyHostToHost));
    OK(vkdnnDeviceSynchronize());
    if (count > 1)
    {
        void* other = NULL;
        OK(vkdnnSetDevice(1));
        OK(vkdnnMalloc(&other, 8));
        CHECK(vkdnnMemcpy(other, a, 8, vkdnnMemcpyDeviceToDevice) == vkdnnErrorNotSupported);
        OK(vkdnnSetDevice(0));
        OK(vkdnnFree(other));
        OK(vkdnnSetDevice(1));
        OK(vkdnnDeviceReset());
        OK(vkdnnSetDevice(0));
    }
    OK(vkdnnFree(a));
    OK(vkdnnFree(b));
    for (int i = 0; i < 8; ++i)
    {
        OK(vkdnnMalloc(&allocationA, 1));
        OK(vkdnnMemcpy(allocationA, input, 1, vkdnnMemcpyHostToDevice));
        OK(vkdnnMemcpy(output, allocationA, 1, vkdnnMemcpyDeviceToHost));
        CHECK(output[0] == input[0]);
        OK(vkdnnFree(allocationA));
    }
    OK(vkdnnMalloc(&allocationA, 32));
    OK(vkdnnDeviceReset());
    INVALID(vkdnnFree(allocationA));
    OK(vkdnnMalloc(&allocationA, 4));
    OK(vkdnnMemcpy(allocationA, input, 4, vkdnnMemcpyDefault));
    OK(vkdnnMemcpy(output, allocationA, 4, vkdnnMemcpyDefault));
    CHECK(memcmp(input, output, 4) == 0);
    OK(vkdnnFree(allocationA));
    OK(vkdnnDeviceReset());
    puts("PASS: CUDA-style device selection, malloc/free, memcpy directions/default, offsets, memset and reset");
    return 0;
}
