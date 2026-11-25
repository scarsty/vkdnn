#pragma once
#include "vkdnn/vkdnn.h"
#include <cstdint>

struct ComputeParams
{
    uint32_t op, count, n, c, h, w, oh, ow;
    uint32_t kh, kw, ph, pw, sh, sw, mode, nan;
    float alpha, beta;
    uint32_t xOffset, yOffset;
    uint32_t outputC;
};

vkdnnError_t runtimeCompute(int device, const void* x, size_t xBytes, void* y, size_t yBytes, ComputeParams params);

struct ActivationBackwardParams
{
    uint32_t count, mode, nan, xOffset, yOffset, dyOffset, dxOffset;
    float alpha, beta;
};
vkdnnError_t runtimeActivationBackward(int device, const void* x, const void* y, const void* dy,
    void* dx, size_t bytes, ActivationBackwardParams params);

struct GemmParams
{
    uint32_t m, n, k, lda, ldb, ldc, transa, transb;
    uint32_t aOffset, bOffset, cOffset;
    float alpha, beta;
};
vkdnnError_t runtimeHostScalar(const float* pointer, float& value);
vkdnnError_t runtimeReduce(int device, const void* x, size_t xBytes, const void* y, size_t yBytes,
    float* result, GemmParams params);
vkdnnError_t runtimeGemm(int device, const void* a, size_t aBytes, const void* b, size_t bBytes,
    void* c, size_t cBytes, GemmParams params);

struct ConvolutionParams
{
    uint32_t count, c, h, w, k, fc, kh, kw, oh, ow;
    uint32_t ph, pw, sh, sw, dh, dw, groups, mode;
    uint32_t aOffset, bOffset, cOffset;
    float alpha, beta;
    uint32_t op, n, nan;
    float gamma, delta;
};
vkdnnError_t runtimeConvolution(int device, const void* x, size_t xBytes, const void* w, size_t wBytes,
    void* y, size_t yBytes, ConvolutionParams params, const void* extra = nullptr, size_t extraBytes = 0);
