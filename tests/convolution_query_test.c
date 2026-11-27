#include "vkdnn/vkdnn_conv_query.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) \
    do \
    { \
        if (!(x)) \
        { \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); \
            return 1; \
        } \
    } while (0)
#define OK(x) CHECK((x) == VKDNN_STATUS_SUCCESS)
int main(void)
{
    int devices = 0, count = -1;
    vkdnnConvolutionFwdAlgoPerf_t f[2], saved[2];
    vkdnnConvolutionBwdDataAlgoPerf_t d[2], ds[2];
    vkdnnConvolutionBwdFilterAlgoPerf_t b[2], bs[2];
    vkdnnHandle_t handle = NULL;
    vkdnnTensorDescriptor_t x = NULL, y = NULL;
    vkdnnFilterDescriptor_t w = NULL;
    vkdnnConvolutionDescriptor_t conv = NULL;
    vkdnnError_t error;
    CHECK(vkdnnSetConvolutionMathType(NULL, VKDNN_DEFAULT_MATH) == VKDNN_STATUS_BAD_PARAM);
    memset(f, 0x5a, sizeof(f));
    memset(d, 0x5a, sizeof(d));
    memset(b, 0x5a, sizeof(b));
    CHECK(vkdnnGetConvolutionForwardAlgorithm_v7(NULL, NULL, NULL, NULL, NULL, 1, &count, f) == VKDNN_STATUS_NOT_INITIALIZED);
    CHECK(count == -1);
    error = vkdnnGetDeviceCount(&devices);
    if (error == vkdnnErrorNoDevice || (error == vkdnnSuccess && !devices)) { return 77; }
    CHECK(error == vkdnnSuccess);
    {
        vkdnnDeviceProp prop;
        CHECK(vkdnnGetDeviceProperties(&prop, 0) == vkdnnSuccess);
        printf("GPU: %s\n", prop.name);
    }
    OK(vkdnnCreate(&handle));
    OK(vkdnnCreateTensorDescriptor(&x));
    OK(vkdnnCreateTensorDescriptor(&y));
    OK(vkdnnCreateFilterDescriptor(&w));
    OK(vkdnnCreateConvolutionDescriptor(&conv));
    OK(vkdnnSetConvolutionMathType(conv, VKDNN_DEFAULT_MATH));
    CHECK(vkdnnSetConvolutionMathType(conv, VKDNN_TENSOR_OP_MATH) == VKDNN_STATUS_NOT_SUPPORTED);
    CHECK(vkdnnSetConvolutionMathType(conv, VKDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) == VKDNN_STATUS_NOT_SUPPORTED);
    CHECK(vkdnnSetConvolutionMathType(conv, VKDNN_FMA_MATH) == VKDNN_STATUS_NOT_SUPPORTED);
    OK(vkdnnSetTensor4dDescriptor(x, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 2, 3, 3));
    OK(vkdnnSetTensor4dDescriptor(y, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 2, 3, 3));
    OK(vkdnnSetFilter4dDescriptor(w, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 2, 1, 1, 1));
    OK(vkdnnSetConvolution2dDescriptor(conv, 0, 0, 1, 1, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
    OK(vkdnnSetConvolutionGroupCount(conv, 2));
    memcpy(saved, f, sizeof(f));
    OK(vkdnnGetConvolutionForwardAlgorithm_v7(handle, x, w, conv, y, 2, &count, f));
    CHECK(count == 1 && f[0].algo == VKDNN_CONVOLUTION_FWD_ALGO_DIRECT);
    CHECK(f[0].status == VKDNN_STATUS_SUCCESS && isnan(f[0].time) && f[0].memory == 0);
    CHECK(f[0].determinism == VKDNN_DETERMINISTIC && f[0].mathType == VKDNN_DEFAULT_MATH);
    CHECK(memcmp(&f[1], &saved[1], sizeof(f[1])) == 0);
    OK(vkdnnGetConvolutionBackwardDataAlgorithm_v7(handle, w, y, conv, x, 2, &count, d));
    CHECK(count == 1 && d[0].algo == VKDNN_CONVOLUTION_BWD_DATA_ALGO_0);
    CHECK(d[0].status == VKDNN_STATUS_SUCCESS && isnan(d[0].time) && d[0].memory == 0);
    CHECK(d[0].determinism == VKDNN_DETERMINISTIC && d[0].mathType == VKDNN_DEFAULT_MATH);
    OK(vkdnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x, y, conv, w, 2, &count, b));
    CHECK(count == 1 && b[0].algo == VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
    CHECK(b[0].status == VKDNN_STATUS_SUCCESS && isnan(b[0].time) && b[0].memory == 0);
    CHECK(b[0].determinism == VKDNN_DETERMINISTIC && b[0].mathType == VKDNN_DEFAULT_MATH);
    {
        void *xp = NULL, *yp = NULL, *wp = NULL;
        float input[18], output[18], weights[2] = { 3.0f, 3.0f };
        int i;
        for (i = 0; i < 18; ++i) { input[i] = 2.0f; }
        CHECK(vkdnnMalloc(&xp, sizeof(input)) == vkdnnSuccess);
        CHECK(vkdnnMalloc(&yp, sizeof(output)) == vkdnnSuccess);
        CHECK(vkdnnMalloc(&wp, sizeof(weights)) == vkdnnSuccess);
        CHECK(vkdnnMemcpy(xp, input, sizeof(input), vkdnnMemcpyHostToDevice) == vkdnnSuccess);
        CHECK(vkdnnMemcpy(wp, weights, sizeof(weights), vkdnnMemcpyHostToDevice) == vkdnnSuccess);
        CHECK(vkdnnMemset(yp, 0xff, sizeof(output)) == vkdnnSuccess);
        OK(vkdnnFindConvolutionForwardAlgorithmEx(handle, x, xp, w, wp, conv, y, yp, 2, &count, f, NULL, 0));
        CHECK(count == 1 && f[0].algo == VKDNN_CONVOLUTION_FWD_ALGO_DIRECT && f[0].status == VKDNN_STATUS_SUCCESS);
        CHECK(isnan(f[0].time) && f[0].memory == 0);
        CHECK(memcmp(&f[1], &saved[1], sizeof(f[1])) == 0);
        CHECK(vkdnnMemcpy(output, yp, sizeof(output), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        for (i = 0; i < 18; ++i) { CHECK(isnan(output[i])); }
        {
            const float alpha = 1, beta = 0;
            OK(vkdnnConvolutionForward(handle, &alpha, x, xp, w, wp, conv, f[0].algo, NULL, 0, &beta, y, yp));
        }
        CHECK(vkdnnMemcpy(output, yp, sizeof(output), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        for (i = 0; i < 18; ++i) { CHECK(output[i] == 6.0f); }
        memcpy(ds, d, sizeof(d));
        OK(vkdnnFindConvolutionBackwardDataAlgorithmEx(handle, w, wp, y, yp, conv, x, xp, 2, &count, d, NULL, 0));
        CHECK(count == 1 && d[0].algo == VKDNN_CONVOLUTION_BWD_DATA_ALGO_0 && d[0].status == VKDNN_STATUS_SUCCESS);
        CHECK(isfinite(d[0].time) && d[0].time >= 0.0f && d[0].memory == 0);
        CHECK(memcmp(&d[1], &ds[1], sizeof(d[1])) == 0);
        CHECK(vkdnnMemcpy(output, xp, sizeof(output), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        for (i = 0; i < 18; ++i) { CHECK(output[i] == 2.0f); }
        {
            const float alpha = 1, beta = 0;
            OK(vkdnnConvolutionBackwardData(handle, &alpha, w, wp, y, yp, conv, d[0].algo, NULL, 0, &beta, x, xp));
        }
        CHECK(vkdnnMemcpy(output, xp, sizeof(output), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        for (i = 0; i < 18; ++i) { CHECK(output[i] == 18.0f); }
        memcpy(bs, b, sizeof(b));
        OK(vkdnnFindConvolutionBackwardFilterAlgorithmEx(handle, x, xp, y, yp, conv, w, wp, 2, &count, b, NULL, 0));
        CHECK(count == 2 && b[0].status == VKDNN_STATUS_SUCCESS && b[1].status == VKDNN_STATUS_SUCCESS);
        CHECK(b[0].time <= b[1].time);
        CHECK(isfinite(b[0].time) && b[0].time >= 0.0f && b[0].memory == 0);
        CHECK(vkdnnMemcpy(weights, wp, sizeof(weights), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        CHECK(weights[0] == 3.0f && weights[1] == 3.0f);
        {
            const float alpha = 1, beta = 0;
            OK(vkdnnConvolutionBackwardFilter(handle, &alpha, x, xp, y, yp, conv, b[0].algo, NULL, 0, &beta, w, wp));
        }
        CHECK(vkdnnMemcpy(weights, wp, sizeof(weights), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        CHECK(weights[0] == 972.0f && weights[1] == 972.0f);
        printf("FindEx ms (forward unmeasured, backward GPU when supported): forward=%g dx=%g dw=%g\n", f[0].time, d[0].time, b[0].time);
        memcpy(saved, f, sizeof(f));
        count = -1;
        OK(vkdnnFindConvolutionForwardAlgorithmEx(handle, x, xp, w, wp, conv, y, xp, 2, &count, f, NULL, 0));
        CHECK(count == 1 && isnan(f[0].time));
        CHECK(vkdnnMemcpy(output, yp, sizeof(output), vkdnnMemcpyDeviceToHost) == vkdnnSuccess);
        for (i = 0; i < 18; ++i) { CHECK(output[i] == 6.0f); }
        CHECK(vkdnnFree(wp) == vkdnnSuccess);
        CHECK(vkdnnFree(yp) == vkdnnSuccess);
        CHECK(vkdnnFree(xp) == vkdnnSuccess);
    }
    memcpy(saved, f, sizeof(f));
    memcpy(ds, d, sizeof(d));
    memcpy(bs, b, sizeof(b));
    count = -1;
    CHECK(vkdnnFindConvolutionForwardAlgorithmEx(handle, x, NULL, w, NULL, conv, y, NULL, 2, &count, f, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionBackwardDataAlgorithmEx(handle, w, NULL, y, NULL, conv, x, NULL, 2, &count, d, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionBackwardFilterAlgorithmEx(handle, x, NULL, y, NULL, conv, w, NULL, 2, &count, b, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionForwardAlgorithmEx(handle, x, NULL, w, NULL, conv, y, NULL, 0, &count, f, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionBackwardDataAlgorithmEx(handle, w, NULL, y, NULL, conv, x, NULL, 1, NULL, d, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionBackwardFilterAlgorithmEx(handle, x, NULL, y, NULL, conv, w, NULL, 1, &count, NULL, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(count == -1 && memcmp(saved, f, sizeof(f)) == 0 && memcmp(ds, d, sizeof(d)) == 0 && memcmp(bs, b, sizeof(b)) == 0);
    CHECK(vkdnnGetConvolutionForwardAlgorithm_v7(handle, x, w, conv, y, 0, &count, f) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnGetConvolutionBackwardDataAlgorithm_v7(handle, w, y, conv, x, -1, &count, d) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x, y, conv, w, 1, NULL, b) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnGetConvolutionForwardAlgorithm_v7(handle, x, w, conv, y, 1, &count, NULL) == VKDNN_STATUS_BAD_PARAM);
    OK(vkdnnSetTensor4dDescriptor(y, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 2, 2, 2));
    CHECK(vkdnnFindConvolutionForwardAlgorithmEx(handle, x, NULL, w, NULL, conv, y, NULL, 1, &count, f, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionBackwardDataAlgorithmEx(handle, w, NULL, y, NULL, conv, x, NULL, 1, &count, d, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnFindConvolutionBackwardFilterAlgorithmEx(handle, x, NULL, y, NULL, conv, w, NULL, 1, &count, b, NULL, 0) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnGetConvolutionForwardAlgorithm_v7(handle, x, w, conv, y, 1, &count, f) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnGetConvolutionBackwardDataAlgorithm_v7(handle, w, y, conv, x, 1, &count, d) == VKDNN_STATUS_BAD_PARAM);
    CHECK(vkdnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x, y, conv, w, 1, &count, b) == VKDNN_STATUS_BAD_PARAM);
    CHECK(count == -1 && memcmp(saved, f, sizeof(f)) == 0 && memcmp(ds, d, sizeof(d)) == 0 && memcmp(bs, b, sizeof(b)) == 0);
    OK(vkdnnDestroyConvolutionDescriptor(conv));
    OK(vkdnnDestroyFilterDescriptor(w));
    OK(vkdnnDestroyTensorDescriptor(y));
    OK(vkdnnDestroyTensorDescriptor(x));
    OK(vkdnnDestroy(handle));
    CHECK(vkdnnDeviceReset() == vkdnnSuccess);
    puts("Convolution query C API passed");
    return 0;
}
