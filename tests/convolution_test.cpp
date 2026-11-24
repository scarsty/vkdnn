#include "vkdnn/vkdnn_ops.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define REQUIRE(x) \
    do \
    { \
        if (!(x)) \
        { \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); \
            std::exit(1); \
        } \
    } while (0)
#define OK(x) REQUIRE((x) == VKDNN_STATUS_SUCCESS)
#define RT(x) REQUIRE((x) == vkdnnSuccess)
int main()
{
    int devices = 0;
    auto status = vkdnnGetDeviceCount(&devices);
    if (status == vkdnnErrorNoDevice || (status == vkdnnSuccess && !devices)) { return 77; }
    RT(status);
    vkdnnDeviceProp prop{};
    RT(vkdnnGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n", prop.name);
    vkdnnHandle_t handle;
    vkdnnTensorDescriptor_t xd, yd;
    vkdnnFilterDescriptor_t fd;
    vkdnnConvolutionDescriptor_t cd;
    OK(vkdnnCreate(&handle));
    OK(vkdnnCreateTensorDescriptor(&xd));
    OK(vkdnnCreateTensorDescriptor(&yd));
    OK(vkdnnCreateFilterDescriptor(&fd));
    OK(vkdnnCreateConvolutionDescriptor(&cd));
    int on = -1, oc = -1, oh = -1, ow = -1;
    REQUIRE(vkdnnGetConvolution2dForwardOutputDim(cd, xd, fd, &on, &oc, &oh, &ow) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(on == -1 && oc == -1 && oh == -1 && ow == -1);
    // N,C,H,W,K,KH,KW,PH,PW,SH,SW,DH,DW,G (includes a depthwise multiplier).
    const int cases[][14] = {
        { 2, 3, 5, 7, 4, 2, 3, 1, 2, 1, 2, 1, 1, 1 },
        { 2, 4, 7, 8, 6, 3, 2, 2, 1, 2, 1, 2, 2, 2 },
        { 2, 3, 5, 7, 6, 2, 3, 1, 2, 1, 2, 2, 1, 3 },
        { 1, 2, 3, 4, 2, 1, 1, 0, 0, 1, 1, 1, 1, 2 },
        { 1, 1, 2, 3, 1, 2, 2, 3, 2, 2, 2, 1, 1, 1 }
    };
    for (const auto& dims : cases)
    {
        int n = dims[0], c = dims[1], h = dims[2], w = dims[3], k = dims[4], kh = dims[5], kw = dims[6];
        int ph = dims[7], pw = dims[8], sh = dims[9], sw = dims[10], dh = dims[11], dw = dims[12], g = dims[13];
        int xs = n * c * h * w, fs = k * (c / g) * kh * kw;
        OK(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w));
        OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, k, c / g, kh, kw));
        OK(vkdnnSetConvolutionGroupCount(cd, g));
        for (int mode = 0; mode < 2; ++mode)
        {
            OK(vkdnnSetConvolution2dDescriptor(cd, ph, pw, sh, sw, dh, dw, vkdnnConvolutionMode_t(mode), VKDNN_DATA_FLOAT));
            OK(vkdnnGetConvolution2dForwardOutputDim(cd, xd, fd, &on, &oc, &oh, &ow));
            REQUIRE(on == n && oc == k && oh == (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1 && ow == (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
            int ys = n * k * oh * ow;
            OK(vkdnnSetTensor4dDescriptor(yd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, k, oh, ow));
            size_t workspace = 123;
            OK(vkdnnGetConvolutionForwardWorkspaceSize(handle, xd, fd, cd, yd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, &workspace));
            REQUIRE(workspace == 0);
            workspace = 123;
            REQUIRE(vkdnnGetConvolutionForwardWorkspaceSize(handle, xd, fd, cd, yd, VKDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(workspace == 123);
            std::vector<float> x(xs), f(fs), initial(ys), actual(ys), expected(ys);
            for (int i = 0; i < xs; ++i) { x[i] = float(i % 19 - 9) * 0.11f; }
            for (int i = 0; i < fs; ++i) { f[i] = float(i % 13 - 6) * 0.13f; }
            void *dx, *df, *dy;
            RT(vkdnnMalloc(&dx, xs * 4 + 4));
            RT(vkdnnMalloc(&df, fs * 4 + 4));
            RT(vkdnnMalloc(&dy, ys * 4 + 4));
            void* xp = static_cast<char*>(dx) + 4;
            void* fp = static_cast<char*>(df) + 4;
            void* yp = static_cast<char*>(dy) + 4;
            RT(vkdnnMemcpy(xp, x.data(), xs * 4, vkdnnMemcpyHostToDevice));
            RT(vkdnnMemcpy(fp, f.data(), fs * 4, vkdnnMemcpyHostToDevice));
            float alpha = 1.25f, beta = 0;
            auto forward = [&](const void* a, const void* input, const void* weights, void* out)
            {
                return vkdnnConvolutionForward(handle, a, xd, input, fd, weights, cd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, nullptr, 0, &beta, yd, out);
            };
            for (float b : { 0.f, -0.5f })
            {
                beta = b;
                initial.assign(ys, NAN);
                if (b != 0) { initial.assign(ys, 2.f); }
                RT(vkdnnMemcpy(yp, initial.data(), ys * 4, vkdnnMemcpyHostToDevice));
                OK(forward(&alpha, xp, fp, yp));
                for (int bn = 0; bn < n; ++bn)
                {
                    for (int out = 0; out < k; ++out)
                    {
                        for (int oy = 0; oy < oh; ++oy)
                        {
                            for (int ox = 0; ox < ow; ++ox)
                            {
                                double sum = 0;
                                for (int ic = 0; ic < c / g; ++ic)
                                {
                                    for (int a = 0; a < kh; ++a)
                                    {
                                        for (int bcol = 0; bcol < kw; ++bcol)
                                        {
                                            int iy = oy * sh - ph + a * dh, ix = ox * sw - pw + bcol * dw;
                                            if (iy < 0 || ix < 0 || iy >= h || ix >= w) { continue; }
                                            int fy = a, fx = bcol;
                                            if (!mode)
                                            {
                                                fy = kh - 1 - a;
                                                fx = kw - 1 - bcol;
                                            }
                                            int in = out / (k / g) * (c / g) + ic;
                                            sum += double(x[((bn * c + in) * h + iy) * w + ix]) * f[((out * (c / g) + ic) * kh + fy) * kw + fx];
                                        }
                                    }
                                }
                                expected[((bn * k + out) * oh + oy) * ow + ox] = float(alpha * sum + b * 2);
                            }
                        }
                    }
                }
                RT(vkdnnMemcpy(actual.data(), yp, ys * 4, vkdnnMemcpyDeviceToHost));
                for (int i = 0; i < ys; ++i) { REQUIRE(std::abs(actual[i] - expected[i]) < 5e-5f * std::max(1.f, std::abs(expected[i]))); }
            }
            alpha = 0;
            beta = 0;
            OK(forward(&alpha, nullptr, nullptr, yp));
            RT(vkdnnMemcpy(actual.data(), yp, ys * 4, vkdnnMemcpyDeviceToHost));
            for (float v : actual) { REQUIRE(v == 0); }
            beta = -0.5f;
            initial.assign(ys, 2.f);
            RT(vkdnnMemcpy(yp, initial.data(), ys * 4, vkdnnMemcpyHostToDevice));
            OK(forward(&alpha, nullptr, nullptr, yp));
            RT(vkdnnMemcpy(actual.data(), yp, ys * 4, vkdnnMemcpyDeviceToHost));
            for (float v : actual) { REQUIRE(v == -1); }
            alpha = 1;
            beta = 0;
            initial.assign(ys, NAN);
            RT(vkdnnMemcpy(yp, initial.data(), ys * 4, vkdnnMemcpyHostToDevice));
            beta = 1;
            OK(forward(&alpha, xp, fp, yp));
            RT(vkdnnMemcpy(actual.data(), yp, ys * 4, vkdnnMemcpyDeviceToHost));
            for (float value : actual) { REQUIRE(std::isnan(value)); }
            beta = 0;
            REQUIRE(forward(nullptr, xp, fp, yp) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(forward(xp, xp, fp, yp) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(forward(&alpha, x.data(), fp, yp) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(forward(&alpha, static_cast<char*>(xp) + 1, fp, yp) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(forward(&alpha, static_cast<char*>(xp) + 4, fp, yp) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(forward(&alpha, xp, static_cast<char*>(fp) + 4, yp) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(forward(&alpha, xp, fp, static_cast<char*>(yp) + 4) == VKDNN_STATUS_BAD_PARAM);
            void* alias;
            RT(vkdnnMalloc(&alias, size_t(std::max({ xs, fs, ys })) * 4));
            REQUIRE(forward(&alpha, alias, fp, alias) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(forward(&alpha, xp, alias, alias) == VKDNN_STATUS_NOT_SUPPORTED);
            RT(vkdnnFree(alias));
            if (devices > 1)
            {
                RT(vkdnnSetDevice(1));
                REQUIRE(forward(&alpha, xp, fp, yp) == VKDNN_STATUS_BAD_PARAM);
                void* foreign;
                RT(vkdnnMalloc(&foreign, xs * 4));
                RT(vkdnnSetDevice(0));
                REQUIRE(forward(&alpha, foreign, fp, yp) == VKDNN_STATUS_BAD_PARAM);
                RT(vkdnnSetDevice(1));
                RT(vkdnnFree(foreign));
                RT(vkdnnSetDevice(0));
            }
            RT(vkdnnDeviceSynchronize());
            RT(vkdnnFree(dx));
            RT(vkdnnFree(df));
            RT(vkdnnFree(dy));
            REQUIRE(forward(&alpha, xp, fp, yp) == VKDNN_STATUS_BAD_PARAM);
        }
    }
    if (devices < 2) { std::puts("SKIP: multi-device ownership (one GPU)"); }
    REQUIRE(vkdnnSetConvolutionGroupCount(cd, 0) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetConvolution2dDescriptor(cd, -1, 0, 1, 1, 1, 1, VKDNN_CONVOLUTION, VKDNN_DATA_FLOAT) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 0, 1, 1, 1, VKDNN_CONVOLUTION, VKDNN_DATA_FLOAT) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 0, 1, VKDNN_CONVOLUTION, VKDNN_DATA_FLOAT) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, vkdnnConvolutionMode_t(99), VKDNN_DATA_FLOAT) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, VKDNN_CONVOLUTION, VKDNN_DATA_HALF) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 0, 1, 1, 1) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_HALF, VKDNN_TENSOR_NCHW, 1, 1, 1, 1) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NHWC, 1, 1, 1, 1) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, INT_MAX, INT_MAX, INT_MAX, INT_MAX) == VKDNN_STATUS_NOT_SUPPORTED);
    OK(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 2, 3, 4));
    OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 3, 1, 2, 2));
    OK(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
    for (int groups : { 1, 2, 3 })
    {
        OK(vkdnnSetConvolutionGroupCount(cd, groups));
        REQUIRE(vkdnnGetConvolution2dForwardOutputDim(cd, xd, fd, &on, &oc, &oh, &ow) == VKDNN_STATUS_BAD_PARAM);
    }
    OK(vkdnnSetConvolutionGroupCount(cd, 1));
    OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 1, 2, 9, 9));
    REQUIRE(vkdnnGetConvolution2dForwardOutputDim(cd, xd, fd, &on, &oc, &oh, &ow) == VKDNN_STATUS_BAD_PARAM);
    OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 1, 2, 2, 2));
    OK(vkdnnSetConvolution2dDescriptor(cd, INT_MAX, INT_MAX, 1, 1, INT_MAX, INT_MAX, VKDNN_CONVOLUTION, VKDNN_DATA_FLOAT));
    REQUIRE(vkdnnGetConvolution2dForwardOutputDim(cd, xd, fd, &on, &oc, &oh, &ow) == VKDNN_STATUS_NOT_SUPPORTED);
    OK(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, VKDNN_CONVOLUTION, VKDNN_DATA_FLOAT));
    OK(vkdnnSetTensor4dDescriptor(yd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 1, 1, 1));
    size_t bytes = 123;
    REQUIRE(vkdnnGetConvolutionForwardWorkspaceSize(handle, xd, fd, cd, yd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, &bytes) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(bytes == 123);
    OK(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 1, 1, 1));
    OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 1, 1, 1, 1));
    void* data;
    RT(vkdnnMalloc(&data, 12));
    void* weights = static_cast<char*>(data) + 4;
    void* output = static_cast<char*>(data) + 8;
    float value = NAN, weight = 1, a = 1, b = 0, result;
    RT(vkdnnMemcpy(data, &value, 4, vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(weights, &weight, 4, vkdnnMemcpyHostToDevice));
    OK(vkdnnConvolutionForward(handle, &a, xd, data, fd, weights, cd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, data, 12, &b, yd, output));
    RT(vkdnnMemcpy(&result, output, 4, vkdnnMemcpyDeviceToHost));
    REQUIRE(std::isnan(result));
    value = 1;
    weight = NAN;
    RT(vkdnnMemcpy(data, &value, 4, vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(weights, &weight, 4, vkdnnMemcpyHostToDevice));
    OK(vkdnnConvolutionForward(handle, &a, xd, data, fd, weights, cd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, nullptr, 0, &b, yd, output));
    RT(vkdnnMemcpy(&result, output, 4, vkdnnMemcpyDeviceToHost));
    REQUIRE(std::isnan(result));
    RT(vkdnnFree(data));
    float one = 1, zero = 0;
    REQUIRE(vkdnnConvolutionForward(handle, &one, xd, nullptr, fd, nullptr, cd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, nullptr, 0, &zero, yd, nullptr) == VKDNN_STATUS_BAD_PARAM);
    OK(vkdnnDestroyConvolutionDescriptor(cd));
    OK(vkdnnDestroyFilterDescriptor(fd));
    REQUIRE(vkdnnDestroyConvolutionDescriptor(cd) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnDestroyFilterDescriptor(fd) == VKDNN_STATUS_BAD_PARAM);
    OK(vkdnnDestroyTensorDescriptor(xd));
    OK(vkdnnDestroyTensorDescriptor(yd));
    OK(vkdnnDestroy(handle));
    RT(vkdnnDeviceReset());
    std::puts("PASS: direct convolution CPU references, groups/depthwise, flip, alpha/beta, bounds");
}
