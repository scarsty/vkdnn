#include "vkdnn/vkdnn_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define REQUIRE(condition) \
    do \
    { \
        if (!(condition)) \
        { \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
            std::exit(1); \
        } \
    } while (0)
#define OK(call) REQUIRE((call) == VKDNN_STATUS_SUCCESS)
#define RT(call) REQUIRE((call) == vkdnnSuccess)
static void compare(const std::vector<float>& a, const std::vector<float>& b)
{
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isnan(b[i]))
        {
            if (!std::isnan(a[i]))
            {
                std::fprintf(stderr, "NaN mismatch index %zu actual %g\n", i, a[i]);
            }
            REQUIRE(std::isnan(a[i]));
        }
        else if (std::isinf(b[i]))
        {
            REQUIRE(a[i] == b[i]);
        }
        else if (!(std::abs(a[i] - b[i]) < 3e-5f * std::max(1.f, std::abs(b[i]))))
        {
            std::fprintf(stderr, "index %zu actual %g expected %g\n", i, a[i], b[i]);
            REQUIRE(false);
        }
    }
}
int main()
{
    int count = 0;
    auto status = vkdnnGetDeviceCount(&count);
    if (status == vkdnnErrorNoDevice || (status == vkdnnSuccess && count == 0))
    {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    RT(status);
    vkdnnDeviceProp prop{};
    RT(vkdnnGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n", prop.name);
    vkdnnHandle_t handle = nullptr;
    vkdnnTensorDescriptor_t xd = nullptr, yd = nullptr;
    vkdnnActivationDescriptor_t activation = nullptr;
    vkdnnPoolingDescriptor_t pool = nullptr;
    OK(vkdnnCreate(&handle));
    OK(vkdnnCreateTensorDescriptor(&xd));
    OK(vkdnnCreateTensorDescriptor(&yd));
    OK(vkdnnCreateActivationDescriptor(&activation));
    OK(vkdnnCreatePoolingDescriptor(&pool));
    const int n = 2, c = 3, h = 3, w = 5, size = n * c * h * w;
    OK(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w));
    OK(vkdnnSetTensor4dDescriptor(yd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w));
    std::vector<float> x(size), initial(size), actual(size), expected(size);
    for (int i = 0; i < size; ++i)
    {
        x[i] = float(i % 17 - 8) * 0.7f;
        initial[i] = float(i % 7 - 3);
    }
    void* dx = nullptr;
    void* dy = nullptr;
    RT(vkdnnMalloc(&dx, size * 4 + 4));
    RT(vkdnnMalloc(&dy, size * 4 + 4));
    // Exercise float-aligned offsets without requiring storage-buffer alignment.
    void* xp = static_cast<char*>(dx) + 4;
    void* yp = static_cast<char*>(dy) + 4;
    auto upload = [&]
    {
        RT(vkdnnMemcpy(xp, x.data(), size * 4, vkdnnMemcpyHostToDevice));
        RT(vkdnnMemcpy(yp, initial.data(), size * 4, vkdnnMemcpyHostToDevice));
    };
    auto download = [&](size_t length)
    {
        actual.resize(length);
        RT(vkdnnMemcpy(actual.data(), yp, length * 4, vkdnnMemcpyDeviceToHost));
    };
    float alpha = 1.5f, beta = -0.5f;
    for (int nan = 0; nan < 2; ++nan)
    {
        x[7] = std::numeric_limits<float>::quiet_NaN();
        OK(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_RELU, vkdnnNanPropagation_t(nan), 0));
        upload();
        OK(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, yp));
        for (int i = 0; i < size; ++i)
        {
            float v = std::max(x[i], 0.f);
            if (std::isnan(v) && !nan)
            {
                v = 0;
            }
            expected[i] = alpha * v + beta * initial[i];
        }
        download(size);
        compare(actual, expected);
    }
    x[7] = 2;
    beta = 0;
    initial.assign(size, std::numeric_limits<float>::quiet_NaN());
    upload();
    OK(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, yp));
    for (int i = 0; i < size; ++i)
    {
        expected[i] = alpha * std::max(x[i], 0.f);
    }
    download(size);
    compare(actual, expected);
    beta = 1;
    upload();
    OK(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, yp));
    download(size);
    for (float value : actual)
    {
        REQUIRE(std::isnan(value));
    }
    beta = 0;
    for (int nan = 0; nan < 2; ++nan)
    {
        OK(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_SIGMOID, vkdnnNanPropagation_t(nan), 0));
        const float values[] = { -1000.f, -100.f, -2.f, 0.f, 2.f, 100.f, 1000.f, INFINITY, -INFINITY, NAN };
        for (int i = 0; i < size; ++i) { x[i] = values[i % 10]; }
        for (float b : { 0.f, -0.5f })
        {
            beta = b;
            initial.assign(size, NAN);
            if (b != 0) { initial.assign(size, 2.f); }
            upload();
            OK(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, yp));
            for (int i = 0; i < size; ++i)
            {
                double e = std::exp(-std::abs(double(x[i])));
                double s = e / (1 + e);
                if (x[i] >= 0) { s = 1 / (1 + e); }
                expected[i] = float(alpha * s + b * 2);
            }
            download(size);
            compare(actual, expected);
        }
    }
    for (int nan = 0; nan < 2; ++nan)
    {
        OK(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_TANH, vkdnnNanPropagation_t(nan), 0));
        REQUIRE(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_TANH, VKDNN_PROPAGATE_NAN, 1) == VKDNN_STATUS_NOT_SUPPORTED);
        const float values[] = { -INFINITY, -3.4e38f, -100.f, -2.f, -0.000001f, -0.f, 0.f, 0.000001f, 0.001f, 2.f, 100.f, 3.4e38f, INFINITY, NAN };
        for (int i = 0; i < size; ++i) { x[i] = values[i % 14]; }
        for (float b : { 0.f, -0.5f, 1.f })
        {
            beta = b;
            initial.assign(size, NAN);
            if (b != 0) { initial.assign(size, 2.f); }
            upload();
            OK(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, yp));
            for (int i = 0; i < size; ++i) { expected[i] = float(alpha * std::tanh(double(x[i])) + b * 2); }
            download(size);
            compare(actual, expected);
        }
    }
    // All 16 per-axis broadcast combinations, including bias and same-shape residual.
    vkdnnTensorDescriptor_t ad = nullptr;
    OK(vkdnnCreateTensorDescriptor(&ad));
    for (int mask = 0; mask < 16; ++mask)
    {
        int an = (mask & 8) ? n : 1, ac = (mask & 4) ? c : 1;
        int ah = (mask & 2) ? h : 1, aw = (mask & 1) ? w : 1;
        OK(vkdnnSetTensor4dDescriptor(ad, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, an, ac, ah, aw));
        for (int i = 0; i < size; ++i) { x[i] = float(i % 13 - 6); }
        for (float a : { 0.f, 1.f, -1.5f })
        {
            for (float b : { 0.f, 1.f, -0.25f })
            {
                initial.assign(size, NAN);
                if (b != 0) { initial.assign(size, 2.f); }
                upload();
                OK(vkdnnAddTensor(handle, &a, ad, xp, &b, yd, yp));
                for (int i = 0; i < size; ++i)
                {
                    int xx = i % w, yy = (i / w) % h, cc = (i / (w * h)) % c, nn = i / (w * h * c);
                    int j = (((nn % an) * ac + cc % ac) * ah + yy % ah) * aw + xx % aw;
                    expected[i] = a * x[j] + b * 2;
                }
                download(size);
                compare(actual, expected);
            }
        }
    }
    OK(vkdnnSetTensor4dDescriptor(ad, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w));
    alpha = 1;
    beta = 0;
    for (int i = 0; i < size; ++i) { x[i] = float(i); }
    x[0] = NAN;
    x[1] = INFINITY;
    x[2] = -INFINITY;
    initial.assign(size, NAN);
    upload();
    OK(vkdnnAddTensor(handle, &alpha, ad, xp, &beta, yd, yp));
    download(size);
    compare(actual, x);
    for (float a : { NAN, INFINITY, -INFINITY })
    {
        alpha = a;
        initial.assign(size, 2.f);
        upload();
        OK(vkdnnAddTensor(handle, &alpha, ad, xp, &beta, yd, yp));
        for (int i = 0; i < size; ++i) { expected[i] = alpha * x[i]; }
        download(size);
        compare(actual, expected);
    }
    alpha = 1;
    beta = NAN;
    OK(vkdnnAddTensor(handle, &alpha, ad, xp, &beta, yd, yp));
    download(size);
    compare(actual, std::vector<float>(size, NAN));
    beta = 0;
    initial.assign(size, 7.f);
    upload();
    auto rejected = [&](vkdnnStatus_t s, vkdnnStatus_t wanted)
    {
        REQUIRE(s == wanted);
        download(size);
        compare(actual, initial);
    };
    rejected(vkdnnAddTensor(nullptr, &alpha, ad, xp, &beta, yd, yp), VKDNN_STATUS_NOT_INITIALIZED);
    rejected(vkdnnAddTensor(handle, nullptr, ad, xp, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, nullptr, xp, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, ad, xp, nullptr, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, xp, ad, xp, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, ad, nullptr, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, ad, x.data(), &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, ad, static_cast<char*>(xp) + 1, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, ad, static_cast<char*>(xp) + 4, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    rejected(vkdnnAddTensor(handle, &alpha, ad, yp, &beta, yd, yp), VKDNN_STATUS_NOT_SUPPORTED);
    rejected(vkdnnAddTensor(handle, &alpha, ad, dy, &beta, yd, yp), VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnAddTensor(handle, &alpha, ad, xp, &beta, yd, static_cast<char*>(yp) + 4) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnAddTensor(handle, &alpha, ad, xp, &beta, yd, static_cast<char*>(yp) + 1) == VKDNN_STATUS_BAD_PARAM);
    for (int axis = 0; axis < 4; ++axis)
    {
        int dims[] = { n, c, h, w };
        dims[axis] += 1;
        OK(vkdnnSetTensor4dDescriptor(ad, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, dims[0], dims[1], dims[2], dims[3]));
        rejected(vkdnnAddTensor(handle, &alpha, ad, xp, &beta, yd, yp), VKDNN_STATUS_BAD_PARAM);
    }
    OK(vkdnnSetTensor4dDescriptor(ad, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w));
    alpha = 0;
    beta = 2;
    OK(vkdnnAddTensor(handle, &alpha, ad, nullptr, &beta, yd, yp));
    download(size);
    compare(actual, std::vector<float>(size, 14.f));
    beta = 0;
    OK(vkdnnAddTensor(handle, &alpha, ad, yp, &beta, yd, yp));
    download(size);
    compare(actual, std::vector<float>(size, 0.f));
    void* shared = nullptr;
    RT(vkdnnMalloc(&shared, size * 8));
    void* sharedOutput = static_cast<char*>(shared) + size * 4;
    RT(vkdnnMemcpy(shared, x.data(), size * 4, vkdnnMemcpyHostToDevice));
    alpha = 1;
    OK(vkdnnAddTensor(handle, &alpha, ad, shared, &beta, yd, sharedOutput));
    RT(vkdnnMemcpy(actual.data(), sharedOutput, size * 4, vkdnnMemcpyDeviceToHost));
    compare(actual, x);
    RT(vkdnnFree(shared));
    alpha = 0;
    if (count > 1)
    {
        RT(vkdnnSetDevice(1));
        REQUIRE(vkdnnAddTensor(handle, &alpha, ad, nullptr, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
        RT(vkdnnSetDevice(0));
        void* foreign = nullptr;
        RT(vkdnnSetDevice(1));
        RT(vkdnnMalloc(&foreign, size * 4));
        RT(vkdnnSetDevice(0));
        alpha = 1;
        REQUIRE(vkdnnAddTensor(handle, &alpha, ad, foreign, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
        REQUIRE(vkdnnActivationForward(handle, activation, &alpha, xd, foreign, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
        RT(vkdnnFree(foreign));
    }
    else
    {
        std::puts("SKIP: cross-device bias/tanh (single GPU)");
    }
    OK(vkdnnDestroyTensorDescriptor(ad));
    alpha = 0;
    beta = 0;
    OK(vkdnnActivationForward(handle, activation, &alpha, xd, nullptr, &beta, yd, yp));
    download(size);
    compare(actual, std::vector<float>(size, 0));
    REQUIRE(vkdnnActivationForward(handle, activation, xp, xd, xp, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    alpha = 1.5f;
    OK(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_RELU, VKDNN_PROPAGATE_NAN, 0));
    REQUIRE(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, xp) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnActivationForward(handle, activation, &alpha, xd, static_cast<char*>(xp) + 1, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnActivationForward(handle, activation, &alpha, xd, static_cast<char*>(xp) + 4, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnActivationForward(handle, activation, nullptr, xd, xp, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnActivationForward(nullptr, activation, &alpha, xd, xp, &beta, yd, yp) == VKDNN_STATUS_NOT_INITIALIZED);
    REQUIRE(vkdnnActivationForward(handle, activation, &alpha, xd, x.data(), &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NHWC, VKDNN_DATA_FLOAT, n, c, h, w) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_HALF, n, c, h, w) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 0, c, h, w) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, INT32_MAX, INT32_MAX, h, w) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_IDENTITY, VKDNN_PROPAGATE_NAN, 0) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_RELU, vkdnnNanPropagation_t(7), 0) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetPooling2dDescriptor(pool, VKDNN_POOLING_MAX, VKDNN_PROPAGATE_NAN, 2, 2, 2, 0, 1, 1) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnSetPooling2dDescriptor(pool, VKDNN_POOLING_MAX_DETERMINISTIC, VKDNN_PROPAGATE_NAN, 2, 2, 0, 0, 1, 1) == VKDNN_STATUS_NOT_SUPPORTED);
    for (int mode = 0; mode < 3; ++mode)
    {
        for (int nan = 0; nan < 2; ++nan)
        {
            for (int withNan = 0; withNan < 2; ++withNan)
            {
                x[7] = 2;
                if (withNan)
                {
                    x[7] = std::numeric_limits<float>::quiet_NaN();
                }
                OK(vkdnnSetPooling2dDescriptor(pool, vkdnnPoolingMode_t(mode), vkdnnNanPropagation_t(nan), 2, 3, 1, 1, 2, 2));
                int on = -1, oc = -1, oh = -1, ow = -1;
                OK(vkdnnGetPooling2dForwardOutputDim(pool, xd, &on, &oc, &oh, &ow));
                REQUIRE(on == n && oc == c && oh == 2 && ow == 3);
                OK(vkdnnSetTensor4dDescriptor(yd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, on, oc, oh, ow));
                initial.assign(size, 2.f);
                beta = 0.25f;
                upload();
                OK(vkdnnPoolingForward(handle, pool, &alpha, xd, xp, &beta, yd, yp));
                expected.resize(on * oc * oh * ow);
                for (int nc = 0; nc < n * c; ++nc)
                {
                    for (int yy = 0; yy < oh; ++yy)
                    {
                        for (int xx = 0; xx < ow; ++xx)
                        {
                            float v = 0;
                            if (mode == 0)
                            {
                                v = -std::numeric_limits<float>::infinity();
                            }
                            int valid = 0;
                            bool bad = false;
                            for (int a = 0; a < 2; ++a)
                            {
                                for (int b = 0; b < 3; ++b)
                                {
                                    int iy = yy * 2 - 1 + a, ix = xx * 2 - 1 + b;
                                    if (iy < 0 || ix < 0 || iy >= h || ix >= w)
                                    {
                                        continue;
                                    }
                                    ++valid;
                                    float t = x[nc * h * w + iy * w + ix];
                                    if (std::isnan(t))
                                    {
                                        bad = bad || nan;
                                        continue;
                                    }
                                    if (mode == 0)
                                    {
                                        v = std::max(v, t);
                                    }
                                    else
                                    {
                                        v += t;
                                    }
                                }
                            }
                            if (mode == 1)
                            {
                                v /= 6;
                            }
                            if (mode == 2)
                            {
                                v /= valid;
                            }
                            if (bad)
                            {
                                v = std::numeric_limits<float>::quiet_NaN();
                            }
                            expected[nc * oh * ow + yy * ow + xx] = alpha * v + beta * 2;
                        }
                    }
                }
                download(expected.size());
                compare(actual, expected);
            }
        }
    }
    // Stable reduction over the entire instance or channels at each spatial point.
    OK(vkdnnSetTensor4dDescriptor(yd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w));
    expected.resize(size);
    for (int i = 0; i < size; ++i)
    {
        x[i] = 1000.f + float(i % 13 - 6);
    }
    for (int mode = 0; mode < 2; ++mode)
    {
        for (int log = 0; log < 2; ++log)
        {
            for (int nonzeroBeta = 0; nonzeroBeta < 2; ++nonzeroBeta)
            {
                beta = nonzeroBeta ? -0.5f : 0.f;
                initial.assign(size, 2.f);
                if (!nonzeroBeta)
                {
                    initial.assign(size, std::numeric_limits<float>::quiet_NaN());
                }
                upload();
                auto algo = VKDNN_SOFTMAX_ACCURATE;
                if (log)
                {
                    algo = VKDNN_SOFTMAX_LOG;
                }
                OK(vkdnnSoftmaxForward(handle, algo, vkdnnSoftmaxMode_t(mode), &alpha, xd, xp, &beta, yd, yp));
                int groups = n, length = c * h * w, stride = 1;
                if (mode)
                {
                    groups = n * h * w;
                    length = c;
                    stride = h * w;
                }
                for (int g = 0; g < groups; ++g)
                {
                    int base = g * length;
                    if (mode)
                    {
                        base = g / (h * w) * c * h * w + g % (h * w);
                    }
                    double m = -INFINITY;
                    for (int j = 0; j < length; ++j)
                    {
                        m = std::max(m, double(x[base + j * stride]));
                    }
                    double sum = 0;
                    for (int j = 0; j < length; ++j)
                    {
                        sum += std::exp(double(x[base + j * stride]) - m);
                    }
                    for (int j = 0; j < length; ++j)
                    {
                        double z = x[base + j * stride] - m;
                        double v = std::exp(z) / sum;
                        if (log)
                        {
                            v = z - std::log(sum);
                        }
                        expected[base + j * stride] = float(alpha * v + beta * 2);
                    }
                }
                download(size);
                compare(actual, expected);
            }
        }
    }
    REQUIRE(vkdnnSoftmaxForward(handle, VKDNN_SOFTMAX_FAST, VKDNN_SOFTMAX_MODE_INSTANCE, &alpha, xd, xp, &beta, yd, yp) == VKDNN_STATUS_NOT_SUPPORTED);
    for (float exceptional : { std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() })
    {
        x.assign(size, exceptional);
        beta = 0;
        upload();
        OK(vkdnnSoftmaxForward(handle, VKDNN_SOFTMAX_LOG, VKDNN_SOFTMAX_MODE_INSTANCE, &alpha, xd, xp, &beta, yd, yp));
        download(size);
        for (float value : actual)
        {
            REQUIRE(std::isnan(value));
        }
    }
    for (int i = 0; i < size; ++i)
    {
        x[i] = float(i % 13);
    }
    x[0] = std::numeric_limits<float>::quiet_NaN();
    beta = 0;
    upload();
    OK(vkdnnSoftmaxForward(handle, VKDNN_SOFTMAX_ACCURATE, VKDNN_SOFTMAX_MODE_INSTANCE, &alpha, xd, xp, &beta, yd, yp));
    download(size);
    for (int i = 0; i < c * h * w; ++i)
    {
        REQUIRE(std::isnan(actual[i]));
    }
    OK(vkdnnSetTensor4dDescriptor(xd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 1, 1, 1));
    OK(vkdnnSetTensor4dDescriptor(yd, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 1, 1, 1));
    x[0] = -1000;
    upload();
    OK(vkdnnSoftmaxForward(handle, VKDNN_SOFTMAX_LOG, VKDNN_SOFTMAX_MODE_CHANNEL, &alpha, xd, xp, &beta, yd, yp));
    download(1);
    REQUIRE(actual[0] == 0);
    OK(vkdnnSetPooling2dDescriptor(pool, VKDNN_POOLING_MAX, VKDNN_NOT_PROPAGATE_NAN, 1, 1, 0, 0, 1, 1));
    x[0] = -7;
    upload();
    OK(vkdnnPoolingForward(handle, pool, &alpha, xd, xp, &beta, yd, yp));
    download(1);
    REQUIRE(actual[0] == alpha * -7);
    OK(vkdnnSetPooling2dDescriptor(pool, VKDNN_POOLING_MAX, VKDNN_PROPAGATE_NAN, 2, 2, 0, 0, 1, 1));
    int on = 11, oc = 12, oh = 13, ow = 14;
    REQUIRE(vkdnnGetPooling2dForwardOutputDim(pool, xd, &on, &oc, &oh, &ow) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(on == 11 && oc == 12 && oh == 13 && ow == 14);
    REQUIRE(vkdnnPoolingForward(handle, pool, &alpha, xd, xp, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    RT(vkdnnFree(dx));
    RT(vkdnnFree(dy));
    REQUIRE(vkdnnActivationForward(handle, activation, &alpha, xd, xp, &beta, yd, yp) == VKDNN_STATUS_BAD_PARAM);
    OK(vkdnnDestroyPoolingDescriptor(pool));
    OK(vkdnnDestroyActivationDescriptor(activation));
    OK(vkdnnDestroyTensorDescriptor(xd));
    OK(vkdnnDestroyTensorDescriptor(yd));
    OK(vkdnnDestroy(handle));
    REQUIRE(vkdnnDestroy(handle) == VKDNN_STATUS_BAD_PARAM);
    REQUIRE(vkdnnDestroyTensorDescriptor(xd) == VKDNN_STATUS_BAD_PARAM);
    RT(vkdnnDeviceReset());
    std::puts("PASS: real Vulkan bias/broadcast/residual, ReLU/sigmoid/tanh/pooling/softmax, CPU references, alpha/beta, NaN, invalid inputs");
}
