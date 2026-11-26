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
struct Buffer
{
    void* base = nullptr;
    void* p = nullptr;
    int size;
    explicit Buffer(int n) : size(n)
    {
        RT(vkdnnMalloc(&base, size_t(n) * 4 + 4));
        p = static_cast<char*>(base) + 4;
    }
    ~Buffer() { RT(vkdnnFree(base)); }
    void put(const std::vector<float>& v) { RT(vkdnnMemcpy(p, v.data(), size_t(size) * 4, vkdnnMemcpyHostToDevice)); }
    std::vector<float> get()
    {
        std::vector<float> v(size);
        RT(vkdnnMemcpy(v.data(), p, size_t(size) * 4, vkdnnMemcpyDeviceToHost));
        return v;
    }
};
void check(const std::vector<float>& actual, const std::vector<double>& ref, float alpha, float beta)
{
    REQUIRE(actual.size() == ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double expected = alpha * ref[i] + beta * 2;
        REQUIRE(std::abs(actual[i] - expected) < 1e-4 * std::max(1.0, std::abs(expected)));
    }
}
void set(vkdnnTensorDescriptor_t d, int n, int c, int h, int w) { OK(vkdnnSetTensor4dDescriptor(d, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w)); }
template <class Objective>
void differences(std::vector<float>& values, const std::vector<double>& gradient, Objective objective)
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        float old = values[i];
        values[i] = old + 0.001f;
        double hi = objective();
        float upper = values[i];
        values[i] = old - 0.001f;
        double lo = objective();
        float lower = values[i];
        values[i] = old;
        REQUIRE(std::abs((hi - lo) / (upper - lower) - gradient[i]) < 2e-5 * std::max(1.0, std::abs(gradient[i])));
    }
}
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
    OK(vkdnnCreate(&handle));
    vkdnnTensorDescriptor_t xd, yd, bd, wrong;
    vkdnnFilterDescriptor_t fd;
    vkdnnConvolutionDescriptor_t cd;
    vkdnnPoolingDescriptor_t pd;
    OK(vkdnnCreateTensorDescriptor(&xd));
    OK(vkdnnCreateTensorDescriptor(&yd));
    OK(vkdnnCreateTensorDescriptor(&bd));
    OK(vkdnnCreateTensorDescriptor(&wrong));
    OK(vkdnnCreateFilterDescriptor(&fd));
    OK(vkdnnCreateConvolutionDescriptor(&cd));
    OK(vkdnnCreatePoolingDescriptor(&pd));
    set(wrong, 1, 1, 1, 1);
    const int cases[][14] = {
        { 2, 4, 4, 5, 6, 2, 3, 1, 2, 2, 1, 1, 2, 2 },
        { 2, 2, 3, 4, 4, 2, 2, 1, 1, 1, 2, 2, 1, 2 },
        { 1, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1, 1, 1, 1 },
        { 1, 1, 2, 3, 1, 2, 2, 3, 2, 2, 2, 1, 1, 1 }
    };
    for (const auto& d : cases)
    {
        for (int mode = 0; mode < 2; ++mode)
        {
            int n = d[0], c = d[1], h = d[2], w = d[3], k = d[4], kh = d[5], kw = d[6], ph = d[7], pw = d[8], sh = d[9], sw = d[10], dh = d[11], dw = d[12], g = d[13];
            int fc = c / g, xs = n * c * h * w, fs = k * fc * kh * kw, on, oc, oh, ow;
            set(xd, n, c, h, w);
            set(bd, 1, k, 1, 1);
            OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, k, fc, kh, kw));
            OK(vkdnnSetConvolutionGroupCount(cd, g));
            OK(vkdnnSetConvolution2dDescriptor(cd, ph, pw, sh, sw, dh, dw, vkdnnConvolutionMode_t(mode), VKDNN_DATA_FLOAT));
            OK(vkdnnGetConvolution2dForwardOutputDim(cd, xd, fd, &on, &oc, &oh, &ow));
            set(yd, n, k, oh, ow);
            int ys = n * k * oh * ow;
            std::vector<float> x(xs), f(fs), dy(ys), bias(k, 0.f);
            for (int i = 0; i < xs; ++i)
            {
                x[i] = float(i % 17 - 8) * 0.11f;
            }
            for (int i = 0; i < fs; ++i)
            {
                f[i] = float(i % 13 - 6) * 0.09f;
            }
            for (int i = 0; i < ys; ++i)
            {
                dy[i] = float(i % 11 - 5) * 0.13f;
            }
            std::vector<double> gx(xs), gf(fs), gb(k);
            auto objective = [&](bool gradients)
            {
                double loss = 0;
                for (int bn = 0; bn < n; ++bn)
                {
                    for (int out = 0; out < k; ++out)
                    {
                        for (int oy = 0; oy < oh; ++oy)
                        {
                            for (int ox = 0; ox < ow; ++ox)
                            {
                                int yi = ((bn * k + out) * oh + oy) * ow + ox;
                                double value = bias[out];
                                if (gradients)
                                {
                                    gb[out] += dy[yi];
                                }
                                for (int ic = 0; ic < fc; ++ic)
                                {
                                    for (int a = 0; a < kh; ++a)
                                    {
                                        for (int b = 0; b < kw; ++b)
                                        {
                                            int iy = oy * sh - ph + a * dh, ix = ox * sw - pw + b * dw;
                                            if (iy < 0 || ix < 0 || iy >= h || ix >= w)
                                            {
                                                continue;
                                            }
                                            int fy = a, fx = b;
                                            if (mode == 0)
                                            {
                                                fy = kh - 1 - a;
                                                fx = kw - 1 - b;
                                            }
                                            int xi = ((bn * c + out / (k / g) * fc + ic) * h + iy) * w + ix, fi = ((out * fc + ic) * kh + fy) * kw + fx;
                                            value += double(x[xi]) * f[fi];
                                            if (gradients)
                                            {
                                                gx[xi] += double(dy[yi]) * f[fi];
                                                gf[fi] += double(dy[yi]) * x[xi];
                                            }
                                        }
                                    }
                                }
                                loss += value * dy[yi];
                            }
                        }
                    }
                }
                return loss;
            };
            objective(true);
            differences(x, gx, [&]
                {
                    return objective(false);
                });
            differences(f, gf, [&]
                {
                    return objective(false);
                });
            differences(bias, gb, [&]
                {
                    return objective(false);
                });
            Buffer bx(xs), bf(fs), by(ys), dx(xs), df(fs), db(k);
            bx.put(x);
            bf.put(f);
            by.put(dy);
            float alpha = 1.25f, beta = 0;
            Buffer forwardOutput(ys), biasBuffer(k);
            float one = 1, zero = 0;
            auto gpuObjective = [&]
            {
                bx.put(x);
                bf.put(f);
                biasBuffer.put(bias);
                OK(vkdnnConvolutionForward(handle, &one, xd, bx.p, fd, bf.p, cd, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, nullptr, 0, &zero, yd, forwardOutput.p));
                OK(vkdnnAddTensor(handle, &one, bd, biasBuffer.p, &one, yd, forwardOutput.p));
                auto result = forwardOutput.get();
                double loss = 0;
                for (int j = 0; j < ys; ++j) { loss += double(result[j]) * dy[j]; }
                return loss;
            };
            auto gpuDifference = [&](std::vector<float>& v, const std::vector<double>& grad)
            {
                for (size_t index : { size_t(0), v.size() / 2, v.size() - 1 })
                {
                    float old = v[index];
                    v[index] = old + 0.01f;
                    double hi = gpuObjective();
                    float upper = v[index];
                    v[index] = old - 0.01f;
                    double lo = gpuObjective();
                    float lower = v[index];
                    v[index] = old;
                    REQUIRE(std::abs((hi - lo) / (upper - lower) - grad[index]) < 0.002 * std::max(1.0, std::abs(grad[index])));
                }
            };
            gpuDifference(x, gx);
            gpuDifference(f, gf);
            gpuDifference(bias, gb);
            bx.put(x);
            bf.put(f);
            auto data = [&](const void* a, const void* weights, const void* gradient, void* output)
            {
                return vkdnnConvolutionBackwardData(handle, a, fd, weights, yd, gradient, cd, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0, &beta, xd, output);
            };
            auto filter = [&](const void* a, const void* input, const void* gradient, void* output)
            {
                return vkdnnConvolutionBackwardFilter(handle, a, xd, input, yd, gradient, cd, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, fd, output);
            };
            auto biasCall = [&](const void* a, const void* gradient, void* output)
            {
                return vkdnnConvolutionBackwardBias(handle, a, yd, gradient, &beta, bd, output);
            };
            size_t bytes = 99;
            OK(vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, fd, yd, cd, xd, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &bytes));
            REQUIRE(bytes == 0);
            bytes = 99;
            OK(vkdnnGetConvolutionBackwardFilterWorkspaceSize(handle, xd, yd, cd, fd, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &bytes));
            REQUIRE(bytes == 0);
            for (float b : { 0.f, -0.5f })
            {
                beta = b;
                float initial = NAN;
                if (b != 0)
                {
                    initial = 2;
                }
                dx.put(std::vector<float>(xs, initial));
                df.put(std::vector<float>(fs, initial));
                db.put(std::vector<float>(k, initial));
                OK(data(&alpha, bf.p, by.p, dx.p));
                OK(filter(&alpha, bx.p, by.p, df.p));
                OK(biasCall(&alpha, by.p, db.p));
                check(dx.get(), gx, alpha, b);
                check(df.get(), gf, alpha, b);
                check(db.get(), gb, alpha, b);
            }
            beta = 0;
            alpha = 0;
            OK(data(&alpha, nullptr, nullptr, dx.p));
            OK(filter(&alpha, nullptr, nullptr, df.p));
            OK(biasCall(&alpha, nullptr, db.p));
            check(dx.get(), gx, 0, 0);
            check(df.get(), gf, 0, 0);
            check(db.get(), gb, 0, 0);
            alpha = 1;
            REQUIRE(data(nullptr, bf.p, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(data(bf.p, bf.p, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(data(&alpha, f.data(), by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(data(&alpha, static_cast<char*>(bf.p) + 1, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(data(&alpha, static_cast<char*>(bf.p) + 4, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(filter(&alpha, static_cast<char*>(bx.p) + 4, by.p, df.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(biasCall(&alpha, static_cast<char*>(by.p) + 4, db.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(data(&alpha, bf.p, by.p, static_cast<char*>(dx.p) + 4) == VKDNN_STATUS_BAD_PARAM);
            Buffer alias(std::max({ xs, fs, ys, k }));
            REQUIRE(data(&alpha, alias.p, by.p, alias.p) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(filter(&alpha, bx.p, alias.p, alias.p) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(biasCall(&alpha, alias.p, alias.p) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(vkdnnConvolutionBackwardBias(handle, &alpha, yd, by.p, &beta, wrong, db.p) == VKDNN_STATUS_BAD_PARAM || k == 1);
            REQUIRE(vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, fd, yd, cd, xd, static_cast<vkdnnConvolutionBwdDataAlgo_t>(99), &bytes) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(vkdnnGetConvolutionBackwardFilterWorkspaceSize(handle, xd, yd, cd, fd, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr) == VKDNN_STATUS_BAD_PARAM);
            if (devices > 1)
            {
                RT(vkdnnSetDevice(1));
                REQUIRE(data(&alpha, bf.p, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
                RT(vkdnnSetDevice(0));
            }
            by.put(std::vector<float>(ys, NAN));
            OK(biasCall(&alpha, by.p, db.p));
            for (float v : db.get())
            {
                REQUIRE(std::isnan(v));
            }
            OK(data(&alpha, bf.p, by.p, dx.p));
            auto nanDx = dx.get();
            for (int i = 0; i < xs; ++i)
            {
                if (gx[i] != 0)
                {
                    REQUIRE(std::isnan(nanDx[i]));
                }
            }
            OK(filter(&alpha, bx.p, by.p, df.p));
            auto nanDf = df.get();
            for (int i = 0; i < fs; ++i)
            {
                if (gf[i] != 0)
                {
                    REQUIRE(std::isnan(nanDf[i]));
                }
            }
            by.put(dy);
            beta = 1;
            dx.put(std::vector<float>(xs, NAN));
            df.put(std::vector<float>(fs, NAN));
            db.put(std::vector<float>(k, NAN));
            OK(data(&alpha, bf.p, by.p, dx.p));
            OK(filter(&alpha, bx.p, by.p, df.p));
            OK(biasCall(&alpha, by.p, db.p));
            for (float v : dx.get())
            {
                REQUIRE(std::isnan(v));
            }
            for (float v : df.get())
            {
                REQUIRE(std::isnan(v));
            }
            for (float v : db.get())
            {
                REQUIRE(std::isnan(v));
            }
        }
    }
    // Tiny convolution isolates NaN products, alpha=0 scaling and failed-call output preservation.
    set(xd, 1, 1, 1, 2);
    set(yd, 1, 1, 1, 2);
    set(bd, 1, 1, 1, 1);
    OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 1, 1, 1, 1));
    OK(vkdnnSetConvolutionGroupCount(cd, 1));
    OK(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
    {
        Buffer x(2), f(1), dy(2), dx(2), df(1), db(1);
        x.put({ 0, 1 });
        f.put({ INFINITY });
        dy.put({ 0, 1 });
        float a = 1, b = 0;
        auto data = [&](const void* weights, const void* gradient, void* output)
        {
            return vkdnnConvolutionBackwardData(handle, &a, fd, weights, yd, gradient, cd, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0, &b, xd, output);
        };
        auto filter = [&](const void* input)
        {
            return vkdnnConvolutionBackwardFilter(handle, &a, xd, input, yd, dy.p, cd, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &b, fd, df.p);
        };
        OK(data(f.p, dy.p, dx.p));
        auto actual = dx.get();
        REQUIRE(std::isnan(actual[0]) && std::isinf(actual[1]));
        dy.put({ NAN, 1 });
        OK(filter(x.p));
        REQUIRE(std::isnan(df.get()[0]));
        OK(vkdnnConvolutionBackwardBias(handle, &a, yd, dy.p, &b, bd, db.p));
        REQUIRE(std::isnan(db.get()[0]));
        a = 0;
        b = -0.5f;
        dx.put({ 2, 2 });
        df.put({ 2 });
        db.put({ 2 });
        OK(data(nullptr, nullptr, dx.p));
        OK(filter(nullptr));
        OK(vkdnnConvolutionBackwardBias(handle, &a, yd, nullptr, &b, bd, db.p));
        check(dx.get(), { 0, 0 }, 0, b);
        check(df.get(), { 0 }, 0, b);
        check(db.get(), { 0 }, 0, b);
        a = 1;
        b = 0;
        dx.put({ 17, 17 });
        df.put({ 17 });
        db.put({ 17 });
        REQUIRE(data(f.p, nullptr, dx.p) == VKDNN_STATUS_BAD_PARAM);
        REQUIRE(filter(nullptr) == VKDNN_STATUS_BAD_PARAM);
        REQUIRE(vkdnnConvolutionBackwardBias(handle, &a, yd, nullptr, &b, bd, db.p) == VKDNN_STATUS_BAD_PARAM);
        check(dx.get(), { 17, 17 }, 1, 0);
        check(df.get(), { 17 }, 1, 0);
        check(db.get(), { 17 }, 1, 0);
        size_t bytes = 123;
        REQUIRE(vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, fd, wrong, cd, xd, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &bytes) == VKDNN_STATUS_BAD_PARAM);
        REQUIRE(bytes == 123);
        OK(vkdnnSetConvolutionGroupCount(cd, 2));
        REQUIRE(vkdnnGetConvolutionBackwardFilterWorkspaceSize(handle, xd, yd, cd, fd, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &bytes) == VKDNN_STATUS_BAD_PARAM);
        REQUIRE(bytes == 123);
        OK(vkdnnSetConvolutionGroupCount(cd, 1));
        OK(vkdnnSetConvolution2dDescriptor(cd, INT_MAX / 2, 0, 1, 1, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
        REQUIRE(vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, fd, yd, cd, xd, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, &bytes) == VKDNN_STATUS_NOT_SUPPORTED);
        REQUIRE(bytes == 123);
        REQUIRE(vkdnnSetTensor4dDescriptor(wrong, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, INT_MAX, INT_MAX, INT_MAX, INT_MAX) == VKDNN_STATUS_NOT_SUPPORTED);
    }
    {
        set(xd, 100, 1, 20, 20);
        set(yd, 100, 1, 20, 20);
        set(bd, 1, 1, 1, 1);
        OK(vkdnnSetFilter4dDescriptor(fd, VKDNN_DATA_FLOAT, VKDNN_TENSOR_NCHW, 1, 1, 1, 1));
        OK(vkdnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, VKDNN_CROSS_CORRELATION, VKDNN_DATA_FLOAT));
        Buffer input(40000), gradient(40000), result(1), biasResult(1);
        std::vector<float> source(40000), dy(40000);
        double reference = 0, biasReference = 0;
        for (size_t index = 0; index < source.size(); ++index)
        {
            source[index] = float(int(index % 17) - 8) * 0.11f;
            dy[index] = float(int(index % 13) - 6) * 0.07f;
            reference += double(source[index]) * dy[index];
            biasReference += dy[index];
        }
        input.put(source);
        gradient.put(dy);
        float alpha = 1.25f, beta = 0.5f;
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            result.put({ 2 });
            OK(vkdnnConvolutionBackwardFilter(handle, &alpha, xd, input.p, yd, gradient.p, cd,
                VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, fd, result.p));
            check(result.get(), { reference }, alpha, beta);
            biasResult.put({ 2 });
            OK(vkdnnConvolutionBackwardBias(handle, &alpha, yd, gradient.p, &beta, bd, biasResult.p));
            check(biasResult.get(), { biasReference }, alpha, beta);
        }
        source[8193] = NAN;
        input.put(source);
        OK(vkdnnConvolutionBackwardFilter(handle, &alpha, xd, input.p, yd, gradient.p, cd,
            VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, fd, result.p));
        REQUIRE(std::isnan(result.get()[0]));
        alpha = 0;
        result.put({ 2 });
        OK(vkdnnConvolutionBackwardFilter(handle, &alpha, xd, nullptr, yd, nullptr, cd,
            VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &beta, fd, result.p));
        check(result.get(), { 0 }, alpha, beta);
        std::puts("PASS: split dW CPU reference, offsets, repeated calls, alpha/beta and NaN");
    }
    std::puts("PASS: convolution dx/dw/db CPU reference, CPU/GPU finite differences, groups/depthwise/flip/stride/dilation/padding, scalars, overflow and ranges");
    for (int mode = 0; mode < 3; ++mode)
    {
        for (int config = 0; config < 3; ++config)
        {
            int n = 2, c = 2, h = 4, w = 5, kh = 2, kw = 3, ph = 1, pw = 1, sh = 1, sw = 2;
            if (config == 1)
            {
                kh = 3;
                kw = 2;
                ph = 1;
                pw = 0;
                sh = 2;
                sw = 1;
            }
            if (config == 2)
            {
                kh = 1;
                kw = 1;
                ph = 0;
                pw = 0;
                sh = 2;
                sw = 3;
            }
            int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1, xs = n * c * h * w, ys = n * c * oh * ow;
            set(xd, n, c, h, w);
            set(yd, n, c, oh, ow);
            OK(vkdnnSetPooling2dDescriptor(pd, vkdnnPoolingMode_t(mode), VKDNN_NOT_PROPAGATE_NAN, kh, kw, ph, pw, sh, sw));
            std::vector<float> x(xs), dy(ys);
            for (int i = 0; i < xs; ++i)
            {
                x[i] = float(i) * 0.13f - 3;
            }
            for (int i = 0; i < ys; ++i)
            {
                dy[i] = float(i % 7 - 3) * 0.17f;
            }
            std::vector<double> gx(xs);
            auto objective = [&](bool gradients)
            {
                double loss = 0;
                for (int nc = 0; nc < n * c; ++nc)
                {
                    for (int oy = 0; oy < oh; ++oy)
                    {
                        for (int ox = 0; ox < ow; ++ox)
                        {
                            int yi = (nc * oh + oy) * ow + ox, winner = -1, valid = 0;
                            double value = 0, best = -INFINITY;
                            for (int a = 0; a < kh; ++a)
                            {
                                for (int b = 0; b < kw; ++b)
                                {
                                    int iy = oy * sh - ph + a, ix = ox * sw - pw + b;
                                    if (iy < 0 || ix < 0 || iy >= h || ix >= w)
                                    {
                                        continue;
                                    }
                                    int j = (nc * h + iy) * w + ix;
                                    ++valid;
                                    value += x[j];
                                    if (x[j] > best)
                                    {
                                        best = x[j];
                                        winner = j;
                                    }
                                }
                            }
                            int divisor = valid;
                            if (mode == 1)
                            {
                                divisor = kh * kw;
                            }
                            if (mode == 0)
                            {
                                value = best;
                                if (gradients && winner >= 0)
                                {
                                    gx[winner] += dy[yi];
                                }
                            }
                            else
                            {
                                value /= divisor;
                                if (gradients)
                                {
                                    for (int a = 0; a < kh; ++a)
                                    {
                                        for (int b = 0; b < kw; ++b)
                                        {
                                            int iy = oy * sh - ph + a, ix = ox * sw - pw + b;
                                            if (iy >= 0 && ix >= 0 && iy < h && ix < w)
                                            {
                                                gx[(nc * h + iy) * w + ix] += double(dy[yi]) / divisor;
                                            }
                                        }
                                    }
                                }
                            }
                            loss += value * dy[yi];
                        }
                    }
                }
                return loss;
            };
            objective(true);
            differences(x, gx, [&]
                {
                    return objective(false);
                });
            Buffer bx(xs), by(ys), bdy(ys), dx(xs);
            bx.put(x);
            bdy.put(dy);
            float one = 1, zero = 0, alpha = 1.25f, beta = 0;
            for (size_t index : { size_t(0), x.size() / 2, x.size() - 1 })
            {
                auto loss = [&]
                {
                    bx.put(x);
                    OK(vkdnnPoolingForward(handle, pd, &one, xd, bx.p, &zero, yd, by.p));
                    auto result = by.get();
                    double value = 0;
                    for (int j = 0; j < ys; ++j) { value += double(result[j]) * dy[j]; }
                    return value;
                };
                float old = x[index];
                x[index] = old + 0.01f;
                double hi = loss();
                float upper = x[index];
                x[index] = old - 0.01f;
                double lo = loss();
                float lower = x[index];
                x[index] = old;
                REQUIRE(std::abs((hi - lo) / (upper - lower) - gx[index]) < 0.002 * std::max(1.0, std::abs(gx[index])));
            }
            bx.put(x);
            OK(vkdnnPoolingForward(handle, pd, &one, xd, bx.p, &zero, yd, by.p));
            auto backward = [&](const void* a, const void* input, const void* gradient, const void* output, void* dst)
            {
                return vkdnnPoolingBackward(handle, pd, a, yd, output, yd, gradient, xd, input, &beta, xd, dst);
            };
            for (float b : { 0.f, -0.5f })
            {
                beta = b;
                float initial = NAN;
                if (b != 0)
                {
                    initial = 2;
                }
                dx.put(std::vector<float>(xs, initial));
                OK(backward(&alpha, bx.p, bdy.p, by.p, dx.p));
                check(dx.get(), gx, alpha, b);
            }
            alpha = 0;
            beta = 0;
            OK(backward(&alpha, nullptr, nullptr, nullptr, dx.p));
            check(dx.get(), gx, 0, 0);
            alpha = 1;
            REQUIRE(backward(&alpha, bx.p, bdy.p, nullptr, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(backward(&alpha, static_cast<char*>(bx.p) + 4, bdy.p, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(backward(&alpha, bx.p, static_cast<char*>(bdy.p) + 1, by.p, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(backward(&alpha, bx.p, bdy.p, static_cast<char*>(by.p) + 4, dx.p) == VKDNN_STATUS_BAD_PARAM);
            REQUIRE(backward(&alpha, bx.p, bdy.p, by.p, bx.p) == VKDNN_STATUS_NOT_SUPPORTED);
            REQUIRE(vkdnnPoolingBackward(handle, pd, &alpha, yd, by.p, wrong, bdy.p, xd, bx.p, &beta, xd, dx.p) == VKDNN_STATUS_BAD_PARAM);
            beta = 1;
            dx.put(std::vector<float>(xs, NAN));
            OK(backward(&alpha, bx.p, bdy.p, by.p, dx.p));
            for (float v : dx.get())
            {
                REQUIRE(std::isnan(v));
            }
        }
    }
    // Non-differentiable ties and NaNs are contractual, not finite-difference cases.
    set(xd, 1, 1, 1, 3);
    set(yd, 1, 1, 1, 2);
    for (int mode = 0; mode < 3; ++mode)
    {
        for (int nan = 0; nan < 2; ++nan)
        {
            OK(vkdnnSetPooling2dDescriptor(pd, vkdnnPoolingMode_t(mode), vkdnnNanPropagation_t(nan), 1, 2, 0, 0, 1, 1));
            Buffer x(3), y(2), dy(2), dx(3);
            float alpha = 1, beta = 0;
            dy.put({ 2, 4 });
            y.put({ 1, 1 });
            x.put({ 1, 1, 1 });
            auto run = [&]
            {
                OK(vkdnnPoolingBackward(handle, pd, &alpha, yd, y.p, yd, dy.p, xd, x.p, &beta, xd, dx.p));
            };
            run();
            if (mode == 0)
            {
                check(dx.get(), { 2, 4, 0 }, 1, 0);
            }
            else
            {
                check(dx.get(), { 1, 3, 2 }, 1, 0);
            }
            x.put({ 1, NAN, 2 });
            run();
            auto actual = dx.get();
            if (nan)
            {
                for (float v : actual)
                {
                    REQUIRE(std::isnan(v));
                }
            }
            else if (mode == 0)
            {
                check(actual, { 2, 0, 4 }, 1, 0);
            }
            else
            {
                check(actual, { 1, 0, 2 }, 1, 0);
            }
            x.put({ NAN, NAN, NAN });
            run();
            if (!nan)
            {
                check(dx.get(), { 0, 0, 0 }, 1, 0);
            }
            x.put({ -INFINITY, -INFINITY, -INFINITY });
            run();
            if (mode == 0)
            {
                check(dx.get(), { 2, 4, 0 }, 1, 0);
            }
            alpha = NAN;
            run();
            for (float v : dx.get())
            {
                REQUIRE(std::isnan(v));
            }
        }
    }
    set(xd, 1, 1, 1, 1);
    set(yd, 1, 1, 1, 1);
    OK(vkdnnSetPooling2dDescriptor(pd, VKDNN_POOLING_MAX, VKDNN_NOT_PROPAGATE_NAN, 1, 1, 0, 0, INT_MAX, INT_MAX));
    {
        Buffer x(1), y(1), dy(1), dx(1);
        x.put({ 2 });
        y.put({ 2 });
        dy.put({ 3 });
        float a = 1, b = 0;
        OK(vkdnnPoolingBackward(handle, pd, &a, yd, y.p, yd, dy.p, xd, x.p, &b, xd, dx.p));
        check(dx.get(), { 3 }, 1, 0);
    }
    RT(vkdnnDeviceSynchronize());
    OK(vkdnnDestroyPoolingDescriptor(pd));
    OK(vkdnnDestroyConvolutionDescriptor(cd));
    OK(vkdnnDestroyFilterDescriptor(fd));
    OK(vkdnnDestroyTensorDescriptor(xd));
    OK(vkdnnDestroyTensorDescriptor(yd));
    OK(vkdnnDestroyTensorDescriptor(bd));
    OK(vkdnnDestroyTensorDescriptor(wrong));
    OK(vkdnnDestroy(handle));
    RT(vkdnnDeviceReset());
    std::puts("PASS: pooling CPU reference, finite differences, overlap/batch/padding/stride/ties/NaN/ranges; synchronization complete");
    if (devices < 2)
    {
        std::puts("SKIP: multi-device ownership (one GPU)");
    }
}
