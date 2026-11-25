#include "vkdnn/vkdnn_helpers.h"
#include "vkdnn/vkdnn_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define REQUIRE(c) \
    do \
    { \
        if (!(c)) \
        { \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); \
            std::exit(1); \
        } \
    } while (0)
#define OK(c) REQUIRE((c) == VKDNN_STATUS_SUCCESS)
#define RT(c) REQUIRE((c) == vkdnnSuccess)
static double activate(int mode, double x)
{
    if (mode == VKDNN_ACTIVATION_RELU) { return std::max(x, 0.0); }
    if (mode == VKDNN_ACTIVATION_SIGMOID) { return 1.0 / (1.0 + std::exp(-x)); }
    return std::tanh(x);
}
static void compare(float actual, double expected, double tolerance = 4e-5)
{
    if (std::isnan(expected)) { REQUIRE(std::isnan(actual)); }
    else if (std::isinf(expected)) { REQUIRE(actual == expected); }
    else
    {
        if (!(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected))))
        {
            std::fprintf(stderr, "actual %.9g expected %.12g\n", actual, expected);
            REQUIRE(false);
        }
    }
}
int main()
{
    int devices = 0;
    auto status = vkdnnGetDeviceCount(&devices);
    if (status == vkdnnErrorNoDevice || (status == vkdnnSuccess && devices == 0))
    {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    RT(status);
    vkdnnDeviceProp prop{};
    RT(vkdnnGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n", prop.name);
    vkdnnHandle_t handle = nullptr;
    vkdnnActivationDescriptor_t activation = nullptr, unsetActivation = nullptr;
    vkdnnTensorDescriptor_t tensor = nullptr, unset = nullptr, different = nullptr;
    OK(vkdnnCreate(&handle));
    {
        void* data = nullptr;
        void* gradient = nullptr;
        RT(vkdnnMalloc(&data, 2 * sizeof(float)));
        RT(vkdnnMalloc(&gradient, 2 * sizeof(float)));
        float values[2] = { -2, 3 };
        float gradients[2] = { 4, 5 };
        RT(vkdnnMemcpy(data, values, sizeof(values), vkdnnMemcpyHostToDevice));
        RT(vkdnnMemcpy(gradient, gradients, sizeof(gradients), vkdnnMemcpyHostToDevice));
        vkdnnTensorView view{ 1, 1, 1, 2, data, gradient };
        OK(vkdnnExecuteActivation(handle, view, view, VKDNN_HELPER_RELU, 2, 1, 0));
        RT(vkdnnMemcpy(values, data, sizeof(values), vkdnnMemcpyDeviceToHost));
        compare(values[0], -2);
        compare(values[1], 9);
        OK(vkdnnExecuteActivation(handle, view, view, VKDNN_HELPER_RELU, 2, 1, 1));
        RT(vkdnnMemcpy(gradients, gradient, sizeof(gradients), vkdnnMemcpyDeviceToHost));
        compare(gradients[0], 4);
        compare(gradients[1], 15);
        REQUIRE(vkdnnExecuteActivation(handle, view, view, VKDNN_HELPER_RELU, 1, 0, 2) == VKDNN_STATUS_BAD_PARAM);
        REQUIRE(vkdnnExecuteActivation(handle, view, view, static_cast<vkdnnActivationKind>(99), 1, 0, 0) == VKDNN_STATUS_NOT_SUPPORTED);
        RT(vkdnnFree(gradient));
        RT(vkdnnFree(data));
        std::puts("Execution helpers aliasing, accumulation and invalid parameters passed");
    }
    OK(vkdnnCreateActivationDescriptor(&activation));
    OK(vkdnnCreateActivationDescriptor(&unsetActivation));
    OK(vkdnnCreateTensorDescriptor(&tensor));
    OK(vkdnnCreateTensorDescriptor(&unset));
    OK(vkdnnCreateTensorDescriptor(&different));
    OK(vkdnnSetTensor4dDescriptor(different, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 1, 1, 1, 1));
    constexpr int capacity = 65541;
    void* allocation[4]{};
    void* pointer[4]{};
    for (int i = 0; i < 4; ++i)
    {
        RT(vkdnnMalloc(&allocation[i], capacity * sizeof(float)));
        pointer[i] = static_cast<char*>(allocation[i]) + 4;
    }
    float alpha = 1, beta = 0;
    auto backward = [&]
    {
        return vkdnnActivationBackward(handle, activation, &alpha, tensor, pointer[1], tensor, pointer[2],
            tensor, pointer[0], &beta, tensor, pointer[3]);
    };
    std::mt19937 random(8321);
    std::uniform_real_distribution<float> distribution(-4, 4);
    for (int size : { 1, 63, 64, 65, 256, 65539 })
    {
        int batch = 1, width = size;
        if (size == 256)
        {
            batch = 4;
            width = 64;
        }
        OK(vkdnnSetTensor4dDescriptor(tensor, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, batch, 1, 1, width));
        std::vector<float> x(size), y(size), dy(size), prior(size), actual(size);
        for (int mode : { VKDNN_ACTIVATION_RELU, VKDNN_ACTIVATION_SIGMOID, VKDNN_ACTIVATION_TANH })
        {
            for (int nan = 0; nan < 2; ++nan)
            {
                OK(vkdnnSetActivationDescriptor(activation, vkdnnActivationMode_t(mode), vkdnnNanPropagation_t(nan), 0));
                for (int i = 0; i < size; ++i)
                {
                    x[i] = distribution(random);
                    dy[i] = distribution(random);
                    prior[i] = distribution(random);
                }
                const float edge[] = { 0, -0.f, 1e-7f, -1e-7f, 1000, -1000, INFINITY, -INFINITY, NAN };
                for (int i = 0; i < std::min(size, 9); ++i) { x[i] = edge[i]; }
                for (int i = 0; i < size; ++i) { y[i] = float(activate(mode, x[i])); }
                if (size > 32)
                {
                    dy[20] = NAN;
                    dy[21] = INFINITY;
                    dy[22] = -INFINITY;
                }
                RT(vkdnnMemcpy(pointer[0], x.data(), size * 4, vkdnnMemcpyHostToDevice));
                RT(vkdnnMemcpy(pointer[1], y.data(), size * 4, vkdnnMemcpyHostToDevice));
                RT(vkdnnMemcpy(pointer[2], dy.data(), size * 4, vkdnnMemcpyHostToDevice));
                for (float a : { 1.f, -1.3f, 0.f })
                {
                    for (float b : { 0.f, 0.7f })
                    {
                        alpha = a;
                        beta = b;
                        if (b == 0) { std::fill(prior.begin(), prior.end(), NAN); }
                        else
                        {
                            for (auto& v : prior) { v = distribution(random); }
                        }
                        RT(vkdnnMemcpy(pointer[3], prior.data(), size * 4, vkdnnMemcpyHostToDevice));
                        OK(backward());
                        RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
                        for (int i = 0; i < size; ++i)
                        {
                            double derivative;
                            if (mode == VKDNN_ACTIVATION_RELU)
                            {
                                derivative = x[i] > 0 ? 1 : 0;
                                if (std::isnan(x[i]) && nan) { derivative = NAN; }
                            }
                            else if (mode == VKDNN_ACTIVATION_SIGMOID) { derivative = double(y[i]) * (1 - double(y[i])); }
                            else
                            {
                                derivative = 1 - double(y[i]) * y[i];
                            }
                            double expected = 0;
                            if (a != 0) { expected = a * (double(dy[i]) * derivative); }
                            if (b != 0) { expected += b * double(prior[i]); }
                            compare(actual[i], expected);
                        }
                    }
                }
            }
        }
    }
    // Differentiate the actual GPU forward objective, not just a duplicate derivative formula.
    constexpr int size = 8;
    OK(vkdnnSetTensor4dDescriptor(tensor, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, 2, 2, 1, 2));
    std::vector<float> x(size), dy(size), actual(size), plus(size), minus(size);
    for (int i = 0; i < size; ++i)
    {
        x[i] = float(i) * 0.4f - 1.3f;
        dy[i] = distribution(random);
    }
    alpha = 1;
    beta = 0;
    for (int mode : { VKDNN_ACTIVATION_RELU, VKDNN_ACTIVATION_SIGMOID, VKDNN_ACTIVATION_TANH })
    {
        OK(vkdnnSetActivationDescriptor(activation, vkdnnActivationMode_t(mode), VKDNN_PROPAGATE_NAN, 0));
        RT(vkdnnMemcpy(pointer[0], x.data(), size * 4, vkdnnMemcpyHostToDevice));
        RT(vkdnnMemcpy(pointer[2], dy.data(), size * 4, vkdnnMemcpyHostToDevice));
        OK(vkdnnActivationForward(handle, activation, &alpha, tensor, pointer[0], &beta, tensor, pointer[1]));
        OK(backward());
        RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
        for (int i = 0; i < size; ++i)
        {
            constexpr float step = 0.002f;
            float original = x[i];
            x[i] = original + step;
            RT(vkdnnMemcpy(pointer[0], x.data(), size * 4, vkdnnMemcpyHostToDevice));
            OK(vkdnnActivationForward(handle, activation, &alpha, tensor, pointer[0], &beta, tensor, pointer[1]));
            RT(vkdnnMemcpy(plus.data(), pointer[1], size * 4, vkdnnMemcpyDeviceToHost));
            x[i] = original - step;
            RT(vkdnnMemcpy(pointer[0], x.data(), size * 4, vkdnnMemcpyHostToDevice));
            OK(vkdnnActivationForward(handle, activation, &alpha, tensor, pointer[0], &beta, tensor, pointer[1]));
            RT(vkdnnMemcpy(minus.data(), pointer[1], size * 4, vkdnnMemcpyDeviceToHost));
            double difference = 0;
            for (int j = 0; j < size; ++j) { difference += double(dy[j]) * (plus[j] - minus[j]); }
            compare(actual[i], difference / (2 * step), 8e-4);
            x[i] = original;
        }
    }
    // y, not x, determines sigmoid/tanh derivatives; the caller supplies unscaled y.
    std::fill(x.begin(), x.end(), NAN);
    std::fill(dy.begin(), dy.end(), 2.f);
    std::vector<float> y(size, 0.25f);
    RT(vkdnnMemcpy(pointer[0], x.data(), size * 4, vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(pointer[1], y.data(), size * 4, vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(pointer[2], dy.data(), size * 4, vkdnnMemcpyHostToDevice));
    for (int mode : { VKDNN_ACTIVATION_SIGMOID, VKDNN_ACTIVATION_TANH })
    {
        OK(vkdnnSetActivationDescriptor(activation, vkdnnActivationMode_t(mode), VKDNN_NOT_PROPAGATE_NAN, 0));
        OK(backward());
        RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
        for (float v : actual) { compare(v, mode == VKDNN_ACTIVATION_SIGMOID ? 0.375 : 1.875); }
    }
    auto call = [&](vkdnnHandle_t h, vkdnnActivationDescriptor_t a, vkdnnTensorDescriptor_t t,
                    const void* yp, const void* dyp, const void* xp, const void* ap, const void* bp, void* output)
    {
        return vkdnnActivationBackward(h, a, ap, t, yp, tensor, dyp, tensor, xp, bp, tensor, output);
    };
    auto bad = [&](vkdnnStatus_t result)
    {
        REQUIRE(result == VKDNN_STATUS_BAD_PARAM);
    };
    REQUIRE(call(nullptr, activation, tensor, pointer[1], pointer[2], pointer[0], &alpha, &beta, pointer[3]) == VKDNN_STATUS_NOT_INITIALIZED);
    for (auto t : { unset, different, vkdnnTensorDescriptor_t(nullptr) })
    {
        bad(call(handle, activation, t, pointer[1], pointer[2], pointer[0], &alpha, &beta, pointer[3]));
    }
    bad(call(handle, unsetActivation, tensor, pointer[1], pointer[2], pointer[0], &alpha, &beta, pointer[3]));
    bad(call(handle, nullptr, tensor, pointer[1], pointer[2], pointer[0], &alpha, &beta, pointer[3]));
    bad(call(handle, activation, tensor, pointer[1], pointer[2], pointer[0], nullptr, &beta, pointer[3]));
    bad(call(handle, activation, tensor, pointer[1], pointer[2], pointer[0], &alpha, nullptr, pointer[3]));
    bad(call(handle, activation, tensor, pointer[1], pointer[2], pointer[0], pointer[0], &beta, pointer[3]));
    bad(call(handle, activation, tensor, pointer[1], pointer[2], pointer[0], &alpha, pointer[1], pointer[3]));
    for (int input = 0; input < 4; ++input)
    {
        RT(vkdnnMemcpy(pointer[3], dy.data(), size * 4, vkdnnMemcpyHostToDevice));
        void* saved = pointer[input];
        for (void* invalid : { static_cast<void*>(nullptr), static_cast<void*>(x.data()),
                 static_cast<void*>(static_cast<char*>(allocation[input]) + 1),
                 static_cast<void*>(static_cast<char*>(allocation[input]) + capacity * 4 - 4) })
        {
            pointer[input] = invalid;
            bad(backward());
        }
        pointer[input] = saved;
        RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
        for (float v : actual) { compare(v, 2); }
    }
    for (int input = 0; input < 3; ++input)
    {
        void* saved = pointer[input];
        pointer[input] = pointer[3];
        REQUIRE(backward() == VKDNN_STATUS_NOT_SUPPORTED);
        pointer[input] = static_cast<char*>(pointer[3]) + 4;
        REQUIRE(backward() == VKDNN_STATUS_NOT_SUPPORTED);
        pointer[input] = saved;
    }
    // Inputs may alias each other; same-allocation disjoint output ranges are supported.
    OK(call(handle, activation, tensor, pointer[1], pointer[1], pointer[1], &alpha, &beta, pointer[3]));
    OK(call(handle, activation, tensor, pointer[0], pointer[0], pointer[0], &alpha, &beta,
        static_cast<char*>(pointer[0]) + size * 4));
    REQUIRE(vkdnnSetActivationDescriptor(activation, VKDNN_ACTIVATION_CLIPPED_RELU, VKDNN_PROPAGATE_NAN, 0) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetTensor4dDescriptor(unset, VKDNN_TENSOR_NHWC, VKDNN_DATA_FLOAT, 1, 1, 1, 1) == VKDNN_STATUS_NOT_SUPPORTED);
    REQUIRE(vkdnnSetTensor4dDescriptor(unset, VKDNN_TENSOR_NCHW, VKDNN_DATA_HALF, 1, 1, 1, 1) == VKDNN_STATUS_NOT_SUPPORTED);
    alpha = 0;
    beta = 0;
    RT(vkdnnMemset(pointer[3], 0xff, size * 4));
    OK(call(handle, activation, tensor, nullptr, nullptr, nullptr, &alpha, &beta, pointer[3]));
    RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
    for (float v : actual) { compare(v, 0); }
    beta = 2;
    RT(vkdnnMemcpy(pointer[3], dy.data(), size * 4, vkdnnMemcpyHostToDevice));
    OK(call(handle, activation, tensor, pointer[3], pointer[3], pointer[3], &alpha, &beta, pointer[3]));
    RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
    for (float v : actual) { compare(v, 4); }
    for (float a : { NAN, INFINITY, 1.f })
    {
        for (float b : { NAN, INFINITY, 0.f, 1.f })
        {
            alpha = a;
            beta = b;
            RT(vkdnnMemset(pointer[3], 0xff, size * 4));
            OK(backward());
            RT(vkdnnMemcpy(actual.data(), pointer[3], size * 4, vkdnnMemcpyDeviceToHost));
            double expected = double(a) * 1.875;
            if (b != 0) { expected = NAN; }
            for (float v : actual) { compare(v, expected); }
        }
    }
    alpha = 1;
    beta = 0;
    if (devices > 1)
    {
        RT(vkdnnSetDevice(1));
        bad(backward());
        void* foreign = nullptr;
        RT(vkdnnMalloc(&foreign, size * 4));
        RT(vkdnnSetDevice(0));
        bad(call(handle, activation, tensor, foreign, pointer[2], pointer[0], &alpha, &beta, pointer[3]));
        RT(vkdnnFree(foreign));
        RT(vkdnnSetDevice(1));
        RT(vkdnnDeviceReset());
        RT(vkdnnSetDevice(0));
    }
    else
    {
        std::puts("SKIP: cross-device ownership (one GPU)");
    }
    RT(vkdnnFree(allocation[0]));
    allocation[0] = nullptr;
    bad(backward());
    RT(vkdnnDeviceReset());
    bad(backward());
    for (int i = 0; i < 4; ++i)
    {
        RT(vkdnnMalloc(&allocation[i], capacity * 4));
        pointer[i] = allocation[i];
        RT(vkdnnMemset(pointer[i], 0, size * 4));
    }
    OK(backward());
    for (void* p : allocation) { RT(vkdnnFree(p)); }
    OK(vkdnnDestroyActivationDescriptor(unsetActivation));
    OK(vkdnnDestroyActivationDescriptor(activation));
    bad(backward());
    OK(vkdnnDestroyTensorDescriptor(tensor));
    OK(vkdnnDestroyTensorDescriptor(unset));
    OK(vkdnnDestroyTensorDescriptor(different));
    OK(vkdnnDestroy(handle));
    RT(vkdnnDeviceReset());
    std::puts("Activation backward CPU reference and GPU finite differences passed");
}
