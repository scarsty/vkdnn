#include "vkdnn/vkdnn_ops.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void check(bool success)
{
    if (!success) { std::exit(1); }
}
int main()
{
    vkdnnDeviceProp prop{};
    check(vkdnnGetDeviceProperties(&prop, 0) == vkdnnSuccess);
    std::printf("GPU: %s driver=%u API=%u; synchronous host us; 10 warmup, 51 samples x 5 calls\n", prop.name, prop.driverVersion, prop.apiVersion);
    vkdnnHandle_t handle = nullptr;
    vkdnnTensorDescriptor_t t = nullptr;
    vkdnnActivationDescriptor_t a = nullptr;
    check(vkdnnCreate(&handle) == VKDNN_STATUS_SUCCESS);
    check(vkdnnCreateTensorDescriptor(&t) == VKDNN_STATUS_SUCCESS);
    check(vkdnnCreateActivationDescriptor(&a) == VKDNN_STATUS_SUCCESS);
    for (int scenario = 0; scenario < 7; ++scenario)
    {
        int n = 32, c = 4099, h = 1, w = 1;
        if (scenario == 1)
        {
            n = 2;
            c = 131;
            h = 17;
            w = 19;
        }
        if (scenario == 2)
        {
            n = 1;
            c = 1;
            w = 16387;
        }
        if (scenario == 3)
        {
            n = 8;
            c = 1027;
        }
        if (scenario >= 4)
        {
            n = 1;
            c = 16;
            h = 257;
            w = 257;
        }
        size_t count = size_t(n) * c * h * w;
        check(vkdnnSetTensor4dDescriptor(t, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w) == VKDNN_STATUS_SUCCESS);
        void *x = nullptr, *y = nullptr, *dy = nullptr, *dx = nullptr;
        for (void** p : { &x, &y, &dy, &dx }) { check(vkdnnMalloc(p, count * 4) == vkdnnSuccess); }
        std::vector<float> input(count), gradient(count);
        for (size_t i = 0; i < count; ++i)
        {
            input[i] = float(int(i % 97) - 48) / 17;
            gradient[i] = float(int(i % 29) - 14) / 19;
        }
        check(vkdnnMemcpy(x, input.data(), count * 4, vkdnnMemcpyHostToDevice) == vkdnnSuccess);
        check(vkdnnMemcpy(dy, gradient.data(), count * 4, vkdnnMemcpyHostToDevice) == vkdnnSuccess);
        float alpha = 1, beta = 0;
        auto mode = VKDNN_SOFTMAX_MODE_INSTANCE;
        if (scenario == 1) { mode = VKDNN_SOFTMAX_MODE_CHANNEL; }
        auto algo = VKDNN_SOFTMAX_ACCURATE;
        if (scenario == 3) { algo = VKDNN_SOFTMAX_LOG; }
        if (scenario < 4) { check(vkdnnSoftmaxForward(handle, algo, mode, &alpha, t, x, &beta, t, y) == VKDNN_STATUS_SUCCESS); }
        else
        {
            check(vkdnnSetActivationDescriptor(a, vkdnnActivationMode_t(scenario - 4), VKDNN_PROPAGATE_NAN, 0) == VKDNN_STATUS_SUCCESS);
            check(vkdnnActivationForward(handle, a, &alpha, t, x, &beta, t, y) == VKDNN_STATUS_SUCCESS);
        }
        auto call = [&]()
        {
            if (scenario < 4) { check(vkdnnSoftmaxBackward(handle, algo, mode, &alpha, t, y, t, dy, &beta, t, dx) == VKDNN_STATUS_SUCCESS); }
            else
            {
                check(vkdnnActivationBackward(handle, a, &alpha, t, y, t, dy, t, x, &beta, t, dx) == VKDNN_STATUS_SUCCESS);
            }
        };
        for (int i = 0; i < 10; ++i) { call(); }
        std::vector<double> samples;
        for (int i = 0; i < 51; ++i)
        {
            auto start = std::chrono::steady_clock::now();
            for (int j = 0; j < 5; ++j) { call(); }
            samples.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 5);
        }
        std::sort(samples.begin(), samples.end());
        std::printf("case=%d NCHW=%d,%d,%d,%d median=%.3f p10=%.3f p90=%.3f\n", scenario, n, c, h, w, samples[25], samples[5], samples[45]);
        for (void* p : { x, y, dy, dx }) { check(vkdnnFree(p) == vkdnnSuccess); }
    }
    check(vkdnnDestroyActivationDescriptor(a) == VKDNN_STATUS_SUCCESS);
    check(vkdnnDestroyTensorDescriptor(t) == VKDNN_STATUS_SUCCESS);
    check(vkdnnDestroy(handle) == VKDNN_STATUS_SUCCESS);
    check(vkdnnDeviceReset() == vkdnnSuccess);
}
