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
    std::printf("GPU: %s driver=%u API=%u; Release; host synchronous API latency, no transfers; 10 warmup, 51 samples x 5 calls\n", prop.name, prop.driverVersion, prop.apiVersion);
    vkdnnHandle_t handle = nullptr;
    vkdnnTensorDescriptor_t tensor = nullptr;
    vkdnnActivationDescriptor_t activation = nullptr;
    check(vkdnnCreate(&handle) == VKDNN_STATUS_SUCCESS);
    check(vkdnnCreateTensorDescriptor(&tensor) == VKDNN_STATUS_SUCCESS);
    check(vkdnnCreateActivationDescriptor(&activation) == VKDNN_STATUS_SUCCESS);
    for (int scenario = 0; scenario < 6; ++scenario)
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
            h = 1;
            w = 1048579;
        }
        if (scenario >= 3)
        {
            n = 1;
            c = 16;
            h = 257;
            w = 257;
        }
        size_t count = size_t(n) * c * h * w;
        check(vkdnnSetTensor4dDescriptor(tensor, VKDNN_TENSOR_NCHW, VKDNN_DATA_FLOAT, n, c, h, w) == VKDNN_STATUS_SUCCESS);
        void *x = nullptr, *y = nullptr;
        check(vkdnnMalloc(&x, count * sizeof(float)) == vkdnnSuccess);
        check(vkdnnMalloc(&y, count * sizeof(float)) == vkdnnSuccess);
        std::vector<float> input(count);
        for (size_t i = 0; i < count; ++i) { input[i] = float(int(i % 97) - 48) / 17; }
        check(vkdnnMemcpy(x, input.data(), count * sizeof(float), vkdnnMemcpyHostToDevice) == vkdnnSuccess);
        float alpha = 1, beta = 0;
        auto mode = VKDNN_SOFTMAX_MODE_INSTANCE;
        if (scenario == 1) { mode = VKDNN_SOFTMAX_MODE_CHANNEL; }
        if (scenario >= 3)
        {
            check(vkdnnSetActivationDescriptor(activation, vkdnnActivationMode_t(scenario - 3), VKDNN_PROPAGATE_NAN, 0) == VKDNN_STATUS_SUCCESS);
        }
        auto call = [&]()
        {
            if (scenario < 3) { check(vkdnnSoftmaxForward(handle, VKDNN_SOFTMAX_ACCURATE, mode, &alpha, tensor, x, &beta, tensor, y) == VKDNN_STATUS_SUCCESS); }
            else
            {
                check(vkdnnActivationForward(handle, activation, &alpha, tensor, x, &beta, tensor, y) == VKDNN_STATUS_SUCCESS);
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
        std::printf("case=%d NCHW=%d,%d,%d,%d median=%.3f us p10=%.3f p90=%.3f\n", scenario, n, c, h, w, samples[25], samples[5], samples[45]);
        check(vkdnnFree(x) == vkdnnSuccess);
        check(vkdnnFree(y) == vkdnnSuccess);
    }
    check(vkdnnDestroyActivationDescriptor(activation) == VKDNN_STATUS_SUCCESS);
    check(vkdnnDestroyTensorDescriptor(tensor) == VKDNN_STATUS_SUCCESS);
    check(vkdnnDestroy(handle) == VKDNN_STATUS_SUCCESS);
    check(vkdnnDeviceReset() == vkdnnSuccess);
}
