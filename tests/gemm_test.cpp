#include "vkdnn/vkblas.h"
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
#define OK(call) REQUIRE((call) == VKBLAS_STATUS_SUCCESS)
#define RT(call) REQUIRE((call) == vkdnnSuccess)

static void asynchronous(vkblasHandle_t handle)
{
    REQUIRE(vkdnnSetExecutionMode(static_cast<vkdnnExecutionMode>(99)) == vkdnnErrorInvalidValue);
    RT(vkdnnSetExecutionMode(vkdnnExecutionAsynchronous));
    float identity[] = { 1, 0, 0, 1 };
    float input[] = { 1, 2, 3, 4 };
    void *weights = nullptr, *first = nullptr, *second = nullptr;
    RT(vkdnnMalloc(&weights, sizeof(identity)));
    RT(vkdnnMalloc(&first, sizeof(input)));
    RT(vkdnnMalloc(&second, sizeof(input)));
    RT(vkdnnMemcpy(weights, identity, sizeof(identity), vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(first, input, sizeof(input), vkdnnMemcpyHostToDevice));
    float alpha = 0.99f, beta = 0;
    for (int iteration = 0; iteration < 97; ++iteration)
    {
        OK(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, &alpha,
            static_cast<float*>(weights), 2, static_cast<float*>(first), 2,
            &beta, static_cast<float*>(second), 2));
        std::swap(first, second);
    }
    float sum = 0;
    OK(vkblasSasum(handle, 4, static_cast<float*>(first), 1, &sum));
    REQUIRE(std::abs(sum - 10 * std::pow(double(alpha), 97)) < 1e-4);
    RT(vkdnnMemcpy(second, first, sizeof(input), vkdnnMemcpyDeviceToDevice));
    OK(vkblasSscal(handle, 4, &alpha, static_cast<float*>(second), 1));
    RT(vkdnnMemcpy(input, second, sizeof(input), vkdnnMemcpyDeviceToHost));
    for (int index = 0; index < 4; ++index)
    {
        REQUIRE(std::abs(input[index] - (index + 1) * std::pow(double(alpha), 98)) < 1e-4);
    }
    OK(vkblasSscal(handle, 4, &alpha, static_cast<float*>(first), 1));
    RT(vkdnnFree(first));
    OK(vkblasSscal(handle, 4, &alpha, static_cast<float*>(second), 1));
    RT(vkdnnSetExecutionMode(vkdnnExecutionSynchronous));
    RT(vkdnnDeviceSynchronize());
    RT(vkdnnFree(second));
    RT(vkdnnFree(weights));
    std::puts("PASS: asynchronous dependent GEMM, slot reuse, reduction, copy, free and mode transition");
}

static void numerical(vkblasHandle_t handle, vkblasOperation_t ta, vkblasOperation_t tb,
    int m, int n, int k, float alpha, float beta)
{
    int ar = ta == VKBLAS_OP_N ? m : k, ac = ta == VKBLAS_OP_N ? k : m;
    int br = tb == VKBLAS_OP_N ? k : n, bc = tb == VKBLAS_OP_N ? n : k;
    int lda = ar + 3, ldb = br + 5, ldc = m + 7;
    const size_t prefix = 3;
    std::vector<float> a(prefix + size_t(lda) * ac + 2), b(prefix + size_t(ldb) * bc + 2);
    std::vector<float> c(prefix + size_t(ldc) * n + 2, -123.0f);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = float(int(i % 17) - 8) / 9; }
    for (size_t i = 0; i < b.size(); ++i) { b[i] = float(int(i % 13) - 6) / 7; }
    for (int col = 0; col < n; ++col)
    {
        for (int row = 0; row < m; ++row)
        {
            c[prefix + row + col * ldc] = beta == 0 ? std::numeric_limits<float>::quiet_NaN() : float(row - col) / 3;
        }
    }
    auto reference = c;
    for (int col = 0; col < n; ++col)
    {
        for (int row = 0; row < m; ++row)
        {
            double sum = 0;
            if (alpha != 0)
            {
                for (int j = 0; j < k; ++j)
                {
                    size_t ai = ta == VKBLAS_OP_N ? row + j * lda : j + row * lda;
                    size_t bi = tb == VKBLAS_OP_N ? j + col * ldb : col + j * ldb;
                    sum += double(a[prefix + ai]) * b[prefix + bi];
                }
            }
            size_t index = prefix + row + col * ldc;
            double value = k == 0 ? 0 : double(alpha) * sum;
            if (beta != 0) { value += double(beta) * c[index]; }
            reference[index] = float(value);
        }
    }
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    RT(vkdnnMalloc(&da, a.size() * 4));
    RT(vkdnnMalloc(&db, b.size() * 4));
    RT(vkdnnMalloc(&dc, c.size() * 4));
    RT(vkdnnMemcpy(da, a.data(), a.size() * 4, vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(db, b.data(), b.size() * 4, vkdnnMemcpyHostToDevice));
    RT(vkdnnMemcpy(dc, c.data(), c.size() * 4, vkdnnMemcpyHostToDevice));
    const float* ap = nullptr;
    const float* bp = nullptr;
    if (alpha != 0 && k)
    {
        ap = static_cast<float*>(da) + prefix;
        bp = static_cast<float*>(db) + prefix;
    }
    const float* alphaPointer = k == 0 ? nullptr : &alpha;
    OK(vkblasSgemm(handle, ta, tb, m, n, k, alphaPointer, ap, lda, bp, ldb, &beta, static_cast<float*>(dc) + prefix, ldc));
    RT(vkdnnMemcpy(c.data(), dc, c.size() * 4, vkdnnMemcpyDeviceToHost));
    for (size_t i = 0; i < c.size(); ++i)
    {
        if (!(std::abs(c[i] - reference[i]) <= 4e-5f * std::max(1.f, std::abs(reference[i]))))
        {
            std::fprintf(stderr, "index %zu: %g vs %g\n", i, c[i], reference[i]);
            REQUIRE(false);
        }
    }
    RT(vkdnnFree(da));
    RT(vkdnnFree(db));
    RT(vkdnnFree(dc));
}

static void reductions(vkblasHandle_t handle)
{
    for (int count : { 1, 255, 256, 257, 65537 })
    {
        for (int stride : { 1, 3 })
        {
            std::vector<float> values(size_t(count) * stride + 3, -99);
            double expectedSum = 0, expectedDot = 0;
            for (int index = 0; index < count; ++index)
            {
                float value = float(index % 17 - 8) / 4;
                values[3 + size_t(index) * stride] = value;
                expectedSum += std::abs(value);
                expectedDot += double(value) * value;
            }
            void* allocation = nullptr;
            RT(vkdnnMalloc(&allocation, values.size() * sizeof(float)));
            RT(vkdnnMemcpy(allocation, values.data(), values.size() * sizeof(float), vkdnnMemcpyHostToDevice));
            auto view = static_cast<float*>(allocation) + 3;
            float actual = -1;
            OK(vkblasSasum(handle, count, view, stride, &actual));
            REQUIRE(std::abs(actual - expectedSum) <= 1e-5 * std::max(1.0, expectedSum));
            OK(vkblasSdot(handle, count, view, stride, view, stride, &actual));
            REQUIRE(std::abs(actual - expectedDot) <= 1e-5 * std::max(1.0, expectedDot));
            REQUIRE(vkblasSasum(handle, count, nullptr, stride, &actual) == VKBLAS_STATUS_INVALID_VALUE);
            REQUIRE(vkblasSdot(handle, count, view, stride, nullptr, stride, &actual) == VKBLAS_STATUS_INVALID_VALUE);
            REQUIRE(vkblasSasum(handle, count, view, stride, nullptr) == VKBLAS_STATUS_INVALID_VALUE);
            OK(vkblasSasum(handle, 0, nullptr, 1, &actual));
            REQUIRE(actual == 0);
            RT(vkdnnFree(allocation));
        }
    }
    std::puts("Native Vulkan asum/dot reductions passed");
}

static void scaling(vkblasHandle_t handle)
{
    for (int n : { 1, 19, 257, 65537 })
    {
        for (int incx : { 1, 3 })
        {
            const size_t prefix = 3;
            size_t extent = size_t(n - 1) * incx + 1;
            std::vector<float> original(prefix + extent + 2, -123.f);
            for (int i = 0; i < n; ++i) { original[prefix + size_t(i) * incx] = float(i % 17 - 8) / 4; }
            for (float alpha : { -2.f, 0.f, 1.f, 0.7f })
            {
                auto actual = original;
                auto expected = original;
                for (int i = 0; i < n; ++i) { expected[prefix + size_t(i) * incx] *= alpha; }
                void* device = nullptr;
                RT(vkdnnMalloc(&device, actual.size() * sizeof(float)));
                RT(vkdnnMemcpy(device, actual.data(), actual.size() * sizeof(float), vkdnnMemcpyHostToDevice));
                OK(vkblasSscal(handle, n, &alpha, static_cast<float*>(device) + prefix, incx));
                RT(vkdnnMemcpy(actual.data(), device, actual.size() * sizeof(float), vkdnnMemcpyDeviceToHost));
                for (size_t i = 0; i < actual.size(); ++i) { REQUIRE(std::abs(actual[i] - expected[i]) <= 1e-6f); }
                RT(vkdnnFree(device));
            }
        }
    }
    float alpha = 0;
    std::vector<float> special = { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -2.f };
    void* device = nullptr;
    RT(vkdnnMalloc(&device, special.size() * sizeof(float)));
    RT(vkdnnMemcpy(device, special.data(), special.size() * sizeof(float), vkdnnMemcpyHostToDevice));
    OK(vkblasSscal(handle, 3, &alpha, static_cast<float*>(device), 1));
    RT(vkdnnMemcpy(special.data(), device, special.size() * sizeof(float), vkdnnMemcpyDeviceToHost));
    REQUIRE(std::isnan(special[0]) && std::isnan(special[1]) && special[2] == 0 && std::signbit(special[2]));
    RT(vkdnnFree(device));
}

int main()
{
    int count = 0;
    auto result = vkdnnGetDeviceCount(&count);
    if (result == vkdnnErrorNoDevice || (result == vkdnnSuccess && !count))
    {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    RT(result);
    vkdnnDeviceProp prop{};
    RT(vkdnnGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n", prop.name);
    vkblasHandle_t handle = nullptr;
    REQUIRE(vkblasCreate(nullptr) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasDestroy(nullptr) == VKBLAS_STATUS_NOT_INITIALIZED);
    REQUIRE(vkblasSetPointerMode(nullptr, VKBLAS_POINTER_MODE_HOST) == VKBLAS_STATUS_NOT_INITIALIZED);
    OK(vkblasCreate(&handle));
    reductions(handle);
    REQUIRE(vkblasSscal(nullptr, 0, nullptr, nullptr, 1) == VKBLAS_STATUS_NOT_INITIALIZED);
    OK(vkblasSscal(handle, 0, nullptr, nullptr, 1));
    OK(vkblasSscal(handle, -1, nullptr, nullptr, 1));
    OK(vkblasSscal(handle, 3, nullptr, nullptr, 0));
    OK(vkblasSscal(handle, 3, nullptr, nullptr, -1));
    scaling(handle);
    REQUIRE(vkblasSetPointerMode(handle, VKBLAS_POINTER_MODE_DEVICE) == VKBLAS_STATUS_NOT_SUPPORTED);
    REQUIRE(vkblasSetPointerMode(handle, static_cast<vkblasPointerMode_t>(99)) == VKBLAS_STATUS_INVALID_VALUE);
    for (auto ta : { VKBLAS_OP_N, VKBLAS_OP_T, VKBLAS_OP_C })
    {
        for (auto tb : { VKBLAS_OP_N, VKBLAS_OP_T, VKBLAS_OP_C })
        {
            numerical(handle, ta, tb, 19, 23, 29, 0.7f, -0.4f);
            numerical(handle, ta, tb, 3, 5, 7, 1.f, 0.f);
        }
    }
    numerical(handle, VKBLAS_OP_N, VKBLAS_OP_N, 16, 32, 16, 1, 1);
    for (auto ta : { VKBLAS_OP_N, VKBLAS_OP_T })
    {
        for (auto tb : { VKBLAS_OP_N, VKBLAS_OP_T })
        {
            numerical(handle, ta, tb, 10, 100, 784, 0.7f, -0.4f);
            numerical(handle, ta, tb, 16, 23, 129, 1.f, 0.f);
        }
    }
    numerical(handle, VKBLAS_OP_T, VKBLAS_OP_C, 1, 1, 1, -1, 2);
    numerical(handle, VKBLAS_OP_N, VKBLAS_OP_N, 17, 9, 0, std::numeric_limits<float>::quiet_NaN(), 0);
    numerical(handle, VKBLAS_OP_T, VKBLAS_OP_N, 17, 9, 0, 1, -2);
    numerical(handle, VKBLAS_OP_N, VKBLAS_OP_T, 17, 9, 20, 0, 0);
    numerical(handle, VKBLAS_OP_T, VKBLAS_OP_N, 17, 9, 20, 0, -2);
    OK(vkblasSetPointerMode(handle, VKBLAS_POINTER_MODE_HOST));
    OK(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 0, 5, 3, nullptr, nullptr, 1, nullptr, 3, nullptr, nullptr, 1));
    OK(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 5, 0, 3, nullptr, nullptr, 5, nullptr, 3, nullptr, nullptr, 5));
    float one = 1;
    void *a = nullptr, *b = nullptr, *c = nullptr;
    RT(vkdnnMalloc(&a, 64));
    RT(vkdnnMalloc(&b, 64));
    RT(vkdnnMalloc(&c, 64));
    REQUIRE(vkblasSscal(handle, 1, nullptr, static_cast<float*>(a), 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSscal(handle, 1, static_cast<float*>(b), static_cast<float*>(a), 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSscal(handle, 1, &one, nullptr, 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSscal(handle, 1, &one, &one, 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSscal(handle, 1, &one, reinterpret_cast<float*>(static_cast<char*>(a) + 1), 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSscal(handle, 6, &one, static_cast<float*>(a) + 1, 3) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSscal(handle, std::numeric_limits<int>::max(), &one, static_cast<float*>(a), 3) == VKBLAS_STATUS_NOT_SUPPORTED);
    RT(vkdnnMemset(a, 0, 64));
    OK(vkblasSscal(handle, 6, &one, static_cast<float*>(a), 3));
    auto call = [&](const float* ap, const float* bp, float* cp, int lda = 2)
    {
        return vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, &one, ap, lda, bp, 2, &one, cp, 2);
    };
    REQUIRE(call(&one, static_cast<float*>(b), static_cast<float*>(c)) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(call(static_cast<float*>(a), nullptr, static_cast<float*>(c)) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(call(static_cast<float*>(a), static_cast<float*>(b), static_cast<float*>(c), 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(call(reinterpret_cast<float*>(static_cast<char*>(a) + 1), static_cast<float*>(b), static_cast<float*>(c)) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(call(static_cast<float*>(a) + 15, static_cast<float*>(b), static_cast<float*>(c)) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(call(static_cast<float*>(a), static_cast<float*>(b), static_cast<float*>(a)) == VKBLAS_STATUS_NOT_SUPPORTED);
    REQUIRE(call(static_cast<float*>(a), static_cast<float*>(b), static_cast<float*>(c) + 15) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, &one, static_cast<float*>(a), 2, static_cast<float*>(b), 1, &one, static_cast<float*>(c), 2) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, &one, static_cast<float*>(a), 2, static_cast<float*>(b), 2, &one, static_cast<float*>(c), 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, -1, &one, static_cast<float*>(a), 2, static_cast<float*>(b), 2, &one, static_cast<float*>(c), 2) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, nullptr, static_cast<float*>(a), 2, static_cast<float*>(b), 2, &one, static_cast<float*>(c), 2) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, &one, static_cast<float*>(a), 2, static_cast<float*>(b), 2, nullptr, static_cast<float*>(c), 2) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, static_cast<vkblasOperation_t>(99), VKBLAS_OP_N, 0, 0, 0, nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr, 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, -1, 0, 0, nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr, 1) == VKBLAS_STATUS_INVALID_VALUE);
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 1, 1, 1, static_cast<float*>(a), static_cast<float*>(a), 1, static_cast<float*>(b), 1, &one, static_cast<float*>(c), 1) == VKBLAS_STATUS_INVALID_VALUE);
    if (count > 1)
    {
        RT(vkdnnSetDevice(1));
        REQUIRE(vkblasSscal(handle, 0, nullptr, nullptr, 1) == VKBLAS_STATUS_INVALID_VALUE);
        REQUIRE(call(static_cast<float*>(a), static_cast<float*>(b), static_cast<float*>(c)) == VKBLAS_STATUS_INVALID_VALUE);
        void* foreign = nullptr;
        RT(vkdnnMalloc(&foreign, 64));
        RT(vkdnnSetDevice(0));
        REQUIRE(vkblasSscal(handle, 1, &one, static_cast<float*>(foreign), 1) == VKBLAS_STATUS_INVALID_VALUE);
        REQUIRE(call(static_cast<float*>(foreign), static_cast<float*>(b), static_cast<float*>(c)) == VKBLAS_STATUS_INVALID_VALUE);
        RT(vkdnnSetDevice(1));
        RT(vkdnnFree(foreign));
        RT(vkdnnSetDevice(0));
    }
    else
    {
        std::puts("SKIP: cross-device ownership test needs two devices");
    }
    // Last-column padding is not required; all three views share one allocation.
    std::vector<float> packed = { 1, 2, -99, -99, -99, 3, 4, 5, 6, -99, -99, -99, 7, 8,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), -99, -99, -99,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN() };
    void* shared = nullptr;
    RT(vkdnnMalloc(&shared, packed.size() * 4));
    RT(vkdnnMemcpy(shared, packed.data(), packed.size() * 4, vkdnnMemcpyHostToDevice));
    float zero = 0;
    auto view = static_cast<float*>(shared);
    OK(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 2, 2, 2, &one, view, 5, view + 7, 5, &zero, view + 14, 5));
    RT(vkdnnMemcpy(packed.data(), shared, packed.size() * 4, vkdnnMemcpyDeviceToHost));
    REQUIRE(packed[14] == 23 && packed[15] == 34 && packed[19] == 31 && packed[20] == 46);
    REQUIRE(packed[16] == -99 && packed[17] == -99 && packed[18] == -99);
    RT(vkdnnFree(shared));
    REQUIRE(vkblasSgemm(handle, VKBLAS_OP_N, VKBLAS_OP_N, 1, 4, 1, &one, static_cast<float*>(a), 1,
                static_cast<float*>(b), 1, &one, static_cast<float*>(c), std::numeric_limits<int>::max())
        == VKBLAS_STATUS_NOT_SUPPORTED);
    RT(vkdnnFree(a));
    REQUIRE(vkblasSscal(handle, 1, &one, static_cast<float*>(a), 1) == VKBLAS_STATUS_INVALID_VALUE);
    RT(vkdnnFree(b));
    RT(vkdnnFree(c));
    RT(vkdnnDeviceReset());
    asynchronous(handle);
    scaling(handle);
    numerical(handle, VKBLAS_OP_N, VKBLAS_OP_T, 7, 3, 5, 1, 0);
    OK(vkblasDestroy(handle));
    RT(vkdnnDeviceReset());
    std::puts("PASS: Vulkan SGEMM/Sscal and CPU reference, bounds, zero-access and lifecycle");
    return 0;
}
