#include "vkdnn/vkdnn_conv_query.h"
#include <chrono>
#include <limits>

namespace
{
template <class Perf, class Algo, class Query>
vkdnnStatus_t queryAlgorithm(int requested, int* returned, Perf* results, Algo algo, Query query)
{
    if (requested <= 0 || !returned || !results) { return VKDNN_STATUS_BAD_PARAM; }
    size_t bytes = 0;
    const auto status = query(&bytes);
    if (status != VKDNN_STATUS_SUCCESS) { return status; }
    Perf result{};
    result.algo = algo;
    result.status = VKDNN_STATUS_SUCCESS;
    result.time = std::numeric_limits<float>::quiet_NaN();
    result.memory = bytes;
    result.determinism = VKDNN_DETERMINISTIC;
    result.mathType = VKDNN_DEFAULT_MATH;
    results[0] = result;
    *returned = 1;
    return VKDNN_STATUS_SUCCESS;
}
template <class Perf, class Query, class Execute>
vkdnnStatus_t findAlgorithm(int requested, int* returned, Perf* results, void* workspace,
    size_t workspaceBytes, Query query, Execute execute)
{
    if (requested <= 0 || !returned || !results) { return VKDNN_STATUS_BAD_PARAM; }
    Perf result{};
    int count = 0;
    auto status = query(&count, &result);
    if (status != VKDNN_STATUS_SUCCESS) { return status; }
    if (result.memory > workspaceBytes || (result.memory && !workspace)) { return VKDNN_STATUS_BAD_PARAM; }
    const auto start = std::chrono::steady_clock::now();
    status = execute();
    const auto end = std::chrono::steady_clock::now();
    if (status != VKDNN_STATUS_SUCCESS) { return status; }
    result.time = std::chrono::duration<float, std::milli>(end - start).count();
    results[0] = result;
    *returned = count;
    return VKDNN_STATUS_SUCCESS;
}
}    //namespace

vkdnnStatus_t vkdnnGetConvolutionForwardAlgorithm_v7(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t x, vkdnnFilterDescriptor_t w, vkdnnConvolutionDescriptor_t conv,
    vkdnnTensorDescriptor_t y, int requested, int* returned, vkdnnConvolutionFwdAlgoPerf_t* results)
{
    return queryAlgorithm(requested, returned, results, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, [&](size_t* bytes)
        {
            return vkdnnGetConvolutionForwardWorkspaceSize(handle, x, w, conv, y, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, bytes);
        });
}
vkdnnStatus_t vkdnnGetConvolutionBackwardDataAlgorithm_v7(vkdnnHandle_t handle,
    vkdnnFilterDescriptor_t w, vkdnnTensorDescriptor_t dy, vkdnnConvolutionDescriptor_t conv,
    vkdnnTensorDescriptor_t dx, int requested, int* returned, vkdnnConvolutionBwdDataAlgoPerf_t* results)
{
    return queryAlgorithm(requested, returned, results, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, [&](size_t* bytes)
        {
            return vkdnnGetConvolutionBackwardDataWorkspaceSize(handle, w, dy, conv, dx, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, bytes);
        });
}
vkdnnStatus_t vkdnnGetConvolutionBackwardFilterAlgorithm_v7(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t x, vkdnnTensorDescriptor_t dy, vkdnnConvolutionDescriptor_t conv,
    vkdnnFilterDescriptor_t dw, int requested, int* returned, vkdnnConvolutionBwdFilterAlgoPerf_t* results)
{
    return queryAlgorithm(requested, returned, results, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, [&](size_t* bytes)
        {
            return vkdnnGetConvolutionBackwardFilterWorkspaceSize(handle, x, dy, conv, dw, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, bytes);
        });
}
vkdnnStatus_t vkdnnFindConvolutionForwardAlgorithmEx(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t x, const void* xp, vkdnnFilterDescriptor_t w, const void* wp,
    vkdnnConvolutionDescriptor_t conv, vkdnnTensorDescriptor_t y, void* yp,
    int requested, int* returned, vkdnnConvolutionFwdAlgoPerf_t* results, void* workspace, size_t workspaceBytes)
{
    const float alpha = 1.0f, beta = 0.0f;
    return findAlgorithm(requested, returned, results, workspace, workspaceBytes, [&](int* count, vkdnnConvolutionFwdAlgoPerf_t* perf)
        {
            return vkdnnGetConvolutionForwardAlgorithm_v7(handle, x, w, conv, y, 1, count, perf);
        },
        [&]
        {
            return vkdnnConvolutionForward(handle, &alpha, x, xp, w, wp, conv, VKDNN_CONVOLUTION_FWD_ALGO_DIRECT, workspace, workspaceBytes, &beta, y, yp);
        });
}
vkdnnStatus_t vkdnnFindConvolutionBackwardDataAlgorithmEx(vkdnnHandle_t handle,
    vkdnnFilterDescriptor_t w, const void* wp, vkdnnTensorDescriptor_t dy, const void* dyp,
    vkdnnConvolutionDescriptor_t conv, vkdnnTensorDescriptor_t dx, void* dxp,
    int requested, int* returned, vkdnnConvolutionBwdDataAlgoPerf_t* results, void* workspace, size_t workspaceBytes)
{
    const float alpha = 1.0f, beta = 0.0f;
    return findAlgorithm(requested, returned, results, workspace, workspaceBytes, [&](int* count, vkdnnConvolutionBwdDataAlgoPerf_t* perf)
        {
            return vkdnnGetConvolutionBackwardDataAlgorithm_v7(handle, w, dy, conv, dx, 1, count, perf);
        },
        [&]
        {
            return vkdnnConvolutionBackwardData(handle, &alpha, w, wp, dy, dyp, conv, VKDNN_CONVOLUTION_BWD_DATA_ALGO_0, workspace, workspaceBytes, &beta, dx, dxp);
        });
}
vkdnnStatus_t vkdnnFindConvolutionBackwardFilterAlgorithmEx(vkdnnHandle_t handle,
    vkdnnTensorDescriptor_t x, const void* xp, vkdnnTensorDescriptor_t dy, const void* dyp,
    vkdnnConvolutionDescriptor_t conv, vkdnnFilterDescriptor_t dw, void* dwp,
    int requested, int* returned, vkdnnConvolutionBwdFilterAlgoPerf_t* results, void* workspace, size_t workspaceBytes)
{
    const float alpha = 1.0f, beta = 0.0f;
    return findAlgorithm(requested, returned, results, workspace, workspaceBytes, [&](int* count, vkdnnConvolutionBwdFilterAlgoPerf_t* perf)
        {
            return vkdnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x, dy, conv, dw, 1, count, perf);
        },
        [&]
        {
            return vkdnnConvolutionBackwardFilter(handle, &alpha, x, xp, dy, dyp, conv, VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0, workspace, workspaceBytes, &beta, dw, dwp);
        });
}
