#include "vkdnn/vkdnn_conv_query.h"
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
    (void)workspace;
    (void)workspaceBytes;
    if (!xp || !wp || !yp) { return VKDNN_STATUS_BAD_PARAM; }
    return vkdnnGetConvolutionForwardAlgorithm_v7(handle, x, w, conv, y, requested, returned, results);
}
