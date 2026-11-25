#include "../core/compute_internal.h"
#include "vkdnn/vkdnn_ops.h"
#include <climits>
#include <cmath>
#include <mutex>
#include <new>
#include <unordered_set>

struct vkdnnDnnContext
{
    int device = 0;
};
struct vkdnnTensorStruct
{
    uint32_t n = 0, c = 0, h = 0, w = 0, count = 0;
};
struct vkdnnActivationStruct
{
    bool initialized = false;
    vkdnnNanPropagation_t nan = VKDNN_NOT_PROPAGATE_NAN;
    vkdnnActivationMode_t mode = VKDNN_ACTIVATION_RELU;
};
struct vkdnnFilterStruct
{
    uint32_t k = 0, c = 0, h = 0, w = 0, count = 0;
};
struct vkdnnConvolutionStruct
{
    bool initialized = false;
    uint32_t ph = 0, pw = 0, sh = 1, sw = 1, dh = 1, dw = 1, groups = 1;
    vkdnnConvolutionMode_t mode = VKDNN_CROSS_CORRELATION;
};
struct vkdnnPoolingStruct
{
    bool initialized = false;
    vkdnnPoolingMode_t mode = VKDNN_POOLING_MAX;
    vkdnnNanPropagation_t nan = VKDNN_NOT_PROPAGATE_NAN;
    uint32_t kh = 0, kw = 0, ph = 0, pw = 0, sh = 0, sw = 0;
};
struct vkdnnOpTensorStruct
{
    bool initialized = false;
    vkdnnOpTensorOp_t op = VKDNN_OP_TENSOR_ADD;
    vkdnnNanPropagation_t nan = VKDNN_NOT_PROPAGATE_NAN;
};

namespace
{
std::mutex mutex;
std::unordered_set<vkdnnHandle_t> handles;
std::unordered_set<vkdnnTensorDescriptor_t> tensors;
std::unordered_set<vkdnnActivationDescriptor_t> activations;
std::unordered_set<vkdnnPoolingDescriptor_t> pools;
std::unordered_set<vkdnnFilterDescriptor_t> filters;
std::unordered_set<vkdnnConvolutionDescriptor_t> convolutions;
std::unordered_set<vkdnnOpTensorDescriptor_t> tensorOps;

template <class F>
vkdnnStatus_t guarded(F function)
{
    try
    {
        std::lock_guard<std::mutex> lock(mutex);
        return function();
    } catch (const std::bad_alloc&) { return VKDNN_STATUS_ALLOC_FAILED; } catch (...)
    {
        return VKDNN_STATUS_INTERNAL_ERROR;
    }
}
template <class T>
vkdnnStatus_t create(T** output, std::unordered_set<T*>& registry)
{
    if (!output)
    {
        return VKDNN_STATUS_BAD_PARAM;
    }
    auto object = new T{};
    try
    {
        registry.insert(object);
    } catch (...)
    {
        delete object;
        throw;
    }
    *output = object;
    return VKDNN_STATUS_SUCCESS;
}
template <class T>
vkdnnStatus_t destroy(T* object, std::unordered_set<T*>& registry)
{
    if (!registry.erase(object))
    {
        return VKDNN_STATUS_BAD_PARAM;
    }
    delete object;
    return VKDNN_STATUS_SUCCESS;
}
bool tensor(vkdnnTensorDescriptor_t t) { return tensors.count(t) && t->count; }
bool nanOption(vkdnnNanPropagation_t n) { return n == VKDNN_NOT_PROPAGATE_NAN || n == VKDNN_PROPAGATE_NAN; }
bool equal(vkdnnTensorDescriptor_t x, vkdnnTensorDescriptor_t y)
{
    return x->n == y->n && x->c == y->c && x->h == y->h && x->w == y->w;
}
vkdnnStatus_t mapStatus(vkdnnError_t status)
{
    switch (status)
    {
    case vkdnnSuccess: return VKDNN_STATUS_SUCCESS;
    case vkdnnErrorMemoryAllocation: return VKDNN_STATUS_ALLOC_FAILED;
    case vkdnnErrorInvalidValue:
    case vkdnnErrorInvalidDevice: return VKDNN_STATUS_BAD_PARAM;
    case vkdnnErrorNoDevice: return VKDNN_STATUS_NOT_INITIALIZED;
    case vkdnnErrorNotSupported: return VKDNN_STATUS_NOT_SUPPORTED;
    default: return VKDNN_STATUS_EXECUTION_FAILED;
    }
}
vkdnnStatus_t execute(vkdnnHandle_t handle, const void* alpha, vkdnnTensorDescriptor_t xd, const void* x,
    const void* beta, vkdnnTensorDescriptor_t yd, void* y, ComputeParams p)
{
    if (!handles.count(handle))
    {
        return VKDNN_STATUS_NOT_INITIALIZED;
    }
    if (!tensor(xd) || !tensor(yd))
    {
        return VKDNN_STATUS_BAD_PARAM;
    }
    auto scalar = runtimeHostScalar(static_cast<const float*>(alpha), p.alpha);
    if (scalar != vkdnnSuccess) { return mapStatus(scalar); }
    scalar = runtimeHostScalar(static_cast<const float*>(beta), p.beta);
    if (scalar != vkdnnSuccess) { return mapStatus(scalar); }
    p.n = xd->n;
    p.c = xd->c;
    p.h = xd->h;
    p.w = xd->w;
    p.oh = yd->h;
    p.ow = yd->w;
    p.outputC = yd->c;
    if (p.op < 2 || p.op == 4 || p.op == 5)
    {
        p.count = yd->count;
    }
    else if (p.mode == VKDNN_SOFTMAX_MODE_INSTANCE)
    {
        // Softmax count denotes independent reductions, not scalar output elements.
        p.count = xd->n;
    }
    else
    {
        p.count = xd->n * xd->h * xd->w;
    }
    size_t xBytes = size_t(xd->count) * 4;
    if (p.op == 5) { xBytes = 0; }
    if ((p.op == 0 || p.op == 4) && p.alpha == 0) { xBytes = 0; }
    return mapStatus(runtimeCompute(handle->device, x, xBytes, y, size_t(yd->count) * 4, p));
}
vkdnnStatus_t convolutionShape(vkdnnConvolutionDescriptor_t p, vkdnnTensorDescriptor_t x,
    vkdnnFilterDescriptor_t f, int& h, int& w)
{
    if (!convolutions.count(p) || !p->initialized || !tensor(x) || !filters.count(f) || !f->count)
    {
        return VKDNN_STATUS_BAD_PARAM;
    }
    if (x->c % p->groups || f->k % p->groups || x->c / p->groups != f->c)
    {
        return VKDNN_STATUS_BAD_PARAM;
    }
    int64_t eh = int64_t(p->dh) * (f->h - 1) + 1;
    int64_t ew = int64_t(p->dw) * (f->w - 1) + 1;
    int64_t hh = int64_t(x->h) + 2 * int64_t(p->ph);
    int64_t ww = int64_t(x->w) + 2 * int64_t(p->pw);
    if (eh > hh || ew > ww) { return VKDNN_STATUS_BAD_PARAM; }
    // Coordinates are signed 32-bit in the shader, including the last dilated tap.
    if (hh > INT_MAX || ww > INT_MAX || eh > INT_MAX || ew > INT_MAX)
    {
        return VKDNN_STATUS_NOT_SUPPORTED;
    }
    hh = (hh - eh) / p->sh + 1;
    ww = (ww - ew) / p->sw + 1;
    uint64_t count = uint64_t(x->n) * f->k;
    if (count > INT_MAX / 4 || count * hh > INT_MAX / 4 || count * hh * ww > INT_MAX / 4)
    {
        return VKDNN_STATUS_NOT_SUPPORTED;
    }
    h = int(hh);
    w = int(ww);
    return VKDNN_STATUS_SUCCESS;
}
vkdnnStatus_t convolutionOutput(vkdnnConvolutionDescriptor_t p, vkdnnTensorDescriptor_t x,
    vkdnnFilterDescriptor_t f, vkdnnTensorDescriptor_t y)
{
    int h, w;
    auto s = convolutionShape(p, x, f, h, w);
    if (s != VKDNN_STATUS_SUCCESS) { return s; }
    if (!tensor(y) || y->n != x->n || y->c != f->k || y->h != uint32_t(h) || y->w != uint32_t(w))
    {
        return VKDNN_STATUS_BAD_PARAM;
    }
    return VKDNN_STATUS_SUCCESS;
}
bool outputDim(vkdnnPoolingDescriptor_t p, vkdnnTensorDescriptor_t x, int& h, int& w)
{
    int64_t hh = int64_t(x->h) + 2 * int64_t(p->ph) - p->kh;
    int64_t ww = int64_t(x->w) + 2 * int64_t(p->pw) - p->kw;
    if (int64_t(x->h) + 2 * int64_t(p->ph) > INT_MAX || int64_t(x->w) + 2 * int64_t(p->pw) > INT_MAX)
    {
        return false;
    }
    if (hh < 0 || ww < 0)
    {
        return false;
    }
    hh = hh / p->sh + 1;
    ww = ww / p->sw + 1;
    if (hh > INT_MAX || ww > INT_MAX)
    {
        return false;
    }
    h = int(hh);
    w = int(ww);
    return true;
}
ConvolutionParams backwardParams(vkdnnTensorDescriptor_t x, vkdnnFilterDescriptor_t f,
    vkdnnConvolutionDescriptor_t p, vkdnnTensorDescriptor_t y, uint32_t op)
{
    ConvolutionParams q{};
    q.op = op;
    q.n = x->n;
    q.c = x->c;
    q.h = x->h;
    q.w = x->w;
    q.k = f->k;
    q.fc = f->c;
    q.kh = f->h;
    q.kw = f->w;
    q.oh = y->h;
    q.ow = y->w;
    q.ph = p->ph;
    q.pw = p->pw;
    q.sh = p->sh;
    q.sw = p->sw;
    q.dh = p->dh;
    q.dw = p->dw;
    q.groups = p->groups;
    q.mode = p->mode;
    return q;
}
vkdnnStatus_t backwardExecute(vkdnnHandle_t handle, const void* alpha, const void* beta,
    const void* a, size_t ab, const void* b, size_t bb, void* out, size_t bytes, ConvolutionParams q,
    const void* extra = nullptr, size_t extraBytes = 0)
{
    auto s = runtimeHostScalar(static_cast<const float*>(alpha), q.alpha);
    if (s != vkdnnSuccess) { return mapStatus(s); }
    s = runtimeHostScalar(static_cast<const float*>(beta), q.beta);
    if (s != vkdnnSuccess) { return mapStatus(s); }
    if (q.alpha == 0)
    {
        ab = 0;
        bb = 0;
        extraBytes = 0;
    }
    return mapStatus(runtimeConvolution(handle->device, a, ab, b, bb, out, bytes, q, extra, extraBytes));
}
}    //namespace

vkdnnStatus_t vkdnnGetConvolutionBackwardDataWorkspaceSize(vkdnnHandle_t handle, vkdnnFilterDescriptor_t f,
    vkdnnTensorDescriptor_t y, vkdnnConvolutionDescriptor_t p, vkdnnTensorDescriptor_t x, vkdnnConvolutionBwdDataAlgo_t algo, size_t* bytes)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            auto s = convolutionOutput(p, x, f, y);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (!bytes) { return VKDNN_STATUS_BAD_PARAM; }
            if (algo != VKDNN_CONVOLUTION_BWD_DATA_ALGO_0) { return VKDNN_STATUS_NOT_SUPPORTED; }
            *bytes = 0;
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnGetConvolutionBackwardFilterWorkspaceSize(vkdnnHandle_t handle, vkdnnTensorDescriptor_t x,
    vkdnnTensorDescriptor_t y, vkdnnConvolutionDescriptor_t p, vkdnnFilterDescriptor_t f, vkdnnConvolutionBwdFilterAlgo_t algo, size_t* bytes)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            auto s = convolutionOutput(p, x, f, y);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (!bytes) { return VKDNN_STATUS_BAD_PARAM; }
            if (algo != VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0) { return VKDNN_STATUS_NOT_SUPPORTED; }
            *bytes = 0;
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnConvolutionBackwardData(vkdnnHandle_t handle, const void* alpha, vkdnnFilterDescriptor_t f, const void* weights,
    vkdnnTensorDescriptor_t y, const void* dy, vkdnnConvolutionDescriptor_t p, vkdnnConvolutionBwdDataAlgo_t algo,
    void* workspace, size_t workspaceBytes, const void* beta, vkdnnTensorDescriptor_t x, void* dx)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            auto s = convolutionOutput(p, x, f, y);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (algo != VKDNN_CONVOLUTION_BWD_DATA_ALGO_0) { return VKDNN_STATUS_NOT_SUPPORTED; }
            (void)workspace;
            (void)workspaceBytes;
            auto q = backwardParams(x, f, p, y, 1);
            q.count = x->count;
            return backwardExecute(handle, alpha, beta, weights, size_t(f->count) * 4, dy, size_t(y->count) * 4, dx, size_t(x->count) * 4, q);
        });
}
vkdnnStatus_t vkdnnConvolutionBackwardFilter(vkdnnHandle_t handle, const void* alpha, vkdnnTensorDescriptor_t x, const void* input,
    vkdnnTensorDescriptor_t y, const void* dy, vkdnnConvolutionDescriptor_t p, vkdnnConvolutionBwdFilterAlgo_t algo,
    void* workspace, size_t workspaceBytes, const void* beta, vkdnnFilterDescriptor_t f, void* dw)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            auto s = convolutionOutput(p, x, f, y);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (algo != VKDNN_CONVOLUTION_BWD_FILTER_ALGO_0) { return VKDNN_STATUS_NOT_SUPPORTED; }
            (void)workspace;
            (void)workspaceBytes;
            auto q = backwardParams(x, f, p, y, 2);
            q.count = f->count;
            return backwardExecute(handle, alpha, beta, input, size_t(x->count) * 4, dy, size_t(y->count) * 4, dw, size_t(f->count) * 4, q);
        });
}
vkdnnStatus_t vkdnnConvolutionBackwardBias(vkdnnHandle_t handle, const void* alpha, vkdnnTensorDescriptor_t y,
    const void* dy, const void* beta, vkdnnTensorDescriptor_t b, void* db)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (!tensor(y) || !tensor(b) || b->n != 1 || b->h != 1 || b->w != 1 || b->c != y->c) { return VKDNN_STATUS_BAD_PARAM; }
            ConvolutionParams q{};
            q.op = 3;
            q.count = b->count;
            q.n = y->n;
            q.k = y->c;
            q.oh = y->h;
            q.ow = y->w;
            return backwardExecute(handle, alpha, beta, dy, size_t(y->count) * 4, nullptr, 0, db, size_t(b->count) * 4, q);
        });
}
vkdnnStatus_t vkdnnPoolingBackward(vkdnnHandle_t handle, vkdnnPoolingDescriptor_t p, const void* alpha,
    vkdnnTensorDescriptor_t y, const void* output, vkdnnTensorDescriptor_t dyd, const void* dy,
    vkdnnTensorDescriptor_t x, const void* input, const void* beta, vkdnnTensorDescriptor_t dxd, void* dx)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (!pools.count(p) || !p->initialized || !tensor(x) || !tensor(y) || !tensor(dyd) || !tensor(dxd)
                || !equal(y, dyd) || !equal(x, dxd)) { return VKDNN_STATUS_BAD_PARAM; }
            int h, w;
            if (!outputDim(p, x, h, w) || y->n != x->n || y->c != x->c || y->h != uint32_t(h) || y->w != uint32_t(w)) { return VKDNN_STATUS_BAD_PARAM; }
            ConvolutionParams q{};
            q.op = 4;
            q.count = x->count;
            q.h = x->h;
            q.w = x->w;
            q.oh = y->h;
            q.ow = y->w;
            q.kh = p->kh;
            q.kw = p->kw;
            q.ph = p->ph;
            q.pw = p->pw;
            q.sh = p->sh;
            q.sw = p->sw;
            q.mode = p->mode;
            q.nan = p->nan;
            // Recompute the winner from x; y still participates in ownership/range/alias validation.
            return backwardExecute(handle, alpha, beta, input, size_t(x->count) * 4, dy, size_t(y->count) * 4, dx, size_t(x->count) * 4, q, output, size_t(y->count) * 4);
        });
}

vkdnnStatus_t vkdnnCreate(vkdnnHandle_t* h)
{
    return guarded([&]
        {
            if (!h)
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            int device = 0;
            auto s = mapStatus(vkdnnGetDevice(&device));
            if (s != VKDNN_STATUS_SUCCESS)
            {
                return s;
            }
            s = create(h, handles);
            if (s == VKDNN_STATUS_SUCCESS)
            {
                (*h)->device = device;
            }
            return s;
        });
}
vkdnnStatus_t vkdnnDestroy(vkdnnHandle_t h)
{
    return guarded([&]
        {
            return destroy(h, handles);
        });
}
vkdnnStatus_t vkdnnCreateTensorDescriptor(vkdnnTensorDescriptor_t* t)
{
    return guarded([&]
        {
            return create(t, tensors);
        });
}
vkdnnStatus_t vkdnnDestroyTensorDescriptor(vkdnnTensorDescriptor_t t)
{
    return guarded([&]
        {
            return destroy(t, tensors);
        });
}
vkdnnStatus_t vkdnnSetTensor4dDescriptor(vkdnnTensorDescriptor_t t, vkdnnTensorFormat_t format, vkdnnDataType_t type, int n, int c, int h, int w)
{
    return guarded([&]
        {
            if (!tensors.count(t) || n <= 0 || c <= 0 || h <= 0 || w <= 0)
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            if (format != VKDNN_TENSOR_NCHW || type != VKDNN_DATA_FLOAT)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            uint64_t count = 1;
            for (int d : { n, c, h, w })
            {
                count *= uint32_t(d);
                if (count > INT_MAX / 4)
                {
                    return VKDNN_STATUS_NOT_SUPPORTED;
                }
            }
            *t = { uint32_t(n), uint32_t(c), uint32_t(h), uint32_t(w), uint32_t(count) };
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnCreateActivationDescriptor(vkdnnActivationDescriptor_t* t)
{
    return guarded([&]
        {
            return create(t, activations);
        });
}
vkdnnStatus_t vkdnnDestroyActivationDescriptor(vkdnnActivationDescriptor_t t)
{
    return guarded([&]
        {
            return destroy(t, activations);
        });
}
vkdnnStatus_t vkdnnSetActivationDescriptor(vkdnnActivationDescriptor_t t, vkdnnActivationMode_t mode, vkdnnNanPropagation_t nan, double coef)
{
    return guarded([&]
        {
            if (!activations.count(t) || !nanOption(nan) || !std::isfinite(coef))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            if ((mode != VKDNN_ACTIVATION_RELU && mode != VKDNN_ACTIVATION_SIGMOID && mode != VKDNN_ACTIVATION_TANH) || coef != 0)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            *t = { true, nan, mode };
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnActivationForward(vkdnnHandle_t handle, vkdnnActivationDescriptor_t a, const void* alpha,
    vkdnnTensorDescriptor_t xd, const void* x, const void* beta, vkdnnTensorDescriptor_t yd, void* y)
{
    return guarded([&]
        {
            if (!activations.count(a) || !a->initialized || !tensor(xd) || !tensor(yd) || !equal(xd, yd))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ComputeParams p{};
            p.nan = a->nan;
            p.mode = a->mode;
            return execute(handle, alpha, xd, x, beta, yd, y, p);
        });
}

vkdnnStatus_t vkdnnActivationBackward(vkdnnHandle_t handle, vkdnnActivationDescriptor_t a, const void* alpha,
    vkdnnTensorDescriptor_t yd, const void* y, vkdnnTensorDescriptor_t dyd, const void* dy,
    vkdnnTensorDescriptor_t xd, const void* x, const void* beta, vkdnnTensorDescriptor_t dxd, void* dx)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (!activations.count(a) || !a->initialized || !tensor(xd) || !tensor(yd) || !tensor(dyd)
                || !tensor(dxd) || !equal(xd, yd) || !equal(xd, dyd) || !equal(xd, dxd))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ActivationBackwardParams p{};
            auto s = runtimeHostScalar(static_cast<const float*>(alpha), p.alpha);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            s = runtimeHostScalar(static_cast<const float*>(beta), p.beta);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            p.count = xd->count;
            p.mode = a->mode;
            p.nan = a->nan;
            return mapStatus(runtimeActivationBackward(handle->device, x, y, dy, dx, size_t(xd->count) * 4, p));
        });
}

vkdnnStatus_t vkdnnAddTensor(vkdnnHandle_t handle, const void* alpha, vkdnnTensorDescriptor_t a,
    const void* x, const void* beta, vkdnnTensorDescriptor_t c, void* y)
{
    return guarded([&]
        {
            if (!tensor(a) || !tensor(c)) { return VKDNN_STATUS_BAD_PARAM; }
            if ((a->n != 1 && a->n != c->n) || (a->c != 1 && a->c != c->c)
                || (a->h != 1 && a->h != c->h) || (a->w != 1 && a->w != c->w))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ComputeParams p{};
            p.op = 4;
            return execute(handle, alpha, a, x, beta, c, y, p);
        });
}

vkdnnStatus_t vkdnnCreateFilterDescriptor(vkdnnFilterDescriptor_t* f)
{
    return guarded([&]
        {
            return create(f, filters);
        });
}
vkdnnStatus_t vkdnnDestroyFilterDescriptor(vkdnnFilterDescriptor_t f)
{
    return guarded([&]
        {
            return destroy(f, filters);
        });
}
vkdnnStatus_t vkdnnSetFilter4dDescriptor(vkdnnFilterDescriptor_t f, vkdnnDataType_t type,
    vkdnnTensorFormat_t format, int k, int c, int h, int w)
{
    return guarded([&]
        {
            if (!filters.count(f) || k <= 0 || c <= 0 || h <= 0 || w <= 0) { return VKDNN_STATUS_BAD_PARAM; }
            if (type != VKDNN_DATA_FLOAT || format != VKDNN_TENSOR_NCHW) { return VKDNN_STATUS_NOT_SUPPORTED; }
            uint64_t count = 1;
            for (int d : { k, c, h, w })
            {
                count *= uint32_t(d);
                if (count > INT_MAX / 4) { return VKDNN_STATUS_NOT_SUPPORTED; }
            }
            *f = { uint32_t(k), uint32_t(c), uint32_t(h), uint32_t(w), uint32_t(count) };
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnCreateConvolutionDescriptor(vkdnnConvolutionDescriptor_t* p)
{
    return guarded([&]
        {
            return create(p, convolutions);
        });
}
vkdnnStatus_t vkdnnDestroyConvolutionDescriptor(vkdnnConvolutionDescriptor_t p)
{
    return guarded([&]
        {
            return destroy(p, convolutions);
        });
}
vkdnnStatus_t vkdnnSetConvolution2dDescriptor(vkdnnConvolutionDescriptor_t p, int ph, int pw,
    int sh, int sw, int dh, int dw, vkdnnConvolutionMode_t mode, vkdnnDataType_t type)
{
    return guarded([&]
        {
            if (!convolutions.count(p) || ph < 0 || pw < 0 || sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0
                || (mode != VKDNN_CONVOLUTION && mode != VKDNN_CROSS_CORRELATION)) { return VKDNN_STATUS_BAD_PARAM; }
            if (type != VKDNN_DATA_FLOAT) { return VKDNN_STATUS_NOT_SUPPORTED; }
            *p = { true, uint32_t(ph), uint32_t(pw), uint32_t(sh), uint32_t(sw), uint32_t(dh), uint32_t(dw), p->groups, mode };
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnSetConvolutionGroupCount(vkdnnConvolutionDescriptor_t p, int groups)
{
    return guarded([&]
        {
            if (!convolutions.count(p) || groups <= 0) { return VKDNN_STATUS_BAD_PARAM; }
            p->groups = uint32_t(groups);
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnSetConvolutionMathType(vkdnnConvolutionDescriptor_t p, vkdnnMathType_t type)
{
    return guarded([&]
        {
            if (!convolutions.count(p)) { return VKDNN_STATUS_BAD_PARAM; }
            if (type != VKDNN_DEFAULT_MATH) { return VKDNN_STATUS_NOT_SUPPORTED; }
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnGetConvolution2dForwardOutputDim(vkdnnConvolutionDescriptor_t p, vkdnnTensorDescriptor_t x,
    vkdnnFilterDescriptor_t f, int* n, int* c, int* h, int* w)
{
    return guarded([&]
        {
            if (!n || !c || !h || !w) { return VKDNN_STATUS_BAD_PARAM; }
            int hh, ww;
            auto s = convolutionShape(p, x, f, hh, ww);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            *n = int(x->n);
            *c = int(f->k);
            *h = hh;
            *w = ww;
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnGetConvolutionForwardWorkspaceSize(vkdnnHandle_t handle, vkdnnTensorDescriptor_t x,
    vkdnnFilterDescriptor_t f, vkdnnConvolutionDescriptor_t p, vkdnnTensorDescriptor_t y,
    vkdnnConvolutionFwdAlgo_t algo, size_t* bytes)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (!bytes) { return VKDNN_STATUS_BAD_PARAM; }
            auto s = convolutionOutput(p, x, f, y);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (algo != VKDNN_CONVOLUTION_FWD_ALGO_DIRECT) { return VKDNN_STATUS_NOT_SUPPORTED; }
            int device;
            s = mapStatus(vkdnnGetDevice(&device));
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (device != handle->device) { return VKDNN_STATUS_BAD_PARAM; }
            *bytes = 0;
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnConvolutionForward(vkdnnHandle_t handle, const void* alpha, vkdnnTensorDescriptor_t x,
    const void* xp, vkdnnFilterDescriptor_t f, const void* fp, vkdnnConvolutionDescriptor_t p,
    vkdnnConvolutionFwdAlgo_t algo, void* workspace, size_t workspaceBytes, const void* beta,
    vkdnnTensorDescriptor_t y, void* yp)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            auto s = convolutionOutput(p, x, f, y);
            if (s != VKDNN_STATUS_SUCCESS) { return s; }
            if (algo != VKDNN_CONVOLUTION_FWD_ALGO_DIRECT) { return VKDNN_STATUS_NOT_SUPPORTED; }
            // DIRECT requires no workspace; the optional pointer/size are intentionally unused.
            (void)workspace;
            (void)workspaceBytes;
            ConvolutionParams q{};
            auto scalar = runtimeHostScalar(static_cast<const float*>(alpha), q.alpha);
            if (scalar != vkdnnSuccess) { return mapStatus(scalar); }
            scalar = runtimeHostScalar(static_cast<const float*>(beta), q.beta);
            if (scalar != vkdnnSuccess) { return mapStatus(scalar); }
            q.count = y->count;
            q.c = x->c;
            q.h = x->h;
            q.w = x->w;
            q.k = f->k;
            q.fc = f->c;
            q.kh = f->h;
            q.kw = f->w;
            q.oh = y->h;
            q.ow = y->w;
            q.ph = p->ph;
            q.pw = p->pw;
            q.sh = p->sh;
            q.sw = p->sw;
            q.dh = p->dh;
            q.dw = p->dw;
            q.groups = p->groups;
            q.mode = p->mode;
            size_t xb = 0, fb = 0;
            if (q.alpha != 0)
            {
                xb = size_t(x->count) * 4;
                fb = size_t(f->count) * 4;
            }
            return mapStatus(runtimeConvolution(handle->device, xp, xb, fp, fb, yp, size_t(y->count) * 4, q));
        });
}
vkdnnStatus_t vkdnnCreatePoolingDescriptor(vkdnnPoolingDescriptor_t* t)
{
    return guarded([&]
        {
            return create(t, pools);
        });
}
vkdnnStatus_t vkdnnDestroyPoolingDescriptor(vkdnnPoolingDescriptor_t t)
{
    return guarded([&]
        {
            return destroy(t, pools);
        });
}
vkdnnStatus_t vkdnnSetPooling2dDescriptor(vkdnnPoolingDescriptor_t t, vkdnnPoolingMode_t mode, vkdnnNanPropagation_t nan,
    int kh, int kw, int ph, int pw, int sh, int sw)
{
    return guarded([&]
        {
            if (!pools.count(t) || !nanOption(nan) || kh <= 0 || kw <= 0 || ph < 0 || pw < 0 || sh <= 0 || sw <= 0 || ph >= kh || pw >= kw)
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            if (mode != VKDNN_POOLING_MAX && mode != VKDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING && mode != VKDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            if (int64_t(kh) * kw > INT_MAX || int64_t(ph) * 2 > INT_MAX || int64_t(pw) * 2 > INT_MAX)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            *t = { true, mode, nan, uint32_t(kh), uint32_t(kw), uint32_t(ph), uint32_t(pw), uint32_t(sh), uint32_t(sw) };
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnGetPooling2dForwardOutputDim(vkdnnPoolingDescriptor_t p, vkdnnTensorDescriptor_t x, int* n, int* c, int* h, int* w)
{
    return guarded([&]
        {
            if (!pools.count(p) || !p->initialized || !tensor(x) || !n || !c || !h || !w)
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            int hh, ww;
            if (!outputDim(p, x, hh, ww))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            *n = int(x->n);
            *c = int(x->c);
            *h = hh;
            *w = ww;
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnPoolingForward(vkdnnHandle_t handle, vkdnnPoolingDescriptor_t pool, const void* alpha,
    vkdnnTensorDescriptor_t xd, const void* x, const void* beta, vkdnnTensorDescriptor_t yd, void* y)
{
    return guarded([&]
        {
            if (!pools.count(pool) || !pool->initialized || !tensor(xd) || !tensor(yd))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            int h, w;
            if (!outputDim(pool, xd, h, w) || yd->n != xd->n || yd->c != xd->c || yd->h != uint32_t(h) || yd->w != uint32_t(w))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ComputeParams p{};
            p.op = 1;
            p.mode = pool->mode;
            p.nan = pool->nan;
            p.kh = pool->kh;
            p.kw = pool->kw;
            p.ph = pool->ph;
            p.pw = pool->pw;
            p.sh = pool->sh;
            p.sw = pool->sw;
            return execute(handle, alpha, xd, x, beta, yd, y, p);
        });
}
vkdnnStatus_t vkdnnSoftmaxForward(vkdnnHandle_t handle, vkdnnSoftmaxAlgorithm_t algo, vkdnnSoftmaxMode_t mode,
    const void* alpha, vkdnnTensorDescriptor_t xd, const void* x, const void* beta, vkdnnTensorDescriptor_t yd, void* y)
{
    return guarded([&]
        {
            if (algo != VKDNN_SOFTMAX_FAST && algo != VKDNN_SOFTMAX_ACCURATE && algo != VKDNN_SOFTMAX_LOG)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            if (mode != VKDNN_SOFTMAX_MODE_INSTANCE && mode != VKDNN_SOFTMAX_MODE_CHANNEL)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            if (!tensor(xd) || !tensor(yd) || !equal(xd, yd))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ComputeParams p{};
            p.op = 2;
            if (algo == VKDNN_SOFTMAX_LOG) { p.op = 3; }
            p.mode = mode;
            return execute(handle, alpha, xd, x, beta, yd, y, p);
        });
}
vkdnnStatus_t vkdnnSetTensor(vkdnnHandle_t handle, vkdnnTensorDescriptor_t yd, void* y, const void* value)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (!tensor(yd)) { return VKDNN_STATUS_BAD_PARAM; }
            ComputeParams p{};
            p.op = 5;
            float zero = 0;
            return execute(handle, value, yd, nullptr, &zero, yd, y, p);
        });
}
vkdnnStatus_t vkdnnCreateOpTensorDescriptor(vkdnnOpTensorDescriptor_t* d)
{
    return guarded([&]
        {
            return create(d, tensorOps);
        });
}
vkdnnStatus_t vkdnnDestroyOpTensorDescriptor(vkdnnOpTensorDescriptor_t d)
{
    return guarded([&]
        {
            return destroy(d, tensorOps);
        });
}
vkdnnStatus_t vkdnnSetOpTensorDescriptor(vkdnnOpTensorDescriptor_t d, vkdnnOpTensorOp_t op,
    vkdnnDataType_t type, vkdnnNanPropagation_t nan)
{
    return guarded([&]
        {
            if (!tensorOps.count(d) || !nanOption(nan)) { return VKDNN_STATUS_BAD_PARAM; }
            if (type != VKDNN_DATA_FLOAT || (op != VKDNN_OP_TENSOR_ADD && op != VKDNN_OP_TENSOR_MUL))
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            *d = { true, op, nan };
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnGetOpTensorDescriptor(vkdnnOpTensorDescriptor_t d, vkdnnOpTensorOp_t* op,
    vkdnnDataType_t* type, vkdnnNanPropagation_t* nan)
{
    return guarded([&]
        {
            if (!tensorOps.count(d) || !d->initialized || !op || !type || !nan) { return VKDNN_STATUS_BAD_PARAM; }
            *op = d->op;
            *type = VKDNN_DATA_FLOAT;
            *nan = d->nan;
            return VKDNN_STATUS_SUCCESS;
        });
}
vkdnnStatus_t vkdnnOpTensor(vkdnnHandle_t handle, vkdnnOpTensorDescriptor_t d,
    const void* alpha1, vkdnnTensorDescriptor_t a, const void* ap,
    const void* alpha2, vkdnnTensorDescriptor_t b, const void* bp,
    const void* beta, vkdnnTensorDescriptor_t c, void* cp)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (!tensorOps.count(d) || !d->initialized || !tensor(a) || !tensor(b) || !tensor(c)
                || !equal(a, c) || (b->n != 1 && b->n != c->n) || (b->c != 1 && b->c != c->c)
                || (b->h != 1 && b->h != c->h) || (b->w != 1 && b->w != c->w))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ConvolutionParams p{};
            auto s = runtimeHostScalar(static_cast<const float*>(alpha1), p.gamma);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            s = runtimeHostScalar(static_cast<const float*>(alpha2), p.delta);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            s = runtimeHostScalar(static_cast<const float*>(beta), p.beta);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            p.alpha = 1;
            p.op = 5;
            p.mode = d->op;
            p.nan = d->nan;
            p.count = c->count;
            p.n = b->n;
            p.c = b->c;
            p.h = b->h;
            p.w = b->w;
            p.k = c->c;
            p.oh = c->h;
            p.ow = c->w;
            return mapStatus(runtimeConvolution(handle->device, ap, p.gamma == 0 ? 0 : size_t(a->count) * 4,
                bp, p.delta == 0 ? 0 : size_t(b->count) * 4, cp, size_t(c->count) * 4, p));
        });
}
vkdnnStatus_t vkdnnSoftmaxBackward(vkdnnHandle_t handle, vkdnnSoftmaxAlgorithm_t algo, vkdnnSoftmaxMode_t mode,
    const void* alpha, vkdnnTensorDescriptor_t yd, const void* y, vkdnnTensorDescriptor_t dyDesc, const void* dy,
    const void* beta, vkdnnTensorDescriptor_t dxDesc, void* dx)
{
    return guarded([&]
        {
            if (!handles.count(handle)) { return VKDNN_STATUS_NOT_INITIALIZED; }
            if (algo != VKDNN_SOFTMAX_FAST && algo != VKDNN_SOFTMAX_ACCURATE && algo != VKDNN_SOFTMAX_LOG)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            if (mode != VKDNN_SOFTMAX_MODE_INSTANCE && mode != VKDNN_SOFTMAX_MODE_CHANNEL)
            {
                return VKDNN_STATUS_NOT_SUPPORTED;
            }
            if (!tensor(yd) || !tensor(dyDesc) || !tensor(dxDesc) || !equal(yd, dyDesc) || !equal(yd, dxDesc))
            {
                return VKDNN_STATUS_BAD_PARAM;
            }
            ConvolutionParams p{};
            auto s = runtimeHostScalar(static_cast<const float*>(alpha), p.alpha);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            s = runtimeHostScalar(static_cast<const float*>(beta), p.beta);
            if (s != vkdnnSuccess) { return mapStatus(s); }
            p.op = 6;
            p.mode = mode;
            p.nan = algo;
            p.count = yd->count;
            p.c = yd->c;
            p.h = yd->h;
            p.w = yd->w;
            size_t bytes = p.alpha == 0 ? 0 : size_t(yd->count) * 4;
            return mapStatus(runtimeConvolution(handle->device, y, bytes, dy, bytes, dx, size_t(dxDesc->count) * 4, p));
        });
}
