# BLAS 支持范围

`vkblasSgemm` 与 `vkblasSscal` 使用同步 FP32 Vulkan compute；句柄绑定创建时的当前设备，标量仅支持 HOST pointer mode，无 CUDA 运行依赖，不承诺 cuBLAS ABI 兼容。

## vkblasSscal

签名核对本机 CUDA 13.2 `cublas_api.h` 的 `cublasSscal_v2`：

```c
vkblasStatus_t vkblasSscal(vkblasHandle_t handle, int n,
                         const float* alpha, float* x, int incx);
```

- 原地执行 `x[i * incx] = *alpha * x[i * incx]`，`0 <= i < n`，不改动间隔元素。
- 依照 BLAS SCAL 快速返回约定，`n <= 0` 或 `incx <= 0` 成功且不访问 alpha/x；仍检查句柄与当前设备。负步长不是反向遍历。
- 非空调用要求宿主 alpha 与当前设备运行时分配的 x，支持 4 字节对齐子视图。完整跨度 `(n-1)*incx+1` 必须落在同一分配内；末尾不要求额外 stride padding。alpha 为 1 也不跳过范围验证。
- alpha 为零仍读取并乘以原始 x，不作为 memset：NaN/Inf 参与 IEEE 浮点乘法。非正规数与精确浮点行为取决于 Vulkan 设备，不保证跨设备逐位一致。
- 复用 GEMM 的 registry、device/range validation、descriptor、同步与 shader 构建路径，不增加构建目标。shader 用独立分支处理向量，二维 dispatch 覆盖较长向量。
- 超过 32 位 shader 索引、设备 storage-buffer 或 dispatch 上限返回 `NOT_SUPPORTED`；不支持设备标量、64 位整数接口或异步调用。当前复用 16×16 工作组与 GEMM shared-memory 能力门槛，未作性能优化。
- 参数及地址错误返回 `INVALID_VALUE`，空句柄返回 `NOT_INITIALIZED`；释放后的地址不可复用。句柄生命周期与串行化契约沿用 SGEMM。

参考：[cuBLAS SCAL](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-scal) 与其链接的 Netlib SCAL。
`tests/gemm_test.cpp` 验证 GPU 数值、正步长、子视图/保护元素、跨工作组及二维 dispatch、空操作、错误参数、释放后地址和 reset 后重建，并保留 SGEMM 回归。
