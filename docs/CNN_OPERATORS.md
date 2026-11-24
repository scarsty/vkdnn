# 基本 CNN 算子与接口清单

## 范围和状态

目标是先支持类似 LeNet 的链式二维分类 CNN：卷积 → 偏置 → ReLU → 池化 → 全连接 → Softmax，同时预留这些算子的反向接口。本轮仅提供接口与实现桩，不能执行网络、创建有效描述符或计算梯度。

- DNN 接口见 [vkdnn_ops.h](../include/vkdnn/vkdnn_ops.h)，使用 `vkdnn` 前缀与独立的 `vkdnnStatus_t`。
- BLAS 接口见 [vkblas.h](../include/vkdnn/vkblas.h)，使用 `vkblas` 前缀与 `vkblasStatus_t`，避免 Create/Destroy/Handle 与 DNN 层冲突。
- 底层设备与内存仍由 [vkdnn.h](../include/vkdnn/vkdnn.h) 的 CUDA Runtime 风格接口提供，返回 `vkdnnError_t`。
- 所有新增桩，包括句柄/描述符管理、形状和 workspace 查询，均返回对应层的 `NOT_SUPPORTED`；不检查参数、不修改输出、不分配资源、不执行 GPU 工作。不存在“workspace 为零表示可用”等伪成功。
- 初步面向 FP32、二维张量/卷积。枚举只是参考接口子集，声明 DOUBLE/HALF、算法或布局并不代表支持。

## 最小前向算子集

| 网络用途 | 参考接口 | 本工程接口 | 关联描述符/参数 |
| --- | --- | --- | --- |
| 张量形状、布局与 dtype | cuDNN Create/Set/DestroyTensorDescriptor | `vkdnnCreateTensorDescriptor`、`vkdnnSetTensor4dDescriptor`、`vkdnnDestroyTensorDescriptor` | NCHW/NHWC、N/C/H/W |
| 卷积权重 | cuDNN FilterDescriptor | `vkdnnCreateFilterDescriptor`、`vkdnnSetFilter4dDescriptor`、`vkdnnDestroyFilterDescriptor` | K/C/H/W |
| 二维卷积 | cuDNN ConvolutionForward | `vkdnnConvolutionForward` | padding、stride、dilation、groups、alpha/beta、algorithm、workspace |
| 卷积输出形状/临时显存 | cuDNN GetConvolution2dForwardOutputDim / GetConvolutionForwardWorkspaceSize | 同名替换前缀 | 查询结果仍为桩，不写输出 |
| 偏置广播、残差加法 | cuDNN AddTensor | `vkdnnAddTensor` | bias 可用 1/C/1/1 描述，残差使用匹配的形状 |
| ReLU 等激活 | cuDNN ActivationForward | `vkdnnActivationForward` | activation descriptor，mode、NaN 规则、coef |
| 最大/平均池化 | cuDNN PoolingForward | `vkdnnPoolingForward` | pooling descriptor、窗口、padding、stride |
| 池化输出形状 | cuDNN GetPooling2dForwardOutputDim | `vkdnnGetPooling2dForwardOutputDim` | N/C/H/W 输出 |
| 全连接 | cuBLAS Sgemm | `vkblasSgemm` | 列主序、transa/transb、M/N/K、lda/ldb/ldc、alpha/beta |
| 分类概率/LogSoftmax | cuDNN SoftmaxForward | `vkdnnSoftmaxForward` | algorithm（含 LOG）、INSTANCE/CHANNEL mode |

计算句柄分别使用 `vkdnnCreate/Destroy`、`vkblasCreate/Destroy`。BLAS 保留 `vkblasSetPointerMode`，明确 alpha/beta 指针所在内存的接口形式。卷积、激活、池化的 Create/Set/Destroy 描述符接口均已列桩。

Flatten 对连续存储通常只是形状解释，不需要独立 GPU kernel；若涉及重排或非连续布局，则需后续布局转换接口，不能仅靠 reshape 冒充转换。全连接使用列主序 GEMM，不能沿用早期计划中的 row-major GEMM 假设；框架需通过矩阵视图及转置参数表达其布局。

## 训练所需反向算子

| 梯度 | 本轮桩接口 |
| --- | --- |
| 卷积输入梯度 | `vkdnnConvolutionBackwardData`、对应 workspace 查询 |
| 卷积权重梯度 | `vkdnnConvolutionBackwardFilter`、对应 workspace 查询 |
| bias 梯度 | `vkdnnConvolutionBackwardBias` |
| 激活输入梯度 | `vkdnnActivationBackward` |
| 池化输入梯度 | `vkdnnPoolingBackward` |
| Softmax 输入梯度 | `vkdnnSoftmaxBackward` |
| 全连接输入/权重梯度 | 复用 `vkblasSgemm`，调整转置和维度 |

分类交叉熵/损失归约、标签处理、初始输出梯度及参数更新仍需框架或后续扩展算子提供；SoftmaxBackward 本身不是交叉熵损失。当前集合不等于完整训练框架，也不能单独完成端到端训练。

## 后续扩展（本轮不声明接口）

| 能力 | 参考方向 | 用途 |
| --- | --- | --- |
| BatchNorm | cuDNN BatchNormalizationForwardInference/Training、Backward | 常见现代 CNN |
| Dropout | cuDNN DropoutForward/Backward、状态与 reserve space 管理 | 训练正则化 |
| 张量变换 | cuDNN TransformTensor | 布局/stride 转换 |
| reduction / elementwise | cuDNN ReduceTensor / OpTensor | 损失、统计与组合算子 |
| AXPY / scale | cuBLAS Saxpy / Sscal | 简单权重更新等基础向量操作 |
| batched GEMM | cuBLAS SgemmStridedBatched | 批处理矩阵工作负载 |

## 参考与兼容边界

- BLAS 参数形式核对本机 CUDA 13.2 的 `cublas_api.h`，使用常见公开名称 Sgemm，不暴露 cuBLAS 内部版本后缀 `_v2`。
- DNN 参数形式核对 NVIDIA 官方 [cuDNN CNN API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html) 与 [cuDNN Ops API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html)（2026-11-24 访问的 cuDNN 9.x 文档）。本机虽有 cuDNN 目录，但未找到可读开发头文件。
- 参考的是传统描述符式接口；其中若干 API 已被当前 cuDNN 标记 deprecated。本工程为清晰表达基本算子保留该风格，不声称实现 cuDNN backend graph API。
- 运行时已有 `vkdnnGetErrorString(vkdnnError_t)`，因此不新增同名但状态类型不同的 cuDNN 错误字符串函数；不要将 DNN/BLAS 状态强制转换为运行时错误码。
- 不引入 CUDA、cuBLAS、cuDNN 链接或运行依赖，不承诺 ABI/完整枚举/数值语义兼容。实现前仍须逐项明确布局、别名、NaN、线程安全和资源生命周期。
