# 基本 CNN 算子与接口清单

## 范围和状态

目标是先支持类似 LeNet 的链式二维分类 CNN：卷积 → 偏置 → 激活 → 池化 → 全连接 → Softmax，同时预留这些算子的反向接口。目前实现 FP32 SGEMM、直接卷积、AddTensor、ReLU/Sigmoid/tanh、Pooling、Softmax 的真实 Vulkan 计算及相应描述符；全部专用反向仍为桩，尚无完整网络示例、预训练参数或训练能力。

- DNN 接口见 [vkdnn_ops.h](../include/vkdnn/vkdnn_ops.h)，使用 `vkdnn` 前缀与独立的 `vkdnnStatus_t`。
- BLAS 接口见 [vkblas.h](../include/vkdnn/vkblas.h)，使用 `vkblas` 前缀与 `vkblasStatus_t`，避免 Create/Destroy/Handle 与 DNN 层冲突。
- 底层设备与内存仍由 [vkdnn.h](../include/vkdnn/vkdnn.h) 的 CUDA Runtime 风格接口提供，返回 `vkdnnError_t`。
- 剩余桩（全部专用反向）返回对应层 `NOT_SUPPORTED`，不检查参数、不修改输出、不分配资源、不执行 GPU 工作。
- 初步面向 FP32、二维张量/卷积。枚举只是参考接口子集，声明 DOUBLE/HALF、算法或布局并不代表支持。

## 最小前向算子集

### 已实现的精确范围

- `src/ops/forward.cpp` 与独立编写的 `forward.comp`：FP32 连续 NCHW、正 N/C/H/W，支持 batch。NHWC、HALF/DOUBLE 不支持；张量元素数上限为 `INT_MAX/4`，还受设备 storage-buffer/dispatch 限制。
- 激活支持 `RELU`/`SIGMOID`/`TANH`、coef=0；其他激活返回 `NOT_SUPPORTED`。ReLU 支持两种 NaN 选项：传播或将输入 NaN 视为零。Sigmoid 使用 `e=exp(-abs(x))`，非负输入取 `1/(1+e)`，负输入取 `e/(1+e)`，避免指数上溢；正/负无穷为 1/0。tanh 使用带符号的 `(1-exp(-2*min(abs(x),80)))/(1+exp(-2*min(abs(x),80)))`，小于 `1e-4` 的输入使用 x 避免消减，极大值/正负无穷为 ±1。Sigmoid/tanh 的 NaN 始终传播（NaN 选项只影响 ReLU）。alpha=0 不读取或校验输入地址，可传 NULL；beta=0 不读旧输出，二者均零写零。宿主 scalar 通过运行时桥接检查，设备 scalar 拒绝。
- `vkdnnAddTensor` 实现 `C=alpha*A+beta*C`，A 每轴维度必须等于 C 或 1，支持 1/C/1/1 bias、任意逐轴广播和同形残差。FP32 连续 NCHW，维度上限与 tensor setter 相同；失败 setter 保留旧设置。alpha=0 不访问/校验 A（可 NULL 或与 C 重叠），仍校验描述符/广播和输出；beta=0 不读旧 C。实际使用的 A 与 C 重叠返回 NOT_SUPPORTED，含原地；同分配非重叠范围允许。复用现有运行时桥的设备、offset、容量、storage-buffer/dispatch 上限检查；参数/能力检查失败不写输出。NaN 输入/非零系数对应的旧输出及 NaN scalar 传播，Inf 遵循 FP32 运算；不走 CPU 计算。
- 二维最大池化、平均池化含/不含 padding；不支持 MAX_DETERMINISTIC。正窗口/stride，非负 padding 且 padding 小于相应窗口；输出使用 `floor((input+2*pad-window)/stride)+1`，非正输出拒绝。边界 max 忽略 padding；average 的分母分别为完整窗口面积或有效输入数。非传播模式忽略 NaN（平均时按零贡献，仍计分母）；全 NaN 的 max 为负无穷。
- Softmax 仅 ACCURATE/LOG，INSTANCE 对每个 N 的 C/H/W 归约，CHANNEL 对每个 N/H/W 的 C 归约。减最大值后指数求和；LOG 直接用 `x-max-log(sum)`。FAST 不支持。组内 NaN、正无穷或全部负无穷使整组输出 NaN；有限值与负无穷混合可用。
- 所有前向执行 `y=alpha*op(x)+beta*y`，alpha/beta 指向宿主 float；beta=0 不读取旧 y（即使旧值 NaN）。支持已登记分配内 4 字节对齐 offset，校验容量/设备；拒绝重叠输入输出（含原地），允许同一分配内不重叠范围。不接受宿主或外部 Vulkan 指针。
- 句柄绑定创建时当前设备，调用时必须是同一当前设备；描述符跨设备纯宿主对象，可修改/复用，失败的 setter 保留旧状态。未初始化/空/未知描述符返回 BAD_PARAM。销毁后禁止复用，重复销毁报错；地址之后可能被分配器复用，不承诺识别任意陈旧指针。
- DNN 调用全局串行化，运行时分配查找与真实 compute 提交在运行时锁内完成，避免与 free/reset 竞争。调用同步完成，不持有跨调用 Vulkan 引用；DeviceReset 后旧设备指针失效，句柄可配合新分配继续用。调用方须保证宿主参数存活，不得并发修改宿主 alpha/beta。
- 不走 CPU 计算或中转；复用运行时 VkBuffer、queue、command pool/fence。shader 由 CMake 使用 Vulkan SDK `glslc` 构建为嵌入式 SPIR-V，部署只需动态库及 Vulkan loader。每次调用创建/销毁 pipeline/descriptor 资源，无缓存；softmax 每归约组单 invocation，正确性基线而非性能优化实现。

`tests/cnn_forward_test.cpp` 在实际 GPU 与 CPU 参考间对照，覆盖 batch、非整工作组长度、padding/stride、三种池化、两种 softmax 模式/算法、alpha/beta、beta=0 的 NaN 旧输出、NaN 规则、越界/对齐/重叠/宿主指针、非法/不支持参数、资源生命周期。容差为 `3e-5 * max(1,abs(reference))`。

偏置/tanh 本轮验证（2026-11-24）：同一测试新增 16 种逐轴广播（含多 batch、空间广播、1/C/1/1、同形残差）、零/非零 alpha/beta、NaN/Inf scalar/input、零输入访问、非重叠同分配与重叠拒绝、错误不写输出；tanh 与 double `std::tanh` 对照，覆盖正负/零/近零/极大有限值/Inf/NaN 和两种 NaN 选项。Windows / RTX 5090 的 Debug、Release 各六项 CTest（runtime memory、device query、CNN stub、CNN forward、GEMM、convolution）全部通过。使用 `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`、`VK_LAYER_VALIDATE_SYNC=1`，未见 validation/synchronization 错误；loader 日志确认 validation DLL 加载。单设备，跨设备 bias/tanh 分支明确跳过。IDE CMake build 无法配置且 diagnostics 为空，按授权改用 CLI build/CTest；两配置 DLL 的 AddTensor/ActivationForward/SetActivationDescriptor C 导出已核对。修改的 C/C++ 及 shader 均通过指定 clang-format dry-run/Werror（shader C++ 模式、RemoveSemicolon=false）。未做跨厂商、性能或完整模型验证。

2026-11-24 在 Windows / NVIDIA GeForce RTX 5090 上，Debug、Release 的四个测试程序全部通过，启用 `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` 和 `VK_LAYER_VALIDATE_SYNC=1`，无 validation 错误。CMake IDE build/CTest 工具均因无法配置项目失败（诊断为空），因此使用实际 CMake CLI 构建、直接执行测试程序；不是将工具失败当作测试通过。

| 网络用途 | 参考接口 | 本工程接口 | 关联描述符/参数 |
| --- | --- | --- | --- |
| 张量形状、布局与 dtype | cuDNN Create/Set/DestroyTensorDescriptor | `vkdnnCreateTensorDescriptor`、`vkdnnSetTensor4dDescriptor`、`vkdnnDestroyTensorDescriptor` | NCHW/NHWC、N/C/H/W |
| 卷积权重 | cuDNN FilterDescriptor | `vkdnnCreateFilterDescriptor`、`vkdnnSetFilter4dDescriptor`、`vkdnnDestroyFilterDescriptor` | K/C/H/W |
| 二维卷积 | cuDNN ConvolutionForward | `vkdnnConvolutionForward` | padding、stride、dilation、groups、alpha/beta、algorithm、workspace |
| 卷积输出形状/临时显存 | cuDNN GetConvolution2dForwardOutputDim / GetConvolutionForwardWorkspaceSize | 同名替换前缀 | 已实现；DIRECT workspace=0 |
| 偏置广播、残差加法 | cuDNN AddTensor | `vkdnnAddTensor` | bias 可用 1/C/1/1 描述，残差使用匹配的形状 |
| ReLU 等激活 | cuDNN ActivationForward | `vkdnnActivationForward` | activation descriptor，mode、NaN 规则、coef |
| 最大/平均池化 | cuDNN PoolingForward | `vkdnnPoolingForward` | pooling descriptor、窗口、padding、stride |
| 池化输出形状 | cuDNN GetPooling2dForwardOutputDim | `vkdnnGetPooling2dForwardOutputDim` | N/C/H/W 输出 |
| 全连接 | cuBLAS Sgemm | `vkblasSgemm` | 列主序、transa/transb、M/N/K、lda/ldb/ldc、alpha/beta |
| 分类概率/LogSoftmax | cuDNN SoftmaxForward | `vkdnnSoftmaxForward` | algorithm（含 LOG）、INSTANCE/CHANNEL mode |

计算句柄分别使用 `vkdnnCreate/Destroy`、`vkblasCreate/Destroy`（均已实现）。BLAS 的 `vkblasSetPointerMode` 当前只支持 HOST；filter/convolution/activation/pooling 描述符均已实现。

### 二维卷积与 Sigmoid 补充

- Filter 为连续 FP32 K/Cg/KH/KW；输入 N/C/H/W，输出 N/K/OH/OW。正 group count 必须整除 C 和 K，filter Cg=C/groups；支持普通、分组与 depthwise（含 multiplier）卷积。
- 正 stride/dilation、非负对称 padding，输出 `floor((input+2*pad-(dilation*(kernel-1)+1))/stride)+1`。有效核大于 padded 输入返回 BAD_PARAM；使用 64 位检查尺寸，元素数超过 INT_MAX/4、padded 坐标超过 INT_MAX 返回 NOT_SUPPORTED。失败不修改查询输出或已有描述符设置。CONVOLUTION 翻转核的两个空间轴，CROSS_CORRELATION 不翻转。
- `convolution.comp` 为独立编写的直接卷积，每 invocation 一个输出元素，FP32 顺序累加，不走 CPU、im2col 或 GEMM。新增 cuDNN 同名 `DIRECT=3` 枚举用于如实表达算法；保留 `IMPLICIT_GEMM=0` 但返回 NOT_SUPPORTED，不能把直接卷积冒称 implicit GEMM。DIRECT workspace 查询返回 0，执行时 optional workspace pointer/size 被忽略，不访问。
- `y=alpha*conv(x,w)+beta*y`，宿主 float scalar；alpha=0 不访问/校验 x 和 w，允许 NULL，仍校验形状及输出。beta=0 不读取旧 y。实际输入/权重须本设备登记、容量足够、4 字节对齐；允许 x/w 别名，拒绝 y 与实际输入/权重重叠，允许同分配不重叠区间。当前设备必须匹配句柄。32 位索引、Vulkan storage buffer range、push constant 与 dispatch 上限均检查，不支持时明确返回错误；无分块 dispatch 或性能承诺。
- 复用 GEMM 的三 buffer 运行时桥接/所有权与 alias 检查、通用 Vulkan pipeline/barrier/fence 提交。NaN 输入/权重（有效采样处）及 NaN scalar 传播；padding 不读取输入。两种 mode、矩形/batch/groups/depthwise、stride/dilation/padding、alpha/beta、零访问、NaN 旧输出、非法参数/容量/对齐/别名/生命周期由 `tests/convolution_test.cpp` 与 double CPU 参考验证（容差 `5e-5*max(1,abs(reference))`）；Sigmoid 在现有 forward 测试覆盖大正负值、0、NaN、Inf 和 alpha/beta。
- 算法参考：2026-11-24 阅读上游 [ncnn convolution Vulkan](https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/convolution_vulkan.cpp) 的有效核/stride/padding 与路径组织、[sigmoid shader](https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/shader/sigmoid.comp) 的指数激活思路；未复制、编译其源码或使用其内部类型/宏。签名核对本机 CUDA 13.2 所附 cuDNN 9.14 `cudnn_cnn.h`/`cudnn_ops.h`；不是 ABI 兼容实现。
- 2026-11-24 Windows / NVIDIA GeForce RTX 5090：Debug、Release 六项 CTest（runtime memory、device query、CNN stub、CNN forward/Sigmoid、GEMM、convolution）全部通过。启用 `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`、`VK_LAYER_VALIDATE_SYNC=1`、`VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`，无 validation/synchronization 错误；loader 日志确认 Khronos validation DLL 实际加载。单 GPU，跨设备测试明确跳过。IDE CMake build/CTest 无法配置（诊断为空），使用 CLI 构建与 CTest。C/C++ 及 `.comp` 按指定规则格式化并通过 dry-run/Werror；新增卷积/描述符与激活 C 符号已核对 DLL exports。未做跨厂商或性能验证。

### LeNet 现在还缺什么

- 经典 LeNet-5 常用 tanh/缩放 tanh；一些教学变体用 sigmoid，现代变体用 ReLU。Sigmoid/ReLU/tanh 已可用；缩放 tanh 可由调用方安排输入缩放及输出 alpha（尚无独立通用输入 scale 算子）。不将 sigmoid 冒充经典模型的等价替换。原版 C3 的非规则稀疏连接也不是普通 groups 的同义词，可由调用方构造带零权重的稠密核表达。
- 最大和平均池化（含/不含 padding）已可用；原版可训练下采样的系数/偏置不等于普通平均池化，仍缺对应逐通道算术。FC 可通过列主序 `vkblasSgemm` 完成；连续 NCHW 每个样本 flatten 只需矩阵视图，不需要复制；布局转换/非连续重排尚无接口，须正确安排权重及 GEMM 转置/leading dimension。
- 卷积、bias、sigmoid/ReLU/tanh、池化、FC、softmax 已有前向积木；网络示例、预训练参数、权重加载、图像归一化/尺寸处理、显存与布局衔接仍未实现。没有端到端模型准确率验证，不能将算子测试宣称为完整 LeNet 已跑通。
- 训练仍缺卷积/激活/池化等专用反向、损失/初始梯度、优化器/参数更新与自动微分；完整训练仍未实现。

### SGEMM 实现与限制

- `src/ops/gemm.cpp`、独立编写的 `gemm.comp` 实现列主序 `C=alpha*op(A)*op(B)+beta*C`。FP32 N/T/C（实数 C=T）、矩形及尾 tile、lda/ldb/ldc padding 均支持。16×16 工作组协作加载 shared-memory tiles；所有 invocation 参加 barrier，越界加载补零、越界输出不写。无 CPU 矩阵计算/中转。
- 宿主 scalar pointer mode 默认 HOST。DEVICE 返回 `NOT_SUPPORTED`，保留 HOST 状态；不是悄悄读取设备 scalar。k=0 不访问 alpha/A/B；alpha=0 不访问 A/B；beta=0 不读取旧 C。m/n=0 在校验枚举、维度、leading dimension、当前设备后成功，不要求任何指针。非空输出始终要求有效 C，包含 beta=1 情况。
- 负维度、非法枚举、过小 leading dimension、空/宿主矩阵、容量不足或非 4 字节对齐 offset 返回 `INVALID_VALUE`。容量按最后一个实际元素检查，不要求最后一列 padding 也在分配内。只接受运行时登记的本设备分配；句柄固定创建设备，调用时该设备必须为当前设备。
- 允许 A/B 互相别名和分配内 offset，拒绝 C 与实际使用的 A/B 范围重叠（`NOT_SUPPORTED`，范围包含列间 padding）；允许同一分配内不重叠范围。32 位 shader 索引、Vulkan storage-buffer range/dispatch 限制超出时返回 `NOT_SUPPORTED`。
- 复用运行时分配、队列、同步、shader 编译链及通用 descriptor/pipeline dispatch；每次调用创建 pipeline，无缓存，不承诺优化性能。调用同步、运行时全局锁保护提交与 free/reset；调用方须串行化同一个 BLAS 句柄的访问和销毁，保证宿主 scalar 存活，禁止复用已销毁句柄。DeviceReset 后须使用新分配，句柄可继续使用。没有 batch、异步或低精度支持。
- `tests/gemm_test.cpp` 使用 double 累加 CPU 参考，覆盖九种 N/T/C 组合、矩形/尾 tile、整 tile、padding 和 offset、alpha/beta、零访问、退化维度、非法参数、别名和 reset 后复用，容差 `4e-5*max(1,abs(reference))`。多设备归属测试仅在至少两设备时运行。
- 算法参考：2026-11-24 阅读 [Tencent/ncnn GEMM Vulkan 调度](https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/gemm_vulkan.cpp) 与 [GEMM shader](https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/shader/gemm.comp) 的分块/shared-memory/转置组织；本实现自行编写，未复制或编译其源码，不采用 ncnn packing、广播 C、类型或宏。接口签名核对本机 CUDA 13.2 `cublas_api.h`；列主序/leading dimension/零 beta 语义参考 [cuBLAS GEMM 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)。
- 2026-11-24 Windows / RTX 5090：Debug 和 Release 的 GEMM、CNN forward/stub、runtime memory、device query 五个程序均通过；启用 `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`、`VK_LAYER_VALIDATE_SYNC=1`，无 validation 错误。已从 loader 日志确认 validation DLL 实际加载；四个 BLAS C 符号经 DLL exports 验证。单设备环境跳过跨设备测试。IDE build/CTest 仍无法配置项目（诊断为空），改用 CMake CLI 构建及直接执行测试程序。C/C++ 和新 shader 均通过规定 clang-format 严格检查。

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
- DNN 参数形式核对本机 CUDA 13.2 include 中 cuDNN 9.14.0 的 `cudnn_ops.h`，及 NVIDIA 官方 [cuDNN CNN API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html) 与 [cuDNN Ops API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html)（2026-11-24）。
- 阅读本地 ncnn Vulkan ReLU 调度、Pooling padding 处理与 Softmax 的 max/exp/sum 归约组织作为算法参考；这里使用自行编写的单 shader 路径，无 ncnn 源码、内部类型或宏参与构建。
- 参考的是传统描述符式接口；其中若干 API 已被当前 cuDNN 标记 deprecated。本工程为清晰表达基本算子保留该风格，不声称实现 cuDNN backend graph API。
- 运行时已有 `vkdnnGetErrorString(vkdnnError_t)`，因此不新增同名但状态类型不同的 cuDNN 错误字符串函数；不要将 DNN/BLAS 状态强制转换为运行时错误码。
- 不引入 CUDA、cuBLAS、cuDNN 链接或运行依赖，不承诺 ABI/完整枚举/数值语义兼容。实现前仍须逐项明确布局、别名、NaN、线程安全和资源生命周期。
