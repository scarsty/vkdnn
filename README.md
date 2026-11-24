# vkdnn

基于 Vulkan Compute 的深度学习计算基础库，目标是提供类似 cuBLAS 的矩阵计算能力与类似 cuDNN 的神经网络算子能力，为推理和训练框架提供跨 GPU 厂商的底层计算后端。

> 已实现 Vulkan 1.2 运行时、FP32 列主序 tiled SGEMM，以及 FP32 连续 NCHW 的直接二维卷积（batch/groups/depthwise、padding/stride/dilation、卷积核翻转）、AddTensor 偏置/逐轴广播及残差加法、ReLU/Sigmoid/tanh、最大/平均池化、稳定 Softmax/LogSoftmax 真实 GPU 前向。卷积选择 `VKDNN_CONVOLUTION_FWD_ALGO_DIRECT`，workspace 为零；IMPLICIT_GEMM 保留但不支持。专用反向、网络示例、预训练参数与完整训练仍未实现，暂无性能数据。实现顺序和验收标准见 [开发计划](PLAN.md)。

## 项目定位

- **计算基础库，而非训练框架**：负责设备、内存、数据传输、矩阵计算和神经网络算子，不提供自动微分、优化器或模型导入。
- **Vulkan 原生实现**：使用 Vulkan Compute 与 SPIR-V，不依赖 CUDA；不承诺 CUDA API、ABI 或执行语义兼容。
- **正确性优先，性能逐步演进**：先建立 CPU 参考结果与可重复测试，再优化 shader、算子选择和执行开销。
- **通用路径与设备优化并存**：基线实现保证受支持设备可用；subgroup、低精度与 cooperative matrix 等能力必须经过检测才能启用。

首期聚焦单机、单 GPU、连续稠密张量和 FP32；先验证通用矩阵乘法，再扩展推理算子，最后补充训练所需的反向计算。

## 能力路线

基本 CNN 的前向/反向算子与参考接口见 [CNN 算子清单及 LeNet 缺口](docs/CNN_OPERATORS.md)。[DNN 头文件](include/vkdnn/vkdnn_ops.h) 中句柄、tensor/filter/convolution/activation/pooling 描述符及卷积、AddTensor、激活、池化、softmax 前向已实现，支持 batch、alpha/beta，确切 NaN/布局/别名限制见该文档；其余 DNN 仍返回 `NOT_SUPPORTED`。[BLAS](include/vkdnn/vkblas.h) 已实现同步 FP32 SGEMM、N/T/C、padded leading dimension 与宿主 alpha/beta；设备 scalar pointer mode 暂不支持。

已在 [third_party/ncnn](third_party/ncnn/README.md) 保存固定版本参考快照，保留上游 BSD-3-Clause 许可；不参与构建。`src/ops/forward.comp` 和宿主调度为独立实现，构建期使用 Vulkan SDK `glslc` 生成嵌入式 SPIR-V，运行时无需 shader 文件或编译器。

以下均为规划内容，具体支持范围以未来发布版本为准。

| 层级 | 初期目标 | 后续扩展 |
| --- | --- | --- |
| 运行时 | 设备枚举、能力查询、上下文、buffer、上传与下载、命令提交、同步、错误报告 | 内存池、pipeline 缓存、异步执行与多队列评估 |
| 矩阵计算 | FP32 GEMM，支持转置、leading dimension、alpha/beta | batched/strided-batched GEMM、向量运算、低精度矩阵乘法 |
| 张量基础 | shape、stride、layout 描述，连续布局校验 | 受限广播、布局转换、非连续布局的明确支持范围 |
| 推理算子 | elementwise、activation、reduction、softmax、conv2d、pooling | normalization、融合算子、更多卷积算法 |
| 训练基础 | 在前向算子稳定后提供反向计算 | 卷积梯度、归一化梯度及混合精度支持 |

不在首期范围：多 GPU、分布式通信、稀疏计算、量化部署、模型格式解析，以及对 cuDNN/cuBLAS 的逐 API 复刻。

## 技术方向

- **宿主端**：C++17 实现，规划以 C ABI 暴露不透明句柄和显式状态码，便于其他语言或框架绑定。
- **GPU 端**：以 GLSL compute shader 编译到 SPIR-V 为起点；优化路径不得成为通用路径的隐式依赖。
- **构建**：采用 CMake，构建时编译 shader，运行时不要求部署 shader 编译器。
- **Vulkan 基线**：暂定 Vulkan 1.2 Compute，具体 feature/limit 要求在最小运行时阶段固化；不要求 graphics queue 或显示窗口。
- **平台**：先在 Windows 完成开发闭环，再验证 Linux；其他平台在完成实际设备验证前不承诺支持。
- **精度**：FP32 优先；FP16/BF16 与矩阵硬件加速作为能力驱动的可选路径，分别验证存储、运算及累加精度。

## 接口原则

1. 基础运行时采用 CUDA 风格的当前设备和隐式上下文，明确设备指针生命周期与 reset 前的同步要求。
2. 内存分配、数据传输和计算分离，避免每次算子调用隐式复制数据。
3. 显式表达布局、数据类型与执行参数；不支持的组合返回错误，不静默改变计算语义。
4. 首版采用同步执行建立闭环；后续异步接口必须明确提交、完成和资源存活边界，不改变已有同步接口的语义。
5. 首版 buffer 使用库自身分配；外部 Vulkan device、buffer 与命令流互操作待运行时稳定后独立设计。
6. 算子参数错误、设备能力不足、内存不足及设备丢失均需向调用者报告。

## 质量与性能

- 使用 CPU 参考实现验证数值结果，覆盖边界尺寸、非对齐尺寸及非法输入。
- 按算子和 dtype 规定绝对误差与相对误差，明确 NaN/Inf 和 reduction 的数值行为。
- 在 Vulkan validation layers 下检查资源、同步与生命周期使用。
- 性能测试区分端到端延迟与 GPU 执行耗时，记录预热、重复次数、设备、驱动与构建配置。
- 矩阵与卷积性能按不同形状评估，不以单个尺寸代表整体表现，也不预先承诺达到 cuBLAS/cuDNN 的性能。

## 构建与使用

### Visual Studio 原生工程

直接打开根目录的 [vkdnn.sln](vkdnn.sln)，工程文件位于 [src/vkdnn.vcxproj](src/vkdnn.vcxproj)。支持 Debug/Release x64，当前工具集为 Visual Studio 2026 的 v145；需要安装 C++ 桌面开发工具、Windows SDK 和 Vulkan SDK，并设置 `VULKAN_SDK`。工程直接编译动态库和三个自有 compute shader，不依赖 CMake 生成工程，不编译 ncnn 参考代码。

输出位于 `build/vs/x64/Debug` 或 `build/vs/x64/Release`，包含 `vkdnn.dll` 和导入库 `vkdnn.lib`。原生解决方案目前仅包含库目标；完整测试继续使用下述 CMake 工程。新增源码或 shader 时须同步维护两套工程列表。

### CMake 构建与测试

构建需要 CMake 3.20+、C11 和 C++17 编译器，以及 Vulkan SDK 提供的头文件、加载器链接库和 `glslc`。CMake 使用 `find_package(Vulkan REQUIRED)` 和 `Vulkan::Vulkan`，不再自动下载头文件。Windows 可通过 `VULKAN_SDK` 指向 SDK 根目录，也可用 `-DVKDNN_GLSLC=绝对路径` 指定 shader 编译器；安装后可能需要重启终端以刷新环境变量。运行时直接依赖 `vulkan-1.dll`（Linux 为 Vulkan loader 共享库）和兼容驱动，不再使用 `LoadLibrary`/`dlopen` 手动加载；缺少 DLL 时系统会拒绝加载本库，而不是返回状态码。使用 CMake 构建并通过 CTest 验证：

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

公共头文件为 [include/vkdnn/vkdnn.h](include/vkdnn/vkdnn.h)，运行时实现位于 [src/core/vkdnn.cpp](src/core/vkdnn.cpp)，测试位于 [tests/](tests/)。可用 `-DBUILD_TESTING=OFF` 关闭测试。当前提供动态库，尚无安装导出。Windows 多配置构建生成 `build/Debug/vkdnn.dll`（Release 对应 `build/Release/vkdnn.dll`）及链接时使用的导入库 `vkdnn.lib`；运行时需将 DLL 放在应用程序目录或系统 DLL 搜索路径中，测试程序与 DLL 输出到同一目录。公共接口使用 `VKDNN_API` 声明导入/导出，调用方不应定义库内部的 `VKDNN_BUILDING_LIBRARY`。设备测试打印实际设备信息；缺少兼容加载器/驱动或没有合格设备时标记为跳过，不作为 GPU 测试通过。

### 运行时接口与实现状态

| 接口 | 用途 |
| --- | --- |
| `vkdnnGetDeviceCount` / `vkdnnGetDeviceProperties` | 枚举计算设备，查询名称、类型、显存和计算限制（已实现） |
| `vkdnnSetDevice` / `vkdnnGetDevice` / `vkdnnDeviceReset` | 选择、查询当前设备，释放当前设备全部运行时资源 |
| `vkdnnMalloc` / `vkdnnFree` | 分配、释放上下文所属的显存 |
| `vkdnnMemcpy(dst, src, count, kind)` | 按方向枚举同步复制；支持 HostToHost、HostToDevice、DeviceToHost、DeviceToDevice、Default |
| `vkdnnMemset` / `vkdnnDeviceSynchronize` | 按字节填充、等待设备执行完成 |
| `vkdnnGetErrorString` | 获取状态码对应的静态错误字符串（已实现） |

基础接口已按本机 CUDA 13.2 Runtime 头文件核对，采用 `vkdnnError_t`、`vkdnnSuccess`、`vkdnnErrorInvalidValue` 等命名，以及 `vkdnnDeviceProp`。这是不兼容旧接口的调整，不再公开上下文/内存句柄或 HtoD/DtoH/DtoD 函数；错误码数值、设备属性字段和 ABI 不承诺与 CUDA 兼容。Vulkan 特有属性保留，不虚构 CUDA 的 SM 等硬件参数。

`vkdnnMalloc(void**, size_t)` 返回保留的不可访问主机虚拟地址，背后对应 Vulkan buffer；支持字节指针偏移，但不能由 CPU 解引用，也不是 Vulkan device address。`vkdnnMemcpyDefault` 按库登记的地址范围推断方向，不代表支持 CUDA managed memory/UVA。显式方向与指针不匹配时返回错误；跨设备复制暂不支持。`vkdnnFree(NULL)` 成功，零大小 malloc 返回 NULL，零长度复制不执行操作。

当前设备为线程局部状态，默认设备 0；设备上下文按需创建，由进程内线程共享。运行时调用内部串行化。`vkdnnFree` 按指针所属设备释放，`vkdnnDeviceReset()` 释放当前设备的所有分配和上下文；调用方必须协调其他线程，并在卸载库前 reset 所有使用过的设备。

设备枚举筛选 Vulkan 1.2 且有非空 compute queue 的设备，允许软件计算设备。首次成功查询后缓存属性列表，索引在进程内稳定；查询结束即销毁临时 Vulkan 实例。查询失败不修改输出，驱动不兼容时返回 `vkdnnErrorNotSupported`。显存信息为 device-local heap 总容量，并非剩余可用显存。

表中运行时接口均已实现。显存使用 device-local buffer，主机传输使用 host-visible staging buffer，处理非 coherent 内存的 flush/invalidate。支持任意字节偏移和长度；4 字节对齐的设备内复制直接执行 GPU transfer，非对齐路径暂经主机中转，上传非对齐边界时保留相邻字节。当前 memset 同样经主机 staging 实现，尚未进行性能优化。

复制调用同步完成，校验方向、容量和范围，重叠复制返回错误。设备地址释放或 reset 后不可继续使用。详细契约见头文件注释。测试覆盖五种复制方向、指针偏移、边界保护、错误参数、重复分配释放及 reset 后重新创建。已有 shader 构建规则；跨平台、跨设备支持清单尚未落地。

```c
void* device = NULL;
vkdnnError_t error = vkdnnSetDevice(0);
if (error == vkdnnSuccess) error = vkdnnMalloc(&device, sizeof(float) * 256);
if (error == vkdnnSuccess) error = vkdnnMemset(device, 0, sizeof(float) * 256);
if (device != NULL) {
    vkdnnError_t cleanup = vkdnnFree(device);
    if (error == vkdnnSuccess) error = cleanup;
}
vkdnnError_t cleanup = vkdnnDeviceReset();
if (error == vkdnnSuccess) error = cleanup;
/* Check error with vkdnnGetErrorString(error). */
```

## 参与开发

开发顺序、目录组织、里程碑和待定决策见 [PLAN.md](PLAN.md)。新增算子应同时提供参数约束、CPU 参考测试及代表性 benchmark；优化必须保留可验证的通用实现或明确报告不支持。

API 风格与验证要求见 [开发标准](DEVELOPMENT.md)，智能体协作约定见 [AGENTS.md](AGENTS.md)。基础运行时、矩阵计算和神经网络算子分别向 CUDA Runtime、cuBLAS 和 cuDNN 的命名与调用形式靠拢；必要偏离需明确说明。

## 许可证

许可证尚未确定，仓库目前未授予开源许可证权限。对外发布前需确定许可证并补充 LICENSE 文件，同时核查依赖与参考实现的许可要求。
