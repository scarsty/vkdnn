# 开发标准

## API 风格

本工程的公开 API 以 **CUDA Runtime、cuBLAS 和 cuDNN** 为形式参考。目标是让熟悉这些库的使用者能够直观迁移调用方式，而不是设计一套无关的新风格。

| 模块 | 参考对象 | 应保持接近的形式 |
| --- | --- | --- |
| 基础运行时 | CUDA Runtime | 当前设备、内存分配/释放、复制方向枚举、内存填充、同步、设备重置与错误查询 |
| 矩阵与 BLAS | cuBLAS | 计算句柄、运算/转置枚举、维度、leading dimension、alpha/beta、数据类型和批处理参数 |
| 神经网络算子 | cuDNN | 计算句柄、张量/算子描述符、前向/反向接口、算法及 workspace 查询与传入方式 |

### 命名与签名

1. 有对应接口时，优先沿用其去掉原库前缀后的函数名、大小写、参数顺序、参数类型形态和枚举形式，仅替换为本工程前缀。基础运行时现用 `vkdnn` 前缀，例如 `cudaMemcpy` 对应 `vkdnnMemcpy`。
2. 基础运行时使用 `vkdnnError_t` 返回状态，成功为 `vkdnnSuccess`，错误使用 `vkdnnError...` 命名；复制方向使用 `vkdnnMemcpyKind`，不另建按方向拆分的公开复制函数。
3. 保持 CUDA 风格的 `vkdnnMalloc(void**, size_t)`、`vkdnnFree(void*)`、`vkdnnMemcpy(void*, const void*, size_t, enum vkdnnMemcpyKind)`、`vkdnnMemset(void*, int, size_t)`。不向这些接口增加 Vulkan 对象、显式上下文或独立 offset 参数。
4. 运行时以当前设备与隐式上下文管理资源；矩阵和 DNN 层按各自参考库使用显式计算句柄/描述符，不将运行时的隐式风格强套到所有模块。
5. 矩阵层统一使用 `vkblas` / `VKBLAS_` 前缀和 `vkblasStatus_t`，DNN 层使用 `vkdnn` / `VKDNN_` 前缀和 `vkdnnStatus_t`；运行时保留 `vkdnnError_t`。三个状态类型不能混用，运行时错误字符串函数不接收 DNN/BLAS 状态。
6. 公开头文件保持 C 兼容并提供 C ABI，C++ 类型与 Vulkan 实现细节留在内部。资源所有权、线程安全、同步和错误语义必须有明确契约。

### 参考依据与兼容边界

- 开发接口前先核对本机 SDK 头文件或对应官方文档，确认函数签名、枚举和相关语义；记录影响实现的参考版本。运行时现已参考本机 CUDA 13.2。
- 参考的是 API 形式与可合理实现的语义，不引入 CUDA/cuBLAS/cuDNN 作为 Vulkan 实现的运行依赖，不复制其实现源码。
- ncnn 等开源项目仅用于理解算法、优化策略与 Vulkan 使用方式；本工程自行编写 shader、调度和运行时实现，不将其源码直接加入编译，也不通过引入其内部类型、宏或框架包装成自己的实现。本地参考快照不属于功能实现；若实际复用受许可保护的代码，仍须如实注明来源并遵守许可，不能仅改名后声称原创。
- 不承诺二进制 ABI、错误码数值、结构体布局或全部功能兼容。不得将“风格接近”描述为“直接兼容”。
- Vulkan 无法等价实现的功能应明确说明限制或返回不支持，不静默降级为不同语义。必要签名偏离必须在变更说明和文档中解释。
- 设备属性如实报告 Vulkan 能力，不虚构 CUDA SM、compute capability 等参数。返回的设备指针仅用于显存标识与字节偏移，不可由 CPU 解引用；`MemcpyDefault` 不意味着支持 CUDA managed memory。
- 未实现的接口不得返回伪成功；实际支持范围以公开头文件、README 和测试为准，不为模仿参考库而一次性铺开大量无实现声明。

## 变更与验证

- 优先做最小、完整的变更；公开 API 调整应同步更新实现、全部直接调用方、测试及文档。
- 使用现有 CMake 构建与测试，验证 C 头文件调用、动态库导出和真实 GPU 行为；无可用设备时明确标记 GPU 测试跳过。
- Vulkan 资源与同步改动应在 validation layers 下验证；错误参数、边界范围和资源生命周期需有针对性测试。
- 修改 C/C++ 头文件或源文件、CUDA 文件后，仅对本次修改文件执行 clang-format；本机规则为 `C:\project\_clang-format`，并用 `--dry-run --Werror` 校验。保持原编码与 BOM，不手工覆盖格式化结果。
- 本工程的 GLSL `.comp` 文件同样纳入 clang-format，按 C++ 语言模式及相同规则整理。可通过标准输入配合 `--assume-filename=shader.cpp` 强制采用 C++ 格式，再写回原文件；校验时使用同一语言模式和 `--dry-run --Werror`。保持 GLSL 语义、编码与 BOM，格式化后重新编译 shader 并运行相关测试；第三方只读参考快照不批量整理。
- GLSL 格式化必须覆盖 `RemoveSemicolon: false`（其余规则继承本机规则文件）：C++ 的分号移除会破坏 GLSL interface block 声明。使用 `--style="{BasedOnStyle: InheritParentConfig, RemoveSemicolon: false}"` 并将 `--assume-filename` 指向项目内的 `.cpp` 路径，使其继承 `C:\project\_clang-format`；格式化与严格校验使用相同参数。
- C++ 禁止嵌套或连续串联三元条件表达式；多分支使用 `if/else` 或 `switch`。
- 不静默吞掉错误，不更改与任务无关的行为，不覆盖他人的工作区变更。
