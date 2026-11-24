# vkdnn 开发计划

## 1. 总体目标

构建基于 Vulkan 的深度学习计算基础库，提供矩阵计算和神经网络算子两类能力，共享设备管理、内存、执行、shader 与测试基础设施。

采用“小闭环、可验收、逐步优化”的路线：**运行时 → GEMM → 推理算子 → 性能优化 → 反向算子**。先做可靠的通用实现，再按真实工作负载投入设备专用优化。

本计划为初始设计；里程碑按交付物推进，不在缺少开发投入与设备信息时承诺日历工期。

当前已落地公共运行时、CMake 动态库、设备/显存测试、FP32 列主序 tiled SGEMM，以及 FP32 NCHW 直接卷积、AddTensor 偏置/逐轴广播及残差加法、ReLU/Sigmoid/tanh、Pooling、ACCURATE/LOG Softmax 前向与 CPU 参考对照测试。卷积覆盖 batch/groups/depthwise、矩形核、padding/stride/dilation 与核翻转，DIRECT 零 workspace，IMPLICIT_GEMM 不支持。M1 已实现 Vulkan 1.2 设备查询、上下文、device-local 显存分配/释放、staging 上传/下载、同步设备内复制、字节填充与同步；非对齐设备复制和 memset 暂经主机中转。已有 GLSL → 嵌入式 SPIR-V 构建链和真实 compute 闭环；GEMM 的 batch/设备 scalar 模式、vector add、反向、性能优化、完整许可证与 CI 工作仍待完成。LeNet 的 bias、tanh 已完成，但网络示例、预训练参数、权重加载/预处理及布局衔接仍待实现，完整训练另缺专用反向、损失和优化器；精确支持范围见 docs/CNN_OPERATORS.md。

## 2. 首期范围与约束

激活反向已落地：FP32 连续 NCHW/batch 的 ReLU、Sigmoid、tanh，独立 Vulkan compute shader；CPU 参考及实际 GPU 前向目标有限差分验收见 `tests/activation_backward_test.cpp`。这是 M5 的第一项增量，其余卷积/bias、池化、Softmax 专用反向仍为桩；不代表训练闭环已完成。

- 单设备、单计算队列，不引入多 GPU 或分布式通信。
- 宿主实现采用 C++17，对外采用 C ABI，基础运行时形式向 CUDA Runtime 靠拢（当前设备、隐式上下文、void* 显存地址及复制方向枚举）；公开 API 在试验阶段允许调整，发布前再制定兼容策略。
- 暂以 Vulkan 1.2 为基线；M0 核实目标设备与必要 feature/limit，M1 将要求固化为可查询、可诊断的契约。
- 首批计算采用 FP32、连续稠密存储；矩阵接口单独支持转置与 leading dimension，不能将二者等同于任意张量 stride 支持。
- 首版同步执行，不默认引入后台线程或异步资源回收。
- 推理先于训练；不实现自动微分、模型加载和优化器。
- 所有可选扩展均需能力检测；设备不满足基线时明确拒绝创建上下文。

## 3. 分层设计

| 层 | 职责 | 边界 |
| --- | --- | --- |
| 公共接口 | 状态码、不透明句柄、描述符、算子入口 | 不泄漏内部 C++ 类型；错误信息应可获取 |
| Core / Runtime | instance、device、queue、buffer、命令、同步、生命周期 | 显式管理资源；先单队列同步执行 |
| Tensor / Matrix 描述 | dtype、维度、shape、stride、layout、矩阵 leading dimension | 统一参数校验、范围检查和容量计算 |
| 算子层 | GEMM、activation、reduction、卷积等参数与实现选择 | 不重复实现运行时；公开说明支持组合 |
| Kernel 层 | SPIR-V shader、workgroup 配置、specialization constants | 保留通用 kernel，可选优化按能力选择 |
| 验证层 | CPU 参考、测试、benchmark、示例 | 正确性与性能分开衡量 |

### 关键执行约定

- 首期 buffer 由库分配并持有，不接受任意外部设备地址。公开 void* 地址仅用于分配标识和字节偏移，不可在 CPU 解引用；内部映射到 Vulkan buffer。当前设备线程局部，设备上下文跨线程共享，调用内部串行化，设备资源通过 DeviceReset 显式清理。
- 区分 host-visible 与 device-local 内存；上传、下载和计算之间建立正确的内存可见性与执行依赖。
- 处理非 coherent 内存的 flush/invalidate 与对齐要求，不假设所有设备都是独立显卡或统一内存。
- 明确禁止或支持输入输出别名，逐算子写入契约；不能依赖偶然有效的原地执行。
- 统一检查尺寸乘法溢出、offset、buffer 容量、dispatch 上限和 shader 索引范围。
- 异步能力后续新增独立接口，先明确提交失败、完成通知、资源占用与设备丢失行为。

## 4. 建议目录

以下为目标代码布局，仅公共头文件、core 实现桩、最小 CMake 构建和 tests 已建立，其余目录按实施需要增加。

```text
vkdnn/
  CMakeLists.txt
  cmake/               构建辅助与 shader 编译规则
  include/vkdnn/       公共 C 接口
  src/
    core/              Vulkan 运行时、内存与执行
    ops/               矩阵与神经网络算子
    kernels/           kernel 注册、pipeline 与选择逻辑
  shaders/             GLSL compute shader
  tests/               CPU 参考和正确性测试
  benchmarks/          性能测试
  examples/            最小调用示例
  docs/                API、数值约定和设备支持文档
```

初期保持单一库目标，避免为尚未实现的功能拆出多个子库；若接口和用户需求稳定，再评估矩阵与 DNN 模块的独立分发。

## 5. 里程碑与验收

### M0：工程与接口契约

**交付物**

- 确定首批测试设备、驱动、Vulkan 能力要求及开发工具最低版本。
- 建立 CMake、shader 构建规则和最小测试入口。
- 定义状态码、句柄所有权、数据类型、矩阵布局、线程安全边界与错误信息获取方式。
- 确定许可证，检查依赖许可；非必要依赖不引入。
- 建立不依赖 GPU 的构建与测试 CI；GPU 测试采用实际可用的设备运行环境。

**验收**：全新检出后能按 README 构建；shader 编译失败会导致构建失败；CPU 侧参数验证测试可以运行；无 Vulkan 设备时清楚区分跳过 GPU 测试和测试通过。

### M1：最小 Vulkan 运行时

**交付物**

- 设备枚举、显式设备选择、能力查询、上下文创建与销毁。
- buffer 分配、释放、上传、下载和 staging 路径。
- compute pipeline、descriptor、命令提交、barrier 与 fence。
- 一个简单的 vector add 示例，验证完整数据路径。

**验收**：上传 → dispatch → 下载的结果与 CPU 一致；覆盖零长度策略、非 workgroup 整数倍长度、重复调用、分配失败与非法容量；validation layers 无错误；资源可重复创建与销毁，无未释放 Vulkan 对象。

### M2：可用的 FP32 GEMM

**交付物**

- 实现矩阵运算 $C = \alpha\,\operatorname{op}(A)\operatorname{op}(B) + \beta C$。
- 首版采用 cuBLAS 风格列主序，支持 A/B 转置及合法 leading dimension；明确 M/N/K 为零时的行为。
- 先提供简单通用 kernel 和 CPU 参考，再实现使用 shared memory 的 tiled kernel。
- 为 alpha/beta 为零的情况定义访问语义，特别是 beta 为零时不读取旧 C。
- 建立 GEMM benchmark，覆盖小矩阵、方阵、长宽比差异和非 tile 对齐尺寸。

**验收**：转置组合、带 padding 的 leading dimension、不同 alpha/beta、K 为零及边界尺寸均通过参考对比；非法参数和不足 buffer 容量返回明确错误；发布可重复的基线性能记录，不预设未经测量的加速倍数。

### M3：基础张量与推理算子

**交付物**

- 张量描述符和连续布局验证；第一批以 NCHW 为卷积布局。
- elementwise add/mul、ReLU、基础 reduction 与数值稳定的 softmax。
- conv2d 前向：先支持明确子集，再扩展 stride、padding、dilation 与 groups；每项能力单独验收。
- 优先实现直接卷积作为参考 GPU 路径；评估 im2col + GEMM 时明确临时内存开销。
- max/average pooling，明确 padding 计数、边界窗口和输出尺寸规则。

**验收**：每个算子公开输入、输出、布局、别名规则和限制；随机与边界用例通过 CPU 对比；softmax 覆盖大幅值输入；卷积覆盖非方形输入和边缘窗口；无效输出形状、非法 axis 或不支持的布局不得静默执行。

### M4：执行与性能优化

**交付物**

- 先用 profiling 定位瓶颈，再引入 pipeline 缓存、descriptor 复用、内存复用和异步接口。
- 为 GEMM/卷积评估 tile、向量化、subgroup 和 cooperative matrix 优化。
- 引入按设备能力、dtype 和形状选择实现的机制，保留通用实现。
- 逐项评估 FP16/BF16；不能由存储支持推断运算或矩阵指令支持。
- 根据测量收益增加 bias + activation 等有限融合，不在首期建立通用图编译器。

**验收**：优化前后执行同一正确性测试集；按固定设备与形状报告收益及退化；异步测试验证等待、复用和销毁边界；timestamp 不支持时明确标记 GPU 时间不可用，不用 CPU 耗时冒充；缓存失效和能力不足行为可验证。

### M5：训练基础与可发布版本

**交付物**

- 为已稳定的前向算子补充反向计算，优先 activation、softmax、GEMM 和 conv2d。
- 评估 normalization 前向/反向、混合精度累加与确定性执行模式。
- 完善 API 文档、安装导出、版本策略和支持矩阵。
- 在 Windows/Linux 及可获得的不同 GPU 厂商设备上验证通用路径。

**验收**：反向算子同时通过 CPU 参考和小规模有限差分检查；发布声明仅覆盖实际测试的平台、驱动和精度；示例可在安装后的库上独立构建运行。

## 6. 测试与性能方法

### 正确性

- CPU 参考尽量使用更高精度累加；逐算子定义绝对/相对误差，不能仅用统一阈值掩盖 reduction 误差。
- 测试覆盖随机输入、全零、正负值、非对齐尺寸、退化维度、非法参数及资源生命周期。
- 对 NaN、Inf、空张量、整型溢出、别名及确定性给出明确契约，并验证已承诺的行为。
- CPU 参数测试、GPU 数值测试、validation layers 检查分别报告；缺少 GPU 的 CI 不作为 GPU 正确性证明。

### 性能

- 分别测量带数据传输的端到端耗时与纯 GPU kernel 耗时。
- 预热后重复运行，记录样本数、中位数和波动；首次 pipeline 编译成本单独报告。
- 记录 GPU 型号、驱动、Vulkan 能力、构建模式、矩阵形状、dtype、临时内存与选中算法。
- GEMM 按约 $2MNK$ 计算 FLOP 数，同时说明计数约定；带宽型算子关注有效带宽。
- 性能回归仅在固定硬件和明确噪声容限下判定；有条件时与相同设备上的成熟后端比较，不能将不同 GPU 的结果直接归因于库实现。

## 7. 主要风险与应对

| 风险 | 应对 |
| --- | --- |
| 厂商特性、subgroup 大小和驱动差异 | 通用 compute 基线、运行时查询、跨设备验证 |
| 小算子被提交和同步开销主导 | 先测量，再做命令复用、异步和有限融合 |
| 卷积算法与临时内存复杂度上升 | 先直接卷积，显式 workspace 查询和容量限制，再增加算法 |
| FP32/低精度在不同设备上误差不同 | 明确累加方式、容差、可选精度路径和确定性约束 |
| GPU CI 资源不足 | CPU CI 常规运行，GPU 专用环境定期及发布前验证 |
| 过早追求 cuDNN/cuBLAS 全量覆盖 | 按里程碑封闭范围，以实际工作负载决定下一步 |

## 8. 下一步实施顺序

1. 完成 M0 的设备、许可证和接口契约决策。
2. 建立工程骨架与 shader 构建链，落地 M1 的 vector add 闭环。
3. 实现并验证通用 FP32 GEMM，然后添加 tiled kernel 与 benchmark。
4. GEMM 稳定后进入推理算子阶段；避免同时铺开训练、量化和多 GPU。

每个里程碑完成后更新 README 的实际支持状态、构建步骤和限制；未实现能力始终保留为计划，不能写成已支持。
