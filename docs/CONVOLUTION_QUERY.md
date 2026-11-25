# 卷积算法查询子集

独立公开头 `vkdnn/vkdnn_conv_query.h` 包含 `vkdnn_ops.h`。接口参数顺序和性能结构字段参考本机 CUDA v13.2 include 中的 `cudnn_cnn.h`，math/determinism 枚举参考同目录 `cudnn_graph.h`、`cudnn_ops.h`。仅参考 API 形式，无 CUDA/cuDNN 运行依赖，**不提供 cuDNN ABI 兼容性**。

- 三个 `GetConvolution*Algorithm_v7` 通过已有公开 workspace 查询验证句柄、描述符和形状，不访问内部描述符。支持范围与对应 workspace 查询一致：FP32 连续 NCHW，支持 groups/depthwise。forward 唯一候选是 DIRECT；backward data/filter 唯一候选各为 ALGO_0。本地 ALGO_0 是确定性 gather，不代表 cuDNN 同名算法实现。
- 请求数量必须大于零；成功只写一项和返回数量 1，不触碰其余数组元素。memory 为实际 workspace 字节数（当前为零），mathType 为 DEFAULT，determinism 为本地确定性路径。reserved 清零。
- `_v7` 的 **time 为 quiet NaN，表示未测量**，不是零耗时或性能预测。该查询不启动 benchmark，不声称具备 cuDNN 的性能排序能力。
- 三个 `FindConvolution*AlgorithmEx` 先经 `_v7` 和公开 workspace 查询验证配置，再实际调用对应同步前向、backward data 或 backward filter 算子一次。固定宿主 FP32 alpha=1、beta=0，覆盖调用方目标缓冲区；成功返回唯一算法及真实测量时间。无预热、无重复平均、无多算法性能排序。
- FindEx 的 time 单位为毫秒，使用 `std::chrono::steady_clock` 测量**完整同步算子调用的宿主墙钟耗时**，包含参数校验、pipeline 创建/缓存、资源准备、提交和等待等开销（以本次路径实际发生为准），不包含此前 workspace 查询或外部数据上传/下载。**不是纯 GPU kernel 时间**，不能作为设备 timestamp benchmark。首次调用可能含较大的 pipeline 开销。
- FindEx 数据范围、容量、设备归属及别名校验复用实际算子的规则，workspace 至少满足查询要求（当前零 workspace）。错误原样传播，返回数量和性能数组保持不变；执行阶段错误可能已修改目标缓冲区，不承诺回滚。成功只写第一项，保留数组尾部。
- `SetConvolutionMathType` 由描述符注册表所在实现负责验证，仅 DEFAULT 成功，其他模式返回 NOT_SUPPORTED；无效描述符返回 BAD_PARAM。DEFAULT 不改变卷积配置。

`tests/convolution_query_test.c` 验证 C 头文件/链接、唯一结果、未测量查询的 NaN 时间、FindEx 有限非负实测时间、错误输出不变、group 配置及 math 限制。三类 FindEx 直接执行真实 GPU 前向、dx/dw 并下载核对结果；覆盖 beta=0 不读取旧目标、非法参数/空地址与别名拒绝。`_v7` 查询自身不执行 GPU kernel。

Debug query test 已在 NVIDIA GeForce RTX 5090 上启用 `VK_LAYER_KHRONOS_validation` 实际通过，未报告 validation 错误。测试打印本次三个同步调用的测量值，仅作为实测路径证明，不是稳定性能基线。
