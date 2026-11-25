# 前向执行优化与实测

## 实现与边界

参考本地 ncnn Vulkan 的长期 pipeline 与 pack4 思路，独立实现本工程 shader 和调度；没有复制、编译 ncnn 源码或引入其类型。Softmax 不机械采用四 dispatch：每组一个 64-thread workgroup，在 shared memory 中归约 maximum、NaN 标志和指数和，再写出结果；INSTANCE 连续访问，CHANNEL 按 NCHW 空间跨度访问。超过 X dispatch 限制的组数映射到二维 workgroup 网格。

ReLU/Sigmoid/tanh 每 invocation 处理四个标量，使用 vec4 运算；输入输出仍是标量 buffer，允许四字节偏移，不要求 vec4 对齐。不足四个元素的尾部走标量路径。保留 alpha/beta、beta=0 不读旧输出、激活 alpha=0 不读输入、NaN 和极值语义。Softmax 归约顺序改变，允许正常 FP32 舍入差异；未改变现有别名、范围、设备校验或公开 API。

运行时 context 持有 shader module、descriptor layout、pipeline layout、pipeline、descriptor pool/set，按嵌入 shader 标识、code size、buffer 数和 push-constant 大小查找；当前最多四个固定 shader 条目，不按形状增长。每次仍更新 buffer descriptors、记录命令、提交并等待 fence。已有全局运行时串行化保证 descriptor 复用安全。创建失败由局部 RAII 清理，不发布半成品。DeviceReset 等待设备并销毁 context/cache；静态 runtime 仅保存 context 裸指针，不在 DLL 卸载时调用 Vulkan，调用方仍须 reset 全部使用过的设备。共享执行基础设施也复用于反向/GEMM/卷积，但未修改其算法。

## 可复现 benchmark

`tests/forward_benchmark.cpp` / CMake target `vkdnn_forward_benchmark`，不加入正确性 CTest。构建 Release 后运行 `build\Release\vkdnn_forward_benchmark.exe`。2026-11-25 Windows，NVIDIA GeForce RTX 5090，Vulkan driverVersion=2588573696，apiVersion=4211039，Vulkan SDK 1.4.363.0 / VS 2026 v145。旧 DLL 是本次优化前的**现有工作树**，不是 HEAD；相同 benchmark 二进制分别在旧/新 DLL 同目录下执行。

每例固定 FP32 输入、alpha=1/beta=0；排除分配/传输，10 次预热，51 个样本，每样本连续五次同步 API 调用取平均，再计算中位数与 p10/p90；关闭 validation 测性能。以下单位均为微秒，是**宿主同步 API 暖调用延迟**，包括参数校验、命令录制、提交及等待，**不是 GPU timestamp/kernel 时间**。不测首次编译延迟，不与 ncnn 比较。

| 算子 / NCHW | 旧中位数 [p10,p90] | 新中位数 [p10,p90] |
| --- | --- | --- |
| Softmax INSTANCE / 32,4099,1,1 | 1404.560 [1382.860,1509.260] | 64.240 [61.580,73.320] |
| Softmax CHANNEL / 2,131,17,19 | 210.740 [183.180,314.900] | 47.660 [46.160,57.980] |
| Softmax INSTANCE / 1,1,1,1048579 | 157977.860 [157806.040,168372.460] | 8393.120 [8277.600,8780.620] |
| Sigmoid / 1,16,257,257 | 270.820 [221.400,347.900] | 55.620 [51.100,78.240] |
| ReLU / 1,16,257,257 | 286.140 [219.340,429.700] | 55.760 [51.700,66.340] |
| tanh / 1,16,257,257 | 233.420 [205.540,345.360] | 54.020 [51.620,71.820] |

所有测量例均改善，但这些是缓存和 shader 改动的合计收益，未单独归因 vec4；不承诺所有尺寸/设备更快。单个超大 softmax 组只占一个 workgroup，仍有约 8.4 ms 延迟，后续可评估多阶段分组归约。CHANNEL 短组会闲置 lanes；未实现自适应 subgroup、异步执行或命令缓存。未测 GPU-only 时间、冷调用和其他设备，不能推断其收益。

## 验证

- CMake Tools Build/CTest 均报“无法配置项目”，诊断工具无 CMake diagnostics；改用已有 build 目录的原生 CMake/MSBuild/CTest。
- CMake Release 九项 CTest 全通过；原生 VS Release DLL 使用相同九个测试可执行文件另目录运行，也全部通过。两套工程继续使用同一源码、shader 与编译选项，无须新增库源文件。
- 使用 `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`、`VK_LAYER_VALIDATE_SYNC=1` 运行真实 GPU 测试，无资源/同步 validation 报错。仅一张 GPU，跨设备子用例显式跳过。
- 新增大组 4099、CHANNEL 131、超过 65535 个 channel 组、batch、非 64/4 尾部、四字节偏移、guard、alpha/beta、NaN/Inf 与 reset 后重新创建缓存的 CPU 参考测试；原有前向/反向有限差分和边界测试保留。
- dumpbin 比较旧 DLL、新 CMake DLL 与原生 VS DLL，63 个 C ABI 导出名称完全一致。修改的 C++/GLSL 均按 `C:\project\_clang-format` 格式化并严格校验，GLSL 使用 C++ 模式且覆盖 `RemoveSemicolon: false`。
