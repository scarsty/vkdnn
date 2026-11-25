# 反向执行优化与实测

## 实现与边界

独立实现并行归约、向量化和局部消除重复扫描；未复制 ncnn 源码，本地 ncnn 快照是前向参考，不提供本工程训练反向实现。不与 ncnn 比较性能。

- SoftmaxBackward FAST/ACCURATE/LOG：每组一个 64-thread workgroup，共享内存归约 `sum(y*dy)` 或 `sum(dy)`，再分线程写结果；每组工作量由 O(L²) 降到 O(L)。INSTANCE 连续访问，CHANNEL 空间跨步访问，组数超过 X 上限时使用二维 dispatch。归约树固定，无 float atomic；FP32 累加次序变化允许舍入差异。单个超长组仍仅使用一个 workgroup，未实现多阶段归约。
- ActivationBackward：每 invocation 处理四个标量，vec4 计算，尾部标量处理；仍允许四字节偏移，无 vec4 对齐要求。保留 ReLU NaN 模式、sigmoid/tanh 使用未缩放 y、alpha=0 不读输入、beta=0 不读旧输出和别名范围校验。
- 平均池化反向仅在 NOT_PROPAGATE_NAN 下以窗口交集直接计算 divisor，省去重复扫描整个窗口；NaN 输入位置仍不接收梯度。PROPAGATE_NAN 和最大池化保留窗口扫描，保持 NaN 传播及第一个最大值 tie 规则。该局部优化未单独测性能，不能声称提速。
- 卷积 dx/dw/db 保持既有确定性 gather 和累加顺序；评估后未重写为 scatter/atomic 或大算法，不宣称卷积提速。batch/groups/dilation/核翻转路径不变。
- 复用现有 context pipeline/descriptor 缓存及 reset 生命周期，不增加缓存或公开 API。CMake/VS 同用原 shader 文件，无新增库源码，VS 无须增加项目项；benchmark 是 CMake 非 CTest 目标。

## 可复现 benchmark

构建 Release 后运行 `build\Release\vkdnn_backward_benchmark.exe`。相同可执行文件复制到 `build\backward-baseline` 与优化前 DLL 同目录执行。基线为修改前现有工作树 DLL，不是 HEAD（该次开始时工作树 clean）：SHA256 `B0D33129ABC9FB78B5CBC2BF9CCED6AF820801AF8EC13DE4856EBE6A6F1CB947`。原 DLL 保留在该目录；不要用重新构建 HEAD 替代。

2026-11-25，Windows / RTX 5090，driverVersion=2588573696，apiVersion=4211039，SDK 1.4.363.0 / VS 2026 v145。固定 FP32 输入和 dy，alpha=1、beta=0，前向预先计算 y；分配、传输、前向均不计时。10 次预热，51 样本，每样本五次同步 API 调用取平均，中位数和 p10/p90，关闭 validation。下表单位为微秒，含宿主校验、命令录制、提交和等待，**不是 GPU timestamp/kernel 时间**。旧路径最大单组限制到 16387，避免 O(L²) 超大基线。

| 算子 / NCHW | 旧中位数 [p10,p90] | 新中位数 [p10,p90] |
| --- | --- | --- |
| Softmax INSTANCE / 32,4099,1,1 | 342.680 [332.060,400.580] | 64.920 [57.340,81.300] |
| Softmax CHANNEL / 2,131,17,19 | 67.060 [65.500,78.300] | 48.180 [46.260,64.820] |
| Softmax INSTANCE / 1,1,1,16387 | 1108.660 [1063.800,1133.320] | 138.380 [133.680,152.620] |
| LogSoftmax INSTANCE / 8,1027,1,1 | 97.920 [84.840,116.400] | 51.560 [49.280,93.200] |
| Sigmoid / 1,16,257,257 | 55.980 [52.920,64.580] | 46.440 [44.640,61.500] |
| ReLU / 1,16,257,257 | 61.540 [52.860,72.240] | 53.560 [46.260,69.500] |
| tanh / 1,16,257,257 | 54.060 [48.960,62.720] | 48.340 [45.180,80.660] |

这是一次暖同步 API 实测，存在宿主调度噪声；激活区间重叠，不承诺所有形状更快。缓存已存在于基线，不把缓存收益重复归因本次改动。未测其他设备、冷调用、GPU-only 时间或端到端训练。

## 验证证据

- CMake Tools Build/CTest 都无法配置；GetDiagnostics 无诊断。随后现有 build 原生 CMake/MSBuild Release 构建成功。原生 VS 首次因未设置 VULKAN_SDK 失败，设置 SDK 路径后成功。
- Release 九项 CTest 全通过；`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`、`VK_LAYER_VALIDATE_SYNC=1` 下九项全通过，完整日志 `build\backward-validation.log` 无 VUID/SYNC-HAZARD。相同九个测试另目录搭配 VS DLL 全通过。旧 build 中不属于九项注册测试的历史 `vkdnn_stub_test.exe` 不适配当前 ABI，不作为回归测试。
- 增加 65539 激活尾部、4099 Softmax 组、CHANNEL 131、超过 65535 个 channel 组、batch、偏移 guard、beta=0 旧输出 NaN 的 CPU/GPU 差分；同一大组测试在 reset 重建后再次执行。现有有限差分、NaN、alpha=0、范围、别名及池化/卷积反向测试保留。仅一张 GPU，跨设备子用例显式跳过。
- dumpbin 比较基线、CMake 和 VS DLL，63 个 C ABI 导出名称完全一致。修改 C++ 按 `C:\project\_clang-format` 格式化并 dry-run/Werror；GLSL 用相同规则 C++ 模式并覆盖 RemoveSemicolon:false 严格校验。编辑器 C++ diagnostics 无错误。
