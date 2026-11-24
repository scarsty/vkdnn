# 工程智能体指引

## 必须遵循的开发约定

- 修改本工程前阅读 [DEVELOPMENT.md](DEVELOPMENT.md)，其为工程开发标准；项目目标与进度见 [README.md](README.md) 和 [PLAN.md](PLAN.md)。
- **公开 API 风格模仿 CUDA Runtime、cuBLAS 和 cuDNN**：基础运行时参考 CUDA，矩阵计算参考 cuBLAS，神经网络算子参考 cuDNN。
- 有对应 API 时，除工程前缀外，函数名称、大小写、签名、参数顺序及枚举形式尽量保持一致。先核对本机 SDK 头文件或官方文档，不凭记忆另创接口。
- 基础运行时使用 `vkdnnError_t`、`vkdnnSuccess`、`vkdnnError...` 和 `vkdnnMemcpyKind`；复制统一为带方向枚举的 `vkdnnMemcpy`。
- 保持 CUDA 风格的当前设备、隐式运行时上下文与 `void*` 显存地址，不重新向基础内存 API 增加显式上下文、内存句柄或 offset 参数。BLAS/DNN 层仍参考各自库的显式句柄和描述符模式。
- 底层只用 Vulkan 实现，不添加 CUDA/cuBLAS/cuDNN 运行依赖；不宣称 ABI 兼容，不虚构 Vulkan 无法提供的 CUDA 能力。必要偏离应明确说明并测试。
- ncnn 仅作算法和 Vulkan 实践参考，自行实现本工程 shader 与调度；不得直接编译 ncnn 源码或引入其内部类型/宏依赖。已有 third_party/ncnn 是参考快照，不是已实现算子，不能把下载源码当作完成功能。
- API 变更同步更新实现、调用方、测试及文档；保留 C ABI，验证动态库导出。
- 修改 C/C++ 或 CUDA 代码后，使用 `C:\project\_clang-format` 格式化本次修改文件，并通过 `--dry-run --Werror`；保留编码和 BOM，禁止嵌套三元表达式。
- 自有 `.comp` shader 也必须使用上述规则按 C++ 模式 clang-format（标准输入配合 `--assume-filename=shader.cpp`），同模式严格校验，并重新编译及测试；不批量改动第三方参考快照。
- GLSL 必须单独覆盖 `RemoveSemicolon: false`，避免 C++ 格式化删除 buffer block 的必要分号；具体继承命令见 DEVELOPMENT.md。
- GPU 相关改动需运行实际测试，资源/同步改动应启用 Vulkan validation layers；设备不可用时报告跳过，不能冒充通过。

## 已确认决策

- BLAS 层用 `vkblas` 前缀，DNN 层用 `vkdnn` 前缀；基础 CNN 接口与桩状态见 [docs/CNN_OPERATORS.md](docs/CNN_OPERATORS.md)。BLAS 使用 cuBLAS 风格列主序 GEMM，不沿用早期 row-major 草案。

- 基础运行时接口已参考本机 CUDA 13.2，由显式上下文与分方向复制接口调整为 CUDA Runtime 风格；后续工作不得无故退回旧形式。
- 工程构建为动态库，直接链接 Vulkan 加载器；Windows 运行时依赖 `vulkan-1.dll`，不使用手动加载作为默认路径。
- 当前采用同步复制；非对齐设备复制与 memset 可暂经 staging 路径实现，其性能限制需如实保留在文档中。
