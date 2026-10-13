# MicroCLIP 复现可行性审核

审核日期：2026-10-13。对象：本地工作目录 `E:\Microscopy\arxiv\MicroCLIP` 当前文件，而非已确认发布的远端版本。模式：artifact / 可复现性。

## 结论

**Agent 可以协助完成环境配置、数据重建和模型评测，但当前仓库尚不能独立闭环复现论文。** 已有较完整的评测实现；获取 MicroCLIP 权重和部分测试数据仍是前提。完整重新训练还缺少训练语料及其构建流程。

| 复现目标 | 当前判断 | 成立条件 |
|---|---|---|
| 随机初始化模型的前向检查 | 代码具备入口，运行未验证 | 可用 Python / PyTorch 环境 |
| 自有图片推理 | 条件可行，运行未验证 | 可获取且兼容的 MicroCLIP checkpoint |
| 五个分类设置和 1,000 对检索评测 | 条件可行，未得到实测指标 | checkpoint、完整原始数据、本地导入及所需授权、重建校验通过 |
| 按原配方重新训练 | 当前不能仅依靠仓库完成 | 三类训练语料、冻结划分与 caption、原始数据加工配方、足够算力 |
| 完整主表、基线和消融复现 | 当前交付不足 | 各基线配置和入口、训练资源、完整实验协议 |

## 已实际核验

通过 PowerShell 的 GZip 解压、JSON 解析和 SHA-256 核验检查了发布元数据；未用替代数据或重新采样。

| 清单 | 记录数 | 分类候选数 | 清单 SHA-256 | 分类计数 |
|---|---:|---:|---|---|
| iid | 5,900 | 59 | 通过 | 通过 |
| partial_ood / compositional_shift | 6,400 | 64 | 通过 | 通过 |
| cell_ood / cell_line_shift | 5,900 | 59 | 通过 | 通过 |
| structure_ood / organelle_set_shift | 5,400 | 54 | 通过 | 通过 |
| lmc_source_shift | 1,158 | 13 | 通过 | 通过 |
| retrieval1000 | 1,000 | 不适用 | 通过 | 不适用 |

六份清单共 25,758 条记录，按大小写敏感的 image_path 去重得到 **25,739 张唯一图像**。每条记录均有配方对应，未发现同一路径对应冲突图像哈希。配方文件及 MicroPool 内两份语义表的 SHA-256 均与 release.json 一致。

这证明交付元数据内部一致，不证明原始下载链接全部可用、转换后的图像字节吻合，或模型指标达到论文数值。

## 阻塞与需要补齐的材料

### 1. MicroCLIP 权重没有可执行的获取步骤（评测阻塞）

README.md 和 MODEL_CARD.md 只要求另行获取 checkpoint，并给出 `/path/to/microclip_epoch015.safetensors` 占位路径。当前目录未发现 MicroCLIP 的 .pt / .pth / .safetensors 文件，文档没有给出对应下载地址、版本和 SHA-256。

DFN2B 初始化模型的官方页面可访问，但它是微调前的 backbone，不能代替 MicroCLIP epoch-15 权重。需要补齐实际发布位置、精确文件名、校验和、checkpoint 格式及权重使用条款。

### 2. 测试集存在明确的人工输入环节（评测阻塞）

按冻结配方统计，25,108 张唯一图像标记为自动获取类型；这只是配方分类，尚未逐源联网下载验证。其余输入如下：

| 来源 | 需要本地导入的唯一图像数 |
|---|---:|
| BPAEC | 246 |
| DynamicNuclearNet segmentation | 160 |
| DynamicNuclearNet tracking | 164 |
| FMD | 54 |
| BioSR | 2 |
| SR-CACO-2 | 2 |
| 合计 | 628 |

另有 3 张 CTC Fluo 图像需要文档指定的授权。账号、凭证、手工下载及权限获取不能由 agent 凭空补足；已有合法本地文件时，agent 可协助映射路径、转换、打包和核验。`--authorized-source fluo` 只是声明实际已获授权，不能授予权限。

构建器默认 all 阶段会在缺少上述输入时停止。可先用 `--stage fetch` 获取可用来源，但这不会形成完整论文评测集。不得删去这些样本后继续声称复现原表。

### 3. 训练数据交付不完整（完整训练阻塞）

MicroPool/README.md 明确只重建评测数据，不重建约 250 万张训练图像。paths.sh 仅指定 microscopy、auxiliary、augmentation 三类 WebDataset 路径和 shard 范围，未给出可下载的实际训练发布包。

完整训练至少需要：冻结训练样本列表及防泄漏划分、L2 长描述、辅助自然图像及文本来源、稀有类增强规则和随机种子、图像加工与打包过程、各文件校验和。重建测试集不会自动获得这些训练资源。

### 4. 训练规模文档不一致（需要作者确认）

README.md 使用 2,505,017 张 microscopy 图像；paths.sh 默认和 MODEL_CARD.md 使用 2,498,617 条基础训练记录，差 **6,400** 条。差值恰与 compositional 测试集大小一致，但不能据此推断排除原因或数据泄漏；必须提供冻结训练清单及解释。

默认总样本数为 2,498,617 + 609,740 + 175,516 = 3,283,873。现有 loader 按全局 batch 8,192 取整，并按 4 个 worker 对齐，得到每 epoch 400 个 batch、30 epoch 12,000 次更新，与 MODEL_CARD 的 schedule 一致。样本数差异仍影响数据组成，不能仅凭 update 数相同忽略。

### 5. 环境和当前运行验证的边界

仓库启动命令针对 Linux / WSL + Bash。当前 agent 会话的 python / python3 命令解析到无法执行的 WindowsApps 别名，未找到可调用的 Bash 或 nvidia-smi；这不能证明用户整台机器没有 Python、GPU 或 WSL 环境。

因此本次没有执行 `MicroPool/tests/test_release.py`、模型前向 smoke test、真实 checkpoint 加载、下载转换或 GPU 训练。不能将此报告称为运行复现成功。

根 requirements.txt 使用 NumPy 2.4.5，MicroPool/requirements.txt 使用 NumPy 2.2.6；两者不是同一套可同时满足的精确依赖约束。建议分别交付重建环境和训练/评测环境，并注明切换流程。MicroPool 的 imagecodecs 仅有下限约束，对要求图像字节级一致的转换还应记录 codec 版本及验证平台。

CI 使用 Python 3.12、未锁定的 CPU torch/torchvision；它不能证明文档声称的 Python 3.13 / CUDA 13 精确环境已经通过。现有 smoke test 不需要 checkpoint 和真实图像，也未覆盖训练 import、优化器更新和完整评测闭环。

训练配方宣称 8 × 72 GB GPU、约 55 小时，即约 440 GPU 小时；本次未验证其性能。减少 GPU、batch 或更改 schedule 可用于代码检查，但不能直接保证复现原配方结果。

### 6. 评测可信性与实验范围

evaluate.py 的分类实现使用每个子集内部的候选标题，检索使用冻结 long_caption、全体 1,000 对相似度及双向 Recall@1，静态上与 README 所述范围相符；加载 checkpoint 使用 strict=True。这些是已有的良好基础。

不过 evaluate.py 只检查 verification.json 存在，没有读取其通过状态、完整 split 覆盖，也没有把当前 tar 内容绑定到该验证报告。数据在验证后变化时，评测可能继续执行。建议评测前重新验证，或记录并校验 shard 哈希。

根评测入口针对 MicroCLIP 固定 ViT-B/16、256-token 配置，仓库 model_configs 当前只有 ViT-B-16。README 主表中的 SigLIP、SigLIP 2、PE-Core、Cell-DINO 和监督分类器没有形成同等完整的独立复现入口；不能把跑通 MicroCLIP 一行等同于复现整张主表。

补充模型配置核查：train_DFN_B16.sh 明确使用 `ViT-B-16` 和 `dfn2b`，pretrained.py 将该组合映射到 `apple/DFN2B-CLIP-ViT-B-16`。因此 README 的“从 DFN2B-CLIP ViT-B/16 初始化”与当前训练入口一致；`ViT-B-16` 是结构名，`dfn2b` 是初始化权重来源。inference.py 使用 `pretrained=None` 后加载完整 MicroCLIP checkpoint，不能仅凭这个参数判断它没有 DFN 初始化背景。

但激活函数需要核对实际训练记录：dfn2b 权重元数据声明 `quick_gelu=True`，本地 ViT-B-16.json 没有此项，train.sh 的 `BACKBONE_FORCE_QUICK_GELU` 默认是 0。当前 factory.py 对这个不一致只发出警告，不自动切换，因此默认训练和推理均构建 GELU。若论文实际训练时显式启用了 QuickGELU，当前推理入口就不匹配；若实际使用仓库默认 GELU，两者则一致。GELU / QuickGELU 无可区分的可训练参数，strict=True 加载成功也不能证明激活函数一致。发布权重时应同时注明实际训练的激活函数及构建配置，不能只提供结构名和权重字节。

### 7. 本地交付与远端交付的区别

当前工作目录存在大量修改；MicroPool/、evaluate.py、eval.sh、src/open_clip_train/train.py 等在本地 git status 中仍显示为未跟踪。报告检查的是这些实际存在的文件，不能保证他人 clone 当前远端就拥有相同内容。应发布固定 commit / tag，并从干净 checkout 重跑验收。

本次未读取到 ccfa.yaml，未使用项目状态追踪。GitHub 项目页面的外部访问未成功，不能据此断言远端仓库或权重不存在。

## 建议的验收顺序

1. 提供 MicroCLIP epoch-15 权重的实际发布链接、SHA-256 和使用条款。
2. 固定可运行的 Linux / WSL 环境，分别提供重建与模型环境锁定文件。
3. 在全新 checkout 上运行清单单元测试、256-token 模型前向及训练入口检查。
4. 运行数据 plan；补齐 628 张本地导入图像及 3 张授权图像，再 fetch → build → verify。
5. 加载真实 checkpoint，输出五个分类指标与双向检索指标；保存权重、数据、环境标识及与论文的误差。
6. 若目标包括完整重训，先发布三类冻结训练资源并统一样本数，再验证单次训练更新、保存和恢复，最后执行原始 30-epoch schedule。
7. 若目标包括整张主表与消融，补齐每个方法的模型标识、原生预处理、启动参数及结果验收规则。

**最低可复现发布门槛：新复现者在干净环境里，只凭公开文件和明确列出的人工输入，能够获得并验证数据、下载指定权重、运行评测并生成可比对的指标。当前最优先补齐权重获取、测试数据人工输入指南和环境验证记录。**

## 外部核验记录

- DFN2B 初始化权重官方页面：https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16 （可访问；不代表 MicroCLIP 权重已发布）。
- PyPI 精确版本元数据：torch 2.12.0、Pillow 12.2.0、NumPy 2.4.5 的 JSON 页面可访问；本次未进行实际安装或平台 wheel 兼容验证。
- GitHub MicroCLIP 与 CTC 官方页面未成功获取；有关本地包要求的判断取自仓库文档与代码，不作为当前上游政策的新核验结论。
