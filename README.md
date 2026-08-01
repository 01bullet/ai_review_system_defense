# AI Review System — LLM 审稿系统的 Prompt Injection 防御

基于 [Sakana AI — The AI Scientist](https://github.com/SakanaAI/AI-Scientist) 二次开发的安全研究系统：构建 **StruQ 结构化查询微调** 的审稿 LLM，使其天然抵抗隐藏在论文中的提示注入攻击，并提供 Web 审稿界面。

> ⚠️ 本仓库为**源码包**，不含模型权重（约 2.2 GB/adapter）。权重获取方式见下方「模型权重」章节。

---

## 一、项目是什么

恶意论文作者可以在 LaTeX 源码中嵌入**人眼不可见、但 PDF 文本提取后可被 LLM 读取**的注入指令（白字、微缩字体、负 vspace、旋转、零宽字符等），诱导审稿 LLM 给出虚高评分。

本项目构建 **三层防御体系 + StruQ 微调**：

| 层 | 模块 | 机制 |
|----|------|------|
| L1 | `ai_scientist/latex_sanitizer.py` | LaTeX 编译前剥离 6 类隐藏文本 |
| L2 | `ai_scientist/structured_query.py` | `[DATA]...[/DATA]` 指令/数据通道隔离 + 递归过滤保留 token + 指令层级声明 |
| L3 | `ai_scientist/gan_defense/` | T5 Generator 学攻击 → BERT Discriminator 检测，检测到攻击自动升级防御 |
| **微调** | `struq_defense/` | QLoRA 微调 Qwen2.5-7B 基础版，只遵循 `[INST]` 区域指令，将 `[DATA]` 视为不可信 |

攻击模拟：`ai_scientist/attack_injector.py` 内置 **17 种**注入策略（视觉隐藏 / 语义操控 / 结构化攻击 / 组合攻击），**仅用于安全研究，禁止用于真实投稿**。

---

## 二、快速开始

### Windows
双击 `run.bat`（自动安装依赖 → 下载基座模型 → 启动 Web 界面）。

### Linux / macOS
```bash
bash run.sh
```

### 手动
```bash
pip install -r requirements.txt
python ensure_model.py     # 下载 Qwen2.5-7B 基座模型 (~15 GB，首次运行)
python download_models.py  # 安装微调 adapter（见「模型权重」）
python review_app.py       # 启动 http://localhost:8000
```

---

## 三、模型权重

本地审稿需要两部分权重：

| 权重 | 大小 | 获取方式 |
|------|------|---------|
| 基座模型 Qwen2.5-7B | ~15 GB | `python ensure_model.py`（HF / ModelScope 镜像） |
| **微调 adapter**（StruQ 防御审稿模型） | ~2.2 GB / 个 | `python download_models.py` |

`download_models.py` 支持三种来源（按优先级）：

```bash
python download_models.py --list                      # 查看已安装情况
python download_models.py                             # 安装 v2a（推荐）
python download_models.py --all                       # v2a + v3b
python download_models.py --local-dir PATH            # 从本地已有目录安装
python download_models.py --local-archive model.tar.gz # 从本地压缩包安装
```

> 本仓库不托管 2.2 GB 权重文件。若从本仓库克隆，请把 adapter 归档放在自己的网盘/发布附件，并把直链填入 `download_models.py` 的 `ADAPTERS["..."]["urls"]`，或用 `--local-archive/--local-dir` 指向本地文件。

### 本地审稿模型版本（`review_app.py` 下拉框）

| 版本 | 路径 | 评估结论 |
|------|------|---------|
| **v2a** | `models/struq_v2_a/struq_lora_adapter` | ✅ **生产可用**（推荐）— 审稿有区分度，真实防御率 66.7% |
| **v3b** | `models/struq_v3/v3b_api_defense` | ✅ 重训版（max_seq_length=6144，已修复全 1 分崩溃），攻击抵抗良好，评分区分度一般 |
| v1 / v2 | `models/struq/...` / `models/struq_v2/...` | 早期版本 |
| v3a | `models/struq_v3/v3a_human_align` | 基础对齐版 |
| v3c | `models/struq_v3/v3c_heavy_defense` | ❌ **不可用** — 过度防御训练导致「防御崩溃」（对所有论文输出全 1 分） |

---

## 四、Web 界面功能

`review_app.py`（FastAPI）支持：
- **论文上传**：PDF / LaTeX / Markdown
- **攻击检测**：注入模式识别 + LaTeX 隐藏内容分析 + 段落风险热力图
- **清洗防御**：一键清除白字、极小字体、scalebox 等隐藏文本
- **AI 审稿**：综合防护 / 规则防御 / GAN 检测 / 无防御，四模式切换
- **本地 vs 云端审稿**：本地 Qwen2.5-7B+adapter，或 DeepSeek / Claude / GPT-4o API（无需本地权重）
- **防御对比**：defense on/off 并排比较 + 雷达图

---

## 五、目录结构

```
├── review_app.py              # FastAPI Web 审稿界面（前端内嵌）
├── start_system.py            # 一键启动脚本
├── launch_scientist.py        # CLI：完整科研流水线（含攻击注入 + 防御审稿）
├── ensure_model.py            # 基座模型自动下载
├── manual_model_loader.py     # 低显存 4-bit 模型加载器（RTX 5060 等）
├── download_models.py         # adapter 权重安装脚本
├── ai_scientist/              # 核心引擎
│   ├── attack_injector.py     # 17 种注入攻击策略
│   ├── latex_sanitizer.py     # L1 清洗
│   ├── structured_query.py    # L2 StruQ 结构化查询
│   ├── gan_defense/           # L3 GAN 对抗防御
│   └── perform_review.py      # 审稿引擎（ensemble + reflection + 三层防御）
├── struq_defense/             # StruQ 结构化指令微调子系统
│   ├── frontend.py            # Secure Front-End（过滤/编码）
│   ├── reviewer.py            # 本地防御审稿器
│   ├── run.py                 # build-dataset / train / review / evaluate
│   └── config*.py             # 各版本训练配置（v1/v2/v2_a/v3）
└── static/                    # 前端资源
```

---

## 六、环境要求与排障

| 项目 | 要求 |
|------|------|
| Python | 3.10+（推荐 3.13） |
| GPU | 本地审稿需 NVIDIA 6+ GB 显存；纯 API 审稿无需 GPU |
| 磁盘 | ~30 GB（基座模型 15 GB + adapter + 代码） |

**排障**：
- **模型下载失败**：`python ensure_model.py --source huggingface` 或 `--source modelscope`
- **显存不足 (OOM)**：4-bit 模式约占用 5 GB。显存更小请改用 API 审稿模式，或编辑 `manual_model_loader.py` 增加 CPU offload
- **Windows/RTX 5060 崩溃**：加载器已内置 workaround；仍异常时设 `CUDA_LAUNCH_BLOCKING=1` 或改用 API 审稿

---

## 七、许可与免责

- 继承 upstream **The AI Scientist Source Code License**（见原项目）。
- **攻击注入模块仅供安全研究与防御评估**，不得用于向真实学术会议/期刊投稿。
- 使用 AI 生成内容发表时须按原项目许可披露 AI 参与。

## 八、相关引用

```bibtex
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```
