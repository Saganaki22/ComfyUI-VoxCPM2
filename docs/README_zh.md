<div id="readme-top" align="center">
  <h1 align="center">ComfyUI-VoxCPM2</h1>

  <p align="center">
    <a href="../README.md">English</a> | <b>中文</b>
  </p>

  <p align="center">
    <strong>VoxCPM2</strong> 的 ComfyUI 节点 — 无分词器、扩散自回归文本转语音。
    <br>20亿参数、30种语言、48kHz音频输出、语音设计、可控声音克隆和LoRA训练。
    <br /><br />
    <a href="https://github.com/Saganaki22/ComfyUI-VoxCPM2/issues/new?labels=bug&template=bug-report---.md">报告Bug</a>
    ·
    <a href="https://github.com/Saganaki22/ComfyUI-VoxCPM2/issues/new?labels=enhancement&template=feature-request---.md">功能建议</a>
  </p>
</div>

<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<br>

## 简介

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) 是一个无分词器的文本转语音模型，基于超过200万小时的多语言语音数据训练。采用 MiniCPM-4 骨干网络和 AudioVAE V2，输出 **48kHz 演播室级音频**，支持 **30种语言**，无需语言标签。

本自定义节点提供了两个推理节点和一个完整的 LoRA 训练流程，全部直接集成到 ComfyUI 中 — 基于 [@wildminder](https://github.com/wildminder) 的原始 [ComfyUI-VoxCPM](https://github.com/wildminder/ComfyUI-VoxCPM) 开发。

**主要特性：**
* **30种语言多语种** — 直接输入任意支持语言的文本，无需语言标签
* **语音设计** — 仅通过自然语言描述即可生成全新声音（性别、年龄、音色、情感、语速）
* **可控声音克隆** — 从短参考音频克隆任意声音，可选风格引导
* **终极克隆** — 提供参考音频+文字稿，实现最高保真度的声音复刻
* **48kHz演播室级输出** — 接受16kHz参考音频，通过 AudioVAE V2 内置超分辨率输出48kHz
* **LoRA 支持** — 加载微调的 LoRA 检查点，应用特定声音风格
* **原生 LoRA 训练** — 直接在 ComfyUI 中训练 LoRA 适配器
* **自动模型管理** — 模型自动下载，由 ComfyUI 管理，节省显存
* **Torch 编译** — 可选 `torch.compile` 优化，加速推理

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 安装

### 通过 ComfyUI Manager 安装（推荐）

搜索 `ComfyUI-VoxCPM2` 并点击"Install"。

### 手动安装

1. 克隆到 `ComfyUI/custom_nodes/` 目录：
   ```sh
   git clone https://github.com/Saganaki22/ComfyUI-VoxCPM2.git
   ```

2. 安装依赖：
   ```sh
   cd ComfyUI-VoxCPM2
   pip install -r requirements.txt
   ```

3. 重启 ComfyUI。节点将出现在 `audio/tts` 类别下。

## 模型

模型首次使用时自动下载到 `ComfyUI/models/tts/VoxCPM/`。

| 模型 | 参数量 | 采样率 | 说明 | Hugging Face |
|:---|:---:|:---:|:---|:---|
| **VoxCPM2** | 20亿 | 48kHz | 最新版本。30种语言、语音设计、可控克隆。 | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 节点

### VoxCPM2 TTS

文本转语音，支持可选的语音设计。无需参考音频。

| 输入 | 类型 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| `model_name` | 下拉 | — | 选择 VoxCPM2 模型 |
| `lora_name` | 下拉 | None | 从 `models/loras` 选择 LoRA 检查点 |
| `voice_description` | 字符串 | — | 语音设计提示（可选），如"一个年轻女性，温柔甜美的声音"。自动加括号并拼接到文本前 |
| `text` | 字符串 | — | 要合成的目标文本 |
| `cfg_value` | 浮点 | 2.0 | 无分类器引导比例（1.0–10.0） |
| `inference_timesteps` | 整数 | 10 | 扩散步数。越多=质量越好，速度越慢 |
| `max_tokens` | 整数 | 4096 | 最大生成长度（64–8192） |
| `normalize_text` | 开关 | Normalize | 自动处理数字、缩写、标点 |
| `seed` | 整数 | 42 | 可复现种子（-1 = 随机） |
| `force_offload` | 开关 | Auto | 生成后强制卸载显存 |
| `dtype` | 下拉 | auto | 强制模型数据类型。Auto 使用模型原生 dtype |
| `device` | 下拉 | cuda | 推理设备（cuda、mps、cpu） |
| `retry_max_attempts` | 整数 | 3 | 生成失败时自动重试次数（0–10） |
| `retry_threshold` | 浮点 | 6.0 | 检测异常生成的阈值 |
| `torch_compile` | 开关 | Standard | 启用 `torch.compile` 优化 |

### VoxCPM2 Voice Clone

声音克隆，支持可控克隆和终极克隆模式。

| 输入 | 类型 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| `model_name` | 下拉 | — | 选择 VoxCPM2 模型 |
| `lora_name` | 下拉 | None | 从 `models/loras` 选择 LoRA 检查点 |
| `voice_description` | 字符串 | — | 风格控制提示（可选），如"语速稍快，欢快语气"。自动加括号并拼接到文本前 |
| `text` | 字符串 | — | 要合成的目标文本 |
| `reference_audio` | 音频 | **必填** | 声音克隆的参考音频 |
| `prompt_text` | 字符串 | — | 参考音频的文字稿。填写后启用**终极克隆**（最高保真度）。留空则为**可控克隆** |
| `cfg_value` | 浮点 | 2.0 | 无分类器引导比例（1.0–10.0） |
| `inference_timesteps` | 整数 | 10 | 扩散步数。越多=质量越好，速度越慢 |
| `max_tokens` | 整数 | 4096 | 最大生成长度（64–8192） |
| `normalize_text` | 开关 | Normalize | 自动处理数字、缩写、标点 |
| `seed` | 整数 | 42 | 可复现种子（-1 = 随机） |
| `force_offload` | 开关 | Auto | 生成后强制卸载显存 |
| `dtype` | 下拉 | auto | 强制模型数据类型。Auto 使用模型原生 dtype |
| `device` | 下拉 | cuda | 推理设备（cuda、mps、cpu） |
| `retry_max_attempts` | 整数 | 3 | 生成失败时自动重试次数（0–10） |
| `retry_threshold` | 浮点 | 6.0 | 检测异常生成的阈值 |
| `torch_compile` | 开关 | Standard | 启用 `torch.compile` 优化 |

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 使用方法

### 文本转语音（零样本）
1. 将 **VoxCPM2 TTS** 节点添加到工作流。
2. 在 `text` 字段中输入文本。
3. （可选）在 `voice_description` 中描述声音（如"一个低沉男性声音，沉稳而威严"）。
4. 执行队列。

### 语音设计
`voice_description` 字段让你无需参考音频即可创建任意声音：
- "一个年轻女性，温柔甜美的声音"
- "一个老人，沙哑缓慢的声音"
- "一个孩子，兴奋而有活力"

描述会自动加括号并拼接到文本前面，匹配 VoxCPM2 API 格式 `(描述)文本`。

### 可控声音克隆
1. 添加 **VoxCPM2 Voice Clone** 节点。
2. 将 `Load Audio` 节点连接到 `reference_audio`。
3. 在 `text` 中输入目标文本。
4. （可选）在 `voice_description` 中添加风格引导（如"语速稍快，欢快语气"）。
5. `prompt_text` 留空。

### 终极克隆（最高保真度）
1. 同上操作，但还需在 `prompt_text` 中提供参考音频的**精确文字稿**。
2. 模型使用音频续写克隆技术，精确复刻每一个声音细节。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## LoRA 支持

### 推理
1. 将 `.safetensors` LoRA 文件放入 `ComfyUI/models/loras/`。
2. 在 `lora_name` 下拉菜单中选择你的 LoRA。

### 训练
使用训练节点（`VoxCPM2 Train Config`、`VoxCPM2 Dataset Maker`、`VoxCPM2 LoRA Trainer`）直接在 ComfyUI 中训练自定义 LoRA 适配器。

**[点击查看完整的 LoRA 训练指南](readme-lora-training.md)**

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 最佳实践

### 声音克隆
- 使用**干净、高质量的参考音频**（5-15秒连续语音）
- **终极克隆**时，在 `prompt_text` 中提供准确的逐字文字稿
- 文字稿中的标点符号有助于模型捕捉语调

### 生成质量
- **`cfg_value`（默认 2.0）：** 提高以更贴合提示，降低以获得更自然的变化
- **`inference_timesteps`（默认 10）：** 5-10 用于快速草稿，15-25 用于更高质量
- **`normalize_text`：** 自然语言输入时保持开启。仅在输入音标如 `{HH AH0 L OW1}` 时关闭

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 支持语言（30种）

阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印尼语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、他加禄语、泰语、土耳其语、越南语

中文方言：四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 局限性

- 语音设计和风格控制的结果可能在每次运行中有所差异，建议生成1-3次以获得理想输出
- 不同语言的性能取决于训练数据的可用性
- 非常长或高度表现力的输入偶尔可能出现不稳定
- **严禁**用于冒充、欺诈或虚假信息。AI生成内容应明确标注。

<p align="right">(<a href="#readme-top">回到顶部</a>)</p>

## 许可证

VoxCPM 模型及其组件遵循 OpenBMB 提供的 [Apache-2.0 许可证](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE)。

## 致谢

- **[@wildminder](https://github.com/wildminder)** 的原始项目 [ComfyUI-VoxCPM](https://github.com/wildminder/ComfyUI-VoxCPM)，本项目基于此开发
- **OpenBMB & ModelBest** 创建并开源了 [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **ComfyUI 团队** 提供了强大且可扩展的平台

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/Saganaki22/ComfyUI-VoxCPM2.svg?style=for-the-badge
[stars-url]: https://github.com/Saganaki22/ComfyUI-VoxCPM2/stargazers
[issues-shield]: https://img.shields.io/github/issues/Saganaki22/ComfyUI-VoxCPM2.svg?style=for-the-badge
[issues-url]: https://github.com/Saganaki22/ComfyUI-VoxCPM2/issues
[forks-shield]: https://img.shields.io/github/forks/Saganaki22/ComfyUI-VoxCPM2.svg?style=for-the-badge
[forks-url]: https://github.com/Saganaki22/ComfyUI-VoxCPM2/network/members
