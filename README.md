# ComfyUI_RH_SoulX-Singer

![License](https://img.shields.io/badge/License-Apache%202.0-green)

A ComfyUI custom node for **[SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)** — a high-fidelity, zero-shot singing voice synthesis (SVS) model. This node enables users to generate realistic singing voices for unseen singers directly within ComfyUI, supporting both melody-conditioned (F0 contour) and score-conditioned (MIDI notes) control.

<p align="center">
  <a href="https://github.com/Soul-AILab/SoulX-Singer">Original Project</a> |
  <a href="https://huggingface.co/spaces/Soul-AILab/SoulX-Singer">HF Demo</a> |
  <a href="https://arxiv.org/abs/2602.07803">Paper</a>
</p>

## ✨ Features

- **Zero-Shot Singing Voice Synthesis** — Generate high-fidelity singing voices for unseen singers without fine-tuning
- **Flexible Control Modes** — Melody (F0 contour) and Score (MIDI notes) conditioning
- **Multi-Language Support** — Mandarin, English, and Cantonese
- **Full Pipeline** — Includes audio preprocessing (vocal separation, F0 extraction, lyric/note transcription) and SVS inference
- **Timbre Cloning** — Preserve singer identity across languages and styles

## 🛠️ Installation

### Method 1: ComfyUI Manager (Recommended)

Search for `ComfyUI_RH_SoulX-Singer` in ComfyUI Manager and install.

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_SoulX-Singer.git
cd ComfyUI_RH_SoulX-Singer
pip install -r preprocess/requirements.txt
```

## 📦 Model Download & Installation

All models must be placed in `ComfyUI/models/Soul-AILab/` with the following structure:

```
ComfyUI/
└── models/
    └── Soul-AILab/
        ├── SoulX-Singer/                        # SVS model
        │   └── model.pt
        └── SoulX-Singer-Preprocess/             # Preprocessing models
            ├── mel-band-roformer-karaoke/        # Vocal separation
            ├── dereverb_mel_band_roformer/       # Dereverberation
            ├── rmvpe/                            # F0 extraction
            ├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/  # Chinese ASR
            ├── parakeet-tdt-0.6b-v2/             # English ASR
            └── rosvot/                           # Note transcription
                ├── rosvot/model.pt
                └── rwbd/model.pt
```

### Download Methods

#### Method 1: Download from HuggingFace (Recommended)

```bash
pip install -U huggingface_hub

# Download the SoulX-Singer SVS model
huggingface-cli download Soul-AILab/SoulX-Singer --local-dir ComfyUI/models/Soul-AILab/SoulX-Singer

# Download preprocessing models
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir ComfyUI/models/Soul-AILab/SoulX-Singer-Preprocess
```

#### Method 2: Download from ModelScope (For China users)

```bash
pip install modelscope

# Download the SoulX-Singer SVS model
modelscope download --model Soul-AILab/SoulX-Singer --local_dir ComfyUI/models/Soul-AILab/SoulX-Singer

# Download preprocessing models
modelscope download --model Soul-AILab/SoulX-Singer-Preprocess --local_dir ComfyUI/models/Soul-AILab/SoulX-Singer-Preprocess
```

## 🚀 Usage

### Workflow

The typical workflow consists of 4 nodes:

1. **Load Preprocess Pipeline** → Loads the audio preprocessing models
2. **Preprocess Audio** → Extracts vocal, F0, lyrics, and notes from audio
3. **Load SVS Model** → Loads the SoulX-Singer synthesis model
4. **Generate Singing Voice** → Synthesizes singing voice with timbre cloning

### Basic Steps

1. Connect a **prompt audio** (reference singer voice for timbre) and a **target audio** (the song to be synthesized)
2. Preprocess both audios to extract metadata
3. Load the SVS model
4. Generate the singing voice with melody or score control

## 📝 Node Reference

### RunningHub SoulX-Singer Preprocess Pipeline

Loads all preprocessing models (vocal separation, F0 extraction, ASR, note transcription).

| Output | Type | Description |
|--------|------|-------------|
| pipeline | SoulXSinger_Preprocess_Pipeline | Loaded preprocessing pipeline |

### RunningHub SoulX-Singer Preprocessor

Processes audio to extract singing metadata.

| Input | Type | Description |
|-------|------|-------------|
| pipeline | SoulXSinger_Preprocess_Pipeline | Preprocessing pipeline |
| audio | AUDIO | Input audio |
| max_merge_duration | INT | Max segment duration in ms (10000–60000, default: 30000) |
| language | STRING | Language: Mandarin / English / Cantonese |

| Output | Type | Description |
|--------|------|-------------|
| audio metadata | SoulXSinger_Audio_Metadata | Extracted singing metadata (JSON) |

### RunningHub SoulX-Singer SVS Loader

Loads the SoulX-Singer SVS model and data processor.

| Output | Type | Description |
|--------|------|-------------|
| pipeline | SoulXSinger_SVS_Pipeline | Loaded SVS model and processor |

### RunningHub SoulX-Singer SVS Processor

Generates singing voice from metadata.

| Input | Type | Description |
|-------|------|-------------|
| pipeline | SoulXSinger_SVS_Pipeline | Loaded SVS pipeline |
| control | STRING | Control mode: melody / score |
| prompt_wav | AUDIO | Reference singer audio (timbre source) |
| prompt_metadata | SoulXSinger_Audio_Metadata | Prompt audio metadata |
| target_metadata | SoulXSinger_Audio_Metadata | Target audio metadata |
| seed | INT | Random seed (default: 12306) |

| Output | Type | Description |
|--------|------|-------------|
| audio | AUDIO | Generated singing voice audio |

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🔗 Links

- **Original Project**: [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)
- **Online Demo**: [HuggingFace Space](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer)
- **Paper**: [arXiv:2602.07803](https://arxiv.org/abs/2602.07803)
- **SVS Model**: [HuggingFace](https://huggingface.co/Soul-AILab/SoulX-Singer)
- **Preprocess Models**: [HuggingFace](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess)
- **RunningHub**: [www.runninghub.cn](https://www.runninghub.cn)

## 🙏 Acknowledgements

This project is based on [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), developed by [Soul-AILab](https://github.com/Soul-AILab).

We also thank the following open-source projects:

- [F5-TTS](https://github.com/SWivid/F5-TTS) — Text-to-Speech framework
- [Amphion](https://github.com/open-mmlab/Amphion) — Audio/Music/Speech toolkit
- [Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — Vocal separation
- [RMVPE](https://github.com/Dream-High/RMVPE) — F0 extraction
- [ROSVOT](https://github.com/RickyL-2000/ROSVOT) — Note transcription
- [Paraformer](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) — Chinese ASR
- [Parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — English ASR
