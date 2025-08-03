# VLM AGENT  LLaVA 1.6 Mistral

This project is a **Vision-Language AI assistant** for analyzing traffic or surveillance videos. It extracts frames from a video and uses a quantized 4-bit version of [`llava-hf/llava-v1.6-mistral-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) to generate intelligent captions or answers to questions about each frame using RAG and LLM.

---

## 🚀 Features

- 🔍 Extracts frames from uploaded video at 1 FPS.
- 📸 Uses `LLaVA v1.6 Mistral 7B` (4-bit quantized) to understand each frame.
- 🧠 Ask custom questions like:
  - *"What violation is happening?"*
  - *"Is anyone wearing a helmet?"*
  - *"What vehicles are present?"*
- 🖼️ Outputs image + answer.
- ⚡ Runs efficiently with 4-bit quantization on a consumer GPU (16 GB+ VRAM).
- 🖥️ Simple Gradio interface (optional).

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/purushogit/VLM_AGENT.git
cd llava-vlm-captioner
pip3 install -r requirements.tyxt


```bash
python3 pipeline.py 

```bash
 Running on local URL:  http://127.0.0.1:7860
 
```bash
  Start using vlm to upload video and ask Q and A using gradio UI interface
  
  
  
#### Additional features can be added
Can make dockerisation
can generate id for every video uploaded and can store with  key reference value and can do Q and A

