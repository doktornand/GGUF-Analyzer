# GGUF Analyzer 🔬

> Deep inspection and visualization of GGUF model files — quantization analysis, architecture reconstruction, performance estimation, and interactive GUI.

GGUF Analyzer is a Python toolset for thoroughly inspecting `.gguf` files used by [llama.cpp](https://github.com/ggerganov/llama.cpp), Ollama, LM Studio, and compatible runtimes. It parses the binary format at a low level, reconstructs the model's architecture, audits quantization quality, and estimates inference performance — all without requiring any ML framework.

---

## ✨ Features

### 🧠 Architecture Reconstruction
- Reads GGUF v2/v3 headers (magic, version, tensor count, metadata KV pairs)
- Reconstructs model architecture from metadata: number of layers, hidden size, attention heads, KV heads, intermediate size, vocab size, context length, RoPE theta
- Validates and cross-checks metadata against actual tensor shapes

### 📦 Tensor Analysis
- Full tensor inventory: name, shape, GGML type, byte size, element count, quantization block layout
- Automatic role classification: `embedding`, `attention_q/k/v/output`, `mlp_gate/up/down`, `normalization`, `output_head`
- Outlier detection via IQR on tensor sizes
- Layer-by-layer parameter progression with mean/std
- Top largest tensors and shape distribution

### 🔢 Quantization Audit
- Supports all GGML types: F32, F16, Q2\_K through Q8\_K, IQ1\_S, IQ2\_XXS, IQ4\_XS, and more (30+ types)
- Per-type compression ratio and effective bits per weight
- Quality score per quantization type (F32 = 100 → IQ1\_S = 30)
- Mixed-precision analysis across layers
- Optimization opportunities: aggressive quantization, uniformization, embedding reduction, layer sharing

### 📐 Performance Estimation
- Estimated FLOPs per token
- Tokens-per-second estimation for CPU, consumer GPU, and datacenter GPU
- KV cache size estimation
- Hardware compatibility matrix: CPU, GPU (CUDA/Metal), mobile, edge

### 🔗 Compatibility Analysis
- Framework support: llama.cpp, GGML, Ollama, Transformers (via conversion), ONNX
- CPU/GPU acceleration readiness (AVX2, AVX-512, NEON, CUDA, Metal)
- Production/edge/cloud deployment readiness

### 📊 Quality Scoring
- Composite score (0–100) across: quantization quality, architecture coherence, optimization level
- Letter grade (A+ → F) per dimension and globally
- Actionable recommendation per score band

---

## 🖥️ Scripts

The repo contains three entry points:

| File | Description |
|---|---|
| `GGUF_analyzer_v3.py` | Core analysis engine — parse, analyze, report. Use this as a library or run standalone. |
| `GGUF_Inspector_Ultra_v3.py` | Full GUI application (Tkinter + Matplotlib). Interactive tabs, visualizations, export. |
| `GGUF_analyzer2a.py` | Alternate / experimental variant of the analyzer. |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/doktornand/GGUF-Analyzer.git
cd GGUF-Analyzer
```

### 2. Install dependencies

```bash
pip install numpy matplotlib
# Optional: for the modern UI theme
pip install customtkinter
# Optional: for topology graph
pip install networkx
```

> Python 3.8+ required. No ML framework (PyTorch, TensorFlow…) needed.

### 3. Run the GUI

```bash
python GGUF_Inspector_Ultra_v3.py
```

Then click **Parcourir…**, select a `.gguf` file, and click **Lancer l'analyse**.

### 4. Run the CLI analyzer

```bash
python GGUF_analyzer_v3.py path/to/model.gguf
```

---

## 🖼️ GUI Overview

The **GGUF Inspector Ultra** interface has four tabs:

- **📄 Rapport** — full text report with all analysis sections
- **📊 Visualisations** — six interactive Matplotlib charts (quantization distribution, tensor size histogram, layer parameter progression, quality scores, component breakdown, compression efficiency)
- **📋 Tenseurs** — sortable table of all tensors with type, shape, size, and quantization status
- **🔄 Comparaison** — side-by-side summary when multiple models have been loaded

**Export** options: JSON (full analysis), CSV (tensor table), Markdown (text report).

---

## 📊 Analysis Sections

### Structure
Header validation, GGUF version, tensor count, metadata count, quantization type distribution, total model size.

### Tensors
Layer classification, size distribution, largest tensors, shape analysis (1D/2D/3D), global compression ratio and space saved.

### Architecture
Reconstructed model parameters, parameter distribution by component (embedding / attention / MLP / normalization), layer-by-layer analysis, attention pattern breakdown, grouped-query attention (GQA) detection, quantization strategy per layer.

### Advanced Patterns
Mixed-precision per layer, memory layout gaps and fragmentation, optimization opportunities (ranked by priority), performance estimates per hardware tier, framework & hardware compatibility, composite quality assessment.

---

## 📁 Supported GGML Types

| Category | Types |
|---|---|
| Full precision | F32, F16, F64 |
| Legacy quants | Q4\_0, Q4\_1, Q5\_0, Q5\_1, Q8\_0, Q8\_1 |
| K-quants | Q2\_K, Q3\_K, Q4\_K, Q5\_K, Q6\_K, Q8\_K |
| i-quants | IQ1\_S, IQ1\_M, IQ2\_XXS, IQ2\_XS, IQ2\_S, IQ3\_XXS, IQ3\_S, IQ4\_NL, IQ4\_XS |
| Integer | I8, I16, I32, I64 |

---

## 🛠️ Tech Stack

- **Python 3.8+** — zero ML framework dependency
- [NumPy](https://numpy.org/) — safe math utilities (division, mean, std, percentile)
- [Matplotlib](https://matplotlib.org/) — embedded charts in the GUI
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — GUI framework (stdlib)
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter) *(optional)* — modern UI theme
- [NetworkX](https://networkx.org/) *(optional)* — model topology graph

---

## 📄 Related Resource

The repo includes `GGUF-Internals-GPT.pdf` — a reference document on the GGUF binary format internals, useful for understanding the parsing logic.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

**doktornand** — [github.com/doktornand](https://github.com/doktornand)

---

⭐ If this tool is useful to you, a star on the repo is appreciated!
