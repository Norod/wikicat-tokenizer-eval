# WikiCat Multilingual Tokenizer Analysis

A comprehensive analysis of 12 modern LLM tokenizers across 204 languages using the [WikiCat Multilingual dataset](https://huggingface.co/datasets/Norod78/WikiCat-Multilingual).

## 📊 Quick Results

**🏆 Winner: GPT-OSS**
- Wins in 95 languages (46.6%)
- Highest average WPT: 0.345
- Best for multilingual applications

**📈 Interactive Results:** [View Slideshow](https://norod78.github.io/wikicat-tokenizer-eval/tokenizer-slideshow.html)

## 🎯 What This Analysis Does

This project evaluates tokenizer efficiency across 272 languages by measuring:
- **WPT (Words Per Token)**: Higher = more efficient (fewer tokens needed)
- **CPT (Characters Per Token)**: Higher = better compression
- **Language family performance**: Which tokenizers excel at specific linguistic groups

## 🚀 Quick Start

### Prerequisites

1. **Python 3.13+** with `uv` package manager
2. **Gated Model Access**: You must accept terms for these models on HuggingFace:
   - [Meta Llama 4](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) (Meta License)
   - [Google Gemma 3](https://huggingface.co/google/gemma-3-4b-it) (Gemma Terms of Use)
   - [Cohere Aya-Expanse](https://huggingface.co/CohereForAI/aya-expanse-8b) (CC-BY-NC-4.0)
   
3. **HuggingFace Authentication**:
   ```bash
   huggingface-cli login
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/Norod78/wikicat-tokenizer-eval.git
cd wikicat-tokenizer-eval

# Install dependencies (using uv)
uv sync
```

### Running the Analysis

```bash
# Run tokenizer evaluation (takes ~30-60 minutes)
uv run python multilingual-tokenizer-wikicat-eval.py

# Analyze results
uv run python analyze-tokenizer-stats.py --min-words 100
```

The evaluation script:
- Downloads the WikiCat dataset automatically
- Processes 272 languages
- Supports resumability (can stop and restart)
- Outputs to `wikicat-tokenizer-eval.jsonl`

## 📁 Repository Structure

```
.
├── README.md                                    # This file
├── pyproject.toml                               # Python dependencies
├── multilingual-tokenizer-wikicat-eval.py      # Main evaluation script
├── analyze-tokenizer-stats.py                   # Analysis and reporting script
├── tokenizer-slideshow.html                     # Interactive results presentation
├── wikicat-tokenizer-eval.jsonl                # Evaluation results (generated)
└── wikipedia_cat-272_languages-utf8-with-text.jsonl  # Dataset (auto-downloaded)
```

## 🔬 Tokenizers Analyzed

| Tokenizer | Model ID | Gated? |
|-----------|----------|--------|
| GPT-2 | [`gpt2`](https://huggingface.co/gpt2) | No |
| Gemma-3 | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | ⚠️ Yes |
| SmolLM3 | [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | No |
| GPT-OSS | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | No |
| Kimi-K2 | [`moonshotai/Kimi-K2-Instruct`](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | No |
| Aya-Expanse | [`CohereForAI/aya-expanse-8b`](https://huggingface.co/CohereForAI/aya-expanse-8b) | ⚠️ Yes |
| Qwen-3-VL | [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | No |
| DeepSeek-3.2 | [`deepseek-ai/DeepSeek-V3.2-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) | No |
| Nemotron-v2 | [`nvidia/NVIDIA-Nemotron-Nano-12B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2) | No |
| Llama-4 | [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | ⚠️ Yes |
| MiniMax-M2 | [`MiniMaxAI/MiniMax-M2`](https://huggingface.co/MiniMaxAI/MiniMax-M2) | No |
| Granite-4 | [`ibm-granite/granite-4.0-h-1b`](https://huggingface.co/ibm-granite/granite-4.0-h-1b) | No |

## 📊 Key Findings

### Overall Performance
1. **GPT-OSS**: Best overall (0.345 avg WPT, 46.6% wins)
2. **Llama-4**: Strong multilingual (0.338 avg WPT, 19.1% wins)
3. **Gemma-3**: Excellent for Indic languages (0.326 avg WPT, 9.8% wins)

### Language Family Champions
- **Slavic** (70%): Llama-4
- **Germanic** (75%): GPT-OSS
- **Romance** (62.5%): MiniMax-M2
- **Indo-Aryan** (50%): Gemma-3
- **Semitic (Arabic)** (100%): MiniMax-M2
- **Semitic (Hebrew)** (100%): GPT-OSS
- **Japanese** (100%): MiniMax-M2
- **Korean** (100%): Llama-4

### The Legacy Problem
- **GPT-2** is worst in 190/204 languages (93%)
- Trained on English-only data
- Russian WPT: 0.134 (vs. 0.350 for Llama-4)

## 🎨 Viewing Results

### Interactive Slideshow
Open `tokenizer-slideshow.html` in any modern browser for an interactive presentation of results.

**Or view it online:** https://norod78.github.io/wikicat-tokenizer-eval/tokenizer-slideshow.html

### Command-Line Analysis
```bash
# Analyze with different word count thresholds
uv run python analyze-tokenizer-stats.py --min-words 50
uv run python analyze-tokenizer-stats.py --min-words 200

# Specify custom input file
uv run python analyze-tokenizer-stats.py --input my-eval.jsonl
```

## 📖 Understanding the Metrics

**WPT (Words Per Token)**: 
- Measures tokenizer efficiency
- Higher is better (fewer tokens needed)
- Example: WPT of 0.5 means 2 tokens per word on average

**CPT (Characters Per Token)**:
- Measures compression ratio
- Higher is better (more characters per token)
- Useful for estimating API costs

## 🌍 Dataset

This analysis uses the [WikiCat Multilingual dataset](https://huggingface.co/datasets/Norod78/WikiCat-Multilingual), which contains Wikipedia articles about "cats" in 272 languages.

The dataset provides a consistent, real-world text sample across diverse writing systems and language families.

## 🛠️ Advanced Usage

### Customizing Tokenizers

Edit `multilingual-tokenizer-wikicat-eval.py` to add/remove tokenizers:

```python
tokenizers = [
    {"name": "My-Tokenizer", "id": "organization/model-name"},
    # ... add more
]
```

### Custom Analysis

The analysis script supports linguistic family groupings. Edit `analyze-tokenizer-stats.py` to modify language categorization.

## 🤝 Contributing

Found an issue or want to add a tokenizer? Feel free to open an issue or submit a pull request!

## 📝 License

This analysis code is released under the MIT License. The dataset follows its own [license terms](https://huggingface.co/datasets/Norod78/WikiCat-Multilingual).

## 🙏 Acknowledgments

- Dataset: [Norod78/WikiCat-Multilingual](https://huggingface.co/datasets/Norod78/WikiCat-Multilingual)
- All tokenizer creators and maintainers
- HuggingFace for hosting and infrastructure

---

**Made with ❤️ for the multilingual NLP community**
