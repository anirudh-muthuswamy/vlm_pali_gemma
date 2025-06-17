
# PaliGemma Implementation from Scratch

This project implements Google's PaliGemma multimodal model from scratch in PyTorch, following Umar Jamil's video. PaliGemma combines a vision encoder (SigLIP) with a language model (Gemma) to understand both images and text, enabling tasks like image captioning, visual question answering, and more.

## Architecture Overview

The implementation consists of several key components that work together to process both visual and textual inputs:

### **Vision Components**

**SigLIP Vision Model** (`modeling_siglip.py`)

- `SiglipVisionEmbeddings`: Converts input images into patch embeddings using 2D convolution and adds positional encodings
- `SiglipAttention`: Multi-head self-attention mechanism for processing visual patches
- `SiglipMLP`: Feed-forward network with GELU activation
- `SiglipEncoderLayer`: Transformer encoder layer combining attention and MLP with residual connections
- `SiglipVisionTransformer`: Complete vision transformer that processes images into feature representations


### **Language Components**

**Gemma Language Model** (`modelling_gemma.py`)

- `GemmaRotaryEmbedding`: Implements Rotary Position Embedding (RoPE) for better positional understanding
- `GemmaAttention`: Multi-head attention with support for key-value caching and grouped-query attention
- `GemmaMLP`: Feed-forward network using SwiGLU activation (gate and up projections)
- `GemmaRMSNorm`: Root Mean Square Layer Normalization for stable training
- `GemmaDecoderLayer`: Complete transformer decoder layer with pre-normalization
- `GemmaModel`: Full language model combining embeddings, decoder layers, and normalization


### **Multimodal Integration**

**PaliGemma Model** (`modelling_gemma.py`)

- `PaliGemmaMultiModalProjector`: Projects vision features to language model dimensions
- `PaliGemmaForConditionalGeneration`: Main model that combines vision and language components
- `KVCache`: Efficient key-value caching for faster inference during text generation


### **Processing Pipeline**

**PaliGemma Processor** (`processing_paligemma.py`)

- Handles image preprocessing (resizing, normalization, rescaling)
- Manages tokenization and adds special image tokens to prompts
- Combines visual and textual inputs into model-ready format
- Supports object detection and segmentation tokens


## Key Features

- **Efficient Inference**: Implements KV-caching for faster autoregressive generation
- **Flexible Architecture**: Supports different model sizes through configurable parameters
- **Multimodal Processing**: Seamlessly integrates visual and textual information
- **Custom Tokenization**: Handles special tokens for images, object detection, and segmentation
- **Memory Optimization**: Uses grouped-query attention to reduce memory usage


## Setup Requirements

Before running the model, you need to:

1. **Clone the Hugging Face PaliGemma repository**:

```bash
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

2. **Download the model weights** from the Hugging Face repository. The model files should include:
    - Configuration files (`config.json`)
    - Model weights (`.safetensors` files)
    - Tokenizer files
3. **Install dependencies**:

```bash
pip install torch torchvision transformers safetensors pillow fire
```


## Running Inference

The project includes a convenient shell script for launching inference:

```bash
./launch_inference.sh
```

This script will execute the inference pipeline using the parameters defined in `inference.py`. You can also run inference directly:

```bash
python inference.py \
    --model_path /path/to/paligemma/model \
    --prompt "Describe this image" \
    --image_file_path /path/to/your/image.jpg \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample True
```


## Inference Parameters

- `model_path`: Path to the downloaded PaliGemma model directory
- `prompt`: Text prompt for the model
- `image_file_path`: Path to the input image
- `max_tokens_to_generate`: Maximum number of tokens to generate
- `temperature`: Controls randomness in generation (higher = more random)
- `top_p`: Nucleus sampling parameter
- `do_sample`: Whether to use sampling or greedy decoding
- `only_cpu`: Force CPU-only inference


## Model Architecture Details

The implementation follows the original PaliGemma architecture:

- **Vision Encoder**: SigLIP processes 224x224 images into patch embeddings
- **Projection Layer**: Maps vision features to language model dimensions
- **Language Model**: Gemma decoder processes combined visual and textual tokens
- **Generation**: Autoregressive text generation with KV-caching for efficiency

