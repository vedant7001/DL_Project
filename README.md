# Question Answering Model Comparison

This project implements and compares three different encoder-decoder architectures for question answering on the SQuAD v1.1 dataset:

1. Encoder-Decoder without Attention (LSTM/GRU)
2. Encoder-Decoder with Bahdanau Attention
3. Transformer-based Encoder-Decoder with Self-Attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vedant7001/DL_Project/blob/main/colab_demo.ipynb)

## Project Structure

```
.
├── data_utils.py            # Data loading and processing utilities
├── models/                  # Model implementations
│   ├── base_model.py        # Encoder-Decoder without Attention
│   ├── attention_model.py   # Encoder-Decoder with Bahdanau Attention
│   ├── transformer_model.py # Transformer-based Encoder-Decoder
├── train.py                 # Training script for all models
├── evaluate.py              # Evaluation and comparison script
├── evaluate_simple.py       # Simple model comparison script
├── example_usage.py         # Example script for using trained models
├── colab_demo.ipynb         # Google Colab demonstration notebook
├── runs/                    # Directory for training outputs
└── comparison_results/      # Directory for model comparison results
```

## Requirements

The project requires the following packages:

```
torch>=1.10.0
torchvision>=0.11.1
transformers>=4.10.0
datasets>=1.11.0
nltk>=3.6.5
tqdm>=4.62.3
matplotlib>=3.5.0
pandas>=1.3.4
scikit-learn>=1.0.1
tensorboard>=2.7.0
numpy>=1.21.4
prettytable>=2.0.0
```

Install them using:

```bash
pip install -r requirements.txt
```

## Google Colab Usage

You can run this project directly in Google Colab:

1. Click the "Open in Colab" badge at the top of this README
2. Run the cells in the notebook to install dependencies, download the code, and run the models

## Local Setup

### Clone the repository

```bash
git clone https://github.com/vedant7001/DL_Project.git
cd DL_Project
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Training Models

To train a model, use the `train.py` script with appropriate arguments:

```bash
# Train Encoder-Decoder without Attention
python train.py --model_type base --embedding_dim 300 --hidden_dim 256 --num_epochs 20 --batch_size 32

# Train Encoder-Decoder with Bahdanau Attention
python train.py --model_type attention --embedding_dim 300 --hidden_dim 256 --num_epochs 20 --batch_size 32

# Train Transformer-based Encoder-Decoder
python train.py --model_type transformer --embedding_dim 256 --num_heads 8 --num_layers 3 --num_epochs 20 --batch_size 32
```

For a quicker demonstration with a smaller dataset:

```bash
python train.py --model_type base --embedding_dim 128 --hidden_dim 64 --num_epochs 1 --batch_size 16 --max_samples 100
```

### Training Options

- `--model_type`: Model architecture (`base`, `attention`, or `transformer`)
- `--embedding_dim`: Dimension of word embeddings
- `--hidden_dim`: Dimension of hidden state in RNNs
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--dropout`: Dropout rate
- `--use_gru`: Use GRU instead of LSTM (for base and attention models)
- `--num_heads`: Number of attention heads (for transformer model)
- `--num_layers`: Number of layers (for transformer model)
- `--max_samples`: Maximum number of samples to use (default: all samples)

## Evaluating and Comparing Models

To compare trained models, use the `evaluate_simple.py` script which doesn't require complete validation data:

```bash
python evaluate_simple.py
```

This will:
1. Find the latest trained models for each architecture type
2. Measure inference time
3. Count parameters
4. Generate comparison visualizations

For a more detailed evaluation using the validation set:
```bash
python evaluate.py --model_paths runs/base_*/best_model.pt runs/attention_*/best_model.pt runs/transformer_*/best_model.pt --output_dir comparison_results
```

## Using Trained Models

To use the trained models for inference, use the `example_usage.py` script:

```bash
python example_usage.py
```

This will load the latest trained models and test them on example questions.

## Model Architectures

### 1. Encoder-Decoder without Attention

Basic model with:
- Bidirectional LSTM/GRU for passage encoding
- LSTM/GRU for question encoding
- Output layers to predict start and end positions

### 2. Encoder-Decoder with Bahdanau Attention

Enhances the base model with:
- Attention calculation between passage and question
- Context-aware representations
- Visualization of attention weights

### 3. Transformer-based Encoder-Decoder

Uses transformer architecture with:
- Self-attention for both passage and question encoding
- Cross-attention between passage and question
- Position encoding
- Multi-head attention

## Results

The comparison results will be saved in the `comparison_results` directory, including:
- Number of model parameters
- Inference time measurements
- Visualizations of model sizes and inference times 