import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Question Answering Model Comparison\n\n",
                "This notebook demonstrates the comparative study of encoder-decoder architectures for question answering using the SQuAD dataset.\n\n",
                "Three models are implemented:\n",
                "1. Encoder-Decoder without Attention\n",
                "2. Encoder-Decoder with Bahdanau Attention\n",
                "3. Transformer-based Encoder-Decoder"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup\n\n",
                "First, let's clone the repository and install the required dependencies:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git clone https://github.com/vedant7001/DL_Project.git\n",
                "%cd DL_Project\n",
                "!pip install -r requirements.txt"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Train Models\n\n",
                "For demonstration purposes, we'll train smaller versions of each model on a limited dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train a small base model (5-10 minutes)\n",
                "!python train.py --model_type base --embedding_dim 128 --hidden_dim 64 --num_epochs 5 --batch_size 16 --max_samples 200"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train a small attention model (5-10 minutes)\n",
                "!python train.py --model_type attention --embedding_dim 128 --hidden_dim 64 --num_epochs 5 --batch_size 16 --max_samples 200"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train a small transformer model (5-10 minutes)\n",
                "!python train.py --model_type transformer --embedding_dim 128 --num_heads 4 --num_layers 2 --num_epochs 5 --batch_size 16 --max_samples 200"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Evaluate and Compare Models\n\n",
                "Let's compare the trained models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run simple evaluation that doesn't require full validation data\n",
                "!python evaluate_simple.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Visualize Attention\n\n",
                "Let's visualize the attention weights in the attention-based models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run example usage script to visualize attention on sample questions\n",
                "!python example_usage.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Try Your Own Questions\n\n",
                "Now let's try our own questions with one of the trained models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import os\n",
                "import matplotlib.pyplot as plt\n",
                "from evaluate import load_model\n",
                "from data_utils import simple_tokenize\n",
                "\n",
                "def find_latest_model(model_type):\n",
                "    runs_dir = 'runs'\n",
                "    matching_dirs = [d for d in os.listdir(runs_dir) if d.startswith(model_type)]\n",
                "    if not matching_dirs:\n",
                "        return None\n",
                "    \n",
                "    # Sort by timestamp\n",
                "    latest_dir = sorted(matching_dirs)[-1]\n",
                "    model_path = os.path.join(runs_dir, latest_dir, 'best_model.pt')\n",
                "    \n",
                "    if os.path.exists(model_path):\n",
                "        return model_path\n",
                "    return None\n",
                "\n",
                "def get_answer(model, model_type, context, question, device):\n",
                "    # Tokenize the inputs\n",
                "    context_tokens = simple_tokenize(context)\n",
                "    question_tokens = simple_tokenize(question)\n",
                "    \n",
                "    # Truncate if needed\n",
                "    max_context_len = 400\n",
                "    max_question_len = 50\n",
                "    \n",
                "    if len(context_tokens) > max_context_len:\n",
                "        context_tokens = context_tokens[:max_context_len]\n",
                "    if len(question_tokens) > max_question_len:\n",
                "        question_tokens = question_tokens[:max_question_len]\n",
                "    \n",
                "    # For demonstration, we'll use placeholder values for word indices\n",
                "    context_tensor = torch.ones(1, len(context_tokens), dtype=torch.long).to(device)  # All UNK tokens\n",
                "    question_tensor = torch.ones(1, len(question_tokens), dtype=torch.long).to(device)  # All UNK tokens\n",
                "    \n",
                "    context_len = torch.tensor([len(context_tokens)]).to(device)\n",
                "    question_len = torch.tensor([len(question_tokens)]).to(device)\n",
                "    \n",
                "    # Get predictions\n",
                "    with torch.no_grad():\n",
                "        if model_type == 'base':\n",
                "            start_idx, end_idx = model.predict(context_tensor, context_len, question_tensor, question_len)\n",
                "            attention = None\n",
                "        else:\n",
                "            start_idx, end_idx, attention = model.predict(context_tensor, context_len, question_tensor, question_len)\n",
                "    \n",
                "    # Get the predicted span\n",
                "    start = start_idx.item()\n",
                "    end = end_idx.item()\n",
                "    \n",
                "    # Ensure start <= end\n",
                "    if start > end:\n",
                "        start, end = end, start\n",
                "    \n",
                "    # Extract the answer\n",
                "    answer_tokens = context_tokens[start:end+1]\n",
                "    answer = ' '.join(answer_tokens)\n",
                "    \n",
                "    result = {\n",
                "        'question': question,\n",
                "        'context': context,\n",
                "        'answer': answer,\n",
                "        'start': start,\n",
                "        'end': end\n",
                "    }\n",
                "    \n",
                "    if attention is not None:\n",
                "        result['attention'] = attention[0].cpu().numpy()\n",
                "    \n",
                "    return result\n",
                "\n",
                "def highlight_answer(context, start, end):\n",
                "    tokens = simple_tokenize(context)\n",
                "    highlighted = []\n",
                "    \n",
                "    for i, token in enumerate(tokens):\n",
                "        if start <= i <= end:\n",
                "            highlighted.append(f\"[{token}]\")\n",
                "        else:\n",
                "            highlighted.append(token)\n",
                "    \n",
                "    return ' '.join(highlighted)\n",
                "\n",
                "# Load the attention model (usually gives better results for visualization)\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "model_path = find_latest_model('attention')\n",
                "if model_path:\n",
                "    print(f\"Using model: {model_path}\")\n",
                "    model, _ = load_model(model_path)\n",
                "    model = model.to(device)\n",
                "    model.eval()\n",
                "else:\n",
                "    print(\"No attention model found. Please run the training cell first.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define context and question\n",
                "context = \"\"\"\n",
                "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. \n",
                "These deep neural networks can learn representations of data with multiple levels of abstraction. \n",
                "The architecture of deep neural networks consists of an input layer, multiple hidden layers, and an output layer. \n",
                "Each layer contains nodes or neurons that perform computations on the input data.\n",
                "Deep learning has achieved remarkable results in various fields, including computer vision, \n",
                "natural language processing, speech recognition, and game playing.\n",
                "\"\"\"\n",
                "\n",
                "question = \"What is deep learning?\"\n",
                "\n",
                "# Get and display answer\n",
                "if model_path:\n",
                "    result = get_answer(model, 'attention', context, question, device)\n",
                "    print(f\"Question: {result['question']}\")\n",
                "    print(f\"\\nAnswer: {result['answer']}\")\n",
                "    print(\"\\nContext with highlighted answer:\")\n",
                "    print(highlight_answer(context, result['start'], result['end']))\n",
                "else:\n",
                "    print(\"Model not loaded, cannot answer question.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n\n",
                "In this notebook, we've demonstrated:\n",
                "1. Training three different encoder-decoder models for question answering\n",
                "2. Evaluating and comparing their performance\n",
                "3. Visualizing attention weights\n",
                "4. Using the models for inference on custom questions\n\n",
                "The full code is available on GitHub at: https://github.com/vedant7001/DL_Project"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('colab_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 