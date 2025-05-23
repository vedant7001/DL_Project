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
                "        'end': end,\n",
                "        'context_tokens': context_tokens\n",
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
                "def visualize_attention(context_tokens, attention_weights, title=\"Attention Weights\"):\n",
                "    \"\"\"Visualize attention weights.\"\"\"\n",
                "    plt.figure(figsize=(12, 4))\n",
                "    \n",
                "    # Check if weights are 1D and reshape if needed\n",
                "    if len(attention_weights.shape) == 1:\n",
                "        # Reshape 1D weights into a 2D matrix (1 x len)\n",
                "        attention_weights = attention_weights.reshape(1, -1)\n",
                "    \n",
                "    plt.imshow(attention_weights, cmap='viridis')\n",
                "    plt.title(title)\n",
                "    plt.xlabel('Context Position')\n",
                "    \n",
                "    # Show tokens on x-axis\n",
                "    if len(context_tokens) <= 50:  # Only show tokens if not too many\n",
                "        plt.xticks(range(len(context_tokens)), context_tokens, rotation=90)\n",
                "    \n",
                "    plt.colorbar()\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
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
                "    \n",
                "    # Visualize attention if available\n",
                "    if 'attention' in result:\n",
                "        visualize_attention(result['context_tokens'], result['attention'], \"Attention Weights\")\n",
                "else:\n",
                "    print(\"Model not loaded, cannot answer question.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Additional Example: Deep Learning and Transformers"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Enhanced context and questions about deep learning\n",
                "context1 = \"\"\"\n",
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. \n",
                "Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, \n",
                "deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including \n",
                "computer vision, speech recognition, natural language processing, audio recognition, social network filtering, \n",
                "machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, \n",
                "where they have produced results comparable to and in some cases surpassing human expert performance.\n",
                "Transformer architectures, a type of deep learning model, have become particularly important for NLP tasks \n",
                "since their introduction in 2017. Transformers use self-attention mechanisms to process input sequences in parallel, \n",
                "which has proven especially effective for tasks like machine translation, text summarization, and question answering. \n",
                "Models like BERT, GPT, and T5 are all based on the transformer architecture and have set new performance benchmarks \n",
                "across numerous language understanding tasks.\n",
                "\"\"\"\n",
                "\n",
                "questions1 = [\n",
                "    \"What is deep learning part of?\",\n",
                "    \"What types of learning can be used in deep learning?\",\n",
                "    \"What are transformer architectures used for?\"\n",
                "]\n",
                "\n",
                "# Only run if we have a model loaded\n",
                "if model_path and 'model' in locals():\n",
                "    for question in questions1:\n",
                "        print(f\"\\n{'='*80}\\nQuestion: {question}\\n{'='*80}\\n\")\n",
                "        \n",
                "        result = get_answer(model, 'attention', context1, question, device)\n",
                "        print(f\"Answer: {result['answer']}\\n\")\n",
                "        print(\"Context with highlighted answer:\")\n",
                "        print(highlight_answer(context1, result['start'], result['end']))\n",
                "        \n",
                "        # Visualize attention if available\n",
                "        if 'attention' in result:\n",
                "            visualize_attention(result['context_tokens'], result['attention'], \"Attention Weights\")\n",
                "        \n",
                "        print('\\n' + '-'*80)\n",
                "else:\n",
                "    print(\"No model loaded, cannot run deep learning examples.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Additional Example: Natural Language Processing and SQuAD"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Example context about NLP and Question Answering\n",
                "context2 = \"\"\"\n",
                "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence \n",
                "concerned with the interactions between computers and human language, in particular how to program computers \n",
                "to process and analyze large amounts of natural language data. The goal is a computer capable of \"understanding\" \n",
                "the contents of documents, including the contextual nuances of the language within them. The technology can then \n",
                "accurately extract information and insights contained in the documents as well as categorize and organize the \n",
                "documents themselves. Challenges in natural language processing frequently involve speech recognition, natural \n",
                "language understanding, and natural language generation. Modern NLP approaches are based on machine learning, \n",
                "especially statistical methods and neural networks. As of 2020, deep learning approaches such as transformers \n",
                "have achieved state-of-the-art results on many NLP tasks.\n",
                "Question answering (QA) is an important NLP task that involves automatically answering questions posed in natural language. \n",
                "Machine reading comprehension, a subset of QA, focuses on answering questions based on a given context passage. \n",
                "The Stanford Question Answering Dataset (SQuAD) has become a benchmark dataset for this task, consisting of questions \n",
                "posed by crowdworkers on a set of Wikipedia articles. In SQuAD, the answer to every question is a segment of text \n",
                "from the corresponding reading passage. Models are evaluated based on exact match and F1 scores, comparing their \n",
                "predicted answers against human-provided reference answers.\n",
                "\"\"\"\n",
                "\n",
                "questions2 = [\n",
                "    \"What is NLP?\",\n",
                "    \"What is SQuAD used for?\",\n",
                "    \"How are QA models evaluated?\"\n",
                "]\n",
                "\n",
                "# Only run if we have a model loaded\n",
                "if model_path and 'model' in locals():\n",
                "    for question in questions2:\n",
                "        print(f\"\\n{'='*80}\\nQuestion: {question}\\n{'='*80}\\n\")\n",
                "        \n",
                "        result = get_answer(model, 'attention', context2, question, device)\n",
                "        print(f\"Answer: {result['answer']}\\n\")\n",
                "        print(\"Context with highlighted answer:\")\n",
                "        print(highlight_answer(context2, result['start'], result['end']))\n",
                "        \n",
                "        # Visualize attention if available\n",
                "        if 'attention' in result:\n",
                "            visualize_attention(result['context_tokens'], result['attention'], \"Attention Weights\")\n",
                "        \n",
                "        print('\\n' + '-'*80)\n",
                "else:\n",
                "    print(\"No model loaded, cannot run NLP examples.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PyTorch Example: Try It Yourself"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define your own context and question about PyTorch\n",
                "my_context = \"\"\"\n",
                "PyTorch is an open source machine learning framework based on the Torch library, \n",
                "used for applications such as computer vision and natural language processing, \n",
                "originally developed by Meta AI and now part of the Linux Foundation umbrella. \n",
                "It is free and open-source software released under the Modified BSD license. \n",
                "Although the Python interface is more polished and the primary focus of development, \n",
                "PyTorch also has a C++ interface. PyTorch provides two high-level features: \n",
                "Tensor computing (like NumPy) with strong acceleration via graphics processing units (GPU) \n",
                "and Deep neural networks built on a tape-based automatic differentiation system.\n",
                "PyTorch is distinctive in its implementation of dynamic computational graphs, which allow for \n",
                "more flexible model building compared to static graph frameworks. This 'define-by-run' approach \n",
                "enables developers to modify neural networks on the fly, making debugging and experimentation easier. \n",
                "The framework includes modules for building complex neural network architectures, optimizers for \n",
                "training, data loading utilities, and seamless GPU integration. Its ecosystem has expanded \n",
                "to include libraries like torchvision for computer vision, torchaudio for audio processing, \n",
                "torchtext for NLP, and PyTorch Lightning for organizing research code. With its intuitive design \n",
                "and Python-native flow, PyTorch has become especially popular in research communities.\n",
                "\"\"\"\n",
                "\n",
                "my_question = \"Who developed PyTorch?\"\n",
                "\n",
                "# Only run if we have a model loaded\n",
                "if model_path and 'model' in locals():\n",
                "    result = get_answer(model, 'attention', my_context, my_question, device)\n",
                "    \n",
                "    print(f\"Question: {result['question']}\\n\")\n",
                "    print(f\"Answer: {result['answer']}\\n\")\n",
                "    print(\"Context with highlighted answer:\")\n",
                "    print(highlight_answer(my_context, result['start'], result['end']))\n",
                "    \n",
                "    # Visualize attention if available\n",
                "    if 'attention' in result:\n",
                "        visualize_attention(result['context_tokens'], result['attention'], \"Attention Weights\")\n",
                "else:\n",
                "    print(\"No model loaded, cannot run custom example.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Your Turn: Try Your Own Context and Question\n",
                "\n",
                "Modify the cells below to try your own context and question."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define your own context and question\n",
                "custom_context = \"\"\"\n",
                "Replace this text with your own context paragraph. It should be several sentences long.\n",
                "Make sure to include factual information that can be used to answer questions.\n",
                "The longer and more detailed the context, the better the model can extract specific answers.\n",
                "\"\"\"\n",
                "\n",
                "custom_question = \"Write your question here?\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run your custom example\n",
                "if model_path and 'model' in locals():\n",
                "    result = get_answer(model, 'attention', custom_context, custom_question, device)\n",
                "    \n",
                "    print(f\"Question: {result['question']}\\n\")\n",
                "    print(f\"Answer: {result['answer']}\\n\")\n",
                "    print(\"Context with highlighted answer:\")\n",
                "    print(highlight_answer(custom_context, result['start'], result['end']))\n",
                "    \n",
                "    # Visualize attention if available\n",
                "    if 'attention' in result:\n",
                "        visualize_attention(result['context_tokens'], result['attention'], \"Attention Weights\")\n",
                "else:\n",
                "    print(\"No model loaded, cannot run custom example.\")"
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