import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
import os
from tqdm import tqdm
from datasets import load_dataset
import string
import re

# Simple tokenization function using regular expressions
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation."""
    # Replace punctuation with spaces around them
    for punct in string.punctuation:
        text = text.replace(punct, f" {punct} ")
    # Split on whitespace
    return [token.strip() for token in text.split() if token.strip()]

class Vocabulary:
    def __init__(self, max_size=None):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_count = Counter()
        self.max_size = max_size
        self.size = 4  # Initial size with special tokens
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.size
            self.idx2word[self.size] = word
            self.size += 1
    
    def add_sentence(self, sentence):
        for word in sentence:
            self.word_count[word] += 1
    
    def build(self):
        # Create vocabulary with most common words
        if self.max_size:
            most_common = self.word_count.most_common(self.max_size - 4)  # -4 for special tokens
        else:
            most_common = self.word_count.most_common()
        
        for word, _ in most_common:
            self.add_word(word)
    
    def __len__(self):
        return self.size

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return " ".join([word for word in text.split() if word.lower() not in {"a", "an", "the"}])
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def find_answer_span(context_tokens, answer_tokens):
    """Find the start and end indices of the answer in the context."""
    n = len(context_tokens)
    m = len(answer_tokens)
    
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if context_tokens[i + j].lower() != answer_tokens[j].lower():
                match = False
                break
        if match:
            return i, i + m - 1
    
    return -1, -1  # Answer span not found

def load_squad_data(train_split_ratio=0.9, max_context_length=400, max_question_length=50, max_samples=500):
    """Load and preprocess the SQuAD dataset."""
    print("Loading SQuAD dataset...")
    squad_dataset = load_dataset("squad")
    
    train_data = []
    val_data = []
    
    # Process training data
    print("Processing training data...")
    train_examples = squad_dataset["train"]
    
    # Limit the number of samples for demo purposes
    if max_samples > 0:
        print(f"Using only {max_samples} samples for demonstration")
        train_examples = train_examples.select(range(min(max_samples, len(train_examples))))
    
    total_examples = len(train_examples)
    train_size = int(train_split_ratio * total_examples)
    
    # Build vocabulary
    vocab = Vocabulary()
    
    for i, example in enumerate(tqdm(train_examples)):
        context = example["context"]
        question = example["question"]
        answer_text = example["answers"]["text"][0]
        
        # Use simple tokenization
        context_tokens = simple_tokenize(context)
        question_tokens = simple_tokenize(question)
        answer_tokens = simple_tokenize(answer_text)
        
        # Truncate if needed
        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[:max_context_length]
        
        if len(question_tokens) > max_question_length:
            question_tokens = question_tokens[:max_question_length]
        
        # Find answer span
        answer_start, answer_end = find_answer_span(context_tokens, answer_tokens)
        
        # Skip examples where answer span is not found or out of bounds
        if answer_start == -1 or answer_end >= max_context_length:
            continue
        
        # Add words to vocabulary
        vocab.add_sentence(context_tokens)
        vocab.add_sentence(question_tokens)
        
        processed_example = {
            "id": example["id"],
            "context_tokens": context_tokens,
            "question_tokens": question_tokens,
            "answer_text": answer_text,
            "answer_tokens": answer_tokens,
            "answer_start": answer_start,
            "answer_end": answer_end
        }
        
        if i < train_size:
            train_data.append(processed_example)
        else:
            val_data.append(processed_example)
    
    # Build the vocabulary
    vocab.build()
    
    print(f"Processed {len(train_data)} training examples and {len(val_data)} validation examples")
    print(f"Vocabulary size: {len(vocab)}")
    
    return train_data, val_data, vocab

class SQuADDataset(Dataset):
    def __init__(self, data, vocab, max_context_length=400, max_question_length=50):
        self.data = data
        self.vocab = vocab
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Convert tokens to indices
        context_indices = [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) 
                          for token in example["context_tokens"]]
        question_indices = [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) 
                           for token in example["question_tokens"]]
        
        # Pad sequences
        context_length = len(context_indices)
        question_length = len(question_indices)
        
        padded_context = context_indices + [self.vocab.word2idx["<PAD>"]] * (self.max_context_length - context_length)
        padded_question = question_indices + [self.vocab.word2idx["<PAD>"]] * (self.max_question_length - question_length)
        
        return {
            "id": example["id"],
            "context": torch.tensor(padded_context),
            "context_length": context_length,
            "question": torch.tensor(padded_question),
            "question_length": question_length,
            "answer_text": example["answer_text"],
            "answer_start": example["answer_start"],
            "answer_end": example["answer_end"]
        }

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    batch_dict = {
        "id": [item["id"] for item in batch],
        "context": torch.stack([item["context"] for item in batch]),
        "context_length": torch.tensor([item["context_length"] for item in batch]),
        "question": torch.stack([item["question"] for item in batch]),
        "question_length": torch.tensor([item["question_length"] for item in batch]),
        "answer_text": [item["answer_text"] for item in batch],
        "answer_start": torch.tensor([item["answer_start"] for item in batch]),
        "answer_end": torch.tensor([item["answer_end"] for item in batch])
    }
    return batch_dict

def get_squad_dataloaders(batch_size=16, train_split_ratio=0.9, max_context_length=400, max_question_length=50, max_samples=500):
    """Create DataLoaders for SQuAD dataset."""
    train_data, val_data, vocab = load_squad_data(train_split_ratio, max_context_length, max_question_length, max_samples)
    
    train_dataset = SQuADDataset(train_data, vocab, max_context_length, max_question_length)
    val_dataset = SQuADDataset(val_data, vocab, max_context_length, max_question_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, vocab 