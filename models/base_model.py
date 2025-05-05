import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

class EncoderDecoderWithoutAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate=0.2, use_gru=False):
        super(EncoderDecoderWithoutAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        
        # Embedding layer for both passage and question
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder for passage (bidirectional)
        if use_gru:
            self.passage_encoder = nn.GRU(
                embedding_dim, 
                hidden_dim // 2,  # divided by 2 because bidirectional
                bidirectional=True,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0
            )
        else:
            self.passage_encoder = nn.LSTM(
                embedding_dim, 
                hidden_dim // 2,  # divided by 2 because bidirectional
                bidirectional=True,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0
            )
        
        # Encoder for question
        if use_gru:
            self.question_encoder = nn.GRU(
                embedding_dim, 
                hidden_dim,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0
            )
        else:
            self.question_encoder = nn.LSTM(
                embedding_dim, 
                hidden_dim,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers for predicting start and end positions
        self.start_output = nn.Linear(hidden_dim * 2, 1)  # hidden_dim * 2 because passage is bidirectional
        self.end_output = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, context, context_lengths, question, question_lengths):
        """
        Args:
            context: Tensor of shape (batch_size, max_context_length)
            context_lengths: Tensor of shape (batch_size,) containing actual lengths
            question: Tensor of shape (batch_size, max_question_length)
            question_lengths: Tensor of shape (batch_size,) containing actual lengths
        
        Returns:
            start_logits: Logits for start positions (batch_size, context_length)
            end_logits: Logits for end positions (batch_size, context_length)
        """
        batch_size = context.size(0)
        
        # Embed the context and question
        context_embedded = self.embedding(context)  # (batch_size, max_context_length, embedding_dim)
        question_embedded = self.embedding(question)  # (batch_size, max_question_length, embedding_dim)
        
        # Pack padded sequences for efficient processing
        packed_context = nn.utils.rnn.pack_padded_sequence(
            context_embedded, 
            context_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_question = nn.utils.rnn.pack_padded_sequence(
            question_embedded,
            question_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Encode the passage
        packed_context_output, context_hidden = self.passage_encoder(packed_context)
        context_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_context_output, 
            batch_first=True,
            total_length=context.size(1)  # Ensure output has the same length as input
        )  # (batch_size, max_context_length, hidden_dim)
        
        # Encode the question
        _, question_hidden = self.question_encoder(packed_question)
        
        # Process question hidden state
        if self.use_gru:
            # For GRU: question_hidden is (num_layers * num_directions, batch_size, hidden_dim)
            question_hidden = question_hidden.view(1, batch_size, self.hidden_dim)
            question_hidden = question_hidden.transpose(0, 1)  # (batch_size, 1, hidden_dim)
        else:
            # For LSTM: question_hidden is a tuple (h_n, c_n) where h_n is the hidden state
            # and c_n is the cell state, both of shape (num_layers * num_directions, batch_size, hidden_dim)
            h_n, _ = question_hidden
            question_hidden = h_n.view(1, batch_size, self.hidden_dim)
            question_hidden = question_hidden.transpose(0, 1)  # (batch_size, 1, hidden_dim)
        
        # Expand question hidden state to match context length for conditioning
        question_hidden_expanded = question_hidden.expand(-1, context_output.size(1), -1)
        
        # Concatenate question hidden state with context representations
        combined = torch.cat([context_output, question_hidden_expanded], dim=2)
        combined = self.dropout(combined)
        
        # Predict start and end positions
        start_logits = self.start_output(combined).squeeze(-1)  # (batch_size, max_context_length)
        end_logits = self.end_output(combined).squeeze(-1)  # (batch_size, max_context_length)
        
        # Mask out padding tokens
        mask = torch.arange(context.size(1), device=context.device).unsqueeze(0) >= context_lengths.unsqueeze(1)
        start_logits.masked_fill_(mask, float('-inf'))
        end_logits.masked_fill_(mask, float('-inf'))
        
        return start_logits, end_logits
    
    def predict(self, context, context_lengths, question, question_lengths):
        """
        Make predictions for the most likely answer span.
        
        Returns:
            predicted_start: Tensor of shape (batch_size,) with predicted start positions
            predicted_end: Tensor of shape (batch_size,) with predicted end positions
        """
        # Get logits
        with torch.no_grad():
            start_logits, end_logits = self.forward(context, context_lengths, question, question_lengths)
        
        # Get predictions
        batch_size = context.size(0)
        max_context_length = context.size(1)
        
        # For each position i, compute scores for all j >= i
        scores = torch.zeros(batch_size, max_context_length, max_context_length, device=context.device)
        
        for i in range(max_context_length):
            for j in range(i, min(i + 30, max_context_length)):  # Limit span to 30 tokens
                scores[:, i, j] = start_logits[:, i] + end_logits[:, j]
        
        # Find the best score
        scores = scores.view(batch_size, -1)
        best_idx = torch.argmax(scores, dim=1)
        
        # Convert flat index to 2D indices
        predicted_start = best_idx // max_context_length
        predicted_end = best_idx % max_context_length
        
        return predicted_start, predicted_end
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 