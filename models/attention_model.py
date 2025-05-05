import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        
        # Attention mechanism using Bahdanau style (additive attention)
        self.W_question = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, context_hidden, question_hidden, mask=None):
        """
        Calculate attention weights using Bahdanau's additive style attention.
        
        Args:
            context_hidden: Context representations (batch_size, context_len, hidden_dim)
            question_hidden: Question representation (batch_size, 1, hidden_dim)
            mask: Mask for padding tokens (batch_size, context_len)
            
        Returns:
            attention_weights: Attention weights (batch_size, context_len)
            context_vector: Context vector after applying attention (batch_size, hidden_dim)
        """
        # Transform hidden states
        question_transformed = self.W_question(question_hidden)  # (batch_size, 1, hidden_dim)
        context_transformed = self.W_context(context_hidden)     # (batch_size, context_len, hidden_dim)
        
        # Calculate scores
        # question_transformed is (batch_size, 1, hidden_dim)
        # context_transformed is (batch_size, context_len, hidden_dim)
        # We add them with broadcasting to get (batch_size, context_len, hidden_dim)
        scores = self.v(torch.tanh(question_transformed + context_transformed))
        
        # Attention weights (batch_size, context_len)
        attention_weights = F.softmax(scores.squeeze(-1), dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            # Instead of using masked_fill_ which is an in-place operation
            # Create a new tensor with zeros at masked positions
            masked_weights = attention_weights * (~mask).float()
            # Renormalize weights if mask is applied
            attention_sum = masked_weights.sum(dim=1, keepdim=True)
            attention_weights = masked_weights / (attention_sum + 1e-8)
        
        # Create context vector as weighted sum of context hidden states
        context_vector = torch.bmm(attention_weights.unsqueeze(1), context_hidden).squeeze(1)
        
        return attention_weights, context_vector

class EncoderDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate=0.2, use_gru=False):
        super(EncoderDecoderWithAttention, self).__init__()
        
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
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers for predicting start and end positions
        self.start_output = nn.Linear(hidden_dim * 2, 1)  # hidden_dim * 2 because we concatenate context and question info
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
            attention_weights: Attention weights for visualization (batch_size, context_length)
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
        packed_question_output, question_hidden = self.question_encoder(packed_question)
        question_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_question_output,
            batch_first=True,
            total_length=question.size(1)  # Ensure output has the same length as input
        )  # (batch_size, max_question_length, hidden_dim)
        
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
        
        # Create attention mask for padding tokens
        mask = torch.arange(context.size(1), device=context.device).unsqueeze(0) >= context_lengths.unsqueeze(1)
        
        # Apply attention between context and question
        attention_weights, context_vector = self.attention(context_output, question_hidden, mask)
        
        # For each position in context, create an enhanced representation using attention
        # Expand context_vector to have the same sequence length as context_output
        context_vector_expanded = context_vector.unsqueeze(1).expand(-1, context_output.size(1), -1)
        
        # Concatenate context_output with context_vector_expanded
        combined = torch.cat([context_output, context_vector_expanded], dim=2)
        combined = self.dropout(combined)
        
        # Predict start and end positions
        start_logits = self.start_output(combined).squeeze(-1)  # (batch_size, max_context_length)
        end_logits = self.end_output(combined).squeeze(-1)  # (batch_size, max_context_length)
        
        # Mask out padding tokens
        start_logits.masked_fill_(mask, float('-inf'))
        end_logits.masked_fill_(mask, float('-inf'))
        
        return start_logits, end_logits, attention_weights
    
    def predict(self, context, context_lengths, question, question_lengths):
        """
        Make predictions for the most likely answer span.
        
        Returns:
            predicted_start: Tensor of shape (batch_size,) with predicted start positions
            predicted_end: Tensor of shape (batch_size,) with predicted end positions
            attention_weights: Attention weights for visualization
        """
        # Get logits
        with torch.no_grad():
            start_logits, end_logits, attention_weights = self.forward(
                context, context_lengths, question, question_lengths
            )
        
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
        
        return predicted_start, predicted_end, attention_weights
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 