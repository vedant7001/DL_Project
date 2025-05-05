import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (won't be considered model parameters)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings (batch_size, seq_length, embedding_dim)
            
        Returns:
            Embeddings + positional encodings (batch_size, seq_length, embedding_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor (batch_size, query_len, d_model)
            key: Key tensor (batch_size, key_len, d_model)
            value: Value tensor (batch_size, value_len, d_model)
            mask: Optional mask (batch_size, query_len, key_len)
            
        Returns:
            output: Output tensor (batch_size, query_len, d_model)
            attention: Attention weights (batch_size, num_heads, query_len, key_len)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for the multi-head dimension
            if mask.dim() == 3:  # (batch_size, query_len, key_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, query_len, key_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        output = torch.matmul(attention, v)
        
        # Reshape and concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            
        Returns:
            Output tensor (batch_size, seq_length, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            mask: Optional mask (batch_size, seq_length)
            
        Returns:
            Output tensor (batch_size, seq_length, d_model)
        """
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, context, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            context: Context tensor for cross-attention (batch_size, context_length, d_model)
            mask: Optional mask for context (batch_size, seq_length, context_length)
            
        Returns:
            Output tensor (batch_size, seq_length, d_model)
            attention: Cross-attention weights
        """
        # Cross-attention with residual connection and layer normalization
        attn_output, attention = self.cross_attn(x, context, context, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attention

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=3, 
                 d_ff=1024, max_seq_length=512, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Embedding layer for both passage and question
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Context encoder layers
        self.context_encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Question encoder layers
        self.question_encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention layer between context and question
        self.cross_attention = CrossAttentionLayer(d_model, num_heads, d_ff, dropout)
        
        # Output layers for predicting start and end positions
        self.start_output = nn.Linear(d_model, 1)
        self.end_output = nn.Linear(d_model, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following the transformer paper."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, seq_lengths):
        """Create a mask for padding tokens."""
        # seq: (batch_size, seq_length)
        # seq_lengths: (batch_size,)
        batch_size, seq_length = seq.size()
        mask = torch.arange(seq_length, device=seq.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
    
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
            cross_attention_weights: Attention weights for visualization
        """
        batch_size = context.size(0)
        
        # Create masks for padding
        context_mask = torch.arange(context.size(1), device=context.device).unsqueeze(0) < context_lengths.unsqueeze(1)
        question_mask = torch.arange(question.size(1), device=question.device).unsqueeze(0) < question_lengths.unsqueeze(1)
        
        # For self-attention in encoder layers
        context_self_attn_mask = context_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, context_len)
        question_self_attn_mask = question_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, question_len)
        
        # For cross-attention between context and question
        context_question_attn_mask = context_mask.unsqueeze(-1) * question_mask.unsqueeze(1)  # (batch_size, context_len, question_len)
        
        # Embed and add positional encoding
        context_embedded = self.pos_encoding(self.embedding(context) * math.sqrt(self.d_model))  # (batch_size, context_len, d_model)
        question_embedded = self.pos_encoding(self.embedding(question) * math.sqrt(self.d_model))  # (batch_size, question_len, d_model)
        
        # Encode context
        context_encoded = context_embedded
        for layer in self.context_encoder_layers:
            context_encoded = layer(context_encoded, context_self_attn_mask)
        
        # Encode question
        question_encoded = question_embedded
        for layer in self.question_encoder_layers:
            question_encoded = layer(question_encoded, question_self_attn_mask)
        
        # Apply cross-attention between context and question
        combined, cross_attention_weights = self.cross_attention(
            context_encoded, 
            question_encoded, 
            context_question_attn_mask
        )
        
        # Predict start and end positions
        start_logits = self.start_output(combined).squeeze(-1)  # (batch_size, context_len)
        end_logits = self.end_output(combined).squeeze(-1)  # (batch_size, context_len)
        
        # Mask out padding tokens
        mask_1d = ~context_mask
        start_logits.masked_fill_(mask_1d, float('-inf'))
        end_logits.masked_fill_(mask_1d, float('-inf'))
        
        return start_logits, end_logits, cross_attention_weights
    
    def predict(self, context, context_lengths, question, question_lengths):
        """
        Make predictions for the most likely answer span.
        
        Returns:
            predicted_start: Tensor of shape (batch_size,) with predicted start positions
            predicted_end: Tensor of shape (batch_size,) with predicted end positions
            cross_attention_weights: Attention weights for visualization
        """
        # Get logits
        with torch.no_grad():
            start_logits, end_logits, cross_attention_weights = self.forward(
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
        
        return predicted_start, predicted_end, cross_attention_weights
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 