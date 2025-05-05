import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_weights(attention_weights, context_tokens, question_tokens=None, title="Attention Weights"):
    """
    Plot attention weights with improved visualization.
    
    Args:
        attention_weights: numpy array of attention weights
        context_tokens: list of context tokens
        question_tokens: optional list of question tokens (for cross-attention)
        title: plot title
    """
    # Process attention weights
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Normalize weights if needed
    if attention_weights.max() > 1 or attention_weights.min() < 0:
        attention_weights = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min())
    
    # Set up the figure
    plt.figure(figsize=(15, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, 
                cmap='YlOrRd',
                xticklabels=context_tokens,
                yticklabels=question_tokens if question_tokens else ['Attention'],
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(title, pad=20)
    plt.xlabel('Context Tokens', labelpad=10)
    plt.ylabel('Query Tokens' if question_tokens else 'Attention Head', labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return plt.gcf()

def visualize_attention_pattern(model_output, context_tokens, question_tokens=None):
    """
    Visualize different types of attention patterns from the model output.
    
    Args:
        model_output: dictionary containing attention weights and other model outputs
        context_tokens: list of context tokens
        question_tokens: optional list of question tokens
    """
    attention = model_output.get('attention')
    if attention is None:
        print("No attention weights available for visualization")
        return
    
    # Handle different attention patterns
    if len(attention.shape) == 4:  # Multi-head attention
        # Average across heads
        attention_avg = attention.mean(axis=1)
        
        # Plot average attention
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plot_attention_weights(attention_avg[0], 
                             context_tokens,
                             title="Average Attention Across Heads")
        
        # Plot individual heads
        num_heads = attention.shape[1]
        fig, axes = plt.subplots(2, num_heads//2, figsize=(20, 8))
        axes = axes.flatten()
        
        for h in range(num_heads):
            plt.sca(axes[h])
            plot_attention_weights(attention[0, h],
                                 context_tokens,
                                 title=f"Head {h+1}")
        
        plt.tight_layout()
        
    elif len(attention.shape) == 3:  # Single-head or decoder attention
        # Plot attention for each decoder step
        num_steps = attention.shape[0]
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 4))
        
        if num_steps == 1:
            axes = [axes]
        
        for step in range(num_steps):
            plt.sca(axes[step])
            plot_attention_weights(attention[step],
                                 context_tokens,
                                 title=f"Decoder Step {step+1}")
        
        plt.tight_layout()
        
    else:  # Simple attention vector
        plot_attention_weights(attention,
                             context_tokens,
                             title="Attention Weights")

def analyze_attention_pattern(attention_weights, context_tokens):
    """
    Analyze and return insights about the attention pattern.
    
    Args:
        attention_weights: numpy array of attention weights
        context_tokens: list of context tokens
    
    Returns:
        dict: Dictionary containing analysis results
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Flatten if multi-dimensional
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    if len(attention_weights.shape) > 1:
        attention_weights = attention_weights.mean(axis=0)
    
    analysis = {
        'max_attention_token': context_tokens[attention_weights.argmax()],
        'max_attention_value': attention_weights.max(),
        'attention_entropy': -(attention_weights * np.log(attention_weights + 1e-10)).sum(),
        'top_k_tokens': [(context_tokens[i], attention_weights[i]) 
                        for i in attention_weights.argsort()[-5:][::-1]]
    }
    
    return analysis 