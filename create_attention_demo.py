import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and introduction
nb['cells'] = [
    nbf.v4.new_markdown_cell("""# Attention Visualization Demo

This notebook demonstrates various techniques for visualizing attention weights in transformer-based models."""),
    
    # Imports
    nbf.v4.new_code_cell("""import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from attention_visualization import plot_attention_weights, visualize_attention_pattern, analyze_attention_pattern"""),
    
    # Simple Attention Example
    nbf.v4.new_markdown_cell("""## 1. Simple Attention Example

Let's start with a simple example of attention weights between a query and a context."""),
    
    nbf.v4.new_code_cell("""# Example context and attention weights
context_tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
attention_weights = torch.softmax(torch.randn(1, len(context_tokens)), dim=-1)

# Plot simple attention
plt.figure(figsize=(12, 4))
plot_attention_weights(attention_weights[0], context_tokens, title='Simple Attention Example')
plt.show()"""),
    
    # Multi-Head Attention
    nbf.v4.new_markdown_cell("## 2. Multi-Head Attention Visualization"),
    
    nbf.v4.new_code_cell("""# Create example multi-head attention
num_heads = 8
multi_head_attention = torch.softmax(torch.randn(1, num_heads, 1, len(context_tokens)), dim=-1)

# Visualize multi-head attention
visualize_attention_pattern({'attention': multi_head_attention}, context_tokens)
plt.show()"""),
    
    # Cross-Attention Example
    nbf.v4.new_markdown_cell("## 3. Cross-Attention Example"),
    
    nbf.v4.new_code_cell("""# Example question and context for cross-attention
question_tokens = ['What', 'color', 'is', 'the', 'fox']
cross_attention = torch.softmax(torch.randn(len(question_tokens), len(context_tokens)), dim=-1)

# Plot cross-attention
plt.figure(figsize=(12, 6))
plot_attention_weights(cross_attention, context_tokens, question_tokens, title='Cross-Attention Pattern')
plt.show()"""),
    
    # Attention Analysis
    nbf.v4.new_markdown_cell("## 4. Attention Pattern Analysis"),
    
    nbf.v4.new_code_cell("""# Analyze attention patterns
analysis_results = analyze_attention_pattern(cross_attention, context_tokens)

print("Attention Analysis Results:")
print(f"Token with maximum attention: {analysis_results['max_attention_token']}")
print(f"Maximum attention value: {analysis_results['max_attention_value']:.4f}")
print(f"Attention entropy: {analysis_results['attention_entropy']:.4f}")
print("\\nTop 5 attended tokens:")
for token, weight in analysis_results['top_k_tokens']:
    print(f"  {token}: {weight:.4f}")""")
]

# Write the notebook to a file
with open('attention_visualization_demo.ipynb', 'w') as f:
    nbf.write(nb, f) 