import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style
plt.style.use('ggplot')
sns.set_palette("muted")
sns.set_context("talk")

# Model data
models = {
    "Base (LSTM/GRU)": {
        "accuracy": 0.0375,  # F1 score
        "exact_match": 0.0,
        "training_time_per_epoch": 1.77,  # seconds
        "inference_time": 207.22,  # milliseconds
        "parameters": 263810,
        "memory_usage": 1.0,  # relative value
        "interpretability": 2,  # score out of 10
        "parallelization": 2,  # score out of 10
        "scalability": 3,  # score out of 10
        "ease_of_implementation": 8,  # score out of 10
        "convergence_speed": 5,  # score out of 10
        "context_length_handling": 4  # score out of 10
    },
    "Attention (Bahdanau)": {
        "accuracy": 0.04,  # F1 score
        "exact_match": 0.0,
        "training_time_per_epoch": 1.68,  # seconds
        "inference_time": 180.49,  # milliseconds
        "parameters": 272066,
        "memory_usage": 1.2,  # relative value
        "interpretability": 8,  # score out of 10
        "parallelization": 2,  # score out of 10
        "scalability": 5,  # score out of 10
        "ease_of_implementation": 6,  # score out of 10
        "convergence_speed": 6,  # score out of 10
        "context_length_handling": 6  # score out of 10
    },
    "Transformer": {
        "accuracy": 0.069,  # F1 score
        "exact_match": 0.0,
        "training_time_per_epoch": 9.79,  # seconds
        "inference_time": 187.84,  # milliseconds
        "parameters": 1821954,
        "memory_usage": 6.7,  # relative value
        "interpretability": 6,  # score out of 10
        "parallelization": 9,  # score out of 10 
        "scalability": 8,  # score out of 10
        "ease_of_implementation": 4,  # score out of 10
        "convergence_speed": 7,  # score out of 10
        "context_length_handling": 8  # score out of 10
    }
}

# Create output directory if it doesn't exist
os.makedirs("comparison_visualizations", exist_ok=True)

# Helper function for formatting large numbers
def format_with_commas(x, pos):
    return f'{int(x):,}'

# 1. Basic Bar Charts for Key Metrics
def plot_basic_metrics():
    metrics = ['accuracy', 'training_time_per_epoch', 'inference_time']
    titles = ['F1 Score', 'Training Time per Epoch (s)', 'Inference Time (ms)']
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [models[model][metric] for model in models]
        axes[i].bar(models.keys(), values, color=colors)
        axes[i].set_title(title)
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        if metric == 'accuracy':
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.001, f"{v:.4f}", ha='center')
        else:
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.1, f"{v:.2f}", ha='center')
        
        # Rotate x-axis labels
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig('comparison_visualizations/basic_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Model Complexity and Parameters
def plot_model_complexity():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract parameters
    model_names = list(models.keys())
    params = [models[model]['parameters'] for model in model_names]
    
    # Create bar chart
    bars = ax.bar(model_names, params, color=['#3498db', '#f39c12', '#2ecc71'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50000,
                f'{int(height):,}', ha='center', va='bottom')
    
    # Set formatter for y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(format_with_commas))
    
    # Labels and title
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Complexity Comparison')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig('comparison_visualizations/model_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Radar Chart for Model Characteristics
def plot_radar_chart():
    # Features to include in the radar chart
    features = ['interpretability', 'parallelization', 'scalability', 
                'ease_of_implementation', 'convergence_speed', 'context_length_handling']
    
    # Create a DataFrame
    model_names = list(models.keys())
    df = pd.DataFrame({
        'feature': features * len(model_names),
        'model': [model for model in model_names for _ in features],
        'score': [models[model][feature] for model in model_names for feature in features]
    })
    
    # Convert to wide format for radar chart
    df_wide = df.pivot(index='feature', columns='model', values='score')
    
    # Number of features
    N = len(features)
    
    # What will be the angle of each axis in the plot (divide the plot into equal parts)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Feature names with proper capitalization
    feature_names = [f.replace('_', ' ').title() for f in features]
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each model
    for i, model in enumerate(model_names):
        values = df_wide[model].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    
    # Set y-axis limits
    ax.set_ylim(0, 10)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Characteristics Comparison', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig('comparison_visualizations/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Bubble Chart: Accuracy vs Training Time vs Parameters
def plot_bubble_chart():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    model_names = list(models.keys())
    x = [models[model]['training_time_per_epoch'] for model in model_names]
    y = [models[model]['accuracy'] for model in model_names]
    size = [np.sqrt(models[model]['parameters'])/30 for model in model_names]
    colors = ['#3498db', '#f39c12', '#2ecc71']
    
    # Create scatter plot
    scatter = ax.scatter(x, y, s=size, c=colors, alpha=0.6, edgecolors='black')
    
    # Add labels for each point
    for i, model in enumerate(model_names):
        ax.annotate(model, (x[i], y[i]),
                   xytext=(10, 5), textcoords='offset points')
    
    # Labels and title
    ax.set_xlabel('Training Time per Epoch (s)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Accuracy vs Training Time vs Model Size')
    
    # Add a note about bubble size
    ax.text(0.95, 0.05, 'Bubble size represents\nnumber of parameters',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('comparison_visualizations/bubble_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Stacked bar for speed metrics
def plot_speed_metrics():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(models.keys())
    train_times = [models[model]['training_time_per_epoch'] for model in model_names]
    inference_times = [models[model]['inference_time']/1000 for model in model_names]  # convert to seconds
    
    # Create stacked bars
    bar_width = 0.35
    x = np.arange(len(model_names))
    
    ax.bar(x, train_times, bar_width, label='Training Time per Epoch (s)', color='#3498db')
    ax.bar(x, inference_times, bar_width, bottom=train_times, 
           label='Inference Time (s)', color='#f39c12')
    
    # Add labels and title
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training and Inference Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    # Add value labels on top of bars
    for i in range(len(model_names)):
        ax.text(i, train_times[i]/2, f'{train_times[i]:.2f}s', ha='center', va='center', color='white')
        ax.text(i, train_times[i] + inference_times[i]/2, f'{inference_times[i]:.3f}s', 
                ha='center', va='center', color='white')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig('comparison_visualizations/speed_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Generate a comprehensive visualization dashboard
def create_visualization_dashboard():
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid structure
    gs = fig.add_gridspec(3, 3)
    
    # Basic metrics subplot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Get data for basic metrics
    model_names = list(models.keys())
    metrics = ['accuracy', 'inference_time']
    x = np.arange(len(model_names))
    width = 0.35
    
    # Plot F1 scores
    ax1.bar(x - width/2, [models[m]['accuracy'] for m in model_names], width, 
           label='F1 Score', color='#3498db')
    
    # Set up secondary axis for inference time
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + width/2, [models[m]['inference_time'] for m in model_names], width, 
                label='Inference Time (ms)', color='#f39c12')
    
    # Configure axes
    ax1.set_ylabel('F1 Score')
    ax1_twin.set_ylabel('Inference Time (ms)')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Model complexity subplot
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Plot parameters
    params = [models[model]['parameters'] for model in model_names]
    bars = ax2.bar(model_names, params, color=['#3498db', '#f39c12', '#2ecc71'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # Configure axis
    ax2.set_ylabel('Parameters')
    ax2.set_title('Model Complexity')
    ax2.yaxis.set_major_formatter(FuncFormatter(format_with_commas))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Training time subplot
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Plot training times
    train_times = [models[model]['training_time_per_epoch'] for model in model_names]
    bars = ax3.bar(model_names, train_times, color=['#3498db', '#f39c12', '#2ecc71'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # Configure axis
    ax3.set_ylabel('Seconds')
    ax3.set_title('Training Time per Epoch')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Memory usage subplot
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Plot memory usage
    memory = [models[model]['memory_usage'] for model in model_names]
    bars = ax4.bar(model_names, memory, color=['#3498db', '#f39c12', '#2ecc71'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom')
    
    # Configure axis
    ax4.set_ylabel('Relative Value')
    ax4.set_title('Memory Usage')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Radar chart for model characteristics
    ax5 = fig.add_subplot(gs[2, :], polar=True)
    
    # Features for radar chart
    features = ['interpretability', 'parallelization', 'scalability', 
                'ease_of_implementation', 'convergence_speed', 'context_length_handling']
    
    # Number of features
    N = len(features)
    
    # Calculate angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Feature names with proper capitalization
    feature_names = [f.replace('_', ' ').title() for f in features]
    
    # Plot each model
    colors = ['#3498db', '#f39c12', '#2ecc71']
    for i, model in enumerate(model_names):
        values = [models[model][feature] for feature in features]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax5.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax5.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set feature labels
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(feature_names)
    
    # Set y-axis limits
    ax5.set_ylim(0, 10)
    
    # Add legend
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    ax5.set_title('Model Characteristics', y=1.1)
    
    # Overall layout adjustments
    plt.suptitle('Comprehensive Model Comparison Dashboard', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('comparison_visualizations/dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all visualizations
if __name__ == "__main__":
    plot_basic_metrics()
    plot_model_complexity()
    plot_radar_chart()
    plot_bubble_chart()
    plot_speed_metrics()
    create_visualization_dashboard()
    
    print("Visualizations created successfully in the 'comparison_visualizations' folder") 