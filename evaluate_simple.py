import torch
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from evaluate import load_model
from data_utils import simple_tokenize

def find_latest_model(model_type):
    """Find the latest model of the specified type."""
    runs_dir = 'runs'
    matching_dirs = [d for d in os.listdir(runs_dir) if d.startswith(model_type)]
    if not matching_dirs:
        return None
    
    latest_dir = sorted(matching_dirs)[-1]
    model_path = os.path.join(runs_dir, latest_dir, 'best_model.pt')
    
    if os.path.exists(model_path):
        return model_path
    return None

def measure_inference_time(model, device, input_length=200, batch_size=1, num_runs=10):
    """Measure the inference time of a model."""
    # Create dummy input
    context = torch.ones((batch_size, input_length), dtype=torch.long).to(device)
    context_length = torch.tensor([input_length] * batch_size).to(device)
    question = torch.ones((batch_size, 20), dtype=torch.long).to(device)  # Shorter question
    question_length = torch.tensor([20] * batch_size).to(device)
    
    # Warm-up
    with torch.no_grad():
        if hasattr(model, 'predict'):
            for _ in range(3):
                _ = model.predict(context, context_length, question, question_length)
        
    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            if hasattr(model, 'predict'):
                _ = model.predict(context, context_length, question, question_length)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    return (elapsed / num_runs) * 1000  # Convert to milliseconds

def analyze_model(model_path):
    """Analyze a model and return its properties."""
    print(f"Analyzing {model_path}...")
    
    model, config = load_model(model_path)
    model_type = config["model_type"]
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Put model in evaluation mode
    model = model.to(device)
    model.eval()
    
    # Measure inference time
    inference_time = measure_inference_time(model, device)
    
    # Try to get the best metrics from the saved checkpoint
    metrics = {}
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
    except:
        pass  # If metrics not available, just continue
    
    # Get embedding dimension and hidden dimension
    embedding_dim = config.get('embedding_dim', 0)
    hidden_dim = config.get('hidden_dim', 0)
    
    result = {
        "model_name": os.path.basename(os.path.dirname(model_path)),
        "model_type": model_type,
        "num_params": num_params,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "inference_time": inference_time
    }
    
    # Add metrics if available
    if metrics:
        result.update(metrics)
    
    return result

def plot_results(results, output_dir):
    """Create comparison plots for the models."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = [r["model_name"] for r in results]
    params = [r["num_params"] for r in results]
    inf_times = [r["inference_time"] for r in results]
    
    # Plot model sizes
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, params)
    plt.title('Model Size Comparison')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_size_comparison.png'))
    
    # Plot inference times
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, inf_times)
    plt.title('Inference Time Comparison')
    plt.ylabel('Inference Time (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a nice table
    table = PrettyTable()
    table.field_names = ["Model", "Type", "Parameters", "Inference Time (ms)"]
    
    for r in results:
        table.add_row([
            r["model_name"],
            r["model_type"],
            f"{r['num_params']:,}",
            f"{r['inference_time']:.2f}"
        ])
    
    print("\nModel Comparison:")
    print(table)
    
    with open(os.path.join(output_dir, 'comparison_table.txt'), 'w') as f:
        f.write(str(table))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument('--output_dir', type=str, default='comparison_results', 
                        help='Directory to save results')
    parser.add_argument('--model_paths', nargs='*', 
                        help='Paths to model checkpoints (optional)')
    
    args = parser.parse_args()
    
    # Use provided models or find the latest of each type
    model_paths = args.model_paths
    if not model_paths:
        print("No model paths provided, finding latest models...")
        model_paths = []
        for model_type in ['base', 'attention', 'transformer']:
            model_path = find_latest_model(model_type)
            if model_path:
                print(f"Found {model_type} model: {model_path}")
                model_paths.append(model_path)
    
    if not model_paths:
        print("No models found. Please train models first or provide paths.")
        exit(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Analyze models
    results = []
    for model_path in model_paths:
        try:
            result = analyze_model(model_path)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {model_path}: {e}")
    
    # Plot and save results
    if results:
        plot_results(results, args.output_dir)
        print(f"Results saved to {args.output_dir}")
    else:
        print("No models could be analyzed successfully.") 