import torch
import torch.nn as nn
import argparse
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from datetime import datetime

from data_utils import get_squad_dataloaders, compute_exact, compute_f1
from models.base_model import EncoderDecoderWithoutAttention
from models.attention_model import EncoderDecoderWithAttention
from models.transformer_model import TransformerEncoderDecoder
from train import evaluate, plot_learning_curves, create_model

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint["config"]
    
    # Create model
    model_type = config["model_type"]
    
    if model_type == "transformer":
        model_kwargs = {
            "num_heads": config.get("num_heads", 8),
            "num_layers": config.get("num_layers", 3),
            "d_ff": config.get("d_ff", 1024),
            "max_seq_length": max(config.get("max_context_length", 400), config.get("max_question_length", 50)),
            "dropout": config.get("dropout", 0.1)
        }
    else:
        model_kwargs = {
            "dropout_rate": config.get("dropout", 0.1),
            "use_gru": config.get("use_gru", False)
        }
    
    model = create_model(
        model_type,
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        **model_kwargs
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, config

def evaluate_model(model, val_loader, device):
    """Evaluate model and measure inference time."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Standard evaluation
    start_time = time.time()
    results = evaluate(model, val_loader, criterion, device, epoch=0, split="test")
    total_eval_time = time.time() - start_time
    
    # Measure inference time for a single batch
    batch = next(iter(val_loader))
    context = batch["context"].to(device)
    context_length = batch["context_length"].to(device)
    question = batch["question"].to(device)
    question_length = batch["question_length"].to(device)
    
    batch_size = context.size(0)
    
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            if isinstance(model, EncoderDecoderWithoutAttention):
                model.predict(context, context_length, question, question_length)
            else:
                model.predict(context, context_length, question, question_length)
    
    # Measure inference time
    num_runs = 10
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            if isinstance(model, EncoderDecoderWithoutAttention):
                model.predict(context, context_length, question, question_length)
            else:
                model.predict(context, context_length, question, question_length)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = (time.time() - start_time) / (num_runs * batch_size)
    
    results["inference_time"] = inference_time * 1000  # convert to ms
    results["total_eval_time"] = total_eval_time
    
    return results

def compare_models(model_paths, output_dir):
    """Compare multiple models on the validation set."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation data
    print("Loading validation data...")
    _, val_loader, _ = get_squad_dataloaders(batch_size=32)
    
    # Evaluate each model
    results = []
    
    for model_path in model_paths:
        print(f"Evaluating model: {model_path}")
        model, config = load_model(model_path)
        model = model.to(device)
        
        # Count parameters
        num_params = model.count_parameters()
        
        # Evaluate
        eval_results = evaluate_model(model, val_loader, device)
        
        # Add model info to results
        model_name = os.path.basename(os.path.dirname(model_path))
        model_results = {
            "model_name": model_name,
            "model_type": config["model_type"],
            "num_params": num_params,
            "embedding_dim": config["embedding_dim"],
            "hidden_dim": config["hidden_dim"],
            "use_gru": config.get("use_gru", False),
            "EM": eval_results["EM"],
            "F1": eval_results["F1"],
            "loss": eval_results["loss"],
            "inference_time": eval_results["inference_time"],
            "total_eval_time": eval_results["total_eval_time"],
            "predictions": eval_results["predictions"]
        }
        
        results.append(model_results)
    
    # Save all results
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Create comparison table
    table = PrettyTable()
    table.field_names = ["Model", "Type", "Params", "EM", "F1", "Inference Time (ms)"]
    
    for result in results:
        table.add_row([
            result["model_name"],
            result["model_type"],
            f"{result['num_params']:,}",
            f"{result['EM']:.4f}",
            f"{result['F1']:.4f}",
            f"{result['inference_time']:.2f}"
        ])
    
    print(table)
    
    # Save table to file
    with open(os.path.join(output_dir, "comparison_table.txt"), "w") as f:
        f.write(str(table))
    
    # Generate comparison plots
    plot_model_comparison(results, output_dir)
    
    # Compare error cases
    compare_error_cases(results, output_dir)

def plot_model_comparison(results, output_dir):
    """Generate plots comparing model performance."""
    # Extract data
    model_names = [r["model_name"] for r in results]
    params = [r["num_params"] for r in results]
    em_scores = [r["EM"] for r in results]
    f1_scores = [r["F1"] for r in results]
    inf_times = [r["inference_time"] for r in results]
    
    # Plot F1 score vs model size
    plt.figure(figsize=(10, 6))
    plt.scatter(params, f1_scores, s=100)
    for i, name in enumerate(model_names):
        plt.annotate(name, (params[i], f1_scores[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel("Number of Parameters")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Model Size")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "f1_vs_size.png"))
    plt.close()
    
    # Plot F1 score vs inference time
    plt.figure(figsize=(10, 6))
    plt.scatter(inf_times, f1_scores, s=100)
    for i, name in enumerate(model_names):
        plt.annotate(name, (inf_times[i], f1_scores[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Inference Time")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "f1_vs_time.png"))
    plt.close()
    
    # Bar chart of EM and F1 scores
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, em_scores, width, label='Exact Match')
    ax.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, "score_comparison.png"))
    plt.close()
    
    # Bar chart of inference time
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, inf_times)
    plt.xlabel("Model")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Time Comparison")
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, "inference_time.png"))
    plt.close()

def compare_error_cases(results, output_dir):
    """Analyze cases where models differ in their predictions."""
    # Group predictions by question ID
    all_predictions = {}
    model_names = []
    
    for result in results:
        model_name = result["model_name"]
        model_names.append(model_name)
        
        for pred in result["predictions"]:
            q_id = pred["id"]
            
            if q_id not in all_predictions:
                all_predictions[q_id] = {
                    "ground_truth": pred["ground_truth"],
                    "models": {}
                }
            
            all_predictions[q_id]["models"][model_name] = {
                "prediction": pred["predicted_answer"],
                "em": pred["exact_match"],
                "f1": pred["f1"]
            }
    
    # Find interesting cases
    all_correct = []  # All models correct
    all_wrong = []    # All models wrong
    vary_correct = [] # Some models correct, some wrong
    
    for q_id, case in all_predictions.items():
        ground_truth = case["ground_truth"]
        correct_models = []
        wrong_models = []
        
        for model_name, pred_info in case["models"].items():
            if pred_info["em"] == 1:
                correct_models.append(model_name)
            else:
                wrong_models.append(model_name)
        
        if len(correct_models) == len(model_names):
            all_correct.append(q_id)
        elif len(wrong_models) == len(model_names):
            all_wrong.append(q_id)
        else:
            vary_correct.append(q_id)
    
    # Save analysis
    analysis = {
        "total_questions": len(all_predictions),
        "all_correct_count": len(all_correct),
        "all_wrong_count": len(all_wrong),
        "vary_correct_count": len(vary_correct),
        "all_correct_percentage": len(all_correct) / len(all_predictions) * 100,
        "all_wrong_percentage": len(all_wrong) / len(all_predictions) * 100,
        "vary_correct_percentage": len(vary_correct) / len(all_predictions) * 100,
    }
    
    # Save interesting examples where models disagree
    interesting_examples = []
    
    for q_id in vary_correct[:50]:  # Limit to 50 examples
        case = all_predictions[q_id]
        interesting_examples.append({
            "id": q_id,
            "ground_truth": case["ground_truth"],
            "model_predictions": case["models"]
        })
    
    analysis["interesting_examples"] = interesting_examples
    
    with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=4)
    
    # Create a Venn diagram of correct predictions
    model_correct_sets = {}
    for model_name in model_names:
        model_correct_sets[model_name] = set()
    
    for q_id, case in all_predictions.items():
        for model_name, pred_info in case["models"].items():
            if pred_info["em"] == 1:
                model_correct_sets[model_name].add(q_id)
    
    # Print statistics
    print("\nPrediction Analysis:")
    print(f"Total questions: {len(all_predictions)}")
    print(f"All models correct: {len(all_correct)} ({analysis['all_correct_percentage']:.2f}%)")
    print(f"All models wrong: {len(all_wrong)} ({analysis['all_wrong_percentage']:.2f}%)")
    print(f"Models disagree: {len(vary_correct)} ({analysis['vary_correct_percentage']:.2f}%)")
    
    for model_name in model_names:
        print(f"Correct by {model_name}: {len(model_correct_sets[model_name])} ({len(model_correct_sets[model_name])/len(all_predictions)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare QA models")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True,
                        help="Paths to model checkpoints to evaluate")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    
    # Compare models
    compare_models(args.model_paths, output_dir)
    
    print(f"Comparison results saved to {output_dir}")

if __name__ == "__main__":
    main() 