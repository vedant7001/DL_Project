import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from data_utils import get_squad_dataloaders, compute_exact, compute_f1
from models.base_model import EncoderDecoderWithoutAttention
from models.attention_model import EncoderDecoderWithAttention
from models.transformer_model import TransformerEncoderDecoder

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    exact_match = 0
    f1_score = 0
    total_examples = 0
    batch_count = 0
    
    start_time = time.time()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        # Move data to device
        context = batch["context"].to(device)
        context_length = batch["context_length"].to(device)
        question = batch["question"].to(device)
        question_length = batch["question_length"].to(device)
        answer_start = batch["answer_start"].to(device)
        answer_end = batch["answer_end"].to(device)
        answer_texts = batch["answer_text"]
        batch_ids = batch["id"]
        
        # Forward pass
        optimizer.zero_grad()
        
        # Handle different model types
        if isinstance(model, EncoderDecoderWithoutAttention):
            start_logits, end_logits = model(context, context_length, question, question_length)
        elif isinstance(model, EncoderDecoderWithAttention):
            start_logits, end_logits, _ = model(context, context_length, question, question_length)
        elif isinstance(model, TransformerEncoderDecoder):
            start_logits, end_logits, _ = model(context, context_length, question, question_length)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        # Compute loss
        start_loss = criterion(start_logits, answer_start)
        end_loss = criterion(end_logits, answer_end)
        loss = (start_loss + end_loss) / 2
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_size = context.size(0)
        total_examples += batch_size
        epoch_loss += loss.item() * batch_size
        
        # Calculate exact match and F1 scores
        start_pred = torch.argmax(start_logits, dim=1)
        end_pred = torch.argmax(end_logits, dim=1)
        
        for i in range(batch_size):
            pred_start = start_pred[i].item()
            pred_end = end_pred[i].item()
            
            if pred_start <= pred_end:
                # Extract predicted tokens
                predicted_indices = context[i, pred_start:pred_end+1].tolist()
                predicted_tokens = [dataloader.dataset.vocab.idx2word.get(idx, "<UNK>") for idx in predicted_indices if idx != 0]
                predicted_answer = " ".join(predicted_tokens)
                
                # Calculate metrics
                exact_match += compute_exact(answer_texts[i], predicted_answer)
                f1_score += compute_f1(answer_texts[i], predicted_answer)
        
        # Update progress bar
        batch_count += 1
        if batch_count % 10 == 0:
            progress_bar.set_postfix({
                "loss": loss.item(),
                "EM": exact_match / total_examples,
                "F1": f1_score / total_examples
            })
    
    # Calculate epoch metrics
    epoch_loss /= total_examples
    epoch_em = exact_match / total_examples
    epoch_f1 = f1_score / total_examples
    epoch_time = time.time() - start_time
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar("training/loss", epoch_loss, epoch)
        writer.add_scalar("training/EM", epoch_em, epoch)
        writer.add_scalar("training/F1", epoch_f1, epoch)
        writer.add_scalar("training/time", epoch_time, epoch)
    
    return {
        "loss": epoch_loss,
        "EM": epoch_em,
        "F1": epoch_f1,
        "time": epoch_time
    }

def evaluate(model, dataloader, criterion, device, epoch, writer=None, split="val"):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    exact_match = 0
    f1_score = 0
    total_examples = 0
    
    all_predictions = []
    
    start_time = time.time()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [{split.capitalize()}]")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            context = batch["context"].to(device)
            context_length = batch["context_length"].to(device)
            question = batch["question"].to(device)
            question_length = batch["question_length"].to(device)
            answer_start = batch["answer_start"].to(device)
            answer_end = batch["answer_end"].to(device)
            answer_texts = batch["answer_text"]
            batch_ids = batch["id"]
            
            # Forward pass
            if isinstance(model, EncoderDecoderWithoutAttention):
                start_logits, end_logits = model(context, context_length, question, question_length)
                start_pred, end_pred = model.predict(context, context_length, question, question_length)
                attention_weights = None
            elif isinstance(model, EncoderDecoderWithAttention):
                start_logits, end_logits, attention_weights = model(context, context_length, question, question_length)
                start_pred, end_pred, _ = model.predict(context, context_length, question, question_length)
            elif isinstance(model, TransformerEncoderDecoder):
                start_logits, end_logits, attention_weights = model(context, context_length, question, question_length)
                start_pred, end_pred, _ = model.predict(context, context_length, question, question_length)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            # Compute loss
            start_loss = criterion(start_logits, answer_start)
            end_loss = criterion(end_logits, answer_end)
            loss = (start_loss + end_loss) / 2
            
            # Calculate metrics
            batch_size = context.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            
            # Save predictions and calculate metrics
            for i in range(batch_size):
                pred_start = start_pred[i].item()
                pred_end = end_pred[i].item()
                
                # Ensure valid span (start <= end)
                if pred_start > pred_end:
                    pred_start, pred_end = pred_end, pred_start
                
                # Extract predicted tokens
                predicted_indices = context[i, pred_start:pred_end+1].tolist()
                predicted_tokens = [dataloader.dataset.vocab.idx2word.get(idx, "<UNK>") for idx in predicted_indices if idx != 0]
                predicted_answer = " ".join(predicted_tokens)
                
                # Calculate metrics
                exact_match += compute_exact(answer_texts[i], predicted_answer)
                f1_score += compute_f1(answer_texts[i], predicted_answer)
                
                # Save prediction
                all_predictions.append({
                    "id": batch_ids[i],
                    "predicted_answer": predicted_answer,
                    "ground_truth": answer_texts[i],
                    "exact_match": compute_exact(answer_texts[i], predicted_answer),
                    "f1": compute_f1(answer_texts[i], predicted_answer),
                    "start_idx": pred_start,
                    "end_idx": pred_end
                })
                
                # Save attention weights for visualization (first batch only)
                if i == 0 and attention_weights is not None and epoch % 5 == 0:
                    if isinstance(model, EncoderDecoderWithAttention):
                        # For Bahdanau attention
                        attn = attention_weights[i].cpu().numpy()
                        context_tokens = [dataloader.dataset.vocab.idx2word.get(idx.item(), "<PAD>") for idx in context[i, :context_length[i]]]
                        
                        if writer:
                            plot_attention(context_tokens, attn, f"{split}_attention_epoch_{epoch}")
                            writer.add_figure(f"{split}/attention", plt.gcf(), epoch)
                            plt.close()
                    
                    elif isinstance(model, TransformerEncoderDecoder):
                        # For transformer self-attention (first head only)
                        attn = attention_weights[0, 0, :context_length[i], :question_length[i]].cpu().numpy()
                        context_tokens = [dataloader.dataset.vocab.idx2word.get(idx.item(), "<PAD>") for idx in context[i, :context_length[i]]]
                        question_tokens = [dataloader.dataset.vocab.idx2word.get(idx.item(), "<PAD>") for idx in question[i, :question_length[i]]]
                        
                        if writer:
                            plot_cross_attention(context_tokens, question_tokens, attn, f"{split}_cross_attention_epoch_{epoch}")
                            writer.add_figure(f"{split}/cross_attention", plt.gcf(), epoch)
                            plt.close()
    
    # Calculate overall metrics
    avg_loss = total_loss / total_examples
    avg_em = exact_match / total_examples
    avg_f1 = f1_score / total_examples
    eval_time = time.time() - start_time
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar(f"{split}/loss", avg_loss, epoch)
        writer.add_scalar(f"{split}/EM", avg_em, epoch)
        writer.add_scalar(f"{split}/F1", avg_f1, epoch)
        writer.add_scalar(f"{split}/time", eval_time, epoch)
    
    return {
        "loss": avg_loss,
        "EM": avg_em,
        "F1": avg_f1,
        "time": eval_time,
        "predictions": all_predictions
    }

def plot_attention(tokens, attention_weights, title="Attention Weights"):
    """Plot attention weights."""
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_cross_attention(context_tokens, question_tokens, attention_weights, title="Cross Attention Weights"):
    """Plot cross-attention weights between context and question."""
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_weights, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(question_tokens)), question_tokens, rotation=90)
    plt.yticks(range(len(context_tokens)), context_tokens)
    plt.title(title)
    plt.tight_layout()
    return plt

def create_model(model_type, vocab_size, embedding_dim, hidden_dim, **kwargs):
    """Create a model of the specified type."""
    if model_type == "base":
        return EncoderDecoderWithoutAttention(vocab_size, embedding_dim, hidden_dim, **kwargs)
    elif model_type == "attention":
        return EncoderDecoderWithAttention(vocab_size, embedding_dim, hidden_dim, **kwargs)
    elif model_type == "transformer":
        # For transformer, we use d_model instead of embedding_dim and hidden_dim
        d_model = embedding_dim
        return TransformerEncoderDecoder(vocab_size, d_model=d_model, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Train QA models on SQuAD")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "attention", "transformer"],
                        help="Type of model to train")
    parser.add_argument("--embedding_dim", type=int, default=300, help="Dimension of word embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of hidden state in RNNs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--max_context_length", type=int, default=400, help="Maximum context length")
    parser.add_argument("--max_question_length", type=int, default=50, help="Maximum question length")
    parser.add_argument("--use_gru", action="store_true", help="Use GRU instead of LSTM")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for transformer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers for transformer")
    parser.add_argument("--d_ff", type=int, default=1024, help="Dimension of feed-forward layer in transformer")
    parser.add_argument("--output_dir", type=str, default="runs", help="Directory to save outputs")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to use (-1 for all samples)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_type}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, vocab = get_squad_dataloaders(
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        max_question_length=args.max_question_length,
        max_samples=args.max_samples
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    
    if args.model_type == "transformer":
        model_kwargs = {
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "max_seq_length": max(args.max_context_length, args.max_question_length),
            "dropout": args.dropout
        }
    else:
        model_kwargs = {
            "dropout_rate": args.dropout,
            "use_gru": args.use_gru
        }
    
    model = create_model(
        args.model_type,
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        **model_kwargs
    )
    
    # Print model details
    print(f"Model architecture: {type(model).__name__}")
    print(f"Number of parameters: {model.count_parameters():,}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Save model configuration
    config = vars(args)
    config["vocab_size"] = len(vocab)
    config["device"] = str(device)
    config["model_parameters"] = model.count_parameters()
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Train the model
    print("Starting training...")
    best_f1 = 0
    train_metrics = []
    val_metrics = []
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_result = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        train_metrics.append(train_result)
        
        # Evaluate
        val_result = evaluate(model, val_loader, criterion, device, epoch, writer)
        val_metrics.append(val_result)
        
        # Print epoch results
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_result['loss']:.4f}, EM: {train_result['EM']:.4f}, F1: {train_result['F1']:.4f}, Time: {train_result['time']:.2f}s")
        print(f"  Val Loss: {val_result['loss']:.4f}, EM: {val_result['EM']:.4f}, F1: {val_result['F1']:.4f}, Time: {val_result['time']:.2f}s")
        
        # Save best model
        if val_result["F1"] > best_f1:
            best_f1 = val_result["F1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_result["F1"],
                "val_em": val_result["EM"],
                "config": config
            }, os.path.join(output_dir, "best_model.pt"))
            
            # Save best predictions
            with open(os.path.join(output_dir, "best_predictions.json"), "w") as f:
                json.dump(val_result["predictions"], f, indent=4)
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == args.num_epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # Save final metrics
    metrics = {
        "train": train_metrics,
        "val": val_metrics
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Plot learning curves
    plot_learning_curves(train_metrics, val_metrics, output_dir)
    
    print("Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Results saved to {output_dir}")

def plot_learning_curves(train_metrics, val_metrics, output_dir):
    """Plot learning curves for loss and F1 score."""
    epochs = range(1, len(train_metrics) + 1)
    
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [m["loss"] for m in train_metrics], "b-", label="Training Loss")
    plt.plot(epochs, [m["loss"] for m in val_metrics], "r-", label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    
    # F1 curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [m["F1"] for m in train_metrics], "b-", label="Training F1")
    plt.plot(epochs, [m["F1"] for m in val_metrics], "r-", label="Validation F1")
    plt.title("F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "f1_curve.png"))
    plt.close()
    
    # EM curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [m["EM"] for m in train_metrics], "b-", label="Training EM")
    plt.plot(epochs, [m["EM"] for m in val_metrics], "r-", label="Validation EM")
    plt.title("Exact Match Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Exact Match Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "em_curve.png"))
    plt.close()

if __name__ == "__main__":
    main() 