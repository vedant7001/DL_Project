import torch
import os
import json
import matplotlib.pyplot as plt
from evaluate import load_model
from data_utils import simple_tokenize

# Find the latest model checkpoints
def find_latest_model(model_type):
    runs_dir = 'runs'
    matching_dirs = [d for d in os.listdir(runs_dir) if d.startswith(model_type)]
    if not matching_dirs:
        return None
    
    # Sort by timestamp
    latest_dir = sorted(matching_dirs)[-1]
    model_path = os.path.join(runs_dir, latest_dir, 'best_model.pt')
    
    if os.path.exists(model_path):
        return model_path
    return None

def get_answer(model, model_type, context, question, device):
    """
    Get an answer from a model for a given context and question.
    """
    # Tokenize the inputs
    context_tokens = simple_tokenize(context)
    question_tokens = simple_tokenize(question)
    
    # Truncate if needed
    max_context_len = 400
    max_question_len = 50
    
    if len(context_tokens) > max_context_len:
        context_tokens = context_tokens[:max_context_len]
    if len(question_tokens) > max_question_len:
        question_tokens = question_tokens[:max_question_len]
    
    # For demonstration, we'll use placeholder values for word indices
    # In a real scenario, you'd convert tokens to indices using the vocabulary
    context_tensor = torch.ones(1, len(context_tokens), dtype=torch.long).to(device)  # All UNK tokens
    question_tensor = torch.ones(1, len(question_tokens), dtype=torch.long).to(device)  # All UNK tokens
    
    context_len = torch.tensor([len(context_tokens)]).to(device)
    question_len = torch.tensor([len(question_tokens)]).to(device)
    
    # Get predictions
    with torch.no_grad():
        if model_type == 'base':
            start_idx, end_idx = model.predict(context_tensor, context_len, question_tensor, question_len)
            attention = None
        else:
            start_idx, end_idx, attention = model.predict(context_tensor, context_len, question_tensor, question_len)
    
    # Get the predicted span
    start = start_idx.item()
    end = end_idx.item()
    
    # Ensure start <= end
    if start > end:
        start, end = end, start
    
    # Extract the answer
    answer_tokens = context_tokens[start:end+1]
    answer = ' '.join(answer_tokens)
    
    result = {
        'question': question,
        'context': context,
        'answer': answer,
        'start': start,
        'end': end
    }
    
    if attention is not None:
        result['attention'] = attention[0].cpu().numpy()
    
    return result

def highlight_answer(context, start, end):
    """
    Highlight the answer in the context for display.
    """
    tokens = simple_tokenize(context)
    highlighted = []
    
    for i, token in enumerate(tokens):
        if start <= i <= end:
            highlighted.append(f"[{token}]")
        else:
            highlighted.append(token)
    
    return ' '.join(highlighted)

def main():
    # Find model paths
    base_model_path = find_latest_model('base')
    attention_model_path = find_latest_model('attention')
    transformer_model_path = find_latest_model('transformer')
    
    print(f"Base model: {base_model_path}")
    print(f"Attention model: {attention_model_path}")
    print(f"Transformer model: {transformer_model_path}")
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    
    if base_model_path:
        print("Loading base model...")
        base_model, _ = load_model(base_model_path)
        base_model = base_model.to(device)
        base_model.eval()
        models['base'] = base_model
    
    if attention_model_path:
        print("Loading attention model...")
        attention_model, _ = load_model(attention_model_path)
        attention_model = attention_model.to(device)
        attention_model.eval()
        models['attention'] = attention_model
    
    if transformer_model_path:
        print("Loading transformer model...")
        transformer_model, _ = load_model(transformer_model_path)
        transformer_model = transformer_model.to(device)
        transformer_model.eval()
        models['transformer'] = transformer_model
    
    print(f"\nLoaded {len(models)} models")
    
    # Example context and questions
    context = """
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.
    """
    
    questions = [
        "What is deep learning part of?",
        "What types of learning can be used in deep learning?",
        "In which fields has deep learning been applied?"
    ]
    
    # Answer each question with all models
    for question in questions:
        print(f"\n{'='*80}\nQuestion: {question}\n{'='*80}\n")
        
        for model_type, model in models.items():
            print(f"Model: {model_type.upper()}")
            result = get_answer(model, model_type, context, question, device)
            
            print(f"Answer: {result['answer']}\n")
            print("Context with highlighted answer:")
            print(highlight_answer(context, result['start'], result['end']))
            
            print('\n' + '-'*80)

if __name__ == "__main__":
    main() 