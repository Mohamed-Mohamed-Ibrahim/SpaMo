"""
Optional: Computing Embeddings for Similarity-Based Context Retrieval

This script shows how to pre-compute embeddings for the training dataset
to enable similarity-based context retrieval (instead of random).

Usage:
    python compute_embeddings.py --dataset ./preprocess/Phoenix14T --output ./embeddings.pt

The embeddings can then be used in the config:
    embedding_cache_path: ./embeddings.pt
    context_retrieval_mode: similarity
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict
import sys

# Try multiple embedding options
AVAILABLE_MODELS = {
    'text-embedding-ada-002': 'Uses OpenAI API (requires key)',
    'all-MiniLM-L6-v2': 'Fast, requires sentence-transformers',
    'all-mpnet-base-v2': 'Higher quality, slower',
    't5-base': 'Can use model hidden states',
}


def compute_embeddings_with_sentence_transformers(
    dataset_metadata: List[Dict],
    model_name: str = 'all-MiniLM-L6-v2'
) -> torch.Tensor:
    """
    Compute embeddings using sentence-transformers.
    
    Args:
        dataset_metadata: List of dicts with 'text' and 'gloss' keys
        model_name: Model from HuggingFace Hub
        
    Returns:
        Tensor of shape (N, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)
    
    print(f"[Embeddings] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Prepare texts for embedding
    # Use concatenation of gloss + text for richer representation
    texts_to_embed = [
        f"{sample['gloss']} {sample['text']}"
        for sample in dataset_metadata
    ]
    
    print(f"[Embeddings] Computing embeddings for {len(texts_to_embed)} samples...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    
    return embeddings


def compute_embeddings_with_text_only(
    dataset_metadata: List[Dict],
) -> torch.Tensor:
    """
    Compute embeddings using only text (without sentence-transformers).
    This is a simple baseline using TF-IDF-like approach.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    
    texts = [sample['text'] for sample in dataset_metadata]
    
    print("[Embeddings] Computing TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=1024, stop_words='english')
    embeddings = vectorizer.fit_transform(texts)
    embeddings = normalize(embeddings, norm='l2')
    
    return torch.from_numpy(embeddings.toarray()).float()


def compute_embeddings_with_t5(
    dataset_metadata: List[Dict],
    model_name: str = 't5-base',
    max_length: int = 512
) -> torch.Tensor:
    """
    Compute embeddings using T5 encoder (the same model used in training).
    """
    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        print("ERROR: transformers not installed")
        sys.exit(1)
    
    print(f"[Embeddings] Loading T5 model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    
    texts = [f"{s['gloss']} {s['text']}" for s in dataset_metadata]
    
    embeddings_list = []
    
    print(f"[Embeddings] Computing T5 embeddings for {len(texts)} samples...")
    
    with torch.no_grad():
        for i in range(0, len(texts), 32):  # Batch processing
            batch_texts = texts[i:i+32]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)
            
            # Get encoder output
            outputs = model(**inputs)
            
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            
            embeddings_list.append(embeddings.cpu())
            
            # Progress
            if (i + 32) % 128 == 0:
                print(f"  Processed {min(i+32, len(texts))}/{len(texts)} samples")
    
    embeddings = torch.cat(embeddings_list, dim=0)
    
    # Normalize
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    return embeddings


def save_embeddings(embeddings: torch.Tensor, output_path: str):
    """Save embeddings to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Embeddings] Saving embeddings to: {output_path}")
    torch.save(embeddings, output_path)
    print(f"[Embeddings] Shape: {embeddings.shape}")
    print(f"[Embeddings] Size: {output_path.stat().st_size / 1e6:.1f} MB")


def load_dataset_metadata_from_npy(anno_root: str, mode: str = 'train') -> List[Dict]:
    """
    Load dataset metadata from Phoenix14T annotation files.
    
    Args:
        anno_root: Root directory containing annotation .npy files
        mode: 'train', 'dev', or 'test'
        
    Returns:
        List of metadata dicts
    """
    anno_root = Path(anno_root)
    anno_file = anno_root / f'{mode}_info_ml.npy'
    
    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")
    
    data = np.load(anno_file, allow_pickle=True).item()
    
    metadata = []
    for sample in data:
        # Normalize text
        text = sample.get('text', '').strip()
        if not text.endswith('.'):
            text = f"{text}."
        
        metadata.append({
            'id': sample.get('fileid', ''),
            'text': text,
            'gloss': sample.get('gloss', ''),
            'lang': 'German'
        })
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Compute embeddings for context retrieval'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset annotations (e.g., ./preprocess/Phoenix14T)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./embeddings.pt',
        help='Output path for embeddings (default: ./embeddings.pt)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'dev', 'test'],
        default='train',
        help='Dataset split to use'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['sentence-transformers', 'tfidf', 't5'],
        default='sentence-transformers',
        help='Embedding method to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Model name for sentence-transformers or T5'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COMPUTING EMBEDDINGS FOR CONTEXT RETRIEVAL")
    print("="*70)
    
    print(f"\n[Config]")
    print(f"  Dataset path: {args.dataset}")
    print(f"  Output path: {args.output}")
    print(f"  Mode: {args.mode}")
    print(f"  Method: {args.method}")
    print(f"  Device: {args.device}")
    
    # Load metadata
    print(f"\n[Loading] Reading dataset metadata...")
    try:
        metadata = load_dataset_metadata_from_npy(args.dataset, args.mode)
        print(f"[Loading] Loaded {len(metadata)} samples")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)
    
    # Compute embeddings
    print(f"\n[Computing] Using {args.method} method...")
    try:
        if args.method == 'sentence-transformers':
            embeddings = compute_embeddings_with_sentence_transformers(
                metadata,
                model_name=args.model
            )
        elif args.method == 'tfidf':
            embeddings = compute_embeddings_with_text_only(metadata)
        elif args.method == 't5':
            embeddings = compute_embeddings_with_t5(metadata, model_name=args.model)
        else:
            raise ValueError(f"Unknown method: {args.method}")
    except Exception as e:
        print(f"[ERROR] Failed to compute embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save embeddings
    save_embeddings(embeddings, args.output)
    
    print("\n" + "="*70)
    print("✅ SUCCESSFULLY COMPUTED EMBEDDINGS")
    print("="*70)
    print(f"\nNext step: Add to config:")
    print(f"  embedding_cache_path: {args.output}")
    print(f"  context_retrieval_mode: similarity")
    print(f"  num_in_context: 3  # or your preferred k\n")


if __name__ == '__main__':
    main()
