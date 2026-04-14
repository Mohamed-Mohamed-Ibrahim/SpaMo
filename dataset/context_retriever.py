"""
Context Retriever for In-Context Learning without Data Leakage

This module provides a ContextRetriever that samples context examples from the training
dataset, ensuring:
1. No data leakage - current sample is never in its own context
2. Efficient retrieval - supports random sampling and similarity-based retrieval
3. Batch-compatible - works seamlessly with PyTorch DataLoader
"""

import random
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path
import torch


class ContextRetriever:
    """
    Retrieves context examples from a training dataset for few-shot learning.
    
    Supports two modes:
    1. Random Sampling: Randomly select k examples (simple, efficient)
    2. Similarity-Based: Find k most similar examples using embeddings (optional)
    
    Key Feature: Excludes current sample by ID to prevent data leakage.
    """
    
    def __init__(
        self,
        dataset_metadata: List[Dict],
        num_context: int = 0,
        mode: str = 'random',
        seed: int = 42,
        embedding_cache_path: Optional[str] = None,
        similarity_model: Optional[Any] = None,
    ):
        """
        Initialize the ContextRetriever.
        
        Args:
            dataset_metadata: List of dicts with keys 'id', 'text', 'gloss', 'lang'
            num_context: Number of context examples to retrieve (k)
            mode: 'random' or 'similarity'
            seed: Random seed for reproducibility
            embedding_cache_path: Path to pre-computed embeddings for similarity mode
            similarity_model: Pre-loaded SentenceTransformer model for similarity mode
        """
        self.dataset_metadata = dataset_metadata
        self.num_context = num_context
        self.mode = mode
        self.seed = seed
        
        # Build index for O(1) lookup by ID
        self.id_to_idx = {str(meta['id']): idx for idx, meta in enumerate(dataset_metadata)}
        
        # Load embeddings if using similarity-based retrieval
        self.embeddings = None
        self.similarity_model = similarity_model
        if self.mode == 'similarity' and embedding_cache_path:
            self.embeddings = self._load_embeddings(embedding_cache_path)
            # Load the similarity model if not provided
            if self.similarity_model is None:
                from sentence_transformers import SentenceTransformer
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.similarity_model.eval()  # Set to eval mode
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _load_embeddings(self, cache_path: str) -> Optional[torch.Tensor]:
        """Load pre-computed embeddings for similarity-based retrieval."""
        try:
            if cache_path.endswith('.pt'):
                return torch.load(cache_path)
            elif cache_path.endswith('.npy'):
                return torch.from_numpy(np.load(cache_path)).float()
            else:
                print(f"Warning: Unsupported embedding format: {cache_path}")
                return None
        except Exception as e:
            print(f"Warning: Failed to load embeddings from {cache_path}: {e}")
            return None
    
    def retrieve(self, sample_id: str, current_sample: Dict = None) -> List[Dict]:
        """
        Retrieve k context examples for a given sample.
        
        Args:
            sample_id: ID of the current sample (will be excluded from context)
            current_sample: Optional dict with current sample data for similarity mode
            
        Returns:
            List of context examples (each is a dict with 'gloss' and 'text')
        """
        if self.num_context <= 0:
            return []
        
        # Handle None or invalid sample_id
        if sample_id is None:
            print("Warning: sample_id is None, falling back to random sampling")
            # Get all indices as candidates since we can't exclude the current sample
            candidate_indices = list(range(len(self.dataset_metadata)))
            return self._retrieve_random(candidate_indices)
        
        # Ensure sample_id is a string for consistent lookup
        sample_id = str(sample_id)
        
        # Get candidate indices (all except current sample)
        candidate_indices = [
            idx for idx, meta in enumerate(self.dataset_metadata)
            if str(meta['id']) != sample_id
        ]
        
        if len(candidate_indices) == 0:
            print(f"Warning: No candidate context examples found for sample_id '{sample_id}'")
            return []
        
        if self.mode == 'random':
            return self._retrieve_random(candidate_indices)
        
        elif self.mode == 'similarity':
            if current_sample is None:
                print("Warning: current_sample required for similarity mode, falling back to random")
                return self._retrieve_random(candidate_indices)
            return self._retrieve_similar_from_sample(current_sample, candidate_indices)
        
        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")
    
    def _retrieve_random(self, candidate_indices: List[int]) -> List[Dict]:
        """Randomly select k examples from candidates."""
        # Sample up to num_context examples
        k = min(self.num_context, len(candidate_indices))
        selected_indices = random.sample(candidate_indices, k)
        
        # Return formatted context examples
        return [
            {
                'gloss': self.dataset_metadata[idx]['gloss'],
                'text': self.dataset_metadata[idx]['text'],
                'id': self.dataset_metadata[idx]['id'],
            }
            for idx in selected_indices
        ]
    
    def _retrieve_similar_from_sample(
        self,
        current_sample: Dict,
        candidate_indices: List[int],
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Select k most similar examples using embeddings, computing current sample embedding on the fly.
        
        Args:
            current_sample: Dict with current sample data
            candidate_indices: List of valid candidate indices
            temperature: Temperature for softmax (lower = sharper distribution)
            
        Returns:
            List of k most similar context examples
        """
        if self.embeddings is None or self.similarity_model is None:
            print("Warning: Embeddings or similarity model not available, falling back to random sampling")
            return self._retrieve_random(candidate_indices)
        
        # Prepare current sample text for embedding
        gloss = current_sample.get('gloss', '').strip()
        text = current_sample.get('text', '').strip()
        combined = f"{gloss} {text}".strip()
        if not combined:
            combined = "unknown"
        
        # Compute embedding for current sample
        current_embedding = self.similarity_model.encode(combined, convert_to_tensor=True).to(self.embeddings.device)
        
        # Compute similarity scores
        candidate_embeddings = self.embeddings[candidate_indices]  # Shape: (K, D)
        
        # Cosine similarity
        current_norm = torch.norm(current_embedding)
        candidate_norms = torch.norm(candidate_embeddings, dim=1)
        
        # Avoid division by zero
        current_norm = current_norm if current_norm > 0 else 1.0
        candidate_norms = torch.where(
            candidate_norms > 0,
            candidate_norms,
            torch.ones_like(candidate_norms)
        )
        
        similarities = torch.matmul(
            current_embedding / current_norm,
            (candidate_embeddings / candidate_norms.unsqueeze(1)).t()
        )  # Shape: (K,)
        
        # Apply temperature and select top-k
        similarities = similarities / temperature
        
        # Convert to probabilities and sample to add diversity (prevent overfitting)
        probs = torch.softmax(similarities, dim=0).cpu().numpy()
        k = min(self.num_context, len(candidate_indices))
        
        # Sample k indices without replacement using probabilities
        selected_relative_indices = np.random.choice(
            len(candidate_indices), size=k, replace=False, p=probs
        )
        
        top_k_absolute_indices = [candidate_indices[i] for i in selected_relative_indices]
        
        return [
            {
                'gloss': self.dataset_metadata[idx]['gloss'],
                'text': self.dataset_metadata[idx]['text'],
                'id': self.dataset_metadata[idx]['id'],
            }
            for idx in top_k_absolute_indices
        ]
    
    @staticmethod
    def format_context(context_examples: List[Dict], format_type: str = 'gloss_text') -> str:
        """
        Format context examples as a string for the prompt.
        
        Args:
            context_examples: List of context dicts
            format_type: One of 'gloss_text', 'text_only', or 'gloss_only'
            
        Returns:
            Formatted context string
        """
        if not context_examples:
            return ""
        
        formatted_parts = []
        
        for example in context_examples:
            if format_type == 'gloss_text':
                # Format: "Gloss: [gloss] | Text: [text]"
                part = f"Gloss: {example['gloss']} | Text: {example['text']}"
            
            elif format_type == 'text_only':
                # Format: "[text]"
                part = example['text']
            
            elif format_type == 'gloss_only':
                # Format: "[gloss]"
                part = example['gloss']
            
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            formatted_parts.append(part)
        
        # Join with newlines or semicolons
        return " | ".join(formatted_parts) if format_type == 'text_only' else "\n".join(formatted_parts)


class DatasetMetadataBuilder:
    """
    Helper class to extract metadata from a PyTorch Dataset.
    """
    
    @staticmethod
    def build_from_dataset(dataset) -> List[Dict]:
        """
        Extract metadata from a dataset.
        
        Args:
            dataset: PyTorch Dataset (Phoenix14T or similar)
            
        Returns:
            List of metadata dicts with keys: 'id', 'text', 'gloss', 'lang'
        """
        metadata = []
        
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                meta = {
                    'id': str(sample.get('id', str(idx))),
                    'text': sample.get('text', ''),
                    'gloss': sample.get('gloss', ''),
                    'lang': sample.get('lang', 'German'),
                }
                if meta['id'] is None:
                    print(f"Warning: Sample {idx} has None id, using fallback '{str(idx)}'")
                    meta['id'] = str(idx)
                metadata.append(meta)
            except Exception as e:
                print(f"Warning: Failed to extract metadata for sample {idx}: {e}")
                continue
        
        return metadata
