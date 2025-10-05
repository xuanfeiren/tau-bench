import numpy as np
import litellm
from tau_bench.retry_utils import auto_retry_with_exponential_backoff
from opto.optimizers.utils import print_color
from typing import List, Tuple
from opto.features.priority_search.priority_search import ModuleCandidate
import time
from opto.features.priority_search.regressor import RegressorTemplate

def get_parameter_text(candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not candidate.update_dict:
            return "base_module_parameters"
        # Convert parameter nodes to readable names for deterministic embedding
        params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
        return str(params_with_names)
        
class PretrainedLinearRegressor(RegressorTemplate):
    """Linear regressor that loads pre-trained weights and bias for score prediction."""
    
    def __init__(self, weights_path: str, bias_path: str, embedding_model="gemini/text-embedding-004", num_threads=20, **kwargs):
        """Initialize with pre-trained weights and bias."""
        self.embedding_model = embedding_model
        
        # Load pre-trained weights and bias
        self.weights = np.load(weights_path)
        self.bias = np.load(bias_path).item()  # bias is a scalar
        
        self.linear_dim = self.weights.shape[0]
        print(f"Loaded regressor: weights shape {self.weights.shape}, bias {self.bias:.6f}")
        self.max_candidates_to_predict = 500
        self.regularization_strength = 0.0001 # Will not be used in this regressor
        self.num_threads = num_threads
        # For random projection if needed (assuming 768D original embeddings)
        self.original_embedding_dim = 768
        self.random_projector = None
    
    def _get_parameter_text(self, candidate):
        """Get the parameter text for a ModuleCandidate."""
        return get_parameter_text(candidate)
    
    def predict_score(self, candidate):
        """Predict score for a single candidate."""
        if not hasattr(candidate, 'embedding'):
            candidate.embedding = self._get_embedding(candidate)
        
        if candidate.embedding is None:
            return 0.0
        
        # Linear prediction: score = weights @ embedding + bias
        embedding_array = np.array(candidate.embedding)
        predicted_score = np.dot(self.weights, embedding_array) + self.bias
        
        # Clip to [0, 1] range
        # predicted_score = np.clip(predicted_score, 0, 1)
        
        candidate.predicted_score = float(predicted_score)
        return float(predicted_score)
    
    def predict_scores_batch(self, candidates):
        """Predict scores for a batch of candidates."""
        scores = []
        for candidate in candidates:
            score = self.predict_score(candidate)
            scores.append(score)
        return scores
    
    # The following two functions will be used in regressor-based PrioritySearch algorithm.
    def update(self, memory: List[Tuple[float, ModuleCandidate]]):
        """
        Pretrained regressor does not need to be updated.
        """
        pass
    def predict_scores(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Predict scores for all candidates in the memory."""
        # Extract all candidates from memory (memory is a list of (neg_score, candidate) tuples)
        print_color("Predicting scores for all candidates in the memory using the pretrained regressor...", "blue")
        if len(memory) == 0:
            return
        batch = [candidate for _, candidate in memory]

        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Collect all embeddings in order
        embeddings = []
        for candidate in batch:
            embeddings.append(candidate.embedding)
        

        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)
        predicted_scores = X_batch @ self.weights + self.bias
        
        # Transform predictions back to [0,1] range
        # predicted_scores = self._inverse_transform_predictions(predicted_scores_transformed)

        # Clip predicted scores to be between 0 and 1
        # predicted_scores = np.clip(predicted_scores, 0, 1)
        
        # Update each candidate with predicted score as attribute
        for candidate, predicted_score in zip(batch, predicted_scores):
            candidate.predicted_score = float(predicted_score)
            
        return predicted_scores


class PretrainedLogisticRegressor(RegressorTemplate):
    """
    Predict scores using embedding logistic regression for ModuleCandidate objects. 
    Should have two key methods: predict_scores and predict_scores_for_batch. 
    predict_scores has no parameters, it could return predicted scores for all candidates in the memory. 
    predict_scores_for_batch has one parameter, a batch of candidates, it could return predicted scores for the batch of candidates."""
    
    def __init__(self, weights_path: str, bias_path: str, embedding_model="gemini/text-embedding-004", num_threads=None):
        self.embedding_model = embedding_model
        # Load pre-trained weights and bias
        self.weights = np.load(weights_path)
        self.bias = np.load(bias_path).item()  # bias is a scalar
        self.linear_dim = self.weights.shape[0]
        self.num_threads = num_threads
        self.regularization_strength = 0.0001 # Will not be used in this regressor
        self.max_candidates_to_predict = 500
        self.random_projector = None
        
    def _sigmoid(self, z):
        """Sigmoid activation function for logistic regression."""
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, memory: List[Tuple[float, ModuleCandidate]]):
        """
        Pretrained logistic regressor does not need to be updated.
        """
        pass

    def predict_scores(self,memory):
        """Predict scores for all candidates in the memory."""
        # Extract all candidates from memory (memory is a list of (neg_score, candidate) tuples)
        if len(memory) == 0:
            return
        batch = [candidate for _, candidate in memory]

        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Collect all embeddings in order
        embeddings = []
        for candidate in batch:
            if candidate.embedding:
                embeddings.append(candidate.embedding)
            else:
                candidate.embedding = self._get_embedding(candidate)
                embeddings.append(candidate.embedding)
        

        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)
        z = X_batch @ self.weights + self.bias
        predicted_scores = self._sigmoid(z)
        
        # Update each candidate with predicted score as attribute
        for candidate, predicted_score in zip(batch, predicted_scores):
            candidate.predicted_score = predicted_score
            
        return predicted_scores