import numpy as np
import litellm
from tau_bench.retry_utils import auto_retry_with_exponential_backoff


class PretrainedLinearRegressor:
    """Linear regressor that loads pre-trained weights and bias for score prediction."""
    
    def __init__(self, weights_path: str, bias_path: str, embedding_model="gemini/text-embedding-004"):
        """Initialize with pre-trained weights and bias."""
        self.embedding_model = embedding_model
        
        # Load pre-trained weights and bias
        self.weights = np.load(weights_path)
        self.bias = np.load(bias_path).item()  # bias is a scalar
        
        self.linear_dim = self.weights.shape[0]
        print(f"Loaded regressor: weights shape {self.weights.shape}, bias {self.bias:.6f}")
        
        # For random projection if needed (assuming 768D original embeddings)
        self.original_embedding_dim = 768
        # if self.linear_dim != self.original_embedding_dim:
        #     from opto.features.priority_search.regressor import GaussianRandomProjection
        #     self.random_projector = GaussianRandomProjection(
        #         input_dim=self.original_embedding_dim,
        #         output_dim=self.linear_dim,
        #         random_seed=42
        #     )
        # else:
            # self.random_projector = None
    
    def _get_parameter_text(self, candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not candidate.update_dict:
            return "base_module_parameters"
        # Convert parameter nodes to readable names for deterministic embedding
        params_with_names = {k.py_name if hasattr(k, 'py_name') else str(k): v for k, v in candidate.update_dict.items()}
        return str(params_with_names)
    
    def _get_embedding(self, candidate):
        """Get the embedding for a ModuleCandidate."""
        parameter_text = self._get_parameter_text(candidate)
        
        def single_embedding_call():
            return litellm.embedding(
                model=self.embedding_model,
                input=parameter_text
            )
        
        try:
            response = auto_retry_with_exponential_backoff(
                single_embedding_call,
                max_retries=10,
                base_delay=1.0,
                operation_name="Embedding API call"
            )
            embedding = response.data[0].embedding
            
            # if self.random_projector is not None:
            #     # Apply random projection
            #     embedding_array = np.array(embedding).reshape(1, -1)
            #     projected = self.random_projector.transform(embedding_array)
            #     embedding = projected.flatten().tolist()
            
            return embedding
        except Exception as e:
            print(f"ERROR: Embedding API call failed: {e}")
            return None
    
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