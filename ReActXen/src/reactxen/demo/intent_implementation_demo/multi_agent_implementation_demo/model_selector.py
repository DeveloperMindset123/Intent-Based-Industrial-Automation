"""
Model Selection by Root Agent
Intelligently selects optimal LLM model based on task requirements and context window needs.
"""
from typing import Dict, List, Any, Optional, Union
from reactxen.utils.model_inference import get_context_length, count_tokens


class ModelSelector:
    """Selects optimal LLM model based on task requirements."""
    
    # Model configurations with context limits and capabilities
    MODEL_CONFIGS = {
        "watsonx": {
            15: {  # Granite-3-2-8b
                "name": "WatsonX Granite-3-2-8b",
                "context_limit": 128000,
                "cost_per_1k_tokens": 0.001,
                "best_for": ["general", "rul_prediction", "fault_classification"],
                "speed": "medium"
            },
            8: {
                "name": "WatsonX Granite-8b",
                "context_limit": 128000,
                "cost_per_1k_tokens": 0.0008,
                "best_for": ["general", "analysis"],
                "speed": "fast"
            }
        },
        "openai": {
            "gpt-4-turbo": {
                "name": "OpenAI GPT-4 Turbo",
                "context_limit": 128000,
                "cost_per_1k_tokens": 0.01,
                "best_for": ["complex_reasoning", "multi_step_tasks"],
                "speed": "slow"
            },
            "gpt-3.5-turbo": {
                "name": "OpenAI GPT-3.5 Turbo",
                "context_limit": 16384,
                "cost_per_1k_tokens": 0.002,
                "best_for": ["simple_tasks", "fast_execution"],
                "speed": "fast"
            }
        },
        "meta-llama": {
            "llama-3-70b": {
                "name": "Meta Llama-3-70b",
                "context_limit": 8192,
                "cost_per_1k_tokens": 0.0005,
                "best_for": ["general", "efficient"],
                "speed": "medium"
            }
        }
    }
    
    def __init__(self):
        self.available_models = self._detect_available_models()
    
    def _detect_available_models(self) -> Dict[str, List[Any]]:
        """Detect which models are available based on environment."""
        available = {
            "watsonx": [],
            "openai": [],
            "meta-llama": []
        }
        
        # WatsonX is always available (default)
        available["watsonx"] = [15, 8]
        
        # Check for OpenAI API key
        import os
        if os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_APIKEY"):
            available["openai"] = ["gpt-3.5-turbo"]
            if os.getenv("OPENAI_API_KEY"):
                available["openai"].append("gpt-4-turbo")
        
        # Meta-Llama via WatsonX (if available)
        # Would need to check WatsonX model list
        
        return available
    
    def select_model(
        self,
        task_type: str,
        estimated_context_size: int = 4096,
        preference: str = "balanced",  # "speed", "accuracy", "cost", "balanced"
        provider_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Select optimal model for task."""
        candidates = []
        
        # Get available models
        for provider, model_ids in self.available_models.items():
            if provider_preference and provider != provider_preference:
                continue
            
            for model_id in model_ids:
                if provider == "watsonx":
                    config = self.MODEL_CONFIGS.get(provider, {}).get(model_id)
                else:
                    config = self.MODEL_CONFIGS.get(provider, {}).get(model_id)
                
                if not config:
                    continue
                
                # Check context window
                if estimated_context_size > config["context_limit"]:
                    continue
                
                # Calculate suitability score
                score = self._calculate_suitability(
                    config, task_type, estimated_context_size, preference
                )
                
                candidates.append({
                    "provider": provider,
                    "model_id": model_id,
                    "config": config,
                    "score": score
                })
        
        if not candidates:
            # Fallback to default
            return {
                "provider": "watsonx",
                "model_id": 15,
                "config": self.MODEL_CONFIGS["watsonx"][15],
                "reason": "No suitable model found, using default"
            }
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        
        return {
            "provider": best["provider"],
            "model_id": best["model_id"],
            "config": best["config"],
            "reason": f"Best match for {task_type} with {preference} preference"
        }
    
    def _calculate_suitability(
        self,
        config: Dict[str, Any],
        task_type: str,
        context_size: int,
        preference: str
    ) -> float:
        """Calculate suitability score for a model."""
        score = 0.0
        
        # Task type match
        if task_type in config.get("best_for", []):
            score += 30.0
        elif "general" in config.get("best_for", []):
            score += 15.0
        
        # Context window efficiency
        context_usage = context_size / config["context_limit"]
        if context_usage < 0.5:
            score += 20.0  # Good headroom
        elif context_usage < 0.8:
            score += 10.0
        
        # Preference-based scoring
        if preference == "speed":
            if config["speed"] == "fast":
                score += 30.0
            elif config["speed"] == "medium":
                score += 15.0
        elif preference == "accuracy":
            if "complex_reasoning" in config.get("best_for", []):
                score += 30.0
        elif preference == "cost":
            # Lower cost is better
            cost_score = max(0, 30.0 - (config["cost_per_1k_tokens"] * 1000))
            score += cost_score
        else:  # balanced
            # Average of all factors
            if config["speed"] == "fast":
                score += 10.0
            if "complex_reasoning" in config.get("best_for", []):
                score += 10.0
            cost_score = max(0, 10.0 - (config["cost_per_1k_tokens"] * 500))
            score += cost_score
        
        return score
    
    def get_model_info(self, provider: str, model_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.MODEL_CONFIGS.get(provider, {}).get(model_id)


def get_model_selector() -> ModelSelector:
    """Get or create model selector instance."""
    if not hasattr(get_model_selector, '_instance'):
        get_model_selector._instance = ModelSelector()
    return get_model_selector._instance

