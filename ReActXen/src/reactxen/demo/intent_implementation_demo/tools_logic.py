from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List, Any, Dict
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_core.prompts import PromptTemplate
from huggingface_hub import HfApi
from datetime import datetime
from load_data import (
    get_reference_test_data,
    get_reference_train_data,
    get_ground_truth,
)
import os
from dotenv import load_dotenv
import logging

load_dotenv(override=False)


# ==========================================================================================
# GLOBAL STATE FOR WATSONX MODELS
class WatsonXAPIState:
    """
    shared state container for watsonx_api instance across tools
    """

    _instance: Optional["watsonx_api"] = None

    @classmethod
    def get_instance(cls) -> Optional["watsonx_api"]:
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "watsonx_api"):
        cls._instance = instance


# GLOBAL STATE FOR HUGGINGFACE MODELS
class HuggingFaceModelState:
    """Shared state for huggingface model catalog"""

    _models_catalog: Dict[str, Any] = {}

    @classmethod
    def get_catalog(cls) -> Dict[str, Any]:
        return cls._models_catalog

    @classmethod
    def set_catalog(cls, catalog: Dict[str, Any]):
        cls._models_catalog = catalog


# ==========================================================================================
# ANY RELEVANT HELPER FUNCTIONS/CLASSES GETS DEFINED HERE
# ==========================================================================================
def train_rul_model_with_watsonx(
    model_id: str, train_data, test_data=None, ground_truth=None
) -> Dict[str, Any]:
    """
    Train/Adapt RUL model using WatsonX foundation model.
    This uses the WatsonX model for inference-based RUL prediction.

    The agent should select the model_id based on data characteristics.
    """
    try:
        watsonx_instance = WatsonXAPIState.get_instance()
        if not watsonx_instance:
            return {
                "error": "WatsonX API not initialized. Call initialize_watsonx_api() first."
            }

        # Set the model_id
        watsonx_instance.set_model_id(model_id)

        # Create model inference instance
        model_inference = ModelInference(
            model_id=model_id,
            project_id=watsonx_instance.project_id,
            api_client=watsonx_instance.create_api_client(),
        )

        # Prepare training data for RUL prediction
        # Convert sensor data to text prompts for LLM-based prediction
        feature_columns = [
            col for col in train_data.columns if col.startswith("sensor_")
        ]

        # Sample a subset for prompt engineering (LLMs work with text, not raw sensor data)
        # In practice, you'd need to convert sensor readings to descriptive text
        sample_data = train_data.head(100)  # Use subset for efficiency

        # Create training prompts (simplified - in practice, you'd engineer better prompts)
        training_prompts = []
        for idx, row in sample_data.iterrows():
            sensor_values = {
                col: row[col] for col in feature_columns[:5]
            }  # Use first 5 sensors
            rul_value = row.get("RUL", 0)
            prompt = f"""Given sensor readings: {sensor_values}, predict the Remaining Useful Life (RUL) in cycles.
The RUL should be approximately {rul_value} cycles."""
            training_prompts.append(prompt)

        # Test the model with a few examples
        test_results = []
        if test_data is not None and ground_truth is not None:
            test_sample = test_data.head(10)
            for idx, row in test_sample.iterrows():
                sensor_values = {col: row[col] for col in feature_columns[:5]}
                test_prompt = f"""Given sensor readings: {sensor_values}, predict the Remaining Useful Life (RUL) in cycles."""

                try:

                    # Use WatsonX model for prediction
                    response = model_inference.generate(
                        prompt=test_prompt, max_new_tokens=50
                    )  # type: ignore
                    test_results.append(
                        {
                            "engine_id": row.get("unit", idx),
                            "prediction": response,
                            "prompt": test_prompt,
                        }
                    )
                except Exception as e:
                    test_results.append(
                        {"engine_id": row.get("unit", idx), "error": str(e)}
                    )

        return {
            "model_id": model_id,
            "model_type": "watsonx_agentic",
            "training_status": "completed",
            "training_samples": len(training_prompts),
            "test_results": test_results,
            "timestamp": datetime.now().isoformat(),
            "note": "WatsonX model used for inference-based RUL prediction. Model selection should be based on data characteristics.",
        }
    except Exception as e:
        return {"error": f"Error training with WatsonX: {str(e)}"}


# def train_rul_model_with_huggingface(
#     model_id: str, train_data, test_data=None, ground_truth=None
# ) -> Dict[str, Any]:
#     """
#     Train RUL model using HuggingFace model.
#     This loads a HuggingFace model and adapts it for RUL prediction.
#     """
#     try:
#         from transformers import AutoModel, AutoTokenizer, AutoConfig
#         import torch
#         import numpy as np

#         # Load model configuration
#         config = AutoConfig.from_pretrained(model_id)
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModel.from_pretrained(model_id)

#         # Prepare training data
#         feature_columns = [
#             col for col in train_data.columns if col.startswith("sensor_")
#         ]
#         X_train = train_data[feature_columns].values
#         y_train = train_data["RUL"].values

#         # For time-series models, you'd need to reshape data appropriately
#         # This is a simplified version - actual implementation would depend on model architecture

#         # Store model info
#         model_info = {
#             "model_id": model_id,
#             "model_type": "huggingface_traditional",
#             "training_status": "completed",
#             "training_samples": len(X_train),
#             "feature_count": len(feature_columns),
#             "model_config": str(config),
#             "timestamp": datetime.now().isoformat(),
#         }

#         # Evaluate if test data and ground truth are available
#         if test_data is not None and ground_truth is not None:
#             X_test = test_data[feature_columns].values
#             # In practice, you'd make predictions here
#             model_info["evaluation_status"] = "test_data_available"

#         return model_info
#     except Exception as e:
#         return {"error": f"Error training with HuggingFace: {str(e)}"}


class watsonx_api:

    def __init__(
        self,
        api_key: str,
        project_id: str,
        project_url: str,
        model_id: Optional[str] = None,
    ):
        """Description:
        The root agent can use this constructor method to instantiate the class with relevant values
        The api_key, project_id, and project_urls are all available within environmental variables that has been set.

        Once the constructor has been set, the LLM can choose what to do with the methods as intended, as the preliminary configuration has been set.
        """
        self.api_key = api_key
        self.project_id = project_id
        self.project_url = project_url
        self.model_id = model_id  # empty initially

    def set_model_id(self, new_model_id: str):
        """sets the model id of the current instance"""
        self.model_id = new_model_id

    def create_api_credentials(self):
        """create and return credentials object"""
        return Credentials(url=self.project_url, api_key=self.api_key)  # type: ignore

    def create_api_client(self):
        """create and return APIClient object"""
        return APIClient(credentials=self.create_api_credentials())  # type: ignore

    def get_chat_models_list(self):
        """Retrieves list of ChatModels's ids within the foundation_models catalog.
        Returns a list of available model IDs that the agent can choose from.

        NOTE : These aren't being picked up correctly
        """
        try:
            return [
                models_enum.value
                for models_enum in self.create_api_client().foundation_models.ChatModels  # type: ignore
            ]

        except Exception as e:
            return f"Error retrieving models: {str(e)}"

    def get_model_details(self, model_id: Optional[str]):
        """retrieves list of models based on given model_id.
        If model_id is not provided, uses self.model_id.

        The root agent can use this method to obtain detailed info per each model_id,
        this will allow for the agent to determine the optimal model to use to solve the given problem, basically assigns a suitable model_id.
        """
        try:

            # check if a model_id is present
            target_model_id = self.model_id or model_id
            if not target_model_id:
                return "Error: no model_id provided. Use set_model_id() first or provide the model_id parameter before obtaining model details."

            return ModelInference(
                model_id=self.model_id,
                project_id=self.project_id,
                api_client=self.create_api_client(),
            ).get_details()
        except Exception as e:
            return f"Error retrieving model details : {str(e)}"


# ==========================================================================================
# END OF RELEVANT HELPER FUNCTIONS/CLASSES
# ==========================================================================================


# ===============================================================================================
# START OF PYDANTIC BASE MODEL SCHEMAS
# ===============================================================================================
# Pydantic input schema for tool inputs
class InitializeWatsonXInput(BaseModel):
    """Pydantic schemas that serves as input schemas for initializing the WatsonX API"""

    api_key: Optional[str] = Field(
        default=None,
        description="API Key for WatsonX. If not provided, uses WATSONX_APIKEY env var.",
    )

    project_id: Optional[str] = Field(
        default=None,
        description="Project ID for WatsonX. If not provided, uses WATSONX_PROJECT_ID env var instead.",
    )

    project_url: Optional[str] = Field(
        default=None,
        description="Project URL for WatsonX. If not provided, uses WATSONX_URL env var.",
    )

    model_id: Optional[str] = Field(
        default=None, description="Optional initial model_id, which can be set later."
    )


class SetModelIDInput(BaseModel):
    """
    Input schema involving setting model id
    """

    model_id: str = Field(description="The model id to set for WatsonX API")


class GetModelDetailsInput(BaseModel):
    """
    Input schema for getting model details
    """

    model_id: Optional[str] = Field(
        default=None,
        description="Model ID to get details for. If not provided, uses currently set model_id.",
    )


# create an empty pydantic model for tools that doesn't have any arguments
class EmptyInput(BaseModel):
    """This is an empty input schema that doesn't require any parameters"""

    pass


# similar empty pydantic scehma that only references the BaseModel, since it cannot be referenced directly


# These are empty BaseModel input for train, test and ground
# NOTE : intentionally left empty
class LoadTrainDataInput(BaseModel):
    """Input schema for loading training data - no parameter needed"""

    pass


class LoadTestDataInput(BaseModel):
    """Input schema for loading test data - no parameter needed"""

    pass


class LoadGroundTruthInput(BaseModel):
    """Input schema for loading ground truth data - no parameter needed"""

    pass


class RetrieveModelsInput(BaseModel):
    """Input schema for retrieving ML models from huggingface"""

    task: str = Field(
        default="time-series-forecasting",
        description="ML task type (e.g. 'time-series-forecasting', 'regression')",
    )
    library: str = Field(
        default="transformers",
        description="Library filter (e.g. 'transformers', 'sklearn')",
    )
    limit: int = Field(default=10, description="Maximum number of models to retrieve")
    sort_by: str = Field(
        default="downloads",  # Fixed: Changed from default=10 to default="downloads"
        description="Sorting criteria ('downloads', 'likes', 'recent')",
    )
    min_downloads: int = Field(default=1000, description="Minimum download thresholds")


class SelectOptimalModelInput(BaseModel):
    """Input schema for selecting optimal model"""

    task_requirements: str = Field(
        description="Description of the task requirement for model selection"
    )

    performance_weight: float = Field(
        default=0.4, description="Weight for performance metrics (0-1)"
    )

    popularity_weight: float = Field(
        default=0.3, description="Weight for popularity metrics (0-1)"
    )

    recency_weight: float = Field(
        default=0.3, description="Weight for recency metrics (0-1)"
    )


class TrainAgenticModelInput(BaseModel):
    """Input schema for agentic model training"""

    model_type: str = Field(
        description="Model type: 'agentic' (uses WatsonX models) or 'traditional' (uses HuggingFace models)"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional: Specific model_id to use. If not provided, agent will select based on data.",
    )
    task_description: Optional[str] = Field(
        default="RUL prediction for industrial equipment",
        description="Description of the RUL prediction task (helps in model selection)",
    )


class CostEstimationInput(BaseModel):
    """Input schema for cost estimation"""

    engine_id: Optional[int] = Field(
        default=None, description="Specific engine ID for cost estimation (optional)"
    )

    maintenance_type: str = Field(
        description="Type of maintenance : 'ROUTINE_SURVELLIANCE', 'PROACTIVE_INSPECTION', 'CORRECTIVE_ACTION', or 'IMMEDIATE_GROUNDING'"
    )

    estimated_hours: float = Field(
        default=8.0, description="Estimated maintenance hours, could include overtime."
    )

    labor_rate: Optional[float] = Field(
        default=None,
        description="Labor rate per hour (will search online if not provided)",
    )


class CostBenefitAnalysisInput(BaseModel):
    """Input schema for cost benefit analysis"""

    engine_ids: List[int] = Field(description="List of engine IDs to analyze")

    maintenance_threshold: int = Field(
        default=20, description="RUL threshold for maintenance consideration"
    )

    include_safety_costs: bool = Field(
        default=True, description="Include safety-related costs in analysis"
    )


# ===============================================================================================
# End of pydantic BaseModel definition
# ===============================================================================================

# ===============================================================================================
# START OF LANGCHAIN BASETOOLS DEFINITIONS
# ===============================================================================================


# define tool for estimating maintenance cost of engines
class CostEstimationTool(BaseTool):
    """Estimate maintenance costs for engines"""

    name: str = "estimate_maintenance_cost"
    description: str = """Estimate maintenance costs including labor and materials.
    
    This tool:
    - Estimates labor costs based on maintenance type and hours
    - Includes overhead costs (insurance, benefits, facilities)
    - Estimates material and equipment costs
    - Can search online for current labor rates if not provided
    
    Use brave_search or duckduckgo_search to get current labor rates and material costs
    before calling this tool for accurate estimates.
    """
    args_schema: Type[BaseModel] = CostEstimationInput  # type: ignore

    def _run(
        self,
        maintenance_type: str,
        estimated_hours: float = 8.0,
        engine_id: Optional[int] = None,
        labor_rate: Optional[float] = None,
    ) -> str:
        try:
            # Default labor rates (can be overridden by search)
            default_rates = {
                "ROUTINE_SURVEILLANCE": 50.0,
                "PROACTIVE_INSPECTION": 75.0,
                "CORRECTIVE_ACTION": 100.0,
                "IMMEDIATE_GROUNDING": 150.0,
            }

            base_rate = labor_rate or default_rates.get(maintenance_type, 75.0)
            overhead_rate = 1.4  # 40% overhead (insurance, benefits, facilities)

            # Calculate labor costs
            labor_cost = base_rate * estimated_hours * overhead_rate

            # Material costs based on maintenance type
            material_multipliers = {
                "ROUTINE_SURVEILLANCE": 0.1,
                "PROACTIVE_INSPECTION": 0.3,
                "CORRECTIVE_ACTION": 0.6,
                "IMMEDIATE_GROUNDING": 1.0,
            }
            material_cost = labor_cost * material_multipliers.get(maintenance_type, 0.5)

            # Equipment and vehicle costs
            equipment_cost = estimated_hours * 25.0  # $25/hour for equipment

            total_cost = labor_cost + material_cost + equipment_cost

            result = f"""✅ Cost Estimation for {maintenance_type}
            
Engine ID: {engine_id if engine_id else 'N/A'}
Labor Cost: ${labor_cost:,.2f} ({base_rate}/hr × {estimated_hours}hrs × {overhead_rate} overhead)
Material Cost: ${material_cost:,.2f}
Equipment Cost: ${equipment_cost:,.2f}
Total Cost: ${total_cost:,.2f}

Note: Use search tools to get current labor rates for more accurate estimates."""

            return result
        except Exception as e:
            return f"Error estimating costs: {str(e)}"


class CostBenefitAnalysisTool(BaseTool):
    """Perform cost-benefit analysis for multiple engines"""

    name: str = "cost_benefit_analysis"
    description: str = """Perform comprehensive cost-benefit analysis for engines at risk.
    
    This tool:
    - Identifies engines requiring maintenance
    - Estimates costs for each engine
    - Calculates benefit (prevented failure costs)
    - Provides cost-benefit ratios
    - Includes safety considerations
    
    Use this after identifying engines at risk with get_engines_at_risk().
    """
    args_schema: Type[BaseModel] = CostBenefitAnalysisInput

    def _run(
        self,
        engine_ids: List[int],
        maintenance_threshold: int = 20,
        include_safety_costs: bool = True,
    ) -> str:
        try:
            # Load ground truth to get RUL values
            if "ground_truth" not in globals() or ground_truth is None:
                ground_truth_data = get_ground_truth()
            else:
                ground_truth_data = ground_truth

            # Load predictions if available
            try:
                predictions = predict_rul("random_forest")
            except:
                predictions = None

            analysis_results = []
            total_maintenance_cost = 0
            total_prevented_cost = 0

            # Failure cost estimates (cost of unplanned failure)
            failure_cost = 500000  # $500k for catastrophic failure
            downtime_cost_per_day = 10000  # $10k/day downtime

            for engine_id in engine_ids:
                # Get RUL
                if predictions is not None:
                    # Find prediction for this engine
                    try:
                        engine_data = test_data[test_data["unit"] == engine_id]
                        if not engine_data.empty:
                            last_idx = engine_data.index[-1]
                            predicted_rul = (
                                predictions[last_idx]
                                if last_idx < len(predictions)
                                else maintenance_threshold
                            )
                        else:
                            predicted_rul = maintenance_threshold
                    except:
                        predicted_rul = maintenance_threshold
                else:
                    # Use ground truth if available
                    try:
                        engine_gt = ground_truth_data[
                            ground_truth_data["unit"] == engine_id
                        ]
                        predicted_rul = (
                            engine_gt["RUL"].iloc[0]
                            if not engine_gt.empty
                            else maintenance_threshold
                        )
                    except:
                        predicted_rul = maintenance_threshold

                # Determine maintenance type
                if predicted_rul <= 10:
                    maintenance_type = "IMMEDIATE_GROUNDING"
                    estimated_hours = 24.0
                elif predicted_rul <= 20:
                    maintenance_type = "CORRECTIVE_ACTION"
                    estimated_hours = 16.0
                elif predicted_rul <= 50:
                    maintenance_type = "PROACTIVE_INSPECTION"
                    estimated_hours = 8.0
                else:
                    maintenance_type = "ROUTINE_SURVEILLANCE"
                    estimated_hours = 4.0

                # Estimate costs
                cost_tool = CostEstimationTool()
                cost_result = cost_tool._run(
                    maintenance_type=maintenance_type,
                    estimated_hours=estimated_hours,
                    engine_id=engine_id,
                )

                # Extract cost (simplified - in practice, parse the result)
                # For now, use estimates
                cost_estimates = {
                    "IMMEDIATE_GROUNDING": 15000,
                    "CORRECTIVE_ACTION": 8000,
                    "PROACTIVE_INSPECTION": 4000,
                    "ROUTINE_SURVEILLANCE": 1500,
                }
                maintenance_cost = cost_estimates.get(maintenance_type, 5000)

                # Calculate benefit (prevented failure cost)
                # Benefit increases as RUL decreases (more urgent)
                urgency_factor = max(
                    0, (maintenance_threshold - predicted_rul) / maintenance_threshold
                )
                prevented_cost = failure_cost * urgency_factor + (
                    downtime_cost_per_day * 5 * urgency_factor
                )

                benefit_cost_ratio = (
                    prevented_cost / maintenance_cost if maintenance_cost > 0 else 0
                )

                analysis_results.append(
                    {
                        "engine_id": engine_id,
                        "predicted_rul": predicted_rul,
                        "maintenance_type": maintenance_type,
                        "maintenance_cost": maintenance_cost,
                        "prevented_cost": prevented_cost,
                        "benefit_cost_ratio": benefit_cost_ratio,
                    }
                )

                total_maintenance_cost += maintenance_cost
                total_prevented_cost += prevented_cost

            # Generate summary
            summary = f"""✅ Cost-Benefit Analysis Results
            
Engines Analyzed: {len(engine_ids)}
Total Maintenance Cost: ${total_maintenance_cost:,.2f}
Total Prevented Cost: ${total_prevented_cost:,.2f}
Net Benefit: ${total_prevented_cost - total_maintenance_cost:,.2f}
Overall Benefit-Cost Ratio: {total_prevented_cost / total_maintenance_cost:.2f}:1

Detailed Results:
"""
            for result in analysis_results[:10]:  # Show first 10
                summary += f"""
Engine {result['engine_id']}:
  RUL: {result['predicted_rul']:.1f} cycles
  Maintenance: {result['maintenance_type']}
  Cost: ${result['maintenance_cost']:,.2f}
  Prevented Cost: ${result['prevented_cost']:,.2f}
  Benefit-Cost Ratio: {result['benefit_cost_ratio']:.2f}:1
"""

            if len(analysis_results) > 10:
                summary += f"\n... and {len(analysis_results) - 10} more engines"

            return summary
        except Exception as e:
            return f"Error performing cost-benefit analysis: {str(e)}"


# define tool to retrieve ML models
class RetrieveMLModelsTool(BaseTool):
    """Retrieve ML models from huggingface hub for RUL prediction tasks"""

    name: str = "retrieve_ml_models"
    description: str = """Retrieve ML models from huggingface hub dynamically
    
    This tool allows the agent to search for appropriate models based on task type, library, and popularity metrics. The agent can use brave_search or duckduckgo_search to find appropriate task description to insert as parameter online before calling on this tool.
    
    Returns a catalog of models with performance metrics that can be used for model selection.
    """
    args_schema: Type[BaseModel] = RetrieveModelsInput

    def _run(
        self,
        task: str = "time-series forecasting",
        library: str = "transformers",
        limit: int = 10,
        sort_by: str = "downloads",
        min_downloads: int = 1000,
    ) -> str:
        try:
            api = HfApi()
            models_catalog = {}
            models_info = api.list_models(
                filter=[task, library], sort_by=sort_by, direction=-1, limit=limit
            )
            for model in models_info:
                model_id = str(model.model_id)
                try:
                    model_details = api.model_info(model_id)
                    if model_details.downloads < min_downloads:
                        continue

                    model_info = {
                        "model_id": model_id,
                        "name": model_details.modelId,
                        "downloads": model_details.downloads,
                        "likes": model_details.likes,
                        "library_name": model_details.library_name,
                        "pipeline_tag": model_details.pipeline_tag,
                        "tags": model_details.tags,
                        "created_at": model_details.created_at,
                        "total_downloads": model_details.downloads_all_time,
                        "last_modified": model_details.last_modified,
                    }

                    # KV store - model info mapped against the model_id
                    models_catalog[model_id] = model_info

                # output error and continue if the error continues to persist
                except Exception as e:
                    logging.warning(f"Failed to get details for model {model_id} : {e}")
                    continue

            # store in shared state
            HuggingFaceModelState.set_catalog(models_catalog)

            if not models_catalog:
                return "No models found matching the criteria. Try adjusting the parameters."

            models_summary = "\n".join(
                [
                    f"  - {info['model_id']}: {info['downloads']} downloads, {info['likes']} likes"
                    for info in list(models_catalog.values())[:10]
                ]
            )

            return f"✅ Retrieved {len(models_catalog)} models:\n{models_summary}\n\nUse select_optimal_model() to choose the best model for your task."
        except Exception as e:
            logging.warning(f"Error retrieving ML models : {str(e)}")


class SelectOptimalModelTool(BaseTool):
    """Select optimal model from catalog based on multiple criteria"""

    name: str = "select_optimal_model"
    description: str = """Select the optimal model from the retrieved catalog based on task requirements.
    
    This tool uses a scoring system that considers:
    - Performance metrics (downloads, likes)
    - Popularity (user adoption)
    - Recency (how recently the model was updated)
    
    Returns the selected model ID that best matches the requirement
    """
    args_schema: Type[BaseModel] = SelectOptimalModelInput

    def _run(
        self,
        task_requirements: str,  # Fixed: Changed from task_requirement to task_requirements
        performance_weight: float = 0.4,
        popularity_weight: float = 0.3,
        recency_weight: float = 0.3,
    ) -> str:
        try:
            models_catalog = HuggingFaceModelState.get_catalog()

            if not models_catalog:
                return "Error: No models catalog available. Call retrieve_ml_models() first."

            model_scores = {}
            for model_id, model_info in models_catalog.items():
                downloads_score = min(model_info.get("downloads", 0) / 1000000, 1.0)
                likes_score = min(model_info.get("likes", 0) / 10000, 1.0)

                try:
                    last_modified = datetime.fromisoformat(
                        model_info.get("last_modified", "").replace("Z", "+00:00")
                    )
                    days_old = (
                        datetime.now() - last_modified.replace(tzinfo=None)
                    ).days
                    recency_score = max(0, 1 - (days_old / 365))
                except:
                    recency_score = 0.5

                composite_score = (
                    popularity_weight * (downloads_score + likes_score) / 2
                    + recency_weight * recency_score
                    + performance_weight * 0.5
                )

                model_scores[model_id] = {
                    "score": composite_score,
                    "downloads_score": downloads_score,
                    "likes_score": likes_score,
                    "recency_score": recency_score,
                    "model_info": model_info,
                }

            # Fixed: Changed model_scores.item() to model_scores.items()
            sorted_models = sorted(
                model_scores.items(), key=lambda x: x[1]["score"], reverse=True
            )

            if sorted_models:
                selected_model_id = sorted_models[0][0]
                score_info = sorted_models[0][1]

                return f"✅ Selected model: {selected_model_id}\nScore: {score_info['score']:.3f}\nDownloads: {score_info['model_info']['downloads']:,}\nLikes: {score_info['model_info']['likes']}"

            return "Error: No suitable model found."
        except Exception as e:
            return f"Error selecting optimal model: {str(e)}"


# =============================================================================
# HELPER FUNCTIONS FOR MODEL TRAINING
# =============================================================================


class TrainAgenticModelTool(BaseTool):
    """
    Train RUL model using agentic or traditional approach.

    - Agentic: Uses WatsonX foundation models for inference-based RUL prediction
    - Traditional: Uses HuggingFace models for traditional ML-based RUL prediction
    """

    name: str = "train_agentic_model"
    description: str = """Train RUL prediction model using agentic or traditional approach.
    
    Model Types:
    - 'agentic': Uses WatsonX foundation models. The agent should:
      1. Initialize WatsonX API (if not already done)
      2. Get available chat models using get_chat_models_list()
      3. Select appropriate model_id based on data characteristics
      4. Fine-tune/adapt the model with training data
      5. Test with training data
      6. Validate against ground truth
      
    - 'traditional': Uses HuggingFace models. The agent should:
      1. Retrieve models using retrieve_ml_models()
      2. Select optimal model using select_optimal_model()
      3. Load and train the HuggingFace model with training data
      4. Test and validate against ground truth
    
    The agent can choose the model_id dynamically based on the data it has loaded.
    If model_id is not provided, the agent should select it based on task_description and data characteristics.
    """
    args_schema: Type[BaseModel] = TrainAgenticModelInput

    def _run(
        self,
        model_type: str,
        model_id: Optional[str] = None,
        task_description: Optional[str] = "RUL prediction for industrial equipment",
    ) -> str:
        try:
            # Validate model_type
            if model_type not in ["agentic", "traditional"]:
                return f"Error: model_type must be 'agentic' or 'traditional', got '{model_type}'"

            # Load training data
            try:
                if "train_data" not in globals() or train_data is None:
                    train_data = get_reference_train_data()
                else:
                    train_data = globals()["train_data"]
            except:
                return (
                    "Error: Training data not available. Call load_train_data() first."
                )

            # Load test data and ground truth if available
            test_data = None
            ground_truth = None
            try:
                if "test_data" in globals() and globals()["test_data"] is not None:
                    test_data = globals()["test_data"]
                else:
                    test_data = get_reference_test_data()
            except:
                pass  # Test data optional

            try:
                if (
                    "ground_truth" in globals()
                    and globals()["ground_truth"] is not None
                ):
                    ground_truth = globals()["ground_truth"]
                else:
                    ground_truth = get_ground_truth()
            except:
                pass  # Ground truth optional

            # AGENTIC APPROACH: Use WatsonX models
            if model_type == "agentic":
                watsonx_instance = WatsonXAPIState.get_instance()
                if not watsonx_instance:
                    return "Error: WatsonX API not initialized. Call initialize_watsonx_api() first. Then use get_chat_models_list() to see available models and select an appropriate model_id."

                # If model_id not provided, use the one set in WatsonX instance
                if not model_id:
                    model_id = watsonx_instance.model_id
                    if not model_id:
                        return "Error: No model_id provided and no model_id set in WatsonX instance. Use set_model_id() or get_chat_models_list() to select a model first."

                # Train with WatsonX
                result = train_rul_model_with_watsonx(
                    model_id=model_id,
                    train_data=train_data,
                    test_data=test_data,
                    ground_truth=ground_truth,
                )

                if "error" in result:
                    return f"Error training with WatsonX: {result['error']}"

                return f"""✅ Agentic model training completed!

Model ID: {result['model_id']}
Model Type: {result['model_type']}
Training Samples: {result['training_samples']}
Status: {result['training_status']}
Timestamp: {result['timestamp']}

Next Steps:
1. Use predict_rul() to make predictions on test data
2. Use evaluate_model() to validate against ground truth
3. Use get_engines_at_risk() to identify engines requiring attention

Note: The agent selected model_id '{model_id}' based on the data characteristics.
You can use get_model_details() to see why this model was suitable."""

            # TRADITIONAL APPROACH: Use HuggingFace models
            elif model_type == "traditional":
                # Check if models catalog is available
                models_catalog = HuggingFaceModelState.get_catalog()
                if not models_catalog and not model_id:
                    return "Error: No HuggingFace models catalog available and no model_id provided. Call retrieve_ml_models() first, then use select_optimal_model() to choose a model, or provide model_id directly."

                # If model_id not provided, get the first one from catalog (or agent should select)
                if not model_id:
                    if models_catalog:
                        # Get the first model from catalog (in practice, agent would use select_optimal_model)
                        model_id = list(models_catalog.keys())[0]
                        return f"Warning: No model_id provided. Using first model from catalog: {model_id}. For better results, use select_optimal_model() first to choose the best model based on task requirements."
                    else:
                        return "Error: No model_id provided and no models catalog available. Call retrieve_ml_models() first."

                # Train with HuggingFace
                result = train_rul_model_with_huggingface(
                    model_id=model_id,
                    train_data=train_data,
                    test_data=test_data,
                    ground_truth=ground_truth,
                )

                if "error" in result:
                    return f"Error training with HuggingFace: {result['error']}"

                return f"""✅ Traditional model training completed!

Model ID: {result['model_id']}
Model Type: {result['model_type']}
Training Samples: {result['training_samples']}
Features: {result['feature_count']}
Status: {result['training_status']}
Timestamp: {result['timestamp']}

Next Steps:
1. Use predict_rul() to make predictions on test data
2. Use evaluate_model() to validate against ground truth
3. Use get_engines_at_risk() to identify engines requiring attention

Note: This model was selected from HuggingFace catalog. For better model selection, use retrieve_ml_models() and select_optimal_model() based on your task requirements."""

        except Exception as e:
            return f"Error training model: {str(e)}"


### CREATE HUGGINGFACE TOOL (and the watsonx agentic approach)
def create_huggingface_tools() -> List[BaseTool]:
    """creates and returns a list of huggingface base tools"""
    return [
        RetrieveMLModelsTool(),
        SelectOptimalModelTool(),
        TrainAgenticModelTool(),
    ]


# Base Tool Implementation using langchain_core
class InitializeWatsonXTool(BaseTool):
    """Initialize the watsonx api instance with credentials and must be called first."""

    name: str = "initialize_watsonx_api"
    description: str = """Initialize the WatsonX API instance with API Key, project ID and project URL.
    
    This must be called first before using other WatsonX tools. Uses environment variables if parameters are not provided.
    
    Example implementation:
    initialize_watsonx_api(api_key="your_api_key", project_id="your_project_id", project_url="https://us-south.ml.cloud.ibm.com/")
    """
    args_schema: Type[BaseModel] = InitializeWatsonXInput

    # private method - basically sets the parameter here
    def _run(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        project_url: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        try:
            # use environmental variables as default
            api_key = api_key or os.environ.get("WATSONX_APIKEY")
            project_id = project_id or os.environ.get("WATSONX_PROJECT_ID")
            project_url = project_url or os.environ.get(
                "WATSONX_URL", "https://us-south.ml.cloud.ibm.com/"
            )

            # validation check for api_key and project_id
            if not api_key or not project_id:
                return "Error : API Key and project ID are required. Provide them as parameter or set WATSONX_APIKEY and WATSONX_PROJECTID environment variables."

            # create and store instance in shared state
            watsonx_instance = watsonx_api(
                api_key=api_key,
                project_id=project_id,
                project_url=project_url,
                model_id=model_id,
            )
            WatsonXAPIState.set_instance(watsonx_instance)
            return f"✅ WatsonX API initialized successfully. Project ID : {project_id}, URL : {project_url}, Model ID : {model_id or 'Not set (use get_chat_model_list to see available models)'}"
        except Exception as e:
            return f"Error initializing WatsonX API : {str(e)}"


class GetChatModelsListTool(BaseTool):
    """
    Get list of available chat models from WatsonX foundation models catalog.
    """

    name: str = "get_chat_models_list"
    description: str = """Retrieve a list of available ChatModels IDs from WatsonX foundation models catalog. The agent can use this to see what models are available and then select an appropriate one.
    
    Returns a list of model IDs that can be used with set_model_id() or get_model_details()
    """
    args_schema: Type[BaseModel] = EmptyInput  # additional arguments not needed

    def _run(self) -> str:
        try:
            instance = WatsonXAPIState.get_instance()
            if not instance:
                return f"Error : WatsonX API not initialized. Call initialize_watsonx_api() first."
            models = instance.get_chat_models_list()
            if isinstance(models, list):
                models_str = "\n".join([f"  - {model}" for model in models])
                return f"Available ChatModels ({len(models)} models):\n{models_str}\n\nYou can use any of these model IDs with set_model_id() or get_model_details()."
            else:
                return str(models)
        except Exception as e:
            return f"Error retrieving chat models list : {str(e)}"


class SetModelIDTool(BaseTool):
    """
    Set the model ID for the WatsonX API instance.
    """

    name: str = "set_model_id"
    description: str = """
    Set the modelID for the WatsonX API operations.
    
    use get_chat_models_list() first to see available models, then set the desired model ID.
    
    example usage:
    Example: set_model_id(model_id="ibm/granite-3-2-8b-instruct")
    """
    args_schema: Type[BaseModel] = SetModelIDInput

    def _run(self, model_id: str) -> str:
        try:
            instance = WatsonXAPIState.get_instance()
            if not instance:
                return "Error : WatsonX API not initialized. Call initialize_watsonx_api() first."
            result = instance.set_model_id(model_id)
            return f"✅ {result}"
        except Exception as e:
            return f"Error setting model ID : {str(e)}"


class GetModelDetailsTool(BaseTool):
    """
    Get detailed information about a specific model
    """

    name: str = "get_model_details"
    description: str = """retrieve detailed information about a specific watsonX model. This helps the agent evaluate which model is suited for the task.
    
    If model_id is not provided, uses the currently set model_id.
    Example: get_model_details(model_id="ibm/granite-3-2-8b-instruct")
    """
    args_schema: Type[BaseModel] = GetModelDetailsInput

    def _run(self, model_id: Optional[str] = None) -> str:
        try:
            instance = WatsonXAPIState.get_instance()
            if not instance:
                return "Error : WatsonX API not initialized. Call initialize_watsonx_api() first."

            result = instance.get_model_details(model_id)
            return result
        except Exception as e:
            return f"Error getting model details : {str(e)}"


# Data loading tools that references the Pydantic base model
class LoadTrainDataTool(BaseTool):
    """
    Load and pre-process the training data from CMAPSS dataset
    This tool loads the training data, calculates RUL for each engine, and returns a summary of the loaded data including shape, number of engines and RUL statistics
    """

    name: str = "load_train_data"
    description: str = """Load and preprocess the training data from CMAPSS dataset.
    
    This tool:
    - Loads training data from CMAPSS FD001 dataset
    - Calculates Remaining Useful Life (RUL) for each engine cycle
    - Returns Summary statistics including data shape, number of engines and RUL range
    
    Returns a string with data summary information.
    No Parameters required.
    """
    args_schema: Type[BaseModel] = LoadTrainDataInput

    def _run(self) -> str:
        """Executes the tool to load training data"""
        try:
            # call the existing function to retrieve the data
            data = get_reference_train_data()

            if data is None or data.empty:
                return "Error : Failed to load training data or data is empty"

            # Otherwise, generate comprehensive summary
            summary = f"""
             Data Statistics:
- Total samples: {len(data):,}
- Number of features: {data.shape[1]}
- Number of engines: {data['unit'].nunique()}
- RUL range: {data['RUL'].min():.0f} to {data['RUL'].max():.0f} cycles
- Mean RUL: {data['RUL'].mean():.2f} cycles
- Median RUL: {data['RUL'].median():.0f} cycles

Columns: {', '.join(data.columns.tolist()[:10])}{'...' if len(data.columns) > 10 else ''}

The training data is now available in the global scope and can be used for model training."""

            return summary
        except FileNotFoundError as e:
            return f"Error: Training data file not found. {str(e)}"
        except Exception as e:
            return f"Error loading training data: {str(e)}"


class LoadTestDataTool(BaseTool):
    """
    Load and pre-process the test data from CMAPSS dataset
    This tool loads the test data which contains the current sensor readings for engines at various stages of their lifecycle, without RUL labels.
    """

    name: str = "load_test_data"  # don't add comma, will lead to tuple being created
    description: str = """Load and preprocess the test data from CMAPSS dataset
    
    This tool:
    - Loads test data from CMAPSS FD001 dataset
    - Contains current sensor readings for engines at various lifecycle stages
    - Returns summary statistics including data shape and number of engines
    
    Returns a string with data summary information.
    """
    args_schema: Type[BaseModel] = LoadTestDataInput

    def _run(self) -> str:
        """Execute the tool to load test data"""
        try:
            # Call the existing function
            data = get_reference_test_data()

            if data is None or data.empty:
                return "Error: Failed to load test data or data is empty"

            # Generate comprehensive summary
            summary = f"""✅ Test data loaded successfully!

Data Statistics:
- Total samples: {len(data):,}
- Number of features: {data.shape[1]}
- Number of engines: {data['unit'].nunique()}
- Time cycles per engine: {data.groupby('unit')['time'].max().describe().to_dict()}

Columns: {', '.join(data.columns.tolist()[:10])}{'...' if len(data.columns) > 10 else ''}

The test data is now available in the global scope and can be used for RUL predictions."""

            return summary
        except FileNotFoundError as e:
            return f"Error: Test data file not found. {str(e)}"
        except Exception as e:
            return f"Error loading test data: {str(e)}"


class LoadGroundTruthTool(BaseTool):
    """
    Load the ground truth RUL values for test data.
    This tool loads the actual remaining useful life values for each test engine,
    which are used to evaluate prediction accuracy.
    """

    name: str = "load_ground_truth"
    description: str = """Load the ground truth RUL values for test data.
    
    This tool:
    - Loads ground truth RUL values from the CMAPSS FD001 dataset
    - Contains actual remaining useful life for each test engine
    - Returns summary statistics including number of engines and RUL distribution
    
    Returns a string with data summary information.
    No parameters required.
    """
    args_schema: Type[BaseModel] = LoadGroundTruthInput

    def _run(self) -> str:
        """Execute the tool to load ground truth data"""
        try:
            # Call the existing function
            data = get_ground_truth()

            if data is None or data.empty:
                return "Error: Failed to load ground truth data or data is empty"

            # Generate comprehensive summary with risk analysis
            at_risk_count = len(data[data["RUL"] <= 20])
            critical_count = len(data[data["RUL"] <= 10])

            summary = f"""✅ Ground truth data loaded successfully!

Data Statistics:
- Number of test engines: {len(data)}
- RUL range: {data['RUL'].min():.0f} to {data['RUL'].max():.0f} cycles
- Mean RUL: {data['RUL'].mean():.2f} cycles
- Median RUL: {data['RUL'].median():.0f} cycles

Risk Analysis:
- Engines with RUL ≤ 20 cycles: {at_risk_count} ({at_risk_count/len(data)*100:.1f}%)
- Engines with RUL ≤ 10 cycles (CRITICAL): {critical_count} ({critical_count/len(data)*100:.1f}%)

The ground truth data is now available in the global scope and can be used for model evaluation."""

            return summary
        except FileNotFoundError as e:
            return f"Error: Ground truth data file not found. {str(e)}"
        except Exception as e:
            return f"Error loading ground truth data: {str(e)}"


# LOAD SEARCH ENGINE TOOLS - BraveSearch and DuckDuckGo
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun

try:
    brave_api_key = os.environ.get("BRAVE_API_KEY", "")
    if brave_api_key:
        brave_search_tool = BraveSearch.from_api_key(api_key=brave_api_key)
        print("✅ BraveSearch tool initialized successfully.")

    else:
        print(
            "⚠️  BRAVE_API_KEY not found in environment variables. BraveSearch will not be available."
        )
        brave_search_tool = None
except Exception as e:
    print(f"⚠️  Error initializing BraveSearch: {str(e)}")
    brave_search_tool = None

# INIT DUCKDUCKGO search tool
try:
    duckduckgo_search_tool = DuckDuckGoSearchRun()
    print("✅ DuckDuckGoSearch tool initialized successfully")
except Exception as e:
    print(f"⚠️ Error initializing DuckDuckGoSearch : {str(e)}")
    duckduckgo_search_tool = None


def create_data_preprocessing_tools() -> list:
    """
    creates and returns a list of data preprocessing tools.
    Similar to create_watsonx_tools(), this function returns all data loading tools
    """
    return [LoadTrainDataTool(), LoadTestDataTool(), LoadGroundTruthTool()]


# Function to create the data pre-processing tools
data_preprocessing_tools = create_data_preprocessing_tools()


# create list of tools based on the tools defined above
# the _run() method allows for the classes to be executed as a function, with pydantic schema definition as input parameters.
def create_watsonx_tools() -> List[BaseTool]:
    """create and return list of WatsonX API tools, higher order function that calls on the"""
    return [
        InitializeWatsonXTool(),
        GetChatModelsListTool(),
        SetModelIDTool(),
        GetModelDetailsTool(),
    ]


def create_cost_benefit_analysis_tools():
    """creates and returns list of available tools related to costs"""
    return [CostEstimationTool(), CostBenefitAnalysisTool()]


watsonx_tools = create_watsonx_tools()

# create a consolidated version of all the tools available
all_tools = []

# add data pre-processing tools
all_tools.extend(data_preprocessing_tools)
print(f"Added {len(data_preprocessing_tools)} data preprocessing tools")

# add watsonX tools
all_tools.extend(watsonx_tools)
print(f"Added {len(watsonx_tools)} data preprocessing tools")

# Load huggingface tools and extend it
all_tools.extend(create_huggingface_tools())

# Load and add the cost analysis tools
all_tools.extend(create_cost_benefit_analysis_tools())

# add RUL prediction tools (if they exists)
# replaced with huggingface, cost and watsonx specific tools
# try:
#     if "rul_tools" in globals():
#         all_tools.extend(rul_tools)
#         print(f"✅ Added {len(rul_tools)} RUL prediction tools.")
# except NameError:
#     print("⚠️ rul_tools not found. RUL prediction tools will not be available.")

# Update BraveSearch tool description to mention fallback
if brave_search_tool:
    brave_search_tool.description = """Search the web using Brave Search API.
    
    If this tool fails with HTTP 422 or any error, immediately try duckduckgo_search instead.
    
    Input: JSON format with "query" parameter: {{"query": "your search query"}}
    
    Example: {{"query": "OSHA safety protocols for aircraft engine maintenance"}}
    """
# brave_search_tool.description = """Search the web using Brave Search API.

#     If this tool fails with HTTP 422 or any error, immediately try duckduckgo_search instead.

#     Input: JSON format with "query" parameter: {{"query": "your search query"}}

#     Example: {{"query": "OSHA safety protocols for aircraft engine maintenance"}}
#     """
# add brave and duckduckgo search tools
if brave_search_tool and duckduckgo_search_tool:
    all_tools.append(brave_search_tool)
    all_tools.append(duckduckgo_search_tool)
elif brave_search_tool and not duckduckgo_search_tool:
    all_tools.append(brave_search_tool)
    print("appended brave search tool, duckduckgo search tool not available")

elif not brave_search_tool and duckduckgo_search_tool:
    all_tools.append(duckduckgo_search_tool)
    print("appended duckduckgo search tool, brave search tool not available")
else:
    print("⚠️ Either brave or duckduckgo search tool is not available")


# add error handling wrapper for search tools


# Add this to your notebook before creating tools
class SearchToolWrapper(BaseTool):
    """Wrapper that provides fallback between Brave Search and DuckDuckGo"""

    name: str = "smart_search"
    description: str = """Search the web for information. Automatically falls back to DuckDuckGo if Brave Search fails.
    
    Use this instead of brave_search or duckduckgo_search directly.
    Input: JSON with "query" parameter containing your search query.
    """
    args_schema: Type[BaseModel] = RetrieveModelsInput  # Reuse a simple schema

    def _run(self, query: str) -> str:
        """Try Brave Search first, fall back to DuckDuckGo"""
        try:
            if brave_search_tool:
                result = brave_search_tool.run({"query": query})
                return result
        except Exception as e:
            # Try DuckDuckGo if Brave fails
            try:
                if duckduckgo_search_tool:
                    result = duckduckgo_search_tool.run(query)
                    return f"Note: Used DuckDuckGo (Brave Search failed): {result}"
            except Exception as e2:
                return f"Both search tools failed. Brave error: {str(e)}, DuckDuckGo error: {str(e2)}"

        return "No search tools available"


# TODO : maybe brave search tool and duckduckgo search tool being included seperately is not needed
all_tools.append(SearchToolWrapper())
print(all_tools)
# detailed info on the tools available
print(f"\n📊 Total tools available: {len(all_tools)}")
print(f"Tool names: {[tool.name for tool in all_tools]}\n")
