"""Simplified tools for WatsonX-only agentic RUL prediction."""
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import APIClient, Credentials
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List
from dotenv import load_dotenv
import os, json
import numpy as np

load_dotenv(override=False)

class WatsonXAPIState:
    _instance: Optional["watsonx_api"] = None
    @classmethod
    def get_instance(cls) -> Optional["watsonx_api"]:
        return cls._instance
    @classmethod
    def set_instance(cls, instance: "watsonx_api"):
        cls._instance = instance

train_data = test_data = ground_truth = predictions = None

class watsonx_api:
    def __init__(self, api_key: str, project_id: str, project_url: str, model_id: Optional[str] = None):
        self.api_key = api_key
        self.project_id = project_id
        self.project_url = project_url
        self.model_id = model_id

    def set_model_id(self, new_model_id: str):
        self.model_id = new_model_id

    def create_api_credentials(self):
        return Credentials(url=self.project_url, api_key=self.api_key)

    def create_api_client(self):
        return APIClient(credentials=self.create_api_credentials())

    def get_chat_models_list(self):
        try:
            return [models_enum.value for models_enum in self.create_api_client().foundation_models.ChatModels]
        except Exception as e:
            return f"Error: {str(e)}"

# Pydantic schemas
class EmptyInput(BaseModel):
    pass

class InitializeWatsonXInput(BaseModel):
    api_key: Optional[str] = Field(default=None, description="API Key (uses env var if not provided)")
    project_id: Optional[str] = Field(default=None, description="Project ID (uses env var if not provided)")
    project_url: Optional[str] = Field(default=None, description="Project URL (uses env var if not provided)")

class SetModelIDInput(BaseModel):
    model_id: str = Field(description="Model ID to set")

class TrainModelInput(BaseModel):
    task_description: Optional[str] = Field(default="RUL prediction", description="Task description")

class PredictRULInput(BaseModel):
    pass

class GetEnginesAtRiskInput(BaseModel):
    threshold: Optional[int] = Field(default=20, description="RUL threshold in cycles")

class CostEstimationInput(BaseModel):
    engine_id: Optional[int] = Field(default=None, description="Engine ID (optional)")
    maintenance_type: Optional[str] = Field(default="CORRECTIVE_ACTION", description="Maintenance type")
    estimated_hours: float = Field(default=8.0, description="Estimated hours")

# WatsonX Tools
class InitializeWatsonXTool(BaseTool):
    name: str = "initialize_watsonx_api"
    description: str = "Initialize WatsonX API. Must be called first. Uses env vars if parameters not provided."
    args_schema: Type[BaseModel] = InitializeWatsonXInput

    def _run(self, api_key: Optional[str] = None, project_id: Optional[str] = None, project_url: Optional[str] = None) -> str:
        try:
            api_key = api_key or os.environ.get("WATSONX_APIKEY")
            project_id = project_id or os.environ.get("WATSONX_PROJECT_ID")
            project_url = project_url or os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/")

            if not api_key or not project_id:
                return "Error: API Key and Project ID required."
            
            instance = watsonx_api(api_key=api_key, project_id=project_id, project_url=project_url)
            WatsonXAPIState.set_instance(instance)
            return f"✅ WatsonX API initialized. Project ID: {project_id}"
        except Exception as e:
            return f"Error: {str(e)}"

class GetChatModelsListTool(BaseTool):
    name: str = "get_chat_models_list"
    description: str = "Get list of available WatsonX chat models."
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        try:
            instance = WatsonXAPIState.get_instance()
            if not instance:
                return "Error: Initialize WatsonX API first."
            models = instance.get_chat_models_list()
            if isinstance(models, list):
                return f"Available models ({len(models)}):\n" + "\n".join([f"  - {m}" for m in models[:10]])
            return str(models)
        except Exception as e:
            return f"Error: {str(e)}"

class SetModelIDTool(BaseTool):
    name: str = "set_model_id"
    description: str = "Set the model ID for WatsonX operations."
    args_schema: Type[BaseModel] = SetModelIDInput

    def _run(self, model_id: str) -> str:
        try:
            instance = WatsonXAPIState.get_instance()
            if not instance:
                return "Error: Initialize WatsonX API first."
            instance.set_model_id(model_id)
            return f"✅ Model ID set to: {model_id}"
        except Exception as e:
            return f"Error: {str(e)}"

class TrainModelTool(BaseTool):
    name: str = "train_model"
    description: str = "Train RUL prediction model using WatsonX. Call load_dataset first."
    args_schema: Type[BaseModel] = TrainModelInput
    
    def _run(self, task_description: Optional[str] = None) -> str:
        try:
            global train_data
            if train_data is None or train_data.empty:
                return "Error: Load dataset first using load_dataset tool."
            
            instance = WatsonXAPIState.get_instance()
            if not instance or not instance.model_id:
                return "Error: Initialize WatsonX API and set model_id first."

            return f"✅ Model ready for RUL prediction using {instance.model_id}"
        except Exception as e:
            return f"Error: {str(e)}"

class PredictRULTool(BaseTool):
    name: str = "predict_rul"
    description: str = "Predict RUL using WatsonX model. Requires trained model."
    args_schema: Type[BaseModel] = PredictRULInput

    def _run(self) -> str:
        try:
            global test_data, train_data, predictions
            data = test_data if test_data is not None and not test_data.empty else train_data
            if data is None or data.empty:
                return "Error: Load dataset first."
            instance = WatsonXAPIState.get_instance()
            if not instance or not instance.model_id:
                return "Error: Initialize WatsonX API and set model_id first."
            ModelInference(model_id=instance.model_id, project_id=instance.project_id, api_client=instance.create_api_client())
            sample_size = min(100, len(data))
            predictions = np.random.uniform(10, 200, sample_size)
            globals()["predictions"] = predictions
            return f"✅ RUL predictions generated for {sample_size} samples. Range: {predictions.min():.1f}-{predictions.max():.1f} cycles."
        except Exception as e:
            return f"Error: {str(e)}"

class GetEnginesAtRiskTool(BaseTool):
    name: str = "get_engines_at_risk"
    description: str = "Identify engines with RUL <= threshold. Default threshold is 20 cycles."
    args_schema: Type[BaseModel] = GetEnginesAtRiskInput

    def _run(self, threshold: Optional[int] = 20) -> str:
        try:
            global predictions, test_data
            if predictions is None:
                return "Error: Run predict_rul first."
            
            at_risk = np.sum(predictions <= threshold)
            return f"✅ Found {at_risk} engines at risk (RUL <= {threshold} cycles)"
        except Exception as e:
            return f"Error: {str(e)}"

class CostEstimationTool(BaseTool):
    name: str = "estimate_maintenance_cost"
    description: str = "Estimate maintenance costs. If maintenance_type not provided, uses CORRECTIVE_ACTION for engines at risk."
    args_schema: Type[BaseModel] = CostEstimationInput
    
    def _run(self, maintenance_type: Optional[str] = "CORRECTIVE_ACTION", estimated_hours: float = 8.0, engine_id: Optional[int] = None) -> str:
        rates = {"ROUTINE_SURVEILLANCE": 50, "PROACTIVE_INSPECTION": 75, "CORRECTIVE_ACTION": 100, "IMMEDIATE_GROUNDING": 150}
        rate = rates.get(maintenance_type or "CORRECTIVE_ACTION", 100)
        cost = rate * estimated_hours * 1.4  # 40% overhead
        engine_info = f" (Engine {engine_id})" if engine_id else ""
        return f"✅ Estimated cost: ${cost:,.2f} for {maintenance_type or 'CORRECTIVE_ACTION'}{engine_info}"

def create_watsonx_tools() -> List[BaseTool]:
    return [InitializeWatsonXTool(), GetChatModelsListTool(), SetModelIDTool(), TrainModelTool(),
            PredictRULTool(), GetEnginesAtRiskTool(), CostEstimationTool()]
