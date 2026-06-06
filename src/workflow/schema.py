from pydantic import BaseModel, Field, ConfigDict
from typing import List, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd

class GraphState(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  seed: int = Field(default=4)
  messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
  
  # Dataset Parameters (Target values requested by user)
  n_samples: int = Field(default=20000, description="Number of individual records to generate.")
  s_prevalence: float = Field(default=0.5, description="Prevalence of the minority group (S=0).")
  y_prevalence: float = Field(default=0.5, description="Prevalence of the positive outcome (Y=1).")
  feature_map: dict = Field(default_factory=dict, description="JSON map specifying names, types, and causal pathways.")
  
  # Dataset
  df: Optional[pd.DataFrame] = Field(default=None, description="In-memory tabular dataset object.")

  # Dataset downstream impact
  probe_results: Optional[str] = Field(default=None, description="Predictive performance metrics of a downstream classifier trained on the generated dataset.")

  # Pipeline Artifact Tracking
  dataset_path: Optional[str] = Field(default=None, description="Local path to the currently generated CSV dataset.")
  validation_passed: bool = Field(default=False, description="Whether the generated dataset successfully met all acceptance criteria.")

  validation_status: Literal["valid", "retry_extraction", "generation_failure", "pending"] = Field(
    default="pending", 
    description="The resulting status from the dataset evaluation node."
  )
  retry_count: int = Field(
    default=0, 
    description="Tracks how many times the extraction/generation loop has been attempted."
  )
  max_retries: int = Field(
    default=3, 
    description="The upper boundary ceiling to prevent infinite looping."
  )

  def __repr__(self):
    return (f"GraphState(messages_count={len(self.messages)}, "
            f"targets=[N={self.n_samples}, S_prev={self.s_prevalence}, Y_prev={self.y_prevalence}], "
            f"valid={self.validation_passed})")

class ExtractedDatasetParams(BaseModel):
  n_samples: Optional[int] = Field(
    default=None, 
    description="The number of individuals/samples requested. E.g., 5000."
  )
  s_prevalence: Optional[float] = Field(
    default=None, 
    description="The prevalence of the majority group (S=1), expressed as a float between 0 and 1. E.g., 0.60 for 60%."
  )
  y_prevalence: Optional[float] = Field(
    default=None, 
    description="The prevalence of the clinical outcome (Y=1), expressed as a float between 0 and 1. E.g., 0.15 for 15%."
  )

class DatasetValidationResult(BaseModel):
  is_acceptable: bool = Field(
    description="True if the actual metrics are reasonably close to the targets (allowing for small stochastic sampling variance), False if they diverge significantly."
  )
  extracted_params_match_query: bool = Field(
    description="True if the currently extracted parameters correctly reflect the explicit request in the original user query. False if the extraction node made a mistake (e.g., swapped minority/majority rates)."
  )
  corrected_n_samples: Optional[int] = Field(
    default=None, description="The corrected number of samples parsed from the query if a mistake was made."
  )
  corrected_s_prevalence: Optional[float] = Field(
    default=None, description="The corrected majority prevalence (S=1) parsed from the query if a mistake was made."
  )
  corrected_y_prevalence: Optional[float] = Field(
    default=None, description="The corrected outcome prevalence (Y=1) parsed from the query if a mistake was made."
  )
  reasoning: str = Field(
    description="A concise explanation detailing your evaluation of the dataset vs targets vs user query."
  )