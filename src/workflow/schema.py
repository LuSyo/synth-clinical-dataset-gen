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
  n_pop: int = Field(default=20000, description="Number of individual records to generate.")
  s_prevalence: float = Field(default=0.5, description="Prevalence of the minority group (S=0).")
  y_prevalence: float = Field(default=0.5, description="Prevalence of the positive outcome (Y=1).")
  feature_map: dict = Field(default_factory=dict, description="JSON map specifying names, types, and causal pathways.")
  
  # Dataset
  df: Optional[pd.DataFrame] = Field(default=None, description="In-memory tabular dataset object.")
  train_df: Optional[pd.DataFrame] = None
  test_df: Optional[pd.DataFrame] = None

  # Dataset downstream impact
  probe_results: Optional[str] = Field(default=None, description="Predictive performance metrics of a downstream classifier trained on the generated dataset.")

  # Pipeline Artifact Tracking
  dataset_path: Optional[str] = Field(default=None, description="Local path to the currently generated CSV dataset.")
  
  validation_passed: bool = Field(default=False, description="Whether the generated dataset successfully met all acceptance criteria.")
  retry_count: int = Field(
    default=0, 
    description="Tracks how many times the generation loop has been attempted."
  )
  max_retries: int = Field(
    default=3, 
    description="The upper boundary ceiling to prevent infinite looping."
  )
  # validation_status: Literal["valid", "retry_extraction", "generation_failure", "pending"] = Field(
  #   default="pending", 
  #   description="The resulting status from the dataset evaluation node."
  # )
  

  def __repr__(self):
    return (f"GraphState(messages_count={len(self.messages)}, "
            f"targets=[N={self.n_pop}, S_prev={self.s_prevalence}, Y_prev={self.y_prevalence}], "
            f"valid={self.validation_passed})")

class FeatureParameterOverride(BaseModel):
  pathway: Literal["bio", "soc", "ind"] = Field(
    description="The pathway category where the feature resides ('bio', 'soc', or 'ind')."
  )
  name: str = Field(
    description="The exact name of the feature to adjust (e.g., 'b1')."
  )
  gamma: Optional[float] = Field(
    default=None, 
    description="The fine-tuned directional weight coefficient. Adjust to change feature-latent correlation strength."
  )
  beta: Optional[float] = Field(
    default=None, 
    description="The fine-tuned intercept shift coefficient. Adjust to shift binary prevalence baseline rates."
  )
  noise_std: Optional[float] = Field(
    default=None, 
    description="The fine-tuned standard deviation of feature noise. Increase to dilute feature predictive power."
  )
  absolute_thresholds: Optional[List[float]] = Field(
    default=None, 
    description="The fine-tuned list of cutoff marks for categorical variables. Must match length of the classes minus 1."
  )

class DatasetValidationResult(BaseModel):
  is_acceptable: bool = Field(
    description="True if the actual metrics are reasonably close to the targets (allowing for small stochastic sampling variance), False if they diverge significantly."
  )
  reasoning: str = Field(
    description="A concise explanation detailing your evaluation of the dataset vs user expectations and your suggested adjustments to parameters."
  )
  adjusted_parameters: Optional[List[FeatureParameterOverride]] = Field(
    default=None,
    description="If is_acceptable is False, provide a list of specific parameter additions/modifications. Leave as None if is_acceptable is True."
  )

