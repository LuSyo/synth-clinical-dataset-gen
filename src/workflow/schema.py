from pydantic import BaseModel, Field, ConfigDict
from typing import List, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd
from utils import Config
from enums import TargetDisp

class GraphState(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  seed: int = Field(default=4)
  # messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
  
  # Dataset Parameters (Target values requested by user)
  n_pop: int = Field(default=Config.N_POP, description="Number of individual records to generate.")
  s_prevalence: float = Field(default=Config.S_PREV, description="Prevalence of the minority group (S=0).")
  y_prevalence: float = Field(default=Config.Y_PREV, description="Prevalence of the positive outcome (Y=1).")
  diff_y_prev_factor: float = Field(default=1, description="Differential prevalence of the positive outcome (Y=1) between groups in the raw dataset.")
  feature_map: dict = Field(default_factory=dict, description="JSON map specifying names, types, and causal pathways.")

  target_raw_auprc: float = Field(
    default=Config.TARGET_AUPRC, 
    description="Minimum acceptable global AUPRC baseline for the raw dataset probe."
  )
  target_disp: TargetDisp = Field(
    default=TargetDisp.none,
    description="Active target disparities (none, both, recall, ppv)"
  )
  target_biased_recall_disp: float = Field(
    default=Config.TARGET_RECALL_DISP, 
    description="Target Recall disparity, calculated as: Recall(S=1) - Recall(S=0). Default is 0.0 (parity)."
  )
  target_biased_ppv_disp: float = Field(
    default=Config.TARGET_PPV_DISP, 
    description="Target Precision/PPV disparity, calculated as: Precision(S=1) - Precision(S=0). Default is 0.0 (parity)."
  )
  disparity_tolerance: float = Field(
    default=Config.DISP_TOLERANCE,
    description="The acceptable error margin (+/-) allowed around the target disparity values."
  )
  
  # Dataset
  df: Optional[pd.DataFrame] = Field(default=None, description="In-memory tabular dataset object.")
  train_df: Optional[pd.DataFrame] = None
  test_df: Optional[pd.DataFrame] = None

  # Dataset downstream impact
  probe_results: Optional[str] = Field(default=None, description="Predictive performance metrics of a downstream classifier trained on the generated dataset.")

  # Pipeline Artifact Tracking
  dataset_path: Optional[str] = Field(default=None, description="Local path to the currently generated CSV dataset.")
  
  phase: str = Field(default="generation", description="The current phase of the flow (generation, bias, complete)")
  validation_passed: bool = Field(default=False, description="Whether the generated dataset successfully met all acceptance criteria.")
  retry_count: int = Field(
    default=0, 
    description="Tracks how many times the generation loop has been attempted."
  )
  max_retries: int = Field(
    default=3, 
    description="The upper boundary ceiling to prevent infinite looping."
  )

  def __repr__(self):
    return (
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

class MeasurementErrorParams(BaseModel):
  model_config = ConfigDict(extra="forbid")
  mu_bias: float = Field(description="Systematic calibration mean shift delta.")
  noise_std: float = Field(description="Random noise standard deviation.")

class AccessBarrierParams(BaseModel):
  model_config = ConfigDict(extra="forbid")
  alpha: float = Field(description="Multiplicative attenuation factor bounded strictly within (0, 1).")
  noise_std: float = Field(description="Random noise standard deviation.")

class ReferralBiasParams(BaseModel):
  model_config = ConfigDict(extra="forbid")
  p_suppress: float = Field(description="Stochastic probability of a true 1 being overridden to an observed 0, bounded within (0, 1).")

class UnderClassificationParams(BaseModel):
  model_config = ConfigDict(extra="forbid")
  p_down: float = Field(description="Stochastic probability of dropping down exactly 1 severity tier, bounded within (0, 1).")

class ValidationResult(BaseModel):
  is_acceptable: bool = Field(
    description="True if the actual metrics are reasonably close to the targets (allowing for small stochastic sampling variance), False if they diverge significantly."
  )
  reasoning: str = Field(
    description="A concise explanation detailing your evaluation of the dataset vs user expectations and your suggested adjustments to parameters."
  )

class DatasetValidationResult(ValidationResult):
  adjusted_parameters: Optional[List[FeatureParameterOverride]] = Field(
    default=None,
    description="If is_acceptable is False, provide a list of specific feature parameter additions/modifications. Leave as None if is_acceptable is True."
  )

