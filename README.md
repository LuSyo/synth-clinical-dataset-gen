# SCM healthcare bias calibration pipeline

An automated, configuration-driven data simulation and fairness auditing framework built with LangGraph and LangChain. This pipeline generates synthetic, structurally sound Electronic Health Record (EHR) datasets driven by explicit causal topologies, tracks mathematical parameter lineage, and executes downstream predictive validation probes to expose representation and performance disparities among protected subgroups.

---

## Causal structural framework

The synthetic population generation is governed by a defined Structural Causal Model (SCM). The pipeline generates hidden patient health variables from which observed clinical features descend:

* Central health confounder (H): Represents an individual's core health index, sampled from a hardcoded independent Gamma distribution.
* Sensitive attribute (S): Binary group indicator representing biological (Sbio) or sociological (Ssoc) classifications.
* S-dependent latent (U_dep): Descends from H and is directly influenced by S, representing biological correlations.
* S-independent latent (U_indep): Descends from H but remains structurally independent of S.
* Clinical outcome (Y): Target binary label descending from both latents.

| Pathway key | Target feature block | Causal parent anchor | Core analytical domain |
| :--- | :--- | :--- | :--- |
| `bio` | X_bio (S-correlated) | U_dep | Baseline clinical covariates, genetic markers, physiological measurements |
| `soc` | X_soc (S-biased) | U_indep | Features susceptible to systemic bias, clinician interpretation, self-reported symptoms |
| `ind` | X_ind (S-independent) | U_indep | Variables independent of the sensitive attribute |

---

## Key pipeline capabilities

* **Configuration-driven feature synthesiser:** generates tabular EHR features iteratively from a mutable JSON feature schema
* **Multi-datatype primitive support:** Simulates continuous, binary and categorical variables
* **Configurable bias:** Simulates different types of bias (healthcare access, measurement, etc...) applied onto the features in the sociologically biased pathway
* **Distribution audit:** Captures the exact coefficients selected to generate the distribution and added bias for each feature. 
* **Automated visual and statistical analysis:** Generates global and stratified distribution plots and table one. 
* **Out-of-bag bootstrap predictive probe:** Runs a 4-iteration out-of-bag (OOB) bootstrap classifier probe immediately following generation to measure global vs stratified predictive potential of the dataset. 

---

## Graph 

The pipeline processes tasks through a deterministic, sequential state machine graph controlled via an orchestration layout:

```
      ┌──────────────────────────────────────────────┐
  [ START ]                                          │
      │                                              │
      ▼                                              │
[Generate clinical ground truth and features]        │
      │                                              │
      ▼                                              │
[Generate plots and data summary]                    │
      │                                              │
      ▼                                              │
[Run downstream classifier probe]                    │
      │                                              │
      ▼                                              │
[LLM: Evaluate results against target]               │
      │                                              │
      ├───(Validation Failed)───► [LLM: Adapt parameters]
      │
      └───(Validation Passed)───► [Save dataset] ───► [END]
```

## Configuration and schema usage

The dataset blueprint is declared as a nested dictionary, e.g.:

```json
{
  "soc": [
    {"name": "neighborhood_risk", "type": "continuous", "dist": "normal"},
    {"name": "insurance_tier", "type": "categorical", "n": 3},
    {"name": "unemployment_flag", "type": "binary"}
  ],
  "bio": [
    {"name": "systolic_bp_delta", "type": "continuous", "dist": "normal"},
    {"name": "biomarker_positive", "type": "binary"}
  ],
  "ind": [
    {"name": "genetic_score_raw", "type": "continuous", "dist": "lognormal"}
  ]
}
```