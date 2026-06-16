# Clinical data generator for fairness testing

This tool automatically generates synthetic medical datasets to test how machine learning models handle demographic bias and develop effective mitigation methods. Instead of requiring manual data tweaking, it uses an LLM feedback loop built with LangGraph to dynamically adjust the properties, distributions, and shapes of the data to elad to the required population characteristics and predictive disparities. 

The pipeline simulates underlying patient health states alongside configurable clinical features and systemic bias to intentionally induce performance gaps between subgroups. Every run automatically tracks the exact mathematical parameters used to generate the dataset and tests the final dataset with a classifier probe to report predictive disparities.

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
[ Start ]
    │
    ▼
┌──────────────────────────────┐
│ 1. Generate ground truth     │ ◄───────────────────────────┐
└──────────────┬───────────────┘                             │
               │                                             │ (Loop Phase 1:
               ▼                                             │  Tune Raw Gen)
┌─────────────────────────────────────────┐                  │
│ 2. Classifier probe on raw features     │                  │
└──────────────┬──────────────────────────┘                  │
               │                                             │
               ▼                                             │
      /─────────────────\                                    │
     < Valid Raw Targets? > ───[ No: Update Gen Params ] ────┘
      \─────────────────/
               │
               │ Yes (Lock Baseline)
               ▼
┌──────────────────────────────┐ 
│ 3. Apply bias on X_soc       │ ◄───────────────────────────┐
└──────────────┬───────────────┘                             │
               │                                             │ (Loop Phase 2:
               ▼                                             │  Tune Bias Only)
┌─────────────────────────────────────────┐                  │
│ 4. Classifier probe on biased features  │                  │
└──────────────┬──────────────────────────┘                  │
               │                                             │
               ▼                                             │
      /──────────────────\                                   │
     < Valid Bias Targets? > ───[ No: Update Bias Params ] ──┘
      \──────────────────/
               │
               │ Yes (Lock Final Artifacts)
               ▼
┌──────────────────────────────┐
│ 5. Compile Reports           │  (Table One & Histograms)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 6. S-stratified sampling     │  (Train / Test Splits)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 7. Save datasets             │
└──────────────────────────────┘
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