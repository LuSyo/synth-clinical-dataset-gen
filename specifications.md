# Clinical ground truth



# Raw features



# Bias

## Measurement error (additive bias)

**Applicable feature types:** Continuous

**What it simulates:** A medical device, diagnostic test, or clinical assessment tool that is systematically less accurate or miscalibrated when applied to a marginalized group

**Clinical example:** pulse oximeters behaving differently on various skin tones

**Formulation:**

For an individual $i$, the observed feature $X_{obs,i}$ is affected if they belong to the group $S_i = a$

$$X_{obs, i} = X_{true, i} + \mathbb{I}(S_i = a) \cdot \left( \mu_{bias} + \epsilon_i \right)$$

where:

- $\mathbb{I}(.)$ is the indicator function
- $\mu_{bias}$ is a systematic mean shift
- $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ is a stochastic noise injection


## Access barrier bias (acuity-based attenuation)

**Applicable feature type:** Continuous, integer

**What it simulates:** Systemic healthcare access barriers where marginalised patients ($S_i = a$) only present to the clinic or have their measurements recorded when their condition becomes severe. Mild or sub-acute presentations go systematically under-recorded or attenuated.

**Clinical example:** Rheumatoid arthritis symptom tracking or lab markers being artificially suppressed or omitted due to a lack of regular primary care access, meaning they are only fully captured during severe acute flare-ups.

**Formulation**:Let $\tau$ be an empirical threshold marking the transition between mild/stable and severe clinical presentations (e.g., a specific lower quantile of the baseline distribution). If an individual belongs to the group $S_i = a$ and their true value $X_{true, i}$ falls below $\tau$, their observed value undergoes severe suppression:

$$X_{obs, i} = \begin{cases} 
X_{true, i} \cdot \alpha + \epsilon_i & \text{if } S_i = a \text{ and } X_{true, i} < \tau \\ 
X_{true, i} & \text{otherwise} 
\end{cases}$$

where:

- $\tau$ is the clinical acuity threshold boundary
- $\alpha \in (0, 1)$ is the multiplicative attenuation/suppression factor
- $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ is a stochastic noise injection

## Referral bias

**Applicable feature types:** Binary

**What it simulates:** Systemic or implicit practitioner bias in binary decision-making paths (e.g. choosing whether to refer a patient to a specialised cardiac unit, order an advanced scan, or document a highly subjective behavior).

**Clinical example:** Systematically under-referring minority patients for advanced diagnostic angiograms compared to majority patients with identical clinical symptoms.

**Formulation:** Let $X_{true, i} \in \{0, 1\}$ be the fairly generated binary feature. If an individual qualifies for the feature ($X_{true, i} = 1$) but belongs to the marginalized group ($S_i = a$), their probability of actually receiving the positive status is systematically suppressed by a probability factor $p_{suppress}$:

$$P(X_{obs, i} = 1 \mid X_{true, i}, S_i) = X_{true, i} \cdot \left(1 - \mathbb{I}(S_i = a) \cdot p_{suppress}\right)$$

$$X_{obs, i} \sim \text{Bernoulli}\left(P(X_{obs, i} = 1 \mid X_{true, i}, S_i)\right)$$

where:

- $p_{suppress} \in [0, 1]$ is the probability that a qualified minoritised patient gets their referral suppressed.
- If $X_{true, i} = 0$, the observed value remains $0$ (the bias strictly acts as a barrier to entry, not an accidental promotion)
- Stochasticity is natively driven by the Bernoulli trial on the resulting probability vector


## Under-classification bias (severity minimisation)

**Applicable feature types: **Categorical (Ordinal)

**What it simulates:** A systemic failure where clinical severity scales (e.g., triage notes, diagnostic staging, tool assessments) are recorded accurately for the majority group, but marginalized patients have their clinical severity systematically downgraded or under-coded due to systemic bias or stereotyping.

**Clinical example:** In mental health or pain management, clinical guidelines might objectively map a patient's symptoms to a "Severe" or "Moderate" classification ($X_{true} = 2$ or $1$). However, due to systemic minimisation of pain or symptoms in minority patients ($S = a$), their documented chart entry is downgraded to a lower tier, resulting in delayed specialty intervention.

**Formulation:**
Let $X_{true, i} \in \{0, 1, \dots, K-1\}$ be the fairly generated ordinal category index (where higher values represent higher clinical severity). For individuals in the marginalised group ($S_i = a$) who possess a true severity greater than baseline ($X_{true, i} > 0$), there is a stochastic probability $p_{down}$ that their recorded stage is reduced:

$$X_{obs, i} = \begin{cases} 
X_{true, i} - 1 & \text{with probability } p_{down} \cdot \epsilon_i \quad \text{if } S_i = a \text{ and } X_{true, i} > 0 \\ 
X_{true, i} & \text{otherwise} 
\end{cases}$$

where:

- $p_{down} \in [0, 1]$ is the baseline probability that a marginalized patient's severity level is overlooked or minimized.
- $\epsilon_i \sim \text{Bernoulli}(0.95)$ acts as an additional stochastic throttle, ensuring that even within a biased systemic framework, there is a small layer of non-deterministic, individual-level variance.
- If $X_{true, i} = 0$, the observed value remains $0$ (the bias cannot reduce severity below the absolute baseline).