# Reinforcement Learning for Market Making

## 📌 Problem Statement
In modern electronic markets, **market makers** are essential liquidity providers. They continuously quote bid/ask prices to facilitate trading, earning the spread while managing **inventory risk**, **adverse selection**, and **execution uncertainty**.  

The challenge:  
- **Microstructure complexity** — queue priority, partial fills, cancellations, and latency.  
- **Dynamic regimes** — volatility spikes, spread changes, liquidity droughts.  
- **Risk constraints** — inventory limits, capital usage, and tail‑risk control.  

Traditional closed‑form models (e.g., Avellaneda–Stoikov) assume stationary dynamics and parametric order‑flow models. In reality, order‑flow is **non‑stationary** and **path‑dependent**.  

This project develops a **TensorFlow‑based Reinforcement Learning (RL) framework** to learn adaptive quoting policies that maximize **risk‑adjusted PnL** under realistic market microstructure constraints.

---

## 🛠 Quant Research Workflow

1. **Market Data Acquisition & Calibration**
   - Pull historical L2 order book and macro factors via `pandas-datareader` and `gs-quant`.
   - Fit volatility regimes, order arrival intensities, and spread transition matrices using `statsmodels`.

2. **Market Microstructure Simulation**
   - Gym‑style Limit Order Book (LOB) environment with:
     - Queue‑position‑aware fills
     - Latency and slippage modeling
     - Maker/taker fees and rebates
   - Configurable for asset, tick size, and regime.

3. **Feature Engineering**
   - LOB depth tensors, order‑flow imbalance, realized volatility, factor exposures.
   - Normalized and batched as `tf.Tensor` for RL agents.

4. **Baseline Strategies**
   - **Fixed‑Spread with inventory skew**
   - **Avellaneda–Stoikov** optimal control

5. **RL Agent Development**
   - **Soft Actor‑Critic (SAC)** in TensorFlow/Keras.
   - `tensorflow-probability` for stochastic policy heads.
   - Reward = PnL − λ·InventoryPenalty − η·CrossPenalty − ψ·TurnoverPenalty.

6. **Evaluation & Stress Testing**
   - Metrics: Sharpe, Sortino, VaR, ES, drawdowns, fill rate, realized spread.
   - Stress scenarios: volatility spikes, liquidity droughts, latency shocks.

7. **Reporting**
   - `quantstats` tear sheets for performance.
   - `Riskfolio-lib` for portfolio‑style inventory risk analysis.

---

## 🗺 System Architecture (Quant Flow)

```mermaid
flowchart TD
    subgraph Data Layer
        A[Market Data Sources] --> B[Data Loaders]
        B --> C[Calibration Models]
    end

    subgraph Simulation Layer
        C --> D[LOB Environment]
        D --> E[Feature Engineering]
    end

    subgraph Agent Layer
        E --> F[RL Agent (SAC)]
        F -->|Quotes| D
    end

    subgraph Evaluation Layer
        D --> G[Evaluation & Stress Tests]
        G --> H[Reporting & Analytics]
    end

## 💻 Tech Stack
Component	Technology	Why	How
Core Language	Python 3.12	Industry standard for quant research	All modules
Deep Learning	TensorFlow 2.x, tensorflow-probability	Production‑ready DL + probabilistic modeling	RL agent networks, stochastic policies
Market Data	pandas-datareader, gs-quant	Historical prices, macro factors, risk exposures	Data ingestion, factor analysis
Statistical Modeling	statsmodels	Econometrics, time series	Volatility regime calibration
RL Environment	Gymnasium API	Standard RL interface	LOB simulator
Performance Analytics	quantstats	Risk/return analytics	Tear sheets
Portfolio Risk	Riskfolio-lib	Portfolio optimization, CVaR	Inventory risk analysis
Experiment Tracking	MLflow	Reproducibility	Configs, metrics, artifacts
Visualization	Matplotlib, Seaborn	Data & performance plots	Calibration, evaluation


## 📊 Results (TBA)
Metric	RL Agent (SAC)	Avellaneda–Stoikov	Fixed Spread
Annualized Sharpe	TBA	TBA	TBA
Sortino Ratio	TBA	TBA	TBA
Max Drawdown	TBA	TBA	TBA
99% Expected Shortfall	TBA	TBA	TBA
Fill Rate (%)	TBA	TBA	TBA
Realized Spread (bps)	TBA	TBA	TBA

## 🔌 ASCII Module Interaction Map
[ data/loaders.py ]
    └─ Output: Pandas DataFrame (raw L2/L3 order book, trades, macro factors)
        ↓
[ data/preprocess.py ]
    └─ Output: Cleaned Pandas DataFrame (aligned timestamps, filtered assets)
        ↓
[ calibration/vol_models.py ]
    └─ Input: Pandas DataFrame (midprice returns)
    └─ Output: Dict of calibrated parameters (vol regimes, GARCH params)
        ↓
[ calibration/microstructure.py ]
    └─ Input: Pandas DataFrame (order flow, spreads)
    └─ Output: Dict (arrival intensities, fill probabilities, spread transitions)
        ↓
[ envs/lob_env.py ]
    └─ Input: Calibration dicts + config YAML
    └─ Output: Python dict (LOB state: depth arrays, inventory, PnL)
        ↓
[ envs/fillers.py / envs/dynamics.py / envs/costs.py ]
    └─ Input: Python dict (LOB state)
    └─ Output: Updated Python dict (post-fill state, costs applied)
        ↓
[ features/lob_features.py ]
    └─ Input: Python dict (LOB state)
    └─ Output: NumPy array (depth snapshot tensor)
[ features/imbalance.py ]
    └─ Output: NumPy array (order-flow imbalance features)
[ features/volatility.py ]
    └─ Output: NumPy array (realized vol features)
[ features/factor_exposures.py ]
    └─ Output: NumPy array (factor loadings from gs_quant)
        ↓
[ Feature Aggregation Layer ]
    └─ Input: Multiple NumPy arrays
    └─ Output: tf.Tensor (batched state representation for RL agent)
        ↓
[ agents/tf_sac/policy.py ]
    └─ Input: tf.Tensor (state)
    └─ Output: tf.Tensor (action logits or continuous offsets)
[ agents/tf_sac/learner.py ]
    └─ Input: tf.Tensor (state, action, reward, next_state)
    └─ Output: Updated TF model weights
[ agents/tf_sac/replay.py ]
    └─ Input: Python tuple (state, action, reward, next_state, done)
    └─ Output: Sampled batches as tf.Tensor for training
        ↓
[ agents/baselines/*.py ]
    └─ Input: NumPy array or Pandas DataFrame (state features)
    └─ Output: Python dict (quote offsets, sizes)
        ↓
[ experiments/scripts/train.py ]
    └─ Orchestrates: env ↔ agent loop, logs to MLflow
[ experiments/scripts/eval.py / stress_test.py ]
    └─ Output: Pandas DataFrame (PnL paths, metrics)
        ↓
[ evaluation/performance.py ]
    └─ Input: Pandas DataFrame (PnL, trades)
    └─ Output: QuantStats HTML report, Riskfolio-lib portfolio metrics
[ evaluation/stress_tests.py ]
    └─ Output: Pandas DataFrame (stress scenario results)
[ evaluation/reporting.py ]
    └─ Output: PDF/HTML dashboards
        ↓
[ notebooks/*.ipynb ]
    └─ Input: Pandas DataFrame, QuantStats reports
    └─ Output: Visualizations, exploratory analysis
[ tracking/mlflow_utils.py ]
    └─ Input: Metrics, params, artifacts
    └─ Output: MLflow experiment logs


## 🚀 How to Run
# Clone repo
git clone https://github.com/yourusername/rl-mm.git
cd rl-mm

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Train
python experiments/scripts/train.py --config configs/base.yaml

# Evaluate
python experiments/scripts/eval.py --config configs/base.yaml


##📬 Contact
For questions or collaboration inquiries, please reach out via GitHub Issues or email.
If you want, I can also **add an ASCII “module interaction map”** showing exac