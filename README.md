# Introduction
This project uses a Bayesian Hidden Markov Model (HMM) to analyze historical avocado price data. In this milestone, the focus is on data preprocessing and model training using a Gibbs sampling approach for parameter estimation. The agent (implemented in HMM_train.py) estimates the model’s parameters and performs one-step-ahead predictions, thereby laying the foundation for risk assessment in avocado pricing.

# PEAS/Agent Analysis
- Performance:
The agent is evaluated based on its ability to:
	- Accurately estimate HMM parameters (initial state distribution, transition matrix, and Gaussian emission parameters).
	- Generate reliable one-step-ahead predictions for avocado prices.
- Environment:
The agent operates on the cleaned and preprocessed avocado dataset stored in processed_avocado.csv. This dataset includes time-sorted historical records of avocado prices.
- Actuators:
The outputs produced by the agent are:
	- Final estimates for the initial state probabilities (π), transition matrix (A), emission means, and variances.
	- One-step-ahead predictions for future avocado prices, both as a sampled prediction and as a predictive mean.
- Sensors:
The agent reads from the preprocessed dataset, particularly using the AveragePrice column as the continuous observation for the HMM. Other previously processed features (if any) ensure the data is cleaned and temporally consistent.


# HMM utilizing the avocado prices dataset
- Goal is to predict future avocado prices by training an HMM using the given dataset
## Raw Dataset

#### ([Dataset Link](https://www.kaggle.com/datasets/neuromusic/avocado-prices))
#### Historical data on avocado prices and sales volume in multiple US Markets

### Dataset Columns

- Date - The date of the observation.
- AveragePrice - the average price of a single avocado.
- type - conventional or organic.
- year - the year
- Region - the city or region of the observation
- Total Volume - Total number of avocados sold
- 4046 - Total number of avocados with PLU 4046 sold
- 4225 - Total number of avocados with PLU 4225 sold
- 4770 - Total number of avocados with PLU 4770 sold

### Dataset Exploration
- Dataset Overview:
The dataset is a cleaned version of the avocado prices data. The primary variable is:
	- AveragePrice: The main observation used for training the HMM.
	- Visualization Placeholder: An exploratory data analysis plot (e.g., a time series plot of AveragePrice) should be included here to illustrate the data trends.


## Cleaned Dataset for training:

- Unnamed: 0 - index column added when the CSV was saved
- AveragePrice - The average price of avocados on a given day.
- Total Volume - The total volume of avocados sold on that day.
- 4046 - Volume of avocados sold with PLU 4046.
- 4225 - Volume of avocados sold with PLU 4225.
- 4770 - Volume of avocados sold with PLU 4770.
- Total Bags - Total number of bags sold, representing packaged quantities.
- Small Bags - Number of small-sized bags sold.
- Large Bags - Number of large-sized bags sold.
- XLarge Bags - Number of extra-large bags sold.
- type - Indicates the type of avocado sold (e.g., conventional or organic).
- year - The year when the sales were recorded.
- region - Specifies the geographical region or market where the sales took place.
- price_change - Represents the change in average price from a previous period.
- volume_change - Indicates the change in total volume sold compared to a previous period.
- price_moving_avg - The moving average of the avocado price over a set period, used to smooth out fluctuations.
- volume_moving_avg - The moving average of the sales volume, highlighting longer-term trends.
- price_state_encoded - An encoded version of the price data, used for modeling and to capture underlying price patterns.
- volume_state_encoded - An encoded version of the volume data, used for modeling and analysis.

## Variables and their interactions:
- AveragePrice – The key observed variable, representing the average selling price of avocados. It is influenced by supply, demand, and overall market conditions.
- Total Volume, 4046, 4225, 4770 – These variables represent the total sales volume and sales volumes for specific PLU codes. They capture the supply side and consumer purchasing behavior, which often correlate with price changes. For example, high volume might indicate abundant supply or a surge in demand.
- Total Bags, Small Bags, Large Bags, XLarge Bags – These packaging metrics add nuance by showing how products are sold in different formats. They can provide insights into market segmentation and trends in consumer preferences that indirectly affect pricing.
- type, year, region are Categorical variables that segment the data. Different regions or product types (e.g., conventional vs. organic) can experience distinct market dynamics, leading to variations in pricing and volume.
- price_change and volume_change – Derived variables that capture short-term dynamics. They indicate momentum or abrupt shifts that might signal transitions between market regimes.
- price_moving_avg and volume_moving_avg – These smooth out short-term fluctuations and highlight underlying trends. They serve as benchmarks to compare against instantaneous values, which helps in understanding whether current movements are anomalies or part of a longer trend.
- price_state_encoded and volume_state_encoded – These represent discretized or transformed versions of the continuous price and volume data. They can help in segmenting the observations into different states, facilitating the mapping to underlying hidden regimes.
- Feature Relationships:
The HMM is structured to model the temporal dynamics of avocado prices. Key assumptions include:
	•	Continuous Observations: Each observed price is modeled as being generated from a Gaussian distribution whose parameters depend on the underlying hidden state.
	•	Markov Property: The current hidden state is assumed to depend solely on the previous state, enabling sequential modeling.

## Parameter Calculation
- Parameter Estimation via Bayesian Inference and Gibbs Sampling: The agent uses a Gibbs sampling algorithm to iteratively update model parameters:
	- Initial State Distribution (π):
	- Prior: Dirichlet distribution (uniform prior with all ones).
	- Update: Based on the state at the first time step.
- Transition Matrix (A):
	- Prior: Each row is drawn from a Dirichlet distribution.
	- Update: Count transitions between states and sample new rows from the Dirichlet posterior.
- Emission Parameters (Means and Variances):
	- Gaussian Likelihood: The probability of an observation given a state is computed as: P(x | μ, σ²) = (1 / sqrt(2πσ²)) * exp(- (x - μ)² / (2σ²))
- Posterior Updates:
	- Means are updated using a Normal distribution.
	- Variances are updated via an Inverse-Gamma sampling approach (implemented by Gamma sampling inversion).

- Gibbs Sampling Loop:
The algorithm iterates over the following steps:
### 1. Hidden State Sampling:
	- Forward Filtering: Computes the likelihood of each state for every observation using the Gaussian PDF.
	- Backward Sampling: Generates a sample sequence of hidden states.
### 2. Parameter Updates:
	- Update π, A, and the emission parameters based on the sampled hidden states.
### 3. Iteration and Convergence:
 - The process is repeated (e.g., for 1000 iterations), with intermediate results stored for analysis.
 
 ### Gibbs Sampling Explanation
 
 Gibbs sampling is a Markov chain Monte Carlo (MCMC) method used to approximate the joint posterior distribution of complex models, such as our Bayesian HMM. The process involves iteratively sampling each variable from its conditional distribution given the current values of all other variables. In our HMM agent, this means:
 
 - **Hidden State Sampling:** Given the current model parameters, the hidden state sequence is sampled using forward filtering and backward sampling.
 - **Parameter Updates:** With the newly sampled hidden states, the initial state distribution (π), transition matrix (A), and Gaussian emission parameters (means and variances) are updated by sampling from their respective conditional posterior distributions.
 
 **The goal of Gibbs sampling** is to generate samples that collectively approximate the joint posterior distribution. This enables us to infer the model parameters and quantify uncertainty in our predictions, which is especially valuable when direct computation of the joint posterior is infeasible due to its complexity.

Library Usage
- NumPy: For numerical computations (array operations, random sampling, etc.).
NumPy Documentation
- pandas: For data loading and manipulation (reading the CSV file, processing data, etc.).
pandas Documentation
- Custom Functions: The agent uses custom helper functions for:
	- Gaussian PDF Calculation
	- Forward Filtering and Backward Sampling
These functions implement the necessary computations for the Bayesian updates within the Gibbs sampling loop.


## Why HMM
- The data in the avocados set is sequential, and interdependent with its market data. This allows us to get accurate estimations utilizing inferring shifts in underlying market conditions.

## HMM structure
- Markov Chain:
  - The hidden state sequence follows a Markov process, meaning that the state at time t only depends on the state at time t-1

- Transition Matrix with Priors:
  - In this Bayesian framework, we place a prior on the transition probabilities. This allows us to incorporate prior beliefs and to quantify uncertainty in how states change over time.

- Emission Distributions:
  - For each hidden state, the observed variables are generated from a likelihood function parameterized by state-specific parameters. This involves:
  - A joint distribution that models the multivariate observation vector.
  - A factorized model where each variable is conditionally independent given the state.
- Priors on Emission Parameters:
- By placing priors on the emission parameters we capture uncertainty about the typical values of the variables when in a specific state. This enables full posterior inference over both the hidden states and the model parameters.

## Model Training
![lol](image.png)
The training process is entirely contained within HMM_train.py and follows these steps:
### 1. Data Loading:
The agent reads the cleaned avocado data from processed_avocado.csv and extracts the AveragePrice as the observation vector.
### 2. Initialization:
- Hidden States: Set to 3 (modifiable based on model complexity).
- Initial Probabilities (π): Uniformly initialized.
- Transition Matrix (A): Uniformly initialized.
- Emission Parameters:
- Means are initialized using a linear space between the minimum and maximum of the observations.
- Variances are initialized to the overall variance of the observations.
### 3. Gibbs Sampling:
- Iteration: The agent runs the Gibbs sampler for a predefined number of iterations (e.g., 1000).
- Sampling Steps:
- Hidden State Sequence Sampling: Using forward filtering and backward sampling.
- Parameter Updates: π, A, means, and variances are updated based on the posterior distributions.
- Sample Storage: Parameter samples are stored for potential further analysis.
### 4. One-Step-Ahead Prediction:
After convergence, the final model parameters are used to:
- Sample the next hidden state using the last observed state and the transition matrix.
- Generate a future observation (avocado price) from the Gaussian emission of the predicted state.
- Compute a predictive mean by averaging the means weighted by the transition probabilities.


## Conclusion and Model Evaluation
![lol](image-1.png)

### Transition Matrix A:
- State 0 (low-price) is highly self-reinforcing (~97% probability of staying in itself), indicating stable low prices.
- State 1 (mid-price) has a strong self-transition (~94%), but still allows for gradual transitions to State 2 (~5.16%).
- State 2 (high-price) is also highly stable (~95% self-transition), meaning once prices increase, they tend to remain high for a while.
- **Therefore, we come to the conclusion that given the data, the price fluctuates minimally over time**
  
The Bayesian HMM agent successfully estimates the model parameters and performs one-step-ahead predictions. Key outcomes include:
	•	Parameter Estimates:
Final estimates for initial state probabilities (π), transition matrix (A), and emission parameters provide insight into the latent regimes of avocado pricing.

### Emission Distribution:
| Hidden State       | Mean Price $ | Variance (flunctiation) |
|--------------------|--------------|-------------------------|
|State 0(Low Price)  |1.03          |0.0286                   |
|State 1(Med Price)  |1.46          |0.0250                   |
|State 2(High Price) |1.95          |0.0762                   |

- State 0 (low-price state) has the lowest variance, meaning avocado prices in this range are very stable.
- State 1 (mid-range price state) also has low variance, confirming that mid-priced avocados fluctuate very little.
- State 2 (high-price state) has the highest variance, meaning avocado prices in this range experience greater fluctuations.
- **The Bayesian HMM effectively clusters price levels into meaningful groups and correctly models price volatility.**

	•	Baseline Comparison:
A naive predictor (e.g., random guessing) would not capture the persistent regimes observed in the data. The structured Bayesian approach leverages sequential dependencies to provide significantly improved predictions.
	•	Model Evaluation:
	•	Convergence: Gibbs sampling converges to stable estimates.
	•	Prediction Quality: Both the sampled future price and the predictive mean demonstrate the model’s ability to forecast near-term market behavior.
	•	Points of Improvement:
	•	Model Complexity: Experiment with increasing the number of hidden states for finer regime segmentation.
	•	Prediction Horizon: Extend the model to predict multiple time steps ahead.
	•	Enhanced Features: Incorporate additional indicators or features to further refine the model’s performance.

### Hidden State Sequence
- The model maintains stable periods within a given state before transitioning.
- We do not see rapid, unrealistic fluctuations between states.
- The sequence follows a logical progression of price changes over time.
- **The model successfully identifies price regimes and models realistic price trends over time.**

### Possible Improvements
- Increase the Number of Hidden States (K) 
- Currently, we use 3 states (low, medium, high).
  - Adding more states (e.g., K = 4 or K = 5) could capture finer-grained trends, such as:
  - Seasonal price variations
  - Sudden price spikes
  - Short-term fluctuations
- Instead of only predicting one time step ahead, extend the model to forecast prices for multiple weeks ahead.
- This would allow us to evaluate long-term trends rather than just immediate price stability.

# Libraries/ Resources Used:
- Used LabelEncoder, which is a tool in the scikit-learn library that is used to convert categorical values into a numerical form by assigning unique integers for each distinct category.
  - [Source](https://scikit-learn.org/stable/)
- OpenAI - used to debug/get suggestions for techniques to use