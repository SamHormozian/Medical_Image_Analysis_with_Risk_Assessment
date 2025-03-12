import numpy as np
import pandas as pd

# -------------------------------
# Helper Functions
# -------------------------------

def gaussian_pdf(x, mean, var):
    """Compute the Gaussian probability density for observation x."""
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

def forward_filtering(obs, pi, A, means, variances):
    """
    Forward algorithm to compute filtered state probabilities (alpha).
    We normalize at each time step for numerical stability.
    """
    T = len(obs)
    K = len(pi)
    alpha = np.zeros((T, K))
    # Initialization:
    for i in range(K):
        alpha[0, i] = pi[i] * gaussian_pdf(obs[0], means[i], variances[i])
    alpha[0] /= np.sum(alpha[0])
    
    # Recursion:
    for t in range(1, T):
        for j in range(K):
            temp = 0.0
            for i in range(K):
                temp += alpha[t-1, i] * A[i, j]
            alpha[t, j] = temp * gaussian_pdf(obs[t], means[j], variances[j])
        alpha[t] /= np.sum(alpha[t])
    return alpha

def backward_sampling(alpha, A):
    """
    Backward sampling to sample the hidden state sequence.
    Given the filtered probabilities from forward filtering, sample states backwards.
    """
    T, K = alpha.shape
    s = np.zeros(T, dtype=int)
    # Sample last state:
    s[T-1] = np.random.choice(K, p=alpha[T-1])
    # Backward recursion:
    for t in range(T-2, -1, -1):
        probs = np.zeros(K)
        for i in range(K):
            probs[i] = alpha[t, i] * A[i, s[t+1]]
        probs /= np.sum(probs)
        s[t] = np.random.choice(K, p=probs)
    return s

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

# Load the cleaned avocado data (ensure the CSV file is accessible).
df = pd.read_csv("processed_avocado.csv")
# We assume the data is sorted by time. If not, sort by an appropriate time column (e.g., 'year').
observations = df['AveragePrice'].values  # observations (T,)

T = len(observations)  # number of time steps
K = 3                 # number of hidden states (you can adjust this)
num_iter = 1000       # number of Gibbs sampling iterations

# -------------------------------
# Prior Hyperparameters
# -------------------------------

# Prior for initial state distribution (π): Dirichlet with all ones.
alpha_pi = np.ones(K)

# Prior for each row of the transition matrix (A): Dirichlet with ones.
alpha_A = np.ones(K)

# Emission parameters: Gaussian likelihood with Normal–Inverse–Gamma prior.
# Prior for mean: Normal(mu0, sigma2/kappa0)
# Prior for variance: Inverse-Gamma(alpha0, beta0)
mu0 = np.mean(observations)
kappa0 = 1.0
alpha0 = 2.0         # shape parameter for variance prior (must be > 0)
beta0 = np.var(observations)  # scale parameter

# -------------------------------
# Initialize Parameters
# -------------------------------

# Initial state probabilities:
pi = np.full(K, 1.0/K)

# Transition matrix:
A = np.full((K, K), 1.0/K)

# Emission parameters: initialize means and variances.
means = np.linspace(observations.min(), observations.max(), K)
variances = np.full(K, np.var(observations))

# Initialize hidden state sequence randomly:
s = np.random.choice(K, size=T)

# For storing samples (if desired for later analysis)
samples_pi = []
samples_A = []
samples_means = []
samples_variances = []
samples_s = []

# -------------------------------
# Gibbs Sampling Loop
# -------------------------------

for it in range(num_iter):
    # --- 1. Sample Hidden State Sequence ---
    alpha = forward_filtering(observations, pi, A, means, variances)
    s = backward_sampling(alpha, A)
    
    # --- 2. Sample Initial State Probabilities (pi) ---
    # Only one observation at t=0 contributes.
    counts_pi = np.zeros(K)
    counts_pi[s[0]] += 1
    pi = np.random.dirichlet(alpha_pi + counts_pi)
    
    # --- 3. Sample Transition Matrix (A) ---
    counts_A = np.zeros((K, K))
    for t in range(T - 1):
        counts_A[s[t], s[t+1]] += 1
    for i in range(K):
        A[i] = np.random.dirichlet(alpha_A + counts_A[i])
    
    # --- 4. Sample Emission Parameters for Each Hidden State ---
    for k in range(K):
        idx = np.where(s == k)[0]
        n_k = len(idx)
        if n_k > 0:
            y_k = observations[idx]
            # Update posterior hyperparameters for state k:
            kappa_n = kappa0 + n_k
            mu_n = (kappa0 * mu0 + np.sum(y_k)) / kappa_n
            alpha_n = alpha0 + n_k / 2.0
            y_bar = np.mean(y_k)
            beta_n = beta0 + 0.5 * np.sum((y_k - y_bar)**2) + (kappa0 * n_k * (y_bar - mu0)**2) / (2 * kappa_n)
            
            # Sample variance: using Gamma sampling, then invert.
            sigma2_sample = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
            variances[k] = sigma2_sample
            # Sample mean:
            means[k] = np.random.normal(mu_n, np.sqrt(sigma2_sample / kappa_n))
        else:
            # If no observations for state k, sample from the prior.
            variances[k] = 1.0 / np.random.gamma(alpha0, 1.0 / beta0)
            means[k] = np.random.normal(mu0, np.sqrt(variances[k] / kappa0))
    
    # --- Store Samples for Later Analysis (optional) ---
    samples_pi.append(pi.copy())
    samples_A.append(A.copy())
    samples_means.append(means.copy())
    samples_variances.append(variances.copy())
    samples_s.append(s.copy())
    
    if it % 100 == 0:
        print(f"Iteration {it} complete.")

# -------------------------------
# Final Parameter Estimates
# -------------------------------
print("\nFinal estimated initial probabilities (pi):", pi)
print("Final estimated transition matrix (A):\n", A)
print("Final estimated emission means:", means)
print("Final estimated emission variances:", variances)

# -------------------------------
# One-Step-Ahead Prediction
# -------------------------------

# We'll use the final iteration's parameters to predict the next observation.
# 1. Get the last hidden state from the sampled sequence.
last_state = s[-1]

# 2. Use the transition matrix to get the probability distribution for the next state.
next_state_probs = A[last_state]
# Option 1: Sample the next state:
predicted_state = np.random.choice(K, p=next_state_probs)
# 3. Sample a future observation from the Gaussian emission of the predicted state:
predicted_price_sample = np.random.normal(means[predicted_state], np.sqrt(variances[predicted_state]))

# Option 2: Compute the predictive mean by averaging over states:
predictive_mean = np.sum(next_state_probs * means)

print("\nOne-step-ahead prediction:")
print("Predicted next state (sampled):", predicted_state)
print("Predicted future price (sampled):", predicted_price_sample)
print("Predictive mean (weighted average):", predictive_mean)
