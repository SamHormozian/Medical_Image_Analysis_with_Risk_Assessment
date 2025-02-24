import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import sys


def load_data():
    """
    Load the cleaned metadata CSV.
    This file was saved by your process_data.py script.
    """
    data_path = "Data/Processed_data/metadata_cleaned_final.csv.gz"
    df = pd.read_csv(data_path)
    print(f"Loaded data with {df.shape[0]} observations and {df.shape[1]} features.")
    return df

def build_network(df):
    """
    Define the Bayesian network structure and learn the CPDs from the data.
    In this example, we assume:
      - Predictors: sex, anatom_site_general, age_group
      - Target: benign_malignant
    (Note: The column names must match those in your cleaned CSV.)
    """
    # Define the network structure: each predictor influences the target.
    model = BayesianNetwork([
        ("sex", "benign_malignant"),
        ("anatom_site_general", "benign_malignant"),
        ("age_group", "benign_malignant")
    ])
    
    # Learn the CPDs using Bayesian Estimation (with a BDeu prior and smoothing)
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    
    # Print learned CPDs for inspection
    print("\nLearned Conditional Probability Distributions (CPDs):")
    for cpd in model.get_cpds():
        print(f"\nCPD for {cpd.variable}:")
        print(cpd)
    
    return model

def perform_inference(model, evidence):
    """
    Given a Bayesian network and evidence (a dict with variable:value pairs),
    perform inference to get the probability distribution for the target variable.
    """
    infer = VariableElimination(model)
    # Query the target variable 'benign_malignant'
    query_result = infer.query(variables=["benign_malignant"], evidence=evidence)
    return query_result

def get_evidence_from_input():
    """
    Retrieve evidence from command-line arguments.
    If no valid arguments are provided, default evidence is used.
    
    Expected command-line arguments (in order):
    1. sex (int)
    2. anatom_site_general (int)
    3. age_group (int)
    """
    default_evidence = {"sex": 1, "anatom_site_general": 2, "age_group": 3}
    
    if len(sys.argv) == 4:
        try:
            evidence = {
                "sex": int(sys.argv[1]),
                "anatom_site_general": int(sys.argv[2]),
                "age_group": int(sys.argv[3])
            }
            print("Using evidence from command-line arguments.")
            return evidence
        except ValueError:
            print("Invalid command-line input. Falling back to default evidence.")
    else:
        print("No command-line evidence provided. Using default evidence.")
    
    return default_evidence

def plot_inference_result(result, filename='inference_result.png'):
    """
    Plot the probability distribution for 'benign_malignant' obtained from inference.
    
    Parameters:
    - result: The inference result (e.g., a CPD from pgmpy) with a 'values' attribute.
    - filename: The name of the file where the plot will be saved.
    """
    # Extract probability values from the result.
    # The result is typically a DiscreteFactor object from pgmpy.
    probabilities = result.values
    # Create state labels based on the number of states.
    states = [f'benign_malignant({i})' for i in range(len(probabilities))]
    
    # Create the bar chart.
    plt.figure(figsize=(8, 6))
    plt.bar(states, probabilities, color='skyblue')
    plt.xlabel('benign_malignant States')
    plt.ylabel('Probability')
    plt.title('Inference Result: Probability Distribution for benign_malignant')
    plt.ylim(0, 1)  # Since these are probabilities
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot to a file.
    plt.savefig(filename)
    plt.close()
    print(f"Inference result plot saved as '{filename}'")


def main():
    # Step 1: Load the cleaned data
    df = load_data()
    
    # Step 2: Build and train the Bayesian network
    model = build_network(df)
    
    # Step 3: Define some example evidence.
    # Note: Because you used LabelEncoder, the evidence values need to be encoded.
    # For instance, if LabelEncoder encoded 'sex' as {0: "female", 1: "male"}, choose accordingly.
    # Adjust these values based on your encoding.
    evidence = evidence = get_evidence_from_input()

    
    # Step 4: Perform inference using the evidence
    result = perform_inference(model, evidence)
    print("\nInference Result for Evidence:")
    print(evidence)
    print("\nProbability Distribution for 'benign_malignant':")
    print(result)

    plot_inference_result(result, filename='inference_result.png')

if __name__ == "__main__":
    main()