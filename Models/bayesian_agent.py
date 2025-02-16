import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

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

def main():
    # Step 1: Load the cleaned data
    df = load_data()
    
    # Step 2: Build and train the Bayesian network
    model = build_network(df)
    
    # Step 3: Define some example evidence.
    # Note: Because you used LabelEncoder, the evidence values need to be encoded.
    # For instance, if LabelEncoder encoded 'sex' as {0: "female", 1: "male"}, choose accordingly.
    # Adjust these values based on your encoding.
    evidence = {
        "sex": 1,                   # example: 1 might correspond to 'male'
        "anatom_site_general": 2,   # example encoded value (check your data mapping)
        "age_group": 3              # example: if your bins were encoded as 0: '0-20', 1: '21-40', etc.
    }
    
    # Step 4: Perform inference using the evidence
    result = perform_inference(model, evidence)
    print("\nInference Result for Evidence:")
    print(evidence)
    print("\nProbability Distribution for 'benign_malignant':")
    print(result)

if __name__ == "__main__":
    main()