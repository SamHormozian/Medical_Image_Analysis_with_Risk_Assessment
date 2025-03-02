import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

data_path = "Data/Processed_data/metadata_cleaned_final.csv.gz"
df = pd.read_csv(data_path)

required_columns = ["sex", "anatom_site_general", "age_group", "benign_malignant"]
assert all(col in df.columns for col in required_columns), "Missing required columns."

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

model = BayesianNetwork([
    ("sex", "benign_malignant"),
    ("anatom_site_general", "benign_malignant"),
    ("age_group", "benign_malignant")
])

model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

infer = VariableElimination(model)

def predict_instance(row):
    evidence = {
        "sex": row["sex"],
        "anatom_site_general": row["anatom_site_general"],
        "age_group": row["age_group"]
    }
    query_result = infer.query(variables=["benign_malignant"], evidence=evidence)
    distribution = query_result.values
    predicted_label = int(np.argmax(distribution))
    return predicted_label

predictions = test_df.apply(predict_instance, axis=1)
true_labels = test_df["benign_malignant"]

accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, zero_division=0)
conf_mat = confusion_matrix(true_labels, predictions)

print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_mat)