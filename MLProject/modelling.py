import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(n_estimators):
    # Load data
    df = pd.read_csv("liver_preprocessed.csv")
    X = df.drop("Target", axis=1)
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Run"):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log parameter & metric
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        
        # Log model
        mlflow.sklearn.log_model(rf, "model_liver")
        
        with open("metrics.txt", "w") as f:
            f.write(f"Accuracy: {acc}")
        mlflow.log_artifact("metrics.txt")
        
        print(f"Training selesai. Akurasi: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    train(args.n_estimators)