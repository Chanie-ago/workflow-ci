import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train(n_estimators):
    # Load data
    df = pd.read_csv("liver_preprocessed.csv")
    X = df.drop("Target", axis=1)
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Retraining_Run"):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        
        acc = rf.score(X_test, y_test)
        
        # Log parameter & metric ke MLflow/DagsHub
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")
        
        joblib.dump(rf, "model_liver.pkl")
        
        print(f"Training Selesai. Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    train(args.n_estimators)