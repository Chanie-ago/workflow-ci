import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train(n_estimators, max_depth):
    df = pd.read_csv("liver_preprocessed.csv")
    X = df.drop("Target", axis=1)
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(X_train, y_train)
        
        acc = rf.score(X_test, y_test)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")
        print(f"Model trained with accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    train(args.n_estimators, args.max_depth)