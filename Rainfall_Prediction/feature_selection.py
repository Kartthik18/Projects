import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

OUT_DIR = Path("Rainfall_Prediction/output")

def main(k: int = 10, rf_estimators: int = 200, random_state: int = 0):
    X_train = pd.read_csv(OUT_DIR / "X_train.csv")
    y_train = pd.read_csv(OUT_DIR / "y_train.csv").squeeze()

    # Chi2 requires non-negative features
    mm = MinMaxScaler()
    X_pos = pd.DataFrame(mm.fit_transform(X_train), columns=X_train.columns)

    skb = SelectKBest(score_func=chi2, k=min(k, X_pos.shape[1]))
    skb.fit(X_pos, y_train)
    chi2_features = X_train.columns[skb.get_support()].tolist()

    rf = RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    sfm = SelectFromModel(rf, prefit=True, threshold="median")
    rf_features = X_train.columns[sfm.get_support()].tolist()

    selected = sorted(list(set(chi2_features + rf_features)))

    with open(OUT_DIR / "selected_features.json", "w") as f:
        json.dump({
            "chi2_top_k": chi2_features,
            "rf_selected": rf_features,
            "union_selected": selected
        }, f, indent=2)

    print(f"[OK] Saved selected features to {OUT_DIR/'selected_features.json'}")
    print("Chi2:", chi2_features)
    print("RF  :", rf_features)
    print("Union:", selected)

if __name__ == "__main__":
    main()
