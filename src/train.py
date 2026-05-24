"""
Model Training, Evaluation & Saving — 5 algorithms.
Cross-platform paths.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(BASE, "data", "processed", "student_clean.csv")
MDL_DIR = os.path.join(BASE, "models")
CHART   = os.path.join(BASE, "assets", "charts")
os.makedirs(MDL_DIR, exist_ok=True)

MODEL   = os.path.join(MDL_DIR, "best_model.pkl")
SCALER  = os.path.join(MDL_DIR, "scaler.pkl")
ENCODER = os.path.join(MDL_DIR, "label_encoder.pkl")
META    = os.path.join(MDL_DIR, "model_meta.json")

FEATURES = ["attendance_pct","test_scores","exam_scores","assignment_scores",
            "gpa","study_hours_per_week","lms_activity","past_academic_record",
            "class_participation_enc","extracurricular_enc","gender_enc","level_enc",
            "score_composite","engagement_index","academic_consistency"]
TARGET    = "performance_category"
CAT_ORDER = ["Excellent","Good","Average","Pass","Poor"]

ROYAL  = "#1565C0"; GRAY = "#F5F6FA"; TEXT_DARK = "#1A1A2E"; TEXT_MID = "#555"
COLORS = {"Excellent":"#1565C0","Good":"#1976D2","Average":"#42A5F5",
          "Pass":"#90CAF9","Poor":"#EF5350"}

def style_light():
    plt.rcParams.update({
        "figure.facecolor":"white","axes.facecolor":GRAY,
        "axes.edgecolor":"#DDD","text.color":TEXT_DARK,
        "axes.labelcolor":TEXT_MID,"xtick.color":TEXT_MID,
        "ytick.color":TEXT_MID,"grid.color":"#DDD",
        "grid.linestyle":"--","grid.linewidth":0.5,
    })

def save(name):
    p = os.path.join(CHART, name)
    plt.tight_layout()
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {name}")

def train():
    df = pd.read_csv(PROC)
    X  = df[FEATURES].fillna(df[FEATURES].median())
    y  = df[TARGET]

    le = LabelEncoder()
    le.fit(CAT_ORDER)
    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc   = accuracy_score(y_test, preds)
        prec  = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec   = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
        cv    = cross_val_score(model, X_train_s, y_train, cv=skf, scoring="f1_weighted").mean()
        results[name] = dict(accuracy=round(acc,4), precision=round(prec,4),
                             recall=round(rec,4), f1=round(f1,4), cv_score=round(cv,4))
        print(f"  [{name}]  acc={acc:.3f}  f1={f1:.3f}  cv={cv:.3f}")
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    print(f"\n  Best Model: {best_name}  (F1={best_f1:.3f})")

    # Save artefacts
    joblib.dump(best_model, MODEL)
    joblib.dump(scaler,     SCALER)
    joblib.dump(le,         ENCODER)
    meta = {"best_model":best_name,"features":FEATURES,
            "classes":CAT_ORDER,"results":results}
    with open(META,"w") as f: json.dump(meta, f, indent=2)
    print(f"  Model saved -> {MODEL}")

    # ── Evaluation charts (light theme) ──────────────────────────────────────
    style_light()

    # Chart 11: Model accuracy + F1 comparison
    names  = list(results.keys())
    accs   = [results[n]["accuracy"]*100 for n in names]
    f1s    = [results[n]["f1"]*100 for n in names]
    x      = np.arange(len(names))
    fig, ax= plt.subplots(figsize=(11,4.5)); ax.set_facecolor(GRAY)
    b1 = ax.bar(x-0.2, accs, 0.38, label="Accuracy",     color=ROYAL, alpha=0.85)
    b2 = ax.bar(x+0.2, f1s,  0.38, label="F1 (weighted)",color="#42A5F5", alpha=0.85)
    for bars in [b1,b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{bar.get_height():.1f}%", ha="center", color=TEXT_DARK, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_ylim(0,115); ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    # star best
    bi = names.index(best_name)
    ax.patches[bi].set_edgecolor(ROYAL); ax.patches[bi].set_linewidth(2.5)
    fig.patch.set_facecolor("white")
    save("11_model_comparison.png")

    # Chart 12: Confusion matrix
    preds_best = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, preds_best)
    n_classes  = cm.shape[0]
    present    = CAT_ORDER[:n_classes]
    fig, ax = plt.subplots(figsize=(7,5.5))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.light_palette(ROYAL, as_cmap=True),
                xticklabels=present, yticklabels=present,
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink":0.8})
    ax.set_title(f"Confusion Matrix — {best_name}", color=TEXT_DARK, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", color=TEXT_MID)
    ax.set_ylabel("Actual",    color=TEXT_MID)
    fig.patch.set_facecolor("white")
    save("12_confusion_matrix.png")

    # Chart 13: Feature importance (Random Forest)
    rf  = models["Random Forest"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:12]
    fig, ax = plt.subplots(figsize=(9,5)); ax.set_facecolor(GRAY)
    feat_names = [FEATURES[i].replace("_"," ").title() for i in idx]
    bar_colors = [ROYAL if i==0 else "#1976D2" if i<3 else "#42A5F5" for i in range(len(idx))]
    ax.barh(feat_names[::-1], imp[idx][::-1], color=bar_colors[::-1], alpha=0.85)
    ax.set_title("Feature Importances (Random Forest)", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.patch.set_facecolor("white")
    save("13_feature_importance_rf.png")

    # Chart 14: CV scores bar
    cv_scores = [results[n]["cv_score"]*100 for n in names]
    fig, ax   = plt.subplots(figsize=(9,4)); ax.set_facecolor(GRAY)
    bar_cols  = [ROYAL if n==best_name else "#90CAF9" for n in names]
    bars14    = ax.bar(names, cv_scores, color=bar_cols, alpha=0.88, edgecolor="white")
    for bar,v in zip(bars14, cv_scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.1f}%", ha="center", color=TEXT_DARK, fontsize=9)
    ax.set_ylim(0,115); ax.set_ylabel("CV F1-Score (%)")
    ax.set_title("5-Fold Cross-Validation Scores", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.set_xticklabels(names, rotation=12, fontsize=9)
    fig.patch.set_facecolor("white")
    save("14_cv_scores.png")

    return results, best_name

if __name__ == "__main__":
    train()
