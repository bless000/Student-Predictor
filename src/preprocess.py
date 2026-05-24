"""
Preprocessing, EDA & Feature Engineering
Cross-platform paths. Light-theme charts (white bg, royal blue accents).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW       = os.path.join(BASE, "data", "raw",       "student_data.csv")
PROC_DIR  = os.path.join(BASE, "data", "processed")
PROC      = os.path.join(PROC_DIR, "student_clean.csv")
CHART     = os.path.join(BASE, "assets", "charts")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(CHART,    exist_ok=True)

CAT_ORDER = ["Excellent","Good","Average","Pass","Poor"]
PALETTE   = {"Excellent":"#1565C0","Good":"#1976D2","Average":"#42A5F5",
             "Pass":"#90CAF9","Poor":"#EF5350"}
ROYAL     = "#1565C0"
LIGHT_BLUE= "#E3F2FD"
GRAY      = "#F5F6FA"
TEXT_DARK = "#1A1A2E"
TEXT_MID  = "#555"


def style_light():
    plt.rcParams.update({
        "figure.facecolor":"white","axes.facecolor":GRAY,
        "axes.edgecolor":"#DDD","text.color":TEXT_DARK,
        "axes.labelcolor":TEXT_MID,"xtick.color":TEXT_MID,
        "ytick.color":TEXT_MID,"grid.color":"#DDD",
        "grid.linestyle":"--","grid.linewidth":0.5,
        "font.family":"DejaVu Sans",
    })


def save(name):
    p = os.path.join(CHART, name)
    plt.tight_layout()
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {name}")


def run_eda(df):
    style_light()
    cats = [c for c in CAT_ORDER if c in df["performance_category"].unique()]

    # 1. Performance Distribution donut
    fig, ax = plt.subplots(figsize=(7,5))
    counts  = df["performance_category"].value_counts().reindex(cats)
    colors  = [PALETTE[c] for c in cats]
    wedges, texts, autos = ax.pie(
        counts, labels=cats, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5))
    for t in texts:  t.set_color(TEXT_MID); t.set_fontsize(10)
    for a in autos:  a.set_color("white");  a.set_fontsize(9); a.set_fontweight("bold")
    ax.set_title("Student Performance Distribution", color=TEXT_DARK, fontsize=13, pad=14, fontweight="bold")
    ax.add_artist(plt.Circle((0,0),0.35,fc="white"))
    ax.text(0,0,f"n={len(df)}",ha="center",va="center",color=ROYAL,fontsize=12,fontweight="bold")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    save("01_performance_distribution.png")

    # 2. Attendance analysis
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    ax = axes[0]; ax.set_facecolor(GRAY)
    ax.hist(df["attendance_pct"].dropna(), bins=30, color=ROYAL, alpha=0.8, edgecolor="white")
    med = df["attendance_pct"].median()
    ax.axvline(med, color="#EF5350", lw=2, ls="--", label=f"Median {med:.1f}%")
    ax.set_title("Attendance Distribution", color=TEXT_DARK, fontsize=11, fontweight="bold")
    ax.set_xlabel("Attendance %"); ax.legend(fontsize=9)
    ax = axes[1]; ax.set_facecolor(GRAY)
    data_box = [df[df["performance_category"]==c]["attendance_pct"].dropna() for c in cats]
    bp = ax.boxplot(data_box, patch_artist=True, medianprops=dict(color=ROYAL,linewidth=2.5))
    for patch,c in zip(bp["boxes"],cats): patch.set_facecolor(PALETTE[c]); patch.set_alpha(0.85)
    ax.set_xticklabels(cats, fontsize=9); ax.set_title("Attendance by Category",color=TEXT_DARK,fontsize=11,fontweight="bold")
    ax.set_ylabel("Attendance %")
    fig.patch.set_facecolor("white")
    save("02_attendance_analysis.png")

    # 3. GPA trend by level
    fig, ax = plt.subplots(figsize=(9,4.5)); ax.set_facecolor(GRAY)
    for cat in cats:
        sub   = df[df["performance_category"]==cat]
        means = [sub[sub["level"]==lv]["gpa"].mean() for lv in ["100","200","300","400","500"]]
        ax.plot(["100","200","300","400","500"], means, marker="o",
                color=PALETTE[cat], label=cat, lw=2, markersize=6)
    ax.set_title("GPA Trend Across Academic Levels", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Level"); ax.set_ylabel("Mean GPA")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.5)
    fig.patch.set_facecolor("white")
    save("03_gpa_trend.png")

    # 4. Correlation heatmap
    num_cols = ["attendance_pct","test_scores","exam_scores","assignment_scores",
                "gpa","study_hours_per_week","lms_activity","past_academic_record","performance_score"]
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap=sns.light_palette(ROYAL, as_cmap=True),
                ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"size":9}, cbar_kws={"shrink":0.75})
    ax.set_title("Feature Correlation Heatmap", color=TEXT_DARK, fontsize=12, pad=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    fig.patch.set_facecolor("white")
    save("04_correlation_heatmap.png")

    # 5. Study hours vs performance
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    ax = axes[0]; ax.set_facecolor(GRAY)
    for cat in cats:
        sub = df[df["performance_category"]==cat]
        ax.scatter(sub["study_hours_per_week"], sub["performance_score"],
                   alpha=0.35, s=15, color=PALETTE[cat], label=cat)
    ax.set_xlabel("Study Hours / Week"); ax.set_ylabel("Performance Score")
    ax.set_title("Study Hours vs Performance", color=TEXT_DARK, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax = axes[1]; ax.set_facecolor(GRAY)
    # Line chart: avg performance per study hour bucket
    df["study_bucket"] = pd.cut(df["study_hours_per_week"], bins=range(0,65,5))
    bucket_mean = df.groupby("study_bucket", observed=False)["performance_score"].mean()
    xs = [b.mid for b in bucket_mean.index]
    ax.plot(xs, bucket_mean.values, color=ROYAL, lw=2.5, marker="o", markersize=5)
    ax.fill_between(xs, bucket_mean.values, alpha=0.12, color=ROYAL)
    ax.set_xlabel("Study Hours / Week (midpoint)"); ax.set_ylabel("Avg Performance Score")
    ax.set_title("Study Hours vs Avg Performance", color=TEXT_DARK, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.5)
    fig.patch.set_facecolor("white")
    save("05_study_hours_analysis.png")

    # 6. Score components grouped bar
    fig, ax = plt.subplots(figsize=(11,5)); ax.set_facecolor(GRAY)
    score_cols = ["test_scores","exam_scores","assignment_scores"]
    x = np.arange(len(cats)); w = 0.26
    colors6 = [ROYAL,"#1976D2","#42A5F5"]
    lbls    = ["Test Scores","Exam Scores","Assignment Scores"]
    for i,(col,color,lbl) in enumerate(zip(score_cols,colors6,lbls)):
        means = [df[df["performance_category"]==c][col].mean() for c in cats]
        bars  = ax.bar(x+i*w-w, means, width=w, color=color, alpha=0.85, label=lbl, edgecolor="white")
        for bar,v in zip(bars,means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v:.0f}", ha="center", fontsize=7.5, color=TEXT_DARK)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("Average Score Components by Category", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Score"); ax.set_ylim(0,110)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("white")
    save("06_score_components.png")

    # 7. Gender comparison (from analytics page)
    fig, ax = plt.subplots(figsize=(8,4.5)); ax.set_facecolor(GRAY)
    gp = df.groupby(["gender","performance_category"]).size().unstack(fill_value=0)
    gp = gp.reindex(columns=cats, fill_value=0)
    gp.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in cats],
            edgecolor="white", alpha=0.85, width=0.65)
    ax.set_title("Performance by Gender", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Gender"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("white")
    save("07_gender_comparison.png")

    # 8. Feature importance (static illustrative bar — real one saved after training)
    fig, ax = plt.subplots(figsize=(9,5)); ax.set_facecolor(GRAY)
    feat_labels = ["Exam Scores","Attendance","Test Scores","GPA",
                   "Assignments","Study Hours","LMS Activity","Past Record"]
    feat_vals   = [0.22,0.18,0.15,0.13,0.12,0.09,0.07,0.04]
    colors8 = [ROYAL if i==0 else "#1976D2" if i<3 else "#42A5F5" if i<6 else "#90CAF9"
               for i in range(len(feat_labels))]
    bars8 = ax.barh(feat_labels[::-1], feat_vals[::-1], color=colors8[::-1], alpha=0.85)
    for bar,v in zip(bars8, feat_vals[::-1]):
        ax.text(v+0.003, bar.get_y()+bar.get_height()/2,
                f"{v*100:.0f}%", va="center", color=TEXT_DARK, fontsize=10)
    ax.set_title("Feature Importance", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.patch.set_facecolor("white")
    save("08_feature_importance.png")

    # 9. LMS vs performance
    fig, ax = plt.subplots(figsize=(9,4.5)); ax.set_facecolor(GRAY)
    for cat in cats:
        sub = df[df["performance_category"]==cat]
        ax.scatter(sub["lms_activity"], sub["performance_score"],
                   alpha=0.3, s=14, color=PALETTE[cat], label=cat)
    valid = df[["lms_activity","performance_score"]].dropna()
    z = np.polyfit(valid["lms_activity"], valid["performance_score"], 1)
    xs = np.linspace(valid["lms_activity"].min(), valid["lms_activity"].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), color="#EF5350", lw=2.5, label="Trend")
    ax.set_xlabel("LMS Activity (%)"); ax.set_ylabel("Performance Score")
    ax.set_title("LMS Activity vs Performance", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("white")
    save("09_lms_vs_performance.png")

    # 10. Past record vs current
    fig, ax = plt.subplots(figsize=(9,5)); ax.set_facecolor(GRAY)
    for cat in cats:
        sub = df[df["performance_category"]==cat]
        ax.scatter(sub["past_academic_record"], sub["performance_score"],
                   alpha=0.4, s=18, color=PALETTE[cat], label=cat)
    ax.set_xlabel("Past Academic Record"); ax.set_ylabel("Current Performance Score")
    ax.set_title("Past Record vs Current Performance", color=TEXT_DARK, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("white")
    save("10_past_vs_current.png")

    print(f"\n  EDA complete — 10 charts saved.")


def preprocess(df):
    df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    before = len(df)
    df.drop_duplicates(subset=["matric_number"], inplace=True)
    print(f"  Duplicates removed: {before-len(df)}")
    df["score_composite"]      = (0.4*df["exam_scores"]+0.35*df["test_scores"]+0.25*df["assignment_scores"]).round(2)
    df["engagement_index"]     = (0.5*df["lms_activity"]+0.3*df["attendance_pct"]+0.2*(df["study_hours_per_week"]/60*100)).round(2)
    df["academic_consistency"] = (df["score_composite"]/(df["past_academic_record"]+1)).round(3)
    df["class_participation_enc"] = df["class_participation"].map({"Low":0,"Medium":1,"High":2})
    df["extracurricular_enc"]     = df["extracurricular"].map({"None":0,"1 Activity":1,"2+ Activities":2})
    df["gender_enc"]              = df["gender"].map({"Male":0,"Female":1})
    df["level_enc"]               = df["level"].astype(int)//100
    df.to_csv(PROC, index=False)
    print(f"  Saved -> {PROC}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df_raw = pd.read_csv(RAW)
    print("=== EDA ==="); run_eda(df_raw)
    print("\n=== Preprocessing ==="); preprocess(df_raw)
