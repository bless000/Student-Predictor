"""
Dataset Generator — 1,500 student records with 15 features.
Cross-platform paths (Windows / Mac / Linux).
"""
import numpy as np
import pandas as pd
import os

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
OUT     = os.path.join(RAW_DIR, "student_data.csv")

np.random.seed(42)
N = 1500

ages     = np.random.randint(16, 30, N)
genders  = np.random.choice(["Male", "Female"], N, p=[0.52, 0.48])
levels   = np.random.choice(["100","200","300","400","500"], N, p=[0.25,0.25,0.20,0.20,0.10])
programs = np.random.choice(["Undergraduate","Postgraduate","HND"], N, p=[0.70,0.15,0.15])

attendance        = np.clip(np.random.normal(72, 18, N),  10, 100)
test_scores       = np.clip(np.random.normal(58, 18, N),   0, 100)
exam_scores       = np.clip(np.random.normal(55, 20, N),   0, 100)
assignment_scores = np.clip(np.random.normal(65, 17, N),   0, 100)
gpa               = np.clip(np.random.normal(3.0, 0.9, N), 0.5, 5.0).round(2)
study_hours       = np.clip(np.random.normal(14,  8,  N),  0,  60)
lms_activity      = np.clip(np.random.normal(55, 25,  N),  0, 100)
past_record       = np.clip(np.random.normal(60, 20,  N),  0, 100)
class_part        = np.random.choice(["Low","Medium","High"], N, p=[0.30,0.45,0.25])
extracurr         = np.random.choice(["None","1 Activity","2+ Activities"], N, p=[0.35,0.40,0.25])

cp_n = {"Low":0,"Medium":1,"High":2}
ec_n = {"None":0,"1 Activity":1,"2+ Activities":2}

perf = np.clip(
    0.20*exam_scores + 0.18*test_scores + 0.15*assignment_scores
  + 0.14*gpa*20 + 0.12*attendance + 0.08*(study_hours/60*100)
  + 0.06*lms_activity + 0.04*past_record
  + 0.02*np.array([cp_n[c] for c in class_part])*33
  + 0.01*np.array([ec_n[e] for e in extracurr])*50
  + np.random.normal(0, 4, N), 0, 100).round(1)

def cat(s):
    if s >= 75: return "Excellent"
    if s >= 60: return "Good"
    if s >= 50: return "Average"
    if s >= 40: return "Pass"
    return "Poor"

rng = np.random.default_rng(42)
fn  = ["Chukwuemeka","Ngozi","Babatunde","Amina","Emeka","Fatima","Oluwaseun",
       "Aisha","Ibrahim","Chidinma","Yusuf","Adaeze","Musa","Blessing","Tunde",
       "Halima","Ike","Grace","Ahmed","Ruth","Sola","Mary","Bello","Sandra","Kola",
       "Patience","Usman","Joy","Gbenga","Chiamaka","Segun","Vivian","Abdul",
       "Precious","Femi","Ifeoma","Lanre","Mercy","Kayode","Stella","Biodun",
       "Nkechi","Rasheed","Chioma","Gbemi","Anthonia","Dauda","Loveth","Wale","Uche"]
ln  = ["Okafor","Adeyemi","Bello","Ibrahim","Nwachukwu","Eze","Abubakar",
       "Okonkwo","Adeleke","Mohammed","Chukwu","Aliyu","Obi","Lawal","Nwosu",
       "Danjuma","Anikulapo","Ofili","Garba","Onwudiwe","Fashola","Okeke",
       "Suleiman","Onyekwere","Bakare"]

names   = [f"{rng.choice(fn)} {rng.choice(ln)}" for _ in range(N)]
matrics = [f"STU/{np.random.randint(2018,2024)}/{str(i+1).zfill(4)}" for i in range(N)]

df = pd.DataFrame({
    "full_name":              names,
    "matric_number":          matrics,
    "age":                    ages,
    "gender":                 genders,
    "program":                programs,
    "level":                  levels,
    "attendance_pct":         attendance.round(1),
    "test_scores":            test_scores.round(1),
    "exam_scores":            exam_scores.round(1),
    "assignment_scores":      assignment_scores.round(1),
    "gpa":                    gpa,
    "study_hours_per_week":   study_hours.round(1),
    "lms_activity":           lms_activity.round(1),
    "past_academic_record":   past_record.round(1),
    "class_participation":    class_part,
    "extracurricular":        extracurr,
    "performance_score":      perf,
    "performance_category":   [cat(s) for s in perf],
})

for col in ["attendance_pct","lms_activity","study_hours_per_week"]:
    idx = np.random.choice(N, int(N*0.05), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv(OUT, index=False)
print(f"Saved {N} records -> {OUT}")
print(df["performance_category"].value_counts())
