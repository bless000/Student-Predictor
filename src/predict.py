"""
Prediction engine — called by the Streamlit app.
Cross-platform paths.
"""
import os, json
import numpy as np
import joblib

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL_DIR = os.path.join(BASE, "models")
MODEL   = os.path.join(MDL_DIR, "best_model.pkl")
SCALER  = os.path.join(MDL_DIR, "scaler.pkl")
ENCODER = os.path.join(MDL_DIR, "label_encoder.pkl")
META    = os.path.join(MDL_DIR, "model_meta.json")

FEATURES = ["attendance_pct","test_scores","exam_scores","assignment_scores",
            "gpa","study_hours_per_week","lms_activity","past_academic_record",
            "class_participation_enc","extracurricular_enc","gender_enc","level_enc",
            "score_composite","engagement_index","academic_consistency"]

_model = _scaler = _encoder = _meta = None

def _load():
    global _model, _scaler, _encoder, _meta
    if _model is None:
        _model   = joblib.load(MODEL)
        _scaler  = joblib.load(SCALER)
        _encoder = joblib.load(ENCODER)
        with open(META) as f:
            _meta = json.load(f)

def predict(attendance_pct, test_scores, exam_scores, assignment_scores,
            gpa, study_hours_per_week, lms_activity, past_academic_record,
            class_participation, extracurricular, gender, level):
    _load()
    cp = {"Low":0,"Medium":1,"High":2}[class_participation]
    ec = {"None":0,"1 Activity":1,"2+ Activities":2}[extracurricular]
    gn = {"Male":0,"Female":1}[gender]
    lv = int(level)//100

    sc  = round(0.4*exam_scores + 0.35*test_scores + 0.25*assignment_scores, 2)
    ei  = round(0.5*lms_activity + 0.3*attendance_pct + 0.2*(study_hours_per_week/60*100), 2)
    ac  = round(sc/(past_academic_record+1), 3)

    row   = np.array([[attendance_pct, test_scores, exam_scores, assignment_scores,
                        gpa, study_hours_per_week, lms_activity, past_academic_record,
                        cp, ec, gn, lv, sc, ei, ac]])
    row_s = _scaler.transform(row)
    enc   = _model.predict(row_s)[0]
    cat   = _encoder.inverse_transform([enc])[0]

    try:
        proba   = _model.predict_proba(row_s)[0]
        classes = _encoder.inverse_transform(range(len(proba)))
        prob_d  = {c: round(float(p)*100,1) for c,p in zip(classes, proba)}
        conf    = round(float(max(proba))*100, 1)
    except:
        prob_d  = {cat: 100.0}
        conf    = 100.0

    # Estimated score
    score = round(min(100, max(0,
        0.20*exam_scores + 0.18*test_scores + 0.15*assignment_scores
      + 0.14*gpa*20 + 0.12*attendance_pct
      + 0.08*(study_hours_per_week/60*100)
      + 0.06*lms_activity + 0.04*past_academic_record)), 1)

    return cat, prob_d, conf, score

def get_meta():
    _load()
    return _meta
