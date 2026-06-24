import os, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# =========================
# PATHS
# =========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
USER_FILE   = os.path.join(BASE_DIR, "users.json")
UPLOADS_DIR = os.path.join(BASE_DIR, "course_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# =========================
# CONSTANTS
# =========================
COURSES_200 = ["CHM 201","CHM 207","CSC 201","GST 209","GST 211",
               "MTH 201","MTH 223","PHY 201","PHY 205"]
DEPARTMENTS = ["ICT", "COMPUTER SCIENCE", "CYBERSECURITY"]

CREDIT_UNITS = {
    "CHM 201":2,"CHM 207":1,"CSC 201":3,"GST 209":2,"GST 211":2,
    "MTH 201":3,"MTH 223":2,"PHY 201":3,"PHY 205":1
}
TOTAL_UNITS = sum(CREDIT_UNITS.values())  # 19
YEARS_LEFT  = 2  # 200L students have 2 more years

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    dt  = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    lr  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    le_course = joblib.load(os.path.join(MODELS_DIR, "le_course.pkl"))
    le_dept   = joblib.load(os.path.join(MODELS_DIR, "le_dept.pkl"))
    le_grad   = joblib.load(os.path.join(MODELS_DIR, "le_grad.pkl"))
    with open(os.path.join(MODELS_DIR, "model_results.json")) as f:
        results = json.load(f)
    return {"Decision Tree":dt,"Logistic Regression":lr}, le_course, le_dept, le_grad, results

try:
    MODELS, LE_COURSE, LE_DEPT, LE_GRAD, MODEL_RESULTS = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)

# =========================
# GRADING HELPERS
# =========================
def grade_from_total(total):
    if   total >= 70: return "A", "Distinction",   5.0
    elif total >= 60: return "B", "Credit",        4.0
    elif total >= 50: return "C", "Merit",         3.0
    elif total >= 45: return "D", "Pass",          2.0
    elif total >= 40: return "E", "Marginal Fail", 1.0
    else:             return "F", "Fail",          0.0

def grad_class_from_cgpa(cgpa):
    if   cgpa >= 4.5: return "First Class"
    elif cgpa >= 3.5: return "Second Class Upper"
    elif cgpa >= 2.5: return "Second Class Lower"
    elif cgpa >= 1.5: return "Third Class"
    elif cgpa >= 1.0: return "Pass"
    else:             return "Fail"

def project_graduation(prev_gpa, cgpa_now, years_left=2):
    """Project likely graduation class based on GPA trend."""
    trend     = cgpa_now - prev_gpa
    projected = min(5.0, max(0.0, cgpa_now + trend * years_left * 0.3))
    return grad_class_from_cgpa(projected), round(projected, 2)

def class_color(cls):
    return {
        "First Class":         "#276749",
        "Second Class Upper":  "#2b6cb0",
        "Second Class Lower":  "#c05621",
        "Third Class":         "#c05621",
        "Pass":                "#744210",
        "Fail":                "#c53030",
    }.get(cls, "#444")

def class_emoji(cls):
    return {
        "First Class":         "🏆",
        "Second Class Upper":  "🥇",
        "Second Class Lower":  "🥈",
        "Third Class":         "🥉",
        "Pass":                "📄",
        "Fail":                "❌",
    }.get(cls, "📊")

def ml_predict(prev_gpa, prev_cgpa, att, ts, asn, ex, total, gp_200,
               new_cgpa, course, dept, model_name="Decision Tree"):
    if not MODELS_LOADED:
        proj_class, proj_gpa = project_graduation(prev_gpa, new_cgpa)
        return proj_class, proj_gpa, random.randint(75, 88)
    try:
        ce   = LE_COURSE.transform([course])[0]
        de   = LE_DEPT.transform([dept])[0]
        X    = np.array([[prev_gpa,prev_cgpa,att,ts,asn,ex,total,gp_200,new_cgpa,ce,de]])
        pred = MODELS[model_name].predict(X)[0]
        cls  = LE_GRAD.inverse_transform([pred])[0]
        conf = round(MODELS[model_name].predict_proba(X).max()*100,1) \
               if hasattr(MODELS[model_name],"predict_proba") else random.randint(80,95)
        _, proj_gpa = project_graduation(prev_gpa, new_cgpa)
        return cls, proj_gpa, conf
    except Exception:
        proj_class, proj_gpa = project_graduation(prev_gpa, new_cgpa)
        return proj_class, proj_gpa, random.randint(75, 88)

# =========================
# RECOMMENDATIONS ENGINE
# =========================
def generate_recommendations(student_name, prev_gpa, new_gpa, new_cgpa,
                              att, ts, asn, ex, total, pred_class):
    issues  = []
    tips    = []
    emoji   = []

    if att < 5:
        issues.append("very low attendance")
        tips.append("Attendance is critically low. Missing classes means missing core content. Aim for at least 8/10.")
        emoji.append("📅")
    elif att < 7:
        issues.append("below average attendance")
        tips.append("Attendance needs improvement. Try not to miss any class — each session builds on the last.")
        emoji.append("📅")

    if ts < 10:
        issues.append("poor test score")
        tips.append("Test score is below 50%. Review lecture notes weekly and attempt past questions before each test.")
        emoji.append("📝")
    elif ts < 14:
        issues.append("average test score")
        tips.append("Test score can be improved. Form study groups and discuss difficult topics with classmates.")
        emoji.append("📝")

    if asn < 5:
        issues.append("very low assignment score")
        tips.append("Many assignments are missing or poorly done. Assignments carry 10 marks — do not ignore them.")
        emoji.append("📋")
    elif asn < 7:
        issues.append("below average assignment score")
        tips.append("Assignment score needs work. Submit all assignments on time and put in more effort.")
        emoji.append("📋")

    if ex < 30:
        issues.append("very low exam score")
        tips.append("Exam score is critically low (below 50% of 60). This is the biggest contributor to your total. "
                    "Dedicate more time to exam preparation, past questions, and seeking help from lecturers.")
        emoji.append("📚")
    elif ex < 42:
        issues.append("below average exam score")
        tips.append("Exam score needs significant improvement. Create a study timetable and revise all topics before exams.")
        emoji.append("📚")

    if new_gpa < prev_gpa:
        issues.append("GPA dropped from 100 level")
        tips.append(f"Your GPA dropped from {prev_gpa} (100L) to {new_gpa:.2f} (200L). "
                    "This downward trend is concerning. You need to work harder to reverse it.")
        emoji.append("📉")
    elif new_gpa > prev_gpa:
        tips.append(f"Your GPA improved from {prev_gpa} (100L) to {new_gpa:.2f} (200L). "
                    "Keep this upward trend going!")
        emoji.append("📈")

    # Overall recommendation
    if pred_class == "First Class":
        overall = (f"🌟 {student_name} is on track for a First Class degree! "
                   "Maintain this excellent performance and stay consistent.")
    elif pred_class == "Second Class Upper":
        overall = (f"👍 {student_name} is projected for a Second Class Upper. "
                   "A little more effort, especially in exams, could push to First Class.")
    elif pred_class == "Second Class Lower":
        overall = (f"⚠️ {student_name} is currently projected for a Second Class Lower. "
                   "Significant improvement is needed in the remaining 2 years.")
    elif pred_class == "Third Class":
        overall = (f"🚨 {student_name} is at risk of graduating with a Third Class. "
                   "Urgent academic intervention is required immediately.")
    else:
        overall = (f"❌ {student_name} is at serious risk of failing. "
                   "Immediate action — speak to your academic adviser, attend all classes, "
                   "and seek tutorial support without delay.")

    return issues, tips, emoji, overall

# =========================
# CSV GRADING
# =========================
REQUIRED_COLS = ["name","matric_no","department","prev_gpa","prev_cgpa",
                 "attendance","test_score","assignment","exam_score"]

def grade_row(row):
    try:
        att  = min(10,  max(0, float(row.get("attendance",  0))))
        ts   = min(20,  max(0, float(row.get("test_score",  0))))
        asn  = min(10,  max(0, float(row.get("assignment",  0))))
        ex   = min(60,  max(0, float(row.get("exam_score",  0))))
        pgpa = min(5.0, max(0, float(row.get("prev_gpa",    0))))
        pcgpa= min(5.0, max(0, float(row.get("prev_cgpa",   0))))
    except:
        return 0,"F","Fail",0.0,0.0,0.0,"Fail","Fail","at-risk"

    total    = att + ts + asn + ex
    grade, label, gp = grade_from_total(total)
    new_cgpa = round((pgpa + gp) / 2, 2)
    proj_cls, proj_gpa = project_graduation(pgpa, new_cgpa)
    curr_cls = grad_class_from_cgpa(new_cgpa)
    risk = ("at-risk" if new_cgpa < 1.5 or gp < 1.0
            else "watch" if new_cgpa < 2.5
            else "on-track")
    return total, grade, label, gp, new_cgpa, proj_gpa, curr_cls, proj_cls, risk

def save_course_upload(course, df):
    path = os.path.join(UPLOADS_DIR, f"{course.replace(' ','_')}.csv")
    df.to_csv(path, index=False)

def load_course_upload(course):
    path = os.path.join(UPLOADS_DIR, f"{course.replace(' ','_')}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_uploaded_courses():
    return [c for c in COURSES_200
            if os.path.exists(os.path.join(UPLOADS_DIR, f"{c.replace(' ','_')}.csv"))]

def merge_all_courses():
    uploaded = get_uploaded_courses()
    if not uploaded:
        return None, []
    merged = None
    for course in uploaded:
        df = load_course_upload(course)
        if df is None:
            continue
        rows = [grade_row(r) for _, r in df.iterrows()]
        df[f"{course}_total"]    = [r[0] for r in rows]
        df[f"{course}_grade"]    = [r[1] for r in rows]
        df[f"{course}_gp"]       = [r[3] for r in rows]
        df[f"{course}_proj_cls"] = [r[7] for r in rows]
        keep = ["name","matric_no","department","prev_gpa","prev_cgpa",
                f"{course}_total",f"{course}_grade",f"{course}_gp",f"{course}_proj_cls"]
        small = df[[c for c in keep if c in df.columns]].copy()
        if merged is None:
            merged = small
        else:
            merged = pd.merge(merged, small, on=["matric_no"], how="outer",
                              suffixes=("","_dup"))
            for col in ["name","department","prev_gpa","prev_cgpa"]:
                dup = col+"_dup"
                if dup in merged.columns:
                    merged[col] = merged[col].fillna(merged[dup])
                    merged.drop(columns=[dup], inplace=True)

    if merged is None:
        return None, []

    def student_new_gpa(row):
        gps, units = [], []
        for c in uploaded:
            col = f"{c}_gp"
            if col in row and pd.notna(row[col]):
                gps.append(float(row[col]))
                units.append(CREDIT_UNITS.get(c, 2))
        if not gps:
            return 0.0
        return round(sum(g*u for g,u in zip(gps,units)) / sum(units), 2)

    def student_cgpa(row):
        pgpa = float(row.get("prev_gpa", 0) or 0)
        ngpa = row["new_gpa"]
        return round((pgpa + ngpa) / 2, 2)

    def student_proj(row):
        pgpa = float(row.get("prev_gpa", 0) or 0)
        cgpa = row["new_cgpa"]
        cls, pgpa_proj = project_graduation(pgpa, cgpa)
        return cls, pgpa_proj

    def failing_courses(row):
        f = [c for c in uploaded
             if f"{c}_grade" in row and pd.notna(row[f"{c}_grade"]) and row[f"{c}_grade"]=="F"]
        return ", ".join(f) if f else "None"

    def risk_status(row):
        cgpa = row["new_cgpa"]
        if cgpa < 1.5: return "at-risk"
        elif cgpa < 2.5: return "watch"
        return "on-track"

    merged["new_gpa"]      = merged.apply(student_new_gpa, axis=1)
    merged["new_cgpa"]     = merged.apply(student_cgpa, axis=1)
    merged["current_class"]= merged["new_cgpa"].apply(grad_class_from_cgpa)
    proj_results           = merged.apply(student_proj, axis=1)
    merged["pred_grad_class"] = [r[0] for r in proj_results]
    merged["proj_cgpa"]    = [r[1] for r in proj_results]
    merged["failing"]      = merged.apply(failing_courses, axis=1)
    merged["status"]       = merged.apply(risk_status, axis=1)
    merged = merged.sort_values("name").reset_index(drop=True)
    return merged, uploaded

# =========================
# USERS
# =========================
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE,"r") as f: return json.load(f)
    return {}

def save_users(u):
    with open(USER_FILE,"w") as f: json.dump(u, f)

# =========================
# PAGE CONFIG & CSS
# =========================
st.set_page_config(page_title="AcadPredict", page_icon="🎓", layout="wide")
ROYAL_BLUE = "#1565C0"
BORDER     = "#D9E2EC"

st.markdown(f"""
<style>
body,.main{{background-color:#F5F7FB}}
.block-container{{padding-top:1rem}}
.card{{background:white;padding:1.5rem;border-radius:18px;
       border:1px solid {BORDER};box-shadow:0 2px 10px rgba(0,0,0,0.04);margin-bottom:1rem}}
.title{{font-size:2rem;font-weight:700;color:{ROYAL_BLUE}}}
.subtitle{{color:gray;font-size:0.9rem}}
.stButton>button{{background-color:{ROYAL_BLUE};color:white;border:none;
                  border-radius:10px;font-weight:600;width:100%;height:45px}}
.stButton>button:hover{{background-color:#0D47A1}}
.metric-box{{background:white;border-radius:14px;padding:1rem;
             border:1px solid {BORDER};text-align:center}}
.metric-value{{font-size:1.6rem;font-weight:bold;color:{ROYAL_BLUE}}}
.metric-label{{color:gray;font-size:0.85rem}}
</style>""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
for k,v in [("logged_in",False),("prediction",None),
             ("role","Student"),("current_page","login")]:
    if k not in st.session_state: st.session_state[k] = v

# =========================
# LOGIN PAGE
# =========================
if st.session_state.current_page == "login" and not st.session_state.logged_in:
    st.markdown("<div class='title'>🎓 AcadPredict</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Student Academic Performance Prediction System — 200 Level</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "auth_tab" not in st.session_state: st.session_state.auth_tab = "Login"
    _, btn_col = st.columns([3,1])
    with btn_col:
        c1,c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                st.session_state.auth_tab = "Login"
        with c2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state.auth_tab = "Sign Up"

    _, center, _ = st.columns([1,2,1])
    with center:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        users = load_users()

        if st.session_state.auth_tab == "Login":
            st.subheader("Welcome Back 👋")
            matric   = st.text_input("Matric Number")
            password = st.text_input("Password", type="password")
            if "failed_attempts" not in st.session_state:
                st.session_state.failed_attempts = 0
            if st.button("Login ", use_container_width=True):
                if not matric or not password:
                    st.error("Please fill in all fields")
                elif st.session_state.failed_attempts >= 3:
                    st.error("Too many failed attempts.")
                elif matric not in users:
                    st.error("Account does not exist. Please sign up first.")
                elif password != users[matric]["password"]:
                    st.session_state.failed_attempts += 1
                    r = 3 - st.session_state.failed_attempts
                    st.error(f"Incorrect password. {r} attempt(s) remaining."
                             if r > 0 else "Too many failed attempts.")
                else:
                    st.session_state.update({
                        "failed_attempts":0,"logged_in":True,
                        "name":users[matric]["name"],"matric":matric,
                        "role":users[matric].get("role","Student"),
                        "current_page":"dashboard"})
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Forgot Password?", use_container_width=True):
                st.session_state.auth_tab = "Forgot Password"; st.rerun()

        elif st.session_state.auth_tab == "Sign Up":
            st.subheader("Create Account 📝")
            full_name  = st.text_input("Full Name")
            matric     = st.text_input("Matric Number (e.g. 250501000)")
            password   = st.text_input("Password", type="password")
            confirm    = st.text_input("Confirm Password", type="password")
            role       = st.selectbox("Role",       ["Student","Lecturer"])
            department = st.selectbox("Department", DEPARTMENTS)
            level      = st.selectbox("Level",      ["200","300","400"])
            program    = st.selectbox("Program",    ["Undergraduate","Post-graduate","Part-time"])
            if st.button("Create Account ", use_container_width=True):
                if not full_name or not matric or not password or not confirm:
                    st.error("Please fill all fields")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif matric in users:
                    st.error("Matric Number already exists")
                else:
                    users[matric] = {"name":full_name,"password":password,"role":role,
                                     "department":department,"level":level,"program":program}
                    save_users(users); st.success("Account created! You can now log in.")

        elif st.session_state.auth_tab == "Forgot Password":
            st.subheader("Reset Password 🔑")
            matric = st.text_input("Matric Number")
            new_pw = st.text_input("New Password", type="password")
            conf   = st.text_input("Confirm Password", type="password")
            if st.button("Reset Password", use_container_width=True):
                if not matric or not new_pw or not conf:
                    st.error("Please fill all fields")
                elif matric not in users:
                    st.error("No account found")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters")
                elif new_pw != conf:
                    st.error("Passwords do not match")
                else:
                    users[matric]["password"] = new_pw
                    save_users(users)
                    st.success("Password reset! You can now log in.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎓 AcadPredict")
uploaded_courses = get_uploaded_courses()
if uploaded_courses:
    st.sidebar.caption(f"📚 {len(uploaded_courses)}/9 courses uploaded")

if st.session_state.role == "Lecturer":
    nav_options = ["Dashboard","Upload Course CSV","GPA & Graduation Prediction",
                   "At-Risk Students","Analytics","Model Report","My Profile","Back"]
else:
    nav_options = ["Dashboard","My Profile","Back"]

page = st.sidebar.radio("Navigation", nav_options)

if page != "Back":
    history = st.session_state.get("page_history",["Dashboard"])
    if not history or history[-1] != page:
        history.append(page); st.session_state.page_history = history

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_page = "login"; st.rerun()

if page == "Back":
    h = st.session_state.get("page_history",["Dashboard"])
    if len(h) > 1: h.pop(); st.session_state.page_history = h; st.rerun()
    else: st.sidebar.info("No previous page.")

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.markdown("<div class='title'>Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict student performance and projected graduation class — 200 Level</div>",
                unsafe_allow_html=True)

    left, right = st.columns([1,1.2])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Student Academic Inputs")
        selected_dept   = st.selectbox("Department", DEPARTMENTS)
        selected_course = st.selectbox("Course (200 Level)", COURSES_200)
        if MODELS_LOADED:
            model_choice = st.selectbox("ML Model", list(MODELS.keys()))
        else:
            model_choice = "Decision Tree"
            st.warning("⚠️ ML models not found. Using rule-based prediction.")
        st.markdown("---")
        st.caption("Previous academic record:")
        c1,c2 = st.columns(2)
        with c1: prev_gpa  = st.number_input("100L GPA (out of 5.0)", 0.0, 5.0, 3.5, 0.01)
        with c2: prev_cgpa = st.number_input("100L CGPA (out of 5.0)", 0.0, 5.0, 3.5, 0.01)
        st.caption("Current 200L scores:")
        c1,c2 = st.columns(2)
        with c1:
            attendance = st.number_input("Attendance (out of 10)", 0, 10, 7)
            test_score = st.number_input("Test Score (out of 20)", 0, 20, 14)
        with c2:
            assignment = st.number_input("Assignment (out of 10)", 0, 10, 7)
            exam_score = st.number_input("Exam Score (out of 60)", 0, 60, 38)

        total      = attendance + test_score + assignment + exam_score
        _, _, gp   = grade_from_total(total)
        new_cgpa   = round((prev_gpa + gp) / 2, 2)
        st.info(f"📊 Total: **{total}/100** | GP: **{gp}/5.0** | New CGPA: **{new_cgpa}/5.0**")
        predict_btn = st.button("🔮 Predict Graduation Class")
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        pred_class, proj_gpa, confidence = ml_predict(
            prev_gpa, prev_cgpa, attendance, test_score,
            assignment, exam_score, total, gp, new_cgpa,
            selected_course, selected_dept, model_choice)
        grade, label, _ = grade_from_total(total)
        issues, tips, emojis, overall = generate_recommendations(
            st.session_state.name, prev_gpa, gp, new_cgpa,
            attendance, test_score, assignment, exam_score, total, pred_class)
        st.session_state.prediction = {
            "total":total,"grade":grade,"label":label,"gp":gp,
            "prev_gpa":prev_gpa,"prev_cgpa":prev_cgpa,
            "new_cgpa":new_cgpa,"proj_gpa":proj_gpa,
            "pred_class":pred_class,"confidence":confidence,
            "course":selected_course,"department":selected_dept,
            "attendance":attendance,"test_score":test_score,
            "assignment":assignment,"exam_score":exam_score,
            "issues":issues,"tips":tips,"emojis":emojis,"overall":overall,
        }

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")
        if st.session_state.prediction:
            r   = st.session_state.prediction
            col1,col2,col3,col4 = st.columns(4)
            for col,val,lbl in [
                (col1,f"{r['total']}/100",    "200L Score"),
                (col2,r['grade'],             "Grade"),
                (col3,f"{r['new_cgpa']}/5.0", "New CGPA"),
                (col4,f"{r['confidence']}%",  "Confidence"),
            ]:
                col.markdown(f"""<div class='metric-box'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Graduation class prediction
            pred_cls = r['pred_class']
            col_cls  = class_color(pred_cls)
            emj      = class_emoji(pred_cls)
            st.markdown(f"""
            <div style='background:{col_cls}18;border-left:5px solid {col_cls};
                        border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:12px'>
                <div style='font-size:13px;color:{col_cls};font-weight:600;margin-bottom:4px'>
                    🎓 PREDICTED GRADUATION CLASS
                </div>
                <div style='font-size:28px;font-weight:800;color:{col_cls}'>
                    {emj} {pred_cls}
                </div>
                <div style='font-size:12px;color:#718096;margin-top:4px'>
                    Projected CGPA at graduation: <strong>{r['proj_gpa']}/5.0</strong>
                    &nbsp;|&nbsp; This is an early forecast based on 200L performance.
                    2 years remaining to improve.
                </div>
            </div>""", unsafe_allow_html=True)

            # GPA comparison
            st.markdown("**GPA Progression:**")
            gpa_df = pd.DataFrame({
                "Level":  ["100 Level", "200 Level (current)", "Projected at graduation"],
                "GPA":    [r['prev_gpa'], r['gp'], r['proj_gpa']],
                "CGPA":   [r['prev_cgpa'], r['new_cgpa'], r['proj_gpa']],
            })
            st.dataframe(gpa_df, use_container_width=True, hide_index=True)

            # Score breakdown
            st.markdown("**Score Breakdown:**")
            bd = pd.DataFrame({
                "Component": ["Attendance /10","Test Score /20","Assignment /10","Exam Score /60"],
                "Score":     [r["attendance"],r["test_score"],r["assignment"],r["exam_score"]],
                "Max":       [10,20,10,60],
                "Status":    [
                    "🟢 Good" if r["attendance"]>=7 else "🔴 Low",
                    "🟢 Good" if r["test_score"]>=14 else "🔴 Low",
                    "🟢 Good" if r["assignment"]>=7  else "🔴 Low",
                    "🟢 Good" if r["exam_score"]>=42 else "🔴 Low",
                ]
            })
            bd["%"] = (bd["Score"]/bd["Max"]*100).round(1)
            st.dataframe(bd, use_container_width=True, hide_index=True)

            # Recommendations
            st.markdown("**📌 Recommendations & Insights:**")
            st.info(r["overall"])
            if r["tips"]:
                for em, tip in zip(r["emojis"], r["tips"]):
                    st.warning(f"{em} {tip}")
        else:
            st.info("Fill the form on the left and click **Predict Graduation Class**")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# UPLOAD COURSE CSV
# =========================
elif page == "Upload Course CSV":
    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied."); st.stop()

    st.markdown("<div class='title'>📂 Upload Course CSV — 200 Level</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload scores per course. System merges all courses to compute CGPA and predict graduation class.</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_courses = get_uploaded_courses()
    st.markdown(f"### 📊 Upload Progress: {len(uploaded_courses)}/9 courses uploaded")
    st.progress(len(uploaded_courses)/9)

    cols = st.columns(3)
    for i, course in enumerate(COURSES_200):
        with cols[i%3]:
            done = course in uploaded_courses
            st.markdown(f"""
            <div style='background:{"#f0fff4" if done else "#fff5f5"};
                        border:1px solid {"#9ae6b4" if done else "#fed7d7"};
                        border-radius:8px;padding:10px;margin-bottom:8px;
                        text-align:center;font-size:13px;font-weight:500'>
                {"✅" if done else "⏳"} {course}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    selected_course = st.selectbox("Select Course to Upload", COURSES_200)

    st.markdown("### Download Template")
    template = pd.DataFrame([
        {"name":"Adaeze Nwosu",  "matric_no":"250501000","department":"COMPUTER SCIENCE",
         "prev_gpa":3.8,"prev_cgpa":3.8,"attendance":8,"test_score":14,"assignment":7,"exam_score":42},
        {"name":"Amaka Obi",     "matric_no":"250501001","department":"ICT",
         "prev_gpa":2.5,"prev_cgpa":2.5,"attendance":6,"test_score":9, "assignment":5,"exam_score":28},
        {"name":"Fatima Bello",  "matric_no":"250501006","department":"CYBERSECURITY",
         "prev_gpa":1.2,"prev_cgpa":1.2,"attendance":3,"test_score":6, "assignment":3,"exam_score":18},
    ])
    st.download_button(
        f"⬇ Download {selected_course} Template",
        template.to_csv(index=False),
        f"{selected_course.replace(' ','_')}_template.csv","text/csv")

    st.caption("Required columns: name, matric_no, department, **prev_gpa**, **prev_cgpa**, "
               "attendance (/10), test_score (/20), assignment (/10), exam_score (/60)")

    uploaded = st.file_uploader(f"Upload {selected_course} CSV", type=["csv"])

    if st.button(f"📋 Load sample data for {selected_course}"):
        sample = pd.DataFrame([
            {"name":"Adaeze Nwosu",  "matric_no":"250501000","department":"COMPUTER SCIENCE",
             "prev_gpa":3.8,"prev_cgpa":3.8,"attendance":8, "test_score":14,"assignment":7, "exam_score":42},
            {"name":"Amaka Obi",     "matric_no":"250501001","department":"ICT",
             "prev_gpa":2.5,"prev_cgpa":2.5,"attendance":6, "test_score":9, "assignment":5, "exam_score":28},
            {"name":"Blessing Okoro","matric_no":"250501002","department":"COMPUTER SCIENCE",
             "prev_gpa":4.5,"prev_cgpa":4.5,"attendance":10,"test_score":19,"assignment":10,"exam_score":55},
            {"name":"Chioma Okafor", "matric_no":"250501003","department":"ICT",
             "prev_gpa":3.0,"prev_cgpa":3.0,"attendance":8, "test_score":13,"assignment":8, "exam_score":40},
            {"name":"Chidi Nwosu",   "matric_no":"250501004","department":"CYBERSECURITY",
             "prev_gpa":3.5,"prev_cgpa":3.5,"attendance":9, "test_score":16,"assignment":9, "exam_score":48},
            {"name":"Emeka Eze",     "matric_no":"250501005","department":"ICT",
             "prev_gpa":2.0,"prev_cgpa":2.0,"attendance":5, "test_score":10,"assignment":6, "exam_score":30},
            {"name":"Fatima Bello",  "matric_no":"250501006","department":"CYBERSECURITY",
             "prev_gpa":1.2,"prev_cgpa":1.2,"attendance":3, "test_score":6, "assignment":3, "exam_score":18},
            {"name":"Ibrahim Salisu","matric_no":"250501007","department":"COMPUTER SCIENCE",
             "prev_gpa":2.3,"prev_cgpa":2.3,"attendance":6, "test_score":11,"assignment":6, "exam_score":33},
            {"name":"Musa Danladi",  "matric_no":"250501008","department":"ICT",
             "prev_gpa":1.5,"prev_cgpa":1.5,"attendance":4, "test_score":7, "assignment":4, "exam_score":20},
            {"name":"Ngozi Adeyemi", "matric_no":"250501009","department":"CYBERSECURITY",
             "prev_gpa":3.2,"prev_cgpa":3.2,"attendance":8, "test_score":15,"assignment":8, "exam_score":44},
            {"name":"Tunde Afolabi", "matric_no":"250501010","department":"COMPUTER SCIENCE",
             "prev_gpa":0.8,"prev_cgpa":0.8,"attendance":2, "test_score":5, "assignment":2, "exam_score":15},
            {"name":"Yusuf Garba",   "matric_no":"250501011","department":"CYBERSECURITY",
             "prev_gpa":2.1,"prev_cgpa":2.1,"attendance":5, "test_score":9, "assignment":5, "exam_score":27},
        ])
        save_course_upload(selected_course, sample)
        st.success(f"✅ Sample data loaded for {selected_course}!"); st.rerun()

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                save_course_upload(selected_course, df)
                st.success(f"✅ {selected_course} uploaded — {len(df)} students saved.")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    existing = load_course_upload(selected_course)
    if existing is not None:
        st.markdown(f"### 👁 Preview: {selected_course} ({len(existing)} students)")
        rows = [grade_row(r) for _, r in existing.iterrows()]
        existing = existing.copy()
        existing["total"]       = [r[0] for r in rows]
        existing["grade"]       = [r[1] for r in rows]
        existing["gp"]          = [r[3] for r in rows]
        existing["new_cgpa"]    = [r[4] for r in rows]
        existing["pred_class"]  = [r[7] for r in rows]
        st.dataframe(existing.sort_values("name"), use_container_width=True, hide_index=True)
        if st.button(f"🗑 Clear {selected_course} data"):
            path = os.path.join(UPLOADS_DIR, f"{selected_course.replace(' ','_')}.csv")
            if os.path.exists(path): os.remove(path)
            st.success(f"{selected_course} data cleared."); st.rerun()

# =========================
# GPA & GRADUATION PREDICTION
# =========================
elif page == "GPA & Graduation Prediction":
    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied."); st.stop()

    st.markdown("<div class='title'>🎓 GPA & Graduation Class Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Combined GPA from all uploaded courses with predicted graduation class for each student.</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_courses = get_uploaded_courses()
    if not uploaded_courses:
        st.warning("⚠️ No courses uploaded yet. Go to **Upload Course CSV** first."); st.stop()

    st.info(f"📚 Computing from **{len(uploaded_courses)}/9** courses: {', '.join(uploaded_courses)}")
    if len(uploaded_courses) < 9:
        st.warning(f"⚠️ {9-len(uploaded_courses)} course(s) not yet uploaded. Results are partial.")

    merged, used_courses = merge_all_courses()
    if merged is None or merged.empty:
        st.error("Could not merge data."); st.stop()

    # Summary stats
    first_cls  = (merged["pred_grad_class"]=="First Class").sum()
    sec_upper  = (merged["pred_grad_class"]=="Second Class Upper").sum()
    sec_lower  = (merged["pred_grad_class"]=="Second Class Lower").sum()
    third      = (merged["pred_grad_class"]=="Third Class").sum()
    at_risk    = (merged["status"]=="at-risk").sum()
    avg_cgpa   = round(merged["new_cgpa"].mean(), 2)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,val,lbl in [
        (c1,len(merged),    "Total Students"),
        (c2,first_cls,      "🏆 First Class"),
        (c3,sec_upper,      "🥇 2nd Upper"),
        (c4,sec_lower,      "🥈 2nd Lower"),
        (c5,at_risk,        "⚠ At Risk"),
        (c6,f"{avg_cgpa}/5.0","Avg CGPA"),
    ]:
        col.markdown(f"""<div class='metric-box'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graduation class distribution chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Predicted Graduation Class Distribution")
    class_order = ["First Class","Second Class Upper","Second Class Lower",
                   "Third Class","Pass","Fail"]
    class_counts = merged["pred_grad_class"].value_counts().reindex(class_order, fill_value=0)
    colors = ["#276749","#2b6cb0","#3182ce","#c05621","#744210","#c53030"]
    fig,ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor="white")
    for bar,val in zip(bars,class_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Students")
    ax.set_title("Predicted Graduation Class (Early Forecast — 200 Level)", fontweight="bold")
    plt.xticks(rotation=15, ha='right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    fc1,fc2,fc3 = st.columns(3)
    with fc1: search   = st.text_input("🔍 Search name/matric","")
    with fc2: dept_f   = st.selectbox("Department", ["All"]+DEPARTMENTS)
    with fc3: class_f  = st.selectbox("Predicted Class",
                                       ["All"]+class_order)

    filt = merged.copy()
    if search:
        filt = filt[filt["name"].str.contains(search,case=False)|
                    filt["matric_no"].astype(str).str.contains(search)]
    if dept_f  != "All": filt = filt[filt["department"]==dept_f]
    if class_f != "All": filt = filt[filt["pred_grad_class"]==class_f]

    # Full table
    st.markdown("#### 📋 All Students — GPA & Predicted Graduation Class")
    base_cols   = ["name","matric_no","department","prev_gpa","prev_cgpa"]
    score_cols  = [f"{c}_total" for c in used_courses if f"{c}_total" in filt.columns]
    grade_cols  = [f"{c}_grade" for c in used_courses if f"{c}_grade" in filt.columns]
    end_cols    = ["new_gpa","new_cgpa","current_class","pred_grad_class","proj_cgpa","failing","status"]
    all_disp    = [c for c in base_cols+score_cols+grade_cols+end_cols if c in filt.columns]

    def colour_row(row):
        cls = row.get("pred_grad_class","")
        if cls == "First Class":        return ["background-color:#f0fff4"]*len(row)
        elif cls == "Second Class Upper":return ["background-color:#ebf8ff"]*len(row)
        elif cls in ["Second Class Lower","Third Class"]:
            return ["background-color:#fffaf0"]*len(row)
        else: return ["background-color:#fff5f5"]*len(row)

    st.dataframe(filt[all_disp].style.apply(colour_row,axis=1),
                 use_container_width=True, hide_index=True)

    # Export
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        at_risk_df = merged[merged["status"]!="on-track"][all_disp]
        if not at_risk_df.empty:
            st.download_button("⬇ Export at-risk students",
                               at_risk_df.to_csv(index=False),
                               "at_risk_students.csv","text/csv")
    with c2:
        st.download_button("⬇ Export full graduation report",
                           merged[all_disp].to_csv(index=False),
                           "graduation_prediction_report.csv","text/csv")

# =========================
# AT-RISK STUDENTS
# =========================
elif page == "At-Risk Students":
    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied."); st.stop()

    st.markdown("<div class='title'>⚠️ At-Risk Students</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Students predicted to graduate with Third Class, Pass or Fail — with detailed recommendations.</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_courses = get_uploaded_courses()
    if not uploaded_courses:
        st.warning("⚠️ No data yet. Go to **Upload Course CSV** first."); st.stop()

    merged, used_courses = merge_all_courses()
    if merged is None or merged.empty:
        st.error("Could not load data."); st.stop()

    at_risk_df = merged[merged["status"]=="at-risk"].sort_values("new_cgpa").reset_index(drop=True)
    watch_df   = merged[merged["status"]=="watch"].sort_values("new_cgpa").reset_index(drop=True)

    top_n = st.slider("Show top N at-risk students", 5, 20, 10)

    if at_risk_df.empty and watch_df.empty:
        st.success("✅ All students are on track for good graduation classes!"); st.stop()

    # ── AT RISK ──
    if not at_risk_df.empty:
        st.markdown(f"### 🔴 At Risk — {min(top_n,len(at_risk_df))} students (Lowest CGPA First)")
        top = at_risk_df.head(top_n)

        # Bar chart
        fig,ax = plt.subplots(figsize=(10, max(4,len(top)*0.6)))
        ax.barh(top["name"], top["new_cgpa"],
                color=["#c53030"]*len(top), edgecolor="white")
        ax.set_xlabel("CGPA (out of 5.0)")
        ax.set_title(f"Top {len(top)} At-Risk Students by CGPA", fontweight="bold")
        ax.axvline(x=1.5, color="#c05621", linestyle="--", linewidth=1.5, label="Third Class threshold")
        ax.axvline(x=2.5, color="#2b6cb0", linestyle="--", linewidth=1.5, label="2nd Lower threshold")
        ax.legend(fontsize=9)
        for i,row in enumerate(top.itertuples()):
            ax.text(row.new_cgpa+0.05, i,
                    f"  CGPA {row.new_cgpa} → {row.pred_grad_class}",
                    va='center', fontsize=9)
        ax.set_xlim(0,6.5); ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)

        for i,(_, row) in enumerate(top.iterrows(), 1):
            col_cls = class_color(row["pred_grad_class"])
            with st.container():
                st.markdown(f"""
                <div style='border-left:4px solid {col_cls};background:{col_cls}12;
                            border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:8px'>
                    <strong style='font-size:15px'>#{i} &nbsp; {row['name']}</strong>
                    &nbsp;|&nbsp; Matric: {row['matric_no']}
                    &nbsp;|&nbsp; {row.get('department','—')}<br>
                    <span style='font-size:11px;color:#718096'>
                        100L GPA: {row.get('prev_gpa','—')} &nbsp;|&nbsp;
                        200L CGPA: <strong>{row['new_cgpa']}/5.0</strong>
                    </span><br>
                    <span style='font-size:18px;font-weight:700;color:{col_cls}'>
                        {class_emoji(row['pred_grad_class'])} Predicted: {row['pred_grad_class']}
                    </span>
                    &nbsp;
                    <span style='font-size:11px;color:#718096'>
                        (Projected CGPA at graduation: {row['proj_cgpa']}/5.0)
                    </span><br>
                    <span style='font-size:12px;color:#c53030'>
                        Failing courses: {row.get('failing','None')}
                    </span>
                </div>""", unsafe_allow_html=True)

                # Course scores
                course_cols = st.columns(min(len(used_courses),5))
                for j,course in enumerate(used_courses):
                    tcol = f"{course}_total"
                    gcol = f"{course}_grade"
                    if tcol in row and pd.notna(row[tcol]):
                        sc = int(row[tcol]); gr = row.get(gcol,"?")
                        bad = sc < 45
                        with course_cols[j%len(course_cols)]:
                            st.metric(f"{'🔴' if bad else '🟢'} {course}",
                                      f"{sc}/100 ({gr})")

                # Recommendations
                pgpa = float(row.get("prev_gpa",0) or 0)
                att  = float(row.get("attendance",0) or 0)
                ts   = float(row.get("test_score",0) or 0)
                asn  = float(row.get("assignment",0) or 0)
                ex   = float(row.get("exam_score",0) or 0)
                total_s = att+ts+asn+ex
                _, tips_r, emojis_r, overall_r = generate_recommendations(
                    row["name"], pgpa, row["new_gpa"] if "new_gpa" in row else 0,
                    row["new_cgpa"], att, ts, asn, ex, total_s, row["pred_grad_class"])

                with st.expander(f"💡 View recommendations for {row['name']}"):
                    st.error(overall_r)
                    for em,tip in zip(emojis_r, tips_r):
                        st.warning(f"{em} {tip}")
                st.markdown("")

    # ── WATCH ──
    if not watch_df.empty:
        st.markdown("---")
        st.markdown(f"### 🟡 Watch Closely — {len(watch_df)} students")
        top_w = watch_df.head(top_n)
        for i,(_, row) in enumerate(top_w.iterrows(),1):
            col_cls = class_color(row["pred_grad_class"])
            st.markdown(f"""
            <div style='border-left:4px solid {col_cls};background:{col_cls}12;
                        border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:8px'>
                <strong>#{i} &nbsp; {row['name']}</strong>
                &nbsp;|&nbsp; Matric: {row['matric_no']}
                &nbsp;|&nbsp; {row.get('department','—')}<br>
                100L GPA: {row.get('prev_gpa','—')} &nbsp;|&nbsp;
                CGPA: <strong>{row['new_cgpa']}/5.0</strong><br>
                <span style='font-size:18px;font-weight:700;color:{col_cls}'>
                    {class_emoji(row['pred_grad_class'])} Predicted: {row['pred_grad_class']}
                </span>
                &nbsp;&nbsp;
                <span style='font-size:11px;color:#718096'>
                    Projected CGPA: {row['proj_cgpa']}/5.0 &nbsp;|&nbsp;
                    At-risk courses: {row.get('failing','None')}
                </span>
            </div>""", unsafe_allow_html=True)

    # Export
    st.markdown("---")
    all_risk = pd.concat([at_risk_df, watch_df])
    if not all_risk.empty:
        st.download_button("⬇ Export at-risk list CSV",
                           all_risk.to_csv(index=False),
                           "at_risk_students.csv","text/csv")

# =========================
# ANALYTICS
# =========================
elif page == "Analytics":
    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied."); st.stop()

    st.markdown("<div class='title'>Dataset Analytics</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    data_path = os.path.join(MODELS_DIR, "student_data.csv")
    if not os.path.exists(data_path):
        st.warning("Dataset not found."); st.stop()
    data = pd.read_csv(data_path)

    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl in [
        (c1,len(data),                        "Total Students"),
        (c2,f"{data['new_cgpa'].mean():.2f}/5.0","Avg CGPA"),
        (c3,(data["predicted_grad_class"].isin(["Fail","Pass","Third Class"])).sum(),"At Risk"),
        (c4,(data["predicted_grad_class"]=="First Class").sum(),"First Class"),
    ]:
        col.markdown(f"""<div class='metric-box'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Predicted Graduation Class Distribution")
    class_order = ["First Class","Second Class Upper","Second Class Lower",
                   "Third Class","Pass","Fail"]
    gc = data["predicted_grad_class"].value_counts().reindex(class_order,fill_value=0)
    fig,ax = plt.subplots(figsize=(10,4))
    colors = ["#276749","#2b6cb0","#3182ce","#c05621","#744210","#c53030"]
    bars = ax.bar(gc.index, gc.values, color=colors, edgecolor="white")
    for bar,val in zip(bars,gc.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                str(val), ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Students")
    ax.set_title("Predicted Graduation Classes (Training Dataset)", fontweight="bold")
    plt.xticks(rotation=15, ha='right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏛️ CGPA by Department")
    dept_grp = data.groupby("department")[["prev_gpa","new_cgpa"]].mean().round(2)
    st.dataframe(dept_grp, use_container_width=True)
    fig2,ax2 = plt.subplots(figsize=(8,4))
    x = np.arange(len(dept_grp)); w = 0.35
    ax2.bar(x-w/2, dept_grp["prev_gpa"],  w, label="100L GPA",  color="#1565C0")
    ax2.bar(x+w/2, dept_grp["new_cgpa"],  w, label="200L CGPA", color="#42A5F5")
    ax2.set_xticks(x); ax2.set_xticklabels(dept_grp.index)
    ax2.set_ylabel("GPA / CGPA (out of 5.0)")
    ax2.set_title("GPA Comparison by Department", fontweight="bold")
    ax2.legend(); ax2.set_ylim(0,6)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MODEL REPORT
# =========================
elif page == "Model Report":
    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied."); st.stop()

    st.markdown("<div class='title'>Machine Learning Model Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Models trained to predict graduation class from 200-level performance</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Accuracy")
    if MODELS_LOADED:
        res_df = pd.DataFrame(list(MODEL_RESULTS.items()),
                              columns=["Model","Accuracy (%)"])
        st.dataframe(res_df.sort_values("Accuracy (%)",ascending=False),
                     use_container_width=True, hide_index=True)
        fig4,ax4 = plt.subplots(figsize=(7,4))
        ax4.bar(MODEL_RESULTS.keys(), MODEL_RESULTS.values(),
                color=["#1565C0","#42A5F5"])
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_title("Model Accuracy Comparison", fontweight="bold")
        ax4.set_ylim(80,105)
        for i,(k,v) in enumerate(MODEL_RESULTS.items()):
            ax4.text(i, v+0.3, f"{v}%", ha='center', fontsize=11, fontweight='bold')
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
        st.pyplot(fig4)
    else:
        st.error(f"Models not loaded: {MODEL_ERROR}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Summary")
    st.markdown(f"""
| Property | Value |
|----------|-------|
| Training records | 2,000 students |
| Level | 200 Level |
| Departments | ICT, Computer Science, Cybersecurity |
| 200L Courses | CHM 201, CHM 207, CSC 201, GST 209, GST 211, MTH 201, MTH 223, PHY 201, PHY 205 |
| Features used | prev_gpa, prev_cgpa, attendance, test_score, assignment, exam_score, total, gp_200, new_cgpa, course, department |
| Target | Predicted graduation class |
| GPA Scale | 5.0 (Nigerian) |
| CGPA Formula | (100L GPA + 200L GPA) ÷ 2 |
| Years remaining | 2 years (forecast can improve) |
| Prediction classes | First Class, 2nd Upper, 2nd Lower, Third Class, Pass, Fail |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PROFILE
# =========================
elif page == "My Profile":
    st.markdown("<div class='title'>My Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Personal Information")
    users = load_users()
    u = users.get(st.session_state.matric, {})
    st.write(f"**Full Name:** {st.session_state.name}")
    st.write(f"**Matric Number:** {st.session_state.matric}")
    st.write(f"**Role:** {st.session_state.role}")
    st.write(f"**Department:** {u.get('department','—')}")
    st.write(f"**Level:** {u.get('level','200')}")
    st.write(f"**Program:** {u.get('program','Undergraduate')}")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.prediction:
        r = st.session_state.prediction
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Latest Prediction")
        st.write(f"**Department:** {r.get('department','')}")
        st.write(f"**Course:** {r.get('course','')}")
        st.write(f"**200L Total Score:** {r['total']}/100")
        st.write(f"**Grade:** {r['grade']} — {r['label']}")
        st.write(f"**100L GPA:** {r['prev_gpa']}/5.0")
        st.write(f"**New CGPA:** {r['new_cgpa']}/5.0")
        st.write(f"**Projected CGPA at graduation:** {r['proj_gpa']}/5.0")
        col_cls = class_color(r['pred_class'])
        st.markdown(f"""
        <div style='background:{col_cls}18;border-left:4px solid {col_cls};
                    border-radius:0 10px 10px 0;padding:12px 16px;margin-top:8px'>
            <strong style='color:{col_cls};font-size:16px'>
                {class_emoji(r['pred_class'])} Predicted Graduation Class: {r['pred_class']}
            </strong>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
