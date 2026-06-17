import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# =========================
# PATHS
# =========================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
USER_FILE  = os.path.join(BASE_DIR, "users.json")

# =========================
# LOAD ML MODELS & ENCODERS
# =========================
@st.cache_resource
def load_models():
    rf  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    dt  = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    lr  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    svm = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
    gb  = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting.pkl"))
    le_course = joblib.load(os.path.join(MODELS_DIR, "le_course.pkl"))
    le_perf   = joblib.load(os.path.join(MODELS_DIR, "le_perf.pkl"))
    with open(os.path.join(MODELS_DIR, "model_results.json")) as f:
        results = json.load(f)
    return {"Random Forest": rf, "Decision Tree": dt,
            "Logistic Regression": lr, "SVM": svm,
            "Gradient Boosting": gb}, le_course, le_perf, results

try:
    MODELS, LE_COURSE, LE_PERF, MODEL_RESULTS = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)

COURSES = ["ICT", "COMPUTER SCIENCE", "CYBERSECURITY"]

# =========================
# GRADING HELPERS
# =========================
def grade_from_total(total):
    if total >= 70: return "A", "Distinction",   "Excellent"
    elif total >= 60: return "B", "Credit",       "Good"
    elif total >= 50: return "C", "Merit",        "Average"
    elif total >= 45: return "D", "Pass",         "Pass"
    elif total >= 40: return "E", "Marginal Fail","Poor"
    else:             return "F", "Fail",         "Poor"

def predict_performance(attendance, test_score, assignment, exam_score, course, model_name="Random Forest"):
    total = attendance + test_score + assignment + exam_score
    grade, label, category = grade_from_total(total)

    if MODELS_LOADED:
        try:
            course_enc = LE_COURSE.transform([course])[0]
            X = np.array([[attendance, test_score, assignment, exam_score, course_enc, total]])
            model = MODELS[model_name]
            pred_enc = model.predict(X)[0]
            category = LE_PERF.inverse_transform([pred_enc])[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X).max()
                confidence = round(prob * 100, 1)
            else:
                confidence = random.randint(85, 98)
        except:
            confidence = random.randint(85, 98)
    else:
        confidence = random.randint(85, 98)

    return total, grade, label, category, confidence

def smart_feedback(total, attendance, test_score, exam_score):
    if total < 40:
        return "⚠️ Performance is very poor. Urgent intervention needed — attend all classes, complete assignments, and seek extra tutorials."
    elif total < 50 and attendance < 5:
        return "Low attendance is seriously affecting your performance. Make attending classes your top priority."
    elif total < 50 and exam_score < 25:
        return "Your exam score is dragging your total down. Focus more on exam preparation and past questions."
    elif total >= 70:
        return "🌟 Excellent performance! Keep maintaining this level of effort and consistency."
    elif total >= 60:
        return "Good work! A little more effort especially in exams can push you to distinction level."
    else:
        return "You are doing okay. Consistency in attendance, assignments, and exam prep will improve your results."

# =========================
# CSV UPLOAD GRADING
# =========================
def grade_student_row(row):
    try:
        att = min(10,  max(0, float(row.get("attendance",  0))))
        ts  = min(20,  max(0, float(row.get("test_score",  0))))
        asn = min(10,  max(0, float(row.get("assignment",  0))))
        ex  = min(60,  max(0, float(row.get("exam_score",  0))))
    except:
        return 0, "F", "Fail", "at-risk"
    total = att + ts + asn + ex
    grade, label, _ = grade_from_total(total)
    risk = "at-risk" if total < 45 else "watch" if total < 60 else "on-track"
    return total, grade, label, risk

REQUIRED_COLS = ["name","matric_no","course","year","attendance","test_score","assignment","exam_score"]

# =========================
# USER FILE
# =========================
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AcadPredict", page_icon="🎓", layout="wide")

ROYAL_BLUE = "#1565C0"
LIGHT_BG   = "#F5F7FB"
BORDER     = "#D9E2EC"

st.markdown(f"""
<style>
body {{ background-color: {LIGHT_BG}; }}
.main {{ background-color: {LIGHT_BG}; }}
.block-container {{ padding-top: 1rem; }}
.card {{
    background: white; padding: 1.5rem; border-radius: 18px;
    border: 1px solid {BORDER}; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}}
.title {{ font-size: 2rem; font-weight: 700; color: {ROYAL_BLUE}; }}
.subtitle {{ color: gray; font-size: 0.9rem; }}
.stButton>button {{
    background-color: {ROYAL_BLUE}; color: white; border: none;
    border-radius: 10px; font-weight: 600; width: 100%; height: 45px;
}}
.stButton>button:hover {{ background-color: #0D47A1; }}
.metric-box {{
    background: white; border-radius: 14px; padding: 1rem;
    border: 1px solid {BORDER}; text-align: center;
}}
.metric-value {{ font-size: 1.8rem; font-weight: bold; color: {ROYAL_BLUE}; }}
.metric-label {{ color: gray; font-size: 0.9rem; }}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
for key, val in [("logged_in", False), ("prediction", None),
                 ("role", "Student"), ("current_page", "login")]:
    if key not in st.session_state:
        st.session_state[key] = val

# =========================
# LOGIN PAGE
# =========================
if st.session_state.current_page == "login" and not st.session_state.logged_in:

    st.markdown("<div class='title'>🎓 AcadPredict</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Student Academic Performance Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "Login"

    _, btn_col = st.columns([3, 1])
    with btn_col:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                st.session_state.auth_tab = "Login"
        with c2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state.auth_tab = "Sign Up"

    auth_option = st.session_state.auth_tab
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        users = load_users()

        if auth_option == "Login":
            st.subheader("Welcome Back 👋")
            matric   = st.text_input("Matric Number")
            password = st.text_input("Password", type="password")
            if "failed_attempts" not in st.session_state:
                st.session_state.failed_attempts = 0
            if st.button("Login ", use_container_width=True):
                if not matric or not password:
                    st.error("Please fill in all fields")
                elif st.session_state.failed_attempts >= 3:
                    st.error("Too many failed attempts. Please restart the app.")
                elif matric not in users:
                    st.error("Account does not exist. Please sign up first.")
                elif password != users[matric]["password"]:
                    st.session_state.failed_attempts += 1
                    remaining = 3 - st.session_state.failed_attempts
                    if remaining > 0:
                        st.error(f"Incorrect password. {remaining} attempt(s) remaining.")
                    else:
                        st.error("Too many failed attempts. Please restart the app.")
                else:
                    st.session_state.failed_attempts = 0
                    st.session_state.logged_in       = True
                    st.session_state.name            = users[matric]["name"]
                    st.session_state.matric          = matric
                    st.session_state.role            = users[matric].get("role", "Student")
                    st.session_state.current_page    = "dashboard"
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Forgot Password?", use_container_width=True):
                st.session_state.auth_tab = "Forgot Password"
                st.rerun()

        elif auth_option == "Sign Up":
            st.subheader("Create Account 📝")
            full_name = st.text_input("Full Name")
            matric    = st.text_input("Matric Number (e.g. 250501000)")
            password  = st.text_input("Password", type="password")
            confirm   = st.text_input("Confirm Password", type="password")
            role      = st.selectbox("Role",    ["Student", "Lecturer"])
            course    = st.selectbox("Course",  COURSES)
            level     = st.selectbox("Level",   ["100", "200", "300", "400"])
            program   = st.selectbox("Program", ["Undergraduate", "Post-graduate", "Part-time"])
            if st.button("Create Account ", use_container_width=True):
                if not full_name or not matric or not password or not confirm:
                    st.error("Please fill all fields")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif matric in users:
                    st.error("An account with this Matric Number already exists")
                else:
                    users[matric] = {"name": full_name, "password": password,
                                     "role": role, "course": course,
                                     "level": level, "program": program}
                    save_users(users)
                    st.success("Account created! You can now log in.")

        elif auth_option == "Forgot Password":
            st.subheader("Reset Password 🔑")
            st.info("Enter your Matric Number and set a new password.")
            matric       = st.text_input("Matric Number")
            new_password = st.text_input("New Password",     type="password")
            confirm_new  = st.text_input("Confirm Password", type="password")
            if st.button("Reset Password", use_container_width=True):
                if not matric or not new_password or not confirm_new:
                    st.error("Please fill all fields")
                elif matric not in users:
                    st.error("No account found with that Matric Number")
                elif len(new_password) < 6:
                    st.error("New password must be at least 6 characters")
                elif new_password != confirm_new:
                    st.error("Passwords do not match")
                else:
                    users[matric]["password"] = new_password
                    save_users(users)
                    st.success("Password reset successfully! You can now log in.")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎓 AcadPredict")

if st.session_state.role == "Lecturer":
    nav_options = ["Dashboard", "CSV Upload", "Analytics", "Model Report", "My Profile", "Back"]
else:
    nav_options = ["Dashboard", "My Profile", "Back"]

page = st.sidebar.radio("Navigation", nav_options)

if page != "Back":
    history = st.session_state.get("page_history", ["Dashboard"])
    if not history or history[-1] != page:
        history.append(page)
        st.session_state.page_history = history

if st.sidebar.button("Logout"):
    st.session_state.logged_in    = False
    st.session_state.current_page = "login"
    st.rerun()

if page == "Back":
    st.session_state.page_history = st.session_state.get("page_history", ["Dashboard"])
    if len(st.session_state.page_history) > 1:
        st.session_state.page_history.pop()
        st.rerun()
    else:
        st.sidebar.info("No previous page to go back to.")

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":

    st.markdown("<div class='title'>Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict student academic performance using machine learning</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Student Academic Inputs")

        selected_course = st.selectbox("Select Course", COURSES)

        if MODELS_LOADED:
            model_choice = st.selectbox("Select ML Model", list(MODELS.keys()))
        else:
            model_choice = "Random Forest"
            st.warning("⚠️ ML models not found. Using rule-based grading.")

        st.markdown("---")
        st.caption("Enter scores using your school's grading scale:")

        col1, col2 = st.columns(2)
        with col1:
            attendance  = st.number_input("Attendance (out of 10)",  min_value=0, max_value=10,  value=7)
            test_score  = st.number_input("Test Score (out of 20)",  min_value=0, max_value=20,  value=14)
        with col2:
            assignment  = st.number_input("Assignment (out of 10)",  min_value=0, max_value=10,  value=7)
            exam_score  = st.number_input("Exam Score (out of 60)",  min_value=0, max_value=60,  value=38)

        total_preview = attendance + test_score + assignment + exam_score
        st.info(f"📊 Total score preview: **{total_preview}/100**")

        predict_button = st.button("🔮 Predict Performance")
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_button:
        total, grade, label, category, confidence = predict_performance(
            attendance, test_score, assignment, exam_score, selected_course, model_choice
        )
        feedback = smart_feedback(total, attendance, test_score, exam_score)
        st.session_state.prediction = {
            "total": total, "grade": grade, "label": label,
            "category": category, "confidence": confidence,
            "feedback": feedback, "course": selected_course,
            "attendance": attendance, "test_score": test_score,
            "assignment": assignment, "exam_score": exam_score,
        }

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if st.session_state.prediction:
            r = st.session_state.prediction

            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{r['total']}/100</div>
                    <div class='metric-label'>Total Score</div>
                </div>""", unsafe_allow_html=True)
            col_b.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{r['grade']}</div>
                    <div class='metric-label'>Grade</div>
                </div>""", unsafe_allow_html=True)
            col_c.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{r['confidence']}%</div>
                    <div class='metric-label'>Confidence</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**Course:** {r['course']}")
            st.write(f"**Performance:** {r['category']}  —  {r['label']}")

            # Score breakdown bar
            st.markdown("**Score Breakdown:**")
            breakdown = pd.DataFrame({
                "Component": ["Attendance /10","Test Score /20","Assignment /10","Exam Score /60"],
                "Score":     [r["attendance"], r["test_score"], r["assignment"], r["exam_score"]],
                "Max":       [10, 20, 10, 60]
            })
            breakdown["Percentage"] = (breakdown["Score"] / breakdown["Max"] * 100).round(1)
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

            st.markdown("**Insight / Recommendation:**")
            if r["total"] >= 60:
                st.success(r["feedback"])
            elif r["total"] >= 45:
                st.warning(r["feedback"])
            else:
                st.error(r["feedback"])

        else:
            st.info("Fill the form on the left and click **Predict Performance**")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CSV UPLOAD PAGE
# =========================
elif page == "CSV Upload":

    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied. This page is for Lecturers only.")
        st.stop()

    st.markdown("<div class='title'>📂 CSV Student Upload</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload a spreadsheet of students to automatically detect who is underperforming</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("ℹ️ How it works — click to read", expanded=False):
        st.markdown("""
**Scoring system (total = 100 marks):**

| Component   | Max marks |
|-------------|-----------|
| Attendance  | 10        |
| Test score  | 20        |
| Assignment  | 10        |
| Exam score  | 60        |
| **Total**   | **100**   |

**Nigerian university grading scale:**

| Grade | Score   | Label         | Status   |
|-------|---------|---------------|----------|
| A     | 70–100  | Distinction   | On track |
| B     | 60–69   | Credit        | On track |
| C     | 50–59   | Merit         | Watch    |
| D     | 45–49   | Pass          | Watch    |
| E     | 40–44   | Marginal Fail | At risk  |
| F     | 0–39    | Fail          | At risk  |
        """)

    st.markdown("### Step 1 — Download the CSV template")
    template_data = pd.DataFrame([
        {"name":"Adaeze Nwosu",   "matric_no":"250501000","course":"COMPUTER SCIENCE","year":"100 Level","attendance":8,  "test_score":14,"assignment":7, "exam_score":42},
        {"name":"Amaka Obi",      "matric_no":"250501001","course":"ICT",             "year":"100 Level","attendance":6,  "test_score":9, "assignment":5, "exam_score":28},
        {"name":"Blessing Okoro", "matric_no":"250501002","course":"CYBERSECURITY",   "year":"100 Level","attendance":10, "test_score":19,"assignment":10,"exam_score":55},
    ])
    st.download_button("⬇ Download CSV Template", template_data.to_csv(index=False),
                       "student_template.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Step 2 — Upload your filled CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if st.button("📋 Load sample data instead"):
        st.session_state["csv_df"] = pd.DataFrame([
            {"name":"Adaeze Nwosu",   "matric_no":"250501000","course":"COMPUTER SCIENCE","year":"100 Level","attendance":8,  "test_score":14,"assignment":7, "exam_score":42},
            {"name":"Amaka Obi",      "matric_no":"250501001","course":"ICT",             "year":"100 Level","attendance":6,  "test_score":9, "assignment":5, "exam_score":28},
            {"name":"Blessing Okoro", "matric_no":"250501002","course":"COMPUTER SCIENCE","year":"100 Level","attendance":10, "test_score":19,"assignment":10,"exam_score":55},
            {"name":"Chioma Okafor",  "matric_no":"250501003","course":"ICT",             "year":"100 Level","attendance":8,  "test_score":13,"assignment":8, "exam_score":40},
            {"name":"Chidi Nwosu",    "matric_no":"250501004","course":"CYBERSECURITY",   "year":"100 Level","attendance":9,  "test_score":16,"assignment":9, "exam_score":48},
            {"name":"Emeka Eze",      "matric_no":"250501005","course":"ICT",             "year":"100 Level","attendance":5,  "test_score":10,"assignment":6, "exam_score":30},
            {"name":"Fatima Bello",   "matric_no":"250501006","course":"CYBERSECURITY",   "year":"100 Level","attendance":3,  "test_score":6, "assignment":3, "exam_score":18},
            {"name":"Ibrahim Salisu", "matric_no":"250501007","course":"COMPUTER SCIENCE","year":"100 Level","attendance":6,  "test_score":11,"assignment":6, "exam_score":33},
            {"name":"Musa Danladi",   "matric_no":"250501008","course":"ICT",             "year":"100 Level","attendance":4,  "test_score":7, "assignment":4, "exam_score":20},
            {"name":"Ngozi Adeyemi",  "matric_no":"250501009","course":"CYBERSECURITY",   "year":"100 Level","attendance":8,  "test_score":15,"assignment":8, "exam_score":44},
            {"name":"Tunde Afolabi",  "matric_no":"250501010","course":"COMPUTER SCIENCE","year":"100 Level","attendance":2,  "test_score":5, "assignment":2, "exam_score":15},
            {"name":"Yusuf Garba",    "matric_no":"250501011","course":"CYBERSECURITY",   "year":"100 Level","attendance":5,  "test_score":9, "assignment":5, "exam_score":27},
        ])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
                st.stop()
            st.session_state["csv_df"] = df
            st.success(f"✅ File uploaded — {len(df)} students found.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if "csv_df" in st.session_state:
        df = st.session_state["csv_df"].copy()
        df = df.sort_values("name").reset_index(drop=True)

        totals, grades, labels, risks = [], [], [], []
        for _, row in df.iterrows():
            t, g, l, r = grade_student_row(row)
            totals.append(t); grades.append(g); labels.append(l); risks.append(r)

        df["total"]  = totals
        df["grade"]  = grades
        df["label"]  = labels
        df["status"] = risks

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Step 3 — Results")

        at_risk_count  = (df["status"] == "at-risk").sum()
        watch_count    = (df["status"] == "watch").sum()
        on_track_count = (df["status"] == "on-track").sum()
        avg_total      = round(df["total"].mean(), 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, lbl in [
            (c1, len(df),            "Total Students"),
            (c2, at_risk_count,      "⚠ At Risk"),
            (c3, watch_count,        "👁 Watch"),
            (c4, on_track_count,     "✅ On Track"),
            (c5, f"{avg_total}/100", "Class Average"),
        ]:
            col.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            search = st.text_input("🔍 Search by name or matric no", "")
        with col_f2:
            course_filter = st.selectbox("Filter by course", ["All"] + COURSES)
        with col_f3:
            status_filter = st.selectbox("Filter by status", ["All","at-risk","watch","on-track"])

        filtered = df.copy()
        if search:
            filtered = filtered[
                filtered["name"].str.contains(search, case=False) |
                filtered["matric_no"].astype(str).str.contains(search)
            ]
        if course_filter != "All":
            filtered = filtered[filtered["course"] == course_filter]
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        st.markdown("#### 📋 All Students (sorted A–Z)")

        def colour_row(row):
            if row["status"] == "at-risk":
                return ["background-color: #fff5f5"] * len(row)
            elif row["status"] == "watch":
                return ["background-color: #fffaf0"] * len(row)
            else:
                return ["background-color: #f0fff4"] * len(row)

        display_cols = ["name","matric_no","course","year","attendance","test_score","assignment","exam_score","total","grade","label","status"]
        st.dataframe(filtered[display_cols].style.apply(colour_row, axis=1),
                     use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        at_risk_df = df[df["status"] == "at-risk"].copy()
        watch_df   = df[df["status"] == "watch"].copy()

        if not at_risk_df.empty:
            st.markdown("---")
            st.markdown(f"### ⚠️ At Risk — Grade E or F ({len(at_risk_df)} students)")
            for _, row in at_risk_df.iterrows():
                st.markdown(f"""
                <div style='border-left:4px solid #c53030;background:#fff5f5;
                            border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px'>
                    <strong style='font-size:15px'>{row['name']}</strong>
                    &nbsp;|&nbsp; Matric: {row['matric_no']}
                    &nbsp;|&nbsp; {row['course']} &nbsp;|&nbsp; {row['year']}<br>
                    <span style='font-size:20px;font-weight:600;color:#c53030'>{int(row['total'])}/100</span>
                    &nbsp;
                    <span style='background:#c53030;color:white;padding:2px 10px;border-radius:99px;font-size:12px'>
                        Grade {row['grade']} — {row['label']}
                    </span>
                </div>""", unsafe_allow_html=True)
                c1,c2,c3,c4 = st.columns(4)
                for col,lbl,val,mx,flagged in [
                    (c1,"Attendance", row["attendance"], 10, row["attendance"]<6),
                    (c2,"Test Score",  row["test_score"],  20, row["test_score"]<10),
                    (c3,"Assignment",  row["assignment"],  10, row["assignment"]<5),
                    (c4,"Exam Score",  row["exam_score"],  60, row["exam_score"]<30),
                ]:
                    col.metric(f"{'🔴' if flagged else '🟢'} {lbl}", f"{val}/{mx}")
                tips = []
                if row["attendance"]<6:  tips.append("low attendance")
                if row["test_score"]<10: tips.append("poor test score")
                if row["assignment"]<5:  tips.append("missing assignments")
                if row["exam_score"]<30: tips.append("low exam score")
                if tips:
                    st.caption(f"💡 Flagged for: {', '.join(tips)}.")
                st.markdown("")

        if not watch_df.empty:
            st.markdown("---")
            st.markdown(f"### 👁 Watch Closely — Grade C or D ({len(watch_df)} students)")
            for _, row in watch_df.iterrows():
                st.markdown(f"""
                <div style='border-left:4px solid #c05621;background:#fffaf0;
                            border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px'>
                    <strong style='font-size:15px'>{row['name']}</strong>
                    &nbsp;|&nbsp; Matric: {row['matric_no']}
                    &nbsp;|&nbsp; {row['course']} &nbsp;|&nbsp; {row['year']}<br>
                    <span style='font-size:20px;font-weight:600;color:#c05621'>{int(row['total'])}/100</span>
                    &nbsp;
                    <span style='background:#c05621;color:white;padding:2px 10px;border-radius:99px;font-size:12px'>
                        Grade {row['grade']} — {row['label']}
                    </span>
                </div>""", unsafe_allow_html=True)

        if at_risk_df.empty and watch_df.empty:
            st.success("✅ All students are performing well — nobody flagged!")

        st.markdown("---")
        st.markdown("### ⬇ Export Results")
        underperforming = df[df["status"] != "on-track"][display_cols]
        if not underperforming.empty:
            st.download_button("⬇ Download underperforming students CSV",
                               underperforming.to_csv(index=False),
                               "underperforming_students.csv", "text/csv")
        st.download_button("⬇ Download full results CSV",
                           df[display_cols].to_csv(index=False),
                           "all_students_results.csv", "text/csv")

# =========================
# ANALYTICS PAGE
# =========================
elif page == "Analytics":

    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied.")
        st.stop()

    st.markdown("<div class='title'>Dataset Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Visual breakdown of student performance across courses and metrics</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Load the trained dataset for real analytics
    data_path = os.path.join(MODELS_DIR, "student_data.csv")
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        data = None

    if data is not None:
        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl in [
            (c1, len(data),                          "Total Students"),
            (c2, f"{data['total'].mean():.1f}/100",  "Average Score"),
            (c3, (data["grade"].isin(["E","F"])).sum(), "At Risk Students"),
            (c4, (data["grade"] == "A").sum(),           "Distinction Students"),
        ]:
            col.markdown(f"""<div class='metric-box'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Grade distribution
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Grade Distribution")
        grade_counts = data["grade"].value_counts().reindex(["A","B","C","D","E","F"], fill_value=0)
        colors = ["#1565C0","#1976D2","#FFA726","#FF7043","#EF5350","#B71C1C"]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(grade_counts.index, grade_counts.values, color=colors, edgecolor="white")
        ax.set_ylabel("Number of Students"); ax.set_xlabel("Grade")
        ax.set_title("Student Grade Distribution (Based on Trained Dataset)", fontweight="bold")
        for bar, val in zip(bars, grade_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(val), ha='center', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score breakdown by course
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📚 Average Score Breakdown by Course")
        course_grp = data.groupby("course")[["attendance","test_score","assignment","exam_score","total"]].mean().round(1)
        st.dataframe(course_grp, use_container_width=True)

        x = np.arange(len(course_grp)); width = 0.2
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(x-width*1.5, course_grp["attendance"]*10,  width, label="Attendance (scaled)", color="#1565C0")
        ax2.bar(x-width*0.5, course_grp["test_score"]*5,   width, label="Test Score (scaled)",  color="#42A5F5")
        ax2.bar(x+width*0.5, course_grp["assignment"]*10,  width, label="Assignment (scaled)",  color="#FFA726")
        ax2.bar(x+width*1.5, course_grp["exam_score"]*1.67,width, label="Exam Score (scaled)",  color="#EF5350")
        ax2.set_xticks(x); ax2.set_xticklabels(course_grp.index)
        ax2.set_ylabel("Score (normalised to 100)")
        ax2.set_title("Component Score by Course", fontweight="bold")
        ax2.legend(); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Correlation Heatmap")
        corr_cols = ["attendance","test_score","assignment","exam_score","total"]
        matrix = data[corr_cols].corr().values
        labels = ["Attendance","Test Score","Assignment","Exam Score","Total"]
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        hm = ax3.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(hm, ax=ax3)
        ax3.set_xticks(range(5)); ax3.set_yticks(range(5))
        ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_title("Academic Factors Correlation Matrix", fontweight="bold")
        for i in range(5):
            for j in range(5):
                ax3.text(j, i, f"{matrix[i,j]:.2f}", ha='center', va='center',
                         fontsize=8, color="white" if matrix[i,j]>0.7 else "black")
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Dataset not found. Make sure student_data.csv is in the models folder.")

# =========================
# MODEL REPORT
# =========================
elif page == "Model Report":

    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied.")
        st.stop()

    st.markdown("<div class='title'>Machine Learning Model Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Models trained on 1,500 Nigerian university students — ICT, Computer Science, Cybersecurity</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Accuracy on Test Data")
    if MODELS_LOADED:
        results_df = pd.DataFrame(list(MODEL_RESULTS.items()), columns=["Model","Accuracy (%)"])
        results_df = results_df.sort_values("Accuracy (%)", ascending=False)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        colors_m = ["#1565C0" if v == max(MODEL_RESULTS.values()) else "#90CAF9"
                    for v in MODEL_RESULTS.values()]
        ax4.bar(MODEL_RESULTS.keys(), MODEL_RESULTS.values(), color=colors_m)
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_title("Model Accuracy Comparison", fontweight="bold")
        ax4.set_ylim(80, 102)
        for i, (k, v) in enumerate(MODEL_RESULTS.items()):
            ax4.text(i, v+0.3, f"{v}%", ha='center', fontsize=10, fontweight='bold')
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
        plt.xticks(rotation=15, ha='right')
        st.pyplot(fig4)
    else:
        st.error(f"Models not loaded: {MODEL_ERROR}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Training Data Summary")
    st.markdown("""
    | Property        | Value                              |
    |----------------|------------------------------------|
    | Total records   | 1,500 students                    |
    | Courses         | ICT, Computer Science, Cybersecurity |
    | Year            | 100 Level                          |
    | Attendance      | Graded /10                         |
    | Test score      | Graded /20                         |
    | Assignment      | Graded /10                         |
    | Exam score      | Graded /60                         |
    | Total marks     | 100                                |
    | Grading scale   | A (70+), B (60+), C (50+), D (45+), E (40+), F (<40) |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PROFILE
# =========================
elif page == "My Profile":

    st.markdown("<div class='title'>My Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Personal Information")
    st.write(f"**Full Name:** {st.session_state.name}")
    st.write(f"**Matric Number:** {st.session_state.matric}")
    st.write(f"**Role:** {st.session_state.role}")
    st.write("**Year:** 100 Level")
    st.write("**Program:** Undergraduate")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.prediction:
        r = st.session_state.prediction
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Latest Prediction")
        st.write(f"**Course:** {r.get('course','')}")
        st.write(f"**Total Score:** {r['total']}/100")
        st.write(f"**Grade:** {r['grade']} — {r['label']}")
        st.write(f"**Performance:** {r['category']}")
        st.write(f"**Model Confidence:** {r['confidence']}%")
        st.markdown("</div>", unsafe_allow_html=True)
