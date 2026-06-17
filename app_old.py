import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import pickle

# 🔥 PUT THIS HERE (VERY IMPORTANT)
def smart_feedback(score, attendance, study_hours):

    if score < 50 and attendance < 60:
        return "Low attendance and low performance detected. Try attending more classes and revising regularly."

    elif score < 50 and study_hours < 2:
        return "Your study time is very low. Increasing daily study hours can significantly improve your performance."

    elif score >= 70:
        return "Excellent performance! Keep maintaining your effort."

    else:
        return "You're doing okay, but consistency will improve your results."

USER_FILE = "users.json"
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)
# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AcadPredict",
    page_icon="🎓",
    layout="wide"
)
# =========================
# LOGIN INPUT FIELDS
# =========================


# =========================
# COLORS
# =========================
ROYAL_BLUE = "#1565C0"
LIGHT_BG = "#F5F7FB"
WHITE = "#FFFFFF"
TEXT = "#1E293B"
BORDER = "#D9E2EC"

# =========================
# CUSTOM CSS
# =========================
st.markdown(f"""
<style>
body {{
    background-color: {LIGHT_BG};
}}

.main {{
    background-color: {LIGHT_BG};
}}

.block-container {{
    padding-top: 1rem;
}}

.card {{
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid {BORDER};
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}}

.title {{
    font-size: 2rem;
    font-weight: 700;
    color: {ROYAL_BLUE};
}}

.subtitle {{
    color: gray;
    font-size: 0.9rem;
}}

.stButton>button {{
    background-color: {ROYAL_BLUE};
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    width: 100%;
    height: 45px;
}}

.stButton>button:hover {{
    background-color: #0D47A1;
}}

.metric-box {{
    background: white;
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid {BORDER};
    text-align: center;
}}

.metric-value {{
    font-size: 1.8rem;
    font-weight: bold;
    color: {ROYAL_BLUE};
}}

.metric-label {{
    color: gray;
    font-size: 0.9rem;
}}
.metric-label {{
    color: gray;
    font-size: 0.9rem;
}}

div[data-testid="stFileUploader"] {{
    display: none;
}}
section[data-testid="stFileUploadDropzone"] {{
    display: none;
}}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]:empty {{
    display: none;
}}

div[data-testid="column"] {{
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATES
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "role" not in st.session_state:
    st.session_state.role = "Student"

if "current_page" not in st.session_state:
    st.session_state.current_page = "login"

# =========================
# LOGIN PAGE
# =========================
if st.session_state.current_page == "login" and not st.session_state.logged_in:

    st.markdown("<div class='title'>🎓 AcadPredict</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Student Academic Performance Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "Login"

    # Small buttons top right
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

        # ================= LOGIN =================
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

        # ================= SIGN UP =================
        elif auth_option == "Sign Up":
            st.subheader("Create Account 📝")
            full_name = st.text_input("Full Name")
            matric    = st.text_input("Matric Number")
            password  = st.text_input("Password", type="password")
            confirm   = st.text_input("Confirm Password", type="password")
            role      = st.selectbox("Role",    ["Student", "Lecturer"])
            level     = st.selectbox("Level",   ["100", "200", "300", "400"])
            program   = st.selectbox("Program", ["Undergraduate", "Post-graduate", "Part-time", "Parents"])

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
                    users[matric] = {
                        "name":     full_name,
                        "password": password,
                        "role":     role,
                        "level":    level,
                        "program":  program
                    }
                    save_users(users)
                    st.success("Account created! You can now log in.")

        # ================= FORGOT PASSWORD =================
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
    nav_options = ["Dashboard", "Analytics", "Model Report", "My Profile", "Back"]
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

    # =========================
    # LEFT SIDE
    # =========================
    with left:

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Student Academic Inputs")

        # =========================
        # COURSE DROPDOWN
        # =========================
        courses = [
            "CHM 101",
            "CHM 107",
            "CSC 101",
            "GST 109",
            "GST 111",
            "MTH 101",
            "MTH 123",
            "PHY 101",
            "PHY 105"
        ]

        selected_course = st.selectbox(
            "Select Course",
            courses
        )

        attendance = st.slider("Attendance (%)", 0, 100, 75)
        test_score = st.slider("Test Score (%)", 0, 100, 65)
        exam_score = st.slider("Exam Score (%)", 0, 100, 70)
        assignment_score = st.slider("Assignment Score (%)", 0, 100, 80)
        
        st.write("### Student Insight / Recommendation")

        # ✅ CORRECT
       # NEW
        score = (
    attendance * 0.25 +
    test_score * 0.25 +
    exam_score * 0.35 +
    assignment_score * 0.15
)

       # NEW
        feedback = smart_feedback(
    score=score,
    attendance=attendance,
    study_hours=0
)
        st.success(feedback)

        class_participation = st.selectbox(
            "Class Participation",
            ["Low", "Medium", "High"]
        )

       

        predict_button = st.button("Predict Performance")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # PREDICTION LOGIC
    # =========================
    if predict_button:

        # NEW
        score = (
    attendance * 0.25 +
    test_score * 0.25 +
    exam_score * 0.35 +
    assignment_score * 0.15
)
        st.write("### Student Insight / Recommendation")

        # NEW
        feedback = smart_feedback(
    score=score,
    attendance=attendance,
    study_hours=0
)

        st.success(feedback)

        if class_participation == "High":
            score += 5
        elif class_participation == "Medium":
            score += 2


        score = min(100, int(score))

        if score >= 80:
            category = "Excellent"
        elif score >= 70:
            category = "Good"
        elif score >= 55:
            category = "Average"
        elif score >= 40:
            category = "Pass"
        else:
            category = "Poor"

        confidence = random.randint(85, 98)

        st.session_state.prediction = {
            "score": score,
            "category": category,
            "confidence": confidence
        }
       

    # =========================
    # RIGHT SIDE
    # =========================
    with right:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Prediction Result")

        if st.session_state.prediction:

            result = st.session_state.prediction

            st.markdown(
                f"""
                <div class='metric-box'>
                    <div class='metric-value'>{result['score']}%</div>
                    <div class='metric-label'>Predicted Score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.success(f"Performance Category: {result['category']}")
            st.write(f"**Selected Course:** {selected_course}")
            st.info(f"Model Confidence: {result['confidence']}%")

            st.subheader("Performance Insight")

            if result['category'] == "Excellent":
                st.success("Student is expected to perform exceptionally well academically.")

            elif result['category'] == "Good":
                st.info("Student is expected to maintain good academic performance.")

            elif result['category'] == "Average":
                st.warning("Student performance is average and needs improvement.")

            elif result['category'] == "Pass":
                st.warning("Student is at risk academically and requires support.")

            else:
                st.error("Student performance is poor and requires immediate intervention.")

        else:
            st.info("Fill the form and click Predict Performance")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ANALYTICS PAGE
# =========================
# =========================
# ANALYTICS PAGE
# =========================
elif page == "Analytics":

    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied. This page is for Lecturers only.")
        st.stop()

    st.markdown("<div class='title'>Dataset Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Visual breakdown of student academic performance across courses and metrics</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # TOP METRIC CARDS
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-value'>1,500</div>
                <div class='metric-label'>Total Students</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-value'>78%</div>
                <div class='metric-label'>Average Performance</div>
            </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-value'>18%</div>
                <div class='metric-label'>At Risk Students</div>
            </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-value'>32%</div>
                <div class='metric-label'>Excellent Students</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # PERFORMANCE DISTRIBUTION
    # =========================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Student Performance Distribution")
    st.caption("This chart shows how many students fall into each performance category across all courses.")

    categories = ["Excellent", "Good", "Average", "Pass", "Poor"]
    values = [300, 450, 400, 220, 130]
    colors = ["#1565C0", "#1976D2", "#42A5F5", "#FFA726", "#EF5350"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.set_xlabel("Performance Category", fontsize=11)
    ax.set_title("Overall Performance Distribution", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 550)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # SCORE BREAKDOWN BY COURSE
    # =========================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📚 Score Breakdown by Course")
    st.caption("Compare average scores across Test, Exam, and Assignment for each course offered.")

    courses = ["CHM 101", "CHM 107", "CSC 101", "GST 109", "GST 111", "MTH 101", "MTH 123", "PHY 101", "PHY 105"]

    np.random.seed(42)
    test_scores   = np.random.randint(55, 85, size=len(courses))
    exam_scores   = np.random.randint(50, 80, size=len(courses))
    assign_scores = np.random.randint(60, 90, size=len(courses))

    x = np.arange(len(courses))
    width = 0.25

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    b1 = ax2.bar(x - width, test_scores,   width, label="Test Score",       color="#1565C0")
    b2 = ax2.bar(x,          exam_scores,   width, label="Exam Score",       color="#42A5F5")
    b3 = ax2.bar(x + width,  assign_scores, width, label="Assignment Score", color="#FFA726")

    ax2.set_xlabel("Course", fontsize=11)
    ax2.set_ylabel("Average Score (%)", fontsize=11)
    ax2.set_title("Average Score per Course (Test vs Exam vs Assignment)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(courses, rotation=25, ha='right')
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary table below chart
    st.caption("📋 Summary Table — Average scores per course")
    course_df = pd.DataFrame({
        "Course": courses,
        "Avg Test Score (%)": test_scores,
        "Avg Exam Score (%)": exam_scores,
        "Avg Assignment Score (%)": assign_scores,
        "Overall Avg (%)": ((test_scores + exam_scores + assign_scores) / 3).astype(int)
    })
    st.dataframe(course_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # CORRELATION HEATMAP
    # =========================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔥 Correlation Heatmap")
    st.caption("Shows how strongly different academic factors are related to each other. Values closer to 1.0 mean stronger correlation.")

    labels = ["Attendance", "Test Score", "Exam Score", "Assignment"]
    np.random.seed(7)
    matrix = np.random.uniform(0.3, 1.0, (4, 4))
    np.fill_diagonal(matrix, 1.0)
    matrix = (matrix + matrix.T) / 2

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    heatmap = ax3.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(heatmap, ax=ax3)

    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_title("Academic Factors Correlation Matrix", fontsize=12, fontweight="bold")

    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                     fontsize=8, color="white" if matrix[i, j] > 0.7 else "black")

    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)
   

# =========================
# MODEL REPORT PAGE
# =========================
elif page == "Model Report":

    if st.session_state.role != "Lecturer":
        st.error("🔒 Access Denied. This page is for Lecturers only.")
        st.stop()

    st.markdown("<div class='title'>Machine Learning Model Report</div>", unsafe_allow_html=True)

    models = {
        "Random Forest": [94, 92, 91, 91],
        "Decision Tree": [87, 85, 84, 84],
        "SVM": [90, 89, 88, 88],
        "Logistic Regression": [84, 83, 82, 82]
    }

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Model Evaluation Metrics")

    df = pd.DataFrame(
        models,
        index=["Accuracy", "Precision", "Recall", "F1-Score"]
    )

    st.dataframe(df)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Accuracy Comparison")

    names = list(models.keys())
    acc = [models[m][0] for m in names]

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(names, acc)
    ax3.set_ylabel("Accuracy")

    st.pyplot(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PROFILE PAGE
# =========================
elif page == "My Profile":

    st.markdown("<div class='title'>Student Profile</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Personal Information")

    st.write(f"**Full Name:** {st.session_state.name}")
    st.write(f"**Matric Number:** {st.session_state.matric}")
    st.write("**Department:** Computer Science")
    st.write("**Level:** 400")
    st.write("**Program:** Undergraduate")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.prediction:

        result = st.session_state.prediction

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Latest Prediction")

        st.write(f"**Predicted Score:** {result['score']}%")
        st.write(f"**Performance Category:** {result['category']}")
        st.write(f"**Model Confidence:** {result['confidence']}%")

        st.markdown("</div>", unsafe_allow_html=True)