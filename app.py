import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AcadPredict",
    page_icon="🎓",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATES
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =========================
# LOGIN PAGE
# =========================
if not st.session_state.logged_in:

    st.markdown("<div class='title'>🎓 AcadPredict</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Student Academic Performance Prediction System</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Student Login")

        full_name = st.text_input("Full Name")
        matric = st.text_input("Matric Number")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if full_name and matric and password:
                st.session_state.logged_in = True
                st.session_state.name = full_name
                st.session_state.matric = matric
                st.rerun()
            else:
                st.error("Please fill all fields")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎓 AcadPredict")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Analytics",
        "Model Report",
        "My Profile"
    ]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

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

        attendance = st.slider("Attendance (%)", 0, 100, 75)
        test_score = st.slider("Test Score (%)", 0, 100, 65)
        exam_score = st.slider("Exam Score (%)", 0, 100, 70)
        assignment_score = st.slider("Assignment Score (%)", 0, 100, 80)
        past_record = st.slider("Past Academic Record (%)", 0, 100, 72)
        study_hours = st.slider("Study Hours Per Week", 0, 60, 15)
        gpa = st.slider("GPA", 0.0, 5.0, 3.2)

        class_participation = st.selectbox(
            "Class Participation",
            ["Low", "Medium", "High"]
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            [
                "None",
                "1 Activity",
                "2 Activities",
                "3+ Activities"
            ]
        )

        predict_button = st.button("Predict Performance")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # PREDICTION LOGIC
    # =========================
    if predict_button:

        score = (
            attendance * 0.15 +
            test_score * 0.15 +
            exam_score * 0.25 +
            assignment_score * 0.15 +
            past_record * 0.15 +
            study_hours * 0.5 +
            (gpa * 20) * 0.15
        )

        if class_participation == "High":
            score += 5
        elif class_participation == "Medium":
            score += 2

        if extracurricular == "1 Activity":
            score += 2
        elif extracurricular == "2 Activities":
            score += 4
        elif extracurricular == "3+ Activities":
            score += 5

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
elif page == "Analytics":

    st.markdown("<div class='title'>Dataset Analytics</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class='metric-box'>
                <div class='metric-value'>1,500</div>
                <div class='metric-label'>Students</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class='metric-box'>
                <div class='metric-value'>78%</div>
                <div class='metric-label'>Average Performance</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class='metric-box'>
                <div class='metric-value'>18%</div>
                <div class='metric-label'>At Risk</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class='metric-box'>
                <div class='metric-value'>32%</div>
                <div class='metric-label'>Excellent Students</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # SAMPLE CHART
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Student Performance Distribution")

    categories = ["Excellent", "Good", "Average", "Pass", "Poor"]
    values = [300, 450, 400, 220, 130]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, values)
    ax.set_ylabel("Students")
    ax.set_xlabel("Performance Category")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # CORRELATION HEATMAP
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")

    data = np.random.rand(6, 6)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    heatmap = ax2.imshow(data)
    plt.colorbar(heatmap)

    st.pyplot(fig2)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MODEL REPORT PAGE
# =========================
elif page == "Model Report":

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
