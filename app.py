"""
Heart Disease Prediction Dashboard
===================================
A professional, dynamic Streamlit application for heart disease risk prediction
with Explainable AI (SHAP) integration.

Prerequisites:
    Run `models (1).ipynb` first to generate the ./models/ directory with:
    - Trained model pickles (random_forest.pkl, xgboost.pkl, etc.)
    - preprocessor.pkl, feature_names.pkl
    - metrics.json, roc_data.json
    - SHAP artifacts (*_shap.pkl)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease AI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced custom styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e63946 0%, #f77f00 50%, #fcbf49 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    .risk-card {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .risk-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .risk-low     { background: linear-gradient(135deg, #2a9d8f 0%, #52b788 100%); }
    .risk-mild    { background: linear-gradient(135deg, #8ac926 0%, #cddc39 100%); color:#222;}
    .risk-mod     { background: linear-gradient(135deg, #f4a261 0%, #e76f51 100%); }
    .risk-high    { background: linear-gradient(135deg, #e63946 0%, #9d0208 100%); }
    .metric-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 6px solid #e63946;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #e63946;
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(230, 57, 70, 0.15);
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    .stat-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
    }
    .stat-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.3);
    }
    .stat-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 12px; 
        background: #f8f9fa;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(230, 57, 70, 0.1);
    }
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e63946;
        display: inline-block;
    }
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-left: 5px solid #00bcd4;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Resource loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
    "KNN",
    "SVM",
]

@st.cache_resource
def load_model(name: str):
    safe = name.lower().replace(" ", "_")
    path = f"models/{safe}.pkl"
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Please run `models (1).ipynb` first.")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")

@st.cache_resource
def load_feature_names():
    return joblib.load("models/feature_names.pkl")

@st.cache_resource
def load_metrics():
    with open("models/metrics.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_roc_data():
    with open("models/roc_data.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_shap_data(name: str):
    safe = name.lower().replace(" ", "_")
    return joblib.load(f"models/{safe}_shap.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Risk recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
def risk_recommendation(prob: float):
    """Return (label, css_class, emoji, advice) based on probability ranges."""
    if prob <= 0.30:
        return (
            "Low Risk",
            "risk-low",
            "🟢",
            [
                "Your predicted heart disease risk is **low**.",
                "Maintain a balanced diet, regular exercise, and healthy sleep habits.",
                "Routine annual check-ups are sufficient.",
                "Continue monitoring blood pressure and cholesterol yearly.",
            ],
        )
    elif prob <= 0.50:
        return (
            "Mild Risk",
            "risk-mild",
            "🟡",
            [
                "Your predicted heart disease risk is **mild**.",
                "Adopt heart-healthy lifestyle changes (reduce sodium, saturated fats).",
                "Increase aerobic activity to ≥150 minutes/week.",
                "Schedule a follow-up with your primary care physician within 3–6 months.",
            ],
        )
    elif prob <= 0.70:
        return (
            "Moderate Risk",
            "risk-mod",
            "🟠",
            [
                "Your predicted heart disease risk is **moderate**.",
                "Consult a cardiologist for further evaluation.",
                "Consider diagnostic tests: ECG, stress test, lipid panel.",
                "Review medications with your doctor; manage stress actively.",
            ],
        )
    else:
        return (
            "High Risk",
            "risk-high",
            "🔴",
            [
                "Your predicted heart disease risk is **high**.",
                "**Seek immediate medical attention.**",
                "Urgent cardiology consultation is strongly recommended.",
                "Comprehensive cardiac work-up (angiography, advanced imaging) may be required.",
                "Do not ignore symptoms such as chest pain, shortness of breath, or dizziness.",
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ❤️ Heart Disease AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "🧭 Navigation",
    [
        "🏠 Home",
        "🔮 Predict Risk",
        "📊 Model Performance",
        "📈 ROC Curves & AUC",
        "🎯 Feature Importance",
        "🧠 Explainability (XAI)",
        "ℹ️ About",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("📅 June 18, 2026 · v1.0")
st.sidebar.caption("Built with Streamlit + SHAP")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-header">❤️ Heart Disease Prediction Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered risk assessment with explainable predictions</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="stat-card">'
            '<div style="font-size:2.5rem">🤖</div>'
            '<div style="font-size:2rem;font-weight:800">5</div>'
            '<div style="font-size:0.9rem;opacity:0.9">ML Models</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="stat-card green">'
            '<div style="font-size:2.5rem">🧬</div>'
            '<div style="font-size:2rem;font-weight:800">13</div>'
            '<div style="font-size:0.9rem;opacity:0.9">Clinical Features</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="stat-card orange">'
            '<div style="font-size:2.5rem">🧠</div>'
            '<div style="font-size:2rem;font-weight:800">SHAP</div>'
            '<div style="font-size:0.9rem;opacity:0.9">Explainable AI</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            '<div class="stat-card blue">'
            '<div style="font-size:2.5rem">📊</div>'
            '<div style="font-size:2rem;font-weight:800">303</div>'
            '<div style="font-size:0.9rem;opacity:0.9">Patient Records</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("### 🌟 What you can do here")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            '<div class="feature-card">'
            '<h3 style="color:#e63946;margin-top:0">🔮 Instant Predictions</h3>'
            '<p>Enter patient vitals and get an instant heart disease risk prediction with tailored medical recommendations based on probability thresholds.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            '<div class="feature-card">'
            '<h3 style="color:#e63946;margin-top:0">🧠 Explainable AI</h3>'
            '<p>Use SHAP values to understand <em>why</em> each model made its prediction — feature-level interpretability for transparency and trust.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div class="feature-card">'
            '<h3 style="color:#e63946;margin-top:0">📊 Model Comparison</h3>'
            '<p>Compare accuracy, precision, recall, F1, and ROC-AUC across 5 state-of-the-art classifiers to choose the best model for your needs.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            '<div class="feature-card">'
            '<h3 style="color:#e63946;margin-top:0">📈 Advanced Visualizations</h3>'
            '<p>Explore ROC curves, feature importance rankings, SHAP beeswarm plots, and interactive probability distributions.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate. Start with **🔮 Predict Risk** to try a live prediction.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Predict Risk
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Risk":
    st.markdown('<p class="section-header">🔮 Patient Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown("Select a model and enter patient information to generate a personalized risk assessment.")

    model_choice = st.selectbox("🤖 Choose a prediction model", MODEL_NAMES, index=0)
    model = load_model(model_choice)
    feature_names = load_feature_names()

    st.markdown("---")
    st.markdown("### 📝 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", 20, 100, 55, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["typical", "nontypical", "nonanginal", "asymptomatic"],
        )
        rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130, step=1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 230, step=1)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (≤120)", "Yes (>120)"])
        rest_ecg = st.selectbox(
            "Resting ECG Results",
            ["0 – Normal", "1 – ST-T wave abnormality", "2 – Left ventricular hypertrophy"],
        )
        max_hr = st.number_input("Max Heart Rate Achieved", 60, 220, 150, step=1)
        ex_ang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 7.0, 1.0, step=0.1, format="%.1f")

    with col3:
        slope = st.selectbox("Slope of ST Segment", ["1 – Upsloping", "2 – Flat", "3 – Downsloping"])
        ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["normal", "fixed", "reversable"])

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "ChestPain": chest_pain,
        "RestBP": rest_bp,
        "Chol": chol,
        "Fbs": 1 if "Yes" in fbs else 0,
        "RestECG": int(rest_ecg.split("–")[0].strip()),
        "MaxHR": max_hr,
        "ExAng": 1 if ex_ang == "Yes" else 0,
        "Oldpeak": oldpeak,
        "Slope": int(slope.split("–")[0].strip()),
        "Ca": ca,
        "Thal": thal,
    }])

    st.markdown("---")
    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            prob = model.predict_proba(input_df)[0][1]
            pred_class = 1 if prob >= 0.5 else 0
            label, css, emoji, advice = risk_recommendation(prob)

        st.markdown("### 🩺 Prediction Result")
        c1, c2 = st.columns([1.2, 1.8])

        with c1:
            st.markdown(
                f'<div class="risk-card {css}">'
                f'<div style="font-size:3rem;position:relative;z-index:1">{emoji}</div>'
                f'<div style="font-size:1.8rem;font-weight:800;position:relative;z-index:1">{label}</div>'
                f'<div style="font-size:2.8rem;font-weight:900;position:relative;z-index:1">{prob*100:.1f}%</div>'
                f'<div style="font-size:1rem;opacity:0.95;position:relative;z-index:1">Probability of Heart Disease</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-box"><b>Model:</b> {model_choice}</div>'
                f'<div class="metric-box"><b>Predicted Class:</b> {"Disease Present" if pred_class else "No Disease"}</div>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown("#### 📋 Personalized Recommendations")
            for tip in advice:
                st.markdown(f"- {tip}")

            st.markdown("#### 📊 Risk Probability Gauge")
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "#52b788"},
                            {"range": [30, 50], "color": "#cddc39"},
                            {"range": [50, 70], "color": "#f4a261"},
                            {"range": [70, 100], "color": "#e63946"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧠 Why did the model predict this? (SHAP Explanation)")
        try:
            shap_data = load_shap_data(model_choice)
            X_test_sample = shap_data["X_test_sample"]
            sv_pos = shap_data["shap_values_positive"]
            ev = shap_data["expected_value"]
            is_tree = shap_data["is_tree_based"]

            preproc = load_preprocessor()
            X_new_trans = preproc.transform(input_df)

            if is_tree:
                trained_model = model.named_steps["model"]
                explainer = shap.TreeExplainer(trained_model)
                sv_new = explainer.shap_values(X_new_trans)
                if isinstance(sv_new, list):
                    sv_new = sv_new[1]
            else:
                trained_model = model.named_steps["model"]
                background = shap_data.get("background")
                if background is None:
                    background = shap.sample(
                        preproc.transform(X_test_sample.head(80) if isinstance(X_test_sample, pd.DataFrame) else X_test_sample),
                        50, random_state=42,
                    )
                explainer = shap.KernelExplainer(trained_model.predict_proba, background)
                sv_new = explainer.shap_values(X_new_trans, nsamples=100)
                if isinstance(sv_new, list):
                    sv_new = sv_new[1]

            explanation = shap.Explanation(
                values=sv_new[0],
                base_values=ev,
                data=X_new_trans[0],
                feature_names=feature_names,
            )

            fig_waterfall = shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig_waterfall, bbox_inches="tight")

            st.markdown("#### 🔝 Top 5 Contributing Features")
            top_idx = np.argsort(np.abs(sv_new[0]))[::-1][:5]
            top_df = pd.DataFrame({
                "Feature": [feature_names[i] for i in top_idx],
                "SHAP Value": [sv_new[0][i] for i in top_idx],
                "Patient Value": [X_new_trans[0][i] for i in top_idx],
            })
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<p class="section-header">📊 Model Performance Comparison</p>', unsafe_allow_html=True)
    metrics = load_metrics()

    df_metrics = pd.DataFrame(metrics).T
    df_metrics = df_metrics.drop(columns=["confusion_matrix"], errors="ignore")
    df_metrics.index.name = "Model"
    df_metrics = df_metrics.reset_index()

    st.markdown("#### 📋 Metrics Table")
    st.dataframe(
        df_metrics.style
        .background_gradient(cmap="RdYlGn", subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
        .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 📊 Metric Comparison")
    metric_sel = st.multiselect(
        "Select metrics to visualize",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        default=["Accuracy", "F1 Score", "ROC-AUC"],
    )
    if metric_sel:
        fig = px.bar(
            df_metrics,
            x="Model",
            y=metric_sel,
            barmode="group",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=450, legend_title_text="Metric")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🟦 Confusion Matrices")
    cols = st.columns(len(metrics))
    for idx, (name, m) in enumerate(metrics.items()):
        with cols[idx]:
            st.caption(f"**{name}**")
            cm = np.array(m["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(3.2, 2.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        cbar=False, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(name, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page: ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 ROC Curves & AUC":
    st.markdown('<p class="section-header">📈 ROC Curves & AUC</p>', unsafe_allow_html=True)
    roc_data = load_roc_data()

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for idx, (name, d) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(
            x=d["fpr"], y=d["tpr"],
            mode="lines",
            name=f"{name} (AUC = {d['auc']:.3f})",
            line=dict(color=colors[idx % len(colors)], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        name="Random Classifier",
        showlegend=True,
    ))
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=550,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98, font=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🏆 AUC Ranking")
    auc_rank = sorted(roc_data.items(), key=lambda x: x[1]["auc"], reverse=True)
    for rank, (name, d) in enumerate(auc_rank, 1):
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank - 1]
        st.markdown(f"{medal} **{rank}. {name}** — AUC = `{d['auc']:.4f}`")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Feature Importance":
    st.markdown('<p class="section-header">🎯 Feature Importance</p>', unsafe_allow_html=True)
    sel = st.selectbox("Select model", MODEL_NAMES, key="fi_model")
    shap_data = load_shap_data(sel)
    fi_df = shap_data["feature_importance"]
    
    fi_df = fi_df.reset_index(drop=True)
    fi_df["Feature"] = fi_df["Feature"].astype(str)
    fi_df["Mean_SHAP"] = pd.to_numeric(fi_df["Mean_SHAP"], errors="coerce").fillna(0)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(
            fi_df,
            x="Mean_SHAP",
            y="Feature",
            orientation="h",
            color="Mean_SHAP",
            color_continuous_scale="RdYlGn_r",
            template="plotly_white",
        )
        fig.update_layout(
            title=f"Mean |SHAP| — {sel}",
            height=550,
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 📌 Top 5 Drivers")
        top5 = fi_df.head(5)
        for _, row in top5.iterrows():
            st.markdown(
                f'<div class="metric-box"><b>{row["Feature"]}</b><br>'
                f'Mean |SHAP| = {row["Mean_SHAP"]:.4f}</div>',
                unsafe_allow_html=True,
            )
        st.info("Higher mean |SHAP| values indicate features that have a stronger influence on model predictions across the dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Explainability (XAI) - FINAL FIXED VERSION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Explainability (XAI)":
    st.markdown('<p class="section-header">🧠 Explainable AI (XAI)</p>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box">'
        '<strong>🔍 What is SHAP?</strong><br>'
        'SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. '
        'SHAP values quantify each feature\'s contribution to each prediction, ensuring <strong>transparency and trust</strong>.'
        '</div>',
        unsafe_allow_html=True
    )

    sel = st.selectbox("Select model", MODEL_NAMES, key="xai_model")
    shap_data = load_shap_data(sel)

    tab_a, tab_b = st.tabs(["🌊 Global Summary", "🔬 Local (Single Prediction)"])

    with tab_a:
        st.markdown("### Global Feature Impact")
        st.markdown("This visualization shows how each feature affects predictions across all test samples.")
        
        X_sample = shap_data["X_test_sample"]
        sv_pos = shap_data["shap_values_positive"]

        # Get feature names
        if isinstance(X_sample, pd.DataFrame):
            X_arr = X_sample.values
            feat_names = list(X_sample.columns)
        else:
            X_arr = X_sample
            feat_names = load_feature_names()

        # Ensure sv_pos is 2D: (n_samples, n_features)
        sv_pos = np.array(sv_pos)
        if sv_pos.ndim == 1:
            sv_pos = sv_pos.reshape(1, -1)
        elif sv_pos.ndim > 2:
            sv_pos = sv_pos.reshape(sv_pos.shape[0], -1)
        
        # Ensure X_arr matches sv_pos shape
        if X_arr.shape[0] != sv_pos.shape[0]:
            min_samples = min(X_arr.shape[0], sv_pos.shape[0])
            X_arr = X_arr[:min_samples]
            sv_pos = sv_pos[:min_samples]

        try:
            explanation = shap.Explanation(
                values=sv_pos,
                base_values=np.full(sv_pos.shape[0], shap_data["expected_value"]),
                data=X_arr,
                feature_names=feat_names,
            )
            fig_summary = shap.plots.beeswarm(explanation, show=False)
            st.pyplot(fig_summary, bbox_inches="tight")
            st.caption(
                "Each dot is a prediction. Horizontal position = SHAP value (impact on output). "
                "Color = feature value (red = high, blue = low)."
            )
        except Exception as e:
            st.warning(f"Unable to generate beeswarm plot: {e}")
            st.info("Showing alternative visualization...")
            
            # Fallback: bar chart of mean absolute SHAP values
            mean_abs_shap = np.abs(sv_pos).mean(axis=0)
            mean_abs_shap = np.asarray(mean_abs_shap).flatten()
            
            feat_names_list = list(feat_names)
            
            if len(feat_names_list) != len(mean_abs_shap):
                min_len = min(len(feat_names_list), len(mean_abs_shap))
                feat_names_list = feat_names_list[:min_len]
                mean_abs_shap = mean_abs_shap[:min_len]
            
            fi_df = pd.DataFrame({
                "Feature": feat_names_list,
                "Mean |SHAP|": mean_abs_shap.tolist()
            }).sort_values("Mean |SHAP|", ascending=True)
            
            fig = px.bar(
                fi_df,
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                title=f"Mean |SHAP| Values — {sel}",
                template="plotly_white",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        st.markdown("### Local Explanation for a Single Prediction")
        st.markdown(
            '<div class="warning-box">'
            '👉 Go to <strong>🔮 Predict Risk</strong> to generate a patient-specific SHAP waterfall plot. '
            'The waterfall plot shows how each feature contributes to pushing the prediction from the base value to the final output.'
            '</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<p class="section-header">ℹ️ About This Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown(
        """
        This dashboard predicts the presence of **heart disease** based on 13 clinical features
        using five machine learning classifiers trained on the UCI Heart Disease dataset.
        """
    )

    st.markdown("### 🧬 Features Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**\n- Age\n- Sex")
    with col2:
        st.markdown("**Vitals**\n- Resting BP\n- Cholesterol\n- Max Heart Rate")
    with col3:
        st.markdown("**Symptoms**\n- Chest Pain Type\n- Exercise-Induced Angina")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("**Tests**\n- Fasting Blood Sugar\n- Resting ECG\n- Oldpeak\n- Slope")
    with col5:
        st.markdown("**Imaging**\n- Major Vessels (Ca)\n- Thalassemia (Thal)")
    with col6:
        st.markdown("**Target**\n- AHD (Heart Disease)\n  - 0 = No\n  - 1 = Yes")

    st.markdown("### 🤖 Models Trained")
    models_df = pd.DataFrame({
        "#": [1, 2, 3, 4, 5],
        "Model": ["Random Forest", "XGBoost", "Logistic Regression", "KNN", "SVM"],
        "Type": ["Ensemble (Tree)", "Gradient Boosting", "Linear", "Instance-based", "Kernel-based"]
    })
    st.dataframe(models_df, use_container_width=True, hide_index=True)

    st.markdown("### 🧠 Explainability")
    st.markdown(
        """
        - **Tree models** → `shap.TreeExplainer` (exact, fast)
        - **Other models** → `shap.KernelExplainer` (model-agnostic)
        
        SHAP provides both **global** (dataset-level) and **local** (instance-level) explanations.
        """
    )

    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        """
        This tool is for **educational and research purposes only**. It is **not a substitute**
        for professional medical advice, diagnosis, or treatment. Always consult a qualified
        healthcare provider for medical decisions.
        """
    )