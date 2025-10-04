# ================================
# 🖼️ Streamlit App v2 - YouTube Revenue Predictor (Bar Chart Only)
# ================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="📺 YouTube Revenue Predictor", page_icon="💰", layout="centered")

# Load trained model
model = joblib.load("best_model.pkl")

# Title
st.title("💰 YouTube Ad Revenue Predictor")
st.write("Predict YouTube Ad Revenue based on video performance, category, device, and country 🌍")

# ============================
# 📥 Centered Input Form
# ============================
with st.form(key='prediction_form'):
    st.subheader("📌 Enter Video Details")

    col1, col2 = st.columns(2)

    with col1:
        views = st.number_input("Views", min_value=0, value=1000)
        likes = st.number_input("Likes", min_value=0, value=100)
        comments = st.number_input("Comments", min_value=0, value=10)
    with col2:
        watch_time = st.number_input("Watch Time (minutes)", min_value=0, value=500)
        video_length = st.number_input("Video Length (minutes)", min_value=1, value=10)
        subscribers = st.number_input("Subscribers", min_value=0, value=10000)

    st.markdown("---")

    category = st.selectbox("🎮 Category", ["Gaming", "Music", "Lifestyle", "Education", "Tech"])
    device = st.selectbox("💻 Device", ["Mobile", "Desktop", "Tablet", "TV"])
    country = st.selectbox("🌍 Country", ["US", "IN", "UK", "CA", "AU"])

    predict_btn = st.form_submit_button("🚀 Predict Revenue")

# ============================
# 🔮 Prediction Section
# ============================
if predict_btn:
    # Prepare input DataFrame
    engagement_rate = (likes + comments) / (views + 1)
    input_df = pd.DataFrame([{
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time,
        "video_length_minutes": video_length,
        "subscribers": subscribers,
        "engagement_rate": engagement_rate
    }])

    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # One-hot activation
    cat_col = f"category_{category}"
    dev_col = f"device_{device}"
    country_col = f"country_{country}"

    if cat_col in input_df.columns:
        input_df[cat_col] = 1
    if dev_col in input_df.columns:
        input_df[dev_col] = 1
    if country_col in input_df.columns:
        input_df[country_col] = 1

    # Align with training columns
    input_df = input_df[expected_cols]

    # Predict
    prediction = model.predict(input_df)[0]

    # Display Prediction
    st.success(f"💵 **Estimated Ad Revenue:** ${prediction:,.2f} USD")
    st.balloons()

    # ============================
    # 📊 Visualization Section (Bar Chart Only)
    # ============================
    st.markdown("---")
    st.subheader("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": model.feature_names_in_,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False).head(10)

        st.write("### 🧠 Top 10 Most Important Features")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="importance", y="feature", data=importance_df, palette="viridis", ax=ax)
        ax.set_title("Top 10 Feature Importances")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Feature importance is available only for tree-based models.")




# - cd "C:\Users\luna love\Downloads\Content_Monetization_Modeler\app"
#streamlit run streamlit_app_v2.py

