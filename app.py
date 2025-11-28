import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
    layout="wide"
)

# Load model bundle
bundle = joblib.load("mushroom_model.joblib")
pipe = bundle["pipeline"]
features = bundle["features"]

# ---------------------------
# UCI ATTRIBUTE OPTIONS
# ---------------------------
OPTIONS = {
    "cap-shape": ["b","c","x","f","k","s"],
    "cap-surface": ["f","g","y","s"],
    "cap-color": ["n","b","c","g","r","p","u","e","w","y"],
    "bruises": ["t","f"],
    "odor": ["a","l","c","y","f","m","n","p","s"],
    "gill-attachment": ["a","d","f","n"],
    "gill-spacing": ["c","w","d"],
    "gill-size": ["b","n"],
    "gill-color": ["k","n","b","h","g","r","o","p","u","e","w","y"],
    "stalk-shape": ["e","t"],
    "stalk-root": ["b","c","u","e","z","r","?"],
    "stalk-surface-above-ring": ["f","y","k","s"],
    "stalk-surface-below-ring": ["f","y","k","s"],
    "stalk-color-above-ring": ["n","b","c","g","o","p","e","w","y"],
    "stalk-color-below-ring": ["n","b","c","g","o","p","e","w","y"],
    "veil-color": ["n","o","w","y"],
    "ring-number": ["n","o","t"],
    "ring-type": ["c","e","f","l","n","p","s","z"],
    "spore-print-color": ["k","n","b","h","r","o","u","w","y"],
    "population": ["a","c","n","s","v","y"],
    "habitat": ["g","l","m","p","u","w","d"]
}

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("🍄 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Classifier", "Model Comparison", "EDA Visuals", "About Project"]
)


# ---------------------------
# PRESET SAMPLES
# ---------------------------
EDIBLE_SAMPLE = {
    'cap-shape': 'x','cap-surface': 's','cap-color': 'w','bruises': 't',
    'odor': 'a','gill-attachment': 'f','gill-spacing': 'c','gill-size': 'b',
    'gill-color': 'w','stalk-shape': 'e','stalk-root': 'e',
    'stalk-surface-above-ring': 's','stalk-surface-below-ring': 's',
    'stalk-color-above-ring': 'w','stalk-color-below-ring': 'w',
    'veil-color': 'w','ring-number': 'o','ring-type': 'p',
    'spore-print-color': 'w','population': 's','habitat': 'g'
}

POISON_SAMPLE = {
    'cap-shape': 'x','cap-surface': 's','cap-color': 'n','bruises': 't',
    'odor': 'f','gill-attachment': 'f','gill-spacing': 'c','gill-size': 'b',
    'gill-color': 'n','stalk-shape': 'e','stalk-root': 'e',
    'stalk-surface-above-ring': 's','stalk-surface-below-ring': 's',
    'stalk-color-above-ring': 'w','stalk-color-below-ring': 'w',
    'veil-color': 'w','ring-number': 'o','ring-type': 'p',
    'spore-print-color': 'k','population': 'a','habitat': 'u'
}

# ---------------------------
# PAGE 1 — CLASSIFIER UI
# ---------------------------
if page == "Classifier":

    st.title("🍄 Mushroom Edibility Classifier")
    st.write("Enter mushroom attributes using **dropdowns**, or load an example.")

    # ---- Sample buttons (must retain values after clicking Predict) ----
    if "stored_inputs" not in st.session_state:
        st.session_state.stored_inputs = {f: "" for f in features}

    colA, colB = st.columns([1, 1])
    if colA.button("Load Edible Example"):
        st.session_state.stored_inputs = EDIBLE_SAMPLE.copy()
    if colB.button("Load Poisonous Example"):
        st.session_state.stored_inputs = POISON_SAMPLE.copy()

    st.markdown("---")

    # ---- Two-column dropdown layout ----
    left, right = st.columns(2)

    for idx, f in enumerate(features):
        if idx % 2 == 0:
            st.session_state.stored_inputs[f] = left.selectbox(
                f"**{f}**",
                OPTIONS[f],
                index=OPTIONS[f].index(st.session_state.stored_inputs.get(f, "")) 
                      if st.session_state.stored_inputs.get(f, "") in OPTIONS[f] else 0
            )
        else:
            st.session_state.stored_inputs[f] = right.selectbox(
                f"**{f}**",
                OPTIONS[f],
                index=OPTIONS[f].index(st.session_state.stored_inputs.get(f, "")) 
                      if st.session_state.stored_inputs.get(f, "") in OPTIONS[f] else 0
            )

    st.markdown("---")

    # ---- Predict button ----
    if st.button("🔍 Predict", use_container_width=True):

        X = pd.DataFrame([st.session_state.stored_inputs])[features].astype(str)
        yhat = pipe.predict(X)[0]
        proba = pipe.predict_proba(X)[0]

        label = "🍄 EDIBLE" if yhat == 0 else "☠️ POISONOUS"
        color = "green" if yhat == 0 else "red"

        st.markdown(
            f"""
            <div style="
                border-radius: 10px;
                padding: 22px;
                background-color: #222;
                border-left: 10px solid {color};
                margin-top:15px;
            ">
                <h2 style="color:white;">Prediction: {label}</h2>
                <p style="color:white;">
                    <b>Probabilities</b><br>
                    Edible: {proba[0]:.3f} &nbsp;&nbsp;|&nbsp;&nbsp;
                    Poisonous: {proba[1]:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
# ---------------------------
# PAGE 2 — MODEL COMPARISON
# ---------------------------
elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    st.write("Below is the performance of the **3 models** you evaluated in the notebook.")

    # Hardcoded table from your results
    summary = pd.DataFrame([
        ["KNN(k=5)",       1.000000, 0.000000, "S1"],
        ["DecisionTree",   0.999754, 0.000739, "S1"],
        ["CategoricalNB",  0.955106, 0.008363, "S1"],
        ["KNN(k=5)",       1.000000, 0.000000, "S2"],
        ["DecisionTree",   0.999754, 0.000739, "S2"],
        ["CategoricalNB",  0.955072, 0.008237, "S2"],
    ], columns=["Model", "Mean Accuracy", "Std", "Strategy"])

    st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Best Accuracy per Model (Visual Comparison)")

    # Side-by-side layout
    left_col, right_col = st.columns([1, 1])

    # LEFT → Bar Chart
    with left_col:
        st.write("### 📈 Accuracy Bar Chart (Smaller Size)")

        best = summary.groupby("Model")["Mean Accuracy"].max()

        fig, ax = plt.subplots(figsize=(4, 3))  # smaller figure
        ax.bar(best.index, best.values, color=["#2ecc71", "#3498db", "#e74c3c"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.9, 1.01)
        ax.set_title("10-Fold CV Accuracy")
        ax.tick_params(axis='x', rotation=20)

        st.pyplot(fig)

    # RIGHT → Explanation
    with right_col:
        st.write("### 📝 What This Means")
        st.markdown("""
        - **KNN (k=5)** achieved a perfect **1.00 accuracy** on both S1 and S2.  
        - **Decision Tree** is extremely close (0.9997).  
        - **Categorical Naive Bayes** performed well but noticeably lower (~0.955).  

        **Conclusion:**  
        KNN and Decision Tree are nearly identical in performance,  
        but **KNN is chosen as the final model** due to perfect accuracy.
        """)

    st.markdown("---")

# ---------------------------
# PAGE 3 — EDA VISUALS
# ---------------------------
elif page == "EDA Visuals":
    st.title("EDA Visualizations")

    st.write("These visualizations come from the preprocessing notebook.")

    img_folder = "figures"
    for img in os.listdir(img_folder):
        if img.endswith(".png"):
            st.image(os.path.join(img_folder, img))
            st.markdown("---")

# ---------------------------
# PAGE 4 — ABOUT PROJECT
# ---------------------------
else:
    st.title("About the Project")

    st.markdown("""
    ### **ITEC 3040 — Final Project: Mushroom Classification**

    **Goal:**  
    Build a model to classify mushrooms from the UCI dataset as *edible* or *poisonous*  
    using 21 categorical attributes.

    **Project Highlights:**  
    - Full dataset EDA  
    - Missing value strategies  
    - Interactive Streamlit classifier  
    - Achieved **100% accuracy (KNN)**

    **Group Members:**  
    Mirza, Hashmat, Taha, Divine, Maxmillian
    """)

