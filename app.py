import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
    layout="wide"
)

# ---------------------------
# Load model bundle
# ---------------------------
MODEL_PATH = "mushroom_model.joblib"
try:
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    features = bundle["features"]
except Exception as e:
    st.error(f"Failed to load model bundle at '{MODEL_PATH}': {e}")
    st.stop()

# ---------------------------
# UCI ATTRIBUTE OPTIONS (raw codes)
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
# Human readable meanings (MEANINGS)
# ---------------------------
MEANINGS = {
    "cap-shape": {"b":"bell", "c":"conical", "x":"convex", "f":"flat", "k":"knobbed", "s":"sunken"},
    "cap-surface": {"f":"fibrous", "g":"grooves", "y":"scaly", "s":"smooth"},
    "cap-color": {"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"},
    "bruises": {"t":"bruises","f":"no bruises"},
    "odor": {"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"},
    "gill-attachment": {"a":"attached","d":"descending","f":"free","n":"notched"},
    "gill-spacing": {"c":"close","w":"crowded","d":"distant"},
    "gill-size": {"b":"broad","n":"narrow"},
    "gill-color": {"k":"black","n":"brown","b":"buff","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"},
    "stalk-shape": {"e":"enlarging","t":"tapering"},
    "stalk-root": {"b":"bulbous","c":"club","u":"cup","e":"equal","z":"rhizomorphs","r":"rooted","?":"missing"},
    "stalk-surface-above-ring": {"f":"fibrous","y":"scaly","k":"silky","s":"smooth"},
    "stalk-surface-below-ring": {"f":"fibrous","y":"scaly","k":"silky","s":"smooth"},
    "stalk-color-above-ring": {"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"},
    "stalk-color-below-ring": {"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"},
    "veil-color": {"n":"brown","o":"orange","w":"white","y":"yellow"},
    "ring-number": {"n":"none","o":"one","t":"two"},
    "ring-type": {"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"},
    "spore-print-color": {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"},
    "population": {"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"},
    "habitat": {"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"woods","d":"waste"}
}

# ---------------------------
# Helper to format options in selectbox while keeping original code as value
# ---------------------------
def option_formatter(feature, code):
    return f"{code} — {MEANINGS.get(feature, {}).get(code, 'unknown')}"

# ---------------------------
# Helper to pretty-print sample
# ---------------------------
def pretty_sample(sample):
    parts = []
    for f, v in sample.items():
        text = MEANINGS.get(f, {}).get(v, v)
        parts.append(f"{f}: {v} ({text})")
    return " • ".join(parts)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("🍄 Navigation")
page = st.sidebar.radio("Go to:", ["Classifier", "Model Comparison", "EDA Visuals", "About Project"])

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
    'cap-shape': 'x','cap-surface': 's','cap-color': 'n','bruises': 'f',
    'odor': 'p','gill-attachment': 'f','gill-spacing': 'c','gill-size': 'n',
    'gill-color': 'k','stalk-shape': 'e','stalk-root': 'e',
    'stalk-surface-above-ring': 's','stalk-surface-below-ring': 's',
    'stalk-color-above-ring': 'w','stalk-color-below-ring': 'w',
    'veil-color': 'w','ring-number': 'o','ring-type': 'p',
    'spore-print-color': 'k','population': 'a','habitat': 'u'
}

# ---------------------------
# Initialize stored inputs
# ---------------------------
if "stored_inputs" not in st.session_state:
    st.session_state.stored_inputs = {f: (OPTIONS[f][0] if f in OPTIONS else "") for f in features}

# ---------------------------
# PAGE 1 — CLASSIFIER UI
# ---------------------------
if page == "Classifier":
    st.title("🍄 Mushroom Edibility Classifier")
    st.write("Select attributes (dropdowns show code — meaning). Load an example or set values and Predict.")

    colA, colB = st.columns([1, 1])
    if colA.button("Load Edible Example"):
        st.session_state.stored_inputs = EDIBLE_SAMPLE.copy()
        st.success("Loaded edible example: " + pretty_sample(EDIBLE_SAMPLE))
    if colB.button("Load Poisonous Example"):
        st.session_state.stored_inputs = POISON_SAMPLE.copy()
        st.success("Loaded poisonous example: " + pretty_sample(POISON_SAMPLE))

    st.markdown("---")

    # Two-column dropdown layout with human-readable labels
    left, right = st.columns(2)
    for idx, f in enumerate(features):
        opts = OPTIONS.get(f, [""])
        target_col = left if idx % 2 == 0 else right

        # determine default selection index
        current_val = st.session_state.stored_inputs.get(f, "")
        try:
            idx0 = opts.index(current_val) if current_val in opts else 0
        except ValueError:
            idx0 = 0

        # Use format_func to display readable text while returning the raw code
        st.session_state.stored_inputs[f] = target_col.selectbox(
            f"**{f}**",
            opts,
            index=idx0,
            format_func=lambda code, feature=f: option_formatter(feature, code)
        )

    st.markdown("---")

    if st.button("🔍 Predict", use_container_width=True):
        X = pd.DataFrame([st.session_state.stored_inputs])[features].astype(str)
        try:
            yhat = pipe.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Try predict_proba if available
        proba = None
        try:
            proba = pipe.predict_proba(X)[0]
        except Exception:
            proba = None

        label = "🍄 EDIBLE" if yhat == 0 else "☠️ POISONOUS"
        color = "green" if yhat == 0 else "red"

        probs_html = ""
        if proba is not None:
            probs_html = f"Edible: {proba[0]:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; Poisonous: {proba[1]:.3f}"
        else:
            probs_html = "Probabilities unavailable for this model."

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
                    {probs_html}
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
    st.write("Below is the performance of the models evaluated in the notebook.")

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

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.write("### 📈 Accuracy Bar Chart")
        best = summary.groupby("Model")["Mean Accuracy"].max()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(best.index, best.values, color=["#2ecc71", "#3498db", "#e74c3c"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.9, 1.01)
        ax.set_title("10-Fold CV Accuracy")
        ax.tick_params(axis='x', rotation=20)
        st.pyplot(fig)

    with right_col:
        st.write("### 📝 Explanation")
        st.markdown("""
        - **KNN (k=5)** achieved perfect **1.00 accuracy** on both strategies.  
        - **Decision Tree** is extremely close (0.9997).  
        - **Categorical Naive Bayes** performed well but lower (~0.955).  

        **Conclusion:** KNN and Decision Tree are effectively perfect on this dataset, with KNN chosen in our notebook as the final pipeline due to perfect CV accuracy.
        """)

    st.markdown("---")

# ---------------------------
# PAGE 3 — EDA VISUALS
# ---------------------------
elif page == "EDA Visuals":
    st.title("EDA Visualizations")
    st.write("Visualizations generated during preprocessing and evaluation (from `figures/`).")

    img_folder = "figures"
    if not os.path.exists(img_folder):
        st.warning(f"No figures folder found at '{img_folder}'. Run the notebook to generate visuals.")
    else:
        imgs = sorted([p for p in os.listdir(img_folder) if p.endswith(".png")])
        if not imgs:
            st.info("No PNG images found in the figures folder.")
        for img in imgs:
            st.image(os.path.join(img_folder, img), use_column_width=True)
            st.markdown("---")

# ---------------------------
# PAGE 4 — ABOUT PROJECT
# ---------------------------
else:
    st.title("About the Project")
    st.markdown("""
    ### **ITEC 3040 — Final Project: Mushroom Classification**

    **Goal:**  
    Build a model to classify mushrooms from the UCI dataset as *edible* or *poisonous* using categorical attributes.

    **Project Highlights:**  
    - Full dataset EDA  
    - Missing value strategies  
    - Interactive Streamlit classifier  
    - Model artifacts and visualizations saved under `figures/` and `app/`

    **Group Members:**  
    Mirza Baig, Hashmat Jadoon, Taha Hashmi, Divine Consile Dikoka, Maximillian Dow
    """)
