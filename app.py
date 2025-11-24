import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.title("Mushroom Edibility Classifier")

bundle = joblib.load("mushroom_model.joblib")
pipe = bundle["pipeline"]
features = bundle["features"]

# Simple UI: build a row of inputs
st.write("Enter attributes (as in the UCI codes):")
row = {}
for f in features:
    row[f] = st.text_input(f, value="")

if st.button("Predict"):
    X = pd.DataFrame([row])[features].astype(str)
    yhat = pipe.predict(X)[0]
    proba = getattr(pipe, "predict_proba", lambda X: [[None, None]])(X)[0]
    label = "poisonous" if yhat==1 else "edible"
    st.subheader(f"Prediction: **{label.upper()}**")
    if proba[0] is not None:
        st.write(f"Probabilities → edible: {proba[0]:.3f}, poisonous: {proba[1]:.3f}")

st.markdown("""
**Model card (summary)**  
- Data: UCI Mushroom (all categorical; two binary as per variables table; `stalk-root` had '?' missing).  
- Algorithm: pipeline chosen by 10-fold CV among KNN / DecisionTree / CategoricalNB.  
- Caveat: Foraging safety requires human expert verification.
""")

