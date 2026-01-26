from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


nb_model = pickle.load(open("nb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

# categorical columns used during training
CAT_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]


def preprocess_input(df):
    """
    Takes RAW user input dataframe
    Returns MODEL-READY scaled numpy array
    """

    # ---------- 1. FEATURE ENGINEERING ----------
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    # drop original columns (same as training)
    df = df.drop(columns=["Credit_Score", "DTI_Ratio"])

    # ---------- 2. ONE HOT ENCODING ----------
    encoded = ohe.transform(df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=df.index
    )

    df = pd.concat(
        [df.drop(columns=CAT_COLS), encoded_df],
        axis=1
    )

    # ---------- 3. FIX COLUMN ORDER ----------
    df = df.reindex(columns=scaler.feature_names_in_)

    # ---------- 4. SCALING ----------
    df_scaled = scaler.transform(df)

    return df_scaled


@app.route("/")
def home():
    return render_template("index.html", prediction_text=None)


@app.route("/predict", methods=["POST"])
def predict():


    data = {
        "Applicant_Income": float(request.form["Applicant_Income"]),
        "Coapplicant_Income": float(request.form["Coapplicant_Income"]),
        "Age": int(request.form["Age"]),
        "Dependents": int(request.form["Dependents"]),
        "Credit_Score": float(request.form["Credit_Score"]),
        "Existing_Loans": int(request.form["Existing_Loans"]),
        "DTI_Ratio": float(request.form["DTI_Ratio"]),
        "Savings": float(request.form["Savings"]),
        "Collateral_Value": float(request.form["Collateral_Value"]),
        "Loan_Amount": float(request.form["Loan_Amount"]),
        "Loan_Term": int(request.form["Loan_Term"]),
        "Education_Level": int(request.form["Education_Level"]),
        "Employment_Status": request.form["Employment_Status"],
        "Marital_Status": request.form["Marital_Status"],
        "Loan_Purpose": request.form["Loan_Purpose"],
        "Property_Area": request.form["Property_Area"],
        "Gender": request.form["Gender"],
        "Employer_Category": request.form["Employer_Category"]
    }

    raw_df = pd.DataFrame([data])


    final_input = preprocess_input(raw_df)

  
    prediction = nb_model.predict(final_input)[0]

    result = "Loan Approved ✅" if int(prediction) == 1 else "Loan Rejected ❌"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run()

