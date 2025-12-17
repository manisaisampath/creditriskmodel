import pickle
import numpy as np
import pandas as pd

# ===============================
# MODEL PATH
# ===============================
MODEL_PATH = r"C:\Users\Sampath\OneDrive\Desktop\credmain\main_model_data.pkl"

# ===============================
# Load model artifacts
# ===============================
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
features = list(model_data["features"])
cols_to_scale = list(model_data["cols_to_scale"])


# ===============================
# Prepare input
# ===============================
def prepare_input(
    age, income, loan_amount, loan_tenure_months,
    avg_dpd_per_delinquency, delinquency_ratio,
    credit_utilization_ratio, num_open_accounts,
    residence_type, loan_purpose, loan_type
):

    input_data = {
        "age": age,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": num_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income": loan_amount / income if income > 0 else 0,

        # ✅ FIXED COLUMN NAME
        "delinquent_ratio": delinquency_ratio,

        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,

        # One-hot encoded categorical variables
        "residence_type_Owned": 1 if residence_type == "Owned" else 0,
        "residence_type_Rented": 1 if residence_type == "Rented" else 0,

        "loan_purpose_Education": 1 if loan_purpose == "Education" else 0,
        "loan_purpose_Home": 1 if loan_purpose == "Home" else 0,
        "loan_purpose_Personal": 1 if loan_purpose == "Personal" else 0,

        "loan_type_Unsecured": 1 if loan_type == "Unsecured" else 0,

        # Dummy values required for scaler consistency
        "number_of_dependants": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": 1,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": 1,
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts": 1,
        "enquiry_count": 1
    }

    df = pd.DataFrame([input_data])

    # ===============================
    # Ensure all scaling columns exist
    # ===============================
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0

    # Scale numeric columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # ===============================
    # Ensure feature alignment
    # ===============================
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    return df


# ===============================
# Credit score calculation
# ===============================
def calculate_credit_score(input_df, base_score=300, scale_length=600):

    logit = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-logit))
    non_default_probability = 1 - default_probability

    credit_score = int(base_score + non_default_probability.flatten()[0] * scale_length)

    if credit_score < 500:
        rating = "Poor"
    elif credit_score < 650:
        rating = "Average"
    elif credit_score < 750:
        rating = "Good"
    else:
        rating = "Excellent"

    return float(default_probability[0][0]), credit_score, rating


# ===============================
# Final prediction function
# ===============================
def predict(
    age, income, loan_amount, loan_tenure_months,
    avg_dpd_per_delinquency, delinquency_ratio,
    credit_utilization_ratio, num_open_accounts,
    residence_type, loan_purpose, loan_type
):

    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency, delinquency_ratio,
        credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    return calculate_credit_score(input_df)
