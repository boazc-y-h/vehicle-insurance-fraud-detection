from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Set a secret key for sessions (required for redirects)
app.secret_key = 'your_unique_secret_key_here'

# Load your model (assuming it's a machine learning model stored in 'model.pkl')
model = pickle.load(open('models/xgboost_none.pkl', 'rb'))
expected_columns = model.feature_names_in_.tolist()
numeric_columns = ["WeekOfMonth", "WeekOfMonthClaimed", "Age", "PolicyNumber",
                   "RepNumber", "Deductible", "DriverRating", "Year"]

# Load policy details
policy_data = pd.read_csv("data/interim/policy_data.csv")

# Preprocessing function
def preprocess_input(new_data):
    # Ensure it's a DataFrame
    new_df = pd.DataFrame([new_data])

    # Identify categorical columns (same ones used during training)
    categorical_columns = [col for col in new_df.columns if col not in numeric_columns]

    # Apply one-hot encoding to categorical columns
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_columns)

    # Ensure all expected columns exist
    missing_cols = list(set(expected_columns) - set(new_df_encoded.columns))  # Convert to list
    missing_df = pd.DataFrame(0, index=new_df_encoded.index, columns=missing_cols)
    new_df_encoded = pd.concat([new_df_encoded, missing_df], axis=1)

    # Reorder columns to match expected_columns
    new_df_encoded = new_df_encoded[expected_columns]

    return new_df_encoded

# Home route for the input form
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# /predict route for handling prediction logic
@app.route("/predict", methods=["POST"])
def predict():
    policy_number = int(request.form["PolicyNumber"])
    
    # Block policy_number 1517 and redirect to error page
    if policy_number == 1517:
        return redirect(url_for('error'))  # Redirect to the error page
    
    # Query the dataset for vehicle details
    try:
        policy_details = policy_data[policy_data["PolicyNumber"] == policy_number].iloc[0]
    except IndexError:
        return redirect(url_for('error'))  # Redirect to error page if policy not found
    
    
    # Query the dataset for vehicle details
    policy_details = policy_data[policy_data["PolicyNumber"] == policy_number].iloc[0]
    
    # Collect form data
    new_input = {
        "WeekOfMonth": int(request.form["WeekOfMonth"]),
        "WeekOfMonthClaimed": int(request.form["WeekOfMonthClaimed"]),
        "Age": int(policy_details["Age"]),
        "PolicyNumber": policy_number,
        "RepNumber": int(policy_details["RepNumber"]),
        "Deductible": int(policy_details["Deductible"]),
        "DriverRating": int(policy_details["DriverRating"]),
        "Year": int(request.form["Year"]),
        "Month": request.form["Month"],
        "DayOfWeek": request.form["DayOfWeek"],
        "Make": policy_details["Make"],
        "AccidentArea": request.form["AccidentArea"],
        "DayOfWeekClaimed": request.form["DayOfWeekClaimed"],
        "MonthClaimed": request.form["MonthClaimed"],
        "Sex": policy_details["Sex"],
        "MaritalStatus": policy_details["MaritalStatus"],
        "Fault": request.form["Fault"],
        "PolicyType": policy_details["PolicyType"],
        "VehicleCategory": policy_details["VehicleCategory"],
        "VehiclePrice": policy_details["VehiclePrice"],
        "Days:Policy-Accident": request.form["Days:Policy-Accident"],
        "Days:Policy-Claim": request.form["Days:Policy-Claim"],
        "PastNumberOfClaims": policy_details["PastNumberOfClaims"],
        "AgeOfVehicle": policy_details["AgeOfVehicle"],
        "AgeOfPolicyHolder": policy_details["AgeOfPolicyHolder"],
        "PoliceReportFiled": request.form["PoliceReportFiled"],
        "WitnessPresent": request.form["WitnessPresent"],
        "AgentType": policy_details["AgentType"],
        "NumberOfSuppliments": request.form["NumberOfSuppliments"],
        "AddressChange-Claim": policy_details["AddressChange-Claim"],
        "NumberOfCars": request.form["NumberOfCars"],
        "BasePolicy": policy_details["BasePolicy"]
    }
    
    # Create policy details dictionary
    policy_dict = {
        "Policy Number": policy_number,
        "Base Policy": policy_details["BasePolicy"],
        "Deductible": int(policy_details["Deductible"])
    }

    # Create policy holder dictionary
    policy_holder = {
        "Age": int(policy_details["Age"]),
        "Gender": policy_details["Sex"],
        "Marital Status": policy_details["MaritalStatus"],
        "Driver Rating": int(policy_details["DriverRating"]),
        "Past Number Of Claims": policy_details["PastNumberOfClaims"],
        "Last Address Change Since Claim": policy_details["AddressChange-Claim"],
    }

    # Create vehicle details dictionary
    vehicle_details = {
        "Make": policy_details["Make"],
        "Vehicle Category": policy_details["VehicleCategory"],
        "Vehicle Price": policy_details["VehiclePrice"],
        "Age Of Vehicle": policy_details["AgeOfVehicle"],
    }
    
    # Create agent details dictionary
    agent_details = {
        "Agent Type": policy_details["AgentType"],
        "Rep Number": int(policy_details["RepNumber"])
    }
    
    # Create Accident Details dictionary
    accident_details = {
        "Year": int(request.form["Year"]),
        "Month": request.form["Month"],
        "Week Of Month": int(request.form["WeekOfMonth"]),
        "Day Of Week": request.form["DayOfWeek"],
        "Accident Area": request.form["AccidentArea"],
        "Fault": request.form["Fault"],
        "Days From Policy Start": request.form["Days:Policy-Accident"],
        "Number Of Cars Involved": request.form["NumberOfCars"]
    }

    # Create claim details dictionary
    claim_details = {
        "Month": request.form["MonthClaimed"],
        "Week Of Month": int(request.form["WeekOfMonthClaimed"]),
        "Day Of Week": request.form["DayOfWeekClaimed"],
        "Days From Policy Start": request.form["Days:Policy-Claim"],
        "Was a police report filed?": request.form["PoliceReportFiled"],
        "Was any witness present?": request.form["WitnessPresent"],
        "Number of Suppliments": request.form["NumberOfSuppliments"],
    }
    
    # Preprocess input
    preprocessed_input = preprocess_input(new_input)

    # Make prediction
    prediction_raw = model.predict(preprocessed_input)[0]  # Get the first (only) prediction
    prediction = "Legitimate" if prediction_raw == 0 else "Fraudulent"

    # Render result page with prediction
    return render_template("result.html", 
                           prediction=prediction,
                           accident_details=accident_details,
                           claim_details=claim_details,
                           policy_dict=policy_dict,
                           policy_holder=policy_holder, 
                           vehicle_details=vehicle_details,
                           agent_details=agent_details,
                           )

# Error route to handle the error page
@app.route("/error")
def error():
    return render_template("error.html")  # Render the error page


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)