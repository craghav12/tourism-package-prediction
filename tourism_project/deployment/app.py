"""
Streamlit app for the "Visit with Us" Wellness Tourism Package predictor.

Loads the model trained and committed by the MLOps pipeline
(tourism_project/deployment/model.joblib) and predicts whether a customer
is likely to purchase the Wellness Tourism Package.
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Predict whether a customer is likely to purchase the new **Wellness "
    "Tourism Package**, based on their profile and sales-pitch interaction "
    "details."
)

st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0, value=20000, step=500
    )

with col2:
    number_of_persons_visiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2
    )
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting (below age 5)", min_value=0, max_value=10, value=0
    )
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    number_of_trips = st.number_input(
        "Average Number of Trips per Year", min_value=0, max_value=20, value=2
    )
    passport = st.selectbox("Holds Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

st.header("Sales Interaction Details")
col3, col4 = st.columns(2)

with col3:
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=60, value=15
    )

with col4:
    number_of_followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=3
    )
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

if st.button("Predict"):
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": type_of_contact,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": number_of_persons_visiting,
                "NumberOfFollowups": number_of_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": preferred_property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": number_of_trips,
                "Passport": 1 if passport == "Yes" else 0,
                "PitchSatisfactionScore": pitch_satisfaction_score,
                "OwnCar": 1 if own_car == "Yes" else 0,
                "NumberOfChildrenVisiting": number_of_children_visiting,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(
            f"✅ This customer is **likely to purchase** the Wellness Tourism "
            f"Package (probability: {probability:.1%})."
        )
    else:
        st.warning(
            f"❌ This customer is **unlikely to purchase** the Wellness Tourism "
            f"Package (probability: {probability:.1%})."
        )

    st.dataframe(input_df)
