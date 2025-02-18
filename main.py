# ctr_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression

# Generating a synthetic dataset with 10 numeric columns (binary or integers)
np.random.seed(42)

# DataFrame Creation with 10000 rows and 10 binary/integer columns
data_size = 10000
df = pd.DataFrame({
    'Ad_Impressions': np.random.randint(0, 2, data_size),  # Binary: Whether the ad was shown
    'Device_Type': np.random.randint(0, 2, data_size),     # Binary: 0 = Mobile, 1 = Desktop
    'Time_of_Day': np.random.randint(1, 10, data_size),     # Integer: Time of day (1-10 scale)
    'Ad_Category': np.random.randint(0, 2, data_size),      # Binary: 0 = Fashion, 1 = Electronics
    'Age_Group': np.random.randint(1, 10, data_size),       # Integer: Age group (1-10 scale)
    'User_Location': np.random.randint(0, 2, data_size),    # Binary: 0 = Rural, 1 = Urban
    'Feature_1': np.random.randint(1, 10, data_size),       # Integer: Some feature
    'Feature_2': np.random.randint(0, 2, data_size),        # Binary: Some feature
    'Feature_3': np.random.randint(1, 10, data_size),       # Integer: Some feature
    'Feature_4': np.random.randint(0, 2, data_size),        # Binary: Some feature
    'click': np.random.randint(0, 2, data_size)             # Target variable (binary: clicked or not)
})

# Data Preprocessing
# 'click' is the target variable (binary classification: 0 or 1)
y = df['click']

# Dropping the target variable 'click' from the features
X = df.drop(columns=['click'])

# Filling the missing values with the mean of the respective columns
X = X.fillna(X.mean())

# Splitting the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features using StandardScaler
scaler = StandardScaler()

# Fitting the scaler and transforming the data
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Buidling and Training Basic Models
# Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_scaled, y_train)

# Random Forest Classifier Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Bulding and Training Advanced Model
# LightGBM Model
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train_scaled, y_train)

# Evaluation of all models using Log Loss and ROC AUC
# Logistic Regression
y_pred_lr = log_reg_model.predict_proba(X_test_scaled)[:, 1]
logloss_lr = log_loss(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr)

# Random Forest Classifier
y_pred_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
logloss_rf = log_loss(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)

# LightGBM
y_pred_lgb = lgb_model.predict_proba(X_test_scaled)[:, 1]
logloss_lgb = log_loss(y_test, y_pred_lgb)
roc_auc_lgb = roc_auc_score(y_test, y_pred_lgb)

# Print evaluation metrics for Basic Models
print(f"Logistic Regression - Log Loss: {logloss_lr:.4f}, ROC AUC: {roc_auc_lr:.4f}")
print(f"Random Forest - Log Loss: {logloss_rf:.4f}, ROC AUC: {roc_auc_rf:.4f}")

# Print evaluation metrics for LightGBM
print(f"LightGBM - Log Loss: {logloss_lgb:.4f}, ROC AUC: {roc_auc_lgb:.4f}")

# Selection of the Best Model based on ROC AUC
best_model = None
if roc_auc_lr > roc_auc_rf and roc_auc_lr > roc_auc_lgb:
    best_model = log_reg_model
    best_model_name = 'Logistic Regression'
elif roc_auc_rf > roc_auc_lr and roc_auc_rf > roc_auc_lgb:
    best_model = rf_model
    best_model_name = 'Random Forest'
else:
    best_model = lgb_model
    best_model_name = 'LightGBM'

print(f"Best model selected: {best_model_name}")

# Retrain the best model using the scaled and transformed found
best_model.fit(X_train_scaled, y_train)

# Saving best model to a file
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Best model saved as 'best_model.pkl'")

# --- Streamlit Application Code ---
import streamlit as st

# Load the saved model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict the CTR
def predict_ctr(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict_proba(scaled_features)[:, 1]
    return prediction[0]*100

# Streamlit interface
st.title("CTR Prediction Application")
st.write("This application predicts the Click-Through Rate (CTR) based on user inputs.")

# Input form
with st.form(key='input_form'):
    # Binary Features
    ad_impressions = st.selectbox('Ad Impressions (0 = Not Shown, 1 = Shown)', [0, 1])
    device_type = st.selectbox('Device Type (0 = Mobile, 1 = Desktop)', [0, 1])
    time_of_day = st.selectbox('Time of Day (1 = Early Morning, 10 = Late Night)', list(range(1, 11)))
    ad_category = st.selectbox('Ad Category (0 = Fashion, 1 = Electronics)', [0, 1])
    age_group = st.selectbox('Age Group (1 = Young, 10 = Adult)', list(range(1, 11)))
    user_location = st.selectbox('User Location (0 = Rural, 1 = Urban)', [0, 1])

    # Integer Features
    num_clicks = st.number_input('Number of Clicks (1-10)', min_value=1, max_value=10, value=1)
    num_views = st.number_input('Number of Views (1-10)', min_value=1, max_value=10, value=1)
    session_duration = st.number_input('Session Duration (1-10)', min_value=1, max_value=10, value=1)
    ad_position = st.number_input('Ad Position (1-10)', min_value=1, max_value=10, value=1)

    submit_button = st.form_submit_button("Predict CTR")

if submit_button:
    input_features = [ad_impressions, device_type, time_of_day, ad_category, age_group, user_location,
                      num_clicks, num_views, session_duration, ad_position]
    ctr_prediction = predict_ctr(input_features)
    st.subheader("Predicted CTR Probability:")
    st.write(f"The predicted CTR is: {ctr_prediction:.2f}%")
