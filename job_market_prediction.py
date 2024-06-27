import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset (example data)
df = pd.read_csv(r'C:\Users\Dell\Desktop\Project-08\finel_clean_data.csv')
df = df.iloc[:50000]

# Convert 'Published_year_month' to datetime format
df['Published_year_month'] = pd.to_datetime(df['Published_year_month'])

# Encode categorical variables
label_encoder_country = LabelEncoder()
label_encoder_job_title = LabelEncoder()

df['country'] = label_encoder_country.fit_transform(df['country'])
df['job_title'] = label_encoder_job_title.fit_transform(df['job_title'])

# Feature engineering - example: adding a feature for year and month
df['Year'] = df['Published_year_month'].dt.year
df['Month'] = df['Published_year_month'].dt.month

# Train a simple linear regression model (example)
X = df[['country', 'job_title', 'Year', 'Month']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# Streamlit app layout
st.title('Predicting Future Job Market Trends')

# Add sections for predictive analytics and visualization

## Section 1: Future Scenario Analysis
st.header('Future Scenario Analysis')

# Example prediction for future scenario
future_country = 'United States'
future_job_title = 'Full Stack'
future_year = 2025
future_month = 6

# Transform input data for prediction
future_country_encoded = label_encoder_country.transform([future_country])[0]
future_job_title_encoded = label_encoder_job_title.transform([future_job_title])[0]

# Create dataframe for prediction
future_data = pd.DataFrame({
    'country': [future_country_encoded],
    'job_title': [future_job_title_encoded],
    'Year': [future_year],
    'Month': [future_month]
})

# Predict salary
predicted_salary = model.predict(future_data)[0]
st.write(f'Predicted Salary for {future_job_title} job in {future_country} in {future_month}/{future_year}: ${predicted_salary:.2f}')

# Add conclusion or summary
st.header('Conclusion')
st.markdown("""
The predictive model suggests potential trends and future scenarios in the job market based on historical data and predictions. Further refinements and data enhancements can improve the accuracy and reliability of predictions.
""")