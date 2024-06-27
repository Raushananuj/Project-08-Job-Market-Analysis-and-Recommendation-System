import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace with your own dataset)
df = pd.read_csv(r'C:\Users\Dell\Desktop\Project-08\finel_clean_data.csv')
# Preprocess categorical variables
df['country_code'] = LabelEncoder().fit_transform(df['country'])
df['job_title_code'] = LabelEncoder().fit_transform(df['job_title'])

# Selecting features and target
X = df[['is_hourly', 'hourly_low', 'hourly_high', 'country_code', 'job_title_code']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary:')

is_hourly = st.checkbox('Is hourly job?')
hourly_low = st.number_input('Hourly low:', min_value=0.0, max_value=None, step=1.0)
hourly_high = st.number_input('Hourly high:', min_value=0.0, max_value=None, step=1.0)
country = st.selectbox('Country:', df['country'].unique())
job_title = st.selectbox('Job Title:', df['job_title'].unique())

country_code = LabelEncoder().fit_transform([country])[0]
job_title_code = LabelEncoder().fit_transform([job_title])[0]

input_data = [[is_hourly, hourly_low, hourly_high, country_code, job_title_code]]

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Salary: ${prediction[0]:,.2f}')
