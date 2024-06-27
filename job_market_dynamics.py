import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv(r'C:\Users\Dell\Desktop\Project-08\finel_clean_data.csv')
df = df.iloc[:2000]

# Convert 'Published_year_month' to datetime format
df['Published_year_month'] = pd.to_datetime(df['Published_year_month'])

# Extract month and year for grouping
df['Month'] = df['Published_year_month'].dt.month
df['Year'] = df['Published_year_month'].dt.year

# Calculate average salary by month and country
average_salary = df.groupby(['country', 'Year', 'Month'])['Salary'].mean().reset_index()

# Streamlit app layout
st.title('Job Market Dynamics Dashboard')

# Add a dropdown for country selection
selected_country = st.selectbox('Select a Country', df['country'].unique())

# Filter data based on selected country
filtered_data = average_salary[average_salary['country'] == selected_country]

# Plot average salary trend
fig = px.line(filtered_data, x='Month', y='Salary', color='Year',
              labels={'Salary': 'Average Salary', 'Month': 'Month (1-12)'},
              title=f'Average Salary Trend in {selected_country}')
st.plotly_chart(fig)


