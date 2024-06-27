import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
# Load the dataset
df = pd.read_csv(r'C:\Users\Dell\Desktop\Project-08\finel_clean_data.csv')
df = df.iloc[:2000]


# Convert 'Published_year_month' to datetime format
df['Published_year_month'] = pd.to_datetime(df['Published_year_month'])

# Streamlit app layout
st.title('Investigating Trends in the Remote Work Landscape')

# Add sections for data analysis and visualization

## Section 1: Overview of Remote Work Distribution
st.header('Overview of Remote Work Distribution')

# Pie chart to show proportion of remote vs. non-remote jobs
remote_counts = df['is_hourly'].value_counts()
fig1 = px.pie(values=remote_counts.values, names=remote_counts.index, title='Distribution of Remote vs. Non-Remote Jobs')
st.plotly_chart(fig1)

## Section 2: Remote Work Trends Over Time
st.header('Remote Work Trends Over Time')

# Line chart to show trends in remote work by month
remote_trends = df.groupby('Published_year_month')['is_hourly'].mean().reset_index()
fig2 = px.line(remote_trends, x='Published_year_month', y='is_hourly', 
               title='Proportion of Remote Jobs Over Time',
               labels={'is_hourly': 'Proportion of Remote Jobs', 'Published_year_month': 'Month'})
st.plotly_chart(fig2)

## Section 3: Remote Work Salary Comparison
st.header('Remote Work Salary Comparison')

# Box plot to compare salaries of remote vs. non-remote jobs
fig3 = px.box(df, x='is_hourly', y='Salary', color='is_hourly',
              title='Comparison of Salary Distribution for Remote vs. Non-Remote Jobs',
              labels={'is_hourly': 'Remote Job', 'Salary': 'Salary'})
st.plotly_chart(fig3)

## Section 4: Forecasting Remote Work Trends (Optional)
# This section can include forecasting models or trends analysis based on historical data.

# Add conclusion or summary
st.header('Conclusion')
st.markdown("""
The analysis reveals significant trends in the remote work landscape, with a growing proportion of jobs being remote. Further analysis could involve more granular insights into specific job roles or industries.
""")

# Optional: Add further details or insights based on the dataset

# Optional: Add disclaimer or sources of data

# Optional: Add contact information or team details for further inquiries


