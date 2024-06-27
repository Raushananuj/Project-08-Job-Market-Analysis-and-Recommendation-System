import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
file_path = r'C:\Users\Dell\Desktop\Project-08\finel_clean_data.csv'
job_postings = pd.read_csv(file_path)
job_postings = job_postings.iloc[:10000]

# Drop rows with NaN values in 'job_title' column
job_postings.dropna(subset=['job_title'], inplace=True)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
job_titles_tfidf = vectorizer.fit_transform(job_postings['job_title'])

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(job_titles_tfidf, job_titles_tfidf)

# Function to recommend jobs
def get_recommendations(job_title, cosine_sim, job_postings):
    idx = job_postings.index[job_postings['job_title'] == job_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 similar job postings
    job_indices = [i[0] for i in sim_scores]
    return job_postings.iloc[job_indices]

# Streamlit app
def main():
    st.title('Job Recommendation Engine')
    st.sidebar.title('Job Recommendations')
    
    job_titles = job_postings['job_title'].tolist()
    selected_job = st.sidebar.selectbox('Select a job title:', job_titles)
    
    if st.sidebar.button('Show Recommendations'):
        recommendations = get_recommendations(selected_job, cosine_sim, job_postings)
        st.subheader('Recommended Jobs:')
        st.table(recommendations[['job_title', 'country', 'Salary']])
    
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Objective:** To develop a personalized job recommendation engine.')
    st.sidebar.markdown('**Deliverables:** A working prototype of the recommendation engine, API documentation, and a user interface for interaction.')

if __name__ == '__main__':
    main()