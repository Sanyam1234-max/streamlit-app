import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load saved models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Custom CSS for better design and colors
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f7fb;  /* Soft light blue */
        }
        .stTitle {
            font-size: 36px;
            font-family: 'Arial', sans-serif;
            color: #343a40;  /* Dark gray for title */
        }
        .stText {
            font-size: 18px;
            font-family: 'Arial', sans-serif;
            color: #6c757d;  /* Light gray for text */
        }
        .stSubheader {
            font-size: 24px;
            font-family: 'Arial', sans-serif;
            color: #495057;  /* Slightly lighter gray for subheaders */
        }
        .stButton {
            background-color: #007bff;  /* Bright blue for buttons */
            color: white;
            border-radius: 5px;
        }
        .stButton:hover {
            background-color: #0056b3;  /* Darker blue on hover */
        }
        .stProgress {
            background-color: #28a745;  /* Green for progress bar */
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit page title and instructions
st.title("Mall Customers Segmentation")
st.write("""
This app allows you to input new customer data and predict their cluster based on existing models.
We will use Age, Annual Income, and Spending Score to classify the customer into one of the clusters.
""")

# Input: Age, Annual Income (k$), Spending Score (1-100)
st.subheader("Enter Customer Data")
age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Form input into a dataframe
input_data = pd.DataFrame({'Age': [age], 'Annual Income (k$)': [income], 'Spending Score (1-100)': [spending_score]})

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Show a progress bar while the prediction is being made
progress = st.progress(0)

# Apply PCA to the scaled data
input_data_pca = pca.transform(input_data_scaled)
progress.progress(50)

# Predict the cluster label
cluster_label = kmeans.predict(input_data_scaled)
progress.progress(100)

# Show the prediction result
st.subheader(f"Predicted Cluster: {cluster_label[0]}")

# Display the cluster centers and the predicted cluster
cluster_centers = kmeans.cluster_centers_
cluster_centers_pca = pca.transform(cluster_centers)

st.write("Cluster Centers (PCA-transformed):")
st.write(cluster_centers_pca)

# Visualization: Plot the clusters using Plotly for better interactivity
df = pd.read_csv('Mall_Customers.csv')  # Ensure to adjust the path
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Apply the same scaler and PCA transformation to the dataset
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# Perform KMeans clustering and assign the cluster labels
df['Cluster'] = kmeans.predict(X_scaled)

# Create DataFrame for PCA visualization
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

# Plot PCA-transformed data with clusters using Plotly
fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title='Customer Segments (PCA-reduced)', 
                 labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}, 
                 color_continuous_scale='Viridis')

fig.update_layout(
    plot_bgcolor='#f4f7fb',  # Soft light background for the plot
    title_font=dict(size=30, color='#343a40', family='Arial'),
    legend_title=dict(font=dict(size=18)),
    xaxis_title_font=dict(size=18, color='#495057'),
    yaxis_title_font=dict(size=18, color='#495057'),
    hoverlabel=dict(font_size=15, font_family='Arial')
)

# Show the plot
st.plotly_chart(fig)

# Optionally, provide some details about the cluster the customer belongs to
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.write(f"Average Characteristics of Cluster {cluster_label[0]}:")
st.write(cluster_summary.iloc[cluster_label[0]])

