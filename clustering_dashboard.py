# wine_clustering_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Wine Clustering Dashboard", layout="wide")
st.title("🍷 Wine Dataset Clustering Dashboard")
st.write("Perform **KMeans**, **Hierarchical Clustering**, and **DBSCAN** on the Wine dataset with metrics and visualization.")

# Load Wine Dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Sidebar Controls
st.sidebar.header("Dashboard Settings")
menu = st.sidebar.radio("Select Section", ["View Dataset", "EDA", "Clustering"])

# --------------------------
# 1. View Dataset
# --------------------------
if menu == "View Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())
    st.subheader("🔍 Missing Values")
    st.write(df.isnull().sum())

# --------------------------
# 2. EDA
# --------------------------
elif menu == "EDA":
    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------------
# 3. Clustering
# --------------------------
elif menu == "Clustering":
    st.subheader("🧩 Select Clustering Algorithm")
    algorithm = st.selectbox("Algorithm", ["KMeans", "Hierarchical", "DBSCAN"])

    if algorithm == "KMeans":
        k = st.slider("Number of Clusters (k)", 2, 10, 3)
        X = df.values
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

    elif algorithm == "Hierarchical":
        X = df[['alcohol','magnesium']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        k = st.slider("Number of Clusters (for coloring)", 2, 10, 3)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)

    elif algorithm == "DBSCAN":
        X = df.values
        eps = st.slider("eps", 0.1, 5.0, 1.0)
        min_samples = st.slider("min_samples", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

    # Add labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels

    # Compute Metrics safely
    def safe_score(func, X, labels):
        try:
            return func(X, labels)
        except:
            return None

    sil_score = safe_score(silhouette_score, X, labels)
    db_score = safe_score(davies_bouldin_score, X, labels)
    ch_score = safe_score(calinski_harabasz_score, X, labels)

    # Display metrics
    st.subheader("📊 Clustering Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{sil_score:.3f}" if sil_score is not None else "N/A")
    col2.metric("Davies-Bouldin Index", f"{db_score:.3f}" if db_score is not None else "N/A")
    col3.metric("Calinski-Harabasz Index", f"{ch_score:.3f}" if ch_score is not None else "N/A")

    # Plot clusters
    st.subheader("📈 Cluster Visualization")
    if X.shape[1] >= 3:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap="Set2", s=50)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette="Set2", s=80, ax=ax)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig)
