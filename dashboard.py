import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Wine Clustering Dashboard", layout="wide")

st.title("🍷 Wine Dataset Clustering Dashboard")
st.write("This dashboard performs **KMeans**, **Hierarchical Clustering**, and **DBSCAN** on the Wine dataset.")

# ------------------------------
# LOAD DATA
# ------------------------------
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# ------------------------------
# SIDEBAR MENU
# ------------------------------
menu = st.sidebar.radio(
    "Select Dashboard Section",
    ["View Dataset", "EDA", "KMeans Clustering", "Hierarchical Clustering", "DBSCAN"]
)

# =============================
# 1. VIEW DATASET
# =============================
if menu == "View Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    st.subheader("🔍 Missing Values")
    st.write(df.isnull().sum())

# =============================
# 2. EDA SECTION
# =============================
elif menu == "EDA":
    st.subheader("📈 EDA: Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =============================
# 3. KMEANS CLUSTERING
# =============================
elif menu == "KMeans Clustering":
    st.subheader("🎯 K-Means Clustering")

    # Select number of clusters
    k = st.slider("Select number of clusters (k)", 2, 10, 3)

    # Use only first 2 features for 2D visualization
    X = df.iloc[:, :2].values

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis")
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.7)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(f"K-Means Clustering with k = {k}")
    st.pyplot(fig)

# =============================
# 4. HIERARCHICAL CLUSTERING
# =============================
elif menu == "Hierarchical Clustering":
    st.subheader("📌 Hierarchical Clustering (Dendrogram)")

    data = df[['alcohol', 'magnesium']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    linked = linkage(scaled, method='ward')

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked)
    plt.title("Dendrogram (Ward Linkage)")
    st.pyplot(fig)

# =============================
# 5. DBSCAN SECTION
# =============================
elif menu == "DBSCAN":
    st.subheader("🌀 DBSCAN Clustering")

    X = df.iloc[:, :2].values

    eps = st.slider("Select eps", 0.1, 5.0, 1.0)
    min_samples = st.slider("Select min_samples", 2, 20, 5)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    plt.title("DBSCAN Clustering")
    st.pyplot(fig)

# dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.title("Clustering Evaluation Metrics Dashboard")

# Step 1: Sample Data
n_samples = st.slider("Number of Samples", 100, 1000, 300)
n_clusters = st.slider("Number of Clusters", 2, 10, 3)

X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=42)

# Step 2: Apply KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Step 3: Calculate Metrics
sil_score = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)

# Step 4: Display Metrics
st.subheader("Clustering Metrics")
st.write(f"**Silhouette Score:** {sil_score:.3f}")
st.write(f"**Davies-Bouldin Index:** {db_score:.3f}")
st.write(f"**Calinski-Harabasz Index:** {ch_score:.3f}")

# Step 5: Plot clusters
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df['Cluster'] = labels

st.subheader("Cluster Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Cluster", palette="Set2", ax=ax)
st.pyplot(fig)
# dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.title("🌟 Interactive Clustering Dashboard 🌟")

# Sidebar for controls
st.sidebar.header("Settings")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
n_features = st.sidebar.slider("Number of Features", 2, 3, 2)
n_clusters = st.sidebar.slider("Number of Clusters (for KMeans/Agglo)", 2, 10, 3)
algorithm = st.sidebar.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

# Generate synthetic data
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)
df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(n_features)])

# Apply clustering based on selected algorithm
if algorithm == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
elif algorithm == "DBSCAN":
    model = DBSCAN(eps=1.0, min_samples=5)
    labels = model.fit_predict(X)
elif algorithm == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

df['Cluster'] = labels

# Compute Metrics safely
def safe_score(func, X, labels):
    try:
        return func(X, labels)
    except:
        return None

sil_score = safe_score(silhouette_score, X, labels)
db_score = safe_score(davies_bouldin_score, X, labels)
ch_score = safe_score(calinski_harabasz_score, X, labels)

# Display metrics as cards (Professional UI)
st.subheader("📊 Clustering Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Silhouette Score", f"{sil_score:.3f}" if sil_score is not None else "N/A")
col2.metric("Davies-Bouldin Index", f"{db_score:.3f}" if db_score is not None else "N/A")
col3.metric("Calinski-Harabasz Index", f"{ch_score:.3f}" if ch_score is not None else "N/A")

# Plot clusters
st.subheader("📈 Cluster Visualization")
if n_features == 2:
    fig, ax = plt.subplots(figsize=(8,5))
    palette = sns.color_palette("Set2", len(df['Cluster'].unique()))
    sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Cluster", palette=palette, ax=ax, s=80)
    st.pyplot(fig)
elif n_features == 3:
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df["Feature 1"], df["Feature 2"], df["Feature 3"], c=df["Cluster"], cmap="Set2", s=50)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    st.pyplot(fig)

