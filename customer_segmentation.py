import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Generate or load a realistic customer dataset
def generate_customer_data():
    np.random.seed(42)  # For reproducibility
    num_customers = 1000
    data = {
        'customer_id': range(1, num_customers + 1),
        'annual_spending': np.random.normal(5000, 1500, num_customers),  # Annual spending in $
        'purchase_frequency': np.random.poisson(20, num_customers),      # Number of purchases per year
        'age': np.random.normal(35, 10, num_customers)                   # Customer age
    }
    df = pd.DataFrame(data)
    
    # Ensure non-negative values
    df['annual_spending'] = df['annual_spending'].clip(lower=500)
    df['purchase_frequency'] = df['purchase_frequency'].clip(lower=1)
    df['age'] = df['age'].clip(lower=18).round().astype(int)
    
    # Save the raw data
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv('data/customer_data.csv', index=False)
    return df

# Step 2: Preprocess data for clustering
def preprocess_data(df):
    # Select features for clustering
    features = ['annual_spending', 'purchase_frequency', 'age']
    X = df[features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features, scaler

# Step 3: Apply K-means clustering
def apply_clustering(X_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score to evaluate clustering quality
    silhouette = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score: {silhouette:.2f}")
    
    return clusters, kmeans

# Step 4: Visualize clusters and save plot
def visualize_clusters(df, clusters, features):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=features[0], y=features[1], hue=clusters, palette='viridis', size=features[2], sizes=(20, 200))
    plt.title('Customer Segments Based on Annual Spending, Purchase Frequency, and Age')
    plt.xlabel('Annual Spending ($)')
    plt.ylabel('Purchase Frequency')
    plt.legend(title='Cluster')
    plt.savefig('customer_segments_plot.png')
    plt.close()
    print("Cluster visualization saved as 'customer_segments_plot.png'.")

# Step 5: Prepare data for Tableau
def prepare_for_tableau(df, clusters):
    df['cluster'] = clusters
    df['cluster_label'] = df['cluster'].map({
        0: 'Low-Spending, Infrequent Buyers',
        1: 'High-Spending, Frequent Buyers',
        2: 'Moderate-Spending, Occasional Buyers'
    })
    df.to_csv('customer_segments_for_tableau.csv', index=False)
    print("Segmented data saved to 'customer_segments_for_tableau.csv' for Tableau import.")

# Main execution
if __name__ == "__main__":
    print("Starting Customer Segmentation Pipeline...")
    
    # Generate or load customer data
    print("Generating customer dataset...")
    customer_df = generate_customer_data()
    
    # Preprocess data
    print("Preprocessing data for clustering...")
    X_scaled, features, scaler = preprocess_data(customer_df)
    
    # Apply clustering
    print("Applying K-means clustering...")
    clusters, kmeans = apply_clustering(X_scaled, n_clusters=3)
    
    # Visualize clusters
    print("Visualizing clusters...")
    visualize_clusters(customer_df, clusters, features)
    
    # Prepare data for Tableau
    print("Preparing data for Tableau...")
    prepare_for_tableau(customer_df, clusters)
    
    print("Pipeline completed successfully!")