import pytest
import numpy as np
import pandas as pd
from src.clustering.customer_segmentation import CustomerSegmentation

@pytest.fixture
def demo_df():
    np.random.seed(42)
    return pd.DataFrame({
        'visits_per_month': np.random.gamma(5, 1, 100),
        'avg_ticket': np.random.normal(1800, 400, 100).clip(300, 5000),
        'fuel_regular_share': np.random.beta(5, 2, 100),
        'fuel_diesel_share': np.random.beta(3, 4, 100),
        'services_shop': np.random.beta(3, 3, 100),
        'services_cafe': np.random.beta(2, 5, 100),
    })

def test_preprocess_data_shape(demo_df):
    seg = CustomerSegmentation()
    X = seg.preprocess_data(demo_df, demo_df.columns.tolist())
    assert X.shape[0] == demo_df.shape[0]

def test_kmeans_clustering(demo_df):
    seg = CustomerSegmentation()
    X = seg.preprocess_data(demo_df, demo_df.columns.tolist())
    model = seg.fit_kmeans(X, n_clusters=3)
    assert hasattr(model, 'labels_')
    labels = model.labels_
    assert len(labels) == demo_df.shape[0]
    assert len(np.unique(labels)) == 3

def test_optimal_clusters_kmeans(demo_df):
    seg = CustomerSegmentation()
    X = seg.preprocess_data(demo_df, demo_df.columns.tolist())
    optimal_k = seg.find_optimal_clusters(X, algorithm='kmeans', k_range=(2, 5))
    assert 2 <= optimal_k <= 5
