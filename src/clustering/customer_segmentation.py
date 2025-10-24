"""Customer segmentation module using various clustering algorithms."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from config.settings import ML_CONFIG, PERSONA_CONFIG


class CustomerSegmentation:
    """Customer segmentation using multiple clustering algorithms."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize customer segmentation.
        
        Args:
            config: Configuration dictionary, uses default if None
        """
        self.config = config or ML_CONFIG
        self.scaler = None
        self.pca = None
        self.models = {}
        self.feature_names = []
        self.cluster_labels = {}
        self.cluster_metrics = {}
        
    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        features: List[str], 
        scaling_method: str = "standard"
    ) -> np.ndarray:
        """Preprocess data for clustering.
        
        Args:
            data: Input dataframe
            features: List of feature column names
            scaling_method: Scaling method ('standard' or 'robust')
            
        Returns:
            Preprocessed feature matrix
        """
        logger.info(f"Preprocessing data with {len(features)} features")
        
        # Select features
        feature_data = data[features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.median())
        
        # Scale features
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
            
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Apply PCA if configured
        pca_components = self.config["feature_engineering"].get("pca_components")
        if pca_components and pca_components < 1.0:
            self.pca = PCA(n_components=pca_components, random_state=42)
            scaled_data = self.pca.fit_transform(scaled_data)
            logger.info(f"Applied PCA, reduced to {scaled_data.shape[1]} components")
            
        self.feature_names = features
        return scaled_data
    
    def find_optimal_clusters(
        self, 
        data: np.ndarray, 
        algorithm: str = "kmeans",
        k_range: Tuple[int, int] = (2, 10)
    ) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            data: Preprocessed feature matrix
            algorithm: Clustering algorithm ('kmeans', 'hierarchical')
            k_range: Range of k values to test
            
        Returns:
            Optimal number of clusters
        """
        logger.info(f"Finding optimal clusters for {algorithm}")
        
        k_values = range(k_range[0], k_range[1] + 1)
        inertias = []
        silhouette_scores = []
        
        for k in k_values:
            if algorithm == "kmeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif algorithm == "hierarchical":
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            else:
                raise ValueError(f"Algorithm {algorithm} not supported for optimal k search")
                
            labels = model.fit_predict(data)
            
            if algorithm == "kmeans":
                inertias.append(model.inertia_)
            
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(data, labels))
            else:
                silhouette_scores.append(0)
        
        # Find elbow point (for k-means)
        if algorithm == "kmeans" and len(inertias) > 2:
            # Calculate rate of change
            diff1 = np.diff(inertias)
            diff2 = np.diff(diff1)
            elbow_k = k_values[np.argmax(diff2) + 2] if len(diff2) > 0 else k_values[len(k_values)//2]
        else:
            elbow_k = k_values[len(k_values)//2]
        
        # Find best silhouette score
        best_silhouette_k = k_values[np.argmax(silhouette_scores)]
        
        # Choose based on silhouette score if significantly better
        optimal_k = best_silhouette_k if max(silhouette_scores) > 0.3 else elbow_k
        
        logger.info(f"Optimal k: {optimal_k} (elbow: {elbow_k}, silhouette: {best_silhouette_k})")
        return optimal_k
    
    def fit_kmeans(
        self, 
        data: np.ndarray, 
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> KMeans:
        """Fit K-means clustering model.
        
        Args:
            data: Preprocessed feature matrix
            n_clusters: Number of clusters, auto-detected if None
            **kwargs: Additional parameters for KMeans
            
        Returns:
            Fitted KMeans model
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(data, "kmeans")
            
        config = self.config["clustering"]["kmeans"].copy()
        config.update(kwargs)
        config["n_clusters"] = n_clusters
        
        model = KMeans(**config)
        model.fit(data)
        
        self.models["kmeans"] = model
        self.cluster_labels["kmeans"] = model.labels_
        
        # Calculate metrics
        self._calculate_metrics(data, model.labels_, "kmeans")
        
        logger.info(f"K-means fitted with {n_clusters} clusters")
        return model
    
    def fit_dbscan(
        self, 
        data: np.ndarray, 
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        **kwargs
    ) -> DBSCAN:
        """Fit DBSCAN clustering model.
        
        Args:
            data: Preprocessed feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            **kwargs: Additional parameters for DBSCAN
            
        Returns:
            Fitted DBSCAN model
        """
        config = self.config["clustering"]["dbscan"].copy()
        if eps is not None:
            config["eps"] = eps
        if min_samples is not None:
            config["min_samples"] = min_samples
        config.update(kwargs)
        
        model = DBSCAN(**config)
        labels = model.fit_predict(data)
        
        self.models["dbscan"] = model
        self.cluster_labels["dbscan"] = labels
        
        # Calculate metrics (excluding noise points)
        if len(np.unique(labels)) > 1:
            self._calculate_metrics(data, labels, "dbscan")
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"DBSCAN fitted with {n_clusters} clusters and {n_noise} noise points")
        
        return model
    
    def fit_hierarchical(
        self, 
        data: np.ndarray, 
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> AgglomerativeClustering:
        """Fit Hierarchical clustering model.
        
        Args:
            data: Preprocessed feature matrix
            n_clusters: Number of clusters, auto-detected if None
            **kwargs: Additional parameters for AgglomerativeClustering
            
        Returns:
            Fitted AgglomerativeClustering model
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(data, "hierarchical")
            
        config = self.config["clustering"]["hierarchical"].copy()
        config.update(kwargs)
        config["n_clusters"] = n_clusters
        
        model = AgglomerativeClustering(**config)
        labels = model.fit_predict(data)
        
        self.models["hierarchical"] = model
        self.cluster_labels["hierarchical"] = labels
        
        # Calculate metrics
        self._calculate_metrics(data, labels, "hierarchical")
        
        logger.info(f"Hierarchical clustering fitted with {n_clusters} clusters")
        return model
    
    def _calculate_metrics(self, data: np.ndarray, labels: np.ndarray, algorithm: str):
        """Calculate clustering metrics.
        
        Args:
            data: Feature matrix
            labels: Cluster labels
            algorithm: Algorithm name
        """
        # Filter out noise points for DBSCAN
        if algorithm == "dbscan":
            mask = labels != -1
            if mask.sum() < 2:
                return
            data_filtered = data[mask]
            labels_filtered = labels[mask]
        else:
            data_filtered = data
            labels_filtered = labels
        
        unique_labels = np.unique(labels_filtered)
        if len(unique_labels) < 2:
            return
            
        metrics = {
            "silhouette_score": silhouette_score(data_filtered, labels_filtered),
            "calinski_harabasz_score": calinski_harabasz_score(data_filtered, labels_filtered),
            "davies_bouldin_score": davies_bouldin_score(data_filtered, labels_filtered),
            "n_clusters": len(unique_labels)
        }
        
        if algorithm == "dbscan":
            metrics["n_noise"] = (labels == -1).sum()
            
        self.cluster_metrics[algorithm] = metrics
        
        logger.info(f"{algorithm} metrics: {metrics}")
    
    def compare_algorithms(self, data: np.ndarray) -> Dict[str, Any]:
        """Compare different clustering algorithms.
        
        Args:
            data: Preprocessed feature matrix
            
        Returns:
            Comparison results
        """
        logger.info("Comparing clustering algorithms")
        
        # Fit all algorithms
        self.fit_kmeans(data)
        self.fit_dbscan(data)
        self.fit_hierarchical(data)
        
        # Create comparison summary
        comparison = {
            "metrics": self.cluster_metrics,
            "best_algorithm": self._select_best_algorithm()
        }
        
        return comparison
    
    def _select_best_algorithm(self) -> str:
        """Select best algorithm based on metrics.
        
        Returns:
            Name of best algorithm
        """
        scores = {}
        
        for alg, metrics in self.cluster_metrics.items():
            # Normalize metrics (higher silhouette and calinski_harabasz are better, lower davies_bouldin is better)
            score = 0
            if "silhouette_score" in metrics:
                score += metrics["silhouette_score"] * 0.4
            if "calinski_harabasz_score" in metrics:
                # Normalize by dividing by 1000 (rough normalization)
                score += min(metrics["calinski_harabasz_score"] / 1000, 1.0) * 0.3
            if "davies_bouldin_score" in metrics:
                # Invert davies_bouldin (lower is better)
                score += (1.0 / (1.0 + metrics["davies_bouldin_score"])) * 0.3
                
            scores[alg] = score
        
        best_algorithm = max(scores.keys(), key=lambda k: scores[k])
        logger.info(f"Best algorithm: {best_algorithm} (score: {scores[best_algorithm]:.3f})")
        
        return best_algorithm
    
    def get_cluster_profiles(
        self, 
        data: pd.DataFrame, 
        algorithm: str = "kmeans"
    ) -> pd.DataFrame:
        """Get cluster profiles with statistics.
        
        Args:
            data: Original dataframe with features
            algorithm: Algorithm to use for profiles
            
        Returns:
            Dataframe with cluster profiles
        """
        if algorithm not in self.cluster_labels:
            raise ValueError(f"Algorithm {algorithm} not fitted yet")
            
        labels = self.cluster_labels[algorithm]
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Calculate cluster statistics
        profiles = data_with_clusters.groupby('cluster').agg({
            feature: ['mean', 'std', 'median', 'count'] 
            for feature in self.feature_names
        }).round(3)
        
        return profiles
    
    def predict_cluster(
        self, 
        data: np.ndarray, 
        algorithm: str = "kmeans"
    ) -> np.ndarray:
        """Predict cluster for new data.
        
        Args:
            data: New data points
            algorithm: Algorithm to use for prediction
            
        Returns:
            Predicted cluster labels
        """
        if algorithm not in self.models:
            raise ValueError(f"Algorithm {algorithm} not fitted yet")
            
        model = self.models[algorithm]
        
        # Scale the data
        scaled_data = self.scaler.transform(data)
        
        # Apply PCA if used
        if self.pca is not None:
            scaled_data = self.pca.transform(scaled_data)
            
        # Predict clusters
        if algorithm == "kmeans":
            return model.predict(scaled_data)
        elif algorithm in ["dbscan", "hierarchical"]:
            # For these algorithms, find closest cluster center
            return self._predict_closest_cluster(scaled_data, algorithm)
        else:
            raise ValueError(f"Prediction not implemented for {algorithm}")
    
    def _predict_closest_cluster(
        self, 
        data: np.ndarray, 
        algorithm: str
    ) -> np.ndarray:
        """Predict cluster by finding closest cluster center.
        
        Args:
            data: Scaled data points
            algorithm: Algorithm name
            
        Returns:
            Predicted cluster labels
        """
        # This is a simplified approach - in practice, you might want to retrain
        # or use more sophisticated methods for DBSCAN/Hierarchical prediction
        labels = self.cluster_labels[algorithm]
        unique_labels = np.unique(labels[labels != -1])  # Exclude noise
        
        # Calculate cluster centers from training data
        # Note: This assumes you have access to training data
        # In practice, you'd save these centers during training
        
        predictions = []
        for point in data:
            # Simple approach: assign to cluster 0 as placeholder
            # This should be replaced with proper prediction logic
            predictions.append(unique_labels[0] if len(unique_labels) > 0 else 0)
            
        return np.array(predictions)