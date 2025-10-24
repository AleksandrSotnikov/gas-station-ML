"""Persona generator module for creating synthetic customer profiles."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import json

from src.models.customer_persona import (
    CustomerPersona, BehavioralAttributes, ContextualTriggers, DecisionRules,
    FuelType, WeatherCondition, TrafficLevel, DayOfWeek, Season
)
from config.settings import PERSONA_CONFIG


class PersonaGenerator:
    """Generator for creating synthetic customer personas from cluster data."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or PERSONA_CONFIG
        self.scaler = MinMaxScaler()
        self.personas = []
        self.cluster_profiles = None
        
    def generate_from_clusters(
        self, 
        cluster_data: pd.DataFrame,
        cluster_labels: np.ndarray,
        n_personas: Optional[int] = None
    ) -> List[CustomerPersona]:
        logger.info("Starting persona generation from clusters")
        
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
        if n_personas is None:
            n_personas = min(len(unique_clusters), self.config["max_personas"])
        
        logger.info(f"Generating {n_personas} personas from {len(unique_clusters)} clusters")
        
        # Create cluster profiles using only numeric columns
        self.cluster_profiles = self._create_cluster_profiles(cluster_data, cluster_labels)
        
        personas = []
        for cluster_id in unique_clusters[:n_personas]:
            if self._is_cluster_significant(cluster_id):
                persona = self._create_persona_from_cluster(cluster_id)
                if persona:
                    personas.append(persona)
        
        self.personas = personas
        logger.info(f"Generated {len(personas)} personas successfully")
        return personas
    
    def _create_cluster_profiles(
        self, 
        data: pd.DataFrame, 
        labels: np.ndarray
    ) -> pd.DataFrame:
        """Create statistical profiles for each cluster (numeric columns only)."""
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Use only numeric columns to avoid conversion errors
        numeric_cols = data_with_clusters.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'cluster']
        
        if not numeric_cols:
            raise ValueError("No numeric columns available for profiling. Please provide numeric features.")
        
        # Compute profiles and add explicit cluster sizes
        profiles = data_with_clusters.groupby('cluster')[numeric_cols].agg(['mean', 'std', 'median', 'min', 'max', 'count']).round(3)
        sizes = data_with_clusters.groupby('cluster').size().rename('cluster_size')
        
        # Combine profiles with sizes for easier access
        profiles[('meta', 'cluster_size')] = sizes
        return profiles
    
    def _is_cluster_significant(self, cluster_id: int) -> bool:
        if self.cluster_profiles is None:
            return False
        
        try:
            # Get cluster size from meta field
            size = int(self.cluster_profiles.loc[cluster_id][('meta', 'cluster_size')])
            return size >= self.config["min_cluster_size"]
        except Exception:
            return False
    
    def _create_persona_from_cluster(self, cluster_id: int) -> Optional[CustomerPersona]:
        try:
            cluster_stats = self.cluster_profiles.loc[cluster_id]
            
            persona_id = str(uuid.uuid4())
            name = self._generate_persona_name(cluster_id)
            description = self._generate_persona_description(cluster_id, cluster_stats)
            
            behavioral_attrs = self._generate_behavioral_attributes(cluster_stats)
            contextual_triggers = self._generate_contextual_triggers(cluster_stats)
            decision_rules = self._generate_decision_rules(cluster_stats)
            
            segment_size = int(cluster_stats[('meta', 'cluster_size')])
            confidence_score = self._calculate_confidence_score(cluster_stats)
            
            persona = CustomerPersona(
                persona_id=persona_id,
                name=name,
                description=description,
                behavioral_attributes=behavioral_attrs,
                contextual_triggers=contextual_triggers,
                decision_rules=decision_rules,
                segment_size=segment_size,
                confidence_score=confidence_score,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                metadata={'cluster_id': cluster_id}
            )
            
            logger.info(f"Created persona '{name}' from cluster {cluster_id}")
            return persona
            
        except Exception as e:
            logger.error(f"Failed to create persona from cluster {cluster_id}: {e}")
            return None

    # ... (остальной код без изменений) ...
