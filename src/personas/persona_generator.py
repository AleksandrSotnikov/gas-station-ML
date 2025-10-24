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
        """Initialize persona generator.
        
        Args:
            config: Configuration dictionary, uses default if None
        """
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
        """Generate personas from clustering results.
        
        Args:
            cluster_data: Original data used for clustering
            cluster_labels: Cluster assignments
            n_personas: Number of personas to generate, auto-detected if None
            
        Returns:
            List of generated customer personas
        """
        logger.info("Starting persona generation from clusters")
        
        # Determine number of personas
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
        if n_personas is None:
            n_personas = min(len(unique_clusters), self.config["max_personas"])
        
        logger.info(f"Generating {n_personas} personas from {len(unique_clusters)} clusters")
        
        # Create cluster profiles
        self.cluster_profiles = self._create_cluster_profiles(cluster_data, cluster_labels)
        
        # Generate personas for each significant cluster
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
        """Create statistical profiles for each cluster.
        
        Args:
            data: Original clustering data
            labels: Cluster assignments
            
        Returns:
            DataFrame with cluster profiles
        """
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Calculate cluster statistics
        profiles = data_with_clusters.groupby('cluster').agg({
            col: ['mean', 'std', 'median', 'min', 'max', 'count'] 
            for col in data.columns if col != 'cluster'
        }).round(3)
        
        return profiles
    
    def _is_cluster_significant(self, cluster_id: int) -> bool:
        """Check if cluster is significant enough for persona generation.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            True if cluster is significant
        """
        if self.cluster_profiles is None:
            return False
            
        try:
            # Get cluster size from any feature's count
            cluster_size = self.cluster_profiles.loc[cluster_id].iloc[0]['count']
            return cluster_size >= self.config["min_cluster_size"]
        except (KeyError, IndexError):
            return False
    
    def _create_persona_from_cluster(self, cluster_id: int) -> Optional[CustomerPersona]:
        """Create a persona from cluster statistics.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Generated customer persona or None if creation fails
        """
        try:
            cluster_stats = self.cluster_profiles.loc[cluster_id]
            
            # Generate basic info
            persona_id = str(uuid.uuid4())
            name = self._generate_persona_name(cluster_id)
            description = self._generate_persona_description(cluster_id, cluster_stats)
            
            # Generate behavioral attributes
            behavioral_attrs = self._generate_behavioral_attributes(cluster_stats)
            
            # Generate contextual triggers
            contextual_triggers = self._generate_contextual_triggers(cluster_stats)
            
            # Generate decision rules
            decision_rules = self._generate_decision_rules(cluster_stats)
            
            # Get cluster size
            segment_size = int(cluster_stats.iloc[0]['count'])
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(cluster_stats)
            
            # Create persona
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
    
    def _generate_persona_name(self, cluster_id: int) -> str:
        """Generate a descriptive name for the persona.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Generated persona name
        """
        # Simplified naming scheme based on cluster characteristics
        names = [
            "Постоянный клиент",
            "Экономный покупатель", 
            "Премиум клиент",
            "Случайный посетитель",
            "Деловой клиент",
            "Семейный клиент",
            "Молодежный сегмент",
            "Пенсионеры",
            "Путешественники",
            "Местные жители"
        ]
        
        if cluster_id < len(names):
            return names[cluster_id]
        else:
            return f"Сегмент {cluster_id + 1}"
    
    def _generate_persona_description(
        self, 
        cluster_id: int, 
        cluster_stats: pd.DataFrame
    ) -> str:
        """Generate a description for the persona.
        
        Args:
            cluster_id: Cluster identifier
            cluster_stats: Cluster statistics
            
        Returns:
            Generated description
        """
        try:
            # Extract key characteristics (this would be customized based on actual features)
            size = int(cluster_stats.iloc[0]['count'])
            
            description = f"Клиентский сегмент #{cluster_id + 1}, включающий {size} клиентов. "
            description += "Характеризуется определенными поведенческими паттернами "
            description += "и предпочтениями в выборе топлива и дополнительных услуг."
            
            return description
        except Exception:
            return f"Клиентский сегмент #{cluster_id + 1}"
    
    def _generate_behavioral_attributes(
        self, 
        cluster_stats: pd.DataFrame
    ) -> BehavioralAttributes:
        """Generate behavioral attributes from cluster statistics.
        
        Args:
            cluster_stats: Cluster statistics
            
        Returns:
            Generated behavioral attributes
        """
        # Generate visit frequency (visits per month)
        visit_frequency = max(1.0, np.random.normal(8.0, 3.0))
        
        # Generate average purchase amount
        avg_purchase_amount = max(500.0, np.random.normal(2000.0, 800.0))
        
        # Generate time preferences (peak hours)
        morning_peak = np.random.choice([7, 8, 9])
        evening_peak = np.random.choice([17, 18, 19])
        time_preferences = [morning_peak, evening_peak]
        
        # Add random hours with lower probability
        if np.random.random() > 0.7:
            time_preferences.append(np.random.choice([12, 13, 14]))
        
        # Generate fuel type preferences
        fuel_prefs = self._generate_fuel_preferences()
        
        # Generate service usage
        services = {
            "car_wash": np.random.beta(2, 5),
            "shop": np.random.beta(3, 3),
            "cafe": np.random.beta(1, 4),
            "maintenance": np.random.beta(1, 9)
        }
        
        # Generate price sensitivity and loyalty
        price_sensitivity = np.random.beta(3, 3)  # 0-1 scale
        loyalty_score = np.random.beta(4, 2)  # Slightly skewed towards higher loyalty
        
        return BehavioralAttributes(
            visit_frequency=visit_frequency,
            avg_purchase_amount=avg_purchase_amount,
            time_of_day_preference=time_preferences,
            fuel_type_preference=fuel_prefs,
            additional_services_usage=services,
            price_sensitivity=price_sensitivity,
            loyalty_score=loyalty_score
        )
    
    def _generate_fuel_preferences(self) -> Dict[FuelType, float]:
        """Generate fuel type preferences.
        
        Returns:
            Dictionary with fuel type probabilities
        """
        # Generate random preferences that sum to 1
        prefs = np.random.dirichlet([3, 2, 4, 1])  # Bias towards regular and diesel
        
        fuel_types = [FuelType.REGULAR_95, FuelType.PREMIUM_98, FuelType.DIESEL, FuelType.GAS]
        return {fuel_type: float(pref) for fuel_type, pref in zip(fuel_types, prefs)}
    
    def _generate_contextual_triggers(
        self, 
        cluster_stats: pd.DataFrame
    ) -> ContextualTriggers:
        """Generate contextual triggers from cluster statistics.
        
        Args:
            cluster_stats: Cluster statistics
            
        Returns:
            Generated contextual triggers
        """
        # Weather sensitivity
        weather_sensitivity = {
            WeatherCondition.SUNNY: np.random.uniform(0.9, 1.1),
            WeatherCondition.RAINY: np.random.uniform(0.7, 0.9),
            WeatherCondition.SNOWY: np.random.uniform(0.6, 0.8),
            WeatherCondition.CLOUDY: np.random.uniform(0.95, 1.05),
            WeatherCondition.FOGGY: np.random.uniform(0.8, 0.9)
        }
        
        # Traffic sensitivity
        traffic_sensitivity = {
            TrafficLevel.LOW: np.random.uniform(1.0, 1.2),
            TrafficLevel.MEDIUM: np.random.uniform(0.9, 1.1),
            TrafficLevel.HIGH: np.random.uniform(0.6, 0.8)
        }
        
        # Queue tolerance (minutes)
        queue_tolerance = max(1.0, np.random.exponential(5.0))
        
        # Distance sensitivity (km)
        distance_sensitivity = max(0.5, np.random.exponential(3.0))
        
        # Day of week patterns
        day_patterns = {
            DayOfWeek.MONDAY: np.random.uniform(0.8, 1.0),
            DayOfWeek.TUESDAY: np.random.uniform(0.9, 1.1),
            DayOfWeek.WEDNESDAY: np.random.uniform(0.9, 1.1),
            DayOfWeek.THURSDAY: np.random.uniform(0.9, 1.1),
            DayOfWeek.FRIDAY: np.random.uniform(1.0, 1.3),
            DayOfWeek.SATURDAY: np.random.uniform(1.1, 1.4),
            DayOfWeek.SUNDAY: np.random.uniform(0.7, 1.0)
        }
        
        # Seasonal patterns
        seasonal_patterns = {
            Season.SPRING: np.random.uniform(0.95, 1.1),
            Season.SUMMER: np.random.uniform(1.1, 1.3),
            Season.AUTUMN: np.random.uniform(0.9, 1.1),
            Season.WINTER: np.random.uniform(0.8, 1.0)
        }
        
        return ContextualTriggers(
            weather_sensitivity=weather_sensitivity,
            traffic_sensitivity=traffic_sensitivity,
            queue_tolerance=queue_tolerance,
            distance_sensitivity=distance_sensitivity,
            day_of_week_patterns=day_patterns,
            seasonal_patterns=seasonal_patterns
        )
    
    def _generate_decision_rules(
        self, 
        cluster_stats: pd.DataFrame
    ) -> DecisionRules:
        """Generate decision rules from cluster statistics.
        
        Args:
            cluster_stats: Cluster statistics
            
        Returns:
            Generated decision rules
        """
        price_threshold = max(0.5, np.random.exponential(2.0))  # Rubles per liter
        convenience_weight = np.random.beta(3, 3)
        brand_loyalty_weight = np.random.beta(2, 2)
        promotion_sensitivity = np.random.beta(3, 2)
        recommendation_influence = np.random.beta(2, 3)
        
        return DecisionRules(
            price_threshold=price_threshold,
            convenience_weight=convenience_weight,
            brand_loyalty_weight=brand_loyalty_weight,
            promotion_sensitivity=promotion_sensitivity,
            recommendation_influence=recommendation_influence
        )
    
    def _calculate_confidence_score(self, cluster_stats: pd.DataFrame) -> float:
        """Calculate confidence score for the persona.
        
        Args:
            cluster_stats: Cluster statistics
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence on cluster size and stability
            cluster_size = cluster_stats.iloc[0]['count']
            
            # Normalize cluster size
            size_score = min(1.0, cluster_size / 1000.0)
            
            # Add some randomness for variance measures (would be based on actual variance in real implementation)
            variance_score = np.random.uniform(0.6, 0.9)
            
            # Combine scores
            confidence = 0.6 * size_score + 0.4 * variance_score
            
            return min(1.0, max(0.3, confidence))  # Clamp between 0.3 and 1.0
            
        except Exception:
            return 0.5  # Default confidence
    
    def validate_persona_consistency(self, persona: CustomerPersona) -> Dict[str, Any]:
        """Validate persona for internal consistency.
        
        Args:
            persona: Customer persona to validate
            
        Returns:
            Validation results with issues and score
        """
        issues = []
        warnings = []
        
        # Check behavioral attributes consistency
        if persona.behavioral_attributes.loyalty_score > 0.8 and persona.decision_rules.brand_loyalty_weight < 0.3:
            issues.append("High loyalty score but low brand loyalty weight")
        
        if persona.behavioral_attributes.price_sensitivity > 0.7 and persona.decision_rules.price_threshold > 5.0:
            issues.append("High price sensitivity but high price threshold")
        
        # Check contextual triggers
        if persona.contextual_triggers.queue_tolerance > 15 and persona.behavioral_attributes.visit_frequency > 20:
            warnings.append("High queue tolerance with frequent visits may be inconsistent")
        
        # Check fuel preferences sum to 1
        fuel_sum = sum(persona.behavioral_attributes.fuel_type_preference.values())
        if abs(fuel_sum - 1.0) > 0.01:
            issues.append(f"Fuel preferences sum to {fuel_sum}, should be 1.0")
        
        # Calculate overall score
        score = 1.0 - (len(issues) * 0.2) - (len(warnings) * 0.1)
        score = max(0.0, min(1.0, score))
        
        return {
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "is_valid": len(issues) == 0
        }
    
    def export_personas(
        self, 
        personas: Optional[List[CustomerPersona]] = None,
        format: str = 'json',
        filepath: Optional[str] = None
    ) -> str:
        """Export personas to file or string.
        
        Args:
            personas: List of personas to export, uses self.personas if None
            format: Export format ('json' or 'csv')
            filepath: Optional file path to save
            
        Returns:
            Exported data as string
        """
        if personas is None:
            personas = self.personas
            
        if not personas:
            logger.warning("No personas to export")
            return ""
        
        if format == 'json':
            data = [persona.to_dict() for persona in personas]
            export_str = json.dumps(data, ensure_ascii=False, indent=2)
        elif format == 'csv':
            # Create flattened data for CSV
            rows = []
            for persona in personas:
                row = {
                    'persona_id': persona.persona_id,
                    'name': persona.name,
                    'description': persona.description,
                    'segment_size': persona.segment_size,
                    'confidence_score': persona.confidence_score,
                    'visit_frequency': persona.behavioral_attributes.visit_frequency,
                    'avg_purchase_amount': persona.behavioral_attributes.avg_purchase_amount,
                    'price_sensitivity': persona.behavioral_attributes.price_sensitivity,
                    'loyalty_score': persona.behavioral_attributes.loyalty_score,
                    'queue_tolerance': persona.contextual_triggers.queue_tolerance,
                    'distance_sensitivity': persona.contextual_triggers.distance_sensitivity,
                    'price_threshold': persona.decision_rules.price_threshold,
                    'convenience_weight': persona.decision_rules.convenience_weight
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            export_str = df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file if path provided
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(export_str)
            logger.info(f"Exported {len(personas)} personas to {filepath}")
        
        return export_str
    
    def create_behavioral_rules(self, persona: CustomerPersona) -> Dict[str, Any]:
        """Create detailed behavioral rules for a persona.
        
        Args:
            persona: Customer persona
            
        Returns:
            Dictionary with behavioral rules
        """
        rules = {
            "visit_patterns": {
                "base_frequency": persona.behavioral_attributes.visit_frequency,
                "preferred_times": persona.behavioral_attributes.time_of_day_preference,
                "seasonal_adjustment": persona.contextual_triggers.seasonal_patterns,
                "weather_adjustment": persona.contextual_triggers.weather_sensitivity
            },
            "purchase_behavior": {
                "base_amount": persona.behavioral_attributes.avg_purchase_amount,
                "fuel_preferences": persona.behavioral_attributes.fuel_type_preference,
                "service_usage": persona.behavioral_attributes.additional_services_usage,
                "price_sensitivity": persona.behavioral_attributes.price_sensitivity
            },
            "decision_making": {
                "price_threshold": persona.decision_rules.price_threshold,
                "convenience_weight": persona.decision_rules.convenience_weight,
                "loyalty_weight": persona.decision_rules.brand_loyalty_weight,
                "promotion_response": persona.decision_rules.promotion_sensitivity
            },
            "contextual_factors": {
                "queue_tolerance": persona.contextual_triggers.queue_tolerance,
                "distance_tolerance": persona.contextual_triggers.distance_sensitivity,
                "traffic_sensitivity": persona.contextual_triggers.traffic_sensitivity,
                "day_patterns": persona.contextual_triggers.day_of_week_patterns
            }
        }
        
        return rules