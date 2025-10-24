"""Customer persona model for gas station ML project."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class FuelType(Enum):
    """Enum for fuel types."""
    REGULAR_95 = "regular_95"
    PREMIUM_98 = "premium_98"
    DIESEL = "diesel"
    GAS = "gas"


class WeatherCondition(Enum):
    """Enum for weather conditions."""
    SUNNY = "sunny"
    RAINY = "rainy"
    SNOWY = "snowy"
    CLOUDY = "cloudy"
    FOGGY = "foggy"


class TrafficLevel(Enum):
    """Enum for traffic levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DayOfWeek(Enum):
    """Enum for days of week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class Season(Enum):
    """Enum for seasons."""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


@dataclass
class BehavioralAttributes:
    """Behavioral attributes of a customer persona."""
    visit_frequency: float  # visits per month
    avg_purchase_amount: float  # average purchase in rubles
    time_of_day_preference: List[int]  # preferred hours (0-23)
    fuel_type_preference: Dict[FuelType, float]  # probability distribution
    additional_services_usage: Dict[str, float]  # service -> usage probability
    price_sensitivity: float  # 0-1 scale, 1 = very sensitive
    loyalty_score: float  # 0-1 scale, 1 = very loyal
    
    def __post_init__(self):
        """Validate behavioral attributes."""
        assert 0 <= self.price_sensitivity <= 1, "Price sensitivity must be between 0 and 1"
        assert 0 <= self.loyalty_score <= 1, "Loyalty score must be between 0 and 1"
        assert all(0 <= hour <= 23 for hour in self.time_of_day_preference), "Hours must be 0-23"
        assert abs(sum(self.fuel_type_preference.values()) - 1.0) < 0.01, "Fuel preferences must sum to 1"


@dataclass
class ContextualTriggers:
    """Contextual triggers that influence customer behavior."""
    weather_sensitivity: Dict[WeatherCondition, float]  # weather -> behavior modifier
    traffic_sensitivity: Dict[TrafficLevel, float]  # traffic -> behavior modifier
    queue_tolerance: float  # max acceptable queue length
    distance_sensitivity: float  # willingness to travel distance
    day_of_week_patterns: Dict[DayOfWeek, float]  # day -> activity modifier
    seasonal_patterns: Dict[Season, float]  # season -> activity modifier
    
    def __post_init__(self):
        """Validate contextual triggers."""
        assert self.queue_tolerance >= 0, "Queue tolerance must be non-negative"
        assert self.distance_sensitivity >= 0, "Distance sensitivity must be non-negative"


@dataclass
class DecisionRules:
    """Rules for customer decision making."""
    price_threshold: float  # maximum acceptable price difference
    convenience_weight: float  # weight of convenience vs price (0-1)
    brand_loyalty_weight: float  # weight of brand loyalty (0-1)
    promotion_sensitivity: float  # response to promotions (0-1)
    recommendation_influence: float  # influence of recommendations (0-1)
    
    def __post_init__(self):
        """Validate decision rules."""
        assert 0 <= self.convenience_weight <= 1, "Convenience weight must be 0-1"
        assert 0 <= self.brand_loyalty_weight <= 1, "Brand loyalty weight must be 0-1"
        assert 0 <= self.promotion_sensitivity <= 1, "Promotion sensitivity must be 0-1"
        assert 0 <= self.recommendation_influence <= 1, "Recommendation influence must be 0-1"


@dataclass
class CustomerPersona:
    """Complete customer persona model."""
    persona_id: str
    name: str
    description: str
    behavioral_attributes: BehavioralAttributes
    contextual_triggers: ContextualTriggers
    decision_rules: DecisionRules
    segment_size: int  # number of real customers in this segment
    confidence_score: float  # confidence in persona accuracy (0-1)
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate customer persona."""
        assert 0 <= self.confidence_score <= 1, "Confidence score must be 0-1"
        assert self.segment_size >= 0, "Segment size must be non-negative"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "description": self.description,
            "behavioral_attributes": {
                "visit_frequency": self.behavioral_attributes.visit_frequency,
                "avg_purchase_amount": self.behavioral_attributes.avg_purchase_amount,
                "time_of_day_preference": self.behavioral_attributes.time_of_day_preference,
                "fuel_type_preference": {k.value: v for k, v in self.behavioral_attributes.fuel_type_preference.items()},
                "additional_services_usage": self.behavioral_attributes.additional_services_usage,
                "price_sensitivity": self.behavioral_attributes.price_sensitivity,
                "loyalty_score": self.behavioral_attributes.loyalty_score
            },
            "contextual_triggers": {
                "weather_sensitivity": {k.value: v for k, v in self.contextual_triggers.weather_sensitivity.items()},
                "traffic_sensitivity": {k.value: v for k, v in self.contextual_triggers.traffic_sensitivity.items()},
                "queue_tolerance": self.contextual_triggers.queue_tolerance,
                "distance_sensitivity": self.contextual_triggers.distance_sensitivity,
                "day_of_week_patterns": {k.value: v for k, v in self.contextual_triggers.day_of_week_patterns.items()},
                "seasonal_patterns": {k.value: v for k, v in self.contextual_triggers.seasonal_patterns.items()}
            },
            "decision_rules": {
                "price_threshold": self.decision_rules.price_threshold,
                "convenience_weight": self.decision_rules.convenience_weight,
                "brand_loyalty_weight": self.decision_rules.brand_loyalty_weight,
                "promotion_sensitivity": self.decision_rules.promotion_sensitivity,
                "recommendation_influence": self.decision_rules.recommendation_influence
            },
            "segment_size": self.segment_size,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert persona to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomerPersona":
        """Create persona from dictionary."""
        behavioral_attrs = BehavioralAttributes(
            visit_frequency=data["behavioral_attributes"]["visit_frequency"],
            avg_purchase_amount=data["behavioral_attributes"]["avg_purchase_amount"],
            time_of_day_preference=data["behavioral_attributes"]["time_of_day_preference"],
            fuel_type_preference={FuelType(k): v for k, v in data["behavioral_attributes"]["fuel_type_preference"].items()},
            additional_services_usage=data["behavioral_attributes"]["additional_services_usage"],
            price_sensitivity=data["behavioral_attributes"]["price_sensitivity"],
            loyalty_score=data["behavioral_attributes"]["loyalty_score"]
        )
        
        contextual_triggers = ContextualTriggers(
            weather_sensitivity={WeatherCondition(k): v for k, v in data["contextual_triggers"]["weather_sensitivity"].items()},
            traffic_sensitivity={TrafficLevel(k): v for k, v in data["contextual_triggers"]["traffic_sensitivity"].items()},
            queue_tolerance=data["contextual_triggers"]["queue_tolerance"],
            distance_sensitivity=data["contextual_triggers"]["distance_sensitivity"],
            day_of_week_patterns={DayOfWeek(k): v for k, v in data["contextual_triggers"]["day_of_week_patterns"].items()},
            seasonal_patterns={Season(k): v for k, v in data["contextual_triggers"]["seasonal_patterns"].items()}
        )
        
        decision_rules = DecisionRules(
            price_threshold=data["decision_rules"]["price_threshold"],
            convenience_weight=data["decision_rules"]["convenience_weight"],
            brand_loyalty_weight=data["decision_rules"]["brand_loyalty_weight"],
            promotion_sensitivity=data["decision_rules"]["promotion_sensitivity"],
            recommendation_influence=data["decision_rules"]["recommendation_influence"]
        )
        
        return cls(
            persona_id=data["persona_id"],
            name=data["name"],
            description=data["description"],
            behavioral_attributes=behavioral_attrs,
            contextual_triggers=contextual_triggers,
            decision_rules=decision_rules,
            segment_size=data["segment_size"],
            confidence_score=data["confidence_score"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata", {})
        )
    
    def predict_visit_probability(self, context: Dict[str, Any]) -> float:
        """Predict probability of customer visit given context."""
        base_prob = min(self.behavioral_attributes.visit_frequency / 30, 1.0)  # daily probability
        
        # Apply contextual modifiers
        modifiers = 1.0
        
        if "weather" in context:
            weather = WeatherCondition(context["weather"])
            modifiers *= self.contextual_triggers.weather_sensitivity.get(weather, 1.0)
        
        if "traffic" in context:
            traffic = TrafficLevel(context["traffic"])
            modifiers *= self.contextual_triggers.traffic_sensitivity.get(traffic, 1.0)
        
        if "day_of_week" in context:
            day = DayOfWeek(context["day_of_week"])
            modifiers *= self.contextual_triggers.day_of_week_patterns.get(day, 1.0)
        
        if "season" in context:
            season = Season(context["season"])
            modifiers *= self.contextual_triggers.seasonal_patterns.get(season, 1.0)
        
        if "queue_length" in context:
            if context["queue_length"] > self.contextual_triggers.queue_tolerance:
                modifiers *= 0.5  # Significant reduction if queue too long
        
        return min(base_prob * modifiers, 1.0)