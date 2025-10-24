import pytest
from src.models.customer_persona import CustomerPersona, BehavioralAttributes, ContextualTriggers, DecisionRules, FuelType, WeatherCondition, TrafficLevel, DayOfWeek, Season
from datetime import datetime
import uuid

@pytest.fixture
def behavioral_attrs():
    return BehavioralAttributes(
        visit_frequency=10,
        avg_purchase_amount=1500,
        time_of_day_preference=[8, 18],
        fuel_type_preference={FuelType.REGULAR_95: 0.6, FuelType.PREMIUM_98: 0.1, FuelType.DIESEL: 0.2, FuelType.GAS: 0.1},
        additional_services_usage={"car_wash": 0.2, "shop": 0.6, "cafe": 0.3, "maintenance": 0.1},
        price_sensitivity=0.7,
        loyalty_score=0.8
    )

@pytest.fixture
def contextual_triggers():
    return ContextualTriggers(
        weather_sensitivity={WeatherCondition.SUNNY: 1.0, WeatherCondition.RAINY: 0.9, WeatherCondition.SNOWY: 0.8, WeatherCondition.CLOUDY: 1.0, WeatherCondition.FOGGY: 0.85},
        traffic_sensitivity={TrafficLevel.LOW: 1.1, TrafficLevel.MEDIUM: 1.0, TrafficLevel.HIGH: 0.7},
        queue_tolerance=8,
        distance_sensitivity=2.5,
        day_of_week_patterns={day: 1.0 for day in DayOfWeek},
        seasonal_patterns={season: 1.0 for season in Season}
    )

@pytest.fixture
def decision_rules():
    return DecisionRules(
        price_threshold=2.0,
        convenience_weight=0.5,
        brand_loyalty_weight=0.6,
        promotion_sensitivity=0.7,
        recommendation_influence=0.5
    )


def test_customer_persona_integrity(behavioral_attrs, contextual_triggers, decision_rules):
    persona_id = str(uuid.uuid4())
    persona = CustomerPersona(
        persona_id=persona_id,
        name="Тестовый портрет",
        description="Описание тестового портрета",
        behavioral_attributes=behavioral_attrs,
        contextual_triggers=contextual_triggers,
        decision_rules=decision_rules,
        segment_size=300,
        confidence_score=0.92,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        metadata={"test": True}
    )
    d = persona.to_dict()
    assert d["persona_id"] == persona_id
    assert 0 <= d["confidence_score"] <= 1
    assert d["segment_size"] == 300

def test_fuel_preferences_sum(behavioral_attrs):
    assert abs(sum(behavioral_attrs.fuel_type_preference.values()) - 1.0) < 0.01

def test_behavioral_validation(behavioral_attrs):
    assert behavioral_attrs.price_sensitivity <= 1
    assert behavioral_attrs.loyalty_score <= 1

def test_contextual_validation(contextual_triggers):
    assert contextual_triggers.queue_tolerance >= 0
    assert contextual_triggers.distance_sensitivity >= 0

def test_decision_rules_validation(decision_rules):
    assert 0 <= decision_rules.convenience_weight <= 1
    assert 0 <= decision_rules.brand_loyalty_weight <= 1
