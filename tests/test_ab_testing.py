import pytest
import numpy as np
from src.experiments.ab_testing import ExperimentDesigner
from src.personas.persona_generator import PersonaGenerator
from src.clustering.customer_segmentation import CustomerSegmentation
import pandas as pd

@pytest.fixture
def test_personas():
    np.random.seed(42)
    df = pd.DataFrame({
        'visits_per_month': np.random.gamma(5, 1, 50),
        'avg_ticket': np.random.normal(1800, 400, 50).clip(300, 5000),
        'fuel_regular_share': np.random.beta(5, 2, 50),
        'fuel_diesel_share': np.random.beta(3, 4, 50),
        'services_shop': np.random.beta(3, 3, 50),
        'services_cafe': np.random.beta(2, 5, 50),
    })
    seg = CustomerSegmentation()
    X = seg.preprocess_data(df, df.columns.tolist())
    model = seg.fit_kmeans(X, n_clusters=2)
    pg = PersonaGenerator()
    return pg.generate_from_clusters(df, model.labels_, n_personas=2)

def test_experiment_design_simulation(test_personas):
    designer = ExperimentDesigner()
    hyp = designer.create_hypothesis(
        name="Тестовая гипотеза",
        description="Проверка A/B",
        target_metric="conversion_rate",
        expected_effect_size=0.08,
        affected_personas=[p.persona_id for p in test_personas],
        feature_description="Тестовая фича"
    )
    exp = designer.design_ab_test(hyp, test_personas)
    assert exp.sample_size > 0
    res = designer.simulate_experiment(exp, test_personas)
    ana = designer.analyze_results(res)
    assert "experiment_id" in ana
    assert "summary" in ana
