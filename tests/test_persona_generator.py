import pytest
import numpy as np
import pandas as pd
from src.personas.persona_generator import PersonaGenerator
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

def test_generate_personas_from_clusters(demo_df):
    seg = CustomerSegmentation()
    X = seg.preprocess_data(demo_df, demo_df.columns.tolist())
    model = seg.fit_kmeans(X, n_clusters=3)
    pg = PersonaGenerator()
    personas = pg.generate_from_clusters(demo_df, model.labels_, n_personas=3)
    assert len(personas) == 3
    for p in personas:
        d = p.to_dict()
        assert 'persona_id' in d
        assert d['segment_size'] > 0


def test_persona_export_json(demo_df):
    seg = CustomerSegmentation()
    X = seg.preprocess_data(demo_df, demo_df.columns.tolist())
    model = seg.fit_kmeans(X, n_clusters=2)
    pg = PersonaGenerator()
    personas = pg.generate_from_clusters(demo_df, model.labels_, n_personas=2)
    json_str = pg.export_personas(personas, format='json')
    assert json_str.startswith('[') and json_str.endswith(']')
