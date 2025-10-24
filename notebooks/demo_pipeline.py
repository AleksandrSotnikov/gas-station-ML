import numpy as np
import pandas as pd

from src.clustering.customer_segmentation import CustomerSegmentation
from src.personas.persona_generator import PersonaGenerator
from src.experiments.ab_testing import ExperimentDesigner

# 1) Mock-данные
np.random.seed(42)
df = pd.DataFrame({
    'visits_per_month': np.random.gamma(5, 1, 2000),
    'avg_ticket': np.random.normal(1800, 400, 2000).clip(300, 5000),
    'fuel_regular_share': np.random.beta(5, 2, 2000),
    'fuel_diesel_share': np.random.beta(3, 4, 2000),
    'services_shop': np.random.beta(3, 3, 2000),
    'services_cafe': np.random.beta(2, 5, 2000),
})

# 2) Кластеризация
features = df.columns.tolist()
seg = CustomerSegmentation()
X = seg.preprocess_data(df, features)
model = seg.fit_kmeans(X, n_clusters=5)

# 3) Генерация портретов
pg = PersonaGenerator()
personas = pg.generate_from_clusters(df, model.labels_, n_personas=5)
print(f"Generated personas: {len(personas)}")

# 4) Дизайн эксперимента
ed = ExperimentDesigner()
hyp = ed.create_hypothesis(
    name="Персонифицированные рекомендации",
    description="Рекомендации на кассе по сервисам для лояльных клиентов",
    target_metric="conversion_rate",
    expected_effect_size=0.07,
    affected_personas=[p.persona_id for p in personas[:2]],
    feature_description="UI-карточки рекомендаций в кассовом интерфейсе"
)
exp = ed.design_ab_test(hyp, personas)
res = ed.simulate_experiment(exp, personas)
ana = ed.analyze_results(res)
print(ana)