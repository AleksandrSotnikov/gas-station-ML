"""FastAPI application exposing endpoints for personas, segmentation, experiments, and predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.clustering.customer_segmentation import CustomerSegmentation
from src.personas.persona_generator import PersonaGenerator
from src.experiments.ab_testing import ExperimentDesigner
from src.models.customer_persona import CustomerPersona

app = FastAPI(title="Gas Station ML API", version="0.1.0")

# In-memory stores (for demo)
personas_store: Dict[str, Dict[str, Any]] = {}
segmentation_store: Dict[str, Any] = {}
experiments_store: Dict[str, Any] = {}


class SegmentationRequest(BaseModel):
    data: List[Dict[str, Any]]
    features: List[str]
    n_clusters: Optional[int] = None


class PersonasFromClustersRequest(BaseModel):
    data: List[Dict[str, Any]]
    labels: List[int]
    n_personas: Optional[int] = None


class HypothesisRequest(BaseModel):
    name: str
    description: str
    target_metric: str
    expected_effect_size: float
    affected_personas: List[str]
    feature_description: str
    business_context: Optional[str] = ""


@app.post("/api/v1/segmentation/kmeans")
def segment_kmeans(req: SegmentationRequest):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(req.data)
    seg = CustomerSegmentation()
    X = seg.preprocess_data(df, req.features)
    model = seg.fit_kmeans(X, n_clusters=req.n_clusters)

    labels = model.labels_.tolist()
    segmentation_store["kmeans"] = {
        "labels": labels,
        "metrics": seg.cluster_metrics.get("kmeans", {})
    }
    return segmentation_store["kmeans"]


@app.post("/api/v1/personas/from-clusters")
def personas_from_clusters(req: PersonasFromClustersRequest):
    import pandas as pd
    df = pd.DataFrame(req.data)
    pg = PersonaGenerator()
    personas = pg.generate_from_clusters(df, np.array(req.labels), n_personas=req.n_personas)

    result = [p.to_dict() for p in personas]
    for p in personas:
        personas_store[p.persona_id] = p.to_dict()
    return result


@app.get("/api/v1/personas/{persona_id}")
def get_persona(persona_id: str):
    if persona_id not in personas_store:
        raise HTTPException(status_code=404, detail="Persona not found")
    return personas_store[persona_id]


@app.post("/api/v1/experiments/design")
def design_experiment(req: HypothesisRequest):
    pg_personas = [CustomerPersona.from_dict(p) for p in personas_store.values() if p["persona_id"] in req.affected_personas]
    if not pg_personas:
        raise HTTPException(status_code=400, detail="No affected personas found")

    designer = ExperimentDesigner()
    hypothesis = designer.create_hypothesis(
        name=req.name,
        description=req.description,
        target_metric=req.target_metric,
        expected_effect_size=req.expected_effect_size,
        affected_personas=req.affected_personas,
        feature_description=req.feature_description,
        business_context=req.business_context or ""
    )
    design = designer.design_ab_test(hypothesis, pg_personas)
    experiments_store[design.experiment_id] = design

    return {
        "experiment_id": design.experiment_id,
        "sample_size": design.sample_size,
        "control_size": design.control_size,
        "treatment_size": design.treatment_size,
        "test_duration_days": design.test_duration_days,
        "confidence_level": design.confidence_level,
        "statistical_power": design.statistical_power,
    }


@app.post("/api/v1/experiments/{experiment_id}/simulate")
def simulate_experiment(experiment_id: str):
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")

    design = experiments_store[experiment_id]
    personas = [CustomerPersona.from_dict(p) for p in personas_store.values() if p["persona_id"] in design.hypothesis.affected_personas]

    designer = ExperimentDesigner()
    results = designer.simulate_experiment(design, personas)
    analysis = designer.analyze_results(results)

    return {
        "results": {
            "control_metrics": results.control_metrics,
            "treatment_metrics": results.treatment_metrics,
            "effect_sizes": results.effect_sizes,
            "significance": results.statistical_significance,
            "confidence_intervals": results.confidence_intervals,
        },
        "analysis": analysis
    }


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "personas": len(personas_store)}
