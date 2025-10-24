from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.clustering.customer_segmentation import CustomerSegmentation
from src.personas.persona_generator import PersonaGenerator
from src.experiments.ab_testing import ExperimentDesigner
from src.models.customer_persona import CustomerPersona

app = FastAPI(title="Gas Station ML UI", version="0.1.1")

# Mount static and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# In-memory cache
CACHE = {
    "data": None,
    "features": None,
    "labels": None,
    "personas": [],
    "experiment": None,
}

class DataPayload(BaseModel):
    data: List[Dict[str, Any]]
    features: List[str]
    n_clusters: Optional[int] = 5

class ExperimentPayload(BaseModel):
    name: str
    description: str
    target_metric: str = "conversion_rate"
    expected_effect_size: float = 0.07
    affected_personas: List[str]
    feature_description: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "personas_count": len(CACHE["personas"]),
        "has_data": CACHE["data"] is not None,
        "has_labels": CACHE["labels"] is not None,
        "has_experiment": CACHE["experiment"] is not None,
        "features": CACHE["features"] or []
    })

@app.post("/ui/upload", response_class=HTMLResponse)
async def ui_upload(request: Request):
    form = await request.form()
    csv_file = form.get("data_file")
    if not csv_file:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Файл не загружен"})
    try:
        df = pd.read_csv(csv_file.file)
        # Сохраняем только числовые фичи по умолчанию
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        CACHE["data"] = df
        CACHE["features"] = numeric_cols
        return templates.TemplateResponse("index.html", {"request": request, "message": "Данные загружены", "features": numeric_cols, "has_data": True})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Ошибка загрузки: {e}"})

@app.post("/ui/cluster", response_class=HTMLResponse)
async def ui_cluster(request: Request):
    try:
        form = await request.form()
        n_clusters = int(form.get("n_clusters", 5))
        selected = form.getlist("features")
        if CACHE["data"] is None:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Сначала загрузите данные"})
        # если не выбрали — используем сохранённые
        features = selected or (CACHE["features"] or [])
        # фильтруем только числовые
        numeric_cols = CACHE["data"].select_dtypes(include=["number"]).columns.tolist()
        features_final = [f for f in features if f in numeric_cols]
        if not features_final:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Выберите хотя бы один числовой признак", "features": numeric_cols, "has_data": True})
        seg = CustomerSegmentation()
        X = seg.preprocess_data(CACHE["data"], features_final)
        model = seg.fit_kmeans(X, n_clusters=n_clusters)
        CACHE["labels"] = model.labels_.tolist()
        CACHE["features"] = features_final
        profiles = seg.get_cluster_profiles(CACHE["data"], "kmeans").to_html(classes="table table-sm table-striped")
        return templates.TemplateResponse("index.html", {"request": request, "message": "Кластеризация завершена", "features": features_final, "labels_count": len(CACHE["labels"]), "profiles_html": profiles, "has_data": True, "has_labels": True})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Ошибка кластеризации: {e}", "features": CACHE["features"] or [], "has_data": CACHE["data"] is not None})

@app.post("/ui/personas", response_class=HTMLResponse)
async def ui_personas(request: Request):
    try:
        if CACHE["data"] is None or CACHE["labels"] is None:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Требуются данные и кластера", "features": CACHE["features"] or []})
        pg = PersonaGenerator()
        personas = pg.generate_from_clusters(CACHE["data"], np.array(CACHE["labels"]), n_personas=5)
        CACHE["personas"] = personas
        return templates.TemplateResponse("index.html", {"request": request, "message": f"Сгенерировано портретов: {len(personas)}", "personas": [p.to_dict() for p in personas], "has_data": True, "has_labels": True, "features": CACHE["features"] or []})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Ошибка генерации портретов: {e}", "features": CACHE["features"] or []})

@app.post("/ui/experiment", response_class=HTMLResponse)
async def ui_experiment(request: Request):
    try:
        form = await request.form()
        name = form.get("name", "Перс. рекомендации")
        description = form.get("description", "A/B тест")
        metric = form.get("metric", "conversion_rate")
        effect = float(form.get("effect", 0.07))
        selected_ids = form.getlist("persona_ids")
        if not CACHE["personas"]:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Сначала сгенерируйте портреты", "features": CACHE["features"] or []})
        personas = CACHE["personas"]
        designer = ExperimentDesigner()
        hypothesis = designer.create_hypothesis(
            name=name,
            description=description,
            target_metric=metric,
            expected_effect_size=effect,
            affected_personas=selected_ids or [p.persona_id for p in personas[:2]],
            feature_description="UI рекомендации"
        )
        exp = designer.design_ab_test(hypothesis, personas)
        results = designer.simulate_experiment(exp, personas)
        analysis = designer.analyze_results(results)
        CACHE["experiment"] = analysis
        return templates.TemplateResponse("index.html", {"request": request, "message": "Эксперимент смоделирован", "analysis": analysis, "personas": [p.to_dict() for p in personas], "has_data": True, "has_labels": True, "features": CACHE["features"] or []})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Ошибка эксперимента: {e}", "features": CACHE["features"] or []})
