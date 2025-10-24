"""A/B testing and experiment design module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import uuid
from scipy import stats
from scipy.stats import norm, chi2_contingency, ttest_ind
from loguru import logger
import json

from src.models.customer_persona import CustomerPersona
from config.settings import EXPERIMENT_CONFIG


class ExperimentHypothesis:
    """Class representing an experiment hypothesis."""
    
    def __init__(
        self,
        hypothesis_id: str,
        name: str,
        description: str,
        target_metric: str,
        expected_effect_size: float,
        affected_personas: List[str],
        feature_description: str,
        business_context: str
    ):
        self.hypothesis_id = hypothesis_id
        self.name = name
        self.description = description
        self.target_metric = target_metric
        self.expected_effect_size = expected_effect_size
        self.affected_personas = affected_personas
        self.feature_description = feature_description
        self.business_context = business_context
        self.created_at = datetime.now().isoformat()


class ExperimentDesign:
    """Class representing an A/B test design."""
    
    def __init__(
        self,
        experiment_id: str,
        hypothesis: ExperimentHypothesis,
        sample_size: int,
        test_duration_days: int,
        control_ratio: float = 0.5,
        confidence_level: float = 0.95,
        statistical_power: float = 0.8
    ):
        self.experiment_id = experiment_id
        self.hypothesis = hypothesis
        # sanitize numeric params
        try:
            if not np.isfinite(sample_size) or sample_size <= 0:
                sample_size = EXPERIMENT_CONFIG.get("ab_test", {}).get("min_sample_size", 1000)
            sample_size = int(np.ceil(float(sample_size)))
        except Exception:
            sample_size = EXPERIMENT_CONFIG.get("ab_test", {}).get("min_sample_size", 1000)
        try:
            if control_ratio is None or not np.isfinite(control_ratio) or control_ratio <= 0 or control_ratio >= 1:
                control_ratio = 0.5
            control_ratio = float(control_ratio)
        except Exception:
            control_ratio = 0.5
        self.sample_size = sample_size
        self.test_duration_days = int(max(1, np.nan_to_num(test_duration_days, nan=7)))
        self.control_ratio = control_ratio
        self.treatment_ratio = 1.0 - control_ratio
        self.confidence_level = float(confidence_level) if np.isfinite(confidence_level) else 0.95
        self.statistical_power = float(statistical_power) if np.isfinite(statistical_power) else 0.8
        self.created_at = datetime.now().isoformat()
        
        # Calculate group sizes using safe math
        self.control_size = int(max(1, np.floor(self.sample_size * self.control_ratio)))
        self.treatment_size = int(max(1, self.sample_size - self.control_size))


class ExperimentResults:
    """Class for storing experiment results."""
    
    def __init__(
        self,
        experiment_id: str,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        statistical_significance: Dict[str, Any],
        effect_sizes: Dict[str, float],
        confidence_intervals: Dict[str, Tuple[float, float]]
    ):
        self.experiment_id = experiment_id
        self.control_metrics = control_metrics
        self.treatment_metrics = treatment_metrics
        self.statistical_significance = statistical_significance
        self.effect_sizes = effect_sizes
        self.confidence_intervals = confidence_intervals
        self.analyzed_at = datetime.now().isoformat()


class ExperimentDesigner:
    """Designer for A/B tests and experiment planning."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize experiment designer.
        
        Args:
            config: Configuration dictionary, uses default if None
        """
        self.config = config or EXPERIMENT_CONFIG
        self.experiments = []
        self.hypotheses = []
        
    def create_hypothesis(
        self,
        name: str,
        description: str,
        target_metric: str,
        expected_effect_size: float,
        affected_personas: List[str],
        feature_description: str,
        business_context: str = ""
    ) -> ExperimentHypothesis:
        """Create an experiment hypothesis.
        
        Args:
            name: Short name for the hypothesis
            description: Detailed description
            target_metric: Primary metric to measure
            expected_effect_size: Expected relative change (e.g., 0.05 for 5% increase)
            affected_personas: List of persona IDs that might be affected
            feature_description: Description of the feature being tested
            business_context: Business justification
            
        Returns:
            Created hypothesis
        """
        hypothesis_id = str(uuid.uuid4())
        
        # sanitize inputs
        try:
            if not np.isfinite(expected_effect_size) or expected_effect_size == 0:
                expected_effect_size = 0.05
        except Exception:
            expected_effect_size = 0.05
        affected_personas = [p for p in (affected_personas or []) if p]
        
        hypothesis = ExperimentHypothesis(
            hypothesis_id=hypothesis_id,
            name=name,
            description=description,
            target_metric=target_metric,
            expected_effect_size=float(expected_effect_size),
            affected_personas=affected_personas,
            feature_description=feature_description,
            business_context=business_context
        )
        
        self.hypotheses.append(hypothesis)
        logger.info(f"Created hypothesis '{name}' with ID {hypothesis_id}")
        return hypothesis
    
    def design_ab_test(
        self,
        hypothesis: ExperimentHypothesis,
        personas: List[CustomerPersona],
        confidence_level: float = 0.95,
        statistical_power: float = 0.8,
        control_ratio: float = 0.5,
        max_duration_days: int = 30
    ) -> ExperimentDesign:
        """Design an A/B test for a hypothesis.
        
        Args:
            hypothesis: Experiment hypothesis
            personas: List of customer personas
            confidence_level: Statistical confidence level
            statistical_power: Statistical power
            control_ratio: Ratio of control group (0.5 = 50/50 split)
            max_duration_days: Maximum test duration
            
        Returns:
            Designed experiment
        """
        logger.info(f"Designing A/B test for hypothesis '{hypothesis.name}'")
        
        # Calculate required sample size safely
        sample_size = self.calculate_sample_size(
            effect_size=hypothesis.expected_effect_size,
            confidence_level=confidence_level,
            statistical_power=statistical_power,
            baseline_metric=self._estimate_baseline_metric(hypothesis, personas)
        )
        
        # Estimate test duration based on persona traffic
        daily_users = self._estimate_daily_users(hypothesis, personas)
        try:
            if not np.isfinite(daily_users) or daily_users <= 0:
                daily_users = 100.0
            estimated_duration = int(max(7, min(max_duration_days, np.ceil(sample_size / float(daily_users)))))
        except Exception:
            estimated_duration = 7
        
        experiment_id = str(uuid.uuid4())
        
        experiment = ExperimentDesign(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            sample_size=sample_size,
            test_duration_days=estimated_duration,
            control_ratio=control_ratio,
            confidence_level=confidence_level,
            statistical_power=statistical_power
        )
        
        self.experiments.append(experiment)
        logger.info(f"Designed experiment with {sample_size} users over {estimated_duration} days")
        return experiment
    
    def calculate_sample_size(
        self,
        effect_size: float,
        confidence_level: float = 0.95,
        statistical_power: float = 0.8,
        baseline_metric: float = 0.1,
        metric_type: str = "conversion"
    ) -> int:
        """Calculate required sample size for experiment.
        
        Args:
            effect_size: Expected relative effect size
            confidence_level: Statistical confidence level
            statistical_power: Statistical power
            baseline_metric: Baseline value of the metric
            metric_type: Type of metric ('conversion', 'continuous')
            
        Returns:
            Required sample size per group
        """
        alpha = 1 - (confidence_level if np.isfinite(confidence_level) else 0.95)
        beta = 1 - (statistical_power if np.isfinite(statistical_power) else 0.8)
        
        # sanitize
        try:
            if not np.isfinite(effect_size) or effect_size == 0:
                effect_size = 0.05
        except Exception:
            effect_size = 0.05
        try:
            if not np.isfinite(baseline_metric) or baseline_metric <= 0:
                baseline_metric = 0.1
        except Exception:
            baseline_metric = 0.1
        
        if metric_type == "conversion":
            p1 = float(baseline_metric)
            p2 = float(min(0.99, max(0.01, p1 * (1 + effect_size))))
            p_pooled = (p1 + p2) / 2.0
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(statistical_power)
            numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
            denominator = (p2 - p1)**2
            if not np.isfinite(numerator) or denominator <= 0:
                n = self.config["ab_test"]["min_sample_size"]
            else:
                n = int(np.ceil(numerator / denominator))
        else:
            cohen_d = float(effect_size)
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(statistical_power)
            denom = cohen_d**2
            if not np.isfinite(z_alpha) or not np.isfinite(z_beta) or denom <= 0:
                n = self.config["ab_test"]["min_sample_size"]
            else:
                n = int(np.ceil(2 * ((z_alpha + z_beta) / cohen_d)**2))
        
        min_sample_size = self.config["ab_test"]["min_sample_size"]
        n = max(int(min_sample_size), int(np.nan_to_num(n, nan=min_sample_size)))
        logger.info(f"Calculated sample size: {n} per group")
        return n
    
    def _estimate_baseline_metric(
        self, 
        hypothesis: ExperimentHypothesis, 
        personas: List[CustomerPersona]
    ) -> float:
        metric_baselines = {
            "conversion_rate": 0.05,
            "average_purchase_value": 2000.0,
            "customer_satisfaction": 4.2,
            "retention_rate": 0.7,
            "cross_selling_rate": 0.15
        }
        val = metric_baselines.get(hypothesis.target_metric, 0.1)
        return float(val)
    
    def _estimate_daily_users(
        self, 
        hypothesis: ExperimentHypothesis, 
        personas: List[CustomerPersona]
    ) -> float:
        total_daily_users = 0.0
        for persona in personas:
            if persona.persona_id in (hypothesis.affected_personas or []):
                vf = getattr(persona.behavioral_attributes, 'visit_frequency', 0) or 0
                ss = getattr(persona, 'segment_size', 0) or 0
                try:
                    vf = float(vf)
                    ss = float(ss)
                except Exception:
                    vf, ss = 0.0, 0.0
                daily_visits_per_user = max(0.0, vf) / 30.0
                total_daily_users += max(0.0, ss) * daily_visits_per_user
        return float(max(100.0, np.nan_to_num(total_daily_users, nan=0.0)))
    
    def simulate_experiment(
        self,
        experiment_design: ExperimentDesign,
        personas: List[CustomerPersona],
        noise_level: float = 0.1
    ) -> ExperimentResults:
        logger.info(f"Simulating experiment {experiment_design.experiment_id}")
        hypothesis = experiment_design.hypothesis
        control_metrics = self._simulate_group_metrics(
            experiment_design.control_size,
            hypothesis,
            personas,
            is_treatment=False,
            noise_level=noise_level
        )
        treatment_metrics = self._simulate_group_metrics(
            experiment_design.treatment_size,
            hypothesis,
            personas,
            is_treatment=True,
            noise_level=noise_level
        )
        significance = self._calculate_statistical_significance(
            control_metrics, treatment_metrics, hypothesis.target_metric
        )
        effect_sizes = self._calculate_effect_sizes(control_metrics, treatment_metrics)
        confidence_intervals = self._calculate_confidence_intervals(
            control_metrics, treatment_metrics, experiment_design.confidence_level
        )
        results = ExperimentResults(
            experiment_id=experiment_design.experiment_id,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            statistical_significance=significance,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals
        )
        logger.info(f"Simulation complete. Primary metric change: {effect_sizes.get(hypothesis.target_metric, 0):.3f}")
        return results
    
    def _simulate_group_metrics(
        self,
        group_size: int,
        hypothesis: ExperimentHypothesis,
        personas: List[CustomerPersona],
        is_treatment: bool,
        noise_level: float
    ) -> Dict[str, float]:
        baseline_metrics = {
            "conversion_rate": 0.05,
            "average_purchase_value": 2000.0,
            "customer_satisfaction": 4.2,
            "retention_rate": 0.7,
            "cross_selling_rate": 0.15
        }
        metrics: Dict[str, float] = {}
        for metric_name in self.config["metrics"]:
            base_value = baseline_metrics.get(metric_name, 0.1)
            if not np.isfinite(base_value):
                base_value = 0.1
            value = float(base_value)
            if is_treatment and metric_name == hypothesis.target_metric:
                treatment_effect = self._calculate_persona_treatment_effect(
                    hypothesis, personas, metric_name
                )
                if not np.isfinite(treatment_effect):
                    treatment_effect = 0.0
                value = value * (1.0 + treatment_effect)
            try:
                noise = np.random.normal(0, abs(noise_level) * abs(value))
                value = value + noise
            except Exception:
                pass
            if "satisfaction" in metric_name:
                value = max(1.0, min(5.0, float(np.nan_to_num(value, nan=base_value))))
            elif "rate" in metric_name:
                value = max(0.0, min(1.0, float(np.nan_to_num(value, nan=base_value))))
            else:
                value = float(np.nan_to_num(value, nan=base_value))
            metrics[metric_name] = value
        return metrics
    
    def _calculate_persona_treatment_effect(
        self,
        hypothesis: ExperimentHypothesis,
        personas: List[CustomerPersona],
        metric_name: str
    ) -> float:
        total_effect = 0.0
        total_weight = 0.0
        base_effect = hypothesis.expected_effect_size if np.isfinite(hypothesis.expected_effect_size) else 0.05
        for persona in personas:
            if persona.persona_id in (hypothesis.affected_personas or []):
                persona_effect = float(base_effect)
                ls = getattr(persona.behavioral_attributes, 'loyalty_score', 0) or 0
                ps = getattr(persona.behavioral_attributes, 'price_sensitivity', 0) or 0
                try:
                    ls = float(ls)
                    ps = float(ps)
                except Exception:
                    ls, ps = 0.0, 0.0
                if metric_name == "conversion_rate":
                    persona_effect *= (1 + max(0.0, ls) * 0.5)
                elif metric_name == "average_purchase_value":
                    persona_effect *= (1 - max(0.0, ps) * 0.3)
                elif metric_name == "retention_rate":
                    persona_effect *= (2 - max(0.0, ls))
                weight = float(getattr(persona, 'segment_size', 0) or 0)
                if np.isfinite(weight) and weight > 0:
                    total_effect += persona_effect * weight
                    total_weight += weight
        return float(total_effect / total_weight) if total_weight > 0 else float(base_effect)
    
    def _calculate_statistical_significance(
        self,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        primary_metric: str
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for metric_name in control_metrics.keys():
            cval = control_metrics.get(metric_name, 0.0)
            tval = treatment_metrics.get(metric_name, 0.0)
            if "rate" in metric_name:
                n_control = 1000
                n_treatment = 1000
                c = float(np.nan_to_num(cval, nan=0.0))
                t = float(np.nan_to_num(tval, nan=0.0))
                control_successes = int(max(0, min(n_control, round(c * n_control))))
                treatment_successes = int(max(0, min(n_treatment, round(t * n_treatment))))
                observed = np.array([
                    [control_successes, n_control - control_successes],
                    [treatment_successes, n_treatment - treatment_successes]
                ])
                try:
                    chi2, p_value, _, _ = chi2_contingency(observed)
                    test_statistic = chi2
                except Exception:
                    p_value = 0.5
                    test_statistic = 0.0
            else:
                n_control = 1000
                n_treatment = 1000
                c = float(np.nan_to_num(cval, nan=0.0))
                t = float(np.nan_to_num(tval, nan=0.0))
                std_c = max(1e-8, abs(c) * 0.2)
                std_t = max(1e-8, abs(t) * 0.2)
                try:
                    control_samples = np.random.normal(c, std_c, n_control)
                    treatment_samples = np.random.normal(t, std_t, n_treatment)
                    test_statistic, p_value = ttest_ind(treatment_samples, control_samples)
                except Exception:
                    p_value = 0.5
                    test_statistic = 0.0
            results[metric_name] = {
                "p_value": float(p_value),
                "test_statistic": float(test_statistic),
                "is_significant": bool(p_value < 0.05),
                "is_primary": bool(metric_name == primary_metric)
            }
        return results
    
    def _calculate_effect_sizes(self, control_metrics, treatment_metrics) -> Dict[str, float]:
        effect_sizes: Dict[str, float] = {}
        for metric_name in control_metrics.keys():
            c = float(np.nan_to_num(control_metrics.get(metric_name, 0.0), nan=0.0))
            t = float(np.nan_to_num(treatment_metrics.get(metric_name, 0.0), nan=0.0))
            if c > 0 and np.isfinite(c) and np.isfinite(t):
                eff = (t - c) / c
            else:
                eff = 0.0
            effect_sizes[metric_name] = float(np.nan_to_num(eff, nan=0.0))
        return effect_sizes
    
    def _calculate_confidence_intervals(
        self, 
        control_metrics, 
        treatment_metrics, 
        confidence_level
    ) -> Dict[str, Tuple[float, float]]:
        intervals: Dict[str, Tuple[float, float]] = {}
        alpha = 1 - (confidence_level if np.isfinite(confidence_level) else 0.95)
        z_score = norm.ppf(1 - alpha / 2)
        if not np.isfinite(z_score):
            z_score = 1.96
        for metric_name in control_metrics.keys():
            c = float(np.nan_to_num(control_metrics.get(metric_name, 0.0), nan=0.0))
            t = float(np.nan_to_num(treatment_metrics.get(metric_name, 0.0), nan=0.0))
            effect = float(np.nan_to_num(t - c, nan=0.0))
            se = float(np.nan_to_num(abs(effect) * 0.1, nan=0.0))
            se = max(se, 1e-8)
            margin = float(z_score * se)
            lower = effect - margin
            upper = effect + margin
            intervals[metric_name] = (lower, upper)
        return intervals
    
    def analyze_results(self, results: ExperimentResults) -> Dict[str, Any]:
        analysis = {
            "experiment_id": results.experiment_id,
            "summary": {},
            "recommendations": [],
            "risks": [],
            "next_steps": []
        }
        primary_metrics = [k for k, v in results.statistical_significance.items() if v.get("is_primary", False)]
        if primary_metrics:
            primary_metric = primary_metrics[0]
            effect_size = float(np.nan_to_num(results.effect_sizes.get(primary_metric, 0.0), nan=0.0))
            is_significant = bool(results.statistical_significance[primary_metric]["is_significant"])
            p_value = float(np.nan_to_num(results.statistical_significance[primary_metric]["p_value"], nan=1.0))
            analysis["summary"] = {
                "primary_metric": primary_metric,
                "effect_size": effect_size,
                "is_significant": is_significant,
                "p_value": p_value,
                "control_value": float(np.nan_to_num(results.control_metrics[primary_metric], nan=0.0)),
                "treatment_value": float(np.nan_to_num(results.treatment_metrics[primary_metric], nan=0.0))
            }
            if is_significant and effect_size > 0:
                analysis["recommendations"].append("Внедрить изменение - статистически значимое улучшение")
            elif is_significant and effect_size < 0:
                analysis["recommendations"].append("Отклонить изменение - статистически значимое ухудшение")
            else:
                analysis["recommendations"].append("Продолжить тестирование или пересмотреть гипотезу")
            if abs(effect_size) > 0.2:
                analysis["risks"].append("Большой размер эффекта может указывать на ошибки в данных")
            if not is_significant:
                analysis["next_steps"].append("Увеличить размер выборки или продолжительность теста")
        return analysis
    
    def export_experiment_plan(
        self, 
        experiment: ExperimentDesign, 
        filepath: Optional[str] = None
    ) -> str:
        plan = {
            "experiment_id": experiment.experiment_id,
            "hypothesis": {
                "name": experiment.hypothesis.name,
                "description": experiment.hypothesis.description,
                "target_metric": experiment.hypothesis.target_metric,
                "expected_effect_size": float(np.nan_to_num(experiment.hypothesis.expected_effect_size, nan=0.05)),
                "affected_personas": experiment.hypothesis.affected_personas,
                "feature_description": experiment.hypothesis.feature_description
            },
            "design": {
                "sample_size": int(experiment.sample_size),
                "control_size": int(experiment.control_size),
                "treatment_size": int(experiment.treatment_size),
                "test_duration_days": int(experiment.test_duration_days),
                "confidence_level": float(experiment.confidence_level),
                "statistical_power": float(experiment.statistical_power)
            },
            "created_at": experiment.created_at
        }
        json_str = json.dumps(plan, ensure_ascii=False, indent=2)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Exported experiment plan to {filepath}")
        return json_str
