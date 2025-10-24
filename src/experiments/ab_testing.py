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
        self.sample_size = sample_size
        self.test_duration_days = test_duration_days
        self.control_ratio = control_ratio
        self.treatment_ratio = 1.0 - control_ratio
        self.confidence_level = confidence_level
        self.statistical_power = statistical_power
        self.created_at = datetime.now().isoformat()
        
        # Calculate group sizes
        self.control_size = int(sample_size * control_ratio)
        self.treatment_size = sample_size - self.control_size


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
        
        hypothesis = ExperimentHypothesis(
            hypothesis_id=hypothesis_id,
            name=name,
            description=description,
            target_metric=target_metric,
            expected_effect_size=expected_effect_size,
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
        
        # Calculate required sample size
        sample_size = self.calculate_sample_size(
            effect_size=hypothesis.expected_effect_size,
            confidence_level=confidence_level,
            statistical_power=statistical_power,
            baseline_metric=self._estimate_baseline_metric(hypothesis, personas)
        )
        
        # Estimate test duration based on persona traffic
        daily_users = self._estimate_daily_users(hypothesis, personas)
        estimated_duration = max(7, min(max_duration_days, int(sample_size / daily_users)))
        
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
        alpha = 1 - confidence_level
        beta = 1 - statistical_power
        
        if metric_type == "conversion":
            # For conversion rates (proportions)
            p1 = baseline_metric
            p2 = baseline_metric * (1 + effect_size)
            
            # Ensure p2 is valid probability
            p2 = min(0.99, max(0.01, p2))
            
            # Pooled proportion
            p_pooled = (p1 + p2) / 2
            
            # Z-scores
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(statistical_power)
            
            # Sample size calculation for proportions
            numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                        z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
            denominator = (p2 - p1)**2
            
            n = int(np.ceil(numerator / denominator))
            
        else:
            # For continuous metrics (assuming equal variances)
            # Simplified calculation - in practice, you'd need historical variance
            cohen_d = effect_size  # Treat effect size as Cohen's d
            
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(statistical_power)
            
            n = int(np.ceil(2 * ((z_alpha + z_beta) / cohen_d)**2))
        
        # Apply minimum sample size
        min_sample_size = self.config["ab_test"]["min_sample_size"]
        n = max(n, min_sample_size)
        
        logger.info(f"Calculated sample size: {n} per group")
        return n
    
    def _estimate_baseline_metric(
        self, 
        hypothesis: ExperimentHypothesis, 
        personas: List[CustomerPersona]
    ) -> float:
        """Estimate baseline metric value from personas.
        
        Args:
            hypothesis: Experiment hypothesis
            personas: List of personas
            
        Returns:
            Estimated baseline metric value
        """
        # Simplified estimation - in practice, this would use historical data
        metric_baselines = {
            "conversion_rate": 0.05,
            "average_purchase_value": 2000.0,
            "customer_satisfaction": 4.2,
            "retention_rate": 0.7,
            "cross_selling_rate": 0.15
        }
        
        return metric_baselines.get(hypothesis.target_metric, 0.1)
    
    def _estimate_daily_users(
        self, 
        hypothesis: ExperimentHypothesis, 
        personas: List[CustomerPersona]
    ) -> float:
        """Estimate daily users for affected personas.
        
        Args:
            hypothesis: Experiment hypothesis
            personas: List of personas
            
        Returns:
            Estimated daily users
        """
        total_daily_users = 0
        
        for persona in personas:
            if persona.persona_id in hypothesis.affected_personas:
                # Estimate daily users from visit frequency and segment size
                daily_visits_per_user = persona.behavioral_attributes.visit_frequency / 30
                daily_users_from_persona = persona.segment_size * daily_visits_per_user
                total_daily_users += daily_users_from_persona
        
        return max(100, total_daily_users)  # Minimum 100 daily users
    
    def simulate_experiment(
        self,
        experiment_design: ExperimentDesign,
        personas: List[CustomerPersona],
        noise_level: float = 0.1
    ) -> ExperimentResults:
        """Simulate experiment results based on personas.
        
        Args:
            experiment_design: Experiment design
            personas: List of personas
            noise_level: Amount of random noise to add
            
        Returns:
            Simulated experiment results
        """
        logger.info(f"Simulating experiment {experiment_design.experiment_id}")
        
        hypothesis = experiment_design.hypothesis
        
        # Simulate control group metrics
        control_metrics = self._simulate_group_metrics(
            experiment_design.control_size,
            hypothesis,
            personas,
            is_treatment=False,
            noise_level=noise_level
        )
        
        # Simulate treatment group metrics
        treatment_metrics = self._simulate_group_metrics(
            experiment_design.treatment_size,
            hypothesis,
            personas,
            is_treatment=True,
            noise_level=noise_level
        )
        
        # Calculate statistical significance
        significance = self._calculate_statistical_significance(
            control_metrics, treatment_metrics, hypothesis.target_metric
        )
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(control_metrics, treatment_metrics)
        
        # Calculate confidence intervals
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
        """Simulate metrics for a group.
        
        Args:
            group_size: Size of the group
            hypothesis: Experiment hypothesis
            personas: List of personas
            is_treatment: Whether this is treatment group
            noise_level: Random noise level
            
        Returns:
            Dictionary of simulated metrics
        """
        # Get baseline metrics
        baseline_metrics = {
            "conversion_rate": 0.05,
            "average_purchase_value": 2000.0,
            "customer_satisfaction": 4.2,
            "retention_rate": 0.7,
            "cross_selling_rate": 0.15
        }
        
        metrics = {}
        
        for metric_name in self.config["metrics"]:
            base_value = baseline_metrics.get(metric_name, 0.1)
            
            # Apply treatment effect if this is treatment group
            if is_treatment and metric_name == hypothesis.target_metric:
                # Apply persona-specific treatment effects
                treatment_effect = self._calculate_persona_treatment_effect(
                    hypothesis, personas, metric_name
                )
                value = base_value * (1 + treatment_effect)
            else:
                value = base_value
            
            # Add random noise
            noise = np.random.normal(0, noise_level * value)
            value = max(0, value + noise)
            
            # Ensure values are within reasonable bounds
            if "rate" in metric_name or "satisfaction" in metric_name:
                if "satisfaction" in metric_name:
                    value = max(1.0, min(5.0, value))
                else:
                    value = max(0.0, min(1.0, value))
            
            metrics[metric_name] = value
        
        return metrics
    
    def _calculate_persona_treatment_effect(
        self,
        hypothesis: ExperimentHypothesis,
        personas: List[CustomerPersona],
        metric_name: str
    ) -> float:
        """Calculate treatment effect based on affected personas.
        
        Args:
            hypothesis: Experiment hypothesis
            personas: List of personas
            metric_name: Name of the metric
            
        Returns:
            Calculated treatment effect
        """
        total_effect = 0
        total_weight = 0
        
        for persona in personas:
            if persona.persona_id in hypothesis.affected_personas:
                # Calculate persona-specific effect based on characteristics
                persona_effect = hypothesis.expected_effect_size
                
                # Modify effect based on persona characteristics
                if metric_name == "conversion_rate":
                    # More loyal customers might have higher conversion lift
                    persona_effect *= (1 + persona.behavioral_attributes.loyalty_score * 0.5)
                elif metric_name == "average_purchase_value":
                    # Price-sensitive customers might have lower purchase value lift
                    persona_effect *= (1 - persona.behavioral_attributes.price_sensitivity * 0.3)
                elif metric_name == "retention_rate":
                    # Already loyal customers have less room for improvement
                    persona_effect *= (2 - persona.behavioral_attributes.loyalty_score)
                
                weight = persona.segment_size
                total_effect += persona_effect * weight
                total_weight += weight
        
        return total_effect / total_weight if total_weight > 0 else hypothesis.expected_effect_size
    
    def _calculate_statistical_significance(
        self,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        primary_metric: str
    ) -> Dict[str, Any]:
        """Calculate statistical significance tests.
        
        Args:
            control_metrics: Control group metrics
            treatment_metrics: Treatment group metrics
            primary_metric: Primary metric name
            
        Returns:
            Statistical significance results
        """
        results = {}
        
        for metric_name in control_metrics.keys():
            control_value = control_metrics[metric_name]
            treatment_value = treatment_metrics[metric_name]
            
            # Simulate individual observations (simplified)
            n_control = 1000  # Simplified - would use actual group sizes
            n_treatment = 1000
            
            if "rate" in metric_name:
                # For rates/proportions, use chi-square test simulation
                control_successes = int(control_value * n_control)
                treatment_successes = int(treatment_value * n_treatment)
                
                # Create contingency table
                observed = np.array([
                    [control_successes, n_control - control_successes],
                    [treatment_successes, n_treatment - treatment_successes]
                ])
                
                try:
                    chi2, p_value, _, _ = chi2_contingency(observed)
                    test_statistic = chi2
                except:
                    p_value = 0.5
                    test_statistic = 0
                    
            else:
                # For continuous metrics, simulate t-test
                # Generate samples around the mean values
                control_samples = np.random.normal(control_value, control_value * 0.2, n_control)
                treatment_samples = np.random.normal(treatment_value, treatment_value * 0.2, n_treatment)
                
                try:
                    test_statistic, p_value = ttest_ind(treatment_samples, control_samples)
                except:
                    p_value = 0.5
                    test_statistic = 0
            
            results[metric_name] = {
                "p_value": float(p_value),
                "test_statistic": float(test_statistic),
                "is_significant": p_value < 0.05,
                "is_primary": metric_name == primary_metric
            }
        
        return results
    
    def _calculate_effect_sizes(self, control_metrics, treatment_metrics) -> Dict[str, float]:
        """Calculate effect sizes for metrics."""
        effect_sizes = {}
        
        for metric_name in control_metrics.keys():
            control_value = control_metrics[metric_name]
            treatment_value = treatment_metrics[metric_name]
            
            if control_value > 0:
                effect_size = (treatment_value - control_value) / control_value
            else:
                effect_size = 0
                
            effect_sizes[metric_name] = effect_size
        
        return effect_sizes
    
    def _calculate_confidence_intervals(
        self, 
        control_metrics, 
        treatment_metrics, 
        confidence_level
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for effect sizes."""
        intervals = {}
        
        for metric_name in control_metrics.keys():
            control_value = control_metrics[metric_name]
            treatment_value = treatment_metrics[metric_name]
            
            # Simplified CI calculation
            effect = treatment_value - control_value
            se = abs(effect * 0.1)  # Simplified standard error
            
            alpha = 1 - confidence_level
            z_score = norm.ppf(1 - alpha / 2)
            
            margin = z_score * se
            lower = effect - margin
            upper = effect + margin
            
            intervals[metric_name] = (lower, upper)
        
        return intervals
    
    def analyze_results(self, results: ExperimentResults) -> Dict[str, Any]:
        """Analyze experiment results and provide recommendations.
        
        Args:
            results: Experiment results
            
        Returns:
            Analysis and recommendations
        """
        analysis = {
            "experiment_id": results.experiment_id,
            "summary": {},
            "recommendations": [],
            "risks": [],
            "next_steps": []
        }
        
        # Analyze primary metric
        primary_metrics = [k for k, v in results.statistical_significance.items() if v.get("is_primary", False)]
        
        if primary_metrics:
            primary_metric = primary_metrics[0]
            effect_size = results.effect_sizes[primary_metric]
            is_significant = results.statistical_significance[primary_metric]["is_significant"]
            p_value = results.statistical_significance[primary_metric]["p_value"]
            
            analysis["summary"] = {
                "primary_metric": primary_metric,
                "effect_size": effect_size,
                "is_significant": is_significant,
                "p_value": p_value,
                "control_value": results.control_metrics[primary_metric],
                "treatment_value": results.treatment_metrics[primary_metric]
            }
            
            # Generate recommendations
            if is_significant and effect_size > 0:
                analysis["recommendations"].append("Внедрить изменение - статистически значимое улучшение")
            elif is_significant and effect_size < 0:
                analysis["recommendations"].append("Отклонить изменение - статистически значимое ухудшение")
            else:
                analysis["recommendations"].append("Продолжить тестирование или пересмотреть гипотезу")
            
            # Identify risks
            if abs(effect_size) > 0.2:
                analysis["risks"].append("Большой размер эффекта может указывать на ошибки в данных")
            
            # Next steps
            if not is_significant:
                analysis["next_steps"].append("Увеличить размер выборки или продолжительность теста")
        
        return analysis
    
    def export_experiment_plan(
        self, 
        experiment: ExperimentDesign, 
        filepath: Optional[str] = None
    ) -> str:
        """Export experiment plan to JSON.
        
        Args:
            experiment: Experiment design
            filepath: Optional file path
            
        Returns:
            JSON string of experiment plan
        """
        plan = {
            "experiment_id": experiment.experiment_id,
            "hypothesis": {
                "name": experiment.hypothesis.name,
                "description": experiment.hypothesis.description,
                "target_metric": experiment.hypothesis.target_metric,
                "expected_effect_size": experiment.hypothesis.expected_effect_size,
                "affected_personas": experiment.hypothesis.affected_personas,
                "feature_description": experiment.hypothesis.feature_description
            },
            "design": {
                "sample_size": experiment.sample_size,
                "control_size": experiment.control_size,
                "treatment_size": experiment.treatment_size,
                "test_duration_days": experiment.test_duration_days,
                "confidence_level": experiment.confidence_level,
                "statistical_power": experiment.statistical_power
            },
            "created_at": experiment.created_at
        }
        
        json_str = json.dumps(plan, ensure_ascii=False, indent=2)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Exported experiment plan to {filepath}")
        
        return json_str