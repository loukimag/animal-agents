"""
Post-hoc analysis for ecosystem simulation outputs.

This module does not modify or run simulations. It interprets outputs from
run_simulation(config) and produces structured, reproducible explanations.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class OutcomeEvidence:
    label: str
    evidence: Dict[str, float]
    justification: str


@dataclass
class AnalysisResult:
    labels: List[OutcomeEvidence]
    summary: str
    detailed: List[str]
    json: Dict


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stdev(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _trend_slope(series: List[float]) -> float:
    if len(series) < 2:
        return 0.0
    n = len(series)
    x_mean = (n - 1) / 2.0
    y_mean = _mean(series)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(series))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return float(num / den) if den else 0.0


def _variance_ratio(series: List[float]) -> float:
    mean = _mean(series)
    return _stdev(series) / mean if mean else 0.0


def _window(series: List[float], fraction: float = 0.2) -> List[float]:
    if not series:
        return []
    size = max(1, int(len(series) * fraction))
    return series[-size:]


def _dominant_periodicity(series: List[float]) -> float:
    if len(series) < 10:
        return 0.0
    diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
    sign_changes = sum(
        1 for i in range(len(diffs) - 1) if diffs[i] == 0 or diffs[i] * diffs[i + 1] < 0
    )
    return sign_changes / max(1, len(series) - 2)


def classify_outcomes(results: Dict) -> List[OutcomeEvidence]:
    ts = results.get("time_series", [])
    if not ts:
        return [
            OutcomeEvidence(
                label="no_data",
                evidence={"steps": 0},
                justification="No time series data available.",
            )
        ]

    prey = [row["prey"] for row in ts]
    predators = [row["predators"] for row in ts]
    competitors = [row["competitors"] for row in ts]
    plants = [row["plants_available"] for row in ts]
    steps = [row["step"] for row in ts]

    labels: List[OutcomeEvidence] = []
    extinction = results.get("extinction_steps", {})
    last_step = steps[-1]

    if extinction:
        first_ext = min(extinction.values())
        labels.append(
            OutcomeEvidence(
                label="extinction_event",
                evidence={"first_extinction_step": float(first_ext)},
                justification=f"At least one species extinct at step {first_ext}.",
            )
        )

    end_prey = prey[-1]
    end_pred = predators[-1]
    end_comp = competitors[-1]
    if end_prey == 0 or end_pred == 0:
        labels.append(
            OutcomeEvidence(
                label="collapse",
                evidence={"prey_end": float(end_prey), "predators_end": float(end_pred)},
                justification="At least one trophic level collapsed by the end.",
            )
        )

    prey_var_ratio = _variance_ratio(_window(prey))
    pred_var_ratio = _variance_ratio(_window(predators))
    if prey_var_ratio < 0.2 and pred_var_ratio < 0.2 and end_prey > 0 and end_pred > 0:
        labels.append(
            OutcomeEvidence(
                label="equilibrium",
                evidence={"prey_var_ratio": prey_var_ratio, "pred_var_ratio": pred_var_ratio},
                justification="Low variance in final window for prey and predators.",
            )
        )
    else:
        periodicity = _dominant_periodicity(_window(prey))
        if periodicity > 0.25 and end_prey > 0 and end_pred > 0:
            labels.append(
                OutcomeEvidence(
                    label="oscillation",
                    evidence={"prey_periodicity": periodicity},
                    justification="Frequent sign changes in prey growth indicate oscillation.",
                )
            )

    if end_comp > 0 and extinction.get("competitor") is None:
        comp_trend = _trend_slope(_window(competitors))
        labels.append(
            OutcomeEvidence(
                label="invasion_persistence",
                evidence={"competitor_end": float(end_comp), "competitor_trend": comp_trend},
                justification="Competitors persist through the final window.",
            )
        )

    if end_comp == 0 and "competitor" in extinction:
        labels.append(
            OutcomeEvidence(
                label="invasion_failure",
                evidence={"competitor_extinction_step": float(extinction["competitor"])},
                justification="Competitors went extinct after introduction.",
            )
        )

    if end_prey > 0 and end_pred > 0:
        ratio = end_pred / max(1, end_prey)
        if ratio < 0.1 or ratio > 1.0:
            labels.append(
                OutcomeEvidence(
                    label="trophic_imbalance",
                    evidence={"predator_prey_ratio": ratio},
                    justification="Predator-prey ratio outside typical bounds.",
                )
            )

    if _mean(plants) == 0:
        labels.append(
            OutcomeEvidence(
                label="resource_collapse",
                evidence={"avg_plants": 0.0},
                justification="Plant availability collapsed across the run.",
            )
        )

    return labels


def attribute_causes(results: Dict, scenario_params: Optional[Dict] = None) -> List[str]:
    config = results.get("config")
    ts = results.get("time_series", [])
    if not ts:
        return ["No data for causal attribution."]

    prey = [row["prey"] for row in ts]
    predators = [row["predators"] for row in ts]
    plants = [row["plants_available"] for row in ts]
    end_prey = prey[-1]
    end_pred = predators[-1]

    factors = []
    if config:
        if getattr(config, "plant_regrow_delay", 5) > 6:
            factors.append(
                "Slow plant regrowth likely reduced food availability, consistent with regrowth delay."
            )
        if getattr(config, "initial_predators", 0) > getattr(config, "initial_prey", 1):
            factors.append(
                "High predator-to-prey ratio increases early predation pressure."
            )
        if getattr(config, "learning_enabled", False):
            factors.append(
                "Learning-enabled movement can amplify foraging success via local energy rewards."
            )
        if getattr(config, "competitor_intro_step", 0) < getattr(config, "max_steps", 1) // 2:
            factors.append(
                "Early competitor introduction increases resource competition in mid-run."
            )

    if end_prey == 0:
        factors.append(
            "Prey extinction implies sustained energy deficits relative to reproduction costs."
        )
    if end_pred == 0:
        factors.append(
            "Predator extinction suggests insufficient prey density to offset movement and energy decay."
        )
    if _mean(plants) < max(plants) * 0.3:
        factors.append(
            "Low mean plant availability indicates resource bottleneck driving energy shortfalls."
        )

    if scenario_params:
        for key, value in scenario_params.items():
            factors.append(f"Scenario parameter {key} set to {value}.")

    if not factors:
        factors.append("No dominant causal factors detected in parameterization.")
    return factors


def build_explanations(results: Dict, scenario_params: Optional[Dict] = None) -> AnalysisResult:
    labels = classify_outcomes(results)
    causes = attribute_causes(results, scenario_params=scenario_params)

    summary_parts = []
    if labels:
        summary_parts.append("Outcomes: " + ", ".join(l.label for l in labels))
    summary = "; ".join(summary_parts) if summary_parts else "No outcomes detected."

    detailed = []
    for label in labels:
        detailed.append(f"{label.label}: {label.justification} Evidence: {label.evidence}")
    for cause in causes:
        detailed.append(f"Cause: {cause}")

    json_payload = {
        "labels": [
            {"label": l.label, "evidence": l.evidence, "justification": l.justification}
            for l in labels
        ],
        "causes": causes,
        "summary": summary,
    }

    return AnalysisResult(labels=labels, summary=summary, detailed=detailed, json=json_payload)
