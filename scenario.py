"""
Scenario abstraction layer for the ecosystem simulator.

This module does NOT run simulations. It only translates high-level scenarios
into deterministic, validated simulation configs compatible with the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ecosystem_sim import LearningParams, SpeciesParams


BIOME_MAP = {
    "arid_desert": {
        "terrain_weights": {"plain": 0.75, "forest": 0.05, "hill": 0.15, "mountain": 0.05},
        "plant_regrow_mult": 1.8,
        "plants_mult": 0.6,
    },
    "temperate_forest": {
        "terrain_weights": {"plain": 0.35, "forest": 0.45, "hill": 0.15, "mountain": 0.05},
        "plant_regrow_mult": 0.9,
        "plants_mult": 1.2,
    },
    "grassland": {
        "terrain_weights": {"plain": 0.65, "forest": 0.15, "hill": 0.15, "mountain": 0.05},
        "plant_regrow_mult": 1.0,
        "plants_mult": 1.0,
    },
    "mountainous": {
        "terrain_weights": {"plain": 0.25, "forest": 0.15, "hill": 0.35, "mountain": 0.25},
        "plant_regrow_mult": 1.3,
        "plants_mult": 0.8,
    },
    "tundra": {
        "terrain_weights": {"plain": 0.55, "forest": 0.1, "hill": 0.25, "mountain": 0.1},
        "plant_regrow_mult": 1.6,
        "plants_mult": 0.7,
    },
}

RESOURCE_LEVEL = {
    "low": {"plants_mult": 0.7, "plant_regrow_mult": 1.3},
    "medium": {"plants_mult": 1.0, "plant_regrow_mult": 1.0},
    "high": {"plants_mult": 1.3, "plant_regrow_mult": 0.85},
}

SOIL_NUTRIENTS = {
    "poor": {"plants_mult": 0.8, "plant_regrow_mult": 1.2},
    "normal": {"plants_mult": 1.0, "plant_regrow_mult": 1.0},
    "rich": {"plants_mult": 1.2, "plant_regrow_mult": 0.9},
}

CLIMATE_CONSTRAINTS = {
    "drought": {"plants_mult": 0.75, "plant_regrow_mult": 1.4},
    "wet_season": {"plants_mult": 1.15, "plant_regrow_mult": 0.9},
    "cold": {"reproduce_cooldown_mult": 1.2, "gestation_delay_mult": 1.1},
    "heatwave": {"move_cost_mult": 1.1},
}

SPECIES_TRAIT_KEYS = (
    "initial_energy",
    "move_cost",
    "eat_gain",
    "reproduce_threshold",
    "reproduce_cost",
    "reproduce_cooldown",
    "gestation_delay",
    "offspring_energy",
)


@dataclass
class TraitMultipliers:
    initial_energy: float = 1.0
    move_cost: float = 1.0
    eat_gain: float = 1.0
    reproduce_threshold: float = 1.0
    reproduce_cost: float = 1.0
    reproduce_cooldown: float = 1.0
    gestation_delay: float = 1.0
    offspring_energy: float = 1.0


@dataclass
class SpeciesScenario:
    initial_count: int
    traits: TraitMultipliers = field(default_factory=TraitMultipliers)
    learning_enabled: Optional[bool] = None


@dataclass
class InvasionSpec:
    species: str
    step: int
    count: int


@dataclass
class EnvironmentScenario:
    biome: str
    resource_level: str = "medium"
    soil_nutrients: str = "normal"
    climate_constraints: List[str] = field(default_factory=list)


@dataclass
class ScenarioSpec:
    name: str
    environment: EnvironmentScenario
    prey: SpeciesScenario
    predators: SpeciesScenario
    competitors: SpeciesScenario
    initial_plants: int
    width: int = 60
    height: int = 60
    seed: int = 42
    max_steps: int = 2000
    learning_enabled: bool = True
    invasions: List[InvasionSpec] = field(default_factory=list)
    auto_correct: bool = True


@dataclass
class ScenarioTranslation:
    config: Dict
    explanation: List[str]
    warnings: List[str]
    risk_factors: List[str]


def _apply_multiplier(value: int, mult: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * mult)))


def _apply_traits(base: SpeciesParams, mult: TraitMultipliers) -> SpeciesParams:
    return SpeciesParams(
        initial_energy=_apply_multiplier(base.initial_energy, mult.initial_energy, minimum=1),
        move_cost=_apply_multiplier(base.move_cost, mult.move_cost, minimum=0),
        eat_gain=_apply_multiplier(base.eat_gain, mult.eat_gain, minimum=1),
        reproduce_threshold=_apply_multiplier(
            base.reproduce_threshold, mult.reproduce_threshold, minimum=1
        ),
        reproduce_cost=_apply_multiplier(base.reproduce_cost, mult.reproduce_cost, minimum=1),
        reproduce_cooldown=_apply_multiplier(
            base.reproduce_cooldown, mult.reproduce_cooldown, minimum=1
        ),
        gestation_delay=_apply_multiplier(base.gestation_delay, mult.gestation_delay, minimum=1),
        offspring_energy=_apply_multiplier(
            base.offspring_energy, mult.offspring_energy, minimum=1
        ),
    )


def _validate_scenario(spec: ScenarioSpec) -> None:
    if spec.environment.biome not in BIOME_MAP:
        raise ValueError(f"Unknown biome: {spec.environment.biome}")
    if spec.environment.resource_level not in RESOURCE_LEVEL:
        raise ValueError(f"Unknown resource_level: {spec.environment.resource_level}")
    if spec.environment.soil_nutrients not in SOIL_NUTRIENTS:
        raise ValueError(f"Unknown soil_nutrients: {spec.environment.soil_nutrients}")
    for constraint in spec.environment.climate_constraints:
        if constraint not in CLIMATE_CONSTRAINTS:
            raise ValueError(f"Unknown climate constraint: {constraint}")
    if spec.width <= 0 or spec.height <= 0:
        raise ValueError("width and height must be positive")
    if spec.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if spec.initial_plants <= 0:
        raise ValueError("initial_plants must be > 0")
    if spec.predators.initial_count > 0 and spec.prey.initial_count <= 0:
        raise ValueError("predators require prey > 0")
    for invasion in spec.invasions:
        if invasion.step < 0 or invasion.step > spec.max_steps:
            raise ValueError("invasion step out of bounds")
        if invasion.count <= 0:
            raise ValueError("invasion count must be > 0")


def translate_scenario(spec: ScenarioSpec) -> ScenarioTranslation:
    _validate_scenario(spec)
    explanation: List[str] = []
    warnings: List[str] = []

    biome = BIOME_MAP[spec.environment.biome]
    resource = RESOURCE_LEVEL[spec.environment.resource_level]
    soil = SOIL_NUTRIENTS[spec.environment.soil_nutrients]

    plant_mult = biome["plants_mult"] * resource["plants_mult"] * soil["plants_mult"]
    regrow_mult = biome["plant_regrow_mult"] * resource["plant_regrow_mult"] * soil["plant_regrow_mult"]

    for constraint in spec.environment.climate_constraints:
        regrow_mult *= CLIMATE_CONSTRAINTS[constraint].get("plant_regrow_mult", 1.0)
        plant_mult *= CLIMATE_CONSTRAINTS[constraint].get("plants_mult", 1.0)

    plant_regrow_delay = _apply_multiplier(5, regrow_mult, minimum=1)
    initial_plants = _apply_multiplier(spec.initial_plants, plant_mult, minimum=1)

    explanation.append(
        f"Biome '{spec.environment.biome}' sets terrain weights and regrow multiplier {biome['plant_regrow_mult']}."
    )
    explanation.append(
        f"Resource level '{spec.environment.resource_level}' multiplies plants by {resource['plants_mult']}."
    )
    explanation.append(
        f"Soil nutrients '{spec.environment.soil_nutrients}' multiplies plants by {soil['plants_mult']}."
    )
    if spec.environment.climate_constraints:
        explanation.append(
            f"Climate constraints {spec.environment.climate_constraints} applied to plant regrowth/availability."
        )

    base_prey = SpeciesParams(30, 1, 14, 11, 2, 3, 3, 18)
    base_predator = SpeciesParams(30, 1, 18, 18, 4, 5, 3, 18)
    base_competitor = SpeciesParams(20, 1, 10, 12, 2, 3, 2, 14)

    prey_params = _apply_traits(base_prey, spec.prey.traits)
    predator_params = _apply_traits(base_predator, spec.predators.traits)
    competitor_params = _apply_traits(base_competitor, spec.competitors.traits)

    if spec.prey.initial_count <= 0:
        if spec.auto_correct:
            warnings.append("Prey count was <= 0; auto-corrected to 10.")
            prey_count = 10
        else:
            raise ValueError("prey count must be > 0")
    else:
        prey_count = spec.prey.initial_count

    predator_count = spec.predators.initial_count
    if predator_count > 0 and prey_count <= 0:
        raise ValueError("predators require prey > 0")

    competitor_intro_step = 0
    competitor_count = spec.competitors.initial_count
    for invasion in spec.invasions:
        if invasion.species != "competitor":
            raise ValueError("Only competitor invasions are supported by the engine.")
        competitor_intro_step = invasion.step
        competitor_count = invasion.count
        explanation.append(
            f"Invasion adds competitors at step {invasion.step} with count {invasion.count}."
        )

    config = {
        "width": spec.width,
        "height": spec.height,
        "initial_plants": initial_plants,
        "initial_prey": prey_count,
        "initial_predators": predator_count,
        "plant_regrow_delay": plant_regrow_delay,
        "competitor_intro_step": competitor_intro_step,
        "initial_competitors": competitor_count,
        "terrain_weights": biome["terrain_weights"],
        "prey_params": prey_params,
        "predator_params": predator_params,
        "competitor_params": competitor_params,
        "learning_enabled": spec.learning_enabled,
        "seed": spec.seed,
        "max_steps": spec.max_steps,
    }

    risk_factors = []
    if initial_plants < spec.initial_plants * 0.7:
        risk_factors.append("resource_bottleneck")
    if predator_count > prey_count:
        risk_factors.append("predator_pressure")
    if spec.environment.biome in ("arid_desert", "tundra"):
        risk_factors.append("high_extinction_risk")

    return ScenarioTranslation(
        config=config,
        explanation=explanation,
        warnings=warnings,
        risk_factors=risk_factors,
    )


def expand_seeds(translation: ScenarioTranslation, seeds: List[int]) -> List[Dict]:
    configs = []
    for seed in seeds:
        cfg = dict(translation.config)
        cfg["seed"] = seed
        configs.append(cfg)
    return configs


def summarize_translation(translation: ScenarioTranslation) -> Dict:
    return {
        "config": translation.config,
        "warnings": translation.warnings,
        "risk_factors": translation.risk_factors,
        "explanation": translation.explanation,
    }


PRESET_SCENARIOS = {
    "temperate_baseline": ScenarioSpec(
        name="temperate_baseline",
        environment=EnvironmentScenario(biome="temperate_forest"),
        prey=SpeciesScenario(initial_count=180),
        predators=SpeciesScenario(initial_count=120),
        competitors=SpeciesScenario(initial_count=20),
        initial_plants=480,
        seed=42,
        max_steps=300,
    ),
    "arid_invasion": ScenarioSpec(
        name="arid_invasion",
        environment=EnvironmentScenario(
            biome="arid_desert",
            resource_level="low",
            soil_nutrients="poor",
            climate_constraints=["drought"],
        ),
        prey=SpeciesScenario(initial_count=140),
        predators=SpeciesScenario(initial_count=80),
        competitors=SpeciesScenario(initial_count=10),
        initial_plants=300,
        invasions=[InvasionSpec(species="competitor", step=80, count=40)],
        seed=7,
        max_steps=300,
    ),
    "rainforest_wolf": ScenarioSpec(
        name="rainforest_wolf",
        environment=EnvironmentScenario(biome="rainforest", resource_level="high"),
        prey=SpeciesScenario(initial_count=200),
        predators=SpeciesScenario(initial_count=20),
        competitors=SpeciesScenario(initial_count=10),
        initial_plants=600,
        seed=11,
        max_steps=300,
    ),
}
