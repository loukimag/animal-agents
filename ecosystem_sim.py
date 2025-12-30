"""
Ecosystem simulation with delayed effects using Mesa.

Run:
  python ecosystem_sim.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid


@dataclass
class SpeciesParams:
    initial_energy: int
    move_cost: int
    eat_gain: int
    reproduce_threshold: int
    reproduce_cost: int
    reproduce_cooldown: int
    gestation_delay: int
    offspring_energy: int


class SimpleScheduler:
    def __init__(self, model: Model):
        self.model = model
        self.agents: List[Agent] = []
        self.steps = 0

    def add(self, agent: Agent) -> None:
        self.agents.append(agent)

    def remove(self, agent: Agent) -> None:
        if agent in self.agents:
            self.agents.remove(agent)

    def step(self) -> None:
        self.steps += 1
        agent_list = list(self.agents)
        self.model.random.shuffle(agent_list)
        for agent in agent_list:
            if agent in self.agents:
                agent.step()


class Plant(Agent):
    def __init__(self, unique_id: int, model: Model, regrow_delay: int):
        super().__init__(model)
        self.unique_id = unique_id
        self.regrow_delay = regrow_delay
        self.regrow_timer = 0
        self.available = True

    def step(self) -> None:
        if not self.available:
            self.regrow_timer -= 1
            if self.regrow_timer <= 0:
                self.available = True

    def consume(self) -> bool:
        if self.available:
            self.available = False
            self.regrow_timer = self.regrow_delay
            return True
        return False


class BaseAnimal(Agent):
    species_name = "base"

    def __init__(self, unique_id: int, model: Model, params: SpeciesParams):
        super().__init__(model)
        self.unique_id = unique_id
        self.params = params
        self.energy = params.initial_energy
        self.cooldown = 0
        self.gestation_queue: List[int] = []

    def step(self) -> None:
        self._handle_gestation()
        self._move()
        self._eat()
        self._attempt_reproduce()
        self._age()

    def _handle_gestation(self) -> None:
        if not self.gestation_queue:
            return
        for i in range(len(self.gestation_queue)):
            self.gestation_queue[i] -= 1
        while self.gestation_queue and self.gestation_queue[0] <= 0:
            self.gestation_queue.pop(0)
            self._spawn_offspring()

    def _move(self) -> None:
        self.energy -= self.params.move_cost
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.model.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def _torus_distance(self, a, b) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dx = min(dx, self.model.grid.width - dx)
        dy = min(dy, self.model.grid.height - dy)
        return max(dx, dy)

    def _local_sense(self, pos, vision: int):
        neighborhood = self.model.grid.get_neighborhood(
            pos, moore=True, include_center=True, radius=vision
        )
        counts = {"plants": 0, "prey": 0, "predators": 0}
        nearest = {"plants": None, "prey": None, "predators": None}
        for npos in neighborhood:
            contents = self.model.grid.get_cell_list_contents([npos])
            if not contents:
                continue
            dist = self._torus_distance(pos, npos)
            for obj in contents:
                if isinstance(obj, Plant) and obj.available:
                    counts["plants"] += 1
                    if nearest["plants"] is None or dist < nearest["plants"]:
                        nearest["plants"] = dist
                elif isinstance(obj, Prey):
                    counts["prey"] += 1
                    if nearest["prey"] is None or dist < nearest["prey"]:
                        nearest["prey"] = dist
                elif isinstance(obj, Predator):
                    counts["predators"] += 1
                    if nearest["predators"] is None or dist < nearest["predators"]:
                        nearest["predators"] = dist
        return counts, nearest

    def _move_with_scoring(
        self,
        radius: int,
        vision: int,
        score_fn,
        allow_stay: bool = True,
    ) -> None:
        candidates = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=allow_stay, radius=radius
        )
        best_score = None
        best_positions = []
        for pos in candidates:
            counts, nearest = self._local_sense(pos, vision=vision)
            score = score_fn(counts, nearest)
            if best_score is None or score > best_score:
                best_score = score
                best_positions = [pos]
            elif score == best_score:
                best_positions.append(pos)
        if not best_positions:
            return
        new_position = self.model.random.choice(best_positions)
        if new_position != self.pos:
            self.energy -= self.params.move_cost
            self.model.grid.move_agent(self, new_position)

    def _eat(self) -> None:
        # Override in subclasses
        return

    def _attempt_reproduce(self) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        if self.energy >= self.params.reproduce_threshold:
            self.energy -= self.params.reproduce_cost
            self.cooldown = self.params.reproduce_cooldown
            self.gestation_queue.append(self.params.gestation_delay)

    def _spawn_offspring(self) -> None:
        offspring = self.__class__(self.model._next_id(), self.model, self.params)
        offspring.energy = self.params.offspring_energy
        self.model.grid.place_agent(offspring, self.pos)
        self.model.schedule.add(offspring)

    def _age(self) -> None:
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            self.model.register_extinction(self.species_name)


class Prey(BaseAnimal):
    species_name = "prey"

    def _move(self) -> None:
        def score(counts, nearest):
            plant_bonus = counts["plants"] * 2
            predator_penalty = counts["predators"] * 8
            if nearest["plants"] is not None:
                plant_bonus += max(0, 4 - nearest["plants"]) * 2
            if nearest["predators"] is not None:
                predator_penalty += max(0, 7 - nearest["predators"]) * 8
                # Favor positions that increase distance from predators.
                plant_bonus += min(6, nearest["predators"]) * 2
            return plant_bonus - predator_penalty

        # Panic mode: larger radius when predators are close.
        base_radius = 4
        base_vision = 4
        _, nearest = self._local_sense(self.pos, vision=base_vision)
        panic = nearest["predators"] is not None and nearest["predators"] <= 2
        if panic:
            base_radius = 6
            base_vision = 6
        self._move_with_scoring(
            radius=base_radius,
            vision=base_vision,
            score_fn=score,
            allow_stay=True,
        )
        # Extra burst move during panic to flee faster.
        if panic:
            self._move_with_scoring(
                radius=4,
                vision=4,
                score_fn=score,
                allow_stay=False,
            )

    def _eat(self) -> None:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cellmates:
            if isinstance(obj, Plant) and obj.consume():
                self.energy += self.params.eat_gain
                return

    def _attempt_reproduce(self) -> None:
        predator_count = self.model.count_species(Predator)
        prey_count = self.model.count_species(Prey)
        if prey_count >= 300:
            if self.cooldown > 0:
                self.cooldown -= 1
            return
        if predator_count > 0 and prey_count > int(predator_count * 1.2):
            if self.cooldown > 0:
                self.cooldown -= 1
            return
        dynamic_threshold = self.params.reproduce_threshold
        if prey_count <= 200:
            dynamic_threshold = max(1, self.params.reproduce_threshold - 2)
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        if self.energy >= dynamic_threshold:
            self.energy -= self.params.reproduce_cost
            self.cooldown = self.params.reproduce_cooldown
            self.gestation_queue.append(self.params.gestation_delay)


class Predator(BaseAnimal):
    species_name = "predator"

    def _move(self) -> None:
        def score(counts, nearest):
            prey_bonus = counts["prey"] * 1
            crowd_penalty = counts["predators"] * 2
            if nearest["prey"] is not None:
                prey_bonus += max(0, 2 - nearest["prey"]) * 1
            if nearest["predators"] is not None:
                crowd_penalty += max(0, 3 - nearest["predators"]) * 1
            return prey_bonus - crowd_penalty

        # Shorter vision and weaker scoring makes predators less effective.
        self._move_with_scoring(radius=2, vision=2, score_fn=score, allow_stay=True)

    def _eat(self) -> None:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        prey_agents = [a for a in cellmates if isinstance(a, Prey)]
        if prey_agents:
            prey = self.model.random.choice(prey_agents)
            self.model.grid.remove_agent(prey)
            self.model.schedule.remove(prey)
            self.energy += self.params.eat_gain

    def _attempt_reproduce(self) -> None:
        # Prevent predator growth when prey are scarce or below parity.
        prey_count = self.model.count_species(Prey)
        predator_count = self.model.count_species(Predator)
        if predator_count >= 300:
            if self.cooldown > 0:
                self.cooldown -= 1
            return
        if prey_count < 10 or prey_count < predator_count:
            if self.cooldown > 0:
                self.cooldown -= 1
            return
        dynamic_threshold = self.params.reproduce_threshold
        if predator_count <= 200:
            dynamic_threshold = max(1, self.params.reproduce_threshold - 2)
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        if self.energy >= dynamic_threshold:
            self.energy -= self.params.reproduce_cost
            self.cooldown = self.params.reproduce_cooldown
            self.gestation_queue.append(self.params.gestation_delay)

    def _age(self) -> None:
        prey_count = self.model.count_species(Prey)
        if prey_count < 100:
            self.energy -= 4
        elif prey_count < 200:
            self.energy -= 2
        super()._age()


class Competitor(BaseAnimal):
    species_name = "competitor"

    def _move(self) -> None:
        def score(counts, nearest):
            plant_bonus = counts["plants"] * 2
            predator_penalty = counts["predators"] * 2
            if nearest["plants"] is not None:
                plant_bonus += max(0, 4 - nearest["plants"]) * 2
            if nearest["predators"] is not None:
                predator_penalty += max(0, 4 - nearest["predators"]) * 2
            return plant_bonus - predator_penalty

        self._move_with_scoring(radius=4, vision=4, score_fn=score, allow_stay=True)

    def _eat(self) -> None:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cellmates:
            if isinstance(obj, Plant) and obj.consume():
                self.energy += self.params.eat_gain
                return


class EcosystemModel(Model):
    def __init__(
        self,
        width: int = 60,
        height: int = 60,
        initial_plants: int = 480,
        initial_prey: int = 180,
        initial_predators: int = 120,
        plant_regrow_delay: int = 5,
        prey_params: Optional[SpeciesParams] = None,
        predator_params: Optional[SpeciesParams] = None,
        competitor_params: Optional[SpeciesParams] = None,
        competitor_intro_step: int = 30,
        initial_competitors: int = 30,
        seed: Optional[int] = 42,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.random = random.Random(seed)
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = SimpleScheduler(self)
        self._id_counter = 0
        self.max_steps = max_steps
        self.competitor_intro_step = competitor_intro_step
        self.initial_competitors = initial_competitors
        self.initial_plants = initial_plants
        self.extinction_step: Dict[str, int] = {}

        self.prey_params = prey_params or SpeciesParams(
            initial_energy=30,
            move_cost=1,
            eat_gain=14,
            reproduce_threshold=11,
            reproduce_cost=2,
            reproduce_cooldown=3,
            gestation_delay=3,
            offspring_energy=18,
        )
        self.predator_params = predator_params or SpeciesParams(
            initial_energy=30,
            move_cost=1,
            eat_gain=18,
            reproduce_threshold=18,
            reproduce_cost=4,
            reproduce_cooldown=5,
            gestation_delay=3,
            offspring_energy=18,
        )
        self.competitor_params = competitor_params or SpeciesParams(
            initial_energy=20,
            move_cost=1,
            eat_gain=10,
            reproduce_threshold=12,
            reproduce_cost=2,
            reproduce_cooldown=3,
            gestation_delay=2,
            offspring_energy=14,
        )

        self._init_plants(initial_plants, plant_regrow_delay)
        self._init_animals(initial_prey, initial_predators)

        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.schedule.steps,
                "plants_available": lambda m: m.count_plants_available(),
                "prey": lambda m: m.count_species(Prey),
                "predators": lambda m: m.count_species(Predator),
                "competitors": lambda m: m.count_species(Competitor),
                "prey_health": lambda m: m.average_energy(Prey),
                "predator_health": lambda m: m.average_energy(Predator),
                "competitor_health": lambda m: m.average_energy(Competitor),
            }
        )

    def _init_plants(self, count: int, regrow_delay: int) -> None:
        for _ in range(count):
            plant = Plant(self._next_id(), self, regrow_delay)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(plant, (x, y))
            self.schedule.add(plant)

    def _init_animals(self, prey_count: int, predator_count: int) -> None:
        for _ in range(prey_count):
            prey = Prey(self._next_id(), self, self.prey_params)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(prey, (x, y))
            self.schedule.add(prey)
        for _ in range(predator_count):
            predator = Predator(self._next_id(), self, self.predator_params)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(predator, (x, y))
            self.schedule.add(predator)

    def _init_competitors(self) -> None:
        for _ in range(self.initial_competitors):
            competitor = Competitor(self._next_id(), self, self.competitor_params)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(competitor, (x, y))
            self.schedule.add(competitor)

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def count_species(self, species_cls: type) -> int:
        return sum(1 for a in self.schedule.agents if isinstance(a, species_cls))

    def count_plants_available(self) -> int:
        return sum(
            1
            for a in self.schedule.agents
            if isinstance(a, Plant) and a.available
        )

    def average_energy(self, species_cls: type) -> float:
        energies = [a.energy for a in self.schedule.agents if isinstance(a, species_cls)]
        if not energies:
            return 0.0
        return float(sum(energies) / len(energies))

    def register_extinction(self, species_name: str) -> None:
        if species_name not in self.extinction_step:
            self.extinction_step[species_name] = self.schedule.steps

    def step(self) -> None:
        if self.schedule.steps == self.competitor_intro_step:
            self._init_competitors()
        self.datacollector.collect(self)
        self.schedule.step()

    def run_model(self) -> pd.DataFrame:
        for _ in range(self.max_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_ess(df: pd.DataFrame, max_resource: int) -> float:
    populations = df[["prey", "predators", "competitors"]].sum(axis=1)
    mean_pop = populations.mean()
    pop_std = populations.std(ddof=0) if mean_pop > 0 else 0.0
    pop_stability = clamp(1.0 - (pop_std / mean_pop if mean_pop else 1.0))

    resource_sustainability = clamp(
        df["plants_available"].mean() / max_resource if max_resource else 0.0
    )

    extinction_cols = ["prey", "predators", "competitors"]
    extinction_times = []
    for col in extinction_cols:
        extinct_steps = df.index[df[col] == 0].tolist()
        if extinct_steps:
            extinction_times.append(extinct_steps[0])
    if extinction_times:
        survival = clamp(min(extinction_times) / max(1, len(df) - 1))
    else:
        survival = 1.0

    return float((pop_stability + resource_sustainability + survival) / 3.0)


def run_simulation() -> None:
    model = EcosystemModel()
    df = model.run_model()
    df.to_csv("ecosystem_results.csv", index=False)

    ess = compute_ess(df, max_resource=model.initial_plants)
    print(f"Ecosystem Stability Score (ESS): {ess:.3f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["prey"], label="Prey")
    plt.plot(df["step"], df["predators"], label="Predators")
    plt.plot(df["step"], df["competitors"], label="Competitors")
    plt.title("Population vs Time")
    plt.xlabel("Step")
    plt.ylabel("Population")
    plt.legend()
    plt.tight_layout()
    plt.savefig("population_vs_time.png", dpi=150)

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["prey_health"], label="Prey Health")
    plt.plot(df["step"], df["predator_health"], label="Predator Health")
    plt.plot(df["step"], df["competitor_health"], label="Competitor Health")
    plt.title("Average Energy (Health) vs Time")
    plt.xlabel("Step")
    plt.ylabel("Average Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("health_vs_time.png", dpi=150)

    plt.figure(figsize=(10, 4))
    plt.plot(df["step"], df["plants_available"], color="green", label="Plants")
    plt.title("Resource Level vs Time")
    plt.xlabel("Step")
    plt.ylabel("Available Plants")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resources_vs_time.png", dpi=150)


def _get_agent_positions(model: EcosystemModel):
    plants_available_x = []
    plants_available_y = []
    plants_unavailable_x = []
    plants_unavailable_y = []
    prey_x = []
    prey_y = []
    predator_x = []
    predator_y = []
    competitor_x = []
    competitor_y = []

    for agent in model.schedule.agents:
        x, y = agent.pos
        if isinstance(agent, Plant):
            if agent.available:
                plants_available_x.append(x)
                plants_available_y.append(y)
            else:
                plants_unavailable_x.append(x)
                plants_unavailable_y.append(y)
        elif isinstance(agent, Prey):
            prey_x.append(x)
            prey_y.append(y)
        elif isinstance(agent, Predator):
            predator_x.append(x)
            predator_y.append(y)
        elif isinstance(agent, Competitor):
            competitor_x.append(x)
            competitor_y.append(y)

    return {
        "plants_available": (plants_available_x, plants_available_y),
        "plants_unavailable": (plants_unavailable_x, plants_unavailable_y),
        "prey": (prey_x, prey_y),
        "predators": (predator_x, predator_y),
        "competitors": (competitor_x, competitor_y),
    }


def run_visual_simulation(
    max_steps: int = 500,
    show_population_plot: bool = True,
) -> None:
    model = EcosystemModel(max_steps=max_steps)

    grid_figsize = (6, 6)
    if show_population_plot:
        fig, (ax_grid, ax_pop) = plt.subplots(1, 2, figsize=(11, 5))
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=grid_figsize)
        ax_pop = None

    ax_grid.set_xlim(-0.5, model.grid.width - 0.5)
    ax_grid.set_ylim(-0.5, model.grid.height - 0.5)
    ax_grid.set_aspect("equal")
    ax_grid.set_title("Ecosystem Grid")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    # Plant squares
    plants_available = ax_grid.scatter(
        [],
        [],
        marker="s",
        s=35,
        c="#5cb85c",
        label="Plants (available)",
        linewidths=0,
    )
    plants_unavailable = ax_grid.scatter(
        [],
        [],
        marker="s",
        s=35,
        c="#2f7d32",
        label="Plants (unavailable)",
        linewidths=0,
    )

    # Animal circles
    prey_scatter = ax_grid.scatter(
        [],
        [],
        marker="o",
        s=45,
        c="#2b6cb0",
        label="Prey",
        linewidths=0,
    )
    predator_scatter = ax_grid.scatter(
        [],
        [],
        marker="o",
        s=50,
        c="#c9302c",
        label="Predators",
        linewidths=0,
    )
    competitor_scatter = ax_grid.scatter(
        [],
        [],
        marker="o",
        s=45,
        c="#f0ad4e",
        label="Competitors",
        linewidths=0,
    )

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", label="Plants (available)",
               markerfacecolor="#5cb85c", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Plants (unavailable)",
               markerfacecolor="#2f7d32", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Prey",
               markerfacecolor="#2b6cb0", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Predators",
               markerfacecolor="#c9302c", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Competitors",
               markerfacecolor="#f0ad4e", markersize=8),
    ]
    ax_grid.legend(handles=legend_handles, loc="upper right", fontsize=8)

    counter_text = ax_grid.text(
        0.02,
        0.98,
        "",
        transform=ax_grid.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    if ax_pop:
        ax_pop.set_title("Population Over Time")
        ax_pop.set_xlabel("Step")
        ax_pop.set_ylabel("Count")
        pop_lines = {
            "prey": ax_pop.plot([], [], color="#2b6cb0", label="Prey")[0],
            "predators": ax_pop.plot([], [], color="#c9302c", label="Predators")[0],
            "competitors": ax_pop.plot([], [], color="#f0ad4e", label="Competitors")[0],
        }
        ax_health = ax_pop.twinx()
        ax_health.set_ylabel("Avg Energy")
        health_lines = {
            "prey_health": ax_health.plot(
                [],
                [],
                color="#2b6cb0",
                linestyle="--",
                label="Prey Health",
            )[0],
            "predator_health": ax_health.plot(
                [],
                [],
                color="#c9302c",
                linestyle="--",
                label="Predator Health",
            )[0],
            "competitor_health": ax_health.plot(
                [],
                [],
                color="#f0ad4e",
                linestyle="--",
                label="Competitor Health",
            )[0],
        }
        ax_pop.legend(loc="upper right", fontsize=8)
        ax_health.legend(loc="lower right", fontsize=8)
        pop_history = {
            "step": [],
            "prey": [],
            "predators": [],
            "competitors": [],
            "prey_health": [],
            "predator_health": [],
            "competitor_health": [],
        }

    def _to_offsets(points):
        if points[0]:
            return np.column_stack(points)
        return np.empty((0, 2))

    def init():
        plants_available.set_offsets(np.empty((0, 2)))
        plants_unavailable.set_offsets(np.empty((0, 2)))
        prey_scatter.set_offsets(np.empty((0, 2)))
        predator_scatter.set_offsets(np.empty((0, 2)))
        competitor_scatter.set_offsets(np.empty((0, 2)))
        counter_text.set_text("")
        if ax_pop:
            for line in pop_lines.values():
                line.set_data([], [])
            for line in health_lines.values():
                line.set_data([], [])
        return (
            plants_available,
            plants_unavailable,
            prey_scatter,
            predator_scatter,
            competitor_scatter,
            counter_text,
        )

    def update(_frame):
        model.step()
        positions = _get_agent_positions(model)

        plants_available.set_offsets(_to_offsets(positions["plants_available"]))
        plants_unavailable.set_offsets(_to_offsets(positions["plants_unavailable"]))
        prey_scatter.set_offsets(_to_offsets(positions["prey"]))
        predator_scatter.set_offsets(_to_offsets(positions["predators"]))
        competitor_scatter.set_offsets(_to_offsets(positions["competitors"]))

        prey_count = model.count_species(Prey)
        predator_count = model.count_species(Predator)
        competitor_count = model.count_species(Competitor)
        plant_count = model.count_plants_available()
        counter_text.set_text(
            f"Step: {model.schedule.steps}\n"
            f"Plants: {plant_count}\n"
            f"Prey: {prey_count}\n"
            f"Predators: {predator_count}\n"
            f"Competitors: {competitor_count}"
        )

        if ax_pop:
            pop_history["step"].append(model.schedule.steps)
            pop_history["prey"].append(prey_count)
            pop_history["predators"].append(predator_count)
            pop_history["competitors"].append(competitor_count)
            for name, line in pop_lines.items():
                line.set_data(pop_history["step"], pop_history[name])
            pop_history["prey_health"].append(model.average_energy(Prey))
            pop_history["predator_health"].append(model.average_energy(Predator))
            pop_history["competitor_health"].append(model.average_energy(Competitor))
            for name, line in health_lines.items():
                line.set_data(pop_history["step"], pop_history[name])
            ax_pop.relim()
            ax_pop.autoscale_view()
            ax_health.relim()
            ax_health.autoscale_view()

        return (
            plants_available,
            plants_unavailable,
            prey_scatter,
            predator_scatter,
            competitor_scatter,
            counter_text,
        )

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=max_steps,
        interval=150,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ecosystem simulation")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Run a real-time visualization instead of batch mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to animate in visual mode.",
    )
    parser.add_argument(
        "--no-pop-plot",
        action="store_true",
        help="Disable the live population plot in visual mode.",
    )
    args = parser.parse_args()

    if args.visual:
        run_visual_simulation(
            max_steps=args.steps,
            show_population_plot=not args.no_pop_plot,
        )
    else:
        run_simulation()
