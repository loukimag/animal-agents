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

    def _eat(self) -> None:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cellmates:
            if isinstance(obj, Plant) and obj.consume():
                self.energy += self.params.eat_gain
                return


class Predator(BaseAnimal):
    species_name = "predator"

    def _eat(self) -> None:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        prey_agents = [a for a in cellmates if isinstance(a, Prey)]
        if prey_agents:
            prey = self.model.random.choice(prey_agents)
            self.model.grid.remove_agent(prey)
            self.model.schedule.remove(prey)
            self.energy += self.params.eat_gain


class Competitor(BaseAnimal):
    species_name = "competitor"

    def _eat(self) -> None:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cellmates:
            if isinstance(obj, Plant) and obj.consume():
                self.energy += self.params.eat_gain
                return


class EcosystemModel(Model):
    def __init__(
        self,
        width: int = 25,
        height: int = 25,
        initial_plants: int = 200,
        initial_prey: int = 60,
        initial_predators: int = 25,
        plant_regrow_delay: int = 12,
        prey_params: Optional[SpeciesParams] = None,
        predator_params: Optional[SpeciesParams] = None,
        competitor_params: Optional[SpeciesParams] = None,
        competitor_intro_step: int = 200,
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
            initial_energy=12,
            move_cost=1,
            eat_gain=4,
            reproduce_threshold=18,
            reproduce_cost=6,
            reproduce_cooldown=6,
            gestation_delay=4,
            offspring_energy=8,
        )
        self.predator_params = predator_params or SpeciesParams(
            initial_energy=15,
            move_cost=1,
            eat_gain=8,
            reproduce_threshold=22,
            reproduce_cost=8,
            reproduce_cooldown=8,
            gestation_delay=6,
            offspring_energy=10,
        )
        self.competitor_params = competitor_params or SpeciesParams(
            initial_energy=10,
            move_cost=1,
            eat_gain=5,
            reproduce_threshold=16,
            reproduce_cost=6,
            reproduce_cooldown=4,
            gestation_delay=3,
            offspring_energy=7,
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

    plt.figure(figsize=(10, 4))
    plt.plot(df["step"], df["plants_available"], color="green", label="Plants")
    plt.title("Resource Level vs Time")
    plt.xlabel("Step")
    plt.ylabel("Available Plants")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resources_vs_time.png", dpi=150)


if __name__ == "__main__":
    run_simulation()
