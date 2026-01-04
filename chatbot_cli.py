"""
Small terminal control for chatbot testing.

Commands:
  run <scenario_name>
  seeds <n>
  show
  explain
  help
  quit
"""

from __future__ import annotations

import json

from analysis import build_explanations
from ecosystem_sim import run_simulation, SimulationConfig
from scenario import PRESET_SCENARIOS, ScenarioSpec, translate_scenario


SCENARIOS = PRESET_SCENARIOS


def _run_scenario(spec: ScenarioSpec):
    translation = translate_scenario(spec)
    config = SimulationConfig(**translation.config)
    results = run_simulation(config)
    return translation, results


def main() -> None:
    current = SCENARIOS["temperate_baseline"]
    last_results = None
    last_translation = None
    seeds = [current.seed]

    print("Chatbot test CLI. Type 'help' for commands.")
    while True:
        try:
            raw = input("eco> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit"):
            break
        if cmd == "help":
            print("Commands: run <scenario>, seeds <n>, show, explain, help, quit")
            print("Scenarios: " + ", ".join(SCENARIOS.keys()))
            continue
        if cmd == "show":
            print(f"Current scenario: {current.name}")
            print(json.dumps(current.__dict__, default=str, indent=2))
            continue
        if cmd == "seeds":
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: seeds <n>")
                continue
            count = int(parts[1])
            seeds = list(range(1, count + 1))
            print(f"Seed set: {seeds}")
            continue
        if cmd == "run":
            if len(parts) != 2:
                print("Usage: run <scenario_name>")
                continue
            name = parts[1]
            if name not in SCENARIOS:
                print(f"Unknown scenario: {name}")
                continue
            current = SCENARIOS[name]
            for seed in seeds:
                spec = current
                spec.seed = seed
                translation, results = _run_scenario(spec)
                last_results = results
                last_translation = translation
                print(f"Run {name} seed={seed} ESS={results['ess']:.3f}")
            continue
        if cmd == "explain":
            if not last_results:
                print("No results. Run a scenario first.")
                continue
            analysis = build_explanations(last_results, scenario_params=last_translation.config)
            print(analysis.summary)
            continue

        print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    main()
