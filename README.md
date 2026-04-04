# airfoil-analyzer

A fast airfoil analysis tool that computes airfoil polars locally using [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), plots aerodynamic performance across a filtered set of airfoils, and generates a Pareto front of lift vs. efficiency.

## How it Works

1. **Filter** — Parses a local JSON database (`airfoils.json`) to find candidate airfoils within a specified thickness and camber range.
2. **Compute** — Uses `neuralfoil` and to rapidly predict aerodynamic polar data (Cl, Cd, Cm vs. Alpha) for each airfoil at a given Reynolds number.
3. **Plot polars** — Generates a grid showing Cl, Cd, Cm, L/D vs. Alpha, and the drag polar so you can visually compare performance.
4. **Plot Pareto front** — Scatters all analyzed airfoils in (L/D max, Cl max) space with the optimal Pareto front highlighted, allowing you to easily identify the most efficient airfoils for your specific lift requirements.
