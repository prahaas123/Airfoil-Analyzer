# ✈️ Airfoil Analyzer Dashboard

An interactive web dashboard built with Streamlit that computes airfoil polars locally using [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil).

## How it Works

1. **Interactive Selection** — Use the sidebar to filter out airfoils, and the dashboard automatically updates all plots.
2. **Airfoil Polars** — Uses `neuralfoil` to predict aerodynamic polar data (Cl, Cd, Cm vs. Alpha) for each airfoil at a given Reynolds number.
3. **Plots** — Generates a plots showing Cl vs. Alpha, Cd vs. Alpha, Cm vs. Alpha, L/D vs. Alpha, and the Drag Polar (Cl vs. Cd) for visual performance comparison.
4. **Pareto Front** — Scatters all selected airfoils in (Max L/D, Max Cl) space. The optimal Pareto front is drawn automatically, allowing you to identify the most efficient airfoils.
5. **Geometry Parsing** — Automatically fetches airfoil coordinates from the [UIUC Airfoil Data Site](https://m-selig.ae.illinois.edu/ads/coord_database.html), and plots them.
