# airfoil-analyzer

Scrapes airfoil polars from [AirfoilTools](http://airfoiltools.com/), plots aerodynamic performance across a filtered set of airfoils, and generates a Pareto front of lift vs. efficiency.

## How it Works

1. **Search** — queries AirfoilTools for airfoils within a specified thickness and camber range
2. **Fetch** — downloads XFoil polar data (Cl, Cd, Cm vs. Alpha) at a given Reynolds number
3. **Plot polars** — 2×3 grid showing Cl, Cd, Cm, L/D vs. Alpha, and the drag polar
4. **Plot Pareto front** — scatter of all airfoils in (L/D max, Cl max) space with the Pareto-optimal front highlighted
