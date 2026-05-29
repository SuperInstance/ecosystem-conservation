# ecosystem-conservation

**Biodiversity spectral analysis — conservation ratio of biomass in food web networks via the Tension Graph Laplacian.**

Applies the conservation spectral framework to ecological food webs. Species are nodes, predation links are edges, biomass is the node attribute. Measures how well biomass is "conserved" (smooth) on the food web graph — high conservation means the ecosystem is structurally stable.

## What This Gives You

- **Food web modeling** — directed graphs with species, biomass, trophic levels
- **Conservation of biomass** — measures structural stability via graph Laplacian
- **Keystone species detection** — nodes whose removal most damages conservation
- **Invasive species simulation** — inject a new predator, watch CR drop
- **Real-world food webs** — includes empirical datasets (Chesapeake Bay, Ythan Estuary)
- **Publication-quality plots** — conservation timeseries, heatmaps, network visualizations

## Quick Start

```bash
pip install numpy scipy matplotlib
python ecosystem_conservation.py
```

Outputs go to `results/` with PNG plots.

## How It Fits

Part of the SuperInstance ecosystem:

- **[conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python)** — Core SDK used here
- **ecosystem-conservation** — Application to ecology (this repo)

## License

MIT
