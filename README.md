# Ecosystem Conservation

**Tension Graph Laplacian → Biomass Conservation in Food Webs**

Applies the Conservation Spectral framework to real ecological networks. Computes conservation ratios for food webs (Ythan Estuary, Cascade Mountains), runs extinction cascade simulations, keystone species detection, invasion impact analysis, and trophic-level conservation profiling. Includes both real food web models and synthetic Cascade/Niche model generators.

## The Aha Moment

Ecology is where conservation stops being abstract math and becomes *life or death*. When we compute α(G, biomass) for a food web, we're asking: "Does the Laplacian's smooth structure preserve the biomass distribution?" High conservation means the ecosystem's topology naturally maintains biomass patterns — removing any single species causes proportional ripple effects, not catastrophic cascades. Low conservation means the graph structure doesn't protect biomass — keystone species exist, and their removal is disproportionate. The spectral gap λ₂ predicts ecosystem resilience. This isn't analogy — it's the same math that describes music, code, and geometry, now applied to who eats whom.

## How to Use

```bash
pip install numpy scipy matplotlib

# Run full analysis on Ythan Estuary + Cascade food webs
python ecosystem_conservation.py
```

This generates ~19 visualization plots in `results/` covering:
- Conservation score comparisons across food webs
- Keystone species impact (removal → conservation drop)
- Extinction cascade dynamics
- Trophic level conservation profiles
- Invasion impact analysis
- Synthetic food web correlations (100 generated networks)

## Architecture

- **`FoodWeb` class**: Directed graph with species names, adjacency matrix, and biomass vector
- **Food web builders**: `ythan_estuary_foodweb()` (94 species), `cascades_foodweb()`, synthetic `cascade_model_foodweb()`, `niche_model_foodweb()`
- **Experiments**: Stability vs conservation, keystone detection, invasion simulation, extinction cascades, trophic analysis
- **Visualization suite**: 6+ plot types showing conservation metrics across ecological scenarios

### Key Functions

- `build_graph_laplacian(fw)` → Laplacian from food web adjacency
- `compute_conservation_score(L, biomass)` → α(G, biomass)
- `compute_trophic_levels(fw)` → Trophic level assignment
- `experiment_keystone(fw)` → Remove each species, measure conservation impact
- `experiment_extinction_cascade(fw)` → Sequential extinction dynamics

## Connection to the Conservation Spectral Framework

This is an **application** of the framework rather than an SDK implementation. It takes the core conservation ratio α(G,a) and applies it where a = biomass distribution and G = food web topology. The results connect spectral graph theory to ecological stability, demonstrating that the same Laplacian analysis used for music, code, and geometry also predicts ecosystem resilience.

## Related Repos

- [conservation-spectral-v2](https://github.com/SuperInstance/conservation-spectral-v2) — Reference implementation (Python)
- [conservation-geometry](https://github.com/SuperInstance/conservation-geometry) — Geometric visualizations of the framework
- [conservation-art](https://github.com/SuperInstance/conservation-art) — Artistic visualizations
- [code-conservation](https://github.com/SuperInstance/code-conservation) — Conservation in software dependency graphs

## Testing

```bash
pip install pytest numpy matplotlib scipy
pytest tests/ -v
```

Tests cover:
- Food web construction (Ythan Estuary, Cascade model, Niche model)
- Laplacian construction (symmetry, row sums, PSD)
- Conservation score (range, determinism, edge cases)
- Trophic level computation
- All experiment functions (stability, invasion, extinction, trophic analysis, synthetic)

## License

MIT

---

Part of the [OpenConstruct](https://github.com/SuperInstance) ecosystem — spectral graph theory meets ecology.
