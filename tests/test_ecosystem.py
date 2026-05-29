"""Tests for ecosystem_conservation.py — food web conservation analysis."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecosystem_conservation import (
    FoodWeb, ythan_estuary_foodweb, cascade_model_foodweb, niche_model_foodweb,
    build_graph_laplacian, compute_conservation_score, compute_trophic_levels,
    experiment_stability_vs_conservation, experiment_invasion,
    experiment_extinction_cascade, experiment_trophic_level_analysis,
    run_synthetic_analysis,
)


class TestFoodWeb:
    def test_ythan_basic(self):
        fw = ythan_estuary_foodweb()
        assert fw.n_species == 94
        assert len(fw.interactions) > 0
        assert len(fw.biomass) == 94

    def test_cascade_model(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        assert fw.n_species == 20
        assert 0 < fw.connectance < 1

    def test_niche_model(self):
        fw = niche_model_foodweb(20, 0.15, seed=42)
        assert fw.n_species == 20
        assert len(fw.interactions) > 0

    def test_connectance_range(self):
        fw = cascade_model_foodweb(30, 0.2, seed=1)
        assert 0 < fw.connectance <= 1.0


class TestLaplacian:
    def test_build_laplacian_symmetric(self):
        fw = cascade_model_foodweb(15, 0.2, seed=42)
        L = build_graph_laplacian(fw)
        assert L.shape == (15, 15)
        assert np.allclose(L, L.T)

    def test_laplacian_row_sums_approx_zero(self):
        fw = cascade_model_foodweb(15, 0.2, seed=42)
        L = build_graph_laplacian(fw)
        assert np.allclose(L.sum(axis=1), np.zeros(15), atol=1e-8)

    def test_laplacian_psd(self):
        fw = cascade_model_foodweb(15, 0.2, seed=42)
        L = build_graph_laplacian(fw)
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-10)


class TestConservationScore:
    def test_score_in_range(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        L = build_graph_laplacian(fw)
        score = compute_conservation_score(L, fw.biomass)
        assert 0 <= score <= 1.0 + 1e-10

    def test_score_deterministic(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        L = build_graph_laplacian(fw)
        s1 = compute_conservation_score(L, fw.biomass)
        s2 = compute_conservation_score(L, fw.biomass)
        assert np.isclose(s1, s2)

    def test_zero_biomass_returns_zero(self):
        fw = cascade_model_foodweb(10, 0.2, seed=42)
        L = build_graph_laplacian(fw)
        score = compute_conservation_score(L, np.zeros(10))
        assert score == 0.0


class TestTrophicLevels:
    def test_basal_species_level_one(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        levels = compute_trophic_levels(fw)
        # Species with no prey (basal) should be level 1
        basal = np.where(fw.adjacency.sum(axis=1) == 0)[0]
        for sp in basal:
            assert levels[sp] == 1.0

    def test_trophic_levels_positive(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        levels = compute_trophic_levels(fw)
        assert np.all(levels >= 1.0)

    def test_trophic_levels_count(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        levels = compute_trophic_levels(fw)
        assert len(levels) == 20


class TestExperiments:
    def test_stability_experiment(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        result = experiment_stability_vs_conservation(fw)
        assert 'original' in result
        assert 'reduced_links' in result
        assert 'added_links' in result
        for key in result:
            assert 'conservation' in result[key]
            assert 0 <= result[key]['conservation'] <= 1.0 + 1e-10

    def test_invasion_experiment(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        result = experiment_invasion(fw)
        assert 'original_conservation' in result
        assert 'post_invasion' in result
        assert 'change' in result
        assert isinstance(result['n_prey'], int)

    def test_extinction_cascade(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        result = experiment_extinction_cascade(fw, n_steps=5)
        assert 'initial_conservation' in result
        assert 'final_conservation' in result
        assert len(result['conservation_trace']) > 1

    def test_trophic_level_analysis(self):
        fw = cascade_model_foodweb(20, 0.15, seed=42)
        result = experiment_trophic_level_analysis(fw)
        assert 'n_levels' in result
        assert 'level_analysis' in result

    def test_synthetic_analysis(self):
        result = run_synthetic_analysis(n_webs=10)
        assert len(result['conservation']) == 10
        assert 'correlations' in result
        assert 0 <= result['conservation_mean'] <= 1.0
