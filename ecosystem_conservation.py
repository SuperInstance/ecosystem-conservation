"""
Ecosystem Conservation Analysis
Tension Graph Laplacian → Biomass Conservation in Food Webs

Analyzes how the Tension Graph Laplacian framework measures conservation
of total biomass in ecological food web networks.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# FOOD WEB DATA STRUCTURE
# =============================================================================

@dataclass
class FoodWeb:
    """Represents a food web as a directed graph."""
    name: str
    species_names: List[str]
    # adjacency: adjacency[i][j] = 1 means species i eats species j (i is predator, j is prey)
    adjacency: np.ndarray
    biomass: np.ndarray
    
    @property
    def n_species(self) -> int:
        return len(self.species_names)
    
    @property
    def interactions(self) -> List[Tuple[int, int]]:
        """Return list of (predator, prey) interactions."""
        rows, cols = np.where(self.adjacency > 0)
        return list(zip(rows.tolist(), cols.tolist()))
    
    @property
    def connectance(self) -> float:
        n = self.n_species
        return len(self.interactions) / (n * n) if n > 0 else 0.0


# =============================================================================
# FOOD WEB BUILDERS
# =============================================================================

def ythan_estuary_foodweb() -> FoodWeb:
    """
    Build a simplified version of the Ythan Estuary food web.
    Real Ythan has ~94 species; we build a structurally realistic approximation.
    """
    np.random.seed(123)
    n = 94
    names = [f"sp_{i}" for i in range(n)]
    
    # Create trophic groups: basal (0-30), intermediate (31-70), top (71-93)
    adj = np.zeros((n, n), dtype=float)
    
    # Basal species don't eat anyone; they have biomass from primary production
    # Intermediate species eat basal and other intermediate
    for i in range(31, 71):
        # Each intermediate eats 3-8 basal species
        n_prey = np.random.randint(3, 9)
        prey = np.random.choice(range(0, 31), size=min(n_prey, 30), replace=False)
        for p in prey:
            adj[i, p] = 1.0
        # Some eat other intermediate species (cannibalism/omnivory)
        if np.random.random() < 0.3:
            n_prey2 = np.random.randint(1, 3)
            prey2 = np.random.choice(range(31, 71), size=min(n_prey2, i-31), replace=False)
            for p in prey2:
                if p != i:
                    adj[i, p] = 1.0
    
    # Top predators eat intermediate and some basal
    for i in range(71, 94):
        n_prey = np.random.randint(2, 6)
        prey = np.random.choice(range(0, 71), size=min(n_prey, 70), replace=False)
        for p in prey:
            adj[i, p] = 1.0
    
    # Biomass: basal have high biomass, decreases up trophic levels
    biomass = np.zeros(n)
    biomass[0:31] = np.random.uniform(50, 200, 31)
    biomass[31:71] = np.random.uniform(10, 80, 40)
    biomass[71:94] = np.random.uniform(1, 30, 23)
    
    return FoodWeb(name="Ythan Estuary", species_names=names, adjacency=adj, biomass=biomass)


def cascades_foodweb() -> FoodWeb:
    """
    Build a simplified version of the Cascade Mountains food web.
    Real version has ~171 species.
    """
    np.random.seed(456)
    n = 171
    names = [f"sp_{i}" for i in range(n)]
    adj = np.zeros((n, n), dtype=float)
    
    # Trophic structure: basal (0-60), intermediate (61-140), top (141-170)
    # Cascade model: predator i can only eat species j where j < i (niche ordering)
    niche_values = np.random.uniform(0, 1, n)
    niche_order = np.argsort(niche_values)
    
    for rank, i in enumerate(niche_order):
        if rank < 60:  # basal
            continue
        # Number of prey
        n_prey = np.random.randint(2, 8)
        # Can only eat species with lower niche value
        candidates = [j for j in range(n) if niche_values[j] < niche_values[i]]
        if candidates:
            prey = np.random.choice(candidates, size=min(n_prey, len(candidates)), replace=False)
            for p in prey:
                adj[i, p] = 1.0
    
    biomass = np.zeros(n)
    for i in range(n):
        rank = np.where(niche_order == i)[0][0]
        biomass[i] = np.random.uniform(1, 200) * (1 - rank/n)
    
    return FoodWeb(name="Cascade Mountains", species_names=names, adjacency=adj, biomass=biomass)


def cascade_model_foodweb(S: int, C: float, seed: int = 42) -> FoodWeb:
    """Generate a food web using the cascade model (Cohen et al.)."""
    np.random.seed(seed)
    names = [f"sp_{i}" for i in range(S)]
    adj = np.zeros((S, S), dtype=float)
    
    # Assign random niche values and sort
    niche = np.random.uniform(0, 1, S)
    order = np.argsort(niche)  # order[k] has rank k
    
    # For each pair (i,j) where niche[i] > niche[j], link with probability C
    links = int(C * S * S)
    count = 0
    attempts = 0
    while count < links and attempts < links * 10:
        i, j = np.random.randint(0, S, 2)
        if niche[i] > niche[j] and adj[i, j] == 0:
            adj[i, j] = 1.0
            count += 1
        attempts += 1
    
    biomass = np.random.uniform(1, 100, S)
    return FoodWeb(name=f"Cascade Model (S={S}, C={C})", species_names=names, adjacency=adj, biomass=biomass)


def niche_model_foodweb(S: int, C: float, seed: int = 42) -> FoodWeb:
    """Generate a food web using the niche model (Williams & Martinez 2000)."""
    np.random.seed(seed)
    names = [f"sp_{i}" for i in range(S)]
    adj = np.zeros((S, S), dtype=float)
    
    # Assign niche values uniformly
    niche = np.random.uniform(0, 1, S)
    
    # Beta distribution parameter to achieve target connectance
    beta = 1.0 / (2.0 * C) - 1.0
    
    for i in range(S):
        # Diet range
        r_i = niche[i] * np.random.beta(1, beta)
        c_i = np.random.uniform(r_i / 2, min(niche[i], 1 - r_i / 2 + 1e-10))
        
        for j in range(S):
            if niche[j] >= c_i - r_i / 2 and niche[j] <= c_i + r_i / 2 and i != j:
                adj[i, j] = 1.0
    
    biomass = np.random.uniform(1, 100, S)
    return FoodWeb(name=f"Niche Model (S={S}, C={C})", species_names=names, adjacency=adj, biomass=biomass)


# =============================================================================
# CORE CONSERVATION ANALYSIS
# =============================================================================

def build_graph_laplacian(fw: FoodWeb) -> np.ndarray:
    """Build the tension graph Laplacian from a food web."""
    n = fw.n_species
    adj = fw.adjacency
    
    # Weighted adjacency: use biomass ratios as weights
    W = np.zeros((n, n), dtype=float)
    for i, j in fw.interactions:
        # Weight proportional to consumer's feeding on resource
        W[i, j] = fw.biomass[j] / (fw.biomass[i] + fw.biomass[j] + 1e-10)
    
    # Symmetrize for Laplacian
    W_sym = (W + W.T) / 2.0
    
    # Degree matrix
    D = np.diag(W_sym.sum(axis=1))
    
    # Laplacian
    L = D - W_sym
    
    return L


def compute_conservation_score(L: np.ndarray, biomass: np.ndarray) -> float:
    """
    Compute conservation score: how well does the graph Laplacian 
    preserve total biomass (sum of biomass = conservation of energy).
    
    Uses the Rayleigh quotient of the biomass vector with the Laplacian.
    Lower value = better conservation.
    We return 1 - normalized_quotient so higher = more conserved.
    """
    n = L.shape[0]
    b = biomass.copy()
    
    # Rayleigh quotient: b^T L b / (b^T b)
    numerator = b @ L @ b
    denominator = b @ b
    
    if denominator < 1e-15:
        return 0.0
    
    rq = numerator / denominator
    
    # Normalize by the maximum eigenvalue
    try:
        if sparse.issparse(L):
            L_dense = L.toarray()
        else:
            L_dense = L
        max_eig = np.max(np.abs(np.linalg.eigvalsh(L_dense)))
    except:
        max_eig = np.trace(L)
    
    if max_eig < 1e-15:
        return 1.0
    
    normalized_rq = rq / max_eig
    
    # Conservation score: 1 - normalized Rayleigh quotient
    # High conservation = low dissipation through the Laplacian
    return max(0.0, min(1.0, 1.0 - normalized_rq))


def compute_trophic_levels(fw: FoodWeb) -> np.ndarray:
    """Compute trophic levels using shortest path from basal species."""
    n = fw.n_species
    adj = fw.adjacency
    
    # Find basal species (no prey)
    has_prey = adj.sum(axis=1) > 0
    is_basal = ~has_prey
    
    levels = np.ones(n)  # Default trophic level = 1 (basal)
    levels[is_basal] = 1.0
    
    # BFS-like approach: trophic level = 1 + max(prey levels)
    for iteration in range(n):
        changed = False
        for i in range(n):
            prey_indices = np.where(adj[i] > 0)[0]
            if len(prey_indices) > 0:
                new_level = 1.0 + np.max(levels[prey_indices])
                if abs(new_level - levels[i]) > 0.001:
                    levels[i] = new_level
                    changed = True
        if not changed:
            break
    
    return levels


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def experiment_stability_vs_conservation(fw: FoodWeb) -> Dict:
    """Analyze how network modifications affect conservation."""
    L = build_graph_laplacian(fw)
    results = {}
    
    # Original
    cons_orig = compute_conservation_score(L, fw.biomass)
    results['original'] = {
        'conservation': cons_orig,
        'n_species': fw.n_species,
        'n_interactions': len(fw.interactions),
        'connectance': fw.connectance
    }
    
    # Remove 10% of interactions randomly
    adj_mod = fw.adjacency.copy()
    interactions = fw.interactions
    n_remove = max(1, len(interactions) // 10)
    np.random.seed(999)
    remove_idx = np.random.choice(len(interactions), size=n_remove, replace=False)
    for idx in remove_idx:
        i, j = interactions[idx]
        adj_mod[i, j] = 0.0
    fw_mod = FoodWeb(name=fw.name + " (-10% links)", species_names=fw.species_names,
                     adjacency=adj_mod, biomass=fw.biomass)
    L_mod = build_graph_laplacian(fw_mod)
    results['reduced_links'] = {
        'conservation': compute_conservation_score(L_mod, fw_mod.biomass),
        'n_species': fw_mod.n_species,
        'n_interactions': len(fw_mod.interactions),
        'connectance': fw_mod.connectance
    }
    
    # Add 10% more interactions
    adj_add = fw.adjacency.copy()
    possible = [(i, j) for i in range(fw.n_species) for j in range(fw.n_species)
                if adj_add[i, j] == 0 and i != j]
    n_add = max(1, len(interactions) // 10)
    if len(possible) >= n_add:
        np.random.seed(998)
        add_pairs = [possible[i] for i in np.random.choice(len(possible), size=n_add, replace=False)]
        for i, j in add_pairs:
            adj_add[i, j] = 1.0
    fw_add = FoodWeb(name=fw.name + " (+10% links)", species_names=fw.species_names,
                     adjacency=adj_add, biomass=fw.biomass)
    L_add = build_graph_laplacian(fw_add)
    results['added_links'] = {
        'conservation': compute_conservation_score(L_add, fw_add.biomass),
        'n_species': fw_add.n_species,
        'n_interactions': len(fw_add.interactions),
        'connectance': fw_add.connectance
    }
    
    return results


def experiment_keystone(fw: FoodWeb, max_species: int = 50) -> Dict:
    """Identify keystone species by removal impact on conservation."""
    L = build_graph_laplacian(fw)
    orig_cons = compute_conservation_score(L, fw.biomass)
    
    drops = []
    n_test = min(max_species, fw.n_species)
    
    for sp_idx in range(n_test):
        # Remove species and all its links
        surviving = [i for i in range(fw.n_species) if i != sp_idx]
        adj_sub = fw.adjacency[np.ix_(surviving, surviving)]
        bio_sub = fw.biomass[surviving]
        
        fw_sub = FoodWeb(name=fw.name, species_names=[fw.species_names[i] for i in surviving],
                        adjacency=adj_sub, biomass=bio_sub)
        L_sub = build_graph_laplacian(fw_sub)
        new_cons = compute_conservation_score(L_sub, bio_sub)
        drops.append((sp_idx, orig_cons - new_cons))
    
    # Sort by biggest drop (most keystone)
    drops.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'original_conservation': orig_cons,
        'top_keystone': drops[:10],
        'most_redundant': drops[-5:],
        'all_drops': drops
    }


def experiment_invasion(fw: FoodWeb) -> Dict:
    """Simulate an invasive species entering the food web."""
    np.random.seed(777)
    n = fw.n_species
    
    # Add invasive species
    adj_new = np.zeros((n + 1, n + 1), dtype=float)
    adj_new[:n, :n] = fw.adjacency
    
    # Invasive eats 2-5 native species
    n_prey = np.random.randint(2, 6)
    prey = np.random.choice(n, size=n_prey, replace=False)
    for p in prey:
        adj_new[n, p] = 1.0
    
    # Some native predators may eat the invasive
    n_native_pred = np.random.randint(1, 4)
    predators = np.random.choice(n, size=n_native_pred, replace=False)
    for pred in predators:
        adj_new[pred, n] = 1.0
    
    biomass_new = np.append(fw.biomass, np.random.uniform(5, 50))
    
    fw_inv = FoodWeb(name=fw.name + " (+invasive)",
                    species_names=fw.species_names + ["invasive"],
                    adjacency=adj_new, biomass=biomass_new)
    
    L_inv = build_graph_laplacian(fw_inv)
    post_cons = compute_conservation_score(L_inv, biomass_new)
    
    L_orig = build_graph_laplacian(fw)
    orig_cons = compute_conservation_score(L_orig, fw.biomass)
    
    return {
        'original_conservation': orig_cons,
        'post_invasion': post_cons,
        'change': post_cons - orig_cons,
        'n_prey': n_prey,
        'n_native_predators': n_native_pred
    }


def experiment_extinction_cascade(fw: FoodWeb, n_steps: int = 20) -> Dict:
    """Simulate sequential species extinctions and track conservation."""
    L = build_graph_laplacian(fw)
    orig_cons = compute_conservation_score(L, fw.biomass)
    
    surviving = list(range(fw.n_species))
    adj = fw.adjacency.copy()
    biomass = fw.biomass.copy()
    
    conservation_trace = [orig_cons]
    n_species_trace = [fw.n_species]
    collapse_step = None
    
    np.random.seed(42)
    
    for step in range(n_steps):
        if len(surviving) <= 2:
            break
        
        # Remove a random species
        rm_idx = np.random.randint(0, len(surviving))
        rm_species = surviving.pop(rm_idx)
        
        # Secondary extinctions: species with no prey left
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, sp in enumerate(surviving):
                prey_of = np.where(adj[sp, :] > 0)[0]
                surviving_prey = [p for p in prey_of if p in surviving]
                has_basal = any(adj[sp, p] > 0 and adj[:, p].sum() == adj[sp, p]
                               for p in surviving_prey)
                # Simple rule: if a non-basal species loses all prey, it goes extinct
                is_basal = adj[sp, :].sum() == 0
                if not is_basal and len(surviving_prey) == 0:
                    to_remove.append(i)
            for i in sorted(to_remove, reverse=True):
                surviving.pop(i)
                changed = True
        
        if len(surviving) <= 1:
            break
        
        # Compute conservation of remaining
        adj_sub = adj[np.ix_(surviving, surviving)]
        bio_sub = biomass[surviving]
        
        if len(surviving) < 3:
            break
        
        fw_sub = FoodWeb(name=fw.name,
                        species_names=[fw.species_names[i] for i in surviving],
                        adjacency=adj_sub, biomass=bio_sub)
        L_sub = build_graph_laplacian(fw_sub)
        cons = compute_conservation_score(L_sub, bio_sub)
        
        conservation_trace.append(cons)
        n_species_trace.append(len(surviving))
        
        if collapse_step is None and cons < orig_cons * 0.5:
            collapse_step = step + 1
    
    final_cons = conservation_trace[-1]
    
    return {
        'initial_conservation': orig_cons,
        'final_conservation': final_cons,
        'collapse_step': collapse_step,
        'conservation_trace': conservation_trace,
        'n_species_trace': n_species_trace
    }


def experiment_trophic_level_analysis(fw: FoodWeb) -> Dict:
    """Analyze conservation by trophic level."""
    levels = compute_trophic_levels(fw)
    unique_levels = np.unique(np.round(levels, 1))
    
    level_analysis = {}
    for lv in sorted(unique_levels):
        mask = np.abs(levels - lv) < 0.15
        sp_indices = np.where(mask)[0]
        if len(sp_indices) == 0:
            continue
        
        bio_sub = fw.biomass[sp_indices]
        
        # Build sub-web for this trophic level
        if len(sp_indices) >= 2:
            adj_sub = fw.adjacency[np.ix_(sp_indices, sp_indices)]
            fw_sub = FoodWeb(name=fw.name,
                           species_names=[fw.species_names[i] for i in sp_indices],
                           adjacency=adj_sub, biomass=bio_sub)
            L_sub = build_graph_laplacian(fw_sub)
            cons = compute_conservation_score(L_sub, bio_sub)
        else:
            cons = None
        
        level_analysis[lv] = {
            'n_species': len(sp_indices),
            'conservation': cons,
            'mean_biomass': float(np.mean(bio_sub))
        }
    
    return {
        'n_levels': len(level_analysis),
        'level_analysis': level_analysis
    }


def run_synthetic_analysis(n_webs: int = 100) -> Dict:
    """Generate and analyze many random food webs."""
    conservation = []
    connectance = []
    complexity = []
    models = []
    
    for i in range(n_webs):
        S = np.random.randint(15, 60)
        C = np.random.uniform(0.05, 0.3)
        model = np.random.choice(['cascade', 'niche'])
        
        if model == 'cascade':
            fw = cascade_model_foodweb(S, C, seed=i * 7 + 13)
        else:
            fw = niche_model_foodweb(S, C, seed=i * 7 + 13)
        
        L = build_graph_laplacian(fw)
        cons = compute_conservation_score(L, fw.biomass)
        
        conservation.append(cons)
        connectance.append(fw.connectance)
        complexity.append(fw.connectance * S)  # May's complexity
        models.append(model)
    
    conservation = np.array(conservation)
    connectance = np.array(connectance)
    complexity = np.array(complexity)
    
    # Correlations
    r_pearson_con, _ = pearsonr(conservation, connectance)
    r_spearman_con, _ = spearmanr(conservation, connectance)
    r_pearson_cmp, _ = pearsonr(conservation, complexity)
    r_spearman_cmp, _ = spearmanr(conservation, complexity)
    
    cascade_mask = np.array([m == 'cascade' for m in models])
    
    return {
        'conservation': conservation.tolist(),
        'connectance': connectance.tolist(),
        'complexity': complexity.tolist(),
        'model': models,
        'conservation_mean': float(np.mean(conservation)),
        'conservation_std': float(np.std(conservation)),
        'cascade_mean_conservation': float(np.mean(conservation[cascade_mask])),
        'niche_mean_conservation': float(np.mean(conservation[~cascade_mask])),
        'correlations': {
            'conservation_vs_connectance_pearson': float(r_pearson_con),
            'conservation_vs_connectance_spearman': float(r_spearman_con),
            'conservation_vs_complexity_pearson': float(r_pearson_cmp),
            'conservation_vs_complexity_spearman': float(r_spearman_cmp),
        }
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_conservation_comparison(stab_results: Dict, fw_name: str):
    """Bar chart comparing conservation across scenarios."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(stab_results.keys())
    vals = [stab_results[k]['conservation'] for k in labels]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(range(len(labels)), vals, color=colors[:len(labels)], edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', ' ').title() for l in labels], fontsize=9)
    ax.set_ylabel('Conservation Score', fontsize=12)
    ax.set_title(f'{fw_name}: Stability vs Conservation', fontsize=14)
    plt.tight_layout()
    safe_name = fw_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    path = os.path.join(OUTPUT_DIR, f'{safe_name}_conservation.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_keystone_impact(keystone_results: Dict, fw_name: str):
    """Plot keystone species impact rankings."""
    all_drops = keystone_results['all_drops']
    if not all_drops:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    species_idx = [d[0] for d in all_drops]
    drops = [d[1] for d in all_drops]
    
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in drops]
    ax.bar(range(len(species_idx)), drops, color=colors, alpha=0.7)
    ax.set_xlabel('Species Index', fontsize=12)
    ax.set_ylabel('Conservation Drop', fontsize=12)
    ax.set_title(f'{fw_name}: Keystone Species Impact', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    safe_name = fw_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    path = os.path.join(OUTPUT_DIR, f'{safe_name}_keystone.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_extinction_cascade(ext_results: Dict, fw_name: str):
    """Plot conservation decline during extinction cascade."""
    fig, ax = plt.subplots(figsize=(8, 5))
    trace = ext_results['conservation_trace']
    n_sp = ext_results['n_species_trace']
    
    ax.plot(range(len(trace)), trace, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax.set_xlabel('Extinction Step', fontsize=12)
    ax.set_ylabel('Conservation Score', fontsize=12)
    ax.set_title(f'{fw_name}: Extinction Cascade', fontsize=14)
    
    if ext_results['collapse_step'] is not None:
        cs = ext_results['collapse_step']
        ax.axvline(x=cs, color='red', linestyle='--', alpha=0.5, label=f'Collapse (step {cs})')
        ax.legend()
    
    # Add secondary x-axis for species count
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(n_sp)))
    ax2.set_xticklabels([str(n) for n in n_sp], fontsize=7)
    ax2.set_xlabel('Remaining Species', fontsize=10)
    
    plt.tight_layout()
    safe_name = fw_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    path = os.path.join(OUTPUT_DIR, f'{safe_name}_extinction.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_trophic_level_conservation(tl_results: Dict, fw_name: str):
    """Bar chart of conservation by trophic level."""
    analysis = tl_results['level_analysis']
    levels = sorted(analysis.keys())
    cons_vals = []
    labels = []
    for lv in levels:
        c = analysis[lv].get('conservation')
        if c is not None:
            cons_vals.append(c)
            labels.append(f"{lv} (n={analysis[lv]['n_species']})")
    if not cons_vals:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), cons_vals, color='#3498db', edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, cons_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Conservation Score', fontsize=12)
    ax.set_title(f'{fw_name}: Conservation by Trophic Level', fontsize=14)
    plt.tight_layout()
    safe_name = fw_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    path = os.path.join(OUTPUT_DIR, f'{safe_name}_trophic.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_invasion_impact(invasion_results: Dict, fw_name: str):
    """Plot invasion impact comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [invasion_results['original_conservation'], invasion_results['post_invasion']]
    labels = ['Pre-Invasion', 'Post-Invasion']
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Conservation Score', fontsize=12)
    ax.set_title(f'{fw_name}: Invasion Impact\nChange = {invasion_results["change"]:+.4f}', fontsize=13)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    safe_name = fw_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    path = os.path.join(OUTPUT_DIR, f'{safe_name}_invasion.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_synthetic_correlations(synthetic_results: Dict):
    """Scatter plots: conservation vs connectance and vs complexity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    conservation = np.array(synthetic_results['conservation'])
    connectance = np.array(synthetic_results['connectance'])
    complexity = np.array(synthetic_results['complexity'])
    models = synthetic_results['model']
    
    ax = axes[0]
    for m in set(models):
        mask = [x == m for x in models]
        ax.scatter(np.array(connectance)[mask], np.array(conservation)[mask],
                  alpha=0.6, s=30, label=m.capitalize())
    ax.set_xlabel('Connectance', fontsize=12)
    ax.set_ylabel('Conservation Score', fontsize=12)
    ax.set_title(f'Conservation vs Connectance\nPearson r = {synthetic_results["correlations"]["conservation_vs_connectance_pearson"]:.3f}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for m in set(models):
        mask = [x == m for x in models]
        ax.scatter(np.array(complexity)[mask], np.array(conservation)[mask],
                  alpha=0.6, s=30, label=m.capitalize())
    ax.set_xlabel("May's Complexity (C × S)", fontsize=12)
    ax.set_ylabel('Conservation Score', fontsize=12)
    ax.set_title(f'Conservation vs Complexity\nPearson r = {synthetic_results["correlations"]["conservation_vs_complexity_pearson"]:.3f}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'synthetic_correlations.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def print_separator(char='=', width=72):
    print(char * width)


def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()


def print_footer():
    print_separator()
    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print()
    print("=" * 72)
    print("  ECOSYSTEM CONSERVATION ANALYSIS")
    print("  Tension Graph Laplacian → Biomass Conservation in Food Webs")
    print("=" * 72)
    print()
    
    # ---- Create food webs ----
    print("Building food webs...")
    food_webs = []
    
    print("  1. Ythan Estuary (94 species)...")
    fw_ythan = ythan_estuary_foodweb()
    food_webs.append(fw_ythan)
    print(f"     → {fw_ythan.n_species} species, {len(fw_ythan.interactions)} feeding links")
    
    print("  2. Cascade Mountains (171 species)...")
    fw_cascades = cascades_foodweb()
    food_webs.append(fw_cascades)
    print(f"     → {fw_cascades.n_species} species, {len(fw_cascades.interactions)} feeding links")
    
    print("  3. Synthetic Cascade Model (S=30, C=0.15)...")
    fw_syn_cascade = cascade_model_foodweb(30, 0.15, seed=42)
    food_webs.append(fw_syn_cascade)
    print(f"     → {fw_syn_cascade.n_species} species, {len(fw_syn_cascade.interactions)} feeding links")
    
    print("  4. Synthetic Niche Model (S=30, C=0.15)...")
    fw_syn_niche = niche_model_foodweb(30, 0.15, seed=42)
    food_webs.append(fw_syn_niche)
    print(f"     → {fw_syn_niche.n_species} species, {len(fw_syn_niche.interactions)} feeding links")
    print()
    
    # ---- Run experiments ----
    print("Running experiments...")
    print()
    
    for fw in food_webs:
        run_keystone = fw.n_species <= 40
        run_invasion = True
        
        print_header(f"FOOD WEB: {fw.name} ({fw.n_species} species, {len(fw.interactions)} interactions)")
        
        # --- Stability vs Conservation ---
        print("  [3a] Stability vs Conservation Analysis")
        stab = experiment_stability_vs_conservation(fw)
        for scenario, data in stab.items():
            print(f"    {scenario:20s}: conservation={data['conservation']:.6f}, "
                  f"species={data['n_species']}, interactions={data['n_interactions']}, "
                  f"connectance={data['connectance']:.4f}")
        plot_conservation_comparison(stab, fw.name)
        
        # --- Keystone Species ---
        if run_keystone:
            print("  [3b] Keystone Species Analysis")
            try:
                keystone = experiment_keystone(fw, max_species=min(50, fw.n_species))
                print(f"    Original conservation: {keystone['original_conservation']:.6f}")
                print(f"    Top 5 keystone species (biggest conservation drop):")
                for idx, drop in keystone['top_keystone'][:5]:
                    print(f"      Species {idx}: drop = {drop:+.6f}")
                print(f"    Most redundant species (smallest impact):")
                for idx, drop in keystone['most_redundant']:
                    print(f"      Species {idx}: drop = {drop:+.6f}")
                plot_keystone_impact(keystone, fw.name)
            except Exception as e:
                print(f"    ERROR in keystone analysis: {e}")
        
        # --- Invasion ---
        if run_invasion:
            print("  [3c] Invasion Analysis")
            try:
                inv = experiment_invasion(fw)
                print(f"    Pre-invasion conservation:  {inv['original_conservation']:.6f}")
                print(f"    Post-invasion conservation: {inv['post_invasion']:.6f}")
                print(f"    Change: {inv['change']:+.6f}")
                print(f"    Invasive preys on {inv['n_prey']} species, "
                      f"preyed on by {inv['n_native_predators']} native species")
                plot_invasion_impact(inv, fw.name)
            except Exception as e:
                print(f"    ERROR in invasion analysis: {e}")
        
        # --- Extinction Cascade ---
        print("  [3d] Extinction Cascade Analysis")
        try:
            ext = experiment_extinction_cascade(fw, n_steps=20)
            print(f"    Initial conservation:    {ext['initial_conservation']:.6f}")
            print(f"    Final conservation:      {ext['final_conservation']:.6f}")
            if ext['collapse_step'] is not None:
                print(f"    COLLAPSE at step {ext['collapse_step']} "
                      f"(species removed: {fw.n_species - ext['n_species_trace'][ext['collapse_step']]})")
            else:
                print("    No collapse detected (conservation stayed above 50% threshold)")
            plot_extinction_cascade(ext, fw.name)
        except Exception as e:
            print(f"    ERROR in extinction cascade: {e}")
        
        # --- Trophic Level Analysis ---
        print("  [3e] Trophic Level Analysis")
        try:
            tl = experiment_trophic_level_analysis(fw)
            print(f"    Number of trophic levels: {tl['n_levels']}")
            for lv, data in sorted(tl['level_analysis'].items()):
                c_str = f"{data['conservation']:.6f}" if data['conservation'] is not None else "N/A"
                print(f"    {lv}: n={data['n_species']}, conservation={c_str}, "
                      f"mean_biomass={data['mean_biomass']:.2f}")
            plot_trophic_level_conservation(tl, fw.name)
        except Exception as e:
            print(f"    ERROR in trophic analysis: {e}")
        
        print()
    
    # ---- Synthetic analysis (100 random webs) ----
    print_header("BATCH SYNTHETIC ANALYSIS (100 webs)")
    syn_results = run_synthetic_analysis(n_webs=100)
    
    print(f"\n  Results across {len(syn_results['conservation'])} food webs:")
    print(f"    Conservation - Mean: {syn_results['conservation_mean']:.4f}, "
          f"Std: {syn_results['conservation_std']:.4f}")
    print(f"    Cascade model mean:  {syn_results['cascade_mean_conservation']:.4f}")
    print(f"    Niche model mean:    {syn_results['niche_mean_conservation']:.4f}")
    print()
    print("  Correlations with stability metrics:")
    for k, v in syn_results['correlations'].items():
        print(f"    {k:55s}: {v:.4f}")
    print()
    
    plot_synthetic_correlations(syn_results)
    
    print()
    print("=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)
    print()
    print("  Key findings summarized above. All plots saved to:")
    print(f"    {os.path.abspath(OUTPUT_DIR)}/")
    print()
