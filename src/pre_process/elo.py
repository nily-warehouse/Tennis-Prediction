import numpy as np
import pandas as pd

def add_general_Elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add general Elo rating features to tennis match data.
    
    Computes traditional Elo ratings without considering surface type.
    Snapshots are taken BEFORE each match to prevent data leakage.
    
    Args:
        df: DataFrame with Player_1, Player_2, Winner columns, sorted chronologically
        
    Returns:
        DataFrame with added Elo columns: Elo_1, Elo_2, N_1, N_2, Elo_diff, Elo_mean, N_min
    """
    df_ = df.copy()
    
    BASE = 1500.0  # Initial rating for all players

    def k_factor(n):
        """Adaptive K-factor: decreases as player gains experience"""
        return 250.0 / (n + 5.0) ** 0.4

    def expected_score(ra, rb):
        """Expected win probability for player A against player B"""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    # State tracking dictionaries
    elo = {}        # player -> current rating
    n_played = {}   # player -> matches seen so far

    # Pre-allocate output arrays
    N = len(df_)
    out = {k: np.empty(N) for k in ["Elo_1", "Elo_2", "N_1", "N_2"]}

    # Process matches chronologically
    for i, (a, b, y) in enumerate(zip(df_.Player_1, df_.Player_2, df_.Winner)):
        ra, rb = elo.get(a, BASE), elo.get(b, BASE)
        na, nb = n_played.get(a, 0), n_played.get(b, 0)

        # Snapshot: pre-match state (result not seen yet)
        out["Elo_1"][i], out["Elo_2"][i] = ra, rb
        out["N_1"][i],   out["N_2"][i]   = na, nb

        # Update: apply match result
        d = y - expected_score(ra, rb)  # Performance delta
        elo[a] = ra + k_factor(na) * d
        elo[b] = rb - k_factor(nb) * d

        n_played[a], n_played[b] = na + 1, nb + 1

    # Add arrays to DataFrame
    for k, v in out.items():
        df_[k] = v

    # Derived features
    df_["Elo_diff"] = df_.Elo_1 - df_.Elo_2
    df_["Elo_mean"] = (df_.Elo_1 + df_.Elo_2) / 2
    df_["N_min"]    = np.minimum(df_.N_1, df_.N_2)

    return df_


class SurfaceEloCalculator:
    """
    Multi-surface Elo rating system with adaptive blending.
    
    Maintains separate ratings for each surface (Hard, Clay, Grass, Carpet)
    plus an overall rating. For players with low experience on a surface,
    blends surface rating with similar surfaces and overall rating.
    """
    
    SURFACES = ('Hard', 'Clay', 'Grass', 'Carpet')

    def __init__(self, initial_rating=1500, base_k=32, shrinkage_param=20,
                 overall_weight=0.5, default_surface='Hard'):
        """
        Args:
            initial_rating: Starting rating for all players/surfaces
            base_k: Base K-factor for rating updates
            shrinkage_param: Controls K decay and trust threshold
            overall_weight: Damping coefficient for overall rating updates
            default_surface: Fallback for unknown/NaN surface values
        """
        self.initial_rating = initial_rating
        self.base_k = base_k
        self.shrinkage_param = shrinkage_param
        self.overall_weight = overall_weight
        self.default_surface = default_surface

        self.ratings = {}        # player -> {'overall': r, surface: r, ...}
        self.match_counts = {}   # player -> {'overall': n, surface: n, ...}

        # Surface similarity graph for low-experience priors
        self.surface_similarities = {
            'Hard': ['Carpet'],
            'Clay': [],
            'Grass': ['Carpet'],
            'Carpet': ['Grass', 'Hard'],
        }

    # ========== Internal methods ==========

    def _norm_surface(self, surface):
        """Normalize surface string to canonical form or fallback"""
        if isinstance(surface, str):
            s = surface.strip().title()
            if s in self.SURFACES:
                return s
        return self.default_surface

    def _get_player_data(self, player):
        """Initialize player if new, return (ratings_dict, counts_dict)"""
        if player not in self.ratings:
            self.ratings[player] = {'overall': self.initial_rating}
            self.match_counts[player] = {'overall': 0}
            for s in self.SURFACES:
                self.ratings[player][s] = self.initial_rating
                self.match_counts[player][s] = 0
        return self.ratings[player], self.match_counts[player]

    def _compute_adaptive_k(self, match_count):
        """K-factor with hyperbolic decay as experience grows"""
        return self.base_k * self.shrinkage_param / (match_count + self.shrinkage_param)

    def _get_effective_rating(self, player, surface):
        """
        Compute blended rating for prediction.
        
        For players with little surface experience, blends:
        - Surface-specific rating
        - Similar surfaces (if available)
        - Overall rating
        
        Full trust in surface rating is reached at 2*shrinkage_param matches.
        """
        surface = self._norm_surface(surface)
        ratings, counts = self._get_player_data(player)

        n_surf = counts[surface]
        r_surf = ratings[surface]
        r_all = ratings['overall']

        full_trust = 2 * self.shrinkage_param
        if n_surf >= full_trust:
            return r_surf  # Fully trust surface rating

        # Linear weight ramp: 0 at zero experience, 1 at full_trust matches
        w_surf = n_surf / full_trust
        w_prior = 1.0 - w_surf

        # Compute prior from similar surfaces if available
        sim_r, sim_w = [], []
        for sim_surf in self.surface_similarities.get(surface, []):
            n_sim = counts[sim_surf]
            if n_sim > 0:
                sim_r.append(ratings[sim_surf])
                sim_w.append(n_sim)

        if sim_r:
            # Weighted average of similar surfaces
            avg_similar = sum(r * w for r, w in zip(sim_r, sim_w)) / sum(sim_w)
            prior = 0.7 * avg_similar + 0.3 * r_all  # 70% similar surfaces, 30% overall
        else:
            prior = r_all  # No similar surface data, use overall rating

        return w_surf * r_surf + w_prior * prior

    # ========== Public API ==========

    def get_ratings_snapshot(self, player, surface):
        """
        Get current state of a player on a surface (before match).
        
        Returns:
            dict with keys: elo_surface, elo_overall, elo_effective,
                           matches_surface, matches_overall
        """
        surface = self._norm_surface(surface)
        ratings, counts = self._get_player_data(player)
        return {
            'elo_surface': ratings[surface],
            'elo_overall': ratings['overall'],
            'elo_effective': self._get_effective_rating(player, surface),
            'matches_surface': counts[surface],
            'matches_overall': counts['overall'],
        }

    def update_ratings(self, winner, loser, surface):
        """
        Apply match result: update both surface-specific and overall ratings.
        
        Prediction uses effective rating, but updates are applied separately:
        - Surface rating: full K
        - Overall rating: K * overall_weight (damped)
        """
        surface = self._norm_surface(surface)

        # Compute expected scores using effective ratings
        w_eff = self._get_effective_rating(winner, surface)
        l_eff = self._get_effective_rating(loser, surface)

        exp_w = 1.0 / (1.0 + 10 ** ((l_eff - w_eff) / 400.0))
        exp_l = 1.0 - exp_w

        w_ratings, w_counts = self._get_player_data(winner)
        l_ratings, l_counts = self._get_player_data(loser)

        # Surface-specific update
        w_ratings[surface] += self._compute_adaptive_k(w_counts[surface]) * (1 - exp_w)
        l_ratings[surface] += self._compute_adaptive_k(l_counts[surface]) * (0 - exp_l)

        # Overall update (damped to avoid one surface dominating)
        w_ratings['overall'] += self._compute_adaptive_k(w_counts['overall']) * self.overall_weight * (1 - exp_w)
        l_ratings['overall'] += self._compute_adaptive_k(l_counts['overall']) * self.overall_weight * (0 - exp_l)

        # Increment match counts
        w_counts[surface] += 1
        w_counts['overall'] += 1
        l_counts[surface] += 1
        l_counts['overall'] += 1

    def build_features(self, df, p1_col='Player_1', p2_col='Player_2',
                      winner_col='Winner', surface_col='Surface'):
        """
        Walk through DataFrame chronologically, snapshot before each match, then update.
        
        Returns:
            DataFrame with surface Elo features, same index as input
        """
        p1_arr = df[p1_col].to_numpy()
        p2_arr = df[p2_col].to_numpy()
        y_arr = df[winner_col].to_numpy()
        surf_arr = df[surface_col].to_numpy()

        rows = []
        for p1, p2, y, surf in zip(p1_arr, p2_arr, y_arr, surf_arr):
            # Snapshot strictly before the match
            s1 = self.get_ratings_snapshot(p1, surf)
            s2 = self.get_ratings_snapshot(p2, surf)

            eff_diff = s1['elo_effective'] - s2['elo_effective']

            rows.append({
                # Raw surface rating difference
                'elo_surface_diff': s1['elo_surface'] - s2['elo_surface'],
                
                # Effective (blended) rating difference
                'elo_effective_diff': eff_diff,
                
                # Win probability derived from effective rating
                'elo_prob': 1.0 / (1.0 + 10 ** (-eff_diff / 400.0)),
                
                # Surface specialization: how much better each player is on this surface vs overall
                # (scaled by 100 to keep magnitudes reasonable)
                'spec_diff': ((s1['elo_effective'] - s1['elo_overall']) -
                              (s2['elo_effective'] - s2['elo_overall'])) / 100.0,
                
                # Minimum surface experience between the two players
                'surface_exp_min': min(s1['matches_surface'], s2['matches_surface']),
            })

            # Update ratings after recording snapshot
            if y == 1:
                self.update_ratings(p1, p2, surf)
            else:
                self.update_ratings(p2, p1, surf)

        return pd.DataFrame(rows, index=df.index)


def add_surface_Elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add surface-aware Elo features to tennis match data.
    
    Wraps SurfaceEloCalculator to produce features that account for
    player specialization on different court surfaces.
    
    Args:
        df: DataFrame with Player_1, Player_2, Winner, Surface columns,
            sorted chronologically
            
    Returns:
        DataFrame with original columns plus surface Elo features
    """
    df_ = df.copy()
    elo = SurfaceEloCalculator(base_k=32, shrinkage_param=20)
    surface_feats = elo.build_features(df_)
    df_ = pd.concat([df_, surface_feats], axis=1)
    return df_