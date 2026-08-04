import numpy as np
import pandas as pd

def add_general_Elo(df:pd.DataFrame) -> pd.DataFrame:

    df_ = df.copy()

    BASE = 1500.0

    def k_factor(n):
        return 250.0 / (n + 5.0) ** 0.4

    def expected_score(ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    elo = {}       # player -> current rating
    n_played = {}  # player -> matches seen so far

    N = len(df_)
    out = {k: np.empty(N) for k in ["Elo_1", "Elo_2", "N_1", "N_2"]}

    for i, (a, b, y) in enumerate(zip(df_.Player_1, df_.Player_2, df_.Winner)):
        ra, rb = elo.get(a, BASE), elo.get(b, BASE)
        na, nb = n_played.get(a, 0), n_played.get(b, 0)

        # --- snapshot: pre-match state, result not seen yet ---
        out["Elo_1"][i], out["Elo_2"][i] = ra, rb
        out["N_1"][i],   out["N_2"][i]   = na, nb

        # --- update: result enters here ---
        d = y - expected_score(ra, rb)
        elo[a] = ra + k_factor(na) * d
        elo[b] = rb - k_factor(nb) * d

        n_played[a], n_played[b] = na + 1, nb + 1

    for k, v in out.items():
        df_[k] = v

    df_["Elo_diff"] = df_.Elo_1 - df_.Elo_2
    df_["Elo_mean"] = (df_.Elo_1 + df_.Elo_2) / 2
    df_["N_min"]    = np.minimum(df_.N_1, df_.N_2)

    return df_

class SurfaceEloCalculator:
    SURFACES = ('Hard', 'Clay', 'Grass', 'Carpet')

    def __init__(self, initial_rating=1500, base_k=32, shrinkage_param=20,
                 overall_weight=0.5, default_surface='Hard'):
        self.initial_rating = initial_rating
        self.base_k = base_k
        self.shrinkage_param = shrinkage_param
        self.overall_weight = overall_weight      # damping for the overall rating update
        self.default_surface = default_surface    # fallback for unknown/NaN surfaces

        self.ratings = {}       # player -> {'overall': r, surface: r, ...}
        self.match_counts = {}  # player -> {'overall': n, surface: n, ...}

        # Surface relationships used as prior for low-experience players
        self.surface_similarities = {
            'Hard': ['Carpet'],
            'Clay': [],
            'Grass': ['Carpet'],
            'Carpet': ['Grass', 'Hard'],
        }

    # ---------- internals ----------

    def _norm_surface(self, surface):
        if isinstance(surface, str):
            s = surface.strip().title()
            if s in self.SURFACES:
                return s
        return self.default_surface

    def _get_player_data(self, player):
        if player not in self.ratings:
            self.ratings[player] = {'overall': self.initial_rating}
            self.match_counts[player] = {'overall': 0}
            for s in self.SURFACES:
                self.ratings[player][s] = self.initial_rating
                self.match_counts[player][s] = 0
        return self.ratings[player], self.match_counts[player]

    def _compute_adaptive_k(self, match_count):
        return self.base_k * self.shrinkage_param / (match_count + self.shrinkage_param)

    def _get_effective_rating(self, player, surface):
        surface = self._norm_surface(surface)
        ratings, counts = self._get_player_data(player)

        n_surf = counts[surface]
        r_surf = ratings[surface]
        r_all = ratings['overall']

        full_trust = 2 * self.shrinkage_param
        if n_surf >= full_trust:
            return r_surf

        # Continuous weight: 0 at no experience, 1 at full_trust matches
        w_surf = n_surf / full_trust
        w_prior = 1.0 - w_surf

        sim_r, sim_w = [], []
        for sim_surf in self.surface_similarities.get(surface, []):
            n_sim = counts[sim_surf]
            if n_sim > 0:
                sim_r.append(ratings[sim_surf])
                sim_w.append(n_sim)

        if sim_r:
            avg_similar = sum(r * w for r, w in zip(sim_r, sim_w)) / sum(sim_w)
            prior = 0.7 * avg_similar + 0.3 * r_all
        else:
            prior = r_all

        return w_surf * r_surf + w_prior * prior

    # ---------- public API ----------

    def get_ratings_snapshot(self, player, surface):
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
        surface = self._norm_surface(surface)

        w_eff = self._get_effective_rating(winner, surface)
        l_eff = self._get_effective_rating(loser, surface)

        exp_w = 1.0 / (1.0 + 10 ** ((l_eff - w_eff) / 400.0))
        exp_l = 1.0 - exp_w

        w_ratings, w_counts = self._get_player_data(winner)
        l_ratings, l_counts = self._get_player_data(loser)

        # Surface-specific update
        w_ratings[surface] += self._compute_adaptive_k(w_counts[surface]) * (1 - exp_w)
        l_ratings[surface] += self._compute_adaptive_k(l_counts[surface]) * (0 - exp_l)

        # Overall update, damped
        w_ratings['overall'] += self._compute_adaptive_k(w_counts['overall']) * self.overall_weight * (1 - exp_w)
        l_ratings['overall'] += self._compute_adaptive_k(l_counts['overall']) * self.overall_weight * (0 - exp_l)

        w_counts[surface] += 1
        w_counts['overall'] += 1
        l_counts[surface] += 1
        l_counts['overall'] += 1

    def build_features(self, df, p1_col='Player_1', p2_col='Player_2', winner_col='Winner', surface_col='Surface'):
        p1_arr = df[p1_col].to_numpy()
        p2_arr = df[p2_col].to_numpy()
        y_arr = df[winner_col].to_numpy()
        surf_arr = df[surface_col].to_numpy()

        rows = []
        for p1, p2, y, surf in zip(p1_arr, p2_arr, y_arr, surf_arr):
            # Snapshots strictly before the match
            s1 = self.get_ratings_snapshot(p1, surf)
            s2 = self.get_ratings_snapshot(p2, surf)

            eff_diff = s1['elo_effective'] - s2['elo_effective']

            rows.append({
                # 'elo_surface_p1': s1['elo_surface'],
                # 'elo_surface_p2': s2['elo_surface'],
                # 'elo_overall_p1': s1['elo_overall'],
                # 'elo_overall_p2': s2['elo_overall'],
                # 'elo_effective_p1': s1['elo_effective'],
                # 'elo_effective_p2': s2['elo_effective'],
                # 'matches_surface_p1': s1['matches_surface'],
                # 'matches_surface_p2': s2['matches_surface'],
                # 'matches_overall_p1': s1['matches_overall'],
                # 'matches_overall_p2': s2['matches_overall'],
                'elo_surface_diff': s1['elo_surface'] - s2['elo_surface'],
                # 'elo_overall_diff': s1['elo_overall'] - s2['elo_overall'],
                'elo_effective_diff': eff_diff,
                'elo_prob': 1.0 / (1.0 + 10 ** (-eff_diff / 400.0)),
                # 'surface_exp_diff': s1['matches_surface'] - s2['matches_surface'],
                # 'overall_exp_diff': s1['matches_overall'] - s2['matches_overall'],
                # Surface specialization: how much better a player is on this surface
                'spec_diff': ((s1['elo_effective'] - s1['elo_overall']) -
                              (s2['elo_effective'] - s2['elo_overall'])) / 100.0,
                'surface_exp_min': min(s1['matches_surface'], s2['matches_surface']),
            })

            # Derive winner/loser from the binary target, then update
            if y == 1:
                self.update_ratings(p1, p2, surf)
            else:
                self.update_ratings(p2, p1, surf)

        return pd.DataFrame(rows, index=df.index)

def add_surface_Elo(df:pd.DataFrame) -> pd.DataFrame:

    df_ = df.copy()
    elo = SurfaceEloCalculator(base_k=32, shrinkage_param=20)
    surface_feats = elo.build_features(df_)
    df_ = pd.concat([df_, surface_feats], axis=1)

    return df_