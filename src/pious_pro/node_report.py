from typing import List
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from .mutate import NodeMutationData
from sklearn.cluster import DBSCAN


ACTION_FREQUENCY_THRESHOLD = 0.05


def color_cards(cards):
    return "".join([color_card(cards[i : i + 2]) for i in range(0, len(cards), 2)])


class NodeReport:
    def __init__(self):
        pass

    def can_combine_clusters(self, sim_matrix, ci, cj, threshold=0.9):
        """
        Can we combine clusters i and j?
        """
        if ci == cj:
            return False
        sim_sum = 0
        print('SIM_MATRIX SHAPE', sim_matrix.shape)
        print("Len ci", len(ci))
        print("Len cj", len(cj))
        for h1 in ci:
            for h2 in cj:
                sim_sum += sim_matrix[h1][h2]
        return sim_sum / (len(ci) * len(cj)) >= threshold

    def combine_clusters_for_threshold(
        self, deltas, sim_matrix, clusters: List, threshold=0.1
    ):
        n_combinations = 0
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                ci = clusters[i]
                cj = clusters[j]
                # print(f"Comparing {ci}@{i} and {cj}@{j}")
                if self.can_combine_clusters(sim_matrix, ci, cj, threshold=threshold):
                    n_combinations += 1
                    ci += cj
                    clusters.pop(j)
                else:
                    j += 1
            i += 1
        return n_combinations

    def run(self, args) -> int:
        with open(args.node_mutation_data, "rb") as f:
            nmd: NodeMutationData = pickle.load(f)

        self.summarize_node_for_action(nmd, args.action, args.threshold)

    def separate_clusters_for_action(
        self, nmd: NodeMutationData, action: str, threshold: float
    ):
        """
        Start with one cluster per player and separate them out until FP is
        reached.
        """
        if action is None:
            for a in nmd.actions:
                if a.startswith("b"):
                    action = a
                    break
            if action is None:
                if "c" in nmd.actions:
                    action = "c"
                elif "f" in nmd.actions:
                    action = "f"
                else:
                    raise RuntimeError("Illegal action set")
        action_idx = nmd.actions.index(action)

        _, hero_data, villain_data = nmd.child_matchup_data[action_idx]
        hero_hand_indices, hero_deltas, hero_sim_matrix = hero_data
        villain_hand_indices, villain_deltas, villain_sim_matrix = villain_data

        action_freqs = nmd.strategy[action_idx]
        # These are the pio hand order indices
        action_hand_indices = [
            idx for (idx, freq) in enumerate(action_freqs) if freq >= 0.05
        ]
        hclusters = [[i for i in range(len(hero_sim_matrix))]]
        vclusters = [[i for i in range(len(villain_sim_matrix))]]

        cluster_profile = [hclusters, vclusters]
        delta_profile = [hero_deltas, villain_deltas]
        sim_matrix_profile = [hero_sim_matrix, villain_sim_matrix]
        print("\n\n==================================\n\n")
        print(hero_sim_matrix)

        print("\n\n==================================\n\n")
        print(villain_sim_matrix)
        exit()
        iteration_number = 0

        def print_clusters_for_player(clusters, deltas):
            for cn, nc in enumerate(clusters):
                print("Cluster ", cn)
                hands = [color_cards(PIO_HAND_ORDER[deltas[idx][0]]) for idx in nc]
                print(" ".join(hands))

        iteration_number = 0
        changed = True
        while changed:
            iteration_number += 1
            print("ITERATION", iteration_number)
            changed = False

            for player_number in range(2):
                print("  ITERATION", iteration_number, " |  PLAYER", player_number)
                # We are currently splitting clusters cs1 based on cs2
                this_player_clusters = cluster_profile[player_number]
                # Deltas: one entry per hand of the shape [hidx, DELTA]
                ds = delta_profile[player_number]
                sm = sim_matrix_profile[player_number]
                cs_idx = 0
                while cs_idx < len(this_player_clusters):
                    cs = this_player_clusters[cs_idx]
                    clust_sm = sm[np.ix_(cs, cs)]
                    # print("similarities", clust_sm)
                    distances = 1 - clust_sm
                    print(sm)
                    # print("distances", distances)
                    result = DBSCAN(min_samples=1, eps=threshold).fit_predict(distances)
                    n_clusters = max(result) + 1
                    if n_clusters > 1:
                        changed = True
                    new_clusters = [[] for i in range(n_clusters)]

                    for idx, c in enumerate(result):
                        # each index indexes into deltas
                        new_clusters[c].append(idx)
                    colored_board = " ".join([color_cards(c) for c in nmd.board])
                    print(
                        f"\033[1;33m ITERATION {iteration_number} | PLAYER {player_number}\033[0m"
                    )
                    print(
                        f"\n  \033[1;33m === {n_clusters} CLUSTERS ON {colored_board} \033[1;33mFOR AT \033[30;1m{nmd.node_id}\033[0m \033[1;33mTHRESHOLD {threshold: 5.3f} ===\033[0m \n"
                    )
                    print_clusters(ds, new_clusters)
                    if changed:
                        # First, remove original cluster
                        this_player_clusters.pop(cs_idx)
                        for nc in new_clusters:
                            this_player_clusters.insert(cs_idx, nc)
                            cs_idx += 1
                    cs_idx += 1
                    exit(1)

    def summarize_node_for_action(
        self, nmd: NodeMutationData, action: str, threshold: float
    ):
        if action is None:
            for a in nmd.actions:
                if a.startswith("b"):
                    action = a
                    break
            if action is None:
                if "c" in nmd.actions:
                    action = "c"
                elif "f" in nmd.actions:
                    action = "f"
                else:
                    raise RuntimeError("Illegal action set")

        action_idx = nmd.actions.index(action)

        _, hero_data, villain_data = nmd.child_matchup_data[action_idx]
        hero_hand_indices = hero_data[0]
        hero_deltas = hero_data[1]
        hero_sims_matrix = hero_data[2]
        action_freqs = nmd.strategy[action_idx]
        # These are the pio hand order indices
        action_hand_indices = [
            idx for (idx, freq) in enumerate(action_freqs) if freq >= 0.05
        ]
        pure_action_hand_indices = [
            idx
            for (idx, freq) in enumerate(action_freqs)
            if freq >= 1 - ACTION_FREQUENCY_THRESHOLD
        ]
        pure_no_action_hand_indices = [
            idx
            for (idx, freq) in enumerate(action_freqs)
            if freq <= ACTION_FREQUENCY_THRESHOLD
        ]
        mix_action_hand_indices = [
            idx
            for (idx, freq) in enumerate(action_freqs)
            if ACTION_FREQUENCY_THRESHOLD < freq < (1 - ACTION_FREQUENCY_THRESHOLD)
        ]

        # Now we want to translate them to our deltas indices

        N = len(hero_sims_matrix)

        clusters = [[i] for i in range(N)]
        # clusters = [[i] for i in hero_hand_indices]
        n_clusters = len(clusters)
        target_threshold = threshold
        threshold = 0.99
        colored_board = " ".join([color_card(c) for c in nmd.board])
        while threshold >= target_threshold:
            threshold = max(target_threshold, threshold)
            n_combinations = self.combine_clusters_for_threshold(
                hero_deltas, hero_sims_matrix, clusters, threshold
            )
            while n_combinations > 0:
                # Do it again
                n_combinations = self.combine_clusters_for_threshold(
                    hero_deltas, hero_sims_matrix, clusters, threshold
                )

            if len(clusters) < n_clusters:
                n_clusters = len(clusters)
                print(
                    f"\n  \033[1;34m === {n_clusters} CLUSTERS ON {colored_board} FOR AT \033[1m{nmd.node_id}\033[0m THRESHOLD {threshold: 5.3f} ===\033[0m \n"
                )
                print_clusters(hero_hand_indices, clusters, width=10)
            threshold -= 0.01


def print_clusters(hand_indices, clusters, width=10):
    print()
    for i, clust in enumerate(clusters):
        hand_ids = [hand_indices[x] for x in clust]
        hand_strs = [PIO_HAND_ORDER[x] for x in hand_ids]
        print(f"[\033[1;33mCLUSTER #{i}\033[0m]")
        for i in range(0, len(hand_strs), width):
            print(
                "    ",
                " ".join([color_cards(hands) for hands in hand_strs[i : i + width]]),
            )
