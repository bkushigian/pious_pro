from typing import List
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from .mutate import NodeMutationData, cosine_similarity, compute_sim_matrix
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

        self.combination_clustering_for_action(nmd, args.action, args.threshold)

    def summarize_node(self, nmd: NodeMutationData, action: str, threshold: float):
        return self.separation_clustering_for_action(nmd, action, threshold)

    def separation_clustering_for_action(
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

        hero_range = nmd.hero_range
        villain_range = nmd.villain_range

        hero_range_condensed = hero_range[np.nonzero(hero_range)[0]]
        villain_range_condensed = villain_range[np.nonzero(villain_range)[0]]

        # print("HERO DATA")
        # print(hero_hand_indices)
        # print(len(hero_hand_indices))
        # print(len(hero_deltas))
        # print(len(hero_deltas[0]))
        # print(len(hero_deltas[1]))
        # print(len(hero_sim_matrix))
        # print(hero_range_condensed)
        # print(len(hero_range_condensed))

        # print("VILLAIN DATA")
        # print(villain_hand_indices)
        # print(len(villain_hand_indices))
        # print(len(villain_deltas))
        # print(len(villain_deltas[1]))
        # print(len(villain_deltas[2]))
        # print(len(villain_sim_matrix))
        # print(villain_range_condensed)
        # print(len(villain_range_condensed))

        action_freqs = nmd.strategy[action_idx]
        # These are the pio hand order indices
        action_hand_indices = [
            idx for (idx, freq) in enumerate(action_freqs) if freq >= 0.05
        ]

        hclusters = [list(range(len(hero_hand_indices)))]
        vclusters = [list(range(len(villain_hand_indices)))]

        # We define profiles to make it easy to index into them based on the
        # current player (i.e., cluster = cluster_profile[pos])
        cluster_profile = [hclusters, vclusters]
        delta_profile = [hero_deltas, villain_deltas]
        sim_matrix_profile = [hero_sim_matrix, villain_sim_matrix]
        hand_indices_profile = [hero_hand_indices, villain_hand_indices]
        range_profile = [nmd.hero_range, nmd.villain_range]

        iteration_number = 0
        changed = True
        while changed:
            iteration_number += 1
            changed = False

            for player_number in range(2):
                this_player_changed = False
                print("ITERATION", iteration_number, " |  PLAYER", player_number)
                # We are currently splitting clusters cs1 based on cs2
                p1_clusters = cluster_profile[player_number]
                p2_clusters = cluster_profile[1 - player_number]

                # Deltas: one entry per hand of the shape [hidx, DELTA]
                p1_deltas = delta_profile[player_number]
                p1_sim_matrix = sim_matrix_profile[player_number]
                p1_hand_indices = hand_indices_profile[player_number]

                p2_deltas = delta_profile[1 - player_number]
                p2_hand_indices = hand_indices_profile[1 - player_number]
                p2_range = range_profile[1 - player_number]

                cs_idx = 0
                while cs_idx < len(p1_clusters):
                    cluster_to_split = p1_clusters[cs_idx]
                    # We want to define a new sim matrix. The normal sim matrix
                    # shows similarities of hands based on how they affect
                    # villain's range, hand by hand.
                    #
                    # Instead, we want to show the similarity of each hand in
                    # the this cluster `cs` based on how it affects the other
                    # player's clusters.
                    #
                    # To calculate we create a 
                    # 
                    #      |cluster_to_split| x # |p2_clusters|
                    # 
                    # matrix whose (i,j) entry is the mean change in EV of
                    # p2_clusters[j] when hand i is the only hand to bet.

                    # We calculate entry (i,j) as the weighted average of each
                    # hand in p2_cluster[j] and that hand's ev deltas:
                    # >>> cluster_hand_indices =[p2_hand_indices[idx] for idx in p2_cluster[j]]
                    # >>> cluster_weights = [p2_range[idx] for idx in cluster_hand_indices]
                    # >>> cluster_deltas = [p2_deltas[idx] for idx cluster_hand_indices]
                    # >>> delta = np.dot(cluster_weights, cluster_deltas)

                    cluster_to_split_hand_indices = [p1_hand_indices[i] for i in cluster_to_split]
                    deltas = np.zeros((len(cluster_to_split), len(p2_clusters)), dtype=np.float64)
                    for cidx, hidx in enumerate(cluster_to_split_hand_indices):
                        for cjdx, c2 in enumerate(p2_clusters):
                            # Compute the average ev change experienced by c2
                            # from betting hidx. This is stored in p1's deltas
                            hand_deltas = np.array(p1_deltas[cidx])
                            c2_hand_indices = [p2_hand_indices[idx] for idx in c2]
                            c2_weights = np.array([p2_range[idx] for idx in c2_hand_indices])
                            c2_deltas = np.nan_to_num(np.array([hand_deltas[idx] for idx in c2]), nan=0.0, posinf=0.0, neginf=0.0)
                            ev_shift = np.dot(c2_weights, c2_deltas)
                            deltas[cidx][cjdx] = ev_shift

                    clust_sm = compute_sim_matrix(deltas)
                    distances = 1 - clust_sm
                    result = DBSCAN(min_samples=1, eps=threshold).fit_predict(distances)
                    print(result)
                    n_clusters = max(result) + 1
                    if n_clusters > 1:
                        changed = True
                        this_player_changed = True
                    new_clusters = [[] for i in range(n_clusters)]

                    for idx, c in enumerate(result):
                        # each index indexes into deltas
                        new_clusters[c].append(cluster_to_split[idx])
                    colored_board = " ".join([color_cards(c) for c in nmd.board])
                    print(
                        f"\033[1;33m ITERATION {iteration_number} | PLAYER {player_number}\033[0m"
                    )
                    print(
                        f"\n  \033[1;33m === {n_clusters} CLUSTERS ON {colored_board} \033[1;33mFOR AT \033[30;1m{nmd.node_id}\033[0m \033[1;33mTHRESHOLD {threshold: 5.3f} ===\033[0m \n"
                    )
                    if this_player_changed:
                        print("\n\033[32;1m---- STARTING CLUSTER ----\033[0m")
                        print_clusters(p1_hand_indices, p1_clusters, width=10)
                        # First, remove original cluster
                        old_cluster_cell = p1_clusters.pop(cs_idx)
                        print("\n\033[32;1m---- REMOVING OLD CLUSTER CELL ----\033[0m")
                        print_clusters(p1_hand_indices, [old_cluster_cell])
                        print("\n\033[32;1m---- INSERTING NEW CLUSTER CELLS ----\033[0m")
                        print_clusters(p1_hand_indices, new_clusters)
                        for new_cluster_cell in new_clusters:
                            p1_clusters.insert(cs_idx, list(new_cluster_cell))
                            cs_idx += 1
                        cs_idx -= 1
                        print("\n\033[32;1m---------------------------------\033[0m")
                        print_clusters(p1_hand_indices, p1_clusters, width=10)
                        input("...")
                    cs_idx += 1
                print("\033[32;1m---- FULL CLUSTER ----\033[0m")
                print_clusters(p1_hand_indices, p1_clusters, width=10)
                input(f"Finished Iteration {iteration_number} Player {player_number} cluster")

    def combination_clustering_for_action(
        self, nmd: NodeMutationData, action: str, threshold: float
    ):
        """
        This is a basic clustering approach, implemented as a proof of concept.
        It works by combining similar hands.
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
