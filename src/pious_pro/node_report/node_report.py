from typing import List
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from ..mutate import NodeMutationData, cosine_similarity, compute_sim_matrix
from sklearn.cluster import DBSCAN
import copy


ACTION_FREQUENCY_THRESHOLD = 0.05


def color_cards(cards):
    return "".join([color_card(cards[i : i + 2]) for i in range(0, len(cards), 2)])


class ClusteringIterationData:
    def __init__():
        pass


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
        self, nmd: NodeMutationData, action: str, threshold: float, max_iters=3
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

        # NONE OF THESE SHOULD BE MUTATED
        cluster_profile = [hclusters, vclusters]
        delta_profile = [hero_deltas, villain_deltas]
        sim_matrix_profile = [hero_sim_matrix, villain_sim_matrix]
        hand_indices_profile = [hero_hand_indices, villain_hand_indices]
        range_profile = [nmd.hero_range, nmd.villain_range]

        iteration_number = 0
        changed = True
        clustering_history = {}
        epochs = []
        clustering_history["epochs"] = epochs
        clustering_history["clusters"] = cluster_profile
        clustering_history["deltas"] = delta_profile
        clustering_history["sim_matrices"] = sim_matrix_profile
        clustering_history["hand_indices"] = hand_indices_profile
        clustering_history["ranges"] = range_profile

        # The following gets modified at each iteration
        current_cluster_profile = [copy.deepcopy(hclusters), copy.deepcopy(vclusters)]
        while changed and iteration_number < max_iters:
            new_cluster_profile = []
            epoch = {}  # Data for clustering history
            epochs.append(epoch)
            iteration_number += 1
            changed = False

            # SET UP EPOCH DATA
            epoch["iteration"] = iteration_number
            player_epoch_data = [{}, {}]
            epoch["player_data"] = player_epoch_data
            epoch["starting_clusters"] = copy.deepcopy(current_cluster_profile)

            for player_number in range(2):
                # Convention: p1 stands for the player who we are currently
                # modifying. Thus p1 will be hero, then villain (or vice versa)
                p1_epoch_data = player_epoch_data[player_number]
                p1_changed = False

                # We are currently splitting clusters cs1 based on cs2
                p1_clusters = current_cluster_profile[player_number]
                p2_clusters = current_cluster_profile[1 - player_number]

                # Deltas: one entry per hand of the shape [hidx, DELTA]
                p1_deltas = delta_profile[player_number]
                p1_hand_indices = hand_indices_profile[player_number]

                p2_hand_indices = hand_indices_profile[1 - player_number]
                p2_range = range_profile[1 - player_number]

                # Compute cluster deltas and sim matrices
                hand_v_cluster_deltas = []
                per_cluster_sim_matrices = []
                per_cluster_distances = []

                for cluster_to_split in p1_clusters:
                    cluster_to_split_hand_indices = [
                        p1_hand_indices[i] for i in cluster_to_split
                    ]
                    # We want to compute the deltas of opponent's clusters per
                    # p1's hands
                    deltas = np.zeros(
                        (len(cluster_to_split), len(p2_clusters)), dtype=np.float64
                    )
                    for cidx, _ in enumerate(cluster_to_split_hand_indices):
                        for cjdx, c2 in enumerate(p2_clusters):
                            # Compute the average ev change experienced by c2
                            # from betting hidx. This is stored in p1's deltas
                            hand_deltas = np.array(p1_deltas[cidx])
                            c2_hand_indices = [p2_hand_indices[idx] for idx in c2]
                            c2_weights = np.array(
                                [p2_range[idx] for idx in c2_hand_indices]
                            )
                            c2_deltas = np.nan_to_num(
                                np.array([hand_deltas[idx] for idx in c2]),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            ev_shift = np.dot(c2_weights, c2_deltas)
                            deltas[cidx][cjdx] = ev_shift

                    clust_sm = compute_sim_matrix(deltas)
                    distances = 1 - clust_sm
                    hand_v_cluster_deltas.append(deltas)
                    per_cluster_sim_matrices.append(clust_sm)
                    per_cluster_distances.append(distances)

                p1_epoch_data["cluster_deltas"] = hand_v_cluster_deltas
                p1_epoch_data["sim_matrices"] = per_cluster_sim_matrices
                p1_epoch_data["distances"] = per_cluster_distances

                cs_idx = 0
                new_p1_clusters = []
                while cs_idx < len(p1_clusters):
                    distances = per_cluster_distances[cs_idx]
                    clust_sm = per_cluster_sim_matrices[cs_idx]
                    deltas = hand_v_cluster_deltas[cs_idx]

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

                    result = DBSCAN(min_samples=1, eps=threshold).fit_predict(distances)
                    n_clusters = max(result) + 1
                    if n_clusters > 1:
                        changed = True
                        p1_changed = True
                    new_subclusters = [[] for i in range(n_clusters)]
                    for idx, c in enumerate(result):
                        # each index indexes into deltas
                        new_subclusters[c].append(cluster_to_split[idx])
                    total_hands_in_new_subclusters = sum(
                        [len(sc) for sc in new_subclusters]
                    )
                    total_hands_in_old_cluster = len(cluster_to_split)

                    colored_board = " ".join([color_cards(c) for c in nmd.board])
                    new_p1_clusters += new_subclusters
                    cs_idx += 1
                print(
                    "Split",
                    len(p1_clusters),
                    "clusters into ",
                    len(new_p1_clusters),
                    "clusters",
                )
                new_cluster_profile.append(new_p1_clusters)
                print(
                    f"\033[1;33m ITERATION {iteration_number} | PLAYER {player_number}\033[0m"
                )
                print(
                    f"\n  \033[1;33m === {n_clusters} CLUSTERS ON {colored_board} \033[1;33mFOR AT \033[30;1m{nmd.node_id}\033[0m \033[1;33mTHRESHOLD {threshold: 5.3f} ===\033[0m \n"
                )
                print_clusters(p1_hand_indices, new_p1_clusters, width=10)
                # input(
                #     f"Finished Iteration {iteration_number} Player {player_number} cluster"
                # )
            # Finished per-player loop
            current_cluster_profile = new_cluster_profile
            new_cluster_profile = []
            epoch["ending_clusters"] = copy.deepcopy(current_cluster_profile)
        return clustering_history

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


def print_epoch_summary(clustering_history, epoch_number, player):
    epochs = clustering_history["epochs"]
    epoch = epochs[epoch_number]
    iteration = epoch["iteration"]
    cs0 = epoch["starting_clusters"][player]
    cs1 = epoch["ending_clusters"][player]
    cluster_mapping = [[] for _ in cs0]
    i = -1
    for c0, c0_mapping in zip(cs0, cluster_mapping):
        l0 = len(c0)
        l1 = 0
        while l1 < l0 and i + 1 < len(cs1):
            i += 1
            c1 = cs1[i]
            l1 += len(c1)
            c0_mapping.append(i)
        if len(c0_mapping) > 1:
            print("---------")
            hand_indices = clustering_history["hand_indices"][player]
            print_clusters(hand_indices, [c0])
            print("SPLIT INTO")
            print_clusters(hand_indices, [cs1[c1] for c1 in c0_mapping])

    pass
