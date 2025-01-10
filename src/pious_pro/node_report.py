from typing import List
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from .mutate import NodeMutationData


ACTION_FREQUENCY_THRESHOLD = 0.05


def color_cards(cards):
    return "".join(
        [color_card(cards[i : i + 2], plain_suit=True) for i in range(0, len(cards), 2)]
    )


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
                    # print(f"Combining clusters {i}{ci} and {j}{cj}")
                    # hands_in_ci = [PIO_HAND_ORDER[deltas[x][0]] for x in ci]
                    # hands_in_cj = [PIO_HAND_ORDER[deltas[x][0]] for x in cj]
                    # print(f"  {hands_in_ci}")
                    # print(f"  {hands_in_cj}")

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

    def summarize_node_for_action(
        self, nmd: NodeMutationData, action: str, threshold: float
    ):
        action = action
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

        _, deltas, sim_matrix = nmd.child_matchup_data[action_idx]
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
        delta_hand_indices = []
        for i, (hidx, _) in enumerate(deltas):
            if hidx in action_hand_indices:
                delta_hand_indices.append(i)

        N = len(sim_matrix)

        clusters = [[i] for i in range(N)]
        clusters = [[i] for i in delta_hand_indices]
        n_clusters = len(clusters)
        target_threshold = threshold
        threshold = 0.99
        colored_board = " ".join([color_card(c) for c in nmd.board])
        while threshold >= target_threshold:
            threshold = max(target_threshold, threshold)
            n_combinations = self.combine_clusters_for_threshold(
                deltas, sim_matrix, clusters, threshold
            )
            while n_combinations > 0:
                # Do it again
                n_combinations = self.combine_clusters_for_threshold(
                    deltas, sim_matrix, clusters, threshold
                )

            if len(clusters) < n_clusters:
                n_clusters = len(clusters)
                print(
                    f"\n   === \033[1;34m{n_clusters}\033[0m CLUSTERS ON {colored_board} FOR AT \033[1m{nmd.node_id}\033[0m THRESHOLD {threshold: 5.3f} === \n"
                )
                print_clusters(deltas, clusters, width=10)
            threshold -= 0.01


def print_clusters(deltas, clusters, width=10):
    print()
    for i, clust in enumerate(clusters):
        hand_ids = [deltas[x][0] for x in clust]
        hand_strs = [PIO_HAND_ORDER[x] for x in hand_ids]
        print(f"[\033[34mCLUSTER #{i}\033[0m]")
        for i in range(0, len(hand_strs), width):
            print(
                "    ",
                " ".join([color_cards(hands) for hands in hand_strs[i : i + width]]),
            )
