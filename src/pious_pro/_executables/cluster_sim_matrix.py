from typing import List
from ..cli import CliSubcommand
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from ..mutate import NodeMutationData


def color_cards(cards):
    return "".join([color_card(cards[i : i + 2]) for i in range(0, len(cards), 2)])


class ClusterSimMatrixCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "cluster_sim_matrix",
            "Cluster hand similarity data from the mutate command",
        )
        p = self.parser
        p.add_argument("node_mutation_data", help="The pickled NodeMutationData")
        p.add_argument(
            "action",
            nargs="?",
            default=None,
            help="Action to inspect (defaults to first aggressive action)",
        )
        p.add_argument(
            "--threshold", default=0.7, type=float, help="threshold for similarity"
        )

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
        action = args.action
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

        N = len(sim_matrix)
        clusters = [[i] for i in range(N)]
        n_clusters = len(clusters)
        target_threshold = args.threshold
        threshold = 0.99
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
                    f"\n   === \033[1;34m{n_clusters}\033[0m CLUSTERS AFTER COMBINING FOR THRESHOLD {threshold: 5.3f} === \n"
                )
                print_clusters(deltas, clusters, width=10)
            threshold -= 0.01


def print_clusters(deltas, clusters, width=10):
    for i, clust in enumerate(clusters):
        hand_ids = [deltas[x][0] for x in clust]
        hand_strs = [PIO_HAND_ORDER[x] for x in hand_ids]
        print()
        print(f"[\033[34mCLUSTER #{i}\033[0m]")
        for i in range(0, len(hand_strs), width):
            print(
                "    ",
                " ".join([color_cards(hands) for hands in hand_strs[i : i + width]]),
            )
