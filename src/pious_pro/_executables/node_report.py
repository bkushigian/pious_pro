from typing import List
from ..cli import CliSubcommand
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from ..mutate import NodeMutationData
from ..node_report.node_report import NodeReport


class NodeReportCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "node_report",
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

    def run(self, args) -> int:
        with open(args.node_mutation_data, "rb") as f:
            nmd: NodeMutationData = pickle.load(f)

        nr = NodeReport()
        nr.combination_clustering_for_action(nmd, args.action, args.threshold)
