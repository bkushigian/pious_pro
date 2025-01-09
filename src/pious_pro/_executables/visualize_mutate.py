from ..cli import CliSubcommand
from ..mutate import NodeMutator
from argparse import _SubParsersAction
import pickle
import numpy as np
import matplotlib.pyplot as plt


class VisualizeMutateCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "visualize_mutate",
            "Visualize data from the mutate command",
        )
        p = self.parser
        p.add_argument("deltas", help="the stored deltas")
        p.add_argument(
            "sim_matrix",
            nargs="?",
            help='The node id to investigate (In PioViewer, Right-click the board and select "Copy node id")',
        )
        p.add_argument(
            "--save",
            action="store_true",
            help="Store visualizations to disk",
        )

    def run(self, args) -> int:
        with open(args.deltas, "rb") as f:
            deltas = pickle.load(f)
        with open(args.sim_matrix, "rb") as f:
            sim_matrix = pickle.load(f)
        plt.figure(figsize=(8, 8))
        plt.imshow(sim_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)

        # Add a colorbar
        cbar = plt.colorbar()
        cbar.set_label("Similarity", rotation=270, labelpad=15)

        # Add labels and title
        plt.title("Similarity Matrix Visualization")
        plt.xlabel("Index")
        plt.ylabel("Index")

        # Show the grid lines (optional)
        plt.grid(False)

        # Display the plot
        plt.show()
        if args.save:
            plt.savefig(f"{args.sim_matrix}.png", dpi=300)
