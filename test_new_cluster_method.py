from pious_pro.mutate import NodeMutator, NodeMutationData, TreeNode
from pious_pro.node_report import NodeReport, print_epoch_summary
import numpy as np
from os import path as osp
import pickle


def main():
    clustering_history = run()
    epochs = clustering_history["epochs"]
    for epoch_number in range(len(epochs)):
        print(" --- EPOCH --- ")
        print_epoch_summary(clustering_history, epoch_number, 0)
    with open(osp.join("pp", "mutate", "epochs.pkl"), "wb") as f:
        pickle.dump(clustering_history, f)


def run():
    path = osp.join("pp", "mutate", "Kc8d3c", "r_0.pkl")
    with open(path, "rb") as f:
        nmd = pickle.load(f)

    nr = NodeReport()
    data = nr.separation_clustering_for_action(nmd, "b168", 3.2, max_iters=5)
    return data


if __name__ == "__main__":
    main()
