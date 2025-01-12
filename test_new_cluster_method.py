from pious_pro.mutate import NodeMutator, NodeMutationData, TreeNode
from pious_pro.node_report import NodeReport
import numpy as np
from os import path as osp
import pickle


def main():
    path = osp.join("pp", "mutate", "Kc8d3c", "r_0.pkl")
    with open(path, "rb") as f:
        nmd = pickle.load(f)
    print(nmd)

    nr = NodeReport()
    return nr.separation_clustering_for_action(nmd, "b168", 3.2, max_iters=10)


if __name__ == "__main__":
    main()
