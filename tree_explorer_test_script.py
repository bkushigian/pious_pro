from pious.util import PIO_HAND_ORDER
from pious.pio import make_solver, Node
from pious_pro.mutate import NodeMutator, TreeEVExplorer, TreeNode
from os import path as osp

cfr_file = r"C:\Users\bkush\OneDrive\Desktop\JdTc6c.cfr"

explorer = TreeEVExplorer(cfr_file, "r:0", depth=2)
explorer.run()
explorer.pickle(osp.join("pp", "mutate", "tree_explorer.pkl"))
