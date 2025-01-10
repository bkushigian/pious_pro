from typing import Dict, List, Optional, Tuple
from pious.pio import make_solver, Node, Solver
from pious.pio.aggregate import SpotData
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
from ansi.color import fg
import numpy as np
from os import path as osp
import os
import pickle
import tqdm
import json
import pprint


def color_cards(c):
    c.replace(" ", "")
    result = []
    for i in range(0, len(c), 2):
        result.append(color_card(c[i : i + 2], plain_suit=True))
    return "".join(result)


def cosine_similarity(xs, ys):
    return np.dot(xs, ys) / (np.linalg.norm(xs) * np.linalg.norm(ys))


class NodeMutationData:
    """
    This wraps data computed by NodeMutator for a spot
    """

    def __init__(
        self,
        cfr_file,
        node_id,
        board,
        actions,
        hero_evs,
        villain_evs,
        hero_range,
        villain_range,
        strategy,
    ):
        # METADATA
        self.cfr_file = cfr_file
        self.node_id = node_id

        # SPOT DATA
        self.board = board
        self.actions = actions
        self.hero_evs = hero_evs
        self.villain_evs = villain_evs
        self.strategy = strategy
        self.hero_range = hero_range
        self.villain_range = villain_range

        # MATCHUP DATA
        self.child_matchup_data = []

    def add_child_deltas_data(self, action, deltas, sim_matrix):
        self.child_matchup_data.append((action, deltas, sim_matrix))

    def pickle(self, out_dir=None):
        if out_dir is None:
            out_dir = osp.join("pp", "mutate")

        file_name = osp.basename(self.cfr_file)
        file_name_no_ext, _ = osp.splitext(file_name)
        node_id_no_colon = self.node_id.replace(":", "_")
        qualified_out_dir = osp.join(out_dir, file_name_no_ext)
        os.makedirs(qualified_out_dir, exist_ok=True)
        with open(osp.join(qualified_out_dir, f"{node_id_no_colon}.pkl"), "wb") as f:
            pickle.dump(self, f)


class TreeNode:
    """
    This contains a particular node and its node mutation data, as well
    as its children.

    """

    def __init__(self, node_id: str | Node, depth: int, nmd: NodeMutationData):
        """
        Initialize a TreeNode instance.

        :param node_id: The unique identifier for the node, which could be
                        either a string or another Node object.
        :param depth: The depth level of this node in the tree hierarchy.
        :param nmd: An instance of NodeMutationData that holds mutation-related
                    information for this node.
        """
        self.node_id = node_id
        self.depth = depth
        self.children: Dict[str:TreeNode] = {}
        self.nmd: NodeMutationData = nmd

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        self.children[key] = value

    def __delitem__(self, key):
        del self.children[key]

    def __contains__(self, item):
        return item in self.children

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)


class TreeEVExplorer:
    def __init__(
        self,
        cfr_file: str,
        root_node: str | Node,
        depth=1,
        action_freq_thresh: float = 0.1,
    ):
        self.cfr_file = cfr_file
        self.solver = make_solver()
        self.root_node = root_node
        self.depth = depth
        self.tree_root = None
        self.action_freq_thresh = action_freq_thresh

    def run(self):
        s = self.solver
        s.load_tree(self.cfr_file)
        s.load_all_nodes()
        s.rebuild_forgotten_streets()
        self.compute_tree()

    def compute_tree(self):
        """
        This method does a breadth first expansion of the tree.
        """
        solver = self.solver
        depth = 1
        print(fg.green("Exploring depth 1"))
        print(f"Computing NodeMutationData for node {self.root_node}")
        nmd = NodeMutator(
            cfr_file=self.cfr_file, solver=self.solver, node=self.root_node, save=False
        ).run()
        self.tree_root = TreeNode(self.root_node, depth=depth, nmd=nmd)
        # next_frontier contains a
        next_frontier = [self.tree_root]

        while depth < self.depth and len(next_frontier) > 0:
            print(fg.green(f"Exploring depth {depth}"))
            depth += 1
            current_frontier = list(next_frontier)
            next_frontier = []
            for tree_node in current_frontier:
                print(fg.green(f"  Expanding children of node {tree_node.node_id}"))
                actions = solver.show_children_actions(tree_node.node_id)
                action_freqs = compute_action_frequencies(
                    SpotData(solver, tree_node.node_id)
                )
                for a in actions:
                    print(fg.green(f"  Action {a}"))
                    freq = action_freqs[a]
                    tree_node[a] = None
                    if freq >= self.action_freq_thresh:
                        child_node_id = f"{tree_node.node_id}:{a}"
                        print(
                            fg.green(
                                f"    Computing NodeMutationData for node {child_node_id}"
                            )
                        )
                        nmd = NodeMutator(solver=solver, node=child_node_id).run()
                        tree_node[a] = TreeNode(child_node_id, depth=depth, nmd=nmd)
                        print(
                            fg.green(
                                f"    Adding new tree node for {a} at node {tree_node.node_id}"
                            )
                        )
                        next_frontier.append(tree_node[a])
                    else:
                        print(
                            fg.green(
                                f"    Action {a} at node {tree_node.node_id} has frequency {freq:5.3f} < {self.action_freq_thresh:5.3f}...skipping"
                            )
                        )

    def pickle(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.tree_root, f)


class NodeMutator:
    def __init__(
        self,
        cfr_file: str = None,
        solver: Solver = None,
        node: str | Node = "r:0",
        save=False,
    ):
        """
        Initialize a NodeMutator instance.

        Args:
            cfr_file (str): The path to the Counterfactual Regret Minimization
            (CFR) file.
            node (str | Node): The node that this mutator will operate on. It
            can be a string identifier or a Node object.

        Attributes:
            cfr_file (str): Stores the path to the CFR file.
            node (str | Node): Stores the node identifier or Node object.
            solver (None): Placeholder for a Solver instance, set during
            `run()`, defaults to None.
        """
        self.node = node
        self.save = save

        if solver is None and cfr_file is None:
            raise ValueError(
                "Cannot construct a NodeMutator without a valid solver or cfr_file instance"
            )

        if solver is None:
            solver = make_solver()
            solver.load_tree(cfr_file)
            solver.load_all_nodes()
            solver.rebuild_forgotten_streets()
        elif cfr_file is None:
            cfr_file = solver.cfr_file_path
        elif cfr_file != solver.cfr_file_path:
            raise ValueError(
                f"solver.cfr_file_path={solver.cfr_file_path} != cfr_file={cfr_file}"
            )
        self.cfr_file = cfr_file
        self.solver = solver

        self.assert_solver_is_ready()

    def assert_solver_is_ready(self):
        if not self.solver.is_ready():
            raise ValueError("Solver is not ready")

    def run(self) -> Optional[NodeMutationData]:
        self.assert_solver_is_ready()
        solver = self.solver
        node = self.solver.show_node(self.node)

        # Just in case, we reload and rebuild (this is cheap)
        solver.load_all_nodes()
        solver.rebuild_forgotten_streets()

        spot = SpotData(solver, node)
        actions = spot.available_actions()
        if len(actions) < 2:
            return None

        action_frequencies = compute_action_frequencies(spot)

        print(
            fg.green(
                f"Action frequencies for {node.node_id}: {pprint.pformat(action_frequencies)}"
            )
        )

        child_nodes = solver.show_children(node)
        pos_idx = spot.node.get_position_idx()
        hero_evs = spot.hand_evs(pos_idx)
        villain_evs = spot.hand_evs(1 - pos_idx)
        hero_range = spot.range(pos_idx).range_array
        villain_range = spot.range(1 - pos_idx).range_array
        strategy = spot.strategy()

        nmd = NodeMutationData(
            self.cfr_file,
            node.node_id,
            node.board,
            actions,
            hero_evs,
            villain_evs,
            hero_range,
            villain_range,
            strategy,
        )
        for child in child_nodes:
            deltas = self.compute_matchup_deltas(spot, child)
            sim_matrix = self.compute_sim_matrix(deltas=deltas)
            nmd.add_child_deltas_data(child.last_action, deltas, sim_matrix)

        if self.save:
            nmd.pickle()
        return nmd

    def compute_sim_matrix(self, deltas: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        N = len(deltas)
        sim_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            _, delta0 = deltas[i]
            cleaned0 = np.nan_to_num(delta0, nan=0.0, posinf=0.0, neginf=0.0)
            for j in range(i, N):
                _, delta1 = deltas[j]
                cleaned1 = np.nan_to_num(delta1, nan=0.0, posinf=0.0, neginf=0.0)
                sim_score = cosine_similarity(cleaned0, cleaned1)
                sim_matrix[i][j] = sim_score
                sim_matrix[j][i] = sim_score
        return sim_matrix

    def compute_matchup_deltas(self, spot: SpotData, child: Node):
        """
        Compute EV deltas per matchup.

        Currently this computes the EV in the child node of each of "villain's"
        hands against each of "hero's" hands when they take the given action.

        TODO: We also want to compute the reverse: we want to compute the ev of
        each of hero's hands when villain reacts in different ways with each of
        their hands.
        """

        child_spot = SpotData(self.solver, child)
        pos_idx = spot.node.get_position_idx()
        child_pos_idx = 1 - pos_idx  # 0 -> 1, 1 -> 0
        child_pos = ["OOP", "IP"][child_pos_idx]
        child_evs = child_spot.hand_evs(child_pos_idx)

        orig_strat = spot.strategy()
        orig_strat = [freq for action_freqs in orig_strat for freq in action_freqs]

        new_strat = list(orig_strat)
        rng = spot.range(spot.node.get_position_idx())

        children = self.solver.show_children(spot.node)
        print(children)
        children_node_ids = [n.node_id for n in children]
        action_idx = children_node_ids.index(child.node_id)
        print(action_idx)
        alt_action_idx = (action_idx + 1) % 2

        N_HANDS = 1326
        # Set action_idx to 0
        for hand_index, wt in enumerate(rng.range_array):
            if wt == 0:
                continue
            from_idx = action_idx * N_HANDS + hand_index
            to_idx = alt_action_idx * N_HANDS + hand_index
            new_strat[to_idx] += new_strat[from_idx]
            new_strat[from_idx] = 0

        # Now, for each hand with non-zero weight

        deltas = []  # List of per-hand deltas
        for hand_index, wt in tqdm.tqdm(
            enumerate(rng.range_array),
            desc="computing matchup evs",
            total=len(rng.range_array),
        ):
            if wt == 0:
                continue
            to_idx = action_idx * N_HANDS + hand_index
            # Make this hand only take our action of interest
            for a_idx in range(len(children)):
                if a_idx == action_idx:
                    continue
                from_idx = a_idx * N_HANDS + hand_index
                new_strat[to_idx] += new_strat[from_idx]
                new_strat[from_idx] = 0

            # Next, set strat and calc results
            self.solver.set_strategy(spot.node, new_strat)
            self.solver.calc_results()
            new_child_evs = self.solver.calc_ev(child_pos, child_spot.node)[0]
            deltas_for_hand = new_child_evs - child_evs
            deltas.append((hand_index, deltas_for_hand))

            # Finally, reset `new_strat` to not take this action with this hand
            new_strat[to_idx] = 0
            new_strat[from_idx] = 1

        print("Restoring original strategy...", end="", flush=True)
        self.solver.set_strategy(spot.node, orig_strat)  # Ensure we reset everything
        self.solver.calc_results()
        print("DONE")
        return deltas

    def write_metadata(self, outdir):
        with open(osp.join(outdir, "metadata.json"), "w") as f:
            json.dump({"cfr_file": self.cfr_file, "node_id": self.node}, f)

    def write_spotdata(self, outdir, spot: SpotData):
        pos_idx = spot.node.get_position_idx()
        payload = {
            "hero_evs": spot.hand_evs(pos_idx),
            "villain_evs": spot.hand_evs(1 - pos_idx),
            "strat": spot.strategy(),
            "hero_range": spot.range(pos_idx).range_array,
            "villain_range": spot.range(1 - pos_idx).range_array,
        }
        with open(osp.join(outdir, "spotdata.pkl"), "wb") as f:
            pickle.dump(payload, f)


def get_previous_node(solver: Solver, node: Node) -> Optional[Node]:
    if node.node_id == "r:0":
        return None
    pos_idx = node.get_position_idx()
    target_pos_idx = (pos_idx + 1) % 2
    actions = node.node_id.split(":")
    while True:
        actions = actions[:-1]
        node_id = ":".join(actions)
        pnode = solver.show_node(node_id)
        if pnode.get_position_idx() == target_pos_idx:
            return pnode
        elif node_id == "r:0":
            raise RuntimeError("Illegal State")


def compute_action_frequencies(spot: SpotData) -> Dict[str, np.float64]:
    pos_idx = spot.node.get_position_idx()
    actions = spot.solver.show_children_actions(spot.node)
    strat = spot.strategy()
    spot.hand_evs(pos_idx)  # To compute matchups
    h_matchups = spot.matchups(pos_idx)
    v_evs = spot.hand_evs((pos_idx + 1) % 2)
    total_matchups = sum(h_matchups)
    action_frequencies = {}
    for action, action_strat in zip(actions, strat):
        action_frequencies[action] = np.dot(action_strat, h_matchups) / total_matchups
    return action_frequencies
