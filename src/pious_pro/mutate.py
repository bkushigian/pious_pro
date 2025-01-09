from typing import Dict, List, Optional, Tuple
from pious.pio import make_solver, Node, Solver
from pious.pio.aggregate import SpotData
from pious.util import PIO_HAND_ORDER
from pious.util import color_card
import numpy as np
from os import path as osp
import os
import pickle
import tqdm
import json


def color_cards(c):
    c.replace(" ", "")
    result = []
    for i in range(0, len(c), 2):
        result.append(color_card(c[i : i + 2], plain_suit=True))
    return "".join(result)


def cosine_similarity(xs, ys):
    return np.dot(xs, ys) / (np.linalg.norm(xs) * np.linalg.norm(ys))


class NodeMutator:
    def __init__(self, cfr_file: str, node: str | Node):
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
        self.cfr_file = cfr_file
        self.node = node
        self.solver = None
        self.save = False

    def run(self):
        solver = make_solver()
        solver.load_tree(self.cfr_file)
        self.solver = solver
        print(self.cfr_file)
        print(self.node)
        node = solver.show_node(self.node)
        solver.load_all_nodes()
        solver.rebuild_forgotten_streets()
        actions = solver.show_children_actions(node)
        if len(actions) == 1:
            raise RuntimeError(
                f"Need more than one action at a node to perform mutate: {actions}"
            )

        spot = SpotData(solver, node)

        actions = spot.available_actions()
        action_frequencies = compute_action_frequencies(spot)
        print(action_frequencies)
        child_nodes = solver.show_children(node)
        if len(child_nodes) < 2:
            print("Skipping: need at least 2 actions")
        else:
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
                print("Child:", child)
                deltas = self.compute_matchup_deltas(spot, child)
                sim_matrix = self.compute_sim_matrix(deltas=deltas)
                nmd.add_child_matchup_data(child.last_action, deltas, sim_matrix)

            if self.save:
                nmd.pickle()

        self.solver = None

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
        print("Computing")
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
    h_evs = spot.hand_evs(pos_idx)
    h_matchups = spot.matchups(pos_idx)
    v_evs = spot.hand_evs((pos_idx + 1) % 2)
    total_matchups = sum(h_matchups)
    action_frequencies = {}
    for action, action_strat in zip(actions, strat):
        action_frequencies[action] = np.dot(action_strat, h_matchups) / total_matchups
    return action_frequencies


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

    def add_child_matchup_data(self, action, matchups, sim_matrix):
        self.child_matchup_data.append((action, matchups, sim_matrix))

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
