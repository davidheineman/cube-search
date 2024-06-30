"""
A user-facing interface for using a trained model to solve a cube.
"""
from .model import CubeModel
from .puzzle_cube import PuzzleCube, VALID_MOVES
from .mcts import MCTSAgent, State
from typing import List, Optional
import numpy as np


def cube_to_initial_mcts_state(cube: PuzzleCube, history: int) -> State:
    blank_history = tuple(None for _ in range(history-1))
    internal_state = (cube._inner_cube,) + blank_history
    return State(_internal_state=internal_state)


class CubeSolver:
    """
    A convinient wrapper around the MCTS solver used to solve the cube.
    """
    def __init__(self, cube: PuzzleCube, model: CubeModel):
        """
        :param cube: The starting puzzle cube.
        :param model: The trained model to use.
        """

        # assert (model._model is not None), "model must be loaded"
        # history_length = model._model.history
        history_length = 8
        blank_history = tuple(None for _ in range(history_length - 1))
        internal_state = (cube._inner_cube,) + blank_history
        initial_state = State(_internal_state=internal_state)

        self._mcts_agent = MCTSAgent(model._function(), initial_state, max_depth=100)

    def solve(self, steps: int, stop_early: bool = True) -> None:
        """
        Run the solver for a certain number of steps.
        :param steps: Number of steps to run the MCTS solver.
        :param stop_early: Whether to stop the search once a solution is found.
        (More steps may find shorter solutions.)  Default is True.
        """
        self._mcts_agent.search(steps, stop_early)

    def solution(self) -> Optional[List[str]]:
        if self._mcts_agent.initial_node.terminal:
            return []
        elif np.any(self._mcts_agent.initial_node.connected_to_terminal):
            node = self._mcts_agent.initial_node
            moves = []
            while not node.terminal:
                best_action = np.argmax(node.action_visit_counts() * node.connected_to_terminal)
                moves.append(VALID_MOVES[best_action])
                node = node.children[best_action]
            return moves
        else:
            return None
