"""
A user-facing interface to manipulate a single puzzle cube.
"""
from typing import Optional

from .cube_utils import BatchCube

VALID_MOVES = [
    "L", 
    "L'", 
    "R", 
    "R'", 
    "U", 
    "U'", 
    "D", 
    "D'", 
    "F", 
    "F'", 
    "B", 
    "B'"
]


class PuzzleCube:
    """
    An instance of a PuzzleCube.  The interface treats each instance of this class as immutable.
    """
    def __init__(self, _inner: Optional[BatchCube] = None):
        """
        :return: A new solved puzzle cube.
        """

        if _inner is None:
            self._inner_cube = BatchCube()
        else:
            self._inner_cube = _inner

    def copy(self) -> "PuzzleCube":
        return PuzzleCube(_inner=self._inner_cube.copy())

    def scramble(self, distance: int) -> "PuzzleCube":
        """
        Scrambles a copy of the cube a set number of random moves.
        :param distance: Number of random moves to scramble
        :return: A copy of the cube scrambled.
        """
        assert(distance >= 0)

        inner = self._inner_cube.copy()
        inner.randomize(distance)
        return PuzzleCube(_inner=inner)

    def move(self, action: str) -> "PuzzleCube":
        """
        Perform action on a copy of the cube.
        :param action: One of "L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"
        :return: A copy of the cube with one action performed.
        """
        assert(action in VALID_MOVES)

        move_index = VALID_MOVES.index(action)

        inner = self._inner_cube.copy()
        inner.step(move_index)
        return PuzzleCube(_inner=inner)

    def is_solved(self) -> bool:
        """
        :return: Whether or not the cube is solved.
        """
        return self._inner_cube.done()[0]

    def __str__(self) -> str:
        """
        :return: A flat string representation of the cube.
        """
        return str(self._inner_cube)

    def __repr__(self) -> str:
        return str(self._inner_cube)
