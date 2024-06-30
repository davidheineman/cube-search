from src.model import CubeModel
from src.gpt import CubeGPT
from src.solver import CubeSolver
from src.puzzle_cube import PuzzleCube

import heapq

def astar(start_state, cm, num_chains=10):
    """ 
    GPT-Generated A-star. 
    
    Possible to implement A* search where the value is the entropy of next moves?
    """
    # Heuristic function (to be defined based on your problem)
    def heuristic(state):
        return 0  # Replace with an actual heuristic function

    # Priority queue for the frontier
    frontier = []
    heapq.heappush(frontier, (0, start_state))

    # Dictionaries to store the cost and path
    cost_so_far = {start_state: 0}
    came_from = {start_state: None}

    while frontier:
        print(frontier)
        # Get the state with the lowest cost
        current_priority, curr_cube = heapq.heappop(frontier)

        # Check if the current state is the goal
        if curr_cube.is_solved():
            # Reconstruct the path
            path = []
            while curr_cube is not None:
                path.append(curr_cube)
                curr_cube = came_from[curr_cube]
            path.reverse()
            return path

        # Get next possible moves and their costs
        next_moves = cm.next_move(curr_cube, num_chains)
        print(next_moves)
        for move, cost in next_moves.items():
            new_cost = cost_so_far[curr_cube] + cost
            next_cube = curr_cube.move(move)

            if next_cube not in cost_so_far or new_cost < cost_so_far[next_cube]:
                cost_so_far[next_cube] = new_cost
                priority = new_cost + heuristic(next_cube)
                heapq.heappush(frontier, (priority, next_cube))
                came_from[next_cube] = curr_cube

    return None


def main():
    # Create test Cube!
    pc = PuzzleCube()
    pc = pc.scramble(distance=8)
    pc = pc.move("R'") # Valid moves: L, L', R, R', U, U', D, D', F, F', B, B'
    # print(pc)
    # print(pc.is_solved())

    # Test ChatGPT solver
    pc = PuzzleCube()
    pc = pc.move("R'")

    cm = CubeGPT()
    next_move = cm.next_move(pc, num_chains=10)

    print(next_move)

    astar(pc, cm)

    # Create a solution plan with MCTS and a "solver" model
    # cm = CubeGPT()
    # s = CubeSolver(pc, cm)
    # s.solve(steps=1600)
    # print(s.solution())

    # Verify solution
    # for action in s.solution():
    #     pc = pc.move(action)
    # assert pc.is_solved()


if __name__ == '__main__': main()