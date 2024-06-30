from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.puzzle_cube import PuzzleCube
from src.cube_utils import BatchCube

import numpy as np

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:5174"}}) 
CORS(app)

executor = ThreadPoolExecutor(max_workers=5)

BASIC_MOVES = [
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

CUBE_PROMPT = """You are solving a Rubriks cube. Here is the current state:
{cube_state}
Select the next position from the following options: {options}. Make sure to think step by step about your response, then return the next position at the end (e.g., My next move is NEXT_MOVE)."""

def parse_response(response):
    """ Convert ['My next move is U.', 'My next move is U.', ...] -> [U, U, ...] """
    if not isinstance(response, list): response = [response]
    
    parsed = []
    for resp in response:
        words = resp.split()
        move = words[-1].strip('.')
        if move not in BASIC_MOVES:
            print(f'Could not parse "{resp}". Skipping...')
            continue
        parsed += [move]

    if len(parsed) == 0: return [None]

    return parsed


def process_output(prompt, completion):
    move = parse_response(completion)

    move = move[0]
    
    # Parse the problem into a cube
    cube_text = prompt.split('Here is the current state:')[1].split('Select the next position')[0]
    cube_text = parse_cube_text(cube_text)
    cube_text = cube_text[np.newaxis, ...]

    cube = PuzzleCube()
    cube._inner_cube._cube_array = cube_text
    if move is not None: cube = cube.move(move)

    # Check if that cube is completed
    completed = cube.is_solved()

    # Paste back into original prompt and return as reult
    formatted_moves = ', '.join(BASIC_MOVES)
    formatted_prompt = CUBE_PROMPT.format(cube_state=str(cube), options=formatted_moves)

    return formatted_prompt, completed


def parse_cube_text(cube_text):
    # pc = PuzzleCube()
    # pc = pc.move("R'")
    # print(pc._inner_cube._cube_array)

    color_map = {'r': 0, 'y': 1, 'g': 2, 'w': 3, 'o': 4, 'b': 5}

    cube_text_cleaned = cube_text.replace(" ", "").replace("\n", "")

    cube_numeric = []
    for char in cube_text_cleaned:
        if char in color_map:
            cube_numeric.append(color_map[char])

    mapping = [
        9, 10, 11, 21, 22, 23, 33, 34, 35, 
        0, 1, 2, 3, 4, 5, 6, 7, 8, 
        12, 13, 14, 24, 25, 26, 36, 37, 38,
        45, 46, 47, 48, 49, 50, 51, 52, 53,
        15, 16, 17, 27, 28, 29, 39, 40, 41, 
        18, 19, 20, 30, 31, 32, 42, 43, 44, 
    ]

    cube_numeric = np.array(cube_numeric)
    cube_numeric = cube_numeric[mapping]

    return cube_numeric


@app.route("/execute", methods=["POST"])
def execute():
    data = request.json

    prompt = data.get("problem", "")
    completion = data.get("completion", "")

    if not completion:
        response = jsonify({"error": "No completion provided"})
        response.status_code = HTTPStatus.BAD_REQUEST
        return response
    
    formatted_prompt, completed = process_output(prompt, completion)

    result = {"result": {"passed": (True if completed else False), "result": formatted_prompt}}

    try:
        return jsonify(result)
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        return response


# check if a 500 error code is thrown
@app.errorhandler(500)
def internal_error(error):
    return "500 error: {}".format(str(error)), 500

if __name__ == "__main__":

    TEST_PROMPT = """You are solving a Rubik's cube. Here is the current state:
         [y][y][b]
         [y][y][b]
         [y][y][b]
[r][r][r][g][g][y][o][o][o][w][b][b]
[r][r][r][g][g][y][o][o][o][w][b][b]
[r][r][r][g][g][y][o][o][o][w][b][b]
         [w][w][g]
         [w][w][g]
         [w][w][g]
Select the next position from the following options: L, L', R, R', U, U', D, D', F, F', B, B'. Think about your response, then return the next position at the end (e.g., My next move is NEXT_MOVE)."""

    TEST_COMPLETION = "My next move is R."

    out, completed = process_output(TEST_PROMPT, TEST_COMPLETION)
    print(out)

    app.run(host="localhost", port=5175)