"""
An implementation of an array Rubic's cubes.

Actions are stored in this order (corresponding to these numbers)
    0:L, 1:L', 2:R, 3:R', 4:U, 5:U', 6:D, 7:D', 8:F, 9:F', 10:B, 11:B'

Internally the cube is stored as an integer array of dimension ? x 54
    One row for each cube, stored in a way easy to import into PyCuber (for printing)
    The numbers 0-5 correspond to the colors RYGWOB
    The nth block of 9 numbers corresponds to the nth face (in the order RYGWOB)
    E.g. the solved cube is stored as 
        000000000011111111111222222222233333333334444444444555555555

When it is outputed to a bit array (for the NN) it is stored as ? x 54 x 6
"""

import numpy as np

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

eye6 = np.eye(6, dtype=bool)
# store cubes via 
#    (-1, 6 faces, 9 squares) arrays
#    (-1, 6 colors, 6 faces, 9 squares) bit array

solved_cube_list = np.array([0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9)
solved_cube_bit_array = eye6[np.newaxis, solved_cube_list]

pc_indices = [58, 61, 64, 95, 98, 101, 132, 135, 138, 10, 13, 16, 29, 32, 35, 48, 51, 54, 67, 70, 73, 104, 107, 110, 141, 144, 147, 178, 181, 184, 197, 200, 203, 216, 219, 222, 76, 79, 82, 113, 116, 119, 150, 153, 156, 85, 88, 91, 122, 125, 128, 159, 162, 165]
str_order = [pc_indices.index(i) for i in sorted(pc_indices)]

STR_FORMAT = \
"""         [{}][{}][{}]
         [{}][{}][{}]
         [{}][{}][{}]
[{}][{}][{}][{}][{}][{}][{}][{}][{}][{}][{}][{}]
[{}][{}][{}][{}][{}][{}][{}][{}][{}][{}][{}][{}]
[{}][{}][{}][{}][{}][{}][{}][{}][{}][{}][{}][{}]
         [{}][{}][{}]
         [{}][{}][{}]
         [{}][{}][{}]"""

colors = ['r', 'y', 'g', 'w', 'o', 'b']
color_dict = {'r': 0, 'y': 1, 'g': 2, 'w': 3, 'o': 4, 'b': 5}
color_dict2 = {0: 'r', 1: 'y', 2: 'g', 3: 'w', 4: 'o', 5: 'b'}

# Here forward_action_array[a, n] is the position where square n moves to under action number a
forward_action_array =\
    np.array([[ 2,  5,  8,  1,  4,  7,  0,  3,  6, 18, 10, 11, 21, 13, 14, 24, 16,
        17, 27, 19, 20, 30, 22, 23, 33, 25, 26, 53, 28, 29, 50, 31, 32, 47,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 15, 48, 49, 12,
        51, 52,  9],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 51, 12, 13, 48, 15, 16,
        45, 18, 19, 11, 21, 22, 14, 24, 25, 17, 27, 28, 20, 30, 31, 23, 33,
        34, 26, 38, 41, 44, 37, 40, 43, 36, 39, 42, 35, 46, 47, 32, 49, 50,
        29, 52, 53],
       [45, 46, 47,  3,  4,  5,  6,  7,  8, 11, 14, 17, 10, 13, 16,  9, 12,
        15,  0,  1,  2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 18, 19, 20, 39, 40, 41, 42, 43, 44, 36, 37, 38, 48, 49, 50,
        51, 52, 53],
       [ 0,  1,  2,  3,  4,  5, 24, 25, 26,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 29, 32, 35, 28, 31, 34, 27,
        30, 33, 36, 37, 38, 39, 40, 41, 51, 52, 53, 45, 46, 47, 48, 49, 50,
         6,  7,  8],
       [ 0,  1, 17,  3,  4, 16,  6,  7, 15,  9, 10, 11, 12, 13, 14, 36, 39,
        42, 20, 23, 26, 19, 22, 25, 18, 21, 24,  2,  5,  8, 30, 31, 32, 33,
        34, 35, 29, 37, 38, 28, 40, 41, 27, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53],
       [33,  1,  2, 34,  4,  5, 35,  7,  8,  6,  3,  0, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 44,
        41, 38, 36, 37,  9, 39, 40, 10, 42, 43, 11, 47, 50, 53, 46, 49, 52,
        45, 48, 51],
       [ 6,  3,  0,  7,  4,  1,  8,  5,  2, 53, 10, 11, 50, 13, 14, 47, 16,
        17,  9, 19, 20, 12, 22, 23, 15, 25, 26, 18, 28, 29, 21, 31, 32, 24,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 33, 48, 49, 30,
        51, 52, 27],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 20, 12, 13, 23, 15, 16,
        26, 18, 19, 29, 21, 22, 32, 24, 25, 35, 27, 28, 51, 30, 31, 48, 33,
        34, 45, 42, 39, 36, 43, 40, 37, 44, 41, 38, 17, 46, 47, 14, 49, 50,
        11, 52, 53],
       [18, 19, 20,  3,  4,  5,  6,  7,  8, 15, 12,  9, 16, 13, 10, 17, 14,
        11, 36, 37, 38, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 45, 46, 47, 39, 40, 41, 42, 43, 44,  0,  1,  2, 48, 49, 50,
        51, 52, 53],
       [ 0,  1,  2,  3,  4,  5, 51, 52, 53,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23,  6,  7,  8, 33, 30, 27, 34, 31, 28, 35,
        32, 29, 36, 37, 38, 39, 40, 41, 24, 25, 26, 45, 46, 47, 48, 49, 50,
        42, 43, 44],
       [ 0,  1, 27,  3,  4, 28,  6,  7, 29,  9, 10, 11, 12, 13, 14,  8,  5,
         2, 24, 21, 18, 25, 22, 19, 26, 23, 20, 42, 39, 36, 30, 31, 32, 33,
        34, 35, 15, 37, 38, 16, 40, 41, 17, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53],
       [11,  1,  2, 10,  4,  5,  9,  7,  8, 38, 41, 44, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  0,
         3,  6, 36, 37, 35, 39, 40, 34, 42, 43, 33, 51, 48, 45, 52, 49, 46,
        53, 50, 47]])

# Unfortionately the forward_action_array is in the wrong order.  
# It is in the order L' R' U' D' F' B' L R U D F B
# But we want it in the order L L' R' U' D' F' B' L R U D F B
# Here we fix it.

action_array = forward_action_array[[6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5]]

color_permutations = \
    np.array([[0, 1, 2, 3, 4, 5],
              [0, 1, 5, 3, 4, 2],
              [0, 2, 3, 5, 4, 1],
              [0, 2, 1, 5, 4, 3],
              [0, 3, 5, 1, 4, 2],
              [0, 3, 2, 1, 4, 5],
              [0, 5, 1, 2, 4, 3],
              [0, 5, 3, 2, 4, 1],
              [1, 0, 5, 4, 3, 2],
              [1, 0, 2, 4, 3, 5],
              [1, 2, 0, 5, 3, 4],
              [1, 2, 4, 5, 3, 0],
              [1, 4, 2, 0, 3, 5],
              [1, 4, 5, 0, 3, 2],
              [1, 5, 4, 2, 3, 0],
              [1, 5, 0, 2, 3, 4],
              [2, 0, 1, 4, 5, 3],
              [2, 0, 3, 4, 5, 1],
              [2, 1, 4, 3, 5, 0],
              [2, 1, 0, 3, 5, 4],
              [2, 3, 0, 1, 5, 4],
              [2, 3, 4, 1, 5, 0],
              [2, 4, 3, 0, 5, 1],
              [2, 4, 1, 0, 5, 3],
              [3, 0, 2, 4, 1, 5],
              [3, 0, 5, 4, 1, 2],
              [3, 2, 4, 5, 1, 0],
              [3, 2, 0, 5, 1, 4],
              [3, 4, 5, 0, 1, 2],
              [3, 4, 2, 0, 1, 5],
              [3, 5, 0, 2, 1, 4],
              [3, 5, 4, 2, 1, 0],
              [4, 1, 5, 3, 0, 2],
              [4, 1, 2, 3, 0, 5],
              [4, 2, 1, 5, 0, 3],
              [4, 2, 3, 5, 0, 1],
              [4, 3, 2, 1, 0, 5],
              [4, 3, 5, 1, 0, 2],
              [4, 5, 3, 2, 0, 1],
              [4, 5, 1, 2, 0, 3],
              [5, 0, 3, 4, 2, 1],
              [5, 0, 1, 4, 2, 3],
              [5, 1, 0, 3, 2, 4],
              [5, 1, 4, 3, 2, 0],
              [5, 3, 4, 1, 2, 0],
              [5, 3, 0, 1, 2, 4],
              [5, 4, 1, 0, 2, 3],
              [5, 4, 3, 0, 2, 1]])

position_permutations = \
    np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
               34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
               51, 52, 53],
              [ 2,  1,  0,  5,  4,  3,  8,  7,  6, 15, 16, 17, 12, 13, 14,  9, 10,
               11, 47, 46, 45, 50, 49, 48, 53, 52, 51, 33, 34, 35, 30, 31, 32, 27,
               28, 29, 38, 37, 36, 41, 40, 39, 44, 43, 42, 20, 19, 18, 23, 22, 21,
               26, 25, 24],
              [ 2,  5,  8,  1,  4,  7,  0,  3,  6, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 53, 52, 51, 50, 49, 48, 47,
               46, 45, 42, 39, 36, 43, 40, 37, 44, 41, 38, 17, 16, 15, 14, 13, 12,
               11, 10,  9],
              [ 8,  5,  2,  7,  4,  1,  6,  3,  0, 24, 25, 26, 21, 22, 23, 18, 19,
               20, 15, 16, 17, 12, 13, 14,  9, 10, 11, 47, 46, 45, 50, 49, 48, 53,
               52, 51, 36, 39, 42, 37, 40, 43, 38, 41, 44, 29, 28, 27, 32, 31, 30,
               35, 34, 33],
              [ 8,  7,  6,  5,  4,  3,  2,  1,  0, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 53, 52, 51, 50, 49, 48, 47, 46, 45,  9, 10, 11, 12, 13, 14, 15,
               16, 17, 44, 43, 42, 41, 40, 39, 38, 37, 36, 26, 25, 24, 23, 22, 21,
               20, 19, 18],
              [ 6,  7,  8,  3,  4,  5,  0,  1,  2, 33, 34, 35, 30, 31, 32, 27, 28,
               29, 24, 25, 26, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14,  9,
               10, 11, 42, 43, 44, 39, 40, 41, 36, 37, 38, 51, 52, 53, 48, 49, 50,
               45, 46, 47],
              [ 6,  3,  0,  7,  4,  1,  8,  5,  2, 53, 52, 51, 50, 49, 48, 47, 46,
               45,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
               25, 26, 38, 41, 44, 37, 40, 43, 36, 39, 42, 35, 34, 33, 32, 31, 30,
               29, 28, 27],
              [ 0,  3,  6,  1,  4,  7,  2,  5,  8, 47, 46, 45, 50, 49, 48, 53, 52,
               51, 33, 34, 35, 30, 31, 32, 27, 28, 29, 24, 25, 26, 21, 22, 23, 18,
               19, 20, 44, 41, 38, 43, 40, 37, 42, 39, 36, 11, 10,  9, 14, 13, 12,
               17, 16, 15],
              [15, 12,  9, 16, 13, 10, 17, 14, 11,  2,  5,  8,  1,  4,  7,  0,  3,
                6, 47, 50, 53, 46, 49, 52, 45, 48, 51, 38, 41, 44, 37, 40, 43, 36,
               39, 42, 33, 30, 27, 34, 31, 28, 35, 32, 29, 24, 21, 18, 25, 22, 19,
               26, 23, 20],
              [ 9, 12, 15, 10, 13, 16, 11, 14, 17,  0,  3,  6,  1,  4,  7,  2,  5,
                8, 18, 21, 24, 19, 22, 25, 20, 23, 26, 36, 39, 42, 37, 40, 43, 38,
               41, 44, 27, 30, 33, 28, 31, 34, 29, 32, 35, 53, 50, 47, 52, 49, 46,
               51, 48, 45],
              [17, 16, 15, 14, 13, 12, 11, 10,  9, 20, 23, 26, 19, 22, 25, 18, 21,
               24,  2,  5,  8,  1,  4,  7,  0,  3,  6, 47, 50, 53, 46, 49, 52, 45,
               48, 51, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42, 39, 36, 43, 40, 37,
               44, 41, 38],
              [15, 16, 17, 12, 13, 14,  9, 10, 11, 18, 21, 24, 19, 22, 25, 20, 23,
               26, 36, 39, 42, 37, 40, 43, 38, 41, 44, 45, 48, 51, 46, 49, 52, 47,
               50, 53, 29, 28, 27, 32, 31, 30, 35, 34, 33,  8,  5,  2,  7,  4,  1,
                6,  3,  0],
              [11, 14, 17, 10, 13, 16,  9, 12, 15, 38, 41, 44, 37, 40, 43, 36, 39,
               42, 20, 23, 26, 19, 22, 25, 18, 21, 24,  2,  5,  8,  1,  4,  7,  0,
                3,  6, 29, 32, 35, 28, 31, 34, 27, 30, 33, 51, 48, 45, 52, 49, 46,
               53, 50, 47],
              [17, 14, 11, 16, 13, 10, 15, 12,  9, 36, 39, 42, 37, 40, 43, 38, 41,
               44, 45, 48, 51, 46, 49, 52, 47, 50, 53,  0,  3,  6,  1,  4,  7,  2,
                5,  8, 35, 32, 29, 34, 31, 28, 33, 30, 27, 26, 23, 20, 25, 22, 19,
               24, 21, 18],
              [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 47, 50, 53, 46, 49, 52, 45, 48,
               51, 38, 41, 44, 37, 40, 43, 36, 39, 42, 20, 23, 26, 19, 22, 25, 18,
               21, 24, 35, 34, 33, 32, 31, 30, 29, 28, 27,  6,  3,  0,  7,  4,  1,
                8,  5,  2],
              [11, 10,  9, 14, 13, 12, 17, 16, 15, 45, 48, 51, 46, 49, 52, 47, 50,
               53,  0,  3,  6,  1,  4,  7,  2,  5,  8, 18, 21, 24, 19, 22, 25, 20,
               23, 26, 33, 34, 35, 30, 31, 32, 27, 28, 29, 44, 41, 38, 43, 40, 37,
               42, 39, 36],
              [24, 21, 18, 25, 22, 19, 26, 23, 20,  8,  7,  6,  5,  4,  3,  2,  1,
                0, 15, 12,  9, 16, 13, 10, 17, 14, 11, 36, 37, 38, 39, 40, 41, 42,
               43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51, 33, 30, 27, 34, 31, 28,
               35, 32, 29],
              [18, 21, 24, 19, 22, 25, 20, 23, 26,  2,  1,  0,  5,  4,  3,  8,  7,
                6, 27, 30, 33, 28, 31, 34, 29, 32, 35, 42, 43, 44, 39, 40, 41, 36,
               37, 38, 53, 50, 47, 52, 49, 46, 51, 48, 45,  9, 12, 15, 10, 13, 16,
               11, 14, 17],
              [18, 19, 20, 21, 22, 23, 24, 25, 26, 15, 12,  9, 16, 13, 10, 17, 14,
               11, 36, 37, 38, 39, 40, 41, 42, 43, 44, 29, 32, 35, 28, 31, 34, 27,
               30, 33, 45, 46, 47, 48, 49, 50, 51, 52, 53,  0,  1,  2,  3,  4,  5,
                6,  7,  8],
              [20, 19, 18, 23, 22, 21, 26, 25, 24, 17, 14, 11, 16, 13, 10, 15, 12,
                9,  2,  1,  0,  5,  4,  3,  8,  7,  6, 27, 30, 33, 28, 31, 34, 29,
               32, 35, 47, 46, 45, 50, 49, 48, 53, 52, 51, 38, 37, 36, 41, 40, 39,
               44, 43, 42],
              [26, 25, 24, 23, 22, 21, 20, 19, 18, 29, 32, 35, 28, 31, 34, 27, 30,
               33,  8,  7,  6,  5,  4,  3,  2,  1,  0, 15, 12,  9, 16, 13, 10, 17,
               14, 11, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39,
               38, 37, 36],
              [24, 25, 26, 21, 22, 23, 18, 19, 20, 27, 30, 33, 28, 31, 34, 29, 32,
               35, 42, 43, 44, 39, 40, 41, 36, 37, 38, 17, 14, 11, 16, 13, 10, 15,
               12,  9, 51, 52, 53, 48, 49, 50, 45, 46, 47,  6,  7,  8,  3,  4,  5,
                0,  1,  2],
              [20, 23, 26, 19, 22, 25, 18, 21, 24, 36, 37, 38, 39, 40, 41, 42, 43,
               44, 29, 32, 35, 28, 31, 34, 27, 30, 33,  8,  7,  6,  5,  4,  3,  2,
                1,  0, 51, 48, 45, 52, 49, 46, 53, 50, 47, 11, 14, 17, 10, 13, 16,
                9, 12, 15],
              [26, 23, 20, 25, 22, 19, 24, 21, 18, 42, 43, 44, 39, 40, 41, 36, 37,
               38, 17, 14, 11, 16, 13, 10, 15, 12,  9,  2,  1,  0,  5,  4,  3,  8,
                7,  6, 45, 48, 51, 46, 49, 52, 47, 50, 53, 35, 32, 29, 34, 31, 28,
               33, 30, 27],
              [33, 30, 27, 34, 31, 28, 35, 32, 29,  6,  3,  0,  7,  4,  1,  8,  5,
                2, 24, 21, 18, 25, 22, 19, 26, 23, 20, 42, 39, 36, 43, 40, 37, 44,
               41, 38, 15, 12,  9, 16, 13, 10, 17, 14, 11, 47, 50, 53, 46, 49, 52,
               45, 48, 51],
              [27, 30, 33, 28, 31, 34, 29, 32, 35,  8,  5,  2,  7,  4,  1,  6,  3,
                0, 53, 50, 47, 52, 49, 46, 51, 48, 45, 44, 41, 38, 43, 40, 37, 42,
               39, 36,  9, 12, 15, 10, 13, 16, 11, 14, 17, 18, 21, 24, 19, 22, 25,
               20, 23, 26],
              [27, 28, 29, 30, 31, 32, 33, 34, 35, 24, 21, 18, 25, 22, 19, 26, 23,
               20, 42, 39, 36, 43, 40, 37, 44, 41, 38, 51, 48, 45, 52, 49, 46, 53,
               50, 47, 17, 16, 15, 14, 13, 12, 11, 10,  9,  2,  5,  8,  1,  4,  7,
                0,  3,  6],
              [29, 28, 27, 32, 31, 30, 35, 34, 33, 26, 23, 20, 25, 22, 19, 24, 21,
               18,  8,  5,  2,  7,  4,  1,  6,  3,  0, 53, 50, 47, 52, 49, 46, 51,
               48, 45, 15, 16, 17, 12, 13, 14,  9, 10, 11, 36, 39, 42, 37, 40, 43,
               38, 41, 44],
              [29, 32, 35, 28, 31, 34, 27, 30, 33, 42, 39, 36, 43, 40, 37, 44, 41,
               38, 51, 48, 45, 52, 49, 46, 53, 50, 47,  6,  3,  0,  7,  4,  1,  8,
                5,  2, 11, 14, 17, 10, 13, 16,  9, 12, 15, 20, 23, 26, 19, 22, 25,
               18, 21, 24],
              [35, 32, 29, 34, 31, 28, 33, 30, 27, 44, 41, 38, 43, 40, 37, 42, 39,
               36, 26, 23, 20, 25, 22, 19, 24, 21, 18,  8,  5,  2,  7,  4,  1,  6,
                3,  0, 17, 14, 11, 16, 13, 10, 15, 12,  9, 45, 48, 51, 46, 49, 52,
               47, 50, 53],
              [35, 34, 33, 32, 31, 30, 29, 28, 27, 51, 48, 45, 52, 49, 46, 53, 50,
               47,  6,  3,  0,  7,  4,  1,  8,  5,  2, 24, 21, 18, 25, 22, 19, 26,
               23, 20,  9, 10, 11, 12, 13, 14, 15, 16, 17, 38, 41, 44, 37, 40, 43,
               36, 39, 42],
              [33, 34, 35, 30, 31, 32, 27, 28, 29, 53, 50, 47, 52, 49, 46, 51, 48,
               45, 44, 41, 38, 43, 40, 37, 42, 39, 36, 26, 23, 20, 25, 22, 19, 24,
               21, 18, 11, 10,  9, 14, 13, 12, 17, 16, 15,  0,  3,  6,  1,  4,  7,
                2,  5,  8],
              [36, 37, 38, 39, 40, 41, 42, 43, 44, 17, 16, 15, 14, 13, 12, 11, 10,
                9, 45, 46, 47, 48, 49, 50, 51, 52, 53, 35, 34, 33, 32, 31, 30, 29,
               28, 27,  0,  1,  2,  3,  4,  5,  6,  7,  8, 18, 19, 20, 21, 22, 23,
               24, 25, 26],
              [38, 37, 36, 41, 40, 39, 44, 43, 42, 11, 10,  9, 14, 13, 12, 17, 16,
               15, 20, 19, 18, 23, 22, 21, 26, 25, 24, 29, 28, 27, 32, 31, 30, 35,
               34, 33,  2,  1,  0,  5,  4,  3,  8,  7,  6, 47, 46, 45, 50, 49, 48,
               53, 52, 51],
              [42, 39, 36, 43, 40, 37, 44, 41, 38, 26, 25, 24, 23, 22, 21, 20, 19,
               18, 17, 16, 15, 14, 13, 12, 11, 10,  9, 45, 46, 47, 48, 49, 50, 51,
               52, 53,  2,  5,  8,  1,  4,  7,  0,  3,  6, 27, 28, 29, 30, 31, 32,
               33, 34, 35],
              [36, 39, 42, 37, 40, 43, 38, 41, 44, 20, 19, 18, 23, 22, 21, 26, 25,
               24, 29, 28, 27, 32, 31, 30, 35, 34, 33, 51, 52, 53, 48, 49, 50, 45,
               46, 47,  8,  5,  2,  7,  4,  1,  6,  3,  0, 15, 16, 17, 12, 13, 14,
                9, 10, 11],
              [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28,
               27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
               10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 53, 52, 51, 50, 49, 48,
               47, 46, 45],
              [42, 43, 44, 39, 40, 41, 36, 37, 38, 29, 28, 27, 32, 31, 30, 35, 34,
               33, 51, 52, 53, 48, 49, 50, 45, 46, 47, 11, 10,  9, 14, 13, 12, 17,
               16, 15,  6,  7,  8,  3,  4,  5,  0,  1,  2, 24, 25, 26, 21, 22, 23,
               18, 19, 20],
              [38, 41, 44, 37, 40, 43, 36, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52,
               53, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
               19, 18,  6,  3,  0,  7,  4,  1,  8,  5,  2,  9, 10, 11, 12, 13, 14,
               15, 16, 17],
              [44, 41, 38, 43, 40, 37, 42, 39, 36, 51, 52, 53, 48, 49, 50, 45, 46,
               47, 11, 10,  9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21, 26,
               25, 24,  0,  3,  6,  1,  4,  7,  2,  5,  8, 33, 34, 35, 30, 31, 32,
               27, 28, 29],
              [47, 50, 53, 46, 49, 52, 45, 48, 51,  0,  1,  2,  3,  4,  5,  6,  7,
                8, 33, 30, 27, 34, 31, 28, 35, 32, 29, 44, 43, 42, 41, 40, 39, 38,
               37, 36, 24, 21, 18, 25, 22, 19, 26, 23, 20, 15, 12,  9, 16, 13, 10,
               17, 14, 11],
              [53, 50, 47, 52, 49, 46, 51, 48, 45,  6,  7,  8,  3,  4,  5,  0,  1,
                2,  9, 12, 15, 10, 13, 16, 11, 14, 17, 38, 37, 36, 41, 40, 39, 44,
               43, 42, 18, 21, 24, 19, 22, 25, 20, 23, 26, 27, 30, 33, 28, 31, 34,
               29, 32, 35],
              [45, 46, 47, 48, 49, 50, 51, 52, 53, 11, 14, 17, 10, 13, 16,  9, 12,
               15,  0,  1,  2,  3,  4,  5,  6,  7,  8, 33, 30, 27, 34, 31, 28, 35,
               32, 29, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41,
               42, 43, 44],
              [47, 46, 45, 50, 49, 48, 53, 52, 51,  9, 12, 15, 10, 13, 16, 11, 14,
               17, 38, 37, 36, 41, 40, 39, 44, 43, 42, 35, 32, 29, 34, 31, 28, 33,
               30, 27, 20, 19, 18, 23, 22, 21, 26, 25, 24,  2,  1,  0,  5,  4,  3,
                8,  7,  6],
              [53, 52, 51, 50, 49, 48, 47, 46, 45, 33, 30, 27, 34, 31, 28, 35, 32,
               29, 44, 43, 42, 41, 40, 39, 38, 37, 36, 11, 14, 17, 10, 13, 16,  9,
               12, 15, 26, 25, 24, 23, 22, 21, 20, 19, 18,  8,  7,  6,  5,  4,  3,
                2,  1,  0],
              [51, 52, 53, 48, 49, 50, 45, 46, 47, 35, 32, 29, 34, 31, 28, 33, 30,
               27,  6,  7,  8,  3,  4,  5,  0,  1,  2,  9, 12, 15, 10, 13, 16, 11,
               14, 17, 24, 25, 26, 21, 22, 23, 18, 19, 20, 42, 43, 44, 39, 40, 41,
               36, 37, 38],
              [51, 48, 45, 52, 49, 46, 53, 50, 47, 44, 43, 42, 41, 40, 39, 38, 37,
               36, 11, 14, 17, 10, 13, 16,  9, 12, 15,  0,  1,  2,  3,  4,  5,  6,
                7,  8, 20, 23, 26, 19, 22, 25, 18, 21, 24, 29, 32, 35, 28, 31, 34,
               27, 30, 33],
              [45, 48, 51, 46, 49, 52, 47, 50, 53, 38, 37, 36, 41, 40, 39, 44, 43,
               42, 35, 32, 29, 34, 31, 28, 33, 30, 27,  6,  7,  8,  3,  4,  5,  0,
                1,  2, 26, 23, 20, 25, 22, 19, 24, 21, 18, 17, 14, 11, 16, 13, 10,
               15, 12,  9]])

opp_action_permutations = \
    np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
              [ 1,  0,  3,  2,  5,  4,  7,  6, 11, 10,  9,  8],
              [ 0,  1,  2,  3, 10, 11,  8,  9,  4,  5,  6,  7],
              [ 1,  0,  3,  2,  9,  8, 11, 10,  5,  4,  7,  6],
              [ 0,  1,  2,  3,  6,  7,  4,  5, 10, 11,  8,  9],
              [ 1,  0,  3,  2,  7,  6,  5,  4,  9,  8, 11, 10],
              [ 0,  1,  2,  3,  8,  9, 10, 11,  6,  7,  4,  5],
              [ 1,  0,  3,  2, 11, 10,  9,  8,  7,  6,  5,  4],
              [ 4,  5,  6,  7,  0,  1,  2,  3, 10, 11,  8,  9],
              [ 5,  4,  7,  6,  1,  0,  3,  2,  9,  8, 11, 10],
              [ 8,  9, 10, 11,  0,  1,  2,  3,  4,  5,  6,  7],
              [11, 10,  9,  8,  1,  0,  3,  2,  5,  4,  7,  6],
              [ 6,  7,  4,  5,  0,  1,  2,  3,  8,  9, 10, 11],
              [ 7,  6,  5,  4,  1,  0,  3,  2, 11, 10,  9,  8],
              [10, 11,  8,  9,  0,  1,  2,  3,  6,  7,  4,  5],
              [ 9,  8, 11, 10,  1,  0,  3,  2,  7,  6,  5,  4],
              [ 4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3],
              [ 5,  4,  7,  6, 11, 10,  9,  8,  1,  0,  3,  2],
              [10, 11,  8,  9,  4,  5,  6,  7,  0,  1,  2,  3],
              [ 9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2],
              [ 8,  9, 10, 11,  6,  7,  4,  5,  0,  1,  2,  3],
              [11, 10,  9,  8,  7,  6,  5,  4,  1,  0,  3,  2],
              [ 6,  7,  4,  5, 10, 11,  8,  9,  0,  1,  2,  3],
              [ 7,  6,  5,  4,  9,  8, 11, 10,  1,  0,  3,  2],
              [ 4,  5,  6,  7,  2,  3,  0,  1,  8,  9, 10, 11],
              [ 5,  4,  7,  6,  3,  2,  1,  0, 11, 10,  9,  8],
              [10, 11,  8,  9,  2,  3,  0,  1,  4,  5,  6,  7],
              [ 9,  8, 11, 10,  3,  2,  1,  0,  5,  4,  7,  6],
              [ 6,  7,  4,  5,  2,  3,  0,  1, 10, 11,  8,  9],
              [ 7,  6,  5,  4,  3,  2,  1,  0,  9,  8, 11, 10],
              [ 8,  9, 10, 11,  2,  3,  0,  1,  6,  7,  4,  5],
              [11, 10,  9,  8,  3,  2,  1,  0,  7,  6,  5,  4],
              [ 2,  3,  0,  1,  4,  5,  6,  7, 10, 11,  8,  9],
              [ 3,  2,  1,  0,  5,  4,  7,  6,  9,  8, 11, 10],
              [ 2,  3,  0,  1,  8,  9, 10, 11,  4,  5,  6,  7],
              [ 3,  2,  1,  0, 11, 10,  9,  8,  5,  4,  7,  6],
              [ 2,  3,  0,  1,  6,  7,  4,  5,  8,  9, 10, 11],
              [ 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8],
              [ 2,  3,  0,  1, 10, 11,  8,  9,  6,  7,  4,  5],
              [ 3,  2,  1,  0,  9,  8, 11, 10,  7,  6,  5,  4],
              [ 4,  5,  6,  7, 10, 11,  8,  9,  2,  3,  0,  1],
              [ 5,  4,  7,  6,  9,  8, 11, 10,  3,  2,  1,  0],
              [ 8,  9, 10, 11,  4,  5,  6,  7,  2,  3,  0,  1],
              [11, 10,  9,  8,  5,  4,  7,  6,  3,  2,  1,  0],
              [10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1],
              [ 9,  8, 11, 10,  7,  6,  5,  4,  3,  2,  1,  0],
              [ 6,  7,  4,  5,  8,  9, 10, 11,  2,  3,  0,  1],
              [ 7,  6,  5,  4, 11, 10,  9,  8,  3,  2,  1,  0]])

action_permutations = \
    np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
              [ 1,  0,  3,  2,  5,  4,  7,  6, 11, 10,  9,  8],
              [ 0,  1,  2,  3,  8,  9, 10, 11,  6,  7,  4,  5],
              [ 1,  0,  3,  2,  9,  8, 11, 10,  5,  4,  7,  6],
              [ 0,  1,  2,  3,  6,  7,  4,  5, 10, 11,  8,  9],
              [ 1,  0,  3,  2,  7,  6,  5,  4,  9,  8, 11, 10],
              [ 0,  1,  2,  3, 10, 11,  8,  9,  4,  5,  6,  7],
              [ 1,  0,  3,  2, 11, 10,  9,  8,  7,  6,  5,  4],
              [ 4,  5,  6,  7,  0,  1,  2,  3, 10, 11,  8,  9],
              [ 5,  4,  7,  6,  1,  0,  3,  2,  9,  8, 11, 10],
              [ 4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3],
              [ 5,  4,  7,  6,  9,  8, 11, 10,  3,  2,  1,  0],
              [ 4,  5,  6,  7,  2,  3,  0,  1,  8,  9, 10, 11],
              [ 5,  4,  7,  6,  3,  2,  1,  0, 11, 10,  9,  8],
              [ 4,  5,  6,  7, 10, 11,  8,  9,  2,  3,  0,  1],
              [ 5,  4,  7,  6, 11, 10,  9,  8,  1,  0,  3,  2],
              [ 8,  9, 10, 11,  0,  1,  2,  3,  4,  5,  6,  7],
              [ 9,  8, 11, 10,  1,  0,  3,  2,  7,  6,  5,  4],
              [ 8,  9, 10, 11,  4,  5,  6,  7,  2,  3,  0,  1],
              [ 9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2],
              [ 8,  9, 10, 11,  6,  7,  4,  5,  0,  1,  2,  3],
              [ 9,  8, 11, 10,  7,  6,  5,  4,  3,  2,  1,  0],
              [ 8,  9, 10, 11,  2,  3,  0,  1,  6,  7,  4,  5],
              [ 9,  8, 11, 10,  3,  2,  1,  0,  5,  4,  7,  6],
              [ 6,  7,  4,  5,  0,  1,  2,  3,  8,  9, 10, 11],
              [ 7,  6,  5,  4,  1,  0,  3,  2, 11, 10,  9,  8],
              [ 6,  7,  4,  5,  8,  9, 10, 11,  2,  3,  0,  1],
              [ 7,  6,  5,  4,  9,  8, 11, 10,  1,  0,  3,  2],
              [ 6,  7,  4,  5,  2,  3,  0,  1, 10, 11,  8,  9],
              [ 7,  6,  5,  4,  3,  2,  1,  0,  9,  8, 11, 10],
              [ 6,  7,  4,  5, 10, 11,  8,  9,  0,  1,  2,  3],
              [ 7,  6,  5,  4, 11, 10,  9,  8,  3,  2,  1,  0],
              [ 2,  3,  0,  1,  4,  5,  6,  7, 10, 11,  8,  9],
              [ 3,  2,  1,  0,  5,  4,  7,  6,  9,  8, 11, 10],
              [ 2,  3,  0,  1,  8,  9, 10, 11,  4,  5,  6,  7],
              [ 3,  2,  1,  0,  9,  8, 11, 10,  7,  6,  5,  4],
              [ 2,  3,  0,  1,  6,  7,  4,  5,  8,  9, 10, 11],
              [ 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8],
              [ 2,  3,  0,  1, 10, 11,  8,  9,  6,  7,  4,  5],
              [ 3,  2,  1,  0, 11, 10,  9,  8,  5,  4,  7,  6],
              [10, 11,  8,  9,  0,  1,  2,  3,  6,  7,  4,  5],
              [11, 10,  9,  8,  1,  0,  3,  2,  5,  4,  7,  6],
              [10, 11,  8,  9,  4,  5,  6,  7,  0,  1,  2,  3],
              [11, 10,  9,  8,  5,  4,  7,  6,  3,  2,  1,  0],
              [10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1],
              [11, 10,  9,  8,  7,  6,  5,  4,  1,  0,  3,  2],
              [10, 11,  8,  9,  2,  3,  0,  1,  4,  5,  6,  7],
              [11, 10,  9,  8,  3,  2,  1,  0,  7,  6,  5,  4]])

class BatchActionCombo:
    """ 
    This class stores an array of actions, which can be a combination of the 12 basic actions.
    The actions are represented as permutations of [0, ... , 53]

    This is supposed to behave as an immutable object.
    """
    def __init__(self, permutations):
        self._permutations = permutations.copy()

    def __len__(self):
        return len(self._permutations)
    
    def multiply(self, other):
        """
        Combine actions in parallel.  Assuming the shapes are the same.
        """
        idx = np.indices(self._permutations.shape)[0]
        new_permutations = self._permutations[idx, other._permutations]
        return BatchActionCombo(new_permutations)

    def outer_multiply(self, other):
        """
        Combine actions independently.  Result has size len(self) x len(other).
        """
        self_len = len(self)
        other_len = len(other)

        permutations_0 = np.repeat(self._permutations, repeats=other_len, axis=0)
        idx = np.indices(permutations_0.shape)[0]
        
        permutations_1 = np.tile(other._permutations, (self_len, 1))

        new_permutations = permutations_0[idx, permutations_1]
        return BatchActionCombo(new_permutations)

    def remove_duplicates(self):
        """
        Removes duplicates by passing through a set
        This may change the order.
        """
        return BatchActionCombo(np.array(list({tuple(perm) for perm in self._permutations})))

    # static methods

    @staticmethod
    def identity():
        return BatchActionCombo(np.arange(54)[np.newaxis])

    @staticmethod
    def basic_actions(action_numbers):
        return BatchActionCombo(action_array[action_numbers])

    @staticmethod
    def all_actions_up_to(radius):
        basic_actions = BatchActionCombo.basic_actions(np.arange(12))
        tuples_outer_level = {tuple(range(54))}
        tuples = {tuple(range(54))}
        for _ in range(radius):
            outer_actions = BatchActionCombo(np.array(list(tuples_outer_level)))
            actions = outer_actions.outer_multiply(basic_actions)
            tuples_outer_level = {row_tuple for row_tuple in map(tuple, actions._permutations)
                                            if row_tuple not in tuples_outer_level}
            tuples.update(tuples_outer_level)
            
        return BatchActionCombo(np.array(list(tuples)))


class BatchCube():
    """
    An implementation of a vector of Rubik's cubes using NumPy arrays.
    It has methods for converting to the PyCuber object class.
    """
    def __init__(self, length = 1, cube_array=None, sample_index=None):
        """
        Creates length-many solved cubes
        """
        if cube_array is None:
            self._cube_array = np.repeat(solved_cube_list[np.newaxis], repeats=length, axis=0)
        else:
            self._cube_array = cube_array

        if sample_index is None:
            self._sample_index = np.indices(self._cube_array.shape)[0]
        else:
            self._sample_index = sample_index
    
    def copy(self):
        return BatchCube(cube_array=self._cube_array.copy(), sample_index=self._sample_index)
        
    def __len__(self):
        return self._cube_array.shape[0]
     
    def bit_array(self):
        return eye6[self._cube_array]

    def load_bit_array(self, bit_array):
        """
        Takes in an array of bits of size k * 54 * 6 and converts it to an array of k cubes.
        It overwrites the array in the object.
        """
        
        bit_array = bit_array.reshape((-1, 54, 6))
        self._cube_array = (bit_array * np.arange(6)).sum(axis=2)
        self._sample_index = np.indices(self._cube_array.shape)[0]

    def step(self, actions):
        """
        Assuming actions is a list of length = len(self)
        """
        action_indices = action_array[actions]
        self._cube_array = self._cube_array[self._sample_index, action_indices]

    def perform_action_combo(self, action_combos):
        """
        Assuming the action_combo is an array with same shape as self._cube_array
        """
        self._cube_array = self._cube_array[self._sample_index, action_combos._permutations]

    def perform_action_combo_independent(self, action_combos):
        """
        Performs all action combos independently on each state
        """
        action_len = len(action_combos)
        cubes_len = len(self._cube_array)
        
        self._cube_array = np.repeat(self._cube_array, repeats=action_len, axis=0)
        self._sample_index = np.indices(self._cube_array.shape)[0]
        new_action_combos = BatchActionCombo(np.tile(action_combos._permutations, (cubes_len, 1)))
        
        self.perform_action_combo(new_action_combos)

    def step_independent(self, actions):
        """
        Performs all actions independently on each state
        """
        action_len = len(actions)
        cubes_len = len(self._cube_array)
        
        self._cube_array = np.repeat(self._cube_array, repeats=action_len, axis=0)
        self._sample_index = np.indices(self._cube_array.shape)[0]
        actions = np.tile(actions, cubes_len)
        
        self.step(actions)

    def get_neighbors(self, radius):
        """
        Apply all action combos of radius <= 5 independently.  It only applies equivalent action
        combos once (but if self has length > 1 then there may be duplicates it doesn't remove.)
        """
        self.perform_action_combo_independent(BatchActionCombo.all_actions_up_to(radius))

    def randomize(self, dist=100):
        l = len(self._cube_array)
        for _ in range(dist):
            actions = np.random.choice(12, l)
            self.step(actions)

    def to_pycuber(self):
        """
        Returns a list of PyCuber Cubes
        """
        import pycuber as pc # will raise error if pycuber not installed

        # convert to number representation
        return [pc.Cube(pc.helpers.array_to_cubies([str(int(c)) for c in color_list])) 
                for color_list in self._cube_array]
        
    def from_pycuber(self, pc_list):
        """
        Takes in a list of PyCuber Cubes and converts them to an array.
        It overwrites the array in the object.
        """
        import pycuber as pc # will raise error if pycuber not installed

        # The only good way I know to do this is to first convert the pc.Cube object
        # to a string and read off the values.
        
        pre_array = []
        for cube in pc_list:
            s = str(cube)
            cube_array = np.array([color_dict[s[i]] for i in pc_indices])
            pre_array.append(cube_array)
        self._cube_array = np.array(pre_array)
        self._sample_index = np.indices(self._cube_array.shape)[0]
    
    def done(self):
        return (self._cube_array == solved_cube_list).all(axis=1)
    
    def remove_done(self):
        self._cube_array = self._cube_array[~self.done()]
        self._sample_index = np.indices(self._cube_array.shape)[0]

    def remove_duplicates(self):
        """
        This removes duplicate cubes, but may also change the order
        """
        self._cube_array = np.array(list({tuple(row) for row in self._cube_array}))
        self._sample_index = np.indices(self._cube_array.shape)[0]

    def __str__(self):
        strs = []
        for cube in self._cube_array:
            strs.append(STR_FORMAT.format(*(color_dict2[c] for c in cube[str_order])))
        
        return "\n".join(strs)
        #return "".join(str(c) for c in self.to_pycuber())

    def __eq__(self, other):
        return np.array_equal(self._cube_array, other._cube_array)

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def concat(batch_cube_list):
        return BatchCube(cube_array=np.concatenate([bc._cube_array for bc in batch_cube_list], axis=0))


if __name__ == '__main__':
    import pycuber as pc

    # blank_cube and export
    bc = BatchCube()
    for c in bc.to_pycuber():
        print(c)
    
    # import blank_cube and export it
    bc1 = BatchCube()
    bc1.from_pycuber([pc.Cube()])
    print(len(bc1))
    assert len(bc1) == 1
    assert bc == bc1
    for c in bc1.to_pycuber():
        print(c)
        
    
    # test all actions
    for i, m in enumerate(BASIC_MOVES):
        bc = BatchCube()
        mycube = pc.Cube()
        
        bc.step(np.array([i]))
        mycube.perform_step(m)
        
        for c in bc.to_pycuber():
            print(i, m, c == mycube)
            assert c == mycube, "Action {} (in BatchCube) does not match action {} in PyCuber.  The first gives:\n{}\nThe second gives:\n{}".format(i, m, c, mycube)
    
    # multiple actions
    bc = BatchCube(12)
    cube_list = [pc.Cube() for _ in range(12)]
    
    bc.step(np.arange(12)) # perform all twelve actions in parallel
    for c, m in zip(cube_list, BASIC_MOVES):
        c.perform_step(m)
    
    for c1, c2 in zip(bc.to_pycuber(), cube_list):
        print(c1 == c2)
        assert c1 == c2
    
    # test remove_done
    bc = BatchCube(12)
    
    bc.step(np.arange(12)) # perform all twelve actions in parallel
    bc.step(np.array([0]*12))
    
    print(bc.done())
    assert bc.done().sum() == 1
    assert bc.done()[1] == 1
    bc.remove_done()
    print(len(bc))
    assert len(bc) == 11

    # test independent actions
    bc.step_independent(np.arange(12))
    bc.step_independent(np.arange(12))
    print(len(bc))
    assert len(bc) == 11 * 12 * 12
    print(bc.done().sum())
    assert bc.done().sum() == 14

    # test independent actions
    bc = BatchCube(2)
    bc.step_independent(np.arange(12))
    bc.step_independent(np.arange(12))
    print(len(bc))
    assert len(bc) == 2 * 12 * 12

    #test randomize
    bc = BatchCube(2)
    bc.randomize(1)
    print("Randomized once:")
    for c in bc.to_pycuber():
        print(c)
    print("Randomized many times:")
    bc.randomize()
    for c in bc.to_pycuber():
        print(c)

    #test copy
    bc = BatchCube(1)
    bc2 = bc.copy()
    bc.step_independent(np.arange(12))
    assert len(bc2) == 1
    bc2.step([0])
    bc2.step([1])
    assert bc2.done()[0] == True

    # test remove duplicates
    bc = BatchCube(5)
    bc.step_independent(np.arange(12))
    bc.remove_duplicates()
    assert len(bc) == 12

    # test BatchActionCombo.basic_actions and perform_action_combo (as well as __eq__ and __ne__)
    for a in range(12):
        bc = BatchCube(1)
        bc.step([a])

        bac = BatchActionCombo.basic_actions([a])
        bc2 = BatchCube(1)
        bc2.perform_action_combo(bac)
        assert bc == bc2
        assert not bc != bc2


    # test action combos
    actions1 = np.random.choice(12, 10)
    actions2 = np.random.choice(12, 10)
    actions3 = np.random.choice(12, 10)

    bac1 = BatchActionCombo.basic_actions(actions1)
    bac2 = BatchActionCombo.basic_actions(actions2)
    bac3 = BatchActionCombo.basic_actions(actions3)

    # three sequential methods
    bc_a = BatchCube(10)
    bc_b = BatchCube(10)
    bc_c = BatchCube(10)

    bc_a.step(actions1)
    bc_a.step(actions2)
    bc_a.step(actions3)

    bc_b.perform_action_combo(bac1)
    bc_b.perform_action_combo(bac2)
    bc_b.perform_action_combo(bac3)

    bac = bac1.multiply(bac2).multiply(bac3)
    bc_c.perform_action_combo(bac)

    assert bc_a == bc_b
    assert bc_b == bc_c

    # three independent methods
    bc_a = BatchCube(2)
    bc_b = BatchCube(2)
    bc_c = BatchCube(2)

    bc_a.step_independent(actions1)
    bc_a.step_independent(actions2)
    bc_a.step_independent(actions3)

    bc_b.perform_action_combo_independent(bac1)
    bc_b.perform_action_combo_independent(bac2)
    bc_b.perform_action_combo_independent(bac3)

    bac = bac1.outer_multiply(bac2).outer_multiply(bac3)
    bc_c.perform_action_combo_independent(bac)

    assert bc_a == bc_b
    assert bc_b == bc_c
    

    # test identity and remove_duplicates
    a = BatchActionCombo.identity() \
                        .outer_multiply(BatchActionCombo.basic_actions(np.arange(12))) \
                        .multiply(BatchActionCombo.basic_actions(np.arange(12)^1))
    assert len(a) == 12
    b = a.remove_duplicates()
    assert len(b) == 1


    # test concat
    bc1 = BatchCube(2)
    bc2 = BatchCube(3)
    bc = BatchCube.concat([bc1, bc2])
    assert len(bc) == 5

    # test get_neighbors
    bc3 = BatchCube(1)
    for i in range(3):
        bc1 = BatchCube(1)
        bc1.get_neighbors(i)

        bc2 = BatchCube(1)
        for j in range(i):
            bc2.step_independent(np.arange(12))
            bc2.remove_duplicates()
        
        bc3 = BatchCube.concat([bc3, bc2])
        bc3.remove_duplicates()

        bc4 = BatchCube.concat([bc3, bc1])
        bc4.remove_duplicates()

        print(i, ":", len(bc1), len(bc2), len(bc3), len(bc4))
        assert len(bc1) == len(bc3)
        assert len(bc1) == len(bc4)

    # test load_bit_array
    bc = BatchCube(10)
    bc.randomize(100)
    bit_array = bc.bit_array()
    bc1 = BatchCube(1)
    bc1.load_bit_array(bit_array)
    assert bc == bc1

    # test from_pycuber
    bc = BatchCube(10)
    bc.randomize(100)
    py_cuber_list = bc.to_pycuber()
    bc1 = BatchCube(1)
    bc1.from_pycuber(py_cuber_list)
    print(bc1)
    assert bc == bc1

    # test __str__
    bc = BatchCube(10)
    bc.randomize(100)
    assert str(bc) + "\n" == "".join(str(c) for c in bc.to_pycuber())

    print("All tests successful!")

