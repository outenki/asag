# from all answers near to be correct, find their mean distance to their neighbors
# Used to show the fact that, distance between high score answers are smaller compared to low score answers.

from config import *
from asag_utils import score_distance



if __name__ == '__main__':
    score_distance('euclidean', reliable=True, feature='infer', distance='euclidean', qwise=True)
