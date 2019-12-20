import datetime
from math import floor
from typing import List, Dict

from numpy import matrix, linalg


class FEMException(Exception):
    pass

class WrongElementTypeError(FEMException):
    pass


def print_execution_time(name_to_show: str = ''):
    def decorator(function):
        def wrapper(*args, **kwargs):
            start = datetime.datetime.now()
            out = function(*args, **kwargs)
            end = datetime.datetime.now()
            print(f'{name_to_show} took {round((end - start).total_seconds(), 2)}s')
            return out
        return wrapper
    return decorator


def triangle_area_2d(v1, v2, v3):
    M = matrix([[v1.x - v3.x, v1.y - v3.y],
                [v2.x - v3.x, v2.y - v3.y]])
    area = abs(linalg.det(M) / 2)
    return area


def is_to_the_left(p, p1, p2):
    """ if positive is to the left, if zero is on, if negative is to the right"""
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    D = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return -D


def split_list(list_to_split: List, n_parts) -> List[List]:
    length = len(list_to_split)
    parts = []
    for i in range(n_parts):
        part = list_to_split[int(floor(i / n_parts * length)): int(floor((i + 1) / n_parts * length))]
        parts.append(part)
    return parts


def invert_dict(dictionary: Dict) -> Dict:
    inv_map = {}
    for k, v in dictionary.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map