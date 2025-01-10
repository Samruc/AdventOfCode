import copy
import functools
import sys
from time import sleep

import numpy as np
import re
from collections import Counter, defaultdict
from functools import cmp_to_key
from pprint import pprint

from utils.utils import *

data = read("1", 2024)
col1, col2 = zip(*data)
counter1, counter2 = Counter(col1), Counter(col2)
mark("1A", sum(abs(a - b) for a, b in zip(sorted(col1), sorted(col2))), 2264607)
mark("1B", sum(a * counter1[a] * counter2[a] for a in counter1.keys()), 19457120)

data = read("2", 2024)


def safety(line):
    l_comp, l_base = line[1:], line[:-1]
    return 1 if all(a - b in [1, 2, 3] for a, b in zip(l_base, l_comp)) \
                or all(b - a in [1, 2, 3] for a, b in zip(l_base, l_comp)) else 0


mark("2A", sum([safety(line) for line in data]), 483)
mark("2B", sum([1 if any(safety(line[:i] + line[i + 1:]) for i in range(len(line))) else 0 for line in data]), 528)

data = read("3", 2024, raw=True, join=True)
do_dont = " ".join(s.split("don't()")[0] for s in data.split("do()"))
mark("3A", (sum([int(a) * int(b) for a, b in re.findall(r"mul\((\d{1,3}),(\d{1,3})\)", data)])), 169021493)
mark("3B", (sum([int(a) * int(b) for a, b in re.findall(r"mul\((\d{1,3}),(\d{1,3})\)", do_dont)])), 111762583)

data = read("4", 2024, raw=True, strip=True)
n = len(data)
cols = ["".join(line[i] for line in data) for i in range(n)]
dias = ["".join(data[i - j][j] for j in range(max(i - n + 1, 0), min(i, n - 1) + 1)) for i in range(2 * n - 1)]
dias += ["".join(data[j][n - 1 - i + j] for j in range(max(i - n + 1, 0), min(i, n - 1) + 1)) for i in range(2 * n - 1)]
mark("4A", sum(sum(1 if line[i:i + 4] in ["XMAS", "SAMX"] else 0
                   for i in range(len(line))) for line in data + cols + dias), 2401)
mark("4B", sum(sum(1 if data[i][j] == "A" and
                        all([data[i + 1][j + 1] in ["M", "S"],
                             data[i - 1][j + 1] in ["M", "S"],
                             data[i + 1][j - 1] in ["M", "S"],
                             data[i - 1][j - 1] in ["M", "S"]]) and
                        data[i + 1][j + 1] != data[i - 1][j - 1] and
                        data[i - 1][j + 1] != data[i + 1][j - 1]
                   else 0 for j in range(1, n - 1)) for i in range(1, n - 1)), 1822)

data = read("5", 2024, raw=True, strip=True)
rules = set(data[:data.index("")])
lists = [l.split(",") for l in data[data.index("") + 1:]]
mark("5A", sum(0 if any(i < j and b + "|" + a in rules
                        for i, a in enumerate(l) for j, b in enumerate(l))
               else int(l[len(l) // 2]) for l in lists), 6242)
mark("5B", sum(int(sorted(l, key=cmp_to_key(lambda a, b: 1 if b + "|" + a in rules else
-1 if a + "|" + b in rules else 0))[len(l) // 2])
               if any(i < j and b + "|" + a in rules
                      for i, a in enumerate(l) for j, b in enumerate(l))
               else 0 for l in lists), 5169)

data = read("6", 2024, raw=True, strip=True)
dirs = {"^": (0, -1), ">": (1, 0), "v": (0, 1), "<": (-1, 0)}
turn = {"^": ">", ">": "v", "v": "<", "<": "^"}
start_direction, start_location = None, None
obstacles = set()
for y, l in enumerate(data):
    for x, c in enumerate(l):
        if c in dirs:
            start_location = (x, y)
            start_direction = c
        elif c == "#":
            obstacles.add((x, y))


def guard_path(obstacles, direction, location, add_obstacle):
    loops_found = 0
    visited = {location}
    visited_dir = {str(location) + direction}
    while True:
        try:
            x = location[0] + dirs[direction][0]
            y = location[1] + dirs[direction][1]
            if x < 0 or y < 0 or x >= len(data) or y >= len(data):
                raise IndexError()
            if (x, y) not in obstacles:
                next_location = (x, y)
                if add_obstacle and (x, y) != start_location and next_location not in visited:
                    obstacles.add((x, y))
                    if not guard_path(obstacles, direction, location, add_obstacle=False):
                        loops_found += 1
                    obstacles.remove((x, y))

                location = next_location
                visited.add(location)
                if str(location) + direction in visited_dir:
                    return None
                else:
                    visited_dir.add(str(location) + direction)
            else:
                direction = turn[direction]
        except IndexError:
            break

    return len(visited), loops_found


mark("6A", guard_path(obstacles, start_direction, start_location, add_obstacle=False)[0], 5269)
# mark("6B", guard_path(obstacles, start_direction, start_location, add_obstacle=True)[1], 1957)
mark("6B", None, 1957, skip_and_add_time=4.6)

data = read("7", 2024, raw=True, strip=True)
lines = [l.replace(": ", ":").split(":") for l in data]
lines = [[int(l[0]), [int(s) for s in l[1].split(" ")]] for l in lines]


def all_combinations(target, l, task_b=False):
    combinations = {l[0]}
    l = l[1:]
    while len(l) > 0:
        new_combinations = set()
        for a in combinations:
            if target >= a + l[0]:
                new_combinations.add(a + l[0])
            if target >= a * l[0]:
                new_combinations.add(a * l[0])
            if task_b and target >= int(str(a) + str(l[0])):
                new_combinations.add(int(str(a) + str(l[0])))
        l = l[1:]
        combinations = new_combinations
    return combinations


mark("7A", sum(l[0] if (l[0] in all_combinations(l[0], l[1])) else 0 for l in lines), 2437272016585)
# mark("7B", sum(l[0] if (l[0] in all_combinations(l[0], l[1], True)) else 0 for l in lines), 162987117690649)
mark("7B", None, 162987117690649, skip_and_add_time=2.45)

data = read("8", 2024, raw=True, strip=True)
antennas = defaultdict(list)
x_max, y_max = len(data[0]), len(data)
for y, l in enumerate(data):
    for x, c in enumerate(l):
        if c != ".":
            antennas[c].append((x, y))


def count_hotspots(more_steps=False):
    hotspots = set()
    for key in antennas.keys():
        ants = antennas[key]
        for a in ants:
            for b in ants:
                if a != b:
                    new_x, new_y = 2 * a[0] - b[0], 2 * a[1] - b[1]
                    while 0 <= new_x < x_max and 0 <= new_y < y_max:
                        hotspots.add((new_x, new_y))
                        new_x += a[0] - b[0]
                        new_y += a[1] - b[1]
                        if not more_steps:
                            new_x = -1
    if more_steps:
        for key in antennas.keys():
            for ant in antennas[key]:
                hotspots.add(ant)
    return len(hotspots)


mark("8A", count_hotspots(), 214)
mark("8B", count_hotspots(more_steps=True), 809)

data = read("9", 2024, raw=True, strip=True)[0]


def task9A(data):
    left_digit = 0
    pos = 0
    checksum = 0
    right_digit = len(data) // 2
    while data:
        # Front-fill
        while int(data[0]) > 0:
            checksum += left_digit * pos
            data = str(int(data[0]) - 1) + data[1:]
            pos += 1
        left_digit += 1
        data = data[1:]
        # Back-fill
        while data and int(data[0]) > 0:
            while data and int(data[-1]) == 0:
                data = data[:-2]
                right_digit -= 1
            if data:
                checksum += right_digit * pos
                data = str(int(data[0]) - 1) + data[1:-1] + str(int(data[-1]) - 1)
            pos += 1
        data = data[1:]
    return checksum


def task9B(d=None):
    start_len = len(d)
    digits = list(range(0, len(d) // 2 + 1))
    d = list(int(n) for n in d)
    digit_to_move = len(d) // 2
    while digit_to_move > 0:
        digit_to_move_index = digits.index(digit_to_move) * 2
        space_needed = d[digit_to_move_index]
        space_searcher = 1
        while space_searcher < digit_to_move_index and d[space_searcher] < space_needed:
            if digits[space_searcher // 2] == digit_to_move:
                space_searcher += digit_to_move_index  # Give up
            else:
                space_searcher += 2
        if space_searcher < digit_to_move_index:
            additional_hole_after = 0 if digit_to_move_index >= len(d) - 1 \
                else d[digit_to_move_index + 1]
            if space_searcher == digit_to_move_index - 1:
                d = (d[:space_searcher] + [0] + [space_needed] +
                     [space_needed + d[digit_to_move_index + 1]] +
                     d[digit_to_move_index + 2:])
            else:
                d = (d[:space_searcher] + [0] + [space_needed] +
                     [d[space_searcher] - space_needed] +
                     d[space_searcher + 1:digit_to_move_index - 1] +
                     [d[digit_to_move_index - 1] + space_needed + additional_hole_after] +
                     d[digit_to_move_index + 2:])
            digits = (digits[:space_searcher // 2 + 1] +
                      [digit_to_move] +
                      digits[space_searcher // 2 + 1: digit_to_move_index // 2] +
                      digits[digit_to_move_index // 2 + 1:])
            d = d[:start_len]

        digit_to_move -= 1

    # Checksum with unordered digits
    checksum = 0
    digit_index = 0
    pos = 0
    while d and digit_index < len(digits):
        while d[0] > 0:
            checksum += digits[digit_index] * pos
            d = [d[0] - 1] + d[1:]
            pos += 1
        digit_index += 1
        d = d[1:]
        # Skip gaps
        if d:
            pos += d[0]
            d = d[1:]

    return checksum


mark("9A", task9A(data), 6291146824486)
# mark("9B", task9B(data), 6307279963620)
mark("9B", None, 162987117690649, skip_and_add_time=5.57)

data = read("10", 2024, raw=True, strip=True)


@functools.cache
def nines_reachable_from(x, y, count=False):
    global data
    current_value = int(data[y][x])
    if current_value == 9:
        return 1 if count else {(x, y)}
    neighbor_candidates = [[x, y + 1], [x + 1, y], [x, y - 1], [x - 1, y]]
    candidates = list(filter(lambda p: 0 <= p[0] < len(data) and 0 <= p[1] < len(data), neighbor_candidates))
    step_candidates = list(filter(lambda p: int(data[p[1]][p[0]]) == current_value + 1, candidates))
    if step_candidates:
        if count:
            nines = 0
            for sc in step_candidates:
                nines += nines_reachable_from(sc[0], sc[1], count=True)
            return nines
        else:
            nines = set()
            for sc in step_candidates:
                nines = nines.union({nine for nine in nines_reachable_from(sc[0], sc[1])})
            return nines
    else:
        return 0 if count else set()


def task10(data, count=False):
    trailheads = 0
    for y, l in enumerate(data):
        for x, c in enumerate(l):
            if c == "0":
                trailheads += nines_reachable_from(x, y, count) if count else len(nines_reachable_from(x, y, count))
    return trailheads


mark("10A", task10(data), 587)
mark("10B", task10(data, count=True), 1340)

data = read("11", 2024)[0]


def split(numbers, times):
    if times == 0:
        return numbers
    else:
        split_numbers = Counter([])
        for num in numbers:
            if num == 0:
                split_numbers[1] += numbers[0]
            elif len(str(num)) % 2 == 0:
                new1 = int(str(num)[:len(str(num)) // 2])
                new2 = int(str(num)[len(str(num)) // 2:])
                split_numbers[new1] += numbers[num]
                split_numbers[new2] += numbers[num]
            else:
                split_numbers[num * 2024] += numbers[num]
        return split(split_numbers, times - 1)


def task11(numbers, times):
    return sum(split(Counter(numbers), times).values())


mark("11A", task11(data, 25), 235850)
mark("11B", task11(data, 75), 279903140844645)

data = read("12", 2024, raw=True, strip=True)

unvisited = set()
visited = set()
for y, line in enumerate(data):
    for x, c in enumerate(line):
        unvisited.add((x, y, c))


def score_region(tile, area, perimeter, corners):
    global unvisited, visited
    nbs = [(tile[0] + 1, tile[1], tile[2]),
           (tile[0] - 1, tile[1], tile[2]),
           (tile[0], tile[1] + 1, tile[2]),
           (tile[0], tile[1] - 1, tile[2])]

    area += 1

    # Count perimeter
    for nb in nbs:
        if nb not in unvisited and nb not in visited:
            perimeter += 1

    # Count corners
    dirs = [(+1, +0),
            (+1, +1),
            (+0, +1),
            (-1, +1),
            (-1, +0),
            (-1, -1),
            (+0, -1),
            (+1, -1), ]
    is_inside = []
    for d in dirs:
        nb = (tile[0] + d[0], tile[1] + d[1], tile[2])
        is_inside.append(nb in unvisited or nb in visited)
    for i in [0, 2, 4, 6]:
        if is_inside[i] and is_inside[(i + 2) % 8] and not is_inside[i + 1]:
            corners += 1
        if not is_inside[i] and not is_inside[(i + 2) % 8]:
            corners += 1

    if not any(nb in unvisited for nb in nbs):
        return area, perimeter, corners

    for nb in nbs:
        if nb in unvisited:
            unvisited.remove(nb)
            visited.add(nb)
            a, p, c = score_region(nb, 0, 0, 0)
            area += a
            perimeter += p
            corners += c

    return area, perimeter, corners


def task12():
    global unvisited, visited
    scoreA, scoreB = 0, 0
    while unvisited:
        visited = set()
        next_tile = unvisited.pop()
        visited.add(next_tile)
        a, p, c = score_region(next_tile, 0, 0, 0)
        scoreA += a * p
        scoreB += a * c
    return scoreA, scoreB


ansA, ansB = task12()
mark("12A", ansA, 1471452)
mark("12B", ansB, 863366)

data = read("13", 2024, raw=True, strip=True)
rows = []
while len(data) >= 3:
    row = [int(data[0].split("+")[1].split(",")[0]),
           int(data[0].split("+")[2]),
           int(data[1].split("+")[1].split(",")[0]),
           int(data[1].split("+")[2]),
           int(data[2].split("=")[1].split(",")[0]),
           int(data[2].split("=")[2])]
    data = data[4:]
    rows += [row]


def task13(rows, taskB=False):
    tokens = 0
    for ax, ay, bx, by, X, Y in rows:
        new_cand = None
        if taskB:
            X += 10000000000000
            Y += 10000000000000
        A = np.array([[ax, bx], [ay, by]])
        guess = np.dot(np.linalg.inv(A), [X, Y])
        k = round(guess[0])
        m = round(guess[1])
        if k * ax + m * bx == X and k * ay + m * by == Y:
            new_cand = 3 * k + m
        tokens += new_cand if new_cand else 0
    return tokens


mark("13A", task13(rows), 29517)
mark("13B", task13(rows, taskB=True), 103570327981381)

data = read("14", 2024, raw=True, strip=True)
robots = []
for row in data:
    r = row.replace(",", " ").replace("=", " ").split(" ")
    robots.append([int(a) for a in r[1:3] + r[4:]])


def task14(robots, seconds, max_x, max_y):
    Q1, Q2, Q3, Q4 = 0, 0, 0, 0
    for r in robots:
        r[0] = (r[0] + seconds * r[2]) % max_x
        r[1] = (r[1] + seconds * r[3]) % max_y
        if r[0] < max_x // 2 and r[1] < max_y // 2:
            Q1 += 1
        elif r[0] < max_x // 2 and r[1] >= max_y // 2 + 1:
            Q2 += 1
        elif r[0] >= max_x // 2 + 1 and r[1] < max_y // 2:
            Q3 += 1
        elif r[0] >= max_x // 2 + 1 and r[1] >= max_y // 2 + 1:
            Q4 += 1
    return Q1 * Q2 * Q3 * Q4, robots


mark("14A", task14(robots, 100, 101, 103)[0], 215987200)
# Task 14B solved by inspection!

data = read("15", 2024, raw=True, strip=True)
walls, boxes = set(), set()
pos = None
max_x = len(data[0])
max_y = 0

for y, line in enumerate(data):
    if line == "":
        break
    max_y += 1
    for x, c in enumerate(line):
        if c == "#":
            walls.add((x, y))
        elif c == "O":
            boxes.add((x, y))
        elif c == "@":
            pos = (x, y)
        elif c == ".":
            pass
        else:
            assert False, c


def pushable(d):
    global dirs, walls, boxes, pos
    temp_pos = pos
    offset = dirs[d]
    while (temp_pos[0] + offset[0], temp_pos[1] + offset[1]) not in walls:
        if (temp_pos[0] + offset[0], temp_pos[1] + offset[1]) not in boxes:
            return True
        temp_pos = (temp_pos[0] + offset[0], temp_pos[1] + offset[1])
    return False


def push(d):
    global dirs, boxes, pos
    temp_pos = pos
    offset = dirs[d]
    pos = (pos[0] + offset[0], pos[1] + offset[1])
    if (temp_pos[0] + offset[0], temp_pos[1] + offset[1]) in boxes:
        boxes.remove((temp_pos[0] + offset[0], temp_pos[1] + offset[1]))
        temp_pos = temp_pos[0] + offset[0], temp_pos[1] + offset[1]
        while (temp_pos[0] + offset[0], temp_pos[1] + offset[1]) in boxes:
            temp_pos = temp_pos[0] + offset[0], temp_pos[1] + offset[1]
        boxes.add((temp_pos[0] + offset[0], temp_pos[1] + offset[1]))


moves = "".join(data[max_y + 1:])
while moves:
    if pushable(moves[0]):
        push(moves[0])
    moves = moves[1:]

mark("15A", sum(b[0] + 100 * b[1] for b in boxes), 1430536)

walls, boxes = set(), set()
pos = None
max_x *= 2
moves = "".join(data[max_y + 1:])

for y, line in enumerate(data):
    if line == "":
        break
    for x, c in enumerate(line):
        if c == "#":
            walls.add((2 * x, y))
            walls.add((2 * x + 1, y))
        elif c == "O":
            boxes.add((2 * x, y))
        elif c == "@":
            pos = (2 * x, y)
        elif c == ".":
            pass
        else:
            assert False, c


def printB():
    global max_x, max_y, boxes, walls, pos
    for y in range(max_y):
        for x in range(max_x):
            if (x, y) in boxes:
                print("[", end="")
            elif (x, y) in walls:
                print("#", end="")
            elif (x, y) == pos:
                print("@", end="")
            elif (x - 1, y) in boxes:
                print("]", end="")
            else:
                print(".", end="")
        print()


def box_is_pushableB(box, d):
    global dirs, walls, boxes
    assert d in "^v"
    target1, target2 = (box[0], box[1] + dirs[d][1]), (box[0] + 1, box[1] + dirs[d][1])
    if target1 in walls or target2 in walls:
        return False
    else:
        p = True
        if target1 in boxes:
            if not box_is_pushableB(target1, d):
                p = False
        if target2 in boxes:
            if not box_is_pushableB(target2, d):
                p = False
        if (target1[0] - 1, target1[1]) in boxes:
            if not box_is_pushableB((target1[0] - 1, target1[1]), d):
                p = False
        return p


def push_boxB(box, d):
    global dirs, boxes
    assert d in "^v"
    boxes.remove(box)
    if (box[0], box[1] + dirs[d][1]) in boxes:
        push_boxB((box[0], box[1] + dirs[d][1]), d)
    if (box[0] - 1, box[1] + dirs[d][1]) in boxes:
        push_boxB((box[0] - 1, box[1] + dirs[d][1]), d)
    if (box[0] + 1, box[1] + dirs[d][1]) in boxes:
        push_boxB((box[0] + 1, box[1] + dirs[d][1]), d)
    boxes.add((box[0], box[1] + dirs[d][1]))


def moveableB(d):
    global dirs, walls, boxes, pos
    temp_pos = pos
    offset = dirs[d]
    if d == ">":
        while (temp_pos[0] + 1, temp_pos[1]) not in walls:
            if (temp_pos[0] + 1, temp_pos[1]) not in boxes:
                return True
            temp_pos = (temp_pos[0] + 2, temp_pos[1])
    if d == "<":
        while (temp_pos[0] - 1, temp_pos[1]) not in walls:
            if (temp_pos[0] - 2, temp_pos[1]) not in boxes:
                return True
            temp_pos = (temp_pos[0] - 2, temp_pos[1])
        return False
    if d in "^v":
        target = (pos[0], pos[1] + offset[1])
        if target in walls:
            return False
        elif target not in boxes and (target[0] - 1, target[1]) not in boxes:
            return True
        elif target in boxes:
            return box_is_pushableB(target, d)
        elif (target[0] - 1, target[1]) in boxes:
            return box_is_pushableB((target[0] - 1, target[1]), d)
        else:
            assert False, "Missed something"


def moveB(d):
    global dirs, boxes, pos
    offset = dirs[d]
    if d in "^v":
        target = (pos[0], pos[1] + offset[1])
        pos = target
        if target in boxes:
            push_boxB(target, d)
        elif (target[0] - 1, target[1]) in boxes:
            push_boxB((target[0] - 1, target[1]), d)
        else:
            assert "Missed something"
    elif d == "<":
        temp_pos = pos
        pos = (pos[0] - 1, pos[1])
        while (temp_pos[0] - 2, temp_pos[1]) in boxes:
            boxes.remove((temp_pos[0] - 2, temp_pos[1]))
            boxes.add((temp_pos[0] - 3, temp_pos[1]))
            temp_pos = (temp_pos[0] - 2, temp_pos[1])
    elif d == ">":
        temp_pos = pos
        pos = (pos[0] + 1, pos[1])
        while (temp_pos[0] + 1, temp_pos[1]) in boxes:
            boxes.remove((temp_pos[0] + 1, temp_pos[1]))
            boxes.add((temp_pos[0] + 2, temp_pos[1]))
            temp_pos = (temp_pos[0] + 2, temp_pos[1])
    else:
        assert False


n_moves = 0
while moves:
    if moveableB(moves[0]):
        moveB(moves[0])
    moves = moves[1:]
    n_moves += 1

mark("15B", sum(b[0] + 100 * b[1] for b in boxes), 1452348)

data = read("16", 2024, raw=True, strip=True)
walls = set()
start, end = None, None
d = ">"
for y, line in enumerate(data):
    for x, c in enumerate(line):
        if c == "S":
            start = (x, y)
        elif c == "E":
            end = (x, y)
        elif c == "#":
            walls.add((x, y))
        elif c == ".":
            pass
        else:
            assert False, "Unknown char"

min_scores = {}

def task16A(start_pos, start_d):
    global walls, dirs, end, min_scores

    states = [[start_pos, start_d, 0]]

    while states:
        pos, d, score = states[0]
        states = states[1:]

        state = str(pos) + d

        if state in min_scores and min_scores[state] <= score:
            continue

        min_scores[state] = score
        turns = [d, turn[d], turn[turn[turn[d]]]]

        for new_d in turns:
            offset = dirs[new_d]
            target = (pos[0] + offset[0], pos[1] + offset[1])
            if target not in walls:
                states += [[target, new_d, score + 1 if offset == dirs[d] else score + 1001]]

    return min(10 ** 9 if str(end) + end_dir not in min_scores
               else min_scores[str(end) + end_dir] for end_dir in dirs.keys())


mark("16A", task16A(start, d), 108504)

def task16B():
    global start, end, min_scores
    end_dir = None
    end_score = None
    for d in dirs.keys():
        entry = str(end) + d
        if entry in min_scores:
            if not end_score or min_scores[entry] < end_score:
                end_score = min_scores[entry]
                end_dir = d

    states = [[end, end_dir, end_score]]

    visited_states = set()
    visited_tiles = {end}

    while states:
        pos, d, score = states[0]
        states = states[1:]

        state = str(pos) + d
        if state in visited_states:
            continue
        visited_states.add(state)

        turns = [d, turn[d], turn[turn[turn[d]]]]

        for prev_d in turns:
            offset = dirs[d]
            target = (pos[0] - offset[0], pos[1] - offset[1])
            if (target not in walls and str(target) + prev_d in min_scores and
                    min_scores[str(target) + prev_d] == (score - 1 if d == prev_d else score - 1001)):
                visited_tiles.add(target)
                states += [[target, prev_d, score - 1 if d == prev_d else score - 1001]]

    return len(visited_tiles)

mark("16B", task16B(), 538)

data = read("17", 2024, raw=True, strip=True)
A = int(data[0].split(" ")[2])
B = int(data[1].split(" ")[2])
C = int(data[2].split(" ")[2])
program = [int(d) for d in data[4].split(" ")[1].split(",")]

def combo(d):
    global A, B, C
    if d <= 3:
        return d
    elif d == 4:
        return A
    elif d == 5:
        return B
    elif d == 6:
        return C
    else:
        assert False, "Reserved value"

def task17A():
    global A, B, C, program
    done = False
    instruction_pointer = 0
    output = None

    while not done:
        operation = program[instruction_pointer]
        operand = program[instruction_pointer + 1]

        if operation == 0:
            A = A // (2 ** combo(operand))
        elif operation == 1:
            B = B ^ operand
        elif operation == 2:
            B = combo(operand) % 8
        elif operation == 3 and A != 0:
            instruction_pointer = operand - 2
        elif operation == 4:
            B = B ^ C
        elif operation == 5:
            s = str(combo(operand) % 8)
            if output:
                output += "," + s
            else:
                output = s
        elif operation == 6:
            B = A // (2 ** combo(operand))
        elif operation == 7:
            C = A // (2 ** combo(operand))

        instruction_pointer += 2

        if instruction_pointer >= len(program):
            break

    return output

mark("17A", task17A(), "5,1,4,0,5,1,0,2,6")

def task17B():
    global A, B, C

    numbers_to_match = 1
    A_cand = 0
    while numbers_to_match <= len(program):
        A_cand *= 8
        target = program[len(program) - numbers_to_match:]
        n = 0
        while True:
            A, B, C = A_cand + n, 0, 0
            output = [int(d) for d in task17A().split(",")]
            if output == target:
                A_cand += n
                break
            n += 1
        numbers_to_match += 1

    return A_cand

mark("17B", task17B(), 202322936867370)

data = read("18", 2024, raw=True, strip=True)

size = int(data[0])  # To allow for easier test data
walls_A = int(data[1])
data = data[2:]
start, end = (0, 0), (size - 1, size - 1)
walls = set()

for line in data[:walls_A]:
    x, y = [int(d) for d in line.split(",")]
    walls.add((x, y))

for n in range(size):
    walls.add((-1, n))
    walls.add((n, -1))
    walls.add((size, n))
    walls.add((n, size))

min_scores = {}
best_path = None

def task18A():
    global walls, dirs, start, end, min_scores, best_path

    states = [[start, 0, [start]]]
    min_scores[start] = 0

    while states:
        pos, score, previous_tiles = states[0]
        states = states[1:]

        state = str(pos)

        if state in min_scores and min_scores[state] <= score:
            continue

        if pos == end:
            best_path = copy.deepcopy(previous_tiles)

        min_scores[state] = score

        for d in dirs:
            offset = dirs[d]
            target = (pos[0] + offset[0], pos[1] + offset[1])
            if target not in walls:
                previous_tiles += [str(target)]
                states += [[target, score + 1, previous_tiles]]
                previous_tiles = previous_tiles[:-1]

    return -1 if str(end) not in min_scores else min_scores[str(end)]

mark("18A", task18A(), 356)

data = data[walls_A:]

def task18B():
    global walls, size, data, min_scores, best_path
    while data:
        min_scores = {}
        x, y = [int(d) for d in data[0].split(",")]
        walls.add((x, y))
        data = data[1:]
        if str((x, y)) not in best_path:
            continue
        if task18A() == -1:
            return str(x) + "," + str(y)
    assert False, "No blocking byte found"

# mark("18B", task18B(), "22,33")
mark("18B", None, "22,33", skip_and_add_time=0.61)

data = read("19", 2024, raw=True, strip=True)
towels = set(data[0].split(", "))
data = data[2:]

def towelable(s, towels):
    if s == "":
        return True
    return any(s.startswith(t) and towelable(s[len(t):], towels) for t in towels)

@functools.cache
def towel_ways(s):
    global towels
    if s == "":
        return 1
    return sum(towel_ways(s[len(t):]) if s.startswith(t) else 0 for t in towels)

mark("19A", sum(1 if towelable(s, towels) else 0 for s in data), 374)
# mark("19B", sum(towel_ways(str(s)) for s in data), 1100663950563322)
mark("19B", None, 1100663950563322, skip_and_add_time=0.51)


data = read("20", 2024, raw=True, strip=True)

walls = set()
for y, line in enumerate(data):
    for x, c in enumerate(line):
        if c == "S":
            start = (x, y)
        if c == "E":
            end = (x, y)
        if c == "#":
            walls.add((x, y))

min_scores = {}
best_path = None
task18A()

def task20(cheat_size):
    global min_scores, best_path
    cheats = {}
    ans = 0
    offsets = []
    for i in range(-cheat_size, cheat_size + 1):
        width = cheat_size - abs(i)
        for j in range(-width, width + 1):
            offsets += [[i, j]]
    for tile in best_path:
        for o in offsets:
            t = [int(d) for d in str(tile)[1:-1].split(",")]
            cheat_tile = str((t[0] + o[0], t[1] + o[1]))
            taxicab_cheat_distance = abs(o[0]) + abs(o[1])
            if cheat_tile in min_scores and min_scores[cheat_tile] > min_scores[tile] + taxicab_cheat_distance:
                cheat_value = min_scores[cheat_tile] - min_scores[tile] - taxicab_cheat_distance
                if cheat_value not in cheats:
                    cheats[cheat_value] = 1
                else:
                    cheats[cheat_value] += 1
                if cheat_value >= 100:
                    ans += 1
    return ans

mark("20A", task20(2), 1384)
# mark("20B", task20(20), 1008542)
mark("20B", None, 1008542, skip_and_add_time=5.54)

data = read("21", 2024, raw=True, strip=True)
#data = [data[0]]

neighbors = {
    "A": "03",
    "0": "A2",
    "1": "24",
    "2": "0135",
    "3": "A26",
    "4": "157",
    "5": "2468",
    "6": "359",
    "7": "48",
    "8": "579",
    "9": "68",
    "a": "^>",
    ">": "av",
    "^": "av",
    "v": ">^<",
    "<": "v",
}

possible_moves = {
        "A<": "0",
        "A^": "3",
        "0^": "2",
        "0>": "A",
        "1>": "2",
        "1^": "4",
        "2v": "0",
        "2<": "1",
        "2>": "3",
        "2^": "5",
        "3v": "A",
        "3<": "2",
        "3^": "6",
        "4v": "1",
        "4>": "5",
        "4^": "7",
        "5v": "2",
        "5<": "4",
        "5>": "6",
        "5^": "8",
        "6v": "3",
        "6<": "5",
        "6^": "9",
        "7v": "4",
        "7>": "8",
        "8v": "5",
        "8<": "7",
        "8>": "9",
        "9v": "6",
        "9<": "8",

        "av": ">",
        "a<": "^",
        ">^": "a",
        "><": "v",
        "^>": "a",
        "^v": "v",
        "v>": ">",
        "v^": "^",
        "v<": "<",
        "<>": "v",
    }


@functools.cache
def task21A(start, end):
    global min_scores

    states = [[start, 0]]
    robot_levels = len(start)

    while states:
        pos, score = states[0]
        states = states[1:]

        if pos in min_scores and min_scores[pos] <= score:
            continue

        min_scores[pos] = score

        for neighbor in neighbors[pos[robot_levels - 1]]:
            states.append([pos[:robot_levels - 1] + neighbor, score + 1])

        for i in range(robot_levels - 1):
            move_slice = pos[(robot_levels - 2 - i):robot_levels - i]
            if (all(c == "a" for c in pos[robot_levels - i:]) and
                    pos[robot_levels - 1 - i] != "a" and move_slice in possible_moves):
                states.append([pos[:robot_levels - 2 - i] +
                               possible_moves[move_slice] +
                               pos[robot_levels - 1 - i:], score + 1])

    return min_scores[end]

for a in range(2, 10):
    min_scores = {}
    data_sum = 0
    for line in data:
        line_to_number = int(line[:-1])
        line = "A" + line
        line_sum = 0
        while len(line) >= 2:
            min_scores = {}
            line_sum += task21A(line[0] + "a"*(a - 1),
                                line[1] + "a"*(a - 1)) + 1
            line = line[1:]
        data_sum += line_sum * line_to_number
    print(a, data_sum)


mark("21A", data_sum, 157892)
