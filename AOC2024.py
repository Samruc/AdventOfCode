import functools
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
mark("2B", sum([1 if any(safety(line[:i] + line[i+1:]) for i in range(len(line))) else 0 for line in data]), 528)

data = read("3", 2024, raw=True, join=True)
do_dont = " ".join(s.split("don't()")[0] for s in data.split("do()"))
mark("3A", (sum([int(a) * int(b) for a, b in re.findall(r"mul\((\d{1,3}),(\d{1,3})\)", data)])), 169021493)
mark("3B", (sum([int(a) * int(b) for a, b in re.findall(r"mul\((\d{1,3}),(\d{1,3})\)", do_dont)])), 111762583)

data = read("4", 2024, raw=True, strip=True)
n = len(data)
cols = ["".join(line[i] for line in data) for i in range(n)]
dias = ["".join(data[i-j][j] for j in range(max(i-n+1, 0), min(i, n-1) + 1)) for i in range(2*n - 1)]
dias += ["".join(data[j][n-1-i+j] for j in range(max(i-n+1, 0), min(i, n-1) + 1)) for i in range(2*n - 1)]
mark("4A", sum(sum(1 if line[i:i+4] in ["XMAS", "SAMX"] else 0
                   for i in range(len(line))) for line in data+cols+dias), 2401)
mark("4B", sum(sum(1 if data[i][j] == "A" and
                    all([data[i+1][j+1] in ["M", "S"],
                         data[i-1][j+1] in ["M", "S"],
                         data[i+1][j-1] in ["M", "S"],
                         data[i-1][j-1] in ["M", "S"]]) and
                         data[i+1][j+1] != data[i-1][j-1] and
                         data[i-1][j+1] != data[i+1][j-1]
                   else 0 for j in range(1, n-1)) for i in range(1, n-1)), 1822)

data = read("5", 2024, raw=True, strip=True)
rules = set(data[:data.index("")])
lists = [l.split(",") for l in data[data.index("")+1:]]
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
                    new_x, new_y = 2*a[0] - b[0], 2*a[1] - b[1]
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
                     d[space_searcher + 1:digit_to_move_index-1] +
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
    neighbor_candidates = [[x, y+1], [x+1, y], [x, y-1], [x-1, y]]
    candidates = list(filter(lambda p : 0 <= p[0] < len(data) and 0 <= p[1] < len(data), neighbor_candidates))
    step_candidates = list(filter(lambda p : int(data[p[1]][p[0]]) == current_value + 1, candidates))
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
            (+1, -1),]
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
