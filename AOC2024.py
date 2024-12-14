import re
from collections import Counter
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

mark("6A", guard_path(data, start_direction, start_location, add_obstacle=False)[0], 5269)
mark("6B", guard_path(data, start_direction, start_location, add_obstacle=True)[1], 1957)
