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
location = None
direction = None
for y, l in enumerate(data):
    for x, c in enumerate(l):
        if c in dirs:
            location = (x, y)
            direction = c
visited = {location}
visited_with_dir = {str(location) + direction}
loop_blocks = set()
while True:
    try:
        x = location[0] + dirs[direction][0]
        y = location[1] + dirs[direction][1]
        if data[y][x] in [".", "^"]:
            if data[y][x] == ".":
                if str(location) + turn[direction] in visited_with_dir:
                    loop_blocks.add((x, y))
            location = (x, y)
            visited.add(location)
            visited_with_dir.add(str(location) + direction)
        else:
            direction = turn[direction]
    except IndexError:
        break

mark("6A", len(visited), 5269)
print(len(loop_blocks))  # Not correct, loop test is too simple. Undershoots on test data as well
