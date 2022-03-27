from matplotlib import pyplot as plt
import csv
import sys
from collections import defaultdict
import re
import numpy as np
from scipy import stats

PLOTS = [
    ('master', 'MPI_Barrier', 'MPI_Barrier'),
    ('master', 'slow_full_sync', 'sync (ad-hoc)'),
    ('epochs', 'slow_full_sync', 'sync (epoch-based)'),
    ('master', 'slow_full_sync after data transfers', 'sync (ad-hoc) after data transfers'),
    ('epochs', 'slow_full_sync after data transfers', 'sync (epoch-based) after data transfers'),
]

data = defaultdict(lambda: defaultdict(list))
for fn in sys.argv[1:]:
    branch = None
    if m := re.match('^benchmark-([^-]+)-', fn):
        branch = m.group(1)
    with open(fn, 'r') as f:
        r = csv.reader(f, delimiter=';')
        head = next(r)
        for row in r:
            bench, ranks = row[0], row[1:]
            all_times = [int(ns) * 1e-9 for ra in ranks for ns in ra.split(',')]
            data[branch, bench][len(ranks)].append(np.mean(all_times))

fig, ax = plt.subplots(sharey='all')
for (branch, bench, label) in PLOTS:
    us = data[branch, bench]
    xs, samples = zip(*us.items())
    ys = [np.mean(s) for s in samples]
    mins, maxs = zip(*(stats.norm.interval(0.99, loc=np.mean(s), scale=np.std(s) / np.sqrt(len(s))) for s in samples))
    ax.plot(xs, ys, label=label)
    ax.fill_between(xs, mins, maxs, alpha=0.3)
ax.legend()
ax.set_xlabel('nodes')
ax.set_ylabel('time [s]')
ax.set_xscale('log')
ax.set_xticks(xs)
ax.set_xticklabels(xs)
ax.minorticks_off()
plt.show()
