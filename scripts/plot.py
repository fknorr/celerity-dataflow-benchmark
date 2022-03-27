from matplotlib import pyplot as plt
import csv
import sys
from collections import defaultdict
import re
import numpy as np
from scipy import stats

data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for fn in sys.argv[1:]:
    branch = None
    if m := re.match('^benchmark-([^-]+)-', fn):
        branch = m.group(1)
    with open(fn, 'r') as f:
        r = csv.reader(f, delimiter=';')
        head = next(r)
        for row in r:
            bench, ranks = row[0], row[1:]
            all_times = [int(ns) * 1e-6 for ra in ranks for ns in ra.split(',')]
            data[branch][bench][len(ranks)].append(np.mean(all_times))

fig, axes = plt.subplots(nrows=1, ncols=max(len(bd) for bd in data.values()), sharey='row')
for branch, bd in data.items():
    for c, (bench, us) in enumerate(bd.items()):
        xs, samples = zip(*us.items())
        ys = [np.mean(s) for s in samples]
        mins, maxs = zip(*(stats.norm.interval(0.95, loc=np.mean(s), scale=np.std(s) / np.sqrt(len(s))) for s in samples))
        ax = axes[c]
        ax.plot(xs, ys, label=branch)
        ax.fill_between(xs, mins, maxs, alpha=0.3)
        ax.set_title(bench)
        ax.legend()
        ax.set_xlabel('nodes')
        ax.set_ylabel('time [ms]')
        ax.set_xscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels(xs)
        ax.minorticks_off()
plt.show()
