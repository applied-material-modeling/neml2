import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1]).filter(
    ["test_case", "benchmark", "nbatch", "chunk_size", "mean", "low_mean", "high_mean"])

test_cases = set(data["test_case"])

for test_case in test_cases:
    print("test case: " + test_case)
    benchmarks = set(data.loc[(data["test_case"] == test_case)]["benchmark"])
    chunk_sizes = sorted(
        set(data.loc[(data["test_case"] == test_case)]["chunk_size"]))
    for benchmark in benchmarks:
        print("     benchmark: " + benchmark)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(test_case + "\n" + benchmark)
        for chunk_size in chunk_sizes:
            select = data.loc[(data["test_case"] == test_case) &
                              (data["chunk_size"] == chunk_size) &
                              (data["benchmark"] == benchmark)]
            ax.plot(select["nbatch"], select["mean"], "o-",
                    label="chunk size="+str(chunk_size))
            # ax.fill_between(select["nbatch"], select["low_mean"],
            #                 select["high_mean"], color='r', alpha=.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of batches")
        ax.set_ylabel("Time per iteration [us]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        fig.tight_layout()
        path = Path(sys.argv[2])/test_case
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path/(benchmark+".png"), dpi=300)
        plt.close()