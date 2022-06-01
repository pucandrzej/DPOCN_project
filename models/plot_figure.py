import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_figure(qa, qc, k, c0, ax, graph_type="random", add_label=1):
    if graph_type == "random":
        if c0 == 0.5:
            c0 = "05"
        elif c0 == 1:
            c0 = "1"
        file_name = f"c{k}_{qa}_{qc}_{c0}.txt"
        data = pd.read_csv(file_name, sep=" ", names=["p", "c"]).sort_values(by="p")

        p = np.array(sorted(list(set(data["p"].to_numpy()))))
        data = data.groupby(["p"]).mean()

        mean_values = data.to_numpy()
        mean_values = np.reshape(mean_values, (1, len(p)))

        if c0 == "1":
            marker = "v"
            color = "r"
        elif c0 == "05":
            marker = "^"
            color = "y"
        if add_label:
            ax.scatter(p, mean_values, marker=marker, color=color, label=f"E--R, $c_0={c0}$")
        else:
            ax.scatter(p, mean_values, marker=marker, color=color)

    elif graph_type == "ws":
        if c0 == 0.5:
            c0 = "05"
        elif c0 == 1:
            c0 = "1"

        file_name = f"c{k}_{qa}_{qc}_{c0}_ws.txt"
        data = pd.read_csv(file_name, sep=" ", names=["p", "c"]).sort_values(by="p")

        p = np.array(sorted(list(set(data["p"].to_numpy()))))
        data = data.groupby(["p"]).mean()

        mean_values = data.to_numpy()
        mean_values = np.reshape(mean_values, (1, len(p)))

        if c0 == "1":
            marker = "1"
            color = "b"
        elif c0 == "05":
            marker = "^"
            color = "g"
        if add_label:
            ax.scatter(p, mean_values, marker=marker, color=color, label=f"W--S, $c_0={c0}$")
        else:
            ax.scatter(p, mean_values, marker=marker, color=color)
