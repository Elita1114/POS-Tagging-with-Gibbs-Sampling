from typing import List

import matplotlib.pyplot as plt


def simple_plot(data: List, save_path: str, title: str, xlabel:str, ylabel: str, color: str = "blue") -> None:
    """Create a plot of the given data and save it the given path"""
    plt.figure()
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    x = list(range(len(data)))

    plt.plot(x, data, color=color)

    plt.savefig(save_path)


def plot_scores(scores):
    """Plot the given scores"""
    title = "scores_per_iteration"
    simple_plot(scores, save_path=title, title=title, xlabel="epoch number", ylabel="score")
