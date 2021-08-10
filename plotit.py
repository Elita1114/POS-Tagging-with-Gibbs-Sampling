from typing import List

import matplotlib.pyplot as plt


def simple_plot(
        data: List,
        save_path: str,
        title: str,
        xlabel: str,
        ylabel: str,
        x_axis_are_strings: bool = False,
        x_axis_delta: float = 1,
        color: str = "blue"
) -> None:
    """Create a plot of the given data and save it the given path"""
    plt.figure()
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    x = [i * x_axis_delta for i in range(0, len(data))]

    if x_axis_are_strings:
        x = [str(i) for i in x]

    plt.plot(x, data, color=color)

    plt.savefig(save_path)


def plot_scores(scores):
    """Plot the given scores"""
    title = "Scores per epoch"
    simple_plot(scores,
                save_path=title.lower().replace(" ", "_"),
                title=title,
                xlabel="epoch number",
                ylabel="score",
                x_axis_are_strings=True,
                x_axis_delta=1,
                color="blue"
                )
