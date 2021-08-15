from typing import List

import matplotlib.pyplot as plt


def simple_plot(
        data: List,
        save_path: str,
        title: str,
        xlabel: str,
        ylabel: str,
        fig_size=None,
        x_axis_spacing: float = 0.2,
        x_axis_are_strings: bool = False,
        x_axis_delta: float = 1,
        color: str = "blue"
) -> None:
    """Create a plot of the given data and save it the given path"""
    if fig_size is None:
        fig_size = [8.0, 6.0]
    plt.rcParams["figure.figsize"] = fig_size  # set the fig size

    fig = plt.figure()
    fig.subplots_adjust(bottom=x_axis_spacing)

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
    title = "Score per epoch"
    fig_size = [8.0, 6.0] if len(scores) <= 30 else [9.0, 6.5]

    simple_plot(scores,
                save_path=title.lower().replace(" ", "_"),
                title=title,
                xlabel="epoch number",
                ylabel="score",
                fig_size=fig_size,
                x_axis_are_strings=False,
                x_axis_delta=1,
                color="blue"
                )
