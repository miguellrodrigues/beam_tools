#!/usr/bin/env python3
"""
Tkinter GUI for interactive beam analysis visualisation.

Displays shear force, bending moment, slope, and deflection plots alongside
rendered LaTeX equations in scrollable windows.
"""

import tkinter as tk

import numpy as np
from matplotlib import rc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beam_tools import SupportType


def display_beam_analysis_with_equations(
    equations, x, V, M, y, theta, scaling_factor, supports
):
    """
    Show beam analysis plots and LaTeX equations in separate Tk windows.

    Parameters
    ----------
    equations : list[str]
        LaTeX equation strings.
    x : numpy.ndarray
        Position array along the beam.
    V, M : numpy.ndarray
        Shear force and bending moment arrays.
    y, theta : numpy.ndarray
        Deflection (mm) and slope (degrees) arrays.
    scaling_factor : float
        Visual scaling multiplier for the x-axis.
    supports : list[Support]
        Beam supports (used to draw markers).
    """

    def render_equations_latex():
        rc("text", usetex=True)
        win = tk.Toplevel()
        win.title("Equations (LaTeX)")
        win.geometry("600x800")

        canvas = tk.Canvas(win)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(win, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        frame = tk.Frame(canvas)
        frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for eq in equations:
            fig = Figure(figsize=(6, 1))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"${eq}$", fontsize=20, ha="center", va="center")
            ax.axis("off")
            widget = FigureCanvasTkAgg(fig, master=frame)
            widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            widget.draw()

    def plot_beam_analysis():
        win = tk.Toplevel()
        win.title("Beam Analysis Plots")
        win.geometry("1200x900")

        fig = Figure(figsize=(12, 10))

        ax1 = fig.add_subplot(311)
        ax1.set_title("Slope")
        ax1.plot(scaling_factor * x, theta, "orange", lw=3)
        ax1.set_ylabel("Degrees")
        ax1.grid()

        ax2 = fig.add_subplot(312)
        ax2.set_title("Deflection")
        ax2.plot([0, 10], [0, 0], "k", lw=3, label="Beam")
        ax2.plot(scaling_factor * x, y, "-r", lw=3)
        ax2.set_ylabel("mm")
        ax2.grid()

        for support in supports:
            loc = support.location * scaling_factor
            if support.support_type == SupportType.HINGED:
                ax2.plot(loc, 0, "go", ms=10, label="Hinged")
            elif support.support_type == SupportType.ROLLER:
                ax2.plot(loc, 0, "bo", ms=10, label="Roller")
            elif support.support_type == SupportType.FIXED:
                ax2.plot(loc, 0, "ko", ms=10, label="Fixed")

        ax3 = fig.add_subplot(313)
        ax3.set_title("Shear Force & Bending Moment")
        ax3.plot(x, V, "b-", label="V(x)", lw=3)
        ax3.plot(x, M, "g-", label="M(x)", lw=3)
        ax3.fill_between(x, V, where=(V > 0), color="skyblue", alpha=0.4)
        ax3.fill_between(x, M, where=(M > 0), color="lightgreen", alpha=0.4)
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("Force / Moment")
        ax3.legend()
        ax3.grid()

        widget = FigureCanvasTkAgg(fig, master=win)
        widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        widget.draw()

    root = tk.Tk()
    root.title("Beam Analysis")
    tk.Button(root, text="Show Plots", command=plot_beam_analysis).pack(pady=10)
    tk.Button(root, text="Show Equations", command=render_equations_latex).pack(
        pady=10
    )
    root.mainloop()
