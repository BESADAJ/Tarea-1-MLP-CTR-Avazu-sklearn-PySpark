"""Estilo común de las figuras del libro y formato de números en español."""
import matplotlib.pyplot as plt

C = dict(azul="#2a78d6", naranja="#eb6834", aqua="#1baf7a", amarillo="#eda100",
         magenta="#e87ba4", verde="#008300", violeta="#6250d6", rojo="#e34948", gris="#8a8a84")
SERIE = {"scikit-learn": C["azul"], "PySpark": C["naranja"]}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (8, 3.8), "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ad",
        "axes.grid": True, "grid.color": "#e7e6e0", "grid.linewidth": .7, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
        "axes.formatter.use_locale": False})


def es(x, d=3):
    """Número con coma decimal."""
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def miles(x):
    return f"{int(x):,}".replace(",", ".")
