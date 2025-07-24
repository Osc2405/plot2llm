import matplotlib
from matplotlib.axes import Axes

def identify_plot_type(ax: Axes) -> str:
    """
    Identifica el tipo principal de gráfico en el eje dado.
    Soporta: 'line', 'bar', 'scatter', 'unknown'
    """
    line_count = len(ax.lines)
    bar_count = sum(
        isinstance(p, matplotlib.patches.Rectangle) and p.get_y() == 0 and p.get_height() > 0
        for p in ax.patches
    )
    scatter_count = sum(
        isinstance(c, matplotlib.collections.PathCollection)
        for c in ax.collections
    )
    if bar_count > 0 and line_count == 0:
        return "bar"
    elif line_count > 0 and bar_count == 0:
        return "line"
    elif scatter_count > 0:
        return "scatter"
    else:
        return "unknown"

def extract_axes_section(ax: Axes) -> dict:
    """
    Extrae la sección 'axes' según el tipo de gráfico.
    """
    plot_type = identify_plot_type(ax)
    axes_section = {
        "plot_type": plot_type,
        "x_label": ax.get_xlabel(),
        "y_label": ax.get_ylabel(),
        "x_lim": list(ax.get_xlim()),
        "y_lim": list(ax.get_ylim()),
    }
    if plot_type == "bar":
        # Extraer etiquetas de categorías y valores de las barras
        categories = [tick.get_text() for tick in ax.get_xticklabels()]
        axes_section["categories"] = categories
    elif plot_type == "line":
        # Extraer información de las líneas
        axes_section["lines"] = [
            {
                "label": line.get_label(),
                "xdata": list(line.get_xdata()),
                "ydata": list(line.get_ydata())
            }
            for line in ax.lines
        ]
    # Puedes agregar lógica para otros tipos aquí
    return axes_section 