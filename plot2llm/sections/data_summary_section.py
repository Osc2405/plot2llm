def build_data_summary_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección data_summary para el output semántico.
    """
    # Use the already calculated data_summary if available
    if "data_summary" in semantic_analysis:
        return semantic_analysis["data_summary"]

    # Get data from axes (from specific analyzers)
    axes = semantic_analysis.get("axes", [])
    if axes:
        # Get total_points from the first axis stats
        first_axis = axes[0]
        stats = first_axis.get("stats", {})
        data_quality = stats.get("data_quality", {})
        total_points = data_quality.get("total_points", 0)

        # Get ranges from the first axis
        x_range = first_axis.get("x_range", [None, None])
        y_range = first_axis.get("y_range", [None, None])
        x_type = first_axis.get("x_type")
        y_type = first_axis.get("y_type")

        # Procesar ranges - pueden venir como listas [min, max] o como None
        x_range_dict = None
        y_range_dict = None

        if (
            x_range
            and isinstance(x_range, list)
            and len(x_range) == 2
            and x_range[0] is not None
        ):
            x_range_dict = {"min": x_range[0], "max": x_range[1], "type": x_type}

        if (
            y_range
            and isinstance(y_range, list)
            and len(y_range) == 2
            and y_range[0] is not None
        ):
            y_range_dict = {"min": y_range[0], "max": y_range[1], "type": y_type}

        data_summary = {
            "total_data_points": total_points,
            "data_ranges": {
                "x": x_range_dict,
                "y": y_range_dict,
            },
            "missing_values": {
                "x": data_quality.get("missing_values", 0),
                "y": data_quality.get("missing_values", 0),
            },
            "x_type": x_type,
            "y_type": y_type,
        }
    else:
        # Fallback to legacy method if no axes data
        data_info = semantic_analysis.get("data_info", {})
        x_type = None
        y_type = None
        x_range = [None, None]
        y_range = [None, None]
        data_summary = {
            "total_data_points": data_info.get("data_points", 0),
            "data_ranges": {
                "x": (
                    {"min": x_range[0], "max": x_range[1], "type": x_type}
                    if x_range
                    else None
                ),
                "y": (
                    {"min": y_range[0], "max": y_range[1], "type": y_type}
                    if y_range
                    else None
                ),
            },
            "missing_values": {"x": 0, "y": 0},
            "x_type": x_type,
            "y_type": y_type,
        }

    return data_summary
