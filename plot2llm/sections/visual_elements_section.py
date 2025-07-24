def build_visual_elements_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección visual_elements para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    visual_elements = {
        "lines": [],
        "axes_styling": [],
        "primary_colors": [],
        "accessibility_score": None
    }
    for ax in axes:
        line_elements = []
        if any(pt.get("type") == "line" for pt in ax.get("plot_types", [])):
            for cp in ax.get("curve_points", []):
                if cp.get("label") and cp.get("label") != "_nolegend_":
                    line_elements.append(cp["label"])
        visual_elements["lines"].append(line_elements)
    for ax in axes:
        styling = {
            "has_grid": ax.get("has_grid", False),
            "spine_visibility": ax.get("spine_visibility"),
            "tick_density": ax.get("tick_density"),
        }
        visual_elements["axes_styling"].append(styling)
    visual_info = semantic_analysis.get("visual_info", {})
    if "colors" in visual_info:
        visual_elements["primary_colors"] = [c.get("hex") for c in visual_info["colors"] if c.get("hex")]
    if "accessibility_score" in visual_info:
        visual_elements["accessibility_score"] = visual_info["accessibility_score"]
    return visual_elements 