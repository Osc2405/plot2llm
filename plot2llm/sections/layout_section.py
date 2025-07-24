def build_layout_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección layout para el output semántico.
    """
    seaborn_info = semantic_analysis.get("seaborn_info", {})
    detailed_info = semantic_analysis.get("detailed_info", {})
    axes = semantic_analysis.get("axes", [])
    if "grid_shape" in seaborn_info:
        return {
            "shape": seaborn_info.get("grid_shape"),
            "size": seaborn_info.get("grid_size"),
        }
    elif "grid_layout" in detailed_info:
        return detailed_info["grid_layout"]
    elif axes:
        return {"shape": (1, len(axes)), "size": len(axes), "nrows": 1, "ncols": len(axes)}
    return None 