def build_visual_elements_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección visual_elements para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    
    # Buscar características visuales en el formato moderno
    modern_visual_elements = semantic_analysis.get("visual_elements", {})
    
    # Si ya tenemos visual_elements del análisis moderno, usarlos
    if modern_visual_elements:
        return modern_visual_elements
    
    # Fallback: construir desde los ejes individuales
    visual_elements = {
        "lines": [],
        "axes_styling": [],
        "primary_colors": [],
        "accessibility_score": None
    }
    
    # Extraer líneas de los ejes
    for ax in axes:
        line_elements = []
        # Buscar en plot_types (formato moderno)
        plot_types = ax.get("plot_types", [])
        for pt in plot_types:
            if pt.get("type") == "line":
                # Buscar líneas en el eje
                lines_data = ax.get("lines", [])
                for line in lines_data:
                    if line.get("label") and line.get("label") != "_nolegend_":
                        line_elements.append(line["label"])
                break
        visual_elements["lines"].append(line_elements)
    
    # Extraer estilos de ejes
    for ax in axes:
        styling = {
            "has_grid": ax.get("has_grid", False),
            "spine_visibility": ax.get("spine_visibility"),
            "tick_density": ax.get("tick_density"),
        }
        visual_elements["axes_styling"].append(styling)
    
    # Buscar colores en diferentes fuentes
    colors = semantic_analysis.get("colors", [])
    if colors:
        visual_elements["primary_colors"] = [c.get("hex") for c in colors if c.get("hex")]
    
    # Buscar en visual_info (formato legacy)
    visual_info = semantic_analysis.get("visual_info", {})
    if not visual_elements["primary_colors"] and "colors" in visual_info:
        visual_elements["primary_colors"] = [c.get("hex") for c in visual_info["colors"] if c.get("hex")]
    
    # Buscar accessibility_score
    if "accessibility_score" in visual_info:
        visual_elements["accessibility_score"] = visual_info["accessibility_score"]
    
    return visual_elements 