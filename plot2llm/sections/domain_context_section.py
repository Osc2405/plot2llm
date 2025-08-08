def build_domain_context_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección domain_context para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    domain_context_list = [ax.get("domain_context", {}) for ax in axes]
    return (
        domain_context_list[0]
        if domain_context_list
        else {
            "likely_domain": None,
            "purpose_inference": None,
            "complexity_level": None,
            "mathematical_properties": None,
        }
    )
