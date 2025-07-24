def build_metadata_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección de metadata para el output semántico.
    """
    metadata = semantic_analysis.get("metadata", {})
    if not metadata:
        metadata = {
            "figure_type": semantic_analysis.get("figure_type"),
            "detail_level": semantic_analysis.get("detail_level", "medium"),
            "analysis_timestamp": semantic_analysis.get("analysis_timestamp"),
            "analyzer_version": semantic_analysis.get("analyzer_version", "unknown"),
        }
    return {
        "figure_type": metadata.get("figure_type", "unknown"),
        "detail_level": metadata.get("detail_level", "medium"),
        "analysis_timestamp": metadata.get("analysis_timestamp", None),
        "analyzer_version": metadata.get("analyzer_version", "unknown"),
    } 