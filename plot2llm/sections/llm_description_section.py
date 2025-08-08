from plot2llm.utils import generate_unified_key_insights


def build_llm_description_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección llm_description para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    if not axes:
        return {}
    primary_axis = axes[0]
    pattern = primary_axis.get("pattern", {})
    shape = primary_axis.get("shape", {})
    domain_context = primary_axis.get("domain_context", {})
    stats = primary_axis.get("stats", {})
    pattern_type = pattern.get("pattern_type", "unknown")
    confidence = pattern.get("confidence_score", 0)
    domain = domain_context.get("likely_domain", "")
    purpose = domain_context.get("purpose", "")
    summary_parts = []
    if pattern_type != "unknown" and confidence > 0.7:
        summary_parts.append(f"a {pattern_type} relationship")
    if domain:
        summary_parts.append(f"in the {domain} domain")
    if purpose:
        summary_parts.append(f"used for {purpose}")
    one_sentence_summary = f"This visualization shows {' '.join(summary_parts)}."
    what_parts = []
    if pattern_type != "unknown":
        what_parts.append(f"{pattern_type} pattern")
    if domain:
        what_parts.append(f"in {domain} context")
    what = " ".join(what_parts) if what_parts else "Data visualization"
    x_semantics = primary_axis.get("x_semantics", "")
    when = "Time-series analysis" if x_semantics == "time" else "Point-in-time analysis"
    why_parts = []
    if purpose:
        why_parts.append(purpose)
    if pattern_type != "unknown" and confidence > 0.8:
        why_parts.append(f"showing clear {pattern_type} behavior")
    why = " ".join(why_parts) if why_parts else "Data analysis"
    # Generate unified key insights
    insights_data = {}

    # Pattern insights
    if pattern_type != "unknown" and confidence > 0.7:
        equation = pattern.get("equation_estimate", "")
        if equation:
            insights_data["equation"] = equation
        insights_data["pattern_confidence"] = confidence

    # Correlation insights
    correlations = stats.get("correlations", [])
    if correlations:
        for corr in correlations:
            if isinstance(corr, dict) and abs(corr.get("value", 0)) > 0.7:
                insights_data["correlation_value"] = corr.get("value", 0)
                insights_data["correlation_strength"] = (
                    "strong" if abs(corr.get("value", 0)) > 0.7 else "moderate"
                )
                insights_data["correlation_direction"] = (
                    "positive" if corr.get("value", 0) > 0 else "negative"
                )
                break

    # Shape insights
    monotonicity = shape.get("monotonicity")
    if monotonicity:
        insights_data["trend"] = monotonicity

    # Outlier insights
    outliers = stats.get("outliers", {})
    if isinstance(outliers, list):
        outliers = outliers[0] if outliers and isinstance(outliers[0], dict) else {}
    if outliers.get("detected", False):
        insights_data["outliers_count"] = outliers.get("count", 0)

    key_insights = generate_unified_key_insights(insights_data)
    return {
        "one_sentence_summary": one_sentence_summary,
        "structured_analysis": {
            "what": what,
            "when": when,
            "why": why,
            "how": "Through data visualization and statistical analysis",
        },
        "key_insights": key_insights,
    }
