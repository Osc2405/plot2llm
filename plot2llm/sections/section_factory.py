from plot2llm.sections.metadata_section import build_metadata_section
from plot2llm.sections.axes_section import build_axes_section
from plot2llm.sections.layout_section import build_layout_section
from plot2llm.sections.data_summary_section import build_data_summary_section
from plot2llm.sections.statistical_insights_section import build_statistical_insights_section
from plot2llm.sections.pattern_analysis_section import build_pattern_analysis_section
from plot2llm.sections.visual_elements_section import build_visual_elements_section
from plot2llm.sections.domain_context_section import build_domain_context_section
from plot2llm.sections.llm_description_section import build_llm_description_section
from plot2llm.sections.llm_context_section import build_llm_context_section

SECTION_BUILDERS = {
    "metadata": build_metadata_section,
    "axes": build_axes_section,
    "layout": build_layout_section,
    "data_summary": build_data_summary_section,
    "statistical_insights": build_statistical_insights_section,
    "pattern_analysis": build_pattern_analysis_section,
    "visual_elements": build_visual_elements_section,
    "domain_context": build_domain_context_section,
    "llm_description": build_llm_description_section,
    "llm_context": build_llm_context_section,
}

def get_section_builder(section_name: str):
    """
    Devuelve la función constructora de la sección correspondiente.
    """
    return SECTION_BUILDERS.get(section_name) 