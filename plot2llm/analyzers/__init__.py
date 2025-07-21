"""
Analyzers package for different figure types.
"""

from .base_analyzer import BaseAnalyzer
from .matplotlib_analyzer import MatplotlibAnalyzer
from .seaborn_analyzer import SeabornAnalyzer


# Main analyzer class that coordinates all analyzers
class FigureAnalyzer:
    """
    Main analyzer class that coordinates analysis of different figure types.
    """

    def __init__(self):
        """Initialize the FigureAnalyzer with specific analyzers."""
        self.matplotlib_analyzer = MatplotlibAnalyzer()
        self.seaborn_analyzer = SeabornAnalyzer()

        import logging

        logger = logging.getLogger(__name__)
        logger.info("FigureAnalyzer initialized")

    def analyze(
        self,
        figure,
        figure_type,
        detail_level="medium",
        include_data=True,
        include_colors=True,
        include_statistics=True,
    ) -> dict:
        """
        Analyze a figure and extract relevant information.

        Args:
            figure: The figure object to analyze
            figure_type: Type of figure ("matplotlib", "plotly", "seaborn", etc.)
            detail_level: Level of detail ("low", "medium", "high")
            include_data: Whether to include data analysis
            include_colors: Whether to include color analysis
            include_statistics: Whether to include statistical analysis

        Returns:
            Dictionary containing the analysis results
        """
        import logging

        import pandas as pd

        logger = logging.getLogger(__name__)

        try:
            # Route to appropriate analyzer
            logger.debug(f"FigureAnalyzer: Routing figure_type={figure_type}")

            if figure_type == "matplotlib":
                analyzer = self.matplotlib_analyzer
                logger.debug("Using matplotlib_analyzer")
            elif figure_type == "seaborn":
                analyzer = self.seaborn_analyzer
                logger.debug("Using seaborn_analyzer")
            else:
                # For now, use matplotlib analyzer as fallback
                analyzer = self.matplotlib_analyzer
                logger.debug(
                    f"Unknown figure_type={figure_type}, using matplotlib_analyzer as fallback"
                )

            # Perform analysis
            analysis = analyzer.analyze(
                figure=figure,
                detail_level=detail_level,
                include_data=include_data,
                include_colors=include_colors,
                include_statistics=include_statistics,
            )

            # Add metadata
            analysis["metadata"] = {
                "figure_type": figure_type,
                "detail_level": detail_level,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "analyzer_version": "0.1.0",
            }

            logger.info(f"Analysis completed for {figure_type} figure")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing figure: {str(e)}")
            raise

    def get_available_analyzers(self) -> list[str]:
        """Get list of available analyzers."""
        return ["matplotlib", "seaborn"]


__all__ = ["BaseAnalyzer", "MatplotlibAnalyzer", "SeabornAnalyzer", "FigureAnalyzer"]
