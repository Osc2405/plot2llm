from plot2llm.formatters import SemanticFormatter
from plot2llm.analyzers import FigureAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Utilidad para imprimir secciones de forma clara
def print_section(title, section):
    print(f"\n--- {title} ---")
    print(json.dumps(section, indent=2, ensure_ascii=False))

analyzer = FigureAnalyzer()
formatter = SemanticFormatter()

# --- Matplotlib Example (Linear Pattern with Outliers) ---
fig1, ax1 = plt.subplots()
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2 * x + 5 + np.random.normal(0, 0.5, 10) # y = 2x + 5 with some noise
y[8] = 30 # Add an outlier
ax1.plot(x, y, label="Sales Data")
ax1.set_title("Matplotlib Plot with Sales Trend and Outlier")
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Revenue (in millions)")

analysis_mpl = analyzer.analyze(fig1, figure_type="matplotlib")
# Test default: curve_points should NOT be included
semantic_output_mpl = formatter.format(analysis_mpl)

# print("\n=== Matplotlib Semantic Output (FULL) ===")
# print(json.dumps(semantic_output_mpl, indent=2, ensure_ascii=False))

# print("\n=== Matplotlib Semantic Output (SECTIONS, NO curve_points) ===")
# for key in [
#     "metadata", "axes", "layout", "data_summary", "statistical_insights", "pattern_analysis", "visual_elements", "domain_context", "llm_description", "llm_context"
# ]:
#     if key in semantic_output_mpl:
#         print_section(key, semantic_output_mpl[key])

# Test with curve_points explicitly included
semantic_output_mpl_with_curves = formatter.format(analysis_mpl, include_curve_points=True)

print("\n=== Matplotlib Semantic Output (SECTIONS, WITH curve_points) ===")
for key in [
    "metadata", "axes", "layout", "data_summary", "statistical_insights", "pattern_analysis", "visual_elements", "domain_context", "llm_description", "llm_context"
]:
    if key in semantic_output_mpl_with_curves:
        print_section(key, semantic_output_mpl_with_curves[key])

# --- Seaborn Example (No Clear Pattern) ---
# import pandas as pd
# df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8], "y": [5, 2, 4, 8, 7, 6, 9, 8]})
# fig2, ax2 = plt.subplots()
# sns.scatterplot(data=df, x="x", y="y", ax=ax2)
# ax2.set_title("Seaborn Scatter Plot (No Clear Pattern)")

# analysis_sns = analyzer.analyze(fig2, figure_type="seaborn")
# semantic_output_sns = formatter.format(analysis_sns)

# print("\n=== Seaborn Semantic Output (FULL) ===")
# print(json.dumps(semantic_output_sns, indent=2, ensure_ascii=False))

# print("\n=== Seaborn Semantic Output (SECTIONS) ===")
# for key in [
#     "metadata", "axes", "layout", "data_summary", "statistical_insights", "pattern_analysis", "visual_elements", "domain_context", "llm_description", "llm_context"
# ]:
#     if key in semantic_output_sns:
#         print_section(key, semantic_output_sns[key]) 