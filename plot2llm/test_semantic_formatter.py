from plot2llm.formatters import SemanticFormatter
from plot2llm.analyzers import FigureAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Utilidad para imprimir secciones de forma clara
def print_section(title, section):
    print(f"\n--- {title} ---")
    print(json.dumps(section, indent=2, ensure_ascii=False))

analyzer = FigureAnalyzer()
formatter = SemanticFormatter()

# --- Matplotlib Example ---
fig1, ax1 = plt.subplots()
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
ax1.plot(x, y, label="Line")
ax1.set_title("Matplotlib Line Plot")
ax1.set_xlabel("X Axis")
ax1.set_ylabel("Y Axis")

analysis_mpl = analyzer.analyze(fig1, figure_type="matplotlib")
semantic_output_mpl = formatter.format(analysis_mpl)

print("\n=== Matplotlib Semantic Output (FULL) ===")
print(json.dumps(semantic_output_mpl, indent=2, ensure_ascii=False))

print("\n=== Matplotlib Semantic Output (SECTIONS) ===")
for key in [
    "metadata", "axes", "layout", "data_summary", "statistical_insights", "pattern_analysis", "visual_elements", "domain_context", "llm_description", "llm_context"
]:
    if key in semantic_output_mpl:
        print_section(key, semantic_output_mpl[key])

# --- Seaborn Example ---
import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 7, 6, 8, 7]})
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="x", y="y", ax=ax2)
ax2.set_title("Seaborn Scatter Plot")

analysis_sns = analyzer.analyze(fig2, figure_type="seaborn")
semantic_output_sns = formatter.format(analysis_sns)

print("\n=== Seaborn Semantic Output (FULL) ===")
print(json.dumps(semantic_output_sns, indent=2, ensure_ascii=False))

print("\n=== Seaborn Semantic Output (SECTIONS) ===")
for key in [
    "metadata", "axes", "layout", "data_summary", "statistical_insights", "pattern_analysis", "visual_elements", "domain_context", "llm_description", "llm_context"
]:
    if key in semantic_output_sns:
        print_section(key, semantic_output_sns[key]) 