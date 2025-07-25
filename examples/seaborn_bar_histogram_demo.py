import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from plot2llm.analyzers.seaborn_analyzer import SeabornAnalyzer
import json

def to_native_type(value):
    import numpy as np
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [to_native_type(v) for v in value]
    if isinstance(value, dict):
        return {k: to_native_type(v) for k, v in value.items()}
    return value

def print_section(title, content):
    print(f"\n{'='*20} {title} {'='*20}")
    print(json.dumps(to_native_type(content), indent=2, ensure_ascii=False))

def analyze_and_show(fig, description):
    analyzer = SeabornAnalyzer()
    result = analyzer.analyze(fig)
    print(f"\n{'='*40}\n{description}\n{'='*40}")
    for key, value in result.items():
        print_section(key.upper(), value)
    plt.close(fig)

# Barplot con seaborn
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 12, 36]
fig1, ax1 = plt.subplots()
sns.barplot(x=categories, y=values, ax=ax1, palette='pastel')
ax1.set_title("Seaborn Bar Plot Example")
ax1.set_xlabel("Category")
ax1.set_ylabel("Value")
analyze_and_show(fig1, "Seaborn Bar Plot")

# Histograma con seaborn
# samples = np.random.normal(0, 1, 1000)
# fig2, ax2 = plt.subplots()
# sns.histplot(samples, bins=30, kde=False, color='skyblue', ax=ax2)
# ax2.set_title("Seaborn Histogram Example")
# ax2.set_xlabel("Value")
# ax2.set_ylabel("Frequency")
# analyze_and_show(fig2, "Seaborn Histogram") 