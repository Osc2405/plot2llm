import matplotlib.pyplot as plt
import numpy as np
from plot2llm.analyzers.matplotlib_analyzer import MatplotlibAnalyzer
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

def remove_curve_points(data):
    """Recursively remove curve_points from the data structure"""
    if isinstance(data, dict):
        # Remove curve_points if present
        if "curve_points" in data:
            del data["curve_points"]
        # Recursively process all values
        for key, value in data.items():
            data[key] = remove_curve_points(value)
        return data
    elif isinstance(data, list):
        # Recursively process all items in the list
        return [remove_curve_points(item) for item in data]
    else:
        return data

def print_section(title, content):
    print(f"\n{'='*20} {title} {'='*20}")
    print(json.dumps(to_native_type(content), indent=2, ensure_ascii=False))

def analyze_and_show(fig, description):
    analyzer = MatplotlibAnalyzer()
    result = analyzer.analyze(fig)
    
    # Remove curve_points from the result
    result = remove_curve_points(result)
    
    print(f"\n{'='*40}\n{description}\n{'='*40}")
    print_section("FIGURE", result.get("figure", {}))
    print_section("AXES", result.get("axes", {}))
    print_section("STATISTICS", result.get("statistics", {}))
    plt.close(fig)

# 1. Line Plot
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)
fig1, ax1 = plt.subplots()
ax1.plot(x, y, label="Linear Trend")
ax1.set_title("Line Plot Example")
ax1.set_xlabel("X Axis")
ax1.set_ylabel("Y Axis")
ax1.legend()
analyze_and_show(fig1, "Line Plot")

# # 2. Bar Plot
# categories = ['A', 'B', 'C', 'D']
# values = [23, 45, 12, 36]
# fig2, ax2 = plt.subplots()
# ax2.bar(categories, values, color=['#e74c3c', '#3498db', '#2ecc71', '#f1c40f'])
# ax2.set_title("Bar Plot Example")
# ax2.set_xlabel("Category")
# ax2.set_ylabel("Value")
# analyze_and_show(fig2, "Bar Plot")

# # 3. Scatter Plot
# x = np.random.rand(50)
# y = 2 * x + np.random.normal(0, 0.2, 50)
# fig3, ax3 = plt.subplots()
# ax3.scatter(x, y, c=y, cmap='viridis', label="Data Points")
# ax3.set_title("Scatter Plot Example")
# ax3.set_xlabel("Random X")
# ax3.set_ylabel("Random Y")
# ax3.legend()
# analyze_and_show(fig3, "Scatter Plot")

# # 4. Histogram
# samples = np.random.normal(0, 1, 1000)
# fig4, ax4 = plt.subplots()
# ax4.hist(samples, bins=30, color='#9b59b6', alpha=0.7)
# ax4.set_title("Histogram Example")
# ax4.set_xlabel("Value")
# ax4.set_ylabel("Frequency")
# analyze_and_show(fig4, "Histogram")

# 5. Boxplot
# data = [np.random.normal(0, std, 100) for std in range(1, 4)]
# fig5, ax5 = plt.subplots()
# ax5.boxplot(data, vert=True, patch_artist=True, labels=['std=1', 'std=2', 'std=3'])
# ax5.set_title("Boxplot Example")
# ax5.set_xlabel("Distribution")
# ax5.set_ylabel("Value")
# analyze_and_show(fig5, "Boxplot")

# # 6. Pie Chart
# sizes = [15, 30, 45, 10]
# labels = ['Group A', 'Group B', 'Group C', 'Group D']
# fig6, ax6 = plt.subplots()
# ax6.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# ax6.set_title("Pie Chart Example")
# analyze_and_show(fig6, "Pie Chart")

# # 7. Area Plot
# x = np.arange(0, 10, 1)
# y1 = np.random.rand(10)
# y2 = np.random.rand(10)
# fig7, ax7 = plt.subplots()
# ax7.stackplot(x, y1, y2, labels=['y1', 'y2'], colors=['#2ecc71', '#e74c3c'])
# ax7.set_title("Area Plot Example")
# ax7.set_xlabel("X")
# ax7.set_ylabel("Y")
# ax7.legend(loc='upper left')
# analyze_and_show(fig7, "Area Plot")

# # 8. Heatmap
# matrix = np.random.rand(5, 5)
# fig8, ax8 = plt.subplots()
# cax = ax8.imshow(matrix, cmap='coolwarm')
# fig8.colorbar(cax)
# ax8.set_title("Heatmap Example")
# analyze_and_show(fig8, "Heatmap") 