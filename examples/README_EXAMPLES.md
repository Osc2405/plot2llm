# Ejemplos de Uso de Plot2LLM

Este directorio contiene ejemplos completos que demuestran las capacidades de Plot2LLM para diferentes casos de uso y formatos de salida.

## 📁 Archivos Disponibles

### Ejemplos Básicos
- **`minimal_matplotlib.py`**: Ejemplo mínimo con matplotlib
- **`minimal_seaborn.py`**: Ejemplo mínimo con seaborn
- **`seaborn_bar_histogram_demo.py`**: Demo de barras e histogramas con seaborn

### Ejemplos Avanzados
- **`advanced_matplotlib.py`**: Visualizaciones complejas con matplotlib
- **`advanced_seaborn.py`**: Visualizaciones avanzadas con seaborn
- **`multi_plot_analysis_demo.py`**: Análisis de múltiples gráficos

### Ejemplos de Casos de Uso Reales
- **`real_world_analysis.py`**: Análisis financiero, marketing y segmentación de clientes
- **`llm_integration_demo.py`**: Integración con LLMs y diferentes formatos de salida

### Ejemplos de Output Semántico
- **`semantic_output_matplotlib_example.py`**: Output semántico completo para matplotlib
- **`semantic_output_seaborn_example.py`**: Output semántico completo para seaborn
- **`test_semantic_formatter.py`**: Tests del formateador semántico

## 🚀 Cómo Ejecutar

### Requisitos
```bash
pip install plot2llm matplotlib seaborn numpy pandas scipy
```

### Ejemplos Básicos
```bash
# Ejemplos mínimos
python examples/minimal_matplotlib.py
python examples/minimal_seaborn.py

# Demo de barras e histogramas
python examples/seaborn_bar_histogram_demo.py
```

### Ejemplos Avanzados
```bash
# Visualizaciones complejas
python examples/advanced_matplotlib.py
python examples/advanced_seaborn.py

# Análisis de múltiples gráficos
python examples/multi_plot_analysis_demo.py
```

### Casos de Uso Reales
```bash
# Análisis financiero y marketing
python examples/real_world_analysis.py

# Integración con LLMs
python examples/llm_integration_demo.py
```

### Output Semántico
```bash
# Output semántico completo
python examples/semantic_output_matplotlib_example.py
python examples/semantic_output_seaborn_example.py

# Tests del formateador
python examples/test_semantic_formatter.py
```

## 📊 Casos de Uso Reales

### 1. Análisis Financiero (`real_world_analysis.py`)
- **Precios de acciones**: Evolución temporal con línea de tendencia
- **Distribución de retornos**: Histograma con estadísticas
- **Insights**: Análisis de volatilidad y tendencias

### 2. Análisis de Marketing (`real_world_analysis.py`)
- **Conversiones por canal**: Gráfico de barras con valores
- **ROI por canal**: Análisis de rentabilidad
- **Costo vs conversión**: Dispersión con etiquetas
- **Distribución de costos**: Gráfico circular

### 3. Segmentación de Clientes (`real_world_analysis.py`)
- **Distribución demográfica**: Histogramas por segmento
- **Análisis de ingresos**: Boxplots y dispersión
- **Matriz de correlaciones**: Heatmap de variables
- **Satisfacción por segmento**: Violin plots

### 4. Integración con LLMs (`llm_integration_demo.py`)
- **Pipeline de análisis**: Visualización compleja con múltiples subplots
- **Generación de prompts**: Formatos optimizados para diferentes LLMs
- **Comparación de formatos**: Text, JSON, Semantic
- **Manejo de errores**: Demostración de robustez

## 🔧 Formatos de Salida

### Texto
```python
result = plot2llm.convert(fig, format='text')
# Descripción narrativa para documentación
```

### JSON
```python
result = plot2llm.convert(fig, format='json')
# Estructura de datos para procesamiento programático
```

### Semántico
```python
result = plot2llm.convert(fig, format='semantic')
# Análisis completo optimizado para LLMs
```

## 📈 Tipos de Visualización Soportados

### Matplotlib
- ✅ Line plots
- ✅ Scatter plots
- ✅ Bar charts
- ✅ Histograms
- ✅ Box plots
- ✅ Subplots
- ✅ Multiple axes

### Seaborn
- ✅ Line plots
- ✅ Scatter plots
- ✅ Bar plots
- ✅ Histograms
- ✅ Violin plots
- ✅ Heatmaps
- ✅ Faceted plots

## 🎯 Características Destacadas

### 1. Detección Automática
- **Tipos de datos**: Numérico, categórico, fecha
- **Tipos de gráfico**: Línea, barras, dispersión, histograma
- **Análisis estadístico**: Correlaciones, distribuciones, outliers

### 2. Análisis Estadístico Avanzado
- **Central tendency**: Media, mediana, moda
- **Variability**: Desviación estándar, rango, IQR
- **Distribution**: Skewness, kurtosis, normalidad
- **Correlations**: Pearson, Spearman
- **Outliers**: Detección por IQR

### 3. Output Optimizado para LLMs
- **Contexto semántico**: Descripciones naturales
- **Insights automáticos**: Patrones y tendencias
- **Recomendaciones**: Sugerencias de análisis adicional
- **Estructura consistente**: Formato estandarizado

## 🔍 Niveles de Detalle

### Low
- Información básica del gráfico
- Estadísticas principales
- Descripción general

### Medium
- Análisis detallado
- Patrones identificados
- Insights estadísticos

### High
- Análisis completo
- Correlaciones y tendencias
- Recomendaciones específicas
- Contexto semántico rico

## 🛠️ Configuración Avanzada

### Personalización de Analizadores
```python
from plot2llm import FigureConverter

converter = FigureConverter(
    detail_level='high',
    include_data=True,
    include_colors=True,
    include_statistics=True
)
```

### Registro de Formateadores Personalizados
```python
converter.register_formatter('custom', my_custom_formatter)
result = converter.convert(fig, 'custom')
```

## 📋 Verificación de Ejemplos

### Comando de Verificación Completa
```bash
# Ejecutar todos los ejemplos
for example in examples/*.py; do
    if [[ $example != *"__init__"* && $example != *"test_"* ]]; then
        echo "Ejecutando: $example"
        python "$example"
        echo "---"
    fi
done
```

### Verificación de Formatos
```bash
# Verificar que todos los formatos funcionan
python -c "
import matplotlib.pyplot as plt
import plot2llm
fig, ax = plt.subplots()
ax.plot([1,2,3], [1,4,2])
for fmt in ['text', 'json', 'semantic']:
    result = plot2llm.convert(fig, fmt)
    print(f'✅ {fmt}: {len(str(result))} caracteres')
"
```

## 🎉 Resultados Esperados

### Ejemplos Básicos
- ✅ Gráficos simples convertidos correctamente
- ✅ Output en formato texto legible
- ✅ Sin errores de importación

### Ejemplos Avanzados
- ✅ Visualizaciones complejas procesadas
- ✅ Múltiples subplots analizados
- ✅ Estadísticas detalladas generadas

### Casos de Uso Reales
- ✅ Análisis financiero con insights
- ✅ Marketing con métricas de ROI
- ✅ Segmentación con patrones demográficos
- ✅ Integración LLM con prompts optimizados

### Output Semántico
- ✅ Estructura JSON válida
- ✅ Contexto semántico rico
- ✅ Insights automáticos
- ✅ Recomendaciones específicas

## 📞 Soporte

Si encuentras problemas con los ejemplos:

1. **Verificar instalación**: `pip install plot2llm[all,test,dev]`
2. **Verificar dependencias**: `python -c "import matplotlib, seaborn, plot2llm"`
3. **Ejecutar tests**: `pytest tests/ -v`
4. **Revisar documentación**: `docs/API_REFERENCE.md`

## 🔄 Actualizaciones

Los ejemplos se actualizan regularmente para reflejar las nuevas características de Plot2LLM. Para la versión más reciente:

```bash
pip install --upgrade plot2llm
git pull origin main
```

---

**¡Disfruta explorando las capacidades de Plot2LLM con estos ejemplos!**
