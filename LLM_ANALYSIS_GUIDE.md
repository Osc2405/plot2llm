# 🧠 Plot2LLM: Guía de Análisis para LLMs

Esta guía explica cómo usar `plot2llm` para extraer información rica que permite a los LLMs (Large Language Models) entender y analizar visualizaciones de datos de manera efectiva.

## 📋 Índice

1. [Introducción](#introducción)
2. [Capacidades de Análisis](#capacidades-de-análisis)
3. [Formatos de Salida](#formatos-de-salida)
4. [Casos de Uso para LLMs](#casos-de-uso-para-llms)
5. [Ejemplos Prácticos](#ejemplos-prácticos)
6. [Mejores Prácticas](#mejores-prácticas)
7. [Integración con LLMs](#integración-con-llms)

## 🎯 Introducción

`plot2llm` es una biblioteca Python que convierte figuras y visualizaciones en formatos que los LLMs pueden entender y procesar fácilmente. Esto permite a los LLMs:

- **Analizar patrones** en los datos visualizados
- **Extraer insights** estadísticos y de negocio
- **Generar reportes** basados en visualizaciones
- **Responder preguntas** sobre gráficos y datos
- **Comparar múltiples** visualizaciones

## 🔍 Capacidades de Análisis

### 1. **Análisis de Tendencias**
- Detección de patrones lineales, exponenciales y estacionales
- Identificación de puntos de inflexión
- Análisis de crecimiento y decrecimiento

### 2. **Análisis de Correlaciones**
- Detección de correlaciones positivas y negativas
- Identificación de patrones de dispersión
- Análisis de la fuerza de las relaciones

### 3. **Análisis de Distribuciones**
- Identificación de distribuciones normales, sesgadas, bimodales
- Detección de outliers y valores atípicos
- Análisis de la forma y características de los datos

### 4. **Análisis Estadístico**
- Estadísticas descriptivas (media, mediana, desviación estándar)
- Rangos y percentiles
- Análisis de variabilidad

### 5. **Análisis de Negocio**
- Extracción de contexto empresarial
- Identificación de métricas clave
- Análisis de rendimiento y comparaciones

## 📊 Formatos de Salida

### 1. **Formato de Texto** (`text`)
```python
from plot2llm import FigureConverter

converter = FigureConverter()
text_result = converter.convert(fig, output_format='text')
```

**Ventajas para LLMs:**
- Fácil de procesar y entender
- Información estructurada en lenguaje natural
- Incluye estadísticas y metadatos

**Ejemplo de salida:**
```
Figure type: matplotlib.figure
Dimensions (inches): [10. 6.]
Title: Sales Growth Analysis
Number of axes: 1

Axis 0: type=linear, x_label=Year, y_label=Sales, 
x_range=(2018, 2023), y_range=(100, 300), grid=True, legend=True

Data points: 6
Data types: ['line_plot']
Statistics: mean=200.0, std=71.4, min=100, max=300, median=175.0

Colors: ['#1f77b4']
Markers: [<matplotlib.markers.MarkerStyle object>]
Line styles: ['-']
Background color: #ffffff
```

### 2. **Formato JSON** (`json`)
```python
json_result = converter.convert(fig, output_format='json')
```

**Ventajas para LLMs:**
- Estructura de datos clara y parseable
- Fácil extracción de campos específicos
- Compatible con APIs y sistemas

### 3. **Formato Semántico** (`semantic`)
```python
semantic_result = converter.convert(fig, output_format='semantic')
```

**Ventajas para LLMs:**
- Información más rica y contextual
- Metadatos adicionales
- Estructura optimizada para análisis

## 🎯 Casos de Uso para LLMs

### 1. **Análisis Automático de Reportes**
```python
# El LLM puede analizar automáticamente múltiples gráficos
def analyze_report_charts(charts):
    insights = []
    for chart in charts:
        result = converter.convert(chart, output_format='semantic')
        insights.append(extract_insights(result))
    return generate_report(insights)
```

### 2. **Respuesta a Preguntas sobre Datos**
```python
# El LLM puede responder preguntas específicas sobre visualizaciones
def answer_chart_question(chart, question):
    analysis = converter.convert(chart, output_format='text')
    return llm.answer(f"Based on this chart analysis: {analysis}\nQuestion: {question}")
```

### 3. **Comparación de Visualizaciones**
```python
# El LLM puede comparar múltiples gráficos
def compare_charts(chart1, chart2):
    analysis1 = converter.convert(chart1, output_format='semantic')
    analysis2 = converter.convert(chart2, output_format='semantic')
    return llm.compare(analysis1, analysis2)
```

### 4. **Generación de Insights de Negocio**
```python
# El LLM puede extraer insights empresariales
def extract_business_insights(chart):
    analysis = converter.convert(chart, output_format='text')
    return llm.extract_insights(analysis, context="business_analysis")
```

## 💡 Ejemplos Prácticos

### Ejemplo 1: Análisis de Tendencias de Ventas
```python
import matplotlib.pyplot as plt
import numpy as np
from plot2llm import FigureConverter

# Crear gráfico de ventas
fig, ax = plt.subplots()
years = [2019, 2020, 2021, 2022, 2023]
sales = [100, 120, 150, 200, 280]
ax.plot(years, sales, 'bo-', linewidth=2)
ax.set_title('Company Sales Growth')
ax.set_xlabel('Year')
ax.set_ylabel('Sales (in thousands)')

# Analizar para LLM
converter = FigureConverter()
analysis = converter.convert(fig, output_format='text')

# El LLM puede ahora analizar:
# - Tasa de crecimiento anual
# - Patrones estacionales
# - Proyecciones futuras
# - Insights de negocio
```

### Ejemplo 2: Análisis de Correlaciones
```python
# Crear gráfico de correlación
fig, ax = plt.subplots()
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)
ax.scatter(x, y, alpha=0.6)
ax.set_title('Correlation Analysis')

# Analizar para LLM
analysis = converter.convert(fig, output_format='semantic')

# El LLM puede identificar:
# - Fuerza de la correlación
# - Dirección de la relación
# - Significancia estadística
# - Implicaciones prácticas
```

## 🚀 Mejores Prácticas

### 1. **Configuración Óptima para LLMs**
```python
# Usar nivel de detalle alto para análisis completo
converter = FigureConverter()
result = converter.convert(
    fig, 
    output_format='semantic',
    detail_level='high',
    include_statistics=True,
    include_data=True
)
```

### 2. **Procesamiento de Múltiples Gráficos**
```python
def batch_analyze_charts(charts):
    """Analiza múltiples gráficos de manera eficiente."""
    converter = FigureConverter()
    results = []
    
    for chart in charts:
        try:
            result = converter.convert(chart, output_format='semantic')
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    
    return results
```

### 3. **Extracción de Insights Específicos**
```python
def extract_trend_insights(analysis):
    """Extrae insights específicos sobre tendencias."""
    if 'data_info' in analysis and 'statistics' in analysis['data_info']:
        stats = analysis['data_info']['statistics']
        
        insights = {
            'mean': stats.get('mean'),
            'trend_direction': 'increasing' if stats.get('mean', 0) > 0 else 'decreasing',
            'variability': stats.get('std'),
            'range': f"{stats.get('min')} to {stats.get('max')}"
        }
        
        return insights
    return None
```

## 🔗 Integración con LLMs

### 1. **Prompt Engineering para Análisis**
```python
def create_analysis_prompt(chart_analysis):
    return f"""
    Analiza la siguiente visualización de datos:
    
    {chart_analysis}
    
    Por favor proporciona:
    1. Un resumen ejecutivo de los datos
    2. Los patrones principales identificados
    3. Insights de negocio relevantes
    4. Recomendaciones basadas en los datos
    5. Limitaciones o consideraciones importantes
    """
```

### 2. **Integración con APIs de LLMs**
```python
import openai

def analyze_with_gpt4(chart):
    converter = FigureConverter()
    analysis = converter.convert(chart, output_format='text')
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un experto analista de datos."},
            {"role": "user", "content": create_analysis_prompt(analysis)}
        ]
    )
    
    return response.choices[0].message.content
```

### 3. **Análisis Comparativo**
```python
def compare_multiple_charts(charts, comparison_criteria):
    converter = FigureConverter()
    analyses = []
    
    for chart in charts:
        analysis = converter.convert(chart, output_format='semantic')
        analyses.append(analysis)
    
    # El LLM puede comparar múltiples análisis
    comparison_prompt = f"""
    Compara las siguientes visualizaciones basándote en: {comparison_criteria}
    
    Análisis 1: {analyses[0]}
    Análisis 2: {analyses[1]}
    ...
    
    Proporciona una comparación estructurada.
    """
    
    return llm.analyze(comparison_prompt)
```

## 📈 Métricas de Calidad

### Indicadores de Calidad del Análisis:
- **Completitud**: ¿Se extrajo toda la información relevante?
- **Precisión**: ¿Los datos extraídos son correctos?
- **Contexto**: ¿Se capturó el contexto empresarial?
- **Estructura**: ¿La información está bien organizada?

### Evaluación de Rendimiento:
```python
def evaluate_analysis_quality(original_chart, llm_analysis):
    """Evalúa la calidad del análisis realizado por el LLM."""
    converter = FigureConverter()
    ground_truth = converter.convert(original_chart, output_format='semantic')
    
    # Comparar con el análisis del LLM
    accuracy = compare_analyses(ground_truth, llm_analysis)
    completeness = check_completeness(llm_analysis)
    relevance = assess_relevance(llm_analysis)
    
    return {
        'accuracy': accuracy,
        'completeness': completeness,
        'relevance': relevance,
        'overall_score': (accuracy + completeness + relevance) / 3
    }
```

## 🎯 Conclusión

`plot2llm` proporciona una base sólida para que los LLMs analicen visualizaciones de datos de manera efectiva. Al combinar la extracción de información estructurada con las capacidades de procesamiento de lenguaje natural de los LLMs, se pueden obtener insights valiosos y análisis profundos de datos visuales.

### Próximos Pasos:
1. **Experimentar** con diferentes tipos de visualizaciones
2. **Refinar prompts** para casos de uso específicos
3. **Integrar** con sistemas de análisis de datos existentes
4. **Desarrollar** capacidades adicionales según necesidades específicas

---

*Para más información, consulta la documentación completa de `plot2llm` y los ejemplos incluidos en el repositorio.* 