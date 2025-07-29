# Ejemplos de Output Semántico

Este directorio contiene ejemplos completos que demuestran el output semántico de Plot2LLM para diferentes tipos de gráficos.

## Archivos Disponibles

### 1. `semantic_output_matplotlib_example.py`
Ejemplo completo para gráficos de **matplotlib** que incluye:
- **Line Plot**: Relación lineal con tendencia
- **Scatter Plot**: Análisis de correlación
- **Bar Plot**: Comparación categórica
- **Histogram**: Análisis de distribución normal

### 2. `semantic_output_seaborn_example.py`
Ejemplo completo para gráficos de **seaborn** que incluye:
- **Line Plot**: Patrón sinusoidal
- **Scatter Plot**: Análisis de correlación con regresión
- **Bar Plot**: Análisis categórico
- **Histogram**: Análisis de distribución con densidad

## Cómo Ejecutar

### Requisitos
```bash
pip install plot2llm matplotlib seaborn numpy scipy
```

### Ejecutar Ejemplo de Matplotlib
```bash
python examples/semantic_output_matplotlib_example.py
```

### Ejecutar Ejemplo de Seaborn
```bash
python examples/semantic_output_seaborn_example.py
```

## Output Semántico Completo

Cada ejemplo muestra el output semántico completo que incluye:

### Campos Principales
- **`metadata`**: Información del archivo y figura
- **`axes`**: Análisis detallado de los ejes
- **`layout`**: Configuración del layout
- **`data_summary`**: Resumen de los datos
- **`statistical_insights`**: Insights estadísticos
- **`pattern_analysis`**: Análisis de patrones
- **`visual_elements`**: Elementos visuales
- **`domain_context`**: Contexto del dominio
- **`llm_description`**: Descripción para LLMs
- **`llm_context`**: Contexto para LLMs

### Campos Específicos por Tipo de Gráfico

#### Line Plot
- **`lines`**: Información de las líneas
- **`trend_analysis`**: Análisis de tendencias
- **`slope`**: Pendiente de la línea

#### Scatter Plot
- **`collections`**: Información de los puntos
- **`correlation_analysis`**: Análisis de correlación
- **`point_density`**: Densidad de puntos

#### Bar Plot
- **`bars`**: Información de las barras
- **`categories`**: Categorías
- **`categorical_analysis`**: Análisis categórico

#### Histogram
- **`bins`**: Información de los bins
- **`distribution_analysis`**: Análisis de distribución
- **`peak_analysis`**: Análisis de picos

## Opciones de Output

Los ejemplos incluyen comentarios que muestran cómo generar otros formatos:

### Output Text
```python
result_text = converter.convert(fig, 'text')
print(result_text)
```

### Output JSON
```python
result_json = converter.convert(fig, 'json')
print(json.dumps(result_json, indent=2))
```

## Características Destacadas

### 1. Detección Automática de Tipos
- **Numeric**: Para datos continuos
- **Categorical**: Para datos categóricos
- **Date**: Para fechas

### 2. Análisis Estadístico Avanzado
- **Central tendency**: Media, mediana, moda
- **Variability**: Desviación estándar, rango
- **Distribution analysis**: Skewness, kurtosis
- **Correlation analysis**: Coeficientes de correlación

### 3. Detección de Patrones
- **Trend detection**: Tendencias lineales y no lineales
- **Pattern recognition**: Patrones cíclicos, estacionales
- **Distribution types**: Normal, multimodal, uniform

### 4. Contexto para LLMs
- **Structured analysis**: Análisis estructurado
- **Key insights**: Insights principales
- **Interpretation hints**: Pistas de interpretación
- **Analysis suggestions**: Sugerencias de análisis

## Uso en Proyectos

Estos ejemplos pueden ser utilizados como:
- **Referencia** para entender el output semántico
- **Template** para implementar análisis personalizados
- **Testing** para validar funcionalidades
- **Documentation** para usuarios de la librería

## Personalización

Los ejemplos pueden ser personalizados para:
- **Diferentes tipos de datos**
- **Configuraciones específicas**
- **Análisis especializados**
- **Integración con otros sistemas** 