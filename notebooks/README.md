# Plot2LLM - Notebooks de Demostración

Este directorio contiene notebooks de Jupyter que demuestran el uso de la librería **Plot2LLM**.

## 📁 Contenido

### `plot2llm_demo.ipynb`
Notebook completo de demostración que incluye:

- ✅ **Instalación y configuración** básica
- ✅ **Ejemplos básicos** de conversión de figuras
- ✅ **Análisis de diferentes tipos** de gráficos
- ✅ **Comparación de formatos** de salida (texto, JSON, semántico)
- ✅ **Integración con LLMs** (simulación)
- ✅ **Caso de uso real** con análisis de datos de clientes
- ✅ **Dashboard completo** de análisis de negocio

## 🚀 Cómo Usar

### 1. Instalación de Dependencias

```bash
# Instalar plot2llm
pip install plot2llm

# Instalar dependencias adicionales para el notebook
pip install jupyter matplotlib seaborn pandas numpy
```

### 2. Ejecutar el Notebook

```bash
# Navegar al directorio de notebooks
cd notebooks

# Iniciar Jupyter
jupyter notebook

# Abrir plot2llm_demo.ipynb
```

### 3. Alternativa con JupyterLab

```bash
# Instalar JupyterLab
pip install jupyterlab

# Iniciar JupyterLab
jupyter lab
```

## 📊 Estructura del Demo

### Sección 1: Configuración Básica
- Verificación de instalación
- Importación de librerías
- Configuración de matplotlib

### Sección 2: Ejemplo Básico
- Creación de figura simple
- Conversión a diferentes formatos
- Comparación de resultados

### Sección 3: Análisis de Tipos de Gráficos
- Gráfico de dispersión
- Gráfico de barras
- Histograma
- Análisis detallado de cada tipo

### Sección 4: Gráfico Complejo
- Múltiples subplots
- Análisis detallado
- Extracción de estadísticas

### Sección 5: Comparación de Formatos
- Formato de texto
- Formato JSON
- Formato semántico
- Análisis comparativo

### Sección 6: Integración con LLMs
- Conversión optimizada para LLMs
- Creación de prompts
- Simulación de análisis

### Sección 7: Caso de Uso Real
- Dataset de clientes
- Dashboard completo
- Análisis de negocio

## 🎯 Objetivos de Aprendizaje

Al completar el notebook, serás capaz de:

1. **Instalar y configurar** Plot2LLM correctamente
2. **Convertir figuras** a diferentes formatos
3. **Analizar diferentes tipos** de gráficos
4. **Entender las diferencias** entre formatos de salida
5. **Integrar Plot2LLM** en flujos de trabajo de análisis
6. **Preparar datos** para análisis con LLMs
7. **Crear dashboards** analíticos completos

## 🔧 Personalización

El notebook está diseñado para ser fácilmente personalizable:

- **Modifica los datos** de ejemplo con tus propios datasets
- **Ajusta los parámetros** de análisis según tus necesidades
- **Agrega nuevos tipos** de gráficos
- **Personaliza los prompts** para LLMs
- **Adapta el dashboard** a tu dominio específico

## 📈 Próximos Pasos

Después de completar el demo, puedes:

1. **Explorar la documentación** completa en `/docs/`
2. **Revisar los ejemplos** en `/examples/`
3. **Probar con tus propios datos**
4. **Integrar en tus proyectos** de análisis
5. **Contribuir** al desarrollo de la librería

## 🐛 Solución de Problemas

### Error de Importación
```python
# Si plot2llm no se encuentra
pip install plot2llm --upgrade
```

### Error de Dependencias
```python
# Instalar todas las dependencias
pip install plot2llm[all]
```

### Problemas con Matplotlib
```python
# Configurar backend no interactivo
import matplotlib
matplotlib.use('Agg')
```

## 📞 Soporte

- **Documentación**: [README principal](../README.md)
- **Ejemplos**: [Directorio de ejemplos](../examples/)
- **Issues**: [GitHub Issues](https://github.com/your-username/plot2llm/issues)

---

**¡Disfruta explorando Plot2LLM! 🚀** 