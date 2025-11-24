# 🤖 Chatbot RAG - TechInnovate Solutions

Sistema de chatbot conversacional que utiliza la técnica RAG (Retrieval-Augmented Generation) con LangChain y OpenAI para responder preguntas basándose en documentos markdown de la empresa TechInnovate Solutions.

## 📋 Descripción

Este proyecto implementa un chatbot inteligente que puede responder preguntas sobre una empresa ficticia utilizando información almacenada en documentos markdown. El sistema utiliza embeddings para vectorizar los documentos y un vector store en memoria para recuperar información relevante que luego es utilizada por un modelo de lenguaje (LLM) para generar respuestas contextualizadas.

## 🏗️ Arquitectura

El proyecto sigue una arquitectura modular:

```
ejercicio/
├── main.py              # Punto de entrada de la aplicación
├── documents/
│   ├── documento1.md    # Información general de la empresa
│   └── documento2.md    # Políticas y procedimientos internos
├── core/
│   ├── __init__.py
│   ├── rag_system.py    # Sistema RAG principal
│   └── chatbot.py       # Lógica del chatbot
├── requirements.txt     # Dependencias del proyecto
├── .env                 # Variables de entorno (API keys)
└── README.md            # Este archivo
```

## 🚀 Características

- ✅ Sistema de embeddings usando `text-embedding-3-small` de OpenAI
- ✅ Vector store en memoria con `InMemoryVectorStore` de LangChain
- ✅ Chatbot conversacional con modelos GPT-4o/GPT-4.1/GPT-4o-mini
- ✅ Técnica RAG para recuperación de información relevante
- ✅ Procesamiento y vectorización de documentos markdown
- ✅ Mantenimiento del contexto conversacional
- ✅ Respuestas basadas únicamente en documentos procesados
- ✅ Sistema de retrieval con búsqueda por similitud
- ✅ Interfaz de línea de comandos (CLI) interactiva

## 📦 Requisitos Previos

- Python 3.8 o superior
- Cuenta de OpenAI con API key
- pip (gestor de paquetes de Python)

## 🔧 Instalación

1. **Clonar o descargar el proyecto**

2. **Instalar las dependencias**

```bash
pip install -r requirements.txt
```

3. **Configurar las variables de entorno**

Asegúrate de que el archivo `.env` en el directorio `ejercicio` contenga tu API key de OpenAI:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

## 💻 Uso

### Iniciar el chatbot

Desde el directorio `ejercicio`, ejecuta:

```bash
python main.py
```

### Comandos disponibles

Una vez iniciado el chatbot, puedes usar los siguientes comandos:

- **Hacer una pregunta**: Simplemente escribe tu pregunta y presiona Enter
- **/salir** o **quit**: Terminar la conversación
- **/reiniciar**: Reiniciar el historial de conversación
- **/historial**: Ver el historial completo de la conversación

### Ejemplos de preguntas

```
👤 Tú: ¿Cuál es la misión de la empresa?
👤 Tú: ¿Qué servicios ofrece TechInnovate Solutions?
👤 Tú: ¿Cuál es la política de vacaciones?
👤 Tú: ¿Qué beneficios tienen los empleados?
👤 Tú: Háblame sobre el horario de trabajo
👤 Tú: ¿Qué certificaciones tiene la empresa?
```

## 🧠 Funcionamiento del Sistema RAG

1. **Carga de documentos**: Los archivos markdown se cargan desde el directorio `documents/`
2. **Chunking**: Los documentos se dividen en fragmentos más pequeños (chunks) de 1000 caracteres con 200 de overlap
3. **Embeddings**: Cada chunk se convierte en un vector usando `text-embedding-3-small`
4. **Vector Store**: Los vectores se almacenan en memoria usando `InMemoryVectorStore`
5. **Retrieval**: Cuando el usuario hace una pregunta:
   - La pregunta se convierte en un vector
   - Se buscan los 4 chunks más similares
   - Se concatenan para formar el contexto
6. **Generación**: El LLM genera una respuesta usando:
   - El contexto recuperado
   - El historial de conversación
   - Un prompt del sistema que define el comportamiento

## 🎯 Componentes Principales

### RAGSystem (`core/rag_system.py`)

Gestiona:
- Configuración de embeddings
- Carga y procesamiento de documentos
- Creación del vector store
- Recuperación de información relevante

### Chatbot (`core/chatbot.py`)

Gestiona:
- Integración con el sistema RAG
- Interacción con el modelo de OpenAI
- Mantenimiento del historial conversacional
- Generación de respuestas contextualizadas

### Main (`main.py`)

Gestiona:
- Interfaz de línea de comandos
- Bucle de conversación
- Comandos del usuario
- Manejo de errores

## 📊 Modelos Disponibles

El chatbot puede usar cualquiera de estos modelos de OpenAI (configurable en `main.py`):

- **gpt-4o**: Modelo más capaz y reciente (por defecto)
- **gpt-4.1**: Versión anterior de GPT-4
- **gpt-4o-mini**: Versión más ligera y económica

Para cambiar el modelo, edita la línea en `main.py`:

```python
chatbot = Chatbot(rag_system=rag_system, api_key=api_key, model="gpt-4o")
```

## 📚 Documentos de la Empresa

El sistema incluye dos documentos markdown ficticios:

1. **documento1.md**: Información general sobre TechInnovate Solutions
   - Historia de la empresa
   - Misión, visión y valores
   - Servicios principales
   - Equipo y cultura
   - Clientes y certificaciones

2. **documento2.md**: Políticas y procedimientos internos
   - Políticas de RRHH
   - Horarios de trabajo
   - Beneficios sociales
   - Código de conducta
   - Procedimientos operativos
   - Evaluación de desempeño

## 🔍 Personalización

### Añadir más documentos

Simplemente añade archivos `.md` al directorio `documents/`. El sistema los procesará automáticamente.

### Ajustar el tamaño de chunks

En `rag_system.py`, modifica los parámetros:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Tamaño del chunk
    chunk_overlap=200,    # Overlap entre chunks
)
```

### Cambiar el número de documentos recuperados

En `chatbot.py`, modifica el parámetro `k`:

```python
context = self.rag_system.get_context_for_query(user_message, k=4)
```

### Ajustar la temperatura del modelo

En `chatbot.py`, modifica:

```python
self.llm = ChatOpenAI(
    model=self.model,
    api_key=self.api_key,
    temperature=0.7  # Más bajo = más determinista, más alto = más creativo
)
```

## ⚠️ Limitaciones

- El chatbot solo puede responder basándose en la información de los documentos
- La calidad de las respuestas depende de la calidad de los documentos
- Los documentos se cargan en memoria (no escalable para miles de documentos)
- Requiere conexión a internet para llamadas a la API de OpenAI
- Tiene un costo asociado por uso de la API de OpenAI

## 🐛 Solución de Problemas

### Error: "No se encontró la API key de OpenAI"
- Verifica que el archivo `.env` existe y contiene `OPENAI_API_KEY`
- Asegúrate de que la API key es válida

### Error: "Import could not be resolved"
- Ejecuta `pip install -r requirements.txt`
- Verifica que estás usando Python 3.8 o superior

### El chatbot no responde correctamente
- Verifica que los documentos markdown están en `documents/`
- Comprueba que el modelo de OpenAI está disponible
- Revisa tu cuota de API de OpenAI

## 📝 Licencia

Este es un proyecto educativo para fines de aprendizaje.

## 👥 Autor

Desarrollado como parte del Bootcamp de IA - Sprint 4

## 🙏 Agradecimientos

- LangChain por el framework RAG
- OpenAI por los modelos de embeddings y LLM
- La comunidad de Python por las excelentes librerías
