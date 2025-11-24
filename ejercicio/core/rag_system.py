"""
Sistema RAG (Retrieval-Augmented Generation) para recuperar información relevante
de documentos markdown y proporcionar contexto a un chatbot.
"""

import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.schema import Document


class RAGSystem:
    """
    Sistema RAG que gestiona la carga de documentos, creación de embeddings,
    almacenamiento en vector store y recuperación de información relevante.
    """
    
    def __init__(self, documents_path: str, api_key: str):
        """
        Inicializa el sistema RAG.
        
        Args:
            documents_path: Ruta al directorio que contiene los documentos markdown
            api_key: Clave API de OpenAI
        """
        self.documents_path = documents_path
        self.api_key = api_key
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        
    def initialize(self) -> None:
        """
        Inicializa el sistema completo: carga documentos, crea embeddings y vector store.
        """
        print("🚀 Inicializando sistema RAG...")
        
        # Configurar embeddings
        self._setup_embeddings()
        
        # Cargar y procesar documentos
        self._load_documents()
        
        # Crear vector store
        self._create_vector_store()
        
        print("✅ Sistema RAG inicializado correctamente")
        print(f"📚 Documentos procesados: {len(self.documents)}")
        
    def _setup_embeddings(self) -> None:
        """
        Configura el modelo de embeddings de OpenAI.
        """
        print("⚙️  Configurando modelo de embeddings...")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key
        )
        
    def _load_documents(self) -> None:
        """
        Carga los documentos markdown del directorio especificado y los divide en chunks.
        """
        print("📖 Cargando documentos markdown...")
        
        # Cargar documentos markdown
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        raw_documents = loader.load()
        
        print(f"   Archivos cargados: {len(raw_documents)}")
        
        # Dividir documentos en chunks más pequeños
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(raw_documents)
        print(f"   Chunks generados: {len(self.documents)}")
        
    def _create_vector_store(self) -> None:
        """
        Crea el vector store en memoria con los documentos embeddeados.
        """
        print("🔢 Creando vector store y generando embeddings...")
        
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        
        print("   Vector store creado exitosamente")
        
    def retrieve_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Recupera los documentos más relevantes para una consulta.
        
        Args:
            query: Consulta del usuario
            k: Número de documentos a recuperar (por defecto 4)
            
        Returns:
            Lista de documentos relevantes
        """
        if not self.vector_store:
            raise ValueError("El sistema RAG no ha sido inicializado. Llama a initialize() primero.")
        
        # Realizar búsqueda por similitud
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        return relevant_docs
    
    def get_context_for_query(self, query: str, k: int = 4) -> str:
        """
        Obtiene el contexto relevante para una consulta como un string.
        
        Args:
            query: Consulta del usuario
            k: Número de documentos a recuperar
            
        Returns:
            String con el contexto relevante concatenado
        """
        relevant_docs = self.retrieve_relevant_documents(query, k)
        
        # Concatenar el contenido de los documentos relevantes
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        return context
