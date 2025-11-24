"""
Chatbot conversacional que utiliza RAG para responder preguntas
basándose en documentos de la empresa.
"""

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from core.rag_system import RAGSystem


class Chatbot:
    """
    Chatbot conversacional que integra el sistema RAG para proporcionar
    respuestas contextualizadas basadas en documentos de la empresa.
    """
    
    def __init__(self, rag_system: RAGSystem, api_key: str, model: str = "gpt-4o"):
        """
        Inicializa el chatbot.
        
        Args:
            rag_system: Sistema RAG para recuperar información relevante
            api_key: Clave API de OpenAI
            model: Modelo de OpenAI a utilizar (gpt-4o, gpt-4.1, o gpt-4o-mini)
        """
        self.rag_system = rag_system
        self.api_key = api_key
        self.model = model
        self.llm = None
        self.conversation_history: List[Dict] = []
        
        # Configurar el LLM
        self._setup_llm()
        
    def _setup_llm(self) -> None:
        """
        Configura el modelo de lenguaje de OpenAI.
        """
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=0.7
        )
        
    def _build_system_prompt(self) -> str:
        """
        Construye el prompt del sistema que define el comportamiento del chatbot.
        
        Returns:
            String con el prompt del sistema
        """
        return """Eres un asistente virtual inteligente de TechInnovate Solutions, una empresa de consultoría tecnológica.

Tu función es responder preguntas sobre la empresa utilizando ÚNICAMENTE la información proporcionada en el contexto.

Reglas importantes:
1. Responde SOLO basándote en la información del contexto proporcionado
2. Si la información no está en el contexto, indica claramente que no tienes esa información
3. Sé claro, conciso y profesional
4. Si el contexto contiene información parcial, menciónalo
5. Mantén un tono amigable pero profesional
6. No inventes información que no esté en el contexto
7. Si te hacen una pregunta general de conversación, responde amablemente pero recuerda tu propósito principal

Recuerda: Tu conocimiento está limitado a los documentos internos de TechInnovate Solutions."""
        
    def _format_context(self, context: str) -> str:
        """
        Formatea el contexto para incluirlo en el prompt.
        
        Args:
            context: Contexto recuperado del sistema RAG
            
        Returns:
            String con el contexto formateado
        """
        return f"""### CONTEXTO RELEVANTE DE LA EMPRESA ###

{context}

### FIN DEL CONTEXTO ###"""
        
    def chat(self, user_message: str) -> str:
        """
        Procesa un mensaje del usuario y genera una respuesta.
        
        Args:
            user_message: Mensaje del usuario
            
        Returns:
            Respuesta del chatbot
        """
        # Recuperar contexto relevante usando RAG
        context = self.rag_system.get_context_for_query(user_message, k=4)
        
        # Construir los mensajes para el LLM
        messages = []
        
        # Mensaje del sistema
        messages.append(SystemMessage(content=self._build_system_prompt()))
        
        # Agregar historial de conversación (últimos 5 intercambios)
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        for entry in recent_history:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                messages.append(AIMessage(content=entry["content"]))
        
        # Mensaje actual con contexto
        current_message = f"""{self._format_context(context)}

### PREGUNTA DEL USUARIO ###
{user_message}"""
        
        messages.append(HumanMessage(content=current_message))
        
        # Obtener respuesta del LLM
        response = self.llm.invoke(messages)
        
        # Guardar en el historial
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response.content
    
    def reset_conversation(self) -> None:
        """
        Reinicia el historial de conversación.
        """
        self.conversation_history = []
        print("🔄 Historial de conversación reiniciado")
        
    def get_conversation_history(self) -> List[Dict]:
        """
        Obtiene el historial completo de la conversación.
        
        Returns:
            Lista con el historial de mensajes
        """
        return self.conversation_history
