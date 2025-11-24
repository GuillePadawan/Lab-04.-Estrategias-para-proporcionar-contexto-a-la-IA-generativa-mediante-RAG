"""
Aplicación principal del chatbot RAG.
Proporciona una interfaz de línea de comandos para interactuar con el chatbot.
"""

import os
from dotenv import load_dotenv
from core.rag_system import RAGSystem
from core.chatbot import Chatbot


def print_welcome_message():
    """Muestra el mensaje de bienvenida."""
    print("\n" + "="*70)
    print("🤖 CHATBOT RAG - TechInnovate Solutions")
    print("="*70)
    print("\n¡Bienvenido! Soy tu asistente virtual de TechInnovate Solutions.")
    print("\nPuedo responder preguntas sobre:")
    print("  • Información general de la empresa")
    print("  • Políticas y procedimientos internos")
    print("  • Beneficios y horarios de trabajo")
    print("  • Servicios que ofrecemos")
    print("  • Y mucho más...")
    print("\n📌 Comandos disponibles:")
    print("  • /salir o quit - Terminar la conversación")
    print("  • /reiniciar - Reiniciar el historial de conversación")
    print("  • /historial - Ver el historial de la conversación")
    print("\n" + "-"*70 + "\n")


def print_separator():
    """Imprime un separador visual."""
    print("\n" + "-"*70 + "\n")


def display_history(chatbot: Chatbot):
    """
    Muestra el historial de conversación.
    
    Args:
        chatbot: Instancia del chatbot
    """
    history = chatbot.get_conversation_history()
    
    if not history:
        print("\n📭 No hay historial de conversación todavía.")
        return
    
    print("\n" + "="*70)
    print("📜 HISTORIAL DE CONVERSACIÓN")
    print("="*70 + "\n")
    
    for i, entry in enumerate(history):
        if entry["role"] == "user":
            print(f"👤 Usuario: {entry['content']}\n")
        else:
            print(f"🤖 Asistente: {entry['content']}\n")
        
        if i < len(history) - 1:
            print("-"*70 + "\n")


def main():
    """Función principal de la aplicación."""
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Obtener la API key de OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: No se encontró la API key de OpenAI en el archivo .env")
        print("   Asegúrate de tener un archivo .env con OPENAI_API_KEY configurado.")
        return
    
    try:
        # Configurar rutas
        current_dir = os.path.dirname(os.path.abspath(__file__))
        documents_path = os.path.join(current_dir, "documents")
        
        # Verificar que existe el directorio de documentos
        if not os.path.exists(documents_path):
            print(f"❌ Error: No se encontró el directorio de documentos en {documents_path}")
            return
        
        # Inicializar el sistema RAG
        print("\n🔧 Inicializando el sistema...")
        rag_system = RAGSystem(documents_path=documents_path, api_key=api_key)
        rag_system.initialize()
        
        # Inicializar el chatbot
        # Puedes cambiar el modelo aquí: "gpt-4o", "gpt-4.1", "gpt-4o-mini"
        chatbot = Chatbot(rag_system=rag_system, api_key=api_key, model="gpt-4o")
        
        # Mostrar mensaje de bienvenida
        print_welcome_message()
        
        # Bucle principal de conversación
        while True:
            try:
                # Obtener entrada del usuario
                user_input = input("👤 Tú: ").strip()
                
                # Verificar si el usuario quiere salir
                if user_input.lower() in ["/salir", "quit", "exit", "salir"]:
                    print("\n👋 ¡Gracias por usar el chatbot de TechInnovate Solutions!")
                    print("   ¡Hasta pronto!\n")
                    break
                
                # Verificar si el usuario quiere reiniciar
                if user_input.lower() in ["/reiniciar", "reiniciar", "reset"]:
                    chatbot.reset_conversation()
                    continue
                
                # Verificar si el usuario quiere ver el historial
                if user_input.lower() in ["/historial", "historial", "history"]:
                    display_history(chatbot)
                    continue
                
                # Ignorar entradas vacías
                if not user_input:
                    continue
                
                # Procesar la consulta
                print("\n🤖 Asistente: ", end="", flush=True)
                response = chatbot.chat(user_input)
                print(response)
                
                print_separator()
                
            except KeyboardInterrupt:
                print("\n\n👋 Conversación interrumpida. ¡Hasta pronto!\n")
                break
            except Exception as e:
                print(f"\n❌ Error al procesar la consulta: {str(e)}")
                print("   Por favor, intenta de nuevo.\n")
                continue
    
    except Exception as e:
        print(f"\n❌ Error al inicializar el sistema: {str(e)}")
        print("   Verifica tu configuración y las dependencias instaladas.\n")
        return


if __name__ == "__main__":
    main()
