import logging
import os
from dotenv import load_dotenv
from typing import Any

from phi.llm.groq import Groq
from phi.agent import Agent
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.vectordb.lancedb import LanceDb
from phi.knowledge.text import TextKnowledgeBase

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
LANCEDB_URI = "tmp/lancedb_air_data"
LANCEDB_TABLE_NAME = "air_quality_docs"
AIR_TEXT_FILE_PATH = "./air.txt"

FORCE_RECREATE_KB = False


def create_knowledge_base(
    text_file_path: str,
    db_uri: str,
    table_name: str,
    embedder_model: str,
    force_recreate: bool = False) -> TextKnowledgeBase | None:
    """
    Creates or loads a TextKnowledgeBase.

    Args:
        text_file_path: Path to the source text file.
        db_uri: URI for LanceDB.
        table_name: Name of the table in LanceDB.
        embedder_model: Name of the sentence transformer model for embeddings.
        force_recreate: If True, forces recreation of the knowledge base.

    Returns:
        A TextKnowledgeBase instance or None if an error occurs.
    """
    
    if not os.path.exists(text_file_path):
        logging.error(f"Source text file not found: {text_file_path}")
        return None

    logging.info(f"Initializing knowledge base with source: {text_file_path}")
    logging.info(f"LanceDB URI: {db_uri}, Table: {table_name}")
    
    if force_recreate:
        logging.info("Forcing recreation of knowledge base.")

    try:
      
        embedder = SentenceTransformerEmbedder(model=embedder_model)
        vector_db = LanceDb(
            table_name=table_name,
            uri=db_uri,
            embedder=embedder,
        )

        knowledge_base = TextKnowledgeBase(
            path=text_file_path,
            vector_db=vector_db,
        )
        
        logging.info("Loading knowledge base into vector store...")
        
        knowledge_base.load(recreate=force_recreate)
        
        logging.info("Knowledge base loaded successfully.")
        
        return knowledge_base
      
    except Exception as e:
        logging.error(f"Failed to create or load knowledge base: {e}", exc_info=True)
        return None


def create_rag_agent(
    llm_model_id: str,
    knowledge_base_instance: Any,
    show_tool_calls: bool = True,
    use_markdown: bool = True) -> Agent | None:
    """
    Creates a RAG agent.

    Args:
        llm_model_id: The ID of the Groq LLM model.
        knowledge_base_instance: An instance of a loaded KnowledgeBase.
        show_tool_calls: Whether to show tool calls in output.
        use_markdown: Whether agent output should be in Markdown.

    Returns:
        An Agent instance or None if an error occurs.
    """
    
    logging.info(f"Creating RAG agent with model: {llm_model_id}")
    
    try:
      
        agent = Agent(
            llm=Groq(model=llm_model_id),
            knowledge_base=knowledge_base_instance,
            description="You are a helpful AI assistant. You answer questions based on the provided knowledge about air quality and related topics. If the information is not in the knowledge base, say so.",
            show_tool_calls=show_tool_calls,
            markdown=use_markdown,
            add_datetime_to_instructions=True,
        )
        
        logging.info("RAG Agent created successfully.")
        
        return agent
      
    except Exception as e:
        logging.error(f"Failed to create RAG agent: {e}", exc_info=True)
        return None


def ask_agent(agent_instance: Agent, question: str):
    """
    Asks a question to the agent and prints the streamed response.

    Args:
        agent_instance: The initialized Agent.
        question: The question to ask.
    """
    
    if not agent_instance:
        logging.error("Agent is not initialized. Cannot ask question.")
        return

    logging.info(f"Asking agent: '{question}'")
    print(f"\n--- Question --- \n{question}")
    print("\n--- Answer ---")
    
    try:
        agent_instance.print_response(question, stream=True)
        
        logging.info("Agent responded successfully.")
        
    except Exception as e:
      
        logging.error(f"Error during agent interaction: {e}", exc_info=True)
        print(f"\n[ERROR] An error occurred while getting the answer: {e}")



def main():
    """
    Main function to set up and run the RAG agent.
    """


    if not os.getenv("GROQ_API_KEY"):
        error_msg = "GROQ_API_KEY not found in environment variables. Please set it in your .env file."
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        
        return

    knowledge = create_knowledge_base(
        text_file_path=AIR_TEXT_FILE_PATH,
        db_uri=LANCEDB_URI,
        table_name=LANCEDB_TABLE_NAME,
        embedder_model=EMBEDDER_MODEL,
        force_recreate=FORCE_RECREATE_KB
    )

    if not knowledge:
        print("Failed to initialize knowledge base. Exiting.")
        return

    rag_agent = create_rag_agent(
        llm_model_id=GROQ_MODEL_ID,
        knowledge_base_instance=knowledge
    )

    if not rag_agent:
        print("Failed to initialize RAG agent. Exiting.")
        return

    questions = "Istanbul hava sıcaklığı kaç derece?"
    ask_agent(rag_agent, questions)


if __name__ == "__main__":
    main()