import logging
import os
from dotenv import load_dotenv

from phi.agent import Agent
from phi.llm.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

def create_research_agent() -> Agent:
    """
    Creates and configures the NYT researcher agent.
    """

    duckduckgo_tool = DuckDuckGo()
    newspaper_tool = Newspaper4k()

    agent_instructions = [
        "You are a senior NYT researcher. Your task is to write an article on a given topic.",
        "1. For the given topic, use the DuckDuckGo tool to search for the top 1 most relevant and authoritative link. Prioritize reputable news sources or academic publications.",
        "2. If DuckDuckGo returns no results or no suitable link, clearly state that you couldn't find a primary source and explain that you cannot proceed with writing the article based on web research.",
        "3. If a link is found, use the Newspaper4k tool to read the URL and extract the full article text. Focus on the main content.",
        "4. If Newspaper4k fails to extract the text (e.g., due to a paywall, non-article page, network error, or if the tool reports an error), clearly state this limitation. If possible, try to explain why it might have failed (e.g., 'The page might be behind a paywall or is not a standard article format.').",
        "5. If the article text is successfully extracted, analyze its content thoroughly. Identify key facts, arguments, perspectives, and any notable quotes.",
        "6. Based *solely* on the information from the successfully extracted article, prepare a comprehensive, well-structured, and engaging NYT-worthy article. Your tone should be objective and informative.",
        "7. If you were unable to extract sufficient information from the web (either no link found or article unreadable), clearly state this and explain that a comprehensive article cannot be produced.",
        "8. Ensure your final output is in Markdown format.",
    ]

    try:
        
        agent = Agent(
            llm=Groq(model=GROQ_MODEL_ID),
            tools=[duckduckgo_tool, newspaper_tool],
            description="""You are a senior NYT researcher tasked with writing an in-depth article 
                            on a specific topic by researching and analyzing a key web source.""",
            instructions=agent_instructions,
            markdown=True,     
            show_tool_calls=True, 
            add_datetime_to_instructions=True,
        )
        
        logging.info(f"Agent created successfully with model: {GROQ_MODEL_ID}")
        return agent
    
    except Exception as e:
        logging.error(f"Failed to create agent: {e}")
        raise



def generate_article_for_topic(topic: str, agent: Agent):
    """
    Uses the provided agent to generate an article for the given topic.
    Streams the response to the console.
    """
    
    if not agent:
        logging.error("Agent not initialized. Cannot generate article.")
        return

    logging.info(f"Attempting to generate an article for the topic: '{topic}'")
    
    try:
        print(f"\n--- NYT Article on: {topic} ---\n")
        agent.print_response(topic, stream=True)
        logging.info(f"Successfully completed request for topic: '{topic}'")
        
    except Exception as e:
        logging.error(f"An error occurred while generating the article for '{topic}': {e}")
        print(f"\n[ERROR] Could not complete the article generation for '{topic}'. Reason: {e}")
        print("Please check the logs and ensure your API keys and model configurations are correct.")



def main():
    """
    Main function to run the research agent.
    """

    if not os.getenv("GROQ_API_KEY"):
        logging.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        print("Error: GROQ_API_KEY is not set. Please create a .env file with GROQ_API_KEY='your_key'.")
        return

    try:
        research_agent = create_research_agent()
        
        topic_to_research = "Simulation Theory"
        generate_article_for_topic(topic_to_research, research_agent)

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")
        print(f"A critical error prevented the script from running: {e}")


if __name__ == "__main__":
    main()