import logging
import os
from dotenv import load_dotenv

from phi.agent import Agent
from phi.llm.groq import Groq
from phi.tools.googlesearch import GoogleSearch


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
NUM_NEWS_ITEMS_TO_FETCH = 2


def create_news_agent() -> Agent | None:
    """
    Creates and configures the news search agent.
    Returns the Agent instance or None if creation fails.
    """

    google_search_tool = GoogleSearch()

    agent_instructions = [
        f"You are a web search agent specializing in finding the latest news.",
        f"Given a topic by the user, your primary goal is to respond with the {NUM_NEWS_ITEMS_TO_FETCH} latest and most relevant news items about that topic.",
        "1. Use the GoogleSearch tool to find news articles related to the provided topic. Explicitly aim to find recent publications.",
        f"2. From the search results, carefully select the top {NUM_NEWS_ITEMS_TO_FETCH} unique and most recent news items. Ensure the items are distinct and not just rehashes of the same story from different minor outlets.",
        "3. For each selected news item, provide a concise summary, the source (publication name), and the direct URL.",
        "4. If the GoogleSearch tool fails to execute or returns no relevant results, clearly state that you were unable to find news on the topic and briefly explain (e.g., 'No recent news found' or 'Search tool error').",
        f"5. If you find fewer than {NUM_NEWS_ITEMS_TO_FETCH} distinct and relevant news items (e.g., only 1), provide information for those you found and explicitly state that fewer items were available.",
        "6. All searches and responses must be in English.",
        "7. Present the final output in Markdown format, clearly listing each news item.",
    ]

    try:
        
        agent = Agent(
            llm=Groq(model=GROQ_MODEL_ID),
            tools=[google_search_tool],
            description="You are a web search agent that helps users find the latest news information.",
            instructions=agent_instructions,
            show_tool_calls=True,
            markdown=True,
            add_datetime_to_instructions=True,
        )
        
        logging.info(f"News agent created successfully with model: {GROQ_MODEL_ID}")
        
        return agent
    
    except Exception as e:
        logging.error(f"Failed to create news agent: {e}")
        return None


def fetch_and_display_news(topic: str, agent: Agent):
    """
    Uses the provided agent to fetch and display news for the given topic.
    Streams the response to the console.
    """
    
    if not agent:
        logging.error("Agent not initialized. Cannot fetch news.")
        print("[ERROR] Agent not available. Please check logs.")
        return

    logging.info(f"Attempting to fetch news for topic: '{topic}'")
    
    try:
        print(f"\n--- Latest News on: {topic} ---\n")
        
        agent.print_response(topic, stream=True)
        
        logging.info(f"Successfully completed news request for topic: '{topic}'")
        
    except Exception as e:
        logging.error(f"An error occurred while fetching news for '{topic}': {e}")
        print(f"\n[ERROR] Could not complete the news fetching for '{topic}'. Reason: {e}")
        print("Please check your API keys (Groq, Google) and model configurations.")



def main():
    """
    Main function to run the news search agent.
    """
    
    if not os.getenv("GROQ_API_KEY"):
        logging.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        print("Error: GROQ_API_KEY is not set. Please create a .env file with GROQ_API_KEY='your_key'.")
        return

    news_agent = create_news_agent()
    
    if not news_agent:
        print("Exiting due to agent creation failure.")
        return

    topic_to_search = "latest developments in large language models (LLMs)"
    fetch_and_display_news(topic_to_search, news_agent)

if __name__ == "__main__":
    main()