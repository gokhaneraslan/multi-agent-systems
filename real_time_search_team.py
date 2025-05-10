import logging
import os
from dotenv import load_dotenv

from phi.agent import Agent
from phi.llm.groq import Groq
from phi.tools.googlesearch import GoogleSearch
from phi.tools.crawl4ai_tools import Crawl4aiTools


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
CRAWL4AI_MAX_LENGTH = 2000
NUM_LINKS_TO_FETCH = 3


def create_web_searcher_agent(llm_model_id: str) -> Agent | None:
    
    """Creates the Web Searcher agent."""
    logging.info("Creating Web Searcher agent...")
    
    try:
        
        agent = Agent(
            llm=Groq(model=llm_model_id),
            name="WebSearcher",
            description="You are a specialized web search assistant. Your task is to find relevant URLs for a given query. Focus on providing diverse and high-quality links.",
            tools=[GoogleSearch(num_results=NUM_LINKS_TO_FETCH + 2)],
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )
        
        logging.info("Web Searcher agent created successfully.")
        
        return agent
    
    except Exception as e:
        
        logging.error(f"Failed to create Web Searcher agent: {e}", exc_info=True)
        return None


def create_web_scraper_agent(llm_model_id: str) -> Agent | None:
    """Creates the Web Scraper agent."""
    
    logging.info("Creating Web Scraper agent...")
    
    if not os.getenv("CRAWL4AI_API_KEY"):
        logging.error("CRAWL4AI_API_KEY not found in environment variables for Web Scraper.")
        return None
    try:
        
        agent = Agent(
            llm=Groq(model=llm_model_id),
            name="WebScraper",
            description="You are a specialized web scraping assistant. Your task is to read and extract the main textual content from a list of provided URLs.",
            tools=[Crawl4aiTools(max_length=CRAWL4AI_MAX_LENGTH, clean_html=True, use_semantic_extractor=True)],
            show_tool_calls=True,
        )
        
        logging.info("Web Scraper agent created successfully.")
        
        return agent
    
    except Exception as e:
        logging.error(f"Failed to create Web Scraper agent: {e}", exc_info=True)
        return None

def create_agent_team(
    llm_model_id: str,
    searcher: Agent,
    scraper: Agent) -> Agent | None:
    """Creates the Agent Team orchestrator."""
    
    logging.info("Creating Agent Team...")
    
    if not searcher or not scraper:
        logging.error("Cannot create agent team due to missing sub-agents.")
        return None

    team_instructions = [
        "You are the lead agent of a web research team. Your goal is to provide a comprehensive summary for a user's query.",
        f"1. Receive the user's query. You MUST pass this query to the `WebSearcher` agent.",
        f"2. Instruct the `WebSearcher` to search for the query and return {NUM_LINKS_TO_FETCH} unique and relevant URLs. Emphasize finding breaking news or very recent information if the query implies it.",
        "3. If the `WebSearcher` fails to return any URLs or returns fewer than expected, acknowledge this in your final summary and explain the limitation.",
        f"4. Once you have the URLs from `WebSearcher`, you MUST pass these URLs to the `WebScraper` agent.",
        "5. Instruct the `WebScraper` to read the content from each of the provided URLs.",
        "6. If the `WebScraper` fails to read content from some or all URLs (e.g., due to errors, paywalls, or non-text content), acknowledge this. Your summary should be based on the content successfully scraped.",
        "7. After receiving the scraped text from `WebScraper` (or an indication of failure), analyze all the gathered information.",
        "8. Finally, provide a thoughtful, engaging, and well-structured summary of the findings in Markdown format. If no information could be gathered, clearly state that.",
        "IMPORTANT: You must explicitly call the `WebSearcher` first, then the `WebScraper` with the results from the searcher. Do not try to use their tools directly yourself."
    ]

    try:
        
        agent_team = Agent(
            llm=Groq(model=llm_model_id),
            name="ResearchTeamLead",
            description="You are the leader of a web research team, coordinating a searcher and a scraper to answer user queries.",
            team=[searcher, scraper],
            instructions=team_instructions,
            show_tool_calls=True,
            markdown=True,
            add_datetime_to_instructions=True,
        )
        
        logging.info("Agent Team created successfully.")
        
        return agent_team
    
    except Exception as e:
        logging.error(f"Failed to create Agent Team: {e}", exc_info=True)
        return None



def run_team_task(team: Agent, query: str):
    """Runs the agent team to process a query and prints the response."""
    
    if not team:
        logging.error("Agent team is not initialized. Cannot run task.")
        print("[ERROR] Agent team not available. Please check logs.")
        return

    logging.info(f"Running agent team for query: '{query}'")
    
    print(f"\n--- Team Task: {query} ---\n")
    
    try:
        team.print_response(query, stream=True)
        
        logging.info(f"Agent team completed task for query: '{query}'")
        
    except Exception as e:
        logging.error(f"An error occurred while the agent team was processing '{query}': {e}", exc_info=True)
        print(f"\n[ERROR] Could not complete the task for '{query}'. Reason: {e}")
        print("Please check API keys (Groq, Google, Crawl4AI) and configurations.")


def main():
    """
    Main function to set up and run the multi-agent system.
    """

    if not os.getenv("GROQ_API_KEY"):
        error_msg = "GROQ_API_KEY not found in environment variables. Please set it in your .env file."
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        
        return


    web_searcher = create_web_searcher_agent(GROQ_MODEL_ID)
    web_scraper = create_web_scraper_agent(GROQ_MODEL_ID)

    if not web_searcher or not web_scraper:
        print("Failed to create one or more sub-agents. Exiting.")
        return

    agent_team = create_agent_team(
        llm_model_id=GROQ_MODEL_ID,
        searcher=web_searcher,
        scraper=web_scraper
    )

    if not agent_team:
        print("Failed to create the agent team. Exiting.")
        return

    query = "What are the latest significant developments in AI ethics this month?"
    run_team_task(agent_team, query)


if __name__ == "__main__":
    main()