import ollama
import requests
from bs4 import BeautifulSoup
import trafilatura
import logging
import sys_msgs


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

OLLAMA_MODEL = 'gemma3:27b'
OLLAMA_REQUEST_TIMEOUT = 120
DUCKDUCKGO_MAX_RESULTS = 5
SEARCH_RETRY_LIMIT = 3
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'


conversation_history = [sys_msgs.asistant_msg]



def _call_ollama_chat(system_prompt_template: str, user_data: dict, model: str = OLLAMA_MODEL) -> str | None:
    """
    Helper function to call the Ollama chat API and extract content.
    Formats the system prompt with user_data.
    """
    
    try:
      
        system_message_content = system_prompt_template.format(**user_data)
        messages = [{'role': 'system', 'content': system_message_content}]
        
        if "user_prompt_for_llm" in user_data:
             messages.append({'role': 'user', 'content': user_data["user_prompt_for_llm"]})

        logging.debug(f"Calling Ollama with model {model}. Messages: {messages}")
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.3},
            #stream=False
            #timeout=OLLAMA_REQUEST_TIMEOUT
        )
        
        content = response['message']['content']
        
        logging.debug(f"Ollama response content: {content}")
        
        return content
      
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}", exc_info=True)
        return None


def _call_ollama_decide(system_prompt: str, last_user_message: dict, model: str = OLLAMA_MODEL) -> str | None:
    """Helper specifically for decision-making prompts (e.g. search_or_not, contains_data)."""
    
    try:
      
        messages=[
            {'role': 'system', 'content': system_prompt},
            last_user_message
        ]
        
        logging.debug(f"Calling Ollama (decide) with model {model}. Messages: {messages}")
        
        response = ollama.chat(model=model, messages=messages, options={"temperature": 0.1})
        content = response['message']['content']
        
        logging.debug(f"Ollama decision response: {content}")
        
        return content
      
    except Exception as e:
        logging.error(f"Error calling Ollama for decision: {e}", exc_info=True)
        return None



def should_search_web(conversation_history: str) -> bool:
    """Determines if a web search is needed based on the user's prompt."""
    
    logging.info("Deciding whether to search the web...")

    if not conversation_history or conversation_history[-1]['role'] != 'user':
        logging.warning("Cannot decide to search without a preceding user message in history.")
        return False
    
    last_user_message = conversation_history[-1]

    content = _call_ollama_decide(sys_msgs.search_or_not_msg, last_user_message)
    
    decision = content and 'true' in content.lower()
    
    logging.info(f"Decision to search: {decision}")
    
    return decision


def generate_search_query(last_user_prompt_content: str) -> str | None:
    """Generates a search query from the user's prompt."""
    
    logging.info("Generating search query...")
    
    user_data_for_llm = {"user_prompt_for_llm": f"CREATE A SEARCH QUERY FOR THIS PROMPT: \n{last_user_prompt_content}"}

    query = _call_ollama_chat(sys_msgs.query_msg, user_data_for_llm)

    if query:

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
            
        if query.startswith("'") and query.endswith("'"):
            query = query[1:-1]
            
        logging.info(f"Generated search query: {query}")
        
    else:
        logging.warning("Failed to generate search query.")
        
    return query


def perform_duckduckgo_search(query: str) -> list[dict]:
    """Performs a DuckDuckGo search and returns formatted results."""
    
    logging.info(f"Performing DuckDuckGo search for: {query}")
    
    headers = {'User-Agent': USER_AGENT}
    url = f"https://html.duckduckgo.com/html/?q={query}"
    
    try:
        response = requests.get(url=url, headers=headers, timeout=10)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"DuckDuckGo search request failed: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    
    for i, result_div in enumerate(soup.find_all('div', class_='result')):
        if i >= DUCKDUCKGO_MAX_RESULTS:
            break
        
        title_tag = result_div.find('a', class_='result__a')
        if not title_tag or not title_tag.get('href'):
            continue
        
        link = title_tag['href']
        title = title_tag.text.strip()
        
        snippet_tag = result_div.find('a', class_='result__snippet')
        snippet = snippet_tag.text.strip() if snippet_tag else 'No description available.'
        
        results.append({
            'id': i,
            'title': title,
            'link': link,
            'snippet': snippet
        })
    
    logging.info(f"Found {len(results)} search results.")
    logging.debug(f"Search results: {results}")
    
    return results

def select_best_search_result_id(search_results: list[dict], user_prompt_content: str, generated_query: str) -> int | None:
    """Asks the LLM to select the best search result ID."""
    
    if not search_results:
        logging.warning("No search results to select from.")
        return None

    logging.info("Selecting the best search result...")
    
    formatted_s_results = "\n".join([
        f"ID: {res['id']}\nTitle: {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}\n---"
        for res in search_results
    ])


    user_data = {
        "s_results": formatted_s_results,
        "user_prompt": user_prompt_content,
        "search_query": generated_query
    }

    for attempt in range(SEARCH_RETRY_LIMIT):
      
        logging.debug(f"Attempt {attempt + 1} to select best result.")
        
        content = _call_ollama_chat(sys_msgs.best_search_msg, user_data)
        if content:
            try:
              
                best_id = int(content)
                if 0 <= best_id < len(search_results):
                    logging.info(f"Selected best search result ID: {best_id}")
                    return best_id
                  
                else:
                    logging.warning(f"LLM returned out-of-bounds ID: {best_id}. Results count: {len(search_results)}")
            
            except ValueError:
                logging.warning(f"LLM did not return a valid integer ID: '{content}'")

    
    logging.error("Failed to select a best search result after multiple attempts.")
    return None


def scrape_webpage_content(url: str) -> str | None:
    """Scrapes the main content of a webpage using Trafilatura."""
    
    logging.info(f"Scraping webpage: {url}")
    
    try:
      
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logging.warning(f"Failed to download content from {url}.")
            return None
          
        content = trafilatura.extract(downloaded, include_formatting=False, include_links=False)
        
        if content:
            logging.info(f"Successfully scraped content (length: {len(content)}).")
            return content
          
        else:
            logging.warning(f"Trafilatura extracted no main content from {url}.")
            return None
          
    except Exception as e:
        logging.error(f"Error scraping webpage {url}: {e}", exc_info=True)
        return None


def is_content_relevant(page_text: str, user_prompt_content: str, generated_query: str) -> bool:
    """Asks the LLM to determine if the scraped content is relevant."""
    logging.info("Checking if scraped content is relevant...")

    max_text_len = 8000
    truncated_page_text = page_text[:max_text_len] + ("..." if len(page_text) > max_text_len else "")

    user_data = {
        "page_text": truncated_page_text,
        "user_prompt": user_prompt_content,
        "search_query": generated_query
    }
    
    content = _call_ollama_chat(sys_msgs.contains_data_msg, user_data)
    
    decision = content and 'true' in content.lower()
    
    logging.info(f"Content relevance decision: {decision}")
    
    return decision

def run_ai_search_pipeline(last_user_prompt_content: str) -> str | None:
    """Orchestrates the AI search pipeline."""
    
    logging.info("--- Starting AI Search Pipeline ---")
    
    generated_query = generate_search_query(last_user_prompt_content)
    if not generated_query:
        logging.warning("Pipeline aborted: Failed to generate search query.")
        return None

    search_results_list = perform_duckduckgo_search(generated_query)
    if not search_results_list:
        logging.warning("Pipeline aborted: No search results found.")
        return None
    
    available_results = list(search_results_list) 

    for _ in range(min(len(available_results), SEARCH_RETRY_LIMIT)):
      
        if not available_results:
            logging.info("No more search results to try.")
            break

        best_result_id = select_best_search_result_id(
            available_results,
            last_user_prompt_content,
            generated_query
        )
        
        if best_result_id is None:
            logging.warning("Could not select a best result from remaining items. Trying next if available, or aborting.")
            
            if not available_results:
              break
            
            selected_result_details = available_results.pop(0)
            
            logging.warning(f"Falling back to trying result ID 0 from remaining: {selected_result_details['link']}")
            
        else:
          
            try:
                selected_result_details = available_results.pop(best_result_id)
                logging.info(f"Attempting to use selected result: {selected_result_details['title']} - {selected_result_details['link']}")
            
            except IndexError:
                logging.error(f"Selected ID {best_result_id} is out of bounds for available_results (len {len(available_results)+1}). Taking first.")
                if not available_results: break
                selected_result_details = available_results.pop(0)


        page_url = selected_result_details['link']
        page_content = scrape_webpage_content(page_url)
        
        if page_content:
            
            if is_content_relevant(page_content, last_user_prompt_content, generated_query):
                logging.info(f"Relevant content found from: {page_url}")
                logging.info("--- AI Search Pipeline Completed Successfully ---")
                
                return page_content
            
            else:
                logging.info(f"Content from {page_url} deemed not relevant. Trying next result.")
        
        else:
            logging.info(f"No content scraped from {page_url} or scraping failed. Trying next result.")
            
    logging.warning("--- AI Search Pipeline Completed: No relevant context found after trying available results. ---")
    
    return None


def stream_and_record_assistant_response():
    """Streams the assistant's response and records it to conversation history."""
    
    global conversation_history
    
    logging.info("Streaming assistant's final response...")
    if not conversation_history:
        logging.error("Conversation history is empty, cannot generate response.")
        return

    try:
        
        response_stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=conversation_history,
            stream=True,
            options={"temperature": 0.7}
        )
        
        complete_response_content = ''
        
        print("\nASSISTANT:")
        
        for chunk in response_stream:
            
            token = chunk['message']['content']
            print(token, end='', flush=True)
            complete_response_content += token
            
        conversation_history.append({'role': 'assistant', 'content': complete_response_content})
        
        print('\n\n')
        logging.info("Assistant response streamed and recorded.")

    except Exception as e:
        logging.error(f"Error streaming assistant response: {e}", exc_info=True)
        print("\n[ERROR] Sorry, I encountered a problem while generating my response.")



def main():
    
    global conversation_history
    logging.info("Starting AI Assistant. Type 'quit' or 'exit' to end.")

    while True:
        
        try:
            
            user_input = input('USER: \n')
            if user_input.lower() in ['quit', 'exit']:
                logging.info("Exiting application.")
                break
            
            if not user_input.strip():
                continue


            conversation_history.append({'role': 'user', 'content': user_input})
            
            last_user_prompt_content = user_input 

            if should_search_web(last_user_prompt_content):
                
                logging.info("Web search is required.")
                
                retrieved_context = run_ai_search_pipeline(last_user_prompt_content)
                
                current_user_message_entry = conversation_history.pop()
                original_user_prompt_for_final_response = current_user_message_entry['content']

                if retrieved_context:
                    
                    prompt_with_context = (
                        f"Based on the following information: \n---BEGIN INFO---\n{retrieved_context}\n---END INFO---\n\n"
                        f"Please answer this question or address this request: \"{original_user_prompt_for_final_response}\""
                    )
                    
                    conversation_history.append({'role': 'user', 'content': prompt_with_context})
                    
                    logging.info("Added retrieved context to user prompt for final response.")
                    
                else:
                    
                    prompt_informing_failed_search = (
                        f"I tried to find information on the web to answer your request: \"{original_user_prompt_for_final_response}\", "
                        "but I couldn't find relevant information or the search failed. "
                        "Please answer based on your general knowledge, or state that you couldn't find the specific information."
                    )
                    
                    conversation_history.append({'role': 'user', 'content': prompt_informing_failed_search})
                    
                    logging.info("Informed LLM about failed search in the user prompt.")
                    
            else:
                logging.info("No web search required. Responding directly.")

            stream_and_record_assistant_response()

        except KeyboardInterrupt:
            logging.info("\nUser interrupted. Exiting application.")
            break
        
        except Exception as e:
            logging.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print("\n[CRITICAL ERROR] An unexpected error occurred. Please check logs. Restarting loop if possible.")


if __name__ == '__main__':
    main()