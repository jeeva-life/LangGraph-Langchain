"""
Simple Playwright Web Browsing Example with LangGraph

This is a simplified, working example that avoids the asyncio issues
by using the synchronous Playwright API and a basic LangGraph setup.
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

def setup_environment():
    """Set up environment variables."""
    load_dotenv(override=True)
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print(" No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")
        return False
    
    print(" Environment setup complete")
    return True

def test_playwright_basic():
    """Test basic Playwright functionality."""
    try:
        from playwright.sync_api import sync_playwright
        
        print(" Testing basic Playwright functionality...")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to a simple page
            page.goto("https://httpbin.org/html")
            
            # Get page title
            title = page.title()
            print(f" Page title: {title}")
            
            # Get some text content
            content = page.text_content("h1")
            print(f" Page content: {content}")
            
            browser.close()
            print(" Basic Playwright test successful!")
            return True
            
    except Exception as e:
        print(f" Playwright test failed: {e}")
        return False

def create_simple_web_agent():
    """Create a simple web browsing agent using LangGraph."""
    print(" Creating simple web browsing agent...")
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # Define the state
    class WebAgentState(TypedDict):
        messages: Annotated[list, add_messages]
        current_url: str
        page_content: str
    
    # Create the graph
    web_graph = StateGraph(WebAgentState)
    
    def web_agent(state: WebAgentState):
        """Main web agent node."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Simple keyword-based routing
        if "navigate" in last_message.lower() or "go to" in last_message.lower():
            return handle_navigation(state)
        elif "search" in last_message.lower():
            return handle_search(state)
        elif "extract" in last_message.lower() or "get content" in last_message.lower():
            return handle_extract_content(state)
        else:
            return handle_general_query(state)
    
    def handle_navigation(state: WebAgentState):
        """Handle navigation requests."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        try:
            from playwright.sync_api import sync_playwright
            
            # Extract URL from message
            url = None
            if "http" in last_message:
                import re
                urls = re.findall(r'https?://[^\s]+', last_message)
                url = urls[0] if urls else "https://www.google.com"
            else:
                url = "https://www.google.com"
            
            # Navigate to URL
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                
                title = page.title()
                content = page.text_content("body")[:500] + "..." if len(page.text_content("body")) > 500 else page.text_content("body")
                
                browser.close()
            
            response = f"I navigated to {url}. Page title: {title}. Content preview: {content}"
            
        except Exception as e:
            response = f"Error navigating: {e}"
        
        return {
            "messages": [AIMessage(content=response)],
            "current_url": url if 'url' in locals() else "",
            "page_content": content if 'content' in locals() else ""
        }
    
    def handle_search(state: WebAgentState):
        """Handle search requests."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        try:
            from playwright.sync_api import sync_playwright
            
            # Extract search term
            search_term = last_message.replace("search for", "").replace("search", "").strip()
            if not search_term:
                search_term = "LangGraph tutorial"
            
            # Perform search on Google
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"https://www.google.com/search?q={search_term}")
                
                # Get search results
                results = page.query_selector_all("h3")
                search_results = [result.text_content() for result in results[:3]]
                
                browser.close()
            
            response = f"I searched for '{search_term}' and found these results:\n" + "\n".join([f"- {result}" for result in search_results])
            
        except Exception as e:
            response = f"Error searching: {e}"
        
        return {
            "messages": [AIMessage(content=response)],
            "current_url": "",
            "page_content": ""
        }
    
    def handle_extract_content(state: WebAgentState):
        """Handle content extraction requests."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        try:
            from playwright.sync_api import sync_playwright
            
            # Use a default URL if none specified
            url = "https://www.wikipedia.org/wiki/Artificial_intelligence"
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                
                # Extract main content
                content = page.text_content("main") or page.text_content("body")
                content = content[:1000] + "..." if len(content) > 1000 else content
                
                browser.close()
            
            response = f"I extracted content from {url}:\n\n{content}"
            
        except Exception as e:
            response = f"Error extracting content: {e}"
        
        return {
            "messages": [AIMessage(content=response)],
            "current_url": url if 'url' in locals() else "",
            "page_content": content if 'content' in locals() else ""
        }
    
    def handle_general_query(state: WebAgentState):
        """Handle general queries using LLM."""
        messages = state["messages"]
        response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "current_url": state.get("current_url", ""),
            "page_content": state.get("page_content", "")
        }
    
    # Add nodes
    web_graph.add_node("agent", web_agent)
    
    # Add edges
    web_graph.add_edge(START, "agent")
    web_graph.add_edge("agent", END)
    
    # Compile the graph
    web_app = web_graph.compile()
    
    print(" Simple web browsing agent created!")
    return web_app

def test_web_agent(web_app):
    """Test the web browsing agent."""
    print("\n Testing Web Browsing Agent...")
    print("=" * 50)
    
    test_queries = [
        "Navigate to https://www.google.com",
        "Search for Python programming tutorials",
        "Extract content from a Wikipedia page about AI",
        "What is LangGraph?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "current_url": "",
                "page_content": ""
            }
            
            # Run the agent
            result = web_app.invoke(initial_state)
            
            # Display results
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f" Response: {last_message.content}")
            else:
                print(" No response generated")
                
        except Exception as e:
            print(f" Error: {e}")
    
    print("\n Web agent tests completed!")

def interactive_mode(web_app):
    """Start interactive mode."""
    print("\n Interactive Web Browsing Mode")
    print("=" * 40)
    print("Type 'quit' to exit")
    print("Example commands:")
    print("  - 'Navigate to https://www.google.com'")
    print("  - 'Search for machine learning tutorials'")
    print("  - 'Extract content from a Wikipedia page'")
    print("  - 'What is artificial intelligence?'")
    
    while True:
        try:
            user_input = input("\n🌐 Your command: ").strip()
            
            if user_input.lower() == 'quit':
                print(" Goodbye!")
                break
            elif not user_input:
                continue
            
            # Create state and run agent
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "current_url": "",
                "page_content": ""
            }
            
            result = web_app.invoke(initial_state)
            
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f" Response: {last_message.content}")
            else:
                print(" No response generated")
            
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")

def main():
    """Main function."""
    print(" Simple Playwright Web Browsing with LangGraph")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Test basic Playwright functionality
    if not test_playwright_basic():
        print(" Playwright is not working properly. Please check installation.")
        return
    
    # Create web browsing agent
    web_app = create_simple_web_agent()
    
    if not web_app:
        print(" Failed to create web browsing agent")
        return
    
    # Test the agent
    test_web_agent(web_app)
    
    # Ask if user wants interactive mode
    try:
        interactive = input("\n Would you like to start interactive mode? (y/n): ").strip().lower()
        if interactive in ['y', 'yes']:
            interactive_mode(web_app)
    except KeyboardInterrupt:
        print("\n Goodbye!")
    
    print("\n Simple Playwright Web Browsing Example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
