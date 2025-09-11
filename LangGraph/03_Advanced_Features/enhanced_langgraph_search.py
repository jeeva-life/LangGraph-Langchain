"""
Enhanced LangGraph with Search Integration

This script demonstrates advanced LangGraph features with multiple search options
and proper error handling for SerperDevTool and other search APIs.
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import SerperDevTool
from langchain_community.utilities import GoogleSearchAPIWrapper, GoogleSerperAPIWrapper

def setup_environment():
    """Set up environment variables and load configuration."""
    load_dotenv(override=True)
    
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    
    print("🔧 Environment Setup:")
    print(f"  OpenAI API: {'✓' if openai_api_key else '✗'}")
    print(f"  Serper API: {'✓' if serper_api_key else '✗'}")
    print(f"  Google Search: {'✓' if google_api_key and google_cse_id else '✗'}")
    
    return openai_api_key, serper_api_key, google_api_key, google_cse_id

def create_search_tool(serper_api_key, google_api_key, google_cse_id):
    """Create the best available search tool."""
    print("\n🔍 Setting up Search Tool...")
    
    # Try SerperDevTool first
    if serper_api_key:
        try:
            serper_tool = SerperDevTool(serper_api_key=serper_api_key)
            result = serper_tool.run("test query")
            print("✓ SerperDevTool working")
            return serper_tool
        except Exception as e:
            print(f"✗ SerperDevTool failed: {e}")
    
    # Try Google Search API
    if google_api_key and google_cse_id:
        try:
            google_search = GoogleSearchAPIWrapper(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id
            )
            result = google_search.run("test query")
            print("✓ Google Search API working")
            return google_search
        except Exception as e:
            print(f"✗ Google Search API failed: {e}")
    
    # Try Google Serper API Wrapper
    if serper_api_key:
        try:
            serper_wrapper = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
            result = serper_wrapper.run("test query")
            print("✓ Google Serper API Wrapper working")
            return serper_wrapper
        except Exception as e:
            print(f"✗ Google Serper API Wrapper failed: {e}")
    
    # Fallback to mock search
    print("⚠️ No working search API found, using mock search")
    return create_mock_search_tool()

def create_mock_search_tool():
    """Create a mock search tool for demonstration."""
    class MockSearchTool:
        def run(self, query):
            mock_results = {
                "test query": "Mock search result for test query",
                "what is the capital of france": "Paris is the capital of France.",
                "what is langchain": "LangChain is a framework for developing applications powered by language models.",
                "what is langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
                "weather": "Mock weather information: Sunny, 22°C",
                "machine learning": "Machine learning is a subset of artificial intelligence that focuses on algorithms."
            }
            return mock_results.get(query.lower(), f"Mock search result for: {query}")
    
    return MockSearchTool()

def create_enhanced_langgraph_workflow(llm, search_tool):
    """Create an enhanced LangGraph workflow with search capabilities."""
    print("\n🔧 Creating Enhanced LangGraph Workflow...")
    
    # Define enhanced state
    class SearchAgentState(TypedDict):
        messages: Annotated[list, add_messages]
        search_results: list
        needs_search: bool
        search_count: int
    
    # Create search tool wrapper
    @tool
    def search_tool_wrapper(query: str) -> str:
        """Search for information using the available search tool."""
        try:
            return search_tool.run(query)
        except Exception as e:
            return f"Search error: {e}"
    
    def should_search(state: SearchAgentState):
        """Determine if we need to search for information."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Enhanced keyword detection
        search_keywords = [
            "search", "find", "look up", "what is", "who is", "when", "where", "how",
            "weather", "news", "current", "latest", "recent", "today", "now"
        ]
        
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            return any(keyword in content for keyword in search_keywords)
        return False
    
    def search_node(state: SearchAgentState):
        """Perform search and update state."""
        messages = state["messages"]
        last_message = messages[-1]
        search_count = state.get("search_count", 0)
        
        if isinstance(last_message, HumanMessage):
            search_query = last_message.content
            search_result = search_tool_wrapper.run(search_query)
            
            return {
                "search_results": [search_result],
                "needs_search": False,
                "search_count": search_count + 1
            }
        
        return {"needs_search": False}
    
    def call_model_with_search(state: SearchAgentState):
        """Call the LLM with search context."""
        messages = state["messages"]
        search_results = state.get("search_results", [])
        search_count = state.get("search_count", 0)
        
        # Add search context to the conversation
        if search_results:
            context_message = f"Search results: {search_results[0]}"
            messages.append(HumanMessage(content=context_message))
        
        # Add search count context
        if search_count > 0:
            context_message = f"Note: {search_count} search(es) performed in this conversation."
            messages.append(HumanMessage(content=context_message))
        
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def should_continue_enhanced(state: SearchAgentState):
        """Enhanced continuation logic."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage):
            return "search" if should_search(state) else "agent"
        else:
            return "end"
    
    # Create enhanced workflow
    enhanced_workflow = StateGraph(SearchAgentState)
    
    # Add nodes
    enhanced_workflow.add_node("search", search_node)
    enhanced_workflow.add_node("agent", call_model_with_search)
    
    # Add edges
    enhanced_workflow.add_edge(START, "agent")
    enhanced_workflow.add_conditional_edges(
        "agent",
        should_continue_enhanced,
        {
            "search": "search",
            "agent": "agent", 
            "end": END
        }
    )
    enhanced_workflow.add_edge("search", "agent")
    
    # Compile the enhanced workflow
    enhanced_app = enhanced_workflow.compile()
    
    print("✓ Enhanced LangGraph workflow created!")
    print("Features:")
    print("  - Automatic search detection")
    print("  - Search result integration")
    print("  - Search count tracking")
    print("  - Enhanced conversation flow")
    
    return enhanced_app

def test_enhanced_workflow(app):
    """Test the enhanced LangGraph workflow."""
    if not app:
        print("⚠️ No workflow to test")
        return
    
    print("\n🧪 Testing Enhanced LangGraph Workflow...")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        "Hello! What is LangGraph?",
        "Search for information about the capital of France",
        "What is the weather like today?",
        "Tell me about machine learning",
        "Find recent news about artificial intelligence"
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_query}")
        print("-" * 40)
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=test_query)],
                "search_results": [],
                "needs_search": False,
                "search_count": 0
            }
            
            result = app.invoke(initial_state)
            response = result['messages'][-1].content
            search_count = result.get('search_count', 0)
            
            print(f"✅ Response: {response[:150]}...")
            
            if search_count > 0:
                print(f"🔍 Search performed: {search_count} search(es)")
                if result.get('search_results'):
                    print(f"📊 Search results: {len(result['search_results'])} items")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced workflow testing completed!")

def demonstrate_workflow_visualization(app):
    """Demonstrate workflow visualization capabilities."""
    if not app:
        print("⚠️ No workflow to visualize")
        return
    
    print("\n📊 Workflow Visualization:")
    print("=" * 40)
    print("Graph Structure:")
    print("  START")
    print("    ↓")
    print("  agent (LLM)")
    print("    ↓")
    print("  [conditional]")
    print("    ├─ search → agent")
    print("    ├─ agent → agent")
    print("    └─ end → END")
    print("\nFlow Logic:")
    print("  1. User sends message")
    print("  2. Agent processes message")
    print("  3. If search keywords detected → search")
    print("  4. If no search needed → continue conversation")
    print("  5. Search results integrated into context")
    print("  6. Agent responds with enhanced context")

def main():
    """Main function to run the enhanced LangGraph demo."""
    print("🚀 Enhanced LangGraph with Search Integration")
    print("=" * 60)
    
    # Setup environment
    openai_api_key, serper_api_key, google_api_key, google_cse_id = setup_environment()
    
    if not openai_api_key:
        print("❌ OpenAI API key required for this demo")
        return
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create search tool
    search_tool = create_search_tool(serper_api_key, google_api_key, google_cse_id)
    
    # Create enhanced workflow
    enhanced_app = create_enhanced_langgraph_workflow(llm, search_tool)
    
    # Test the workflow
    test_enhanced_workflow(enhanced_app)
    
    # Demonstrate visualization
    demonstrate_workflow_visualization(enhanced_app)
    
    print("\n🎯 Summary:")
    print(f"  Search Tool: {type(search_tool).__name__}")
    print(f"  Workflow: {'✓ Working' if enhanced_app else '✗ Failed'}")
    print(f"  Features: Search integration, error handling, mock fallback")
    
    print("\n✅ Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
