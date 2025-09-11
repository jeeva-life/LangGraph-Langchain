"""
LangGraph Introduction with LangSmith

This script demonstrates the basics of LangGraph and LangSmith integration
without Jupyter notebook environment issues.
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

def setup_environment():
    """Set up environment variables and load configuration."""
    load_dotenv(override=True)
    
    # Check for required API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("✓ OpenAI API key found")
    else:
        print(" No OpenAI API key found - LLM functionality will not work")
    
    # Check for optional API keys
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key:
        print(" Serper API key found")
    else:
        print(" No Serper API key found - Google search will use mock")
    
    # LangSmith setup
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    if langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        langchain_project = os.getenv("LANGCHAIN_PROJECT", "langgraph-intro")
        os.environ["LANGCHAIN_PROJECT"] = langchain_project
        print("✓ LangSmith tracing configured")
    else:
        print("ℹ No LangSmith tracking configured")
    
    return openai_api_key, serper_api_key

def test_serper_api(serper_api_key):
    """Test Google Serper API functionality."""
    print("\n Testing Google Serper API...")
    
    if serper_api_key:
        try:
            serper = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
            result = serper.run("what is the capital of france")
            print(f"✓ Search successful: {result}")
            return serper
        except Exception as e:
            print(f"✗ Error with Serper API: {e}")
            return None
    else:
        print(" No Serper API key - using mock search")
        
        class MockSerper:
            def run(self, query):
                return f"Mock search result for: {query}"
        
        serper = MockSerper()
        result = serper.run("what is the capital of france")
        print(f"Mock result: {result}")
        return serper

def create_langgraph_workflow(openai_api_key):
    """Create a simple LangGraph workflow."""
    if not openai_api_key:
        print(" Cannot create LangGraph workflow - no OpenAI API key")
        return None
    
    print("\n🔧 Creating LangGraph workflow...")
    
    try:
        # Define the state for our graph
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
        
        # Create the LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        def call_model(state: AgentState):
            """Call the LLM with the current messages."""
            messages = state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: AgentState):
            """Determine if we should continue the conversation."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Simple logic: continue if it's a human message
            if isinstance(last_message, HumanMessage):
                return "continue"
            else:
                return "end"
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "agent",
                "end": END
            }
        )
        
        # Compile the graph
        app = workflow.compile()
        
        print(" LangGraph workflow created successfully!")
        print("Graph structure:")
        print("  START -> agent -> (continue/end)")
        print("  continue -> agent")
        print("  end -> END")
        
        return app
        
    except Exception as e:
        print(f"✗ Error creating LangGraph workflow: {e}")
        return None

def test_langgraph_workflow(app):
    """Test the LangGraph workflow."""
    if not app:
        print("No workflow to test")
        return
    
    print("\nTesting LangGraph workflow...")
    
    try:
        # Test with a simple message
        initial_state = {
            "messages": [HumanMessage(content="Hello! What is LangGraph?")]
        }
        
        result = app.invoke(initial_state)
        print(" Workflow executed successfully!")
        print(f"Response: {result['messages'][-1].content}")
        
        # Test with a follow-up question
        print("\n Testing follow-up conversation...")
        follow_up_state = {
            "messages": [
                HumanMessage(content="Hello! What is LangGraph?"),
                AIMessage(content=result['messages'][-1].content),
                HumanMessage(content="Can you give me a simple example?")
            ]
        }
        
        follow_up_result = app.invoke(follow_up_state)
        print("✓ Follow-up conversation successful!")
        print(f"Response: {follow_up_result['messages'][-1].content}")
        
    except Exception as e:
        print(f" Error testing workflow: {e}")
        print("This might be due to:")
        print("  1. Missing OpenAI API key")
        print("  2. Network connectivity issues")
        print("  3. LangGraph configuration problems")

def demonstrate_advanced_features(app, serper):
    """Demonstrate advanced LangGraph features."""
    if not app:
        print(" No workflow available for advanced features")
        return
    
    print("\n Advanced Features Demo...")
    
    # Example of a more complex state
    class AdvancedState(TypedDict):
        messages: Annotated[list, add_messages]
        search_results: list
        user_preferences: dict
    
    print("Advanced state structure:")
    print("  - messages: Conversation history")
    print("  - search_results: External search data")
    print("  - user_preferences: User-specific settings")
    
    # Example of conditional routing
    print("\nConditional routing examples:")
    print("  - Route to search if user asks for current information")
    print("  - Route to memory if user references past conversation")
    print("  - Route to tools if user needs specific functionality")

def main():
    """Main function to run the LangGraph introduction demo."""
    print(" LangGraph Introduction with LangSmith")
    print("=" * 50)
    
    # Setup environment
    openai_api_key, serper_api_key = setup_environment()
    
    # Test Serper API
    serper = test_serper_api(serper_api_key)
    
    # Create LangGraph workflow
    app = create_langgraph_workflow(openai_api_key)
    
    # Test the workflow
    test_langgraph_workflow(app)
    
    # Demonstrate advanced features
    demonstrate_advanced_features(app, serper)
    
    # Summary
    print("\n Summary:")
    print(f"  OpenAI API: {'✓ Available' if openai_api_key else '✗ Not available'}")
    print(f"  Serper API: {'✓ Available' if serper_api_key else '✗ Not available'}")
    print(f"  LangGraph: {'✓ Working' if app else '✗ Not available'}")
    
    if app:
        print("\n Demo completed successfully!")
        print("You can now build more complex LangGraph applications.")
    else:
        print("\n Demo completed with issues.")
        print("Please check your API keys and try again.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
