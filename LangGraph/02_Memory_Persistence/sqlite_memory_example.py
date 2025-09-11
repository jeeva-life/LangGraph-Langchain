"""
LangGraph SQLite Memory Persistence Example

This script demonstrates how to use SQLite as a persistent memory store
for LangGraph applications with comprehensive examples and error handling.
"""

import sqlite3
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

def setup_environment():
    """Set up environment variables and database connection."""
    load_dotenv(override=True)
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    print("🔧 Environment Setup:")
    print(f"  OpenAI API: {'✓' if openai_api_key else '✗'}")
    
    return openai_api_key

def create_database_connection(db_path="memory.db"):
    """Create SQLite database connection and checkpoint saver."""
    try:
        # Create database connection
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create SQLite checkpoint saver
        sql_memory = SqliteSaver(conn)
        
        print(f"✅ SQLite checkpoint memory initialized!")
        print(f"  Database path: {os.path.abspath(db_path)}")
        print(f"  Connection status: Connected")
        
        return conn, sql_memory
        
    except Exception as e:
        print(f"❌ Error initializing SQLite memory: {e}")
        return None, None

def create_llm(openai_api_key):
    """Create LLM instance with fallback to mock."""
    if not openai_api_key:
        print("⚠️ No OpenAI API key found. Using mock LLM for demonstration")
        
        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content=f"Mock response to: {messages[-1].content}")
        
        return MockLLM()
    else:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

def create_langgraph_workflow(llm, sql_memory):
    """Create LangGraph workflow with SQLite memory."""
    print("\n🔧 Creating LangGraph Workflow...")
    
    # Define the state for our graph
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        user_name: str
        conversation_count: int

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

    # Compile the graph with SQLite memory
    app = workflow.compile(checkpointer=sql_memory)
    
    print("✅ LangGraph workflow with SQLite memory created!")
    print("Features:")
    print("  - Persistent conversation memory")
    print("  - State checkpointing")
    print("  - Thread-based conversations")
    
    return app

def test_memory_persistence(app):
    """Test the persistent memory functionality."""
    print("\n🧪 Testing SQLite Memory Persistence...")
    print("=" * 60)
    
    # Test 1: Create a new conversation thread
    print("\n📝 Test 1: New conversation thread")
    thread_id = "test_conversation_1"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Start a conversation
        result1 = app.invoke(
            {"messages": [HumanMessage(content="Hello! My name is Alice.")]},
            config=config
        )
        print(f"Response: {result1['messages'][-1].content}")

        # Continue the conversation
        result2 = app.invoke(
            {"messages": [HumanMessage(content="What's my name?")]},
            config=config
        )
        print(f"Response: {result2['messages'][-1].content}")

    except Exception as e:
        print(f"❌ Error in Test 1: {e}")

    # Test 2: Create another conversation thread
    print("\n📝 Test 2: Another conversation thread")
    thread_id_2 = "test_conversation_2"
    config2 = {"configurable": {"thread_id": thread_id_2}}

    try:
        result3 = app.invoke(
            {"messages": [HumanMessage(content="Hello! My name is Bob.")]},
            config=config2
        )
        print(f"Response: {result3['messages'][-1].content}")

    except Exception as e:
        print(f"❌ Error in Test 2: {e}")

    # Test 3: Go back to first thread (should remember Alice)
    print("\n📝 Test 3: Back to first thread (should remember Alice)")
    try:
        result4 = app.invoke(
            {"messages": [HumanMessage(content="What's my name again?")]},
            config=config
        )
        print(f"Response: {result4['messages'][-1].content}")

    except Exception as e:
        print(f"❌ Error in Test 3: {e}")

    print("\n✅ Memory persistence test completed!")
    print("Each thread maintains its own conversation history.")

def demonstrate_advanced_features(app, conn, db_path):
    """Demonstrate advanced memory features."""
    print("\n🔧 Advanced SQLite Memory Features...")
    print("=" * 60)
    
    # List all conversation threads
    print("\n📋 Available conversation threads:")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = cursor.fetchall()
        
        if threads:
            for i, (thread_id,) in enumerate(threads, 1):
                print(f"  {i}. {thread_id}")
        else:
            print("  No conversation threads found")
            
    except Exception as e:
        print(f"  Error retrieving threads: {e}")

    # Get conversation history for a specific thread
    print(f"\n📜 Conversation history for thread: test_conversation_1")
    try:
        config = {"configurable": {"thread_id": "test_conversation_1"}}
        state = app.get_state(config)
        if state and state.values:
            messages = state.values.get("messages", [])
            print(f"  Total messages: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"    {i}. [{msg_type}]: {content}")
        else:
            print("  No conversation history found")
            
    except Exception as e:
        print(f"  Error retrieving history: {e}")

    # Database statistics
    print(f"\n📊 Database Statistics:")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]
        print(f"  Total checkpoints: {checkpoint_count}")
        
        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        thread_count = cursor.fetchone()[0]
        print(f"  Total threads: {thread_count}")
        
        # Get database file size
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path)
            print(f"  Database size: {db_size} bytes ({db_size/1024:.2f} KB)")
        
    except Exception as e:
        print(f"  Error getting statistics: {e}")

    print("\n✅ Advanced memory features demonstrated!")

def interactive_conversation(app):
    """Start an interactive conversation session."""
    print("\n💬 Interactive Conversation Mode")
    print("=" * 40)
    print("Type 'quit' to exit, 'new' to start a new thread")
    print("Type 'threads' to see available threads")
    print("Type 'history' to see current thread history")
    
    current_thread = "interactive_session"
    config = {"configurable": {"thread_id": current_thread}}
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'new':
                current_thread = f"session_{len(os.listdir('.'))}"
                config = {"configurable": {"thread_id": current_thread}}
                print(f"🆕 New thread started: {current_thread}")
                continue
            elif user_input.lower() == 'threads':
                # List threads
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
                threads = cursor.fetchall()
                print("Available threads:")
                for i, (thread_id,) in enumerate(threads, 1):
                    print(f"  {i}. {thread_id}")
                continue
            elif user_input.lower() == 'history':
                # Show history
                state = app.get_state(config)
                if state and state.values:
                    messages = state.values.get("messages", [])
                    print(f"Conversation history ({len(messages)} messages):")
                    for i, msg in enumerate(messages, 1):
                        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                        print(f"  {i}. [{msg_type}]: {msg.content}")
                else:
                    print("No conversation history found")
                continue
            elif not user_input:
                continue
            
            # Send message to the agent
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            response = result['messages'][-1].content
            print(f"AI: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def cleanup_database(conn, db_path):
    """Clean up database resources."""
    try:
        conn.close()
        print(f"\n🧹 Database connection closed")
        print(f"Database file: {os.path.abspath(db_path)}")
    except Exception as e:
        print(f"❌ Error closing database: {e}")

def main():
    """Main function to run the SQLite memory example."""
    print("🚀 LangGraph SQLite Memory Persistence Example")
    print("=" * 60)
    
    # Setup environment
    openai_api_key = setup_environment()
    
    # Create database connection
    conn, sql_memory = create_database_connection()
    if not conn or not sql_memory:
        print("❌ Failed to initialize database. Exiting.")
        return
    
    # Create LLM
    llm = create_llm(openai_api_key)
    
    # Create LangGraph workflow
    app = create_langgraph_workflow(llm, sql_memory)
    if not app:
        print("❌ Failed to create workflow. Exiting.")
        return
    
    # Test memory persistence
    test_memory_persistence(app)
    
    # Demonstrate advanced features
    demonstrate_advanced_features(app, conn, "memory.db")
    
    # Ask if user wants interactive mode
    try:
        interactive = input("\n🤔 Would you like to start an interactive conversation? (y/n): ").strip().lower()
        if interactive in ['y', 'yes']:
            interactive_conversation(app)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    # Cleanup
    cleanup_database(conn, "memory.db")
    
    print("\n✅ SQLite Memory Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
