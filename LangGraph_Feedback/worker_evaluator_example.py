"""
Worker-Evaluator Multi-Agent Workflow

This script demonstrates a multi-agent system using LangGraph with a worker agent
and an evaluator agent that provides feedback and determines when tasks are complete.

Key Features:
- Worker agent that can use tools to complete tasks
- Evaluator agent that provides structured feedback
- Conditional routing based on tool usage and evaluation results
- Memory persistence for conversation history
- Clean error handling and fallback mechanisms

Note: This version uses mock tools instead of Playwright to avoid asyncio issues.
"""

import os
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages


class EvaluatorOutput(BaseModel):
    """Structured output for the evaluator agent."""
    feedback: str = Field(description="Feedback on the worker's response")
    success_criteria_met: bool = Field(description="Whether the success criteria has been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")


class State(TypedDict):
    """State definition for the multi-agent workflow."""
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


def setup_environment():
    """Set up environment variables and check API keys."""
    load_dotenv(override=True)
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("WARNING: No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")
        return False
    
    print("Environment setup complete")
    return True


def create_mock_tools():
    """Create mock tools for demonstration purposes."""
    print("Creating mock tools...")
    
    @tool
    def mock_web_search(query: str) -> str:
        """Mock web search tool for demonstration."""
        return f"Mock search results for '{query}': Found 5 relevant articles about the topic."
    
    @tool
    def mock_calculator(expression: str) -> str:
        """Mock calculator tool for demonstration."""
        try:
            result = eval(expression)
            return f"Mock calculation: {expression} = {result}"
        except:
            return f"Mock calculation: Could not evaluate '{expression}'"
    
    @tool
    def mock_file_reader(filename: str) -> str:
        """Mock file reader tool for demonstration."""
        return f"Mock file content from '{filename}': This is sample content that would be read from the file."
    
    @tool
    def mock_data_analyzer(data_description: str) -> str:
        """Mock data analyzer tool for demonstration."""
        return f"Mock analysis of '{data_description}': Generated 3 key insights and 2 charts."
    
    tools = [mock_web_search, mock_calculator, mock_file_reader, mock_data_analyzer]
    print("Mock tools created successfully!")
    return tools


def create_llms():
    """Create and configure the language models."""
    print("Setting up language models...")
    
    # Worker LLM with tools
    worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Evaluator LLM with structured output
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm_with_structured_output = evaluator_llm.with_structured_output(EvaluatorOutput)
    
    print("Language models configured successfully!")
    return worker_llm, evaluator_llm_with_structured_output


def worker(state: State, worker_llm_with_tools) -> Dict[str, Any]:
    """Worker node that uses tools to complete tasks."""
    system_message = f"""you are a helpful assistant that can use tools to complete tasks
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met
    This is the success criteria: {state["success_criteria"]}
    you should reply either with a question for the user about this assignment, or with your final response.
    If you have a question for the user, you need to reply by clearly stating that you are asking a question, and then ask the question.
    
    An example might be:

    Question: Please clarify whether you want a summary or a detailed answer

    If you have finished, reply with the final answer, and don't ask a question. simply reply with the answer.
"""

    if state.get("feedback_on_work"):
        system_message += f"""
        Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
        Here is the feedback of why this was rejected: {state["feedback_on_work"]}
        With this feedback in mind, please continue the assignment, ensuring that you meet the success criteria or ask for more information if needed.
        """

    # Prepare messages with system message
    messages = state["messages"]
    
    # Check if there's already a system message
    found_system_message = False
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = system_message
            found_system_message = True
            break
    
    if not found_system_message:
        messages = [SystemMessage(content=system_message)] + messages

    # Invoke the LLM with tools
    response = worker_llm_with_tools.invoke(messages)

    # Return the updated state
    return {
        "messages": [response],
        "success_criteria": state["success_criteria"],
        "feedback_on_work": state.get("feedback_on_work"),
        "success_criteria_met": state.get("success_criteria_met", False),
        "user_input_needed": state.get("user_input_needed", False)
    }


def worker_router(state: State) -> str:
    """Route worker output to either tools or evaluator."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "evaluator"


def format_conversation(messages: List[Any]) -> str:
    """Format conversation history for evaluation."""
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content or "[Tools use]"
            conversation += f"Assistant: {text}\n"
    return conversation


def evaluator(state: State, evaluator_llm_with_structured_output) -> Dict[str, Any]:
    """Evaluator node that provides structured feedback on the worker's response."""
    last_response = state["messages"][-1].content

    system_message = """You are an evaluator that determines whether the assistant has met the success criteria.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether the user needs to provide more information."""

    user_message = f"""You are evaluating a conversation between the user and assistant. you decide what action to take based on the last response
    The entire conversation with assistant, with the user's original request and all replies is:
    {format_conversation(state['messages'])}

    The success criteria for this assignment is:
    {state['success_criteria']}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and with your decision on whether the success criteria has been met,
    Also decide if more user input is required, either because the assistant is stuck, or because the user needs to clarify their request."""
    
    if state["feedback_on_work"]:
        user_message += f"Also, note that in a prior attempt from the assistant, you provided this feedback: {state['feedback_on_work']}"
        user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is needed."

    evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
    
    eval_result = evaluator_llm_with_structured_output.invoke(evaluator_messages)
    
    new_state = {
        "messages": [AIMessage(content=f"Evaluator Feedback on this answer: {eval_result.feedback}")],
        "feedback_on_work": eval_result.feedback,
        "success_criteria_met": eval_result.success_criteria_met,
        "user_input_needed": eval_result.user_input_needed,
    }
    return new_state


def route_based_on_evaluator_result(state: State) -> str:
    """Route based on evaluator results."""
    if state["success_criteria_met"] or state["user_input_needed"]:
        return "END"
    else:
        return "worker"


def create_workflow(worker_llm_with_tools, evaluator_llm_with_structured_output, tools, use_memory=True):
    """Create the LangGraph workflow."""
    print("Creating multi-agent workflow...")
    
    # Create the graph
    graph_builder = StateGraph(State)

    # Add nodes with partial functions to pass the LLMs
    graph_builder.add_node("worker", lambda state: worker(state, worker_llm_with_tools))
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_node("evaluator", lambda state: evaluator(state, evaluator_llm_with_structured_output))

    # Add edges
    graph_builder.add_edge(START, "worker")
    graph_builder.add_conditional_edges("worker", worker_router, {"tools": "tools", "evaluator": "evaluator"})
    graph_builder.add_edge("tools", "worker")
    graph_builder.add_conditional_edges("evaluator", route_based_on_evaluator_result, {"END": END, "worker": "worker"})

    # Compile the graph
    if use_memory:
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        print("Multi-agent workflow created successfully with memory!")
    else:
        graph = graph_builder.compile()
        print("Multi-agent workflow created successfully without memory!")
    
    return graph


def test_workflow(graph):
    """Test the multi-agent workflow with various scenarios."""
    print("\nTesting Worker-Evaluator Multi-Agent Workflow")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Web Research Task",
            "message": "Research the latest developments in artificial intelligence",
            "criteria": "Provide a summary of 3 recent AI developments with sources and dates"
        },
        {
            "name": "Mathematical Problem",
            "message": "Calculate the compound interest for $1000 invested at 5% for 3 years",
            "criteria": "Show the calculation steps and provide the final amount"
        },
        {
            "name": "Data Analysis Task",
            "message": "Analyze the sales data from last quarter",
            "criteria": "Provide 5 key insights with specific metrics and recommendations"
        },
        {
            "name": "Creative Writing Task",
            "message": "Write a short story about a time-traveling detective",
            "criteria": "Create a 300-word story with a clear plot and character development"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 40)
        
        initial_state = {
            "messages": [HumanMessage(content=test_case["message"])],
            "success_criteria": test_case["criteria"],
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False
        }
        
        try:
            # Check if graph uses checkpointer
            if hasattr(graph, 'checkpointer') and graph.checkpointer:
                config = {"configurable": {"thread_id": f"test_{i}"}}
                result = graph.invoke(initial_state, config=config)
            else:
                result = graph.invoke(initial_state)
            
            print("Workflow completed!")
            print(f"Success criteria met: {result.get('success_criteria_met', False)}")
            print(f"User input needed: {result.get('user_input_needed', False)}")
            
            if result.get("feedback_on_work"):
                print(f"Feedback: {result['feedback_on_work']}")
            
            # Show the final response
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"Final response: {last_message.content[:200]}...")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nAll tests completed!")


def interactive_mode(graph):
    """Start interactive mode for testing the workflow."""
    print("\nInteractive Worker-Evaluator Mode")
    print("=" * 40)
    print("Type 'quit' to exit")
    print("Example tasks:")
    print("  - 'Research the benefits of renewable energy'")
    print("  - 'Calculate the area of a circle with radius 5'")
    print("  - 'Write a poem about space exploration'")
    print("  - 'Analyze the pros and cons of remote work'")
    
    while True:
        try:
            user_input = input("\nYour task: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif not user_input:
                continue
            
            # Get success criteria from user
            criteria = input("Success criteria: ").strip()
            if not criteria:
                criteria = "Complete the task thoroughly and provide a detailed response"
            
            # Create state and run workflow
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "success_criteria": criteria,
                "feedback_on_work": None,
                "success_criteria_met": False,
                "user_input_needed": False
            }
            
            # Check if graph uses checkpointer
            if hasattr(graph, 'checkpointer') and graph.checkpointer:
                config = {"configurable": {"thread_id": "interactive_session"}}
                result = graph.invoke(initial_state, config=config)
            else:
                result = graph.invoke(initial_state)
            
            print(f"\nWorkflow completed!")
            print(f"Success criteria met: {result.get('success_criteria_met', False)}")
            print(f"User input needed: {result.get('user_input_needed', False)}")
            
            if result.get("feedback_on_work"):
                print(f"Feedback: {result['feedback_on_work']}")
            
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"\nResponse: {last_message.content}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def display_workflow_graph(graph):
    """Display the workflow graph structure."""
    try:
        print("\nWorkflow Graph Structure:")
        print("=" * 40)
        
        # Get the graph structure
        graph_dict = graph.get_graph()
        print("Nodes:", list(graph_dict.nodes.keys()))
        print("Edges:", [(edge.source, edge.target) for edge in graph_dict.edges])
        
        print("\nWorkflow Flow:")
        print("START → worker → [tools OR evaluator]")
        print("tools → worker")
        print("evaluator → [worker OR END]")
        
    except Exception as e:
        print(f"WARNING: Could not display graph: {e}")


def main():
    """Main function to run the worker-evaluator example."""
    print("Worker-Evaluator Multi-Agent Workflow")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Create mock tools
    tools = create_mock_tools()
    
    # Create LLMs
    worker_llm, evaluator_llm_with_structured_output = create_llms()
    
    # Bind tools to worker LLM
    worker_llm_with_tools = worker_llm.bind_tools(tools)
    
    # Create workflow (with memory by default)
    graph = create_workflow(worker_llm_with_tools, evaluator_llm_with_structured_output, tools, use_memory=True)
    
    # Display workflow structure
    display_workflow_graph(graph)
    
    # Test the workflow
    test_workflow(graph)
    
    # Ask if user wants interactive mode
    try:
        interactive = input("\nWould you like to start interactive mode? (y/n): ").strip().lower()
        if interactive in ['y', 'yes']:
            interactive_mode(graph)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    
    print("\nWorker-Evaluator Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
