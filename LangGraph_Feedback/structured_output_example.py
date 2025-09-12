"""
Structured Outputs with Multi-Agent Workflow

This script demonstrates a multi-agent system using LangGraph with structured outputs,
featuring a worker agent and an evaluator agent with feedback loops.

Key Features:
- Structured outputs using Pydantic models
- Multi-agent workflow with worker and evaluator
- Playwright web browsing tools (synchronous API)
- Feedback loop for iterative improvement
- Error handling and fallback mechanisms
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
# ScreenshotTool import removed due to module issues



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
        print(" No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")
        return False
    
    print(" Environment setup complete")
    return True


def setup_playwright_tools():
    """Setup Playwright tools using synchronous API to avoid asyncio issues."""
    try:
        from playwright.sync_api import sync_playwright
        from langchain_community.tools.playwright import (
            ClickTool, ExtractTextTool, NavigateTool,
            NavigateBackTool, NavigateForwardTool, GetCurrentPageTool
        )
        
        print(" Setting up Playwright tools with synchronous API...")
        
        # Create a simple browser instance for tool creation
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Create individual tools manually
            tools = [
                NavigateTool(page=page),
                ClickTool(page=page),
                ExtractTextTool(page=page),
                NavigateBackTool(page=page),
                NavigateForwardTool(page=page),
                GetCurrentPageTool(page=page)
            ]
            
            browser.close()
            print(" Playwright tools created successfully!")
            return tools
            
    except Exception as e:
        print(f" Error setting up Playwright tools: {e}")
        print(" Falling back to mock tools...")
        
        # Create mock tools as fallback
        @tool
        def mock_navigate(url: str) -> str:
            """Mock navigate tool for demonstration."""
            return f"Mock: Would navigate to {url}"
        
        @tool
        def mock_click(selector: str) -> str:
            """Mock click tool for demonstration."""
            return f"Mock: Would click on {selector}"
        
        @tool
        def mock_extract_text(selector: str = "body") -> str:
            """Mock extract text tool for demonstration."""
            return f"Mock: Would extract text from {selector}"
        
        @tool
        def mock_screenshot() -> str:
            """Mock screenshot tool for demonstration."""
            return "Mock: Would take a screenshot"
        
        return [mock_navigate, mock_click, mock_extract_text, mock_screenshot]


def create_llms():
    """Create and configure the language models."""
    print(" Setting up language models...")
    
    # Worker LLM with tools
    worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Evaluator LLM with structured output
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm_with_structured_output = evaluator_llm.with_structured_output(EvaluatorOutput)
    
    print(" Language models configured successfully!")
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


def evaluator(state: State, evaluator_llm_with_structured_output) -> Dict[str, Any]:
    """Evaluator node that provides structured feedback on the worker's response."""
    
    # Get the last message from the worker
    messages = state["messages"]
    if not messages:
        return {
            "feedback_on_work": "No response from worker",
            "success_criteria_met": False,
            "user_input_needed": True
        }
    
    last_message = messages[-1]
    
    # Create evaluation prompt
    evaluation_prompt = f"""
    Evaluate the worker's response based on the success criteria.
    
    Success Criteria: {state["success_criteria"]}
    
    Worker's Response: {last_message.content}
    
    Provide feedback on whether the success criteria was met and if more input is needed.
    """
    
    # Get evaluation from LLM
    evaluation = evaluator_llm_with_structured_output.invoke([HumanMessage(content=evaluation_prompt)])
    
    return {
        "feedback_on_work": evaluation.feedback,
        "success_criteria_met": evaluation.success_criteria_met,
        "user_input_needed": evaluation.user_input_needed
    }


def should_continue(state: State) -> str:
    """Decide whether to continue or end the workflow."""
    if state.get("success_criteria_met", False):
        return "end"
    elif state.get("user_input_needed", False):
        return "end"  # End to get user input
    else:
        return "evaluator"  # Continue to evaluator


def should_continue_from_evaluator(state: State) -> str:
    """Decide whether to continue from evaluator or end."""
    if state.get("success_criteria_met", False):
        return "end"
    elif state.get("user_input_needed", False):
        return "end"  # End to get user input
    else:
        return "worker"  # Go back to worker with feedback


def create_workflow(worker_llm_with_tools, evaluator_llm_with_structured_output):
    """Create the LangGraph workflow."""
    print(" Creating multi-agent workflow...")
    
    # Create the graph
    workflow = StateGraph(State)

    # Add nodes with partial functions to pass the LLMs
    workflow.add_node("worker", lambda state: worker(state, worker_llm_with_tools))
    workflow.add_node("evaluator", lambda state: evaluator(state, evaluator_llm_with_structured_output))

    # Add edges
    workflow.add_edge(START, "worker")
    workflow.add_conditional_edges(
        "worker",
        should_continue,
        {
            "evaluator": "evaluator",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "evaluator",
        should_continue_from_evaluator,
        {
            "worker": "worker",  # Go back to worker with feedback
            "end": END
        }
    )

    # Compile the graph
    app = workflow.compile()
    
    print(" Multi-agent workflow created successfully!")
    return app


def test_workflow(app):
    """Test the multi-agent workflow with various scenarios."""
    print("\n Testing Multi-Agent Workflow with Structured Outputs")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Web Research Task",
            "message": "Research the latest news about artificial intelligence",
            "criteria": "Provide a summary of 3 recent AI news articles with sources"
        },
        {
            "name": "Data Analysis Task",
            "message": "Analyze the stock market trends for the past month",
            "criteria": "Provide 5 key insights with specific data points and charts"
        },
        {
            "name": "Creative Writing Task",
            "message": "Write a short story about a robot learning to paint",
            "criteria": "Create a 500-word story with character development and a clear plot"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n Test {i}: {test_case['name']}")
        print("-" * 40)
        
        initial_state = {
            "messages": [HumanMessage(content=test_case["message"])],
            "success_criteria": test_case["criteria"],
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False
        }
        
        try:
            result = app.invoke(initial_state)
            
            print(" Workflow completed!")
            print(f"Success criteria met: {result.get('success_criteria_met', False)}")
            print(f"User input needed: {result.get('user_input_needed', False)}")
            
            if result.get("feedback_on_work"):
                print(f"Feedback: {result['feedback_on_work']}")
            
            # Show the final response
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"Final response: {last_message.content[:200]}...")
                
        except Exception as e:
            print(f" Error: {e}")
    
    print("\n All tests completed!")


def interactive_mode(app):
    """Start interactive mode for testing the workflow."""
    print("\n🌐 Interactive Multi-Agent Mode")
    print("=" * 40)
    print("Type 'quit' to exit")
    print("Example tasks:")
    print("  - 'Research the latest AI developments'")
    print("  - 'Write a poem about space exploration'")
    print("  - 'Analyze the benefits of renewable energy'")
    
    while True:
        try:
            user_input = input("\n Your task: ").strip()
            
            if user_input.lower() == 'quit':
                print(" Goodbye!")
                break
            elif not user_input:
                continue
            
            # Get success criteria from user
            criteria = input(" Success criteria: ").strip()
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
            
            result = app.invoke(initial_state)
            
            print(f"\n Workflow completed!")
            print(f"Success criteria met: {result.get('success_criteria_met', False)}")
            print(f"User input needed: {result.get('user_input_needed', False)}")
            
            if result.get("feedback_on_work"):
                print(f"Feedback: {result['feedback_on_work']}")
            
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"\n Response: {last_message.content}")
            
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")


def main():
    """Main function to run the structured output example."""
    print(" Structured Outputs with Multi-Agent Workflow")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Setup Playwright tools
    tools = setup_playwright_tools()
    
    # Create LLMs
    worker_llm, evaluator_llm_with_structured_output = create_llms()
    
    # Bind tools to worker LLM
    worker_llm_with_tools = worker_llm.bind_tools(tools)
    
    # Create workflow
    app = create_workflow(worker_llm_with_tools, evaluator_llm_with_structured_output)
    
    # Test the workflow
    test_workflow(app)
    
    # Ask if user wants interactive mode
    try:
        interactive = input("\n Would you like to start interactive mode? (y/n): ").strip().lower()
        if interactive in ['y', 'yes']:
            interactive_mode(app)
    except KeyboardInterrupt:
        print("\n Goodbye!")
    
    print("\n Structured Output Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
