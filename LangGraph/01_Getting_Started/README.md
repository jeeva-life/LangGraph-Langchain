# Getting Started with LangGraph

This folder contains introductory materials and basic examples for getting started with LangGraph.

## Files in this folder:

### 📓 **Intro_Langsmith.ipynb**
- **Purpose**: Comprehensive introduction to LangGraph with LangSmith integration
- **Features**: 
  - Basic LangGraph workflow creation
  - Search API integration (Serper, Google Search)
  - Error handling and fallback mechanisms
  - Enhanced workflow with search capabilities
- **Prerequisites**: OpenAI API key, optional Serper API key

### 🐍 **langgraph_intro.py**
- **Purpose**: Python script version of the LangGraph introduction
- **Features**: Same as notebook but in standalone script format
- **Usage**: `python langgraph_intro.py`

## Learning Path:

1. **Start with**: `Intro_Langsmith.ipynb` for interactive learning
2. **Then try**: `langgraph_intro.py` for script-based execution
3. **Next**: Move to `02_Memory_Persistence/` for persistent state management

## Key Concepts Covered:

- ✅ **StateGraph Creation**: Building workflows with nodes and edges
- ✅ **State Management**: Using TypedDict for structured state
- ✅ **Conditional Logic**: Dynamic flow control based on state
- ✅ **Search Integration**: Multiple search API options with fallbacks
- ✅ **Error Handling**: Robust error handling and recovery

## Prerequisites:

- Python 3.8+
- LangGraph: `pip install langgraph`
- LangChain: `pip install langchain langchain-openai`
- Optional: `pip install langgraph-checkpoint` for memory features
