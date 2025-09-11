# Advanced LangGraph Features

This folder contains advanced LangGraph implementations and complex feature demonstrations.

## Files in this folder:

### 🐍 **enhanced_langgraph_search.py**
- **Purpose**: Advanced LangGraph workflow with multiple search API integration
- **Features**:
  - Multiple search API support (Serper, Google Search, Mock)
  - Intelligent search detection and integration
  - Enhanced conversation flow with search context
  - Comprehensive error handling and fallbacks
  - Search count tracking and analytics
- **Usage**: `python enhanced_langgraph_search.py`

## Key Features:

### ✅ **Multi-API Search Integration**
- **SerperDevTool**: Primary search option
- **Google Search API**: Alternative search option
- **Google Serper API Wrapper**: Another alternative
- **Mock Search Tool**: Fallback for demonstration

### ✅ **Smart Search Detection**
- Automatic keyword-based search triggering
- Context-aware search integration
- Search result processing and formatting

### ✅ **Enhanced Workflow Management**
- Search count tracking
- Context integration
- Error resilience
- Performance monitoring

### ✅ **Comprehensive Error Handling**
- API key validation
- Network error recovery
- Graceful fallbacks
- User-friendly error messages

## Advanced Workflow Features:

1. **Automatic Search Detection**: Detects when search is needed based on keywords
2. **Search Result Integration**: Seamlessly integrates search results into conversation
3. **Search Tracking**: Keeps track of search count and results
4. **Error Handling**: Robust error handling for all search APIs
5. **Mock Fallback**: Always works for demonstration purposes

## Setup Requirements:

### Required APIs (at least one):
- **Serper API**: Sign up at [serper.dev](https://serper.dev/)
- **Google Search API**: Google Cloud Console
- **OpenAI API**: For LLM functionality

### Environment Variables:
```bash
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key  # Optional
GOOGLE_API_KEY=your_google_key  # Optional
GOOGLE_CSE_ID=your_cse_id       # Optional
```

## Usage Examples:

### Basic Usage:
```python
python enhanced_langgraph_search.py
```

### Interactive Mode:
- Type messages to chat with the AI
- Use `'new'` to start a new conversation thread
- Use `'threads'` to see available threads
- Use `'history'` to see current conversation history
- Use `'quit'` to exit

## Workflow Architecture:

```
START → agent → (continue/end)
  ↓
continue → agent
  ↓
search → agent (if search needed)
  ↓
end → END
```

## Next Steps:

1. Explore more complex state management patterns
2. Implement user authentication with thread management
3. Add conversation analytics and insights
4. Build web interfaces with persistent memory
5. Scale to production with proper database management
