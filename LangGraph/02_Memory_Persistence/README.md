# Memory Persistence in LangGraph

This folder contains examples and implementations of persistent memory storage in LangGraph applications.

## Files in this folder:

### 📓 **SQLite_Memory.ipynb**
- **Purpose**: Comprehensive SQLite memory persistence tutorial
- **Features**:
  - SQLite checkpoint integration
  - Thread-based conversation management
  - Database operations and statistics
  - Advanced memory features demonstration
- **Prerequisites**: `langgraph-checkpoint` package

### 🐍 **sqlite_memory_example.py**
- **Purpose**: Standalone Python script for SQLite memory persistence
- **Features**:
  - Complete setup and initialization
  - Interactive conversation mode
  - Thread management utilities
  - Database statistics and monitoring
- **Usage**: `python sqlite_memory_example.py`

### 📓 **Checkpoint.ipynb**
- **Purpose**: Additional checkpoint and state management examples
- **Features**: Various checkpoint strategies and patterns

## Key Features:

### ✅ **Persistent State Management**
- Conversation history survives application restarts
- Multiple conversation threads supported
- Automatic state checkpointing

### ✅ **Thread-Based Conversations**
- Each conversation has a unique thread ID
- Independent conversation histories
- Easy thread switching and management

### ✅ **Database Operations**
- View all conversation threads
- Retrieve conversation history
- Monitor database statistics
- Manage checkpoint data

## Use Cases:

1. **Chat Applications**: Maintain context across sessions
2. **Multi-User Systems**: Separate conversations per user
3. **Complex Workflows**: Save intermediate states
4. **Debugging**: Inspect conversation history
5. **Analytics**: Analyze conversation patterns

## Setup Requirements:

```bash
pip install langgraph-checkpoint
pip install sqlite3  # Built into Python
```

## Database Schema:

The SQLite database automatically creates tables:
- **checkpoints**: Main table storing conversation states
- **thread_id**: Unique identifier for each conversation
- **checkpoint_id**: Sequential ID for each state checkpoint
- **metadata**: Additional information about each checkpoint
