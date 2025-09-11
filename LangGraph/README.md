# LangGraph Learning Repository

This repository contains comprehensive examples, tutorials, and implementations of LangGraph applications organized by topic and complexity level.

## 📁 Folder Structure

### 🚀 **01_Getting_Started**
- **Purpose**: Introduction to LangGraph basics
- **Contents**: Basic workflows, state management, search integration
- **Files**: `Intro_Langsmith.ipynb`, `langgraph_intro.py`
- **Prerequisites**: Basic Python knowledge

### 💾 **02_Memory_Persistence**
- **Purpose**: Persistent state management and memory storage
- **Contents**: SQLite integration, thread management, checkpoint systems
- **Files**: `SQLite_Memory.ipynb`, `sqlite_memory_example.py`, `Checkpoint.ipynb`
- **Prerequisites**: `langgraph-checkpoint` package

### 🔧 **03_Advanced_Features**
- **Purpose**: Advanced LangGraph implementations and complex features
- **Contents**: Multi-API search integration, enhanced workflows, error handling
- **Files**: `enhanced_langgraph_search.py`
- **Prerequisites**: Multiple API keys, advanced Python knowledge

### 📚 **04_Examples**
- **Purpose**: Real-world applications and use cases
- **Contents**: Business applications, educational tools, creative projects
- **Files**: *Planned examples and case studies*
- **Prerequisites**: Varies by example

### 🏗️ **05_State_Management**
- **Purpose**: Advanced state management patterns and techniques
- **Contents**: State reducers, object manipulation, validation
- **Files**: `StateObject_Reducer.ipynb`
- **Prerequisites**: Understanding of state management concepts

### 🌐 **06_Web_Interfaces**
- **Purpose**: Web interfaces and user interfaces for LangGraph apps
- **Contents**: Gradio, Streamlit, FastAPI integrations
- **Files**: `OpenAI_Chatbot_Graph.ipynb`
- **Prerequisites**: Web development basics

### 🛠️ **07_Utilities**
- **Purpose**: Reusable utilities and helper functions
- **Contents**: Common tools, helpers, configuration management
- **Files**: *Planned utility modules*
- **Prerequisites**: Python development experience

## 🎯 Learning Path

### **Beginner Level**
1. Start with `01_Getting_Started/` to understand LangGraph basics
2. Learn about state management and workflow creation
3. Experiment with simple examples

### **Intermediate Level**
1. Move to `02_Memory_Persistence/` for persistent state management
2. Explore `05_State_Management/` for advanced state patterns
3. Try `06_Web_Interfaces/` for user interface development

### **Advanced Level**
1. Study `03_Advanced_Features/` for complex implementations
2. Build real-world applications in `04_Examples/`
3. Use `07_Utilities/` for production-ready code

## 🚀 Quick Start

### **Prerequisites**
```bash
pip install langgraph langchain langchain-openai
pip install langgraph-checkpoint  # For memory features
pip install gradio streamlit      # For web interfaces
```

### **Environment Setup**
Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key    # Optional
LANGCHAIN_API_KEY=your_langchain_key  # Optional
```

### **Getting Started**
1. **Start Here**: `01_Getting_Started/Intro_Langsmith.ipynb`
2. **Add Memory**: `02_Memory_Persistence/SQLite_Memory.ipynb`
3. **Build Interfaces**: `06_Web_Interfaces/OpenAI_Chatbot_Graph.ipynb`

## 📖 Key Concepts Covered

### ✅ **Core LangGraph Concepts**
- StateGraph creation and management
- Node and edge definitions
- Conditional logic and flow control
- State management with TypedDict

### ✅ **Memory and Persistence**
- SQLite checkpoint integration
- Thread-based conversations
- State persistence across sessions
- Database operations and management

### ✅ **Advanced Features**
- Multi-API search integration
- Error handling and fallbacks
- Enhanced workflow patterns
- Performance optimization

### ✅ **Web Development**
- Gradio web interfaces
- Streamlit applications
- FastAPI REST endpoints
- Real-time communication

### ✅ **State Management**
- State reducers and transformers
- Object manipulation patterns
- State validation and sanitization
- Complex state structures

## 🛠️ Development Tools

### **Notebooks (.ipynb)**
- Interactive learning and experimentation
- Step-by-step tutorials
- Visual output and debugging
- Great for learning and prototyping

### **Python Scripts (.py)**
- Production-ready implementations
- Standalone execution
- Easy deployment and integration
- Better for automation and CI/CD

### **Database Files**
- SQLite databases for persistence
- Thread and checkpoint storage
- Conversation history management
- Easy backup and migration

## 🔧 Troubleshooting

### **Common Issues**
1. **Import Errors**: Ensure all required packages are installed
2. **API Key Issues**: Check your `.env` file and API key validity
3. **Memory Errors**: Restart kernel and run setup cells
4. **Database Locked**: Close any open database connections

### **Getting Help**
1. Check the README in each folder
2. Look at the code comments and docstrings
3. Run the examples step by step
4. Check the error messages for guidance

## 📚 Additional Resources

### **Official Documentation**
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)

### **Community Resources**
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/langchain-ai/langgraph/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)

## 🤝 Contributing

### **Adding New Examples**
1. Choose appropriate folder based on complexity
2. Create both notebook and script versions
3. Include comprehensive README
4. Add proper documentation and comments

### **Improving Existing Code**
1. Fix bugs and improve error handling
2. Add new features and capabilities
3. Improve documentation and examples
4. Optimize performance and usability

## 📄 License

This repository is for educational purposes. Please respect the terms of service of any APIs you use.

---

**Happy Learning! 🚀**

Start with the basics in `01_Getting_Started/` and work your way up to advanced implementations. Each folder contains detailed README files to guide you through the learning process.