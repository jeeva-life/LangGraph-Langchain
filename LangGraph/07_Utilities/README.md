# LangGraph Utilities

This folder contains utility functions, helper modules, and common tools for LangGraph development.

## Purpose:

This folder is designed to hold reusable utilities that can be used across different LangGraph projects:

- Helper functions and decorators
- Common state management utilities
- Database connection helpers
- API integration utilities
- Testing and debugging tools
- Configuration management

## Planned Utilities:

### 🔧 **Core Utilities**
- State validation helpers
- Error handling decorators
- Logging utilities
- Configuration loaders

### 🗄️ **Database Utilities**
- Connection managers
- Query builders
- Migration helpers
- Backup utilities

### 🔌 **API Utilities**
- API client wrappers
- Rate limiting helpers
- Retry mechanisms
- Authentication helpers

### 🧪 **Testing Utilities**
- Mock data generators
- Test fixtures
- Assertion helpers
- Performance testing tools

### 📊 **Monitoring Utilities**
- Metrics collectors
- Health check endpoints
- Performance monitors
- Alert systems

## Example Structure:

```
07_Utilities/
├── README.md
├── core/
│   ├── __init__.py
│   ├── state_utils.py
│   ├── error_handlers.py
│   └── config.py
├── database/
│   ├── __init__.py
│   ├── connection_manager.py
│   └── query_builder.py
├── api/
│   ├── __init__.py
│   ├── client_wrapper.py
│   └── rate_limiter.py
├── testing/
│   ├── __init__.py
│   ├── mock_data.py
│   └── test_helpers.py
└── monitoring/
    ├── __init__.py
    ├── metrics.py
    └── health_checks.py
```

## Common Utility Patterns:

### ✅ **State Management Utilities**
```python
def validate_state(state: dict, schema: dict) -> bool:
    """Validate state against schema"""
    pass

def merge_states(base_state: dict, updates: dict) -> dict:
    """Safely merge state updates"""
    pass

def sanitize_state(state: dict) -> dict:
    """Sanitize state for persistence"""
    pass
```

### ✅ **Error Handling Utilities**
```python
def retry_on_failure(max_retries: int = 3):
    """Decorator for retrying failed operations"""
    pass

def log_errors(func):
    """Decorator for logging errors"""
    pass

def handle_api_errors(func):
    """Decorator for handling API errors"""
    pass
```

### ✅ **Configuration Utilities**
```python
def load_config(config_path: str) -> dict:
    """Load configuration from file"""
    pass

def get_env_var(key: str, default: str = None) -> str:
    """Get environment variable with default"""
    pass

def validate_config(config: dict) -> bool:
    """Validate configuration"""
    pass
```

## Usage Examples:

### Importing Utilities
```python
from langgraph_utils.core import validate_state, merge_states
from langgraph_utils.database import get_connection
from langgraph_utils.api import retry_on_failure
```

### Using State Utilities
```python
# Validate state before processing
if validate_state(state, state_schema):
    processed_state = process_state(state)
else:
    raise ValueError("Invalid state")

# Merge state updates safely
updated_state = merge_states(current_state, updates)
```

### Using Database Utilities
```python
# Get database connection
conn = get_connection()

# Execute query with error handling
result = execute_query(conn, query, params)
```

## Best Practices:

### ✅ **Code Organization**
- Group related utilities in submodules
- Use clear, descriptive names
- Include comprehensive docstrings
- Add type hints for better IDE support

### ✅ **Error Handling**
- Always handle exceptions gracefully
- Provide meaningful error messages
- Log errors for debugging
- Include fallback mechanisms

### ✅ **Testing**
- Write unit tests for all utilities
- Test edge cases and error conditions
- Use mock objects for external dependencies
- Maintain high test coverage

### ✅ **Documentation**
- Document all public functions
- Include usage examples
- Explain parameters and return values
- Keep documentation up to date

## Contributing:

To add new utilities to this folder:

1. Create appropriate subfolder if needed
2. Add utility functions with proper documentation
3. Include unit tests
4. Update this README with new utilities
5. Follow the established patterns and conventions

## Next Steps:

1. Implement core utility functions
2. Add database connection utilities
3. Create API integration helpers
4. Build testing and debugging tools
5. Add monitoring and analytics utilities
