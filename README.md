# MCP SSE Client-Server

A Simple Python implementation of a MCP client and Server with Server-Sent Events (SSE).

## Features

- SSE-based communication between client and server
- Integration with MCP platform tools (google_search, get_web_content)
- Asynchronous processing of queries using Python's asyncio
- Comprehensive logging and error handling
- OpenAI API integration with fallback to OpenRouter
- Environment variable configuration support

## Requirements

- Python 3.10+
- Dependencies listed in pyproject.toml:
  - httpx for HTTP requests
  - python-dotenv for environment variables
  - openai for LLM integration
  - mcp[cli] for MCP platform integration

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   uv venv
   # On Windows
   .venv\Scripts\activate
   # On Unix/macOS
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   uv sync .
   ```

## Configuration

Create a `.env` file in the project root with your API keys:
```
# Required - use either OPENAI_API_KEY or OPENROUTER_API_KEY
OPENAI_API_KEY=your_api_key
OPENROUTER_API_KEY=your_api_key

# Optional - specify custom base URL
OPENAI_BASE_URL=https://your-custom-url
OPENROUTER_BASE_URL=https://your-custom-url
```

## Usage
start mcp server firstly, which run on 8081 port.
then start mcp client.

### MCP Server Setup

```bash
uv run mcp-server-search.py
```
### MCP Client Usage
```bash
uv run mcp-client-sse.py --help
usage: mcp-client-sse.py [-h] [--server SERVER] [--verbose] [--quiet]

Run MCP SSE-based client

options:
  -h, --help       show this help message and exit
  --server SERVER  MCP SSE server URL (default: http://localhost:8081/sse)
  --verbose        Enable verbose logging mode (DEBUG level)
  --quiet          Set logging to WARNING level (overrides --verbose)

uv run mcp-client-sse.py --server URL_ADDRESS-server-url:8081/sse --verbose
```

### Available MCP Tools

The client supports these MCP tools:
- google_search: Perform web searches
- get_web_content: Fetch full web page content
- get_weekday_from_date: Date utilities
- get_weather_for_date: Weather information

## Development

### Logging

The client generates logs in `mcp-client.log` with timestamps and detailed information.

### Testing

To be implemented...

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

## License

[MIT](https://choosealicense.com/licenses/mit/)
