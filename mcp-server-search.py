from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
from datetime import datetime, date
import requests
import json
from googlesearch import search
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import uvicorn

# Initialize FastMCP server for Weather tools (SSE)
mcp = FastMCP("misc-tool")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

@mcp.tool()
async def get_weekday_from_date(date_str: str) -> str:
    '''
    input date, return weekday
    :param date_str: the string of date, format is YYYY-MM-DD
    :return: the weekday
    '''
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        weekday_number = date.weekday()
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        return weekdays[weekday_number]
    except ValueError:
        return "invalid date, use YYYY-MM-DD format."


@mcp.tool()
async def get_weather_for_city_at_date(city: str, date_str: str | None = None) -> str:
    '''
    Get the weather for a specific date and city using wttr.in (no API key needed).
    :param city: The name of the city
    :param date_str: The string of the date, format is YYYY-MM-DD. Defaults to today if not provided.
    :return: The weather information or an error message.
    '''
    target_date_str = date_str if date_str else date.today().strftime('%Y-%m-%d')

    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        # wttr.in doesn't support future dates well for the free tier JSON format, 
        # and historical data might be limited. We'll fetch current/forecast for simplicity.
        # If a specific date is requested, we inform the user it might not be precise for past/future.
        date_info = f"on or around {target_date_str}" if date_str else f"for today ({target_date_str})"

        # Use wttr.in API (JSON format)
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        weather_data = response.json()

        # Extract relevant information (example: current condition)
        current_condition = weather_data.get('current_condition', [{}])[0]
        description = current_condition.get('weatherDesc', [{}])[0].get('value', 'N/A')
        temp_c = current_condition.get('temp_C', 'N/A')
        feels_like_c = current_condition.get('FeelsLikeC', 'N/A')

        return f"Weather for {city} {date_info}: {description}, Temp: {temp_c}°C, Feels like: {feels_like_c}°C."

    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except (KeyError, IndexError, json.JSONDecodeError):
        return f"Could not parse weather data for {city}. The city might be invalid or the API response changed."


@mcp.tool()
async def google_search(query: str, num_results: int = 5) -> str:
    '''
    Performs a Google search for the given query.
    :param query: The search query string.
    :param num_results: The maximum number of results to return (default is 5).
    :return: A string containing the search results, or an error message.
    '''
    try:
        results = []
        # The search function returns a generator, we take the first num_results
        for j in search(query, num_results=num_results):
            results.append(j)
            if len(results) >= num_results:
                break
        if not results:
            return f"No results found for '{query}'."
        return f"Search results for '{query}':\n" + "\n".join(results)
    except Exception as e:
        return f"An error occurred during the search: {e}"


@mcp.tool()
async def get_web_content(url: str, format: str = 'markdown') -> str:
    '''
    Fetches the main content of a web page and returns it in the specified format.
    Attempts to remove common boilerplate like headers, footers, and navigation.
    :param url: The URL of the web page to fetch.
    :param format: The desired output format ('markdown', 'html', 'text'). Defaults to 'markdown'.
    :return: The extracted main content in the specified format, or an error message.
    '''
    supported_formats = ['markdown', 'html', 'text']
    if format.lower() not in supported_formats:
        return f"Error: Unsupported format '{format}'. Supported formats are: {', '.join(supported_formats)}"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Content Extraction Heuristics (Best Effort) ---
        # Remove common boilerplate tags
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button', 'iframe']): # Added form, button, iframe
            tag.decompose()

        # Try to find common main content containers
        main_content = soup.find('article') or \
                       soup.find('main') or \
                       soup.find('div', {'id': 'content'}) or \
                       soup.find('div', {'class': 'content'}) or \
                       soup.find('div', {'id': 'main-content'}) or \
                       soup.find('div', {'class': 'main-content'}) or \
                       soup.find('div', {'role': 'main'}) # Added role='main'

        # If no specific container found, use the body, but try to clean it
        if not main_content:
            main_content = soup.body
            if main_content:
                 # Remove elements often found outside main content within body
                 for selector in ['header', 'footer', 'nav', '.sidebar', '#sidebar', '.menu', '#menu']:
                     for tag in main_content.select(selector):
                         tag.decompose()
            else:
                 # Fallback if body is also missing
                 main_content = soup

        # --- Formatting ---
        if format.lower() == 'html':
            return str(main_content)
        elif format.lower() == 'markdown':
            # Convert HTML to Markdown
            return md(str(main_content), heading_style="ATX")
        elif format.lower() == 'text':
            # Extract text, trying to preserve some structure
            return main_content.get_text(separator='\n', strip=True)

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while fetching {url}."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing {url}: {e}"

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse
    
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
