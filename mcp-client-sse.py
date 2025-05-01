import asyncio
import json
import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import argparse
import anyio

# MCP Imports based on documentation examples and user's types.py
from mcp import ClientSession
from mcp.client.sse import sse_client
# Corrected import path for MCP types - Removing AudioContent as it's not in user's types.py
from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource # Import specific types if needed for type hinting or checking

# OpenAI Imports
from openai import OpenAI
# Import the ToolCall model if using OpenAI library v1.x+
try:
    # Assuming usage of openai>=1.0.0
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
except ImportError:
    ChatCompletionMessageToolCall = None # Fallback for older versions or different structures
    ChatCompletionMessage = None

# Environment and Logging
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

# Configure logger
logger = logging.getLogger("mcp_client")
# Basic logging setup - will be refined in main() based on args
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MCPClient:
    def __init__(self, verbose=True):
        # Set log level based on verbosity
        self.verbose = verbose
        log_level = logging.DEBUG if verbose else logging.INFO
        # Ensure logger level is set correctly
        logger.setLevel(log_level)
        # Avoid adding handler if logger already has handlers from basicConfig or previous runs
        if not logger.hasHandlers() or len(logger.handlers) == 0:
             # Create console handler
             console_handler = logging.StreamHandler()
             console_handler.setLevel(log_level)
             # Create file handler
             file_handler = logging.FileHandler('mcp-client.log')
             file_handler.setLevel(log_level)
             # Create formatter
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             console_handler.setFormatter(formatter)
             file_handler.setFormatter(formatter)
             # Add handlers to logger
             logger.addHandler(console_handler)
             logger.addHandler(file_handler)
             logger.propagate = False # Prevent duplicating logs to root logger if basicConfig was called

        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Initial system message with workflow guidance
        self.messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant equipped with several tools.
                \n\n**Workflow Guidance:**\n
                1. Analyze the user's request. \n
                2. If the request requires current information or details from the web, first use `google_search` to find relevant URLs.\n
                3. Based on the search results, decide if fetching full content using `get_web_content` for the most relevant URL(s) is necessary to answer the query comprehensively. *You will receive the web content integrated with the search results if fetched.* \n
                4. Synthesize the information from the search results and any fetched web content to answer the user's question.\n
                5. For other requests, use the appropriate tool directly or answer based on your knowledge."""
            }
        ]

        # Prioritize OPENAI_API_KEY, then OPENROUTER_API_KEY
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file")

        # Support openrouter endpoint
        self.openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")

        self._session_context = None
        self._streams_context = None
        # Create an OpenAI client instance
        self.llm_client = OpenAI(
            base_url = self.openai_base_url,
            api_key = self.openai_api_key,
        )

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        logger.info(f"Attempting to connect to SSE server at {server_url}")
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        try:
            streams = await self._streams_context.__aenter__()
            logger.debug("SSE streams acquired.")
        except Exception as e:
            logger.error(f"Failed to establish SSE connection: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to SSE server at {server_url}") from e


        self._session_context = ClientSession(*streams)
        try:
            self.session: ClientSession = await self._session_context.__aenter__()
            logger.debug("MCP ClientSession entered.")
        except Exception as e:
             logger.error(f"Failed to enter ClientSession context: {e}", exc_info=True)
             # Attempt to clean up streams context if session fails
             if self._streams_context:
                  await self._streams_context.__aexit__(None, None, None)
             raise ConnectionError("Failed to establish MCP session.") from e


        # Initialize
        try:
            await self.session.initialize()
            logger.info("MCP session initialized.")
        except Exception as e:
            logger.error(f"MCP session initialization failed: {e}", exc_info=True)
            # Attempt to clean up contexts
            if self._session_context:
                 await self._session_context.__aexit__(None, None, None)
            if self._streams_context:
                 await self._streams_context.__aexit__(None, None, None)
            raise ConnectionError("MCP session initialization failed.") from e


        # List available tools to verify connection
        logger.info("Listing tools...")
        try:
            response = await self.session.list_tools()
            tools = response.tools
            logger.info("Connected to server with tools: %s", [tool.name for tool in tools])
        except Exception as e:
             logger.error(f"Failed to list tools after connection: {e}", exc_info=True)
             # Consider if connection should be terminated here
             # await self.cleanup() # Optionally cleanup if listing tools is critical
             # raise ConnectionError("Failed to list tools after connection.") from e


    async def cleanup(self):
        """Properly clean up the session and streams"""
        logger.info("Starting cleanup...")
        
        # Create a task group to manage cleanup operations
        async with anyio.create_task_group() as tg:
            # Exit session context first
            if hasattr(self, '_session_context') and self._session_context:
                try:
                    await self._session_context.__aexit__(None, None, None)
                    logger.info("MCP ClientSession exited.")
                except Exception as e:
                    logger.error(f"Error exiting ClientSession context: {e}", exc_info=True)
                finally:
                    self.session = None
                    self._session_context = None

            # Then exit streams context
            if hasattr(self, '_streams_context') and self._streams_context:
                try:
                    await self._streams_context.__aexit__(None, None, None)
                    logger.info("SSE streams context exited.")
                except Exception as e:
                    logger.error(f"Error exiting SSE streams context: {e}", exc_info=True)
                finally:
                    self._streams_context = None
        
        logger.info("Cleanup finished.")

    def _format_tool_content_for_llm(self, tool_content: List[Any]) -> str:
        """
        Formats the tool result content (list of TextContent, etc.) into a
        JSON string suitable for the OpenAI API 'tool' role message content.
        """
        # Convert potentially complex MCP content objects into serializable dicts
        serializable_content = []
        if not isinstance(tool_content, list):
             logger.warning(f"Tool content is not a list: {type(tool_content)}. Attempting to wrap.")
             tool_content = [tool_content] # Wrap non-list content

        for item in tool_content:
            if isinstance(item, TextContent):
                # Ensure 'type' field is included when creating the dictionary
                serializable_content.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                 # Option 3: Simple text placeholder
                 serializable_content.append({"type": "image", "status": f"Image content received ({item.mimeType})"})
            # Removed AudioContent handling as it's not in user's mcp.types
            # elif isinstance(item, AudioContent):
            #      serializable_content.append({"type": "audio", "status": f"Audio content received ({item.mimeType})"})
            elif isinstance(item, EmbeddedResource):
                 # Ensure resource attribute exists before accessing sub-attributes
                 if hasattr(item, 'resource') and item.resource:
                      serializable_content.append({"type": "resource", "uri": str(item.resource.uri), "name": item.resource.name})
                 else:
                      serializable_content.append({"type": "resource", "status": "Malformed EmbeddedResource item"})
            elif isinstance(item, dict) and 'text' in item and 'type' in item: # Handle dicts that already have type and text
                 serializable_content.append(item)
            elif isinstance(item, dict) and 'text' in item: # Handle simple dict case if server returns that (add type)
                 serializable_content.append({"type": "text", "text": item['text']})
            elif isinstance(item, str): # Handle simple string case (add type)
                 serializable_content.append({"type": "text", "text": item})
            else:
                 # Fallback for unknown types
                 logger.warning(f"Unknown item type in tool content: {type(item)}. Converting to string.")
                 serializable_content.append({"type": "unknown", "content": str(item)})

        # Serialize the list of dictionaries to a JSON string
        try:
            # Ensure the final output is a string (JSON representation of the list)
            return json.dumps(serializable_content)
        except TypeError as e:
            logger.error(f"Failed to serialize tool content to JSON: {e}. Content: {serializable_content}")
            # Fallback: return a simple string representation of the error
            return json.dumps([{"type": "error", "text": f"Error serializing tool content: {e}"}])


    async def process_query(self, query: str, top_n: int = 3) -> str:
        """
        Processes a user query, interacts with the LLM, handles tool calls
        (including google_search + get_web_content), and returns the final response.
        Relies on documented MCP client methods and standard OpenAI API format.
        """
        if not self.session:
             logger.error("MCP session is not available.")
             return "Error: Not connected to MCP server."

        # 1. Append user query to message history
        self.messages.append({"role": "user", "content": query})

        # 2. Get available tools from the MCP session
        try:
            response = await self.session.list_tools()
            # Ensure inputSchema is included as per documentation/examples
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema # Directly use the schema provided by the tool
                }
            } for tool in response.tools]
        except Exception as e:
            logger.error(f"Failed to list tools: {e}", exc_info=True)
            # Remove the user message we just added
            self.messages.pop()
            return f"Error: Could not retrieve tools from the server. {e}"

        if self.verbose:
            logger.debug("Available tools: %s", json.dumps(available_tools, indent=2))
            logger.debug("Model: %s", os.getenv("OPENAI_MODEL"))
            logger.debug("Messages before 1st LLM call: %s", self._serialize_messages_for_log(self.messages))

        # --- First LLM Call ---
        try:
            first_response = self.llm_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=self.messages,
                tools=available_tools,
                tool_choice="auto",
                max_tokens=4096,
                temperature=0,
            )
        except Exception as e:
            logger.error("First LLM API call failed: %s", str(e), exc_info=True)
            # Remove the user message we just added, as the call failed
            self.messages.pop()
            return f"Error: Failed to communicate with the language model. {e}"

        if self.verbose:
            try:
                logger.debug("Full first_response: %s", first_response.model_dump_json(indent=2))
            except Exception as log_e:
                 logger.warning(f"Could not serialize first_response for logging: {log_e}")
                 logger.debug("Full first_response (raw): %s", first_response)

        if not first_response.choices:
            logger.error("No choices returned in first_response from LLM.")
            # Remove the user message
            self.messages.pop()
            return "Error: The language model did not provide a response."

        # --- Process First Response ---
        first_message: Optional[ChatCompletionMessage] = first_response.choices[0].message

        # Append assistant's response (content and tool calls) to history
        message_to_append = {"role": "assistant", "content": first_message.content if first_message else None}
        tool_calls_in_response = first_message.tool_calls if first_message else None

        # Handle tool_calls - ensure they are serializable dictionaries for history
        if tool_calls_in_response:
             if ChatCompletionMessageToolCall and isinstance(tool_calls_in_response[0], ChatCompletionMessageToolCall):
                  # Use model_dump() for pydantic models (openai v1+)
                  message_to_append["tool_calls"] = [tc.model_dump() for tc in tool_calls_in_response]
             elif isinstance(tool_calls_in_response, list):
                 # Assume it's already a list of dicts if not the specific object
                 message_to_append["tool_calls"] = tool_calls_in_response
             else:
                  logger.warning(f"Unexpected tool_calls format: {type(tool_calls_in_response)}")
                  message_to_append["tool_calls"] = None # Or handle appropriately
        else:
             message_to_append["tool_calls"] = None

        # Only append if there's content or tool calls
        if message_to_append["content"] or message_to_append["tool_calls"]:
            self.messages.append(message_to_append)
        else:
             logger.warning("Assistant message had no content and no tool calls. Not adding to history.")


        # Determine if tools were called
        tool_calls_requested = message_to_append.get("tool_calls") # Use the dict version added to history

        # --- Handle Tool Calls ---
        if tool_calls_requested:
            tool_results_for_next_call = [] # Store tool results formatted for the next LLM call

            for tool_call in tool_calls_requested:
                # Ensure tool_call is accessed correctly (it's now guaranteed to be a dict)
                tool_call_id = tool_call.get('id')
                function_info = tool_call.get('function', {})
                tool_name = function_info.get('name')
                arguments_str = function_info.get('arguments', '{}')

                if not all([tool_call_id, tool_name]):
                     logger.warning(f"Skipping invalid tool call object in history: {tool_call}")
                     continue

                logger.info(f"LLM requested tool: {tool_name} with ID: {tool_call_id}")

                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse arguments for tool {tool_name}: {arguments_str}")
                    tool_result_content_str = json.dumps([{"type":"error", "text":f"Invalid arguments provided for tool {tool_name}. Arguments must be a valid JSON string."}])
                    tool_results_for_next_call.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result_content_str, # JSON string error
                    })
                    continue # Skip to the next tool call

                # Execute the tool using MCP session
                try:
                    # call_tool returns CallToolResult according to spec/SDK hints
                    # Use the correct type hint here
                    tool_result: CallToolResult = await self.session.call_tool(tool_name, arguments=arguments)

                    # Check for tool execution errors reported by the server
                    if tool_result.isError:
                        logger.warning(f"Tool '{tool_name}' execution reported an error by the server.")
                        # Format the error content provided by the server
                        tool_result_content_str = self._format_tool_content_for_llm(tool_result.content)
                        logger.warning(f"Tool '{tool_name}' error content: {tool_result_content_str}")
                    else:
                        # Process successful result
                        original_tool_content_list = tool_result.content if tool_result.content else []

                        # --- Special Handling for google_search ---
                        if tool_name == "google_search":
                            logger.info("Processing google_search result...")
                            # Extract text content for URL parsing
                            search_result_text = "".join([c.text for c in original_tool_content_list if isinstance(c, TextContent)])

                            combined_content_list = list(original_tool_content_list) # Start with original list
                            urls = [line.strip() for line in search_result_text.splitlines() if line.strip().startswith("http")]

                            if urls:
                                # Limit URLs to top_n results
                                urls = urls[:top_n] if top_n > 0 else urls
                                for url in urls:
                                    logger.info(f"Found URL in search results, attempting to fetch content for: {url}")
                                    try:
                                        # Call get_web_content tool
                                        web_content_result: CallToolResult = await self.session.call_tool(
                                            "get_web_content",
                                            arguments={"url": url, "format": "markdown"}
                                        )
                                        if web_content_result.isError:
                                             logger.warning(f"get_web_content reported an error for {url}")
                                             # Append error info to the results
                                             # Ensure error_content is a list before extending
                                             # FIX: Add type="text" when creating TextContent
                                             error_text = f"Error fetching content for {url}"
                                             if web_content_result.content and isinstance(web_content_result.content, list) and isinstance(web_content_result.content[0], TextContent):
                                                 error_text = web_content_result.content[0].text # Get error text from tool if available
                                             error_content = [TextContent(type="text", text=error_text)]
                                             combined_content_list.append(TextContent(type="text", text="\n--- Web Content Fetch Error ---\n")) # Add type
                                             combined_content_list.extend(error_content)
                                        elif web_content_result.content:
                                             logger.info(f"Successfully fetched web content for {url}.")
                                             # Append fetched content to the results
                                             # FIX: Add type="text" when creating TextContent
                                             combined_content_list.append(TextContent(type="text", text="\n--- Fetched Web Content ---\n")) # Add type
                                             combined_content_list.extend(web_content_result.content)
                                    except Exception as e:
                                        logger.error(f"Error processing URL {url}: {str(e)}")
                                        error_content = [TextContent(type="text", text=f"Error processing URL {url}: {str(e)}")]
                                        combined_content_list.append(TextContent(type="text", text="\n--- Web Content Processing Error ---\n"))
                                        combined_content_list.extend(error_content)
                            else:
                                 logger.info("No URLs found in google_search results to fetch content from.")

                            # Format the potentially combined content list for the LLM
                            tool_result_content_str = self._format_tool_content_for_llm(combined_content_list)
                        # --- End Special Handling for google_search ---
                        else:
                            # For other tools, format the original content list
                            tool_result_content_str = self._format_tool_content_for_llm(original_tool_content_list)

                    # Append the formatted result string for the next LLM call
                    tool_results_for_next_call.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result_content_str, # Use the formatted JSON string
                    })

                except Exception as tool_exec_err:
                    logger.error(f"Failed to execute MCP tool {tool_name}: {tool_exec_err}", exc_info=True)
                    # Provide error feedback as the tool result string
                    tool_result_content_str = json.dumps([{"type":"error", "text":f"Error: Failed to execute tool {tool_name}. Reason: {tool_exec_err}"}])
                    tool_results_for_next_call.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result_content_str, # JSON string error
                    })

            # Append all collected tool results to the message history *after* the loop
            self.messages.extend(tool_results_for_next_call)

            if self.verbose:
                 logger.debug("Messages before 2nd LLM call: %s", self._serialize_messages_for_log(self.messages))

            # --- Second LLM Call ---
            try:
                # Pass tools=[] to prevent the LLM from calling tools again
                second_response = self.llm_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL"),
                    messages=self.messages,
                    tools=[], # Explicitly disable tools
                    max_tokens=4096,
                    temperature=0,
                )
                if not second_response.choices:
                     logger.error("No choices returned in second_response from LLM.")
                     # Don't modify history here, let the error return
                     return "Error: The language model did not provide a final response after processing tools."

                final_message = second_response.choices[0].message
                final_message_content = final_message.content if final_message else None


                if not final_message_content or final_message_content.strip() == '':
                    logger.warning("Received empty final response from LLM after tool execution.")
                    final_message_content = "(The model returned an empty response after processing the tool results.)" # Provide placeholder

            except Exception as e:
                logger.error("Second LLM API call failed: %s", str(e), exc_info=True)
                # History already includes tool results, don't pop user message
                return f"Error: Failed to get final response from the language model after tool use. {e}"

            # Append the final assistant response to history
            self.messages.append({"role": "assistant", "content": final_message_content})
            return final_message_content

        # --- Handle direct response (no tool calls) ---
        else:
            final_message_content = first_message.content if first_message else None
            if not final_message_content or final_message_content.strip() == '':
                logger.warning("Received empty initial response from LLM (no tool calls).")
                 # Assistant message might have already been added, check history
                if self.messages and self.messages[-1]["role"] == "assistant" and not self.messages[-1].get("content"):
                     logger.debug("Updating last empty assistant message.")
                     self.messages[-1]["content"] = "(The model provided an empty response.)"
                     return self.messages[-1]["content"]
                else:
                     # If not already added or last message wasn't empty assistant, add new one
                     empty_response_text = "(The model provided an empty response.)"
                     self.messages.append({"role": "assistant", "content": empty_response_text})
                     return empty_response_text

            # The message was already appended to self.messages earlier if it had content
            return final_message_content

    def _serialize_messages_for_log(self, messages: List[Dict[str, Any]]) -> str:
        """Safely serializes message history for logging, handling potential unserializable objects."""
        try:
            # Attempt standard JSON serialization first
            return json.dumps(messages, indent=2)
        except TypeError:
            # If standard serialization fails, try a more robust approach
            logger.warning("Standard JSON serialization failed for messages, attempting fallback.")
            serializable_messages = []
            for msg in messages:
                serializable_msg = {}
                for key, value in msg.items():
                    # Handle tool_calls specifically if they cause issues
                    if key == "tool_calls" and value is not None:
                         # Ensure tool_calls are dictionaries before adding
                         if isinstance(value, list) and all(isinstance(tc, dict) for tc in value):
                              serializable_msg[key] = value
                         else:
                              # Attempt to convert or represent safely
                              try:
                                   # If they were objects, they should be dicts now from message_to_append logic
                                   # If still not serializable, represent as string
                                   json.dumps(value)
                                   serializable_msg[key] = value
                              except TypeError:
                                   logger.warning(f"Could not serialize tool_calls in message: {msg.get('role')}")
                                   serializable_msg[key] = "[Unserializable Tool Calls]"
                    else:
                        # Convert other potentially problematic types to string
                        try:
                            json.dumps(value) # Test serializability
                            serializable_msg[key] = value
                        except TypeError:
                            serializable_msg[key] = str(value) # Fallback to string representation
                serializable_messages.append(serializable_msg)
            # Try serializing the cleaned list
            try:
                 return json.dumps(serializable_messages, indent=2)
            except Exception as final_e:
                 logger.error(f"Fallback message serialization also failed: {final_e}")
                 return "[Error serializing messages]"
        except Exception as e:
            logger.error(f"Unexpected error during message serialization: {e}")
            return "[Error serializing messages]"


    async def chat_loop(self):
        """Run an interactive chat loop"""
        logger.info("MCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                if not query: # Handle empty input
                     continue

                response = await self.process_query(query)
                print("\nAssistant: " + response)

            except KeyboardInterrupt: # Allow graceful exit with Ctrl+C
                 print("\nExiting chat loop.")
                 break
            except ConnectionError as e: # Handle connection errors specifically
                 logger.error(f"Connection Error: {e}", exc_info=True)
                 print(f"\nConnection Error: {e}. Please check the server and connection details.")
                 break # Exit loop on connection error
            except Exception as e:
                logger.error("Error during chat loop: %s", str(e), exc_info=True)
                print(f"\nAn unexpected error occurred: {e}")


async def main():
    parser = argparse.ArgumentParser(description='Run MCP SSE-based client')
    parser.add_argument('--server', type=str, default='http://localhost:8081/sse', help='MCP SSE server URL (default: http://localhost:8081/sse)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging mode (DEBUG level)')
    parser.add_argument('--quiet', action='store_true', help='Set logging to WARNING level (overrides --verbose)')
    args = parser.parse_args()

    # Configure root logger level based on args
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO # Default level

    # Apply level to the root logger AND our specific logger
    # Use force=True if other libraries might configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Ensure our specific logger also adheres to the level
    logger.setLevel(log_level)
    # Prevent duplicate messages if basicConfig added a handler to root
    if logger.hasHandlers():
         logger.propagate = False


    # Create client instance *after* setting logger level
    client = MCPClient(verbose=args.verbose and not args.quiet)
    try:
        await client.connect_to_sse_server(server_url=args.server)
        await client.chat_loop()
    except ConnectionError as e:
         # Log critical connection errors that prevent startup
         logger.critical(f"Failed to connect or initialize MCP session: {e}", exc_info=True)
         print(f"\nError: Could not connect to the MCP server at {args.server}. Please ensure it is running and accessible.")
    except Exception as e:
         logger.critical(f"An unrecoverable error occurred during startup or chat loop: {e}", exc_info=True)
         print(f"\nAn unexpected error occurred: {e}")
    finally:
        logger.info("Cleaning up MCP client...")
        # Ensure cleanup happens even if client wasn't fully initialized
        if 'client' in locals() and client:
             await client.cleanup()
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    # Consider adding platform-specific event loop policies if needed
    # import sys
    # if sys.platform == 'win32':
    #    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())