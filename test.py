import os
import json
from typing import List, Dict, Annotated, TypedDict, Any
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY not found in environment variables. Please set it in a .env file.")

# Initialize the Mistral LLM
llm = ChatMistralAI(api_key=MISTRAL_API_KEY,
                    model="mistral-large-latest", temperature=0.2)


# --- 1. Define your Graph State ---
class KidMemoryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    kid_profile: Dict[str, Any]


# --- 2. Create Nodes for your Graph ---

# IMPORTANT: The node function now accepts 'config' as an argument.
# LangGraph will automatically pass the current invocation's config here.
def extract_and_update_identities(state: KidMemoryState, config: Dict[str, Any]) -> KidMemoryState:
    """
    Node to extract key identities, preferences, and names of friends/family.
    It now also infers the child's name from the thread_id if not already known.
    """
    messages = state["messages"]
    current_kid_profile = state.get("kid_profile", {})
    latest_message_content = messages[-1].content if messages else ""

    # --- NEW LOGIC: Infer name from thread_id if not already present ---
    thread_id = config["configurable"].get("thread_id")
    if thread_id and "name" not in current_kid_profile:
        # Simple parsing: assume name is before the first underscore, then capitalize
        # e.g., "leo_123" -> "Leo", "sarah" -> "Sarah"
        inferred_name = thread_id.split("_")[0].capitalize()
        if inferred_name:
            current_kid_profile["name"] = inferred_name
            # This name is now part of current_kid_profile for this turn,
            # but can still be overridden by explicit mention in chat.
    # --- END NEW LOGIC ---

    if not latest_message_content:
        return {"kid_profile": current_kid_profile}

    extraction_prompt = f"""
    You are an intelligent assistant. Your task is to extract all factual information and **any expressed preferences** (things a child states they 'like', 'love', 'enjoy', or mention as their 'favorite') about a child from the following conversation snippet.

    Specifically, look for and extract the following:
    - Child's "name" (if explicitly mentioned in the conversation, this will override any inferred name)
    - Child's "age"
    - "favorite_color", "favorite_animal", "favorite_food", "favorite_toy", "favorite_game", "favorite_activity"
    - "favorite_tv_show", "favorite_movie", "favorite_book", "favorite_character" (can be from any media like books/games/movies too)
    - Names of "friends" (extract as a list of names if multiple are mentioned, e.g., ["Sarah", "Tom"])
    - Names of "family_members" (extract as a list of names, specifying relationship if known, e.g., ["Mom (Jane)", "Sister (Emily)"])
    - Any other general "interests" or "things_enjoyed" (describe the topic or item, e.g., "playing with blocks", "drawing").

    If information is mentioned, extract it using the suggested key. If not, omit the key.
    For names of friends/family, consolidate them into a list.
    For general interests/things_enjoyed, if multiple are mentioned, make it a list.

    Return the extracted information as a JSON object.
    Example: {{"name": "Leo", "age": "5", "favorite_color": "blue", "favorite_tv_show": "Paw Patrol", "friends": ["Sarah", "Tom"], "family_members": ["Mom (Jane)", "Dad (John)"], "interests": ["playing outside", "drawing", "building castles"]}}

    Conversation snippet:
    {latest_message_content}

    Extracted information (JSON format):
    """

    extracted_info = {}
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        extracted_info_str = response.content.strip()

        # Clean up the string to ensure it's valid JSON (remove markdown code blocks)
        if extracted_info_str.startswith("```json"):
            extracted_info_str = extracted_info_str[len("```json"):].strip()
        if extracted_info_str.endswith("```"):
            extracted_info_str = extracted_info_str[:-len("```")].strip()

        if extracted_info_str:
            extracted_info = json.loads(extracted_info_str)

    except (json.JSONDecodeError, Exception) as e:
        pass

    # Merge newly extracted info with the existing profile (which now might have inferred name)
    # Start with current profile including inferred name if added
    updated_profile = {**current_kid_profile}
    for key, value in extracted_info.items():
        if key in updated_profile and isinstance(updated_profile[key], list) and isinstance(value, list):
            # If both existing and new are lists for the same key, merge and de-duplicate
            updated_profile[key] = list(set(updated_profile[key] + value))
        elif key == "name" and value:
            # If the LLM explicitly extracts a name, it overrides the inferred one.
            updated_profile[key] = value
        else:
            # For other types or new keys, just set/overwrite
            updated_profile[key] = value

    return {"kid_profile": updated_profile}


def generate_response(state: KidMemoryState) -> KidMemoryState:
    """
    Node to generate a personalized response using the LLM,
    incorporating the kid's profile from short-term memory and limiting words.
    """
    messages = state["messages"]
    kid_profile = state.get("kid_profile", {})

    # Construct a system message to inject the kid's profile into the LLM's context
    profile_facts = ", ".join(
        [f"{k}: {v}" for k, v in kid_profile.items() if v])

    if profile_facts:
        system_message_content = (
            f"You are a friendly and helpful assistant for children. "
            f"Remember these facts about the child you are talking to: {profile_facts}. "
            f"Respond in a fun, encouraging, and age-appropriate way. "
            f"Limit your response to 50 words."
        )
    else:
        system_message_content = (
            "You are a friendly and helpful assistant for children. "
            "Respond in a fun, encouraging, and age-appropriate way. "
            "Limit your response to 50 words."
        )

    # Prepend the system message to the conversation history for the LLM
    llm_input_messages = [SystemMessage(
        content=system_message_content)] + messages

    response = AIMessage(
        content="I'm sorry, I seem to be having a bit of trouble responding right now. Can you try again?")
    try:
        response = llm.invoke(llm_input_messages)
    except Exception as e:
        pass

    return {"messages": [response]}


# --- 3. Build Your LangGraph ---

def build_kid_memory_agent():
    """
    Builds and compiles the LangGraph agent with short-term memory.
    """
    builder = StateGraph(KidMemoryState)

    # Add nodes to the graph. No change here, LangGraph automatically passes 'config'
    # if the function signature for the node includes it.
    builder.add_node("extract_and_update", extract_and_update_identities)
    builder.add_node("generate_response", generate_response)

    builder.set_entry_point("extract_and_update")
    builder.add_edge("extract_and_update", "generate_response")
    builder.add_edge("generate_response", END)

    memory = InMemorySaver()
    app = builder.compile(checkpointer=memory)

    return app, memory

# --- 4. Interact with Your Agent ---


def chat_with_kid_agent():
    """
    Starts an interactive chat session with the kid memory agent.
    Supports chatting, `!recall [child_id]` to retrieve details,
    and `!switch [new_child_id]` to change the active child.
    """
    app, memory = build_kid_memory_agent()

    print("--- Kid Memory Agent Chat ---")
    print("Type 'exit' to end the chat.")
    print(
        "Type '!recall [child_id]' or '!recall[child_id]' to see details about a specific child.")
    print(
        "Type '!switch [new_child_id]' or '!switch[new_child_id]' to change the active child you're chatting with.")

    current_child_id = input(
        "Enter the Child ID for this session (e.g., 'Leo_123'): ")
    if not current_child_id:
        current_child_id = "default_child_001"
        print(f"No ID entered, using default: '{current_child_id}'")
    print(f"Starting chat with: '{current_child_id}'")

    while True:
        user_input = input(f"Child ({current_child_id}): ")
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break

        def parse_command_id(command_prefix, text_input):
            text_input_lower = text_input.lower()
            extracted_id = None

            if text_input_lower.startswith(command_prefix + ' '):
                parts = text_input.split(' ', 1)
                if len(parts) > 1:
                    extracted_id = parts[1].strip()
            elif text_input_lower.startswith(command_prefix + '['):
                start_bracket = text_input.find('[')
                end_bracket = text_input.find(']')
                if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                    extracted_id = text_input[start_bracket +
                                              1:end_bracket].strip()

            return extracted_id

        if user_input.lower().startswith('!recall'):
            query_child_id = parse_command_id('!recall', user_input)

            if query_child_id:
                queried_state = memory.get(
                    {"configurable": {"thread_id": query_child_id}})

                if queried_state and 'channel_values' in queried_state:
                    queried_kid_profile = queried_state['channel_values'].get(
                        'kid_profile', {})
                    if queried_kid_profile:
                        profile_summary_prompt = (
                            f"Here are some facts about a child with ID '{query_child_id}': "
                            f"{json.dumps(queried_kid_profile)}. "
                            f"Please summarize these details in a friendly way for an adult, "
                            f"starting with 'I remember that {query_child_id}...'. "
                            f"If there are no specific details, say 'I don't have many specific details for this child yet, but I'm ready to learn more!' "
                            f"Limit your response to 50 words."
                        )
                        summary_response = AIMessage(
                            content="I don't have specific details for this child right now.")
                        try:
                            llm_response = llm.invoke(
                                [HumanMessage(content=profile_summary_prompt)])
                            summary_response = llm_response
                        except Exception:
                            pass

                        print(f"Assistant: {summary_response.content}")
                    else:
                        print(
                            f"Assistant: I don't have any specific details for '{query_child_id}' yet.")
                else:
                    print(
                        f"Assistant: I don't have any recorded memory for child ID '{query_child_id}'.")
            else:
                print(
                    "Assistant: Please provide a child ID after '!recall' (e.g., '!recall Leo_123' or '!recall[Leo_123]').")
            continue

        if user_input.lower().startswith('!switch'):
            new_child_id = parse_command_id('!switch', user_input)

            if new_child_id:
                current_child_id = new_child_id
                print(
                    f"Assistant: Switched active child to: '{current_child_id}'.")
            else:
                print(
                    "Assistant: Please provide a new child ID after '!switch' (e.g., '!switch Sarah_456' or '!switch[Sarah_456]').")
            continue

        inputs = HumanMessage(content=user_input)

        response = app.invoke(
            {"messages": [inputs]},
            {"configurable": {"thread_id": current_child_id}}
        )

        ai_response_message = response["messages"][-1]
        print(f"Assistant: {ai_response_message.content}")


if __name__ == "__main__":
    chat_with_kid_agent()
