import os
import json
import streamlit as st  # Import Streamlit
from typing import List, Dict, Annotated, TypedDict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
# --- FIX: set_page_config() MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Kid Memory Assistant", layout="centered")
# --- END FIX ---

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MISTRAL_API_KEY = st.secrets["mistral"]["api_key"]
if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()  # Stop the Streamlit app if the key is missing

# Initialize the Mistral LLM (cached to run once)


@st.cache_resource
def get_llm():
    """Caches and returns the initialized Mistral LLM."""
    return ChatMistralAI(api_key=MISTRAL_API_KEY, model="mistral-8x7b", temperature=0.9)


llm = get_llm()


# --- 1. Define your Graph State ---
class KidMemoryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    kid_profile: Dict[str, Any]


# --- 2. Create Nodes for your Graph ---

# Node functions remain the same as in the previous console app
def extract_and_update_identities(state: KidMemoryState, config: Dict[str, Any]) -> KidMemoryState:
    """
    Node to extract key identities, preferences, and names of friends/family.
    It now also infers the child's name from the thread_id if not already known.
    """
    messages = state["messages"]
    current_kid_profile = state.get("kid_profile", {})
    latest_message_content = messages[-1].content if messages else ""

    # --- Infer name from thread_id if not already present ---
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

        if extracted_info_str.startswith("```json"):
            extracted_info_str = extracted_info_str[len("```json"):].strip()
        if extracted_info_str.endswith("```"):
            extracted_info_str = extracted_info_str[:-len("```")].strip()

        if extracted_info_str:
            extracted_info = json.loads(extracted_info_str)

    except (json.JSONDecodeError, Exception) as e:
        pass

    updated_profile = {**current_kid_profile}
    for key, value in extracted_info.items():
        if key in updated_profile and isinstance(updated_profile[key], list) and isinstance(value, list):
            updated_profile[key] = list(set(updated_profile[key] + value))
        elif key == "name" and value:
            updated_profile[key] = value
        else:
            updated_profile[key] = value

    return {"kid_profile": updated_profile}


def generate_response(state: KidMemoryState) -> KidMemoryState:
    """
    Node to generate a personalized response using the LLM,
    incorporating the kid's profile from short-term memory and limiting words.
    """
    messages = state["messages"]
    kid_profile = state.get("kid_profile", {})

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

# Cache the agent and memory to prevent re-initialization on every Streamlit rerun
@st.cache_resource
def build_kid_memory_agent():
    """
    Builds and compiles the LangGraph agent with short-term memory.
    """
    builder = StateGraph(KidMemoryState)
    builder.add_node("extract_and_update", extract_and_update_identities)
    builder.add_node("generate_response", generate_response)
    builder.set_entry_point("extract_and_update")
    builder.add_edge("extract_and_update", "generate_response")
    builder.add_edge("generate_response", END)

    memory = InMemorySaver()
    app = builder.compile(checkpointer=memory)
    return app, memory


# Initialize the LangGraph app and memory
langgraph_app, langgraph_memory = build_kid_memory_agent()


# --- Streamlit UI ---

st.title("🧒 Kid Memory Assistant")
st.markdown("Chat with the AI, and it will remember details about each child. You can switch between children or recall their stored memories!")

# --- Session State Initialization ---
if "active_child_id" not in st.session_state:
    st.session_state.active_child_id = "default_child_001"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores messages to display in UI

# Function to update the displayed chat history (used after agent response or command output)


def update_chat_display(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

# Function to get child's current profile from memory


def get_child_profile_from_memory(child_id):
    queried_state = langgraph_memory.get(
        {"configurable": {"thread_id": child_id}})
    if queried_state and 'channel_values' in queried_state:
        return queried_state['channel_values'].get('kid_profile', {})
    return {}


# --- Sidebar for Controls ---
st.sidebar.header("Controls")

# Child ID Selector
with st.sidebar.container():
    st.subheader("Current Child")
    selected_child_id = st.text_input(
        "Enter or set active Child ID:",
        value=st.session_state.active_child_id,
        key="child_id_input"
    )
    if st.button("Set Active Child"):
        st.session_state.active_child_id = selected_child_id
        update_chat_display(
            "assistant", f"Switched active child to: '{st.session_state.active_child_id}'.")
        st.rerun()  # Rerun to update chat prompt immediately

# Recall Specific Child Memory
with st.sidebar.container():
    st.subheader("Recall Memory")
    recall_id = st.text_input("Recall details for Child ID:")
    if st.button("Recall Details"):
        if recall_id:
            queried_kid_profile = get_child_profile_from_memory(recall_id)
            if queried_kid_profile:
                profile_summary_prompt = (
                    f"Here are some facts about a child with ID '{recall_id}': "
                    f"{json.dumps(queried_kid_profile)}. "
                    f"Please summarize these details in a friendly way for an adult, "
                    f"starting with 'I remember that {recall_id}...'. "
                    f"If there are no specific details, say 'I don't have many specific details for this child yet, but I'm ready to learn more!' "
                    f"Limit your response to 50 words."
                )
                summary_response_content = "I don't have specific details for this child right now."
                try:
                    llm_response = llm.invoke(
                        [HumanMessage(content=profile_summary_prompt)])
                    summary_response_content = llm_response.content
                except Exception:
                    pass  # Keep default error message
                update_chat_display("assistant", summary_response_content)
            else:
                update_chat_display(
                    "assistant", f"I don't have any specific details for '{recall_id}' yet.")
        else:
            update_chat_display(
                "assistant", "Please provide a Child ID to recall details.")

# Clear Chat Display Button
if st.sidebar.button("Clear Chat Display"):
    st.session_state.chat_history = []
    st.rerun()


# --- Main Chat Interface ---

st.header(f"Chatting with: {st.session_state.active_child_id}")

# Display chat messages
chat_container = st.container(height=400, border=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        chat_container.chat_message("user").write(msg["content"])
    else:
        chat_container.chat_message("assistant").write(msg["content"])

# User input text box
user_input = st.chat_input("Say something...")

if user_input:
    update_chat_display("user", user_input)

    with st.spinner("Thinking..."):
        try:
            response = langgraph_app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                {"configurable": {"thread_id": st.session_state.active_child_id}}
            )
            ai_response_message = response["messages"][-1]
            update_chat_display("assistant", ai_response_message.content)

        except Exception as e:
            st.error(f"An error occurred: {e}. Please try again.")
            update_chat_display(
                "assistant", "Oops! Something went wrong. Can you please try that again?")

    st.rerun()  # Rerun to refresh display and ensure auto-scroll
