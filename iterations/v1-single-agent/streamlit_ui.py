from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import json
import logfire
import streamlit as st
import lancedb

from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize LanceDB
DB_PATH = "site_pages_lancedb"
db = lancedb.connect(DB_PATH)
table = db.open_table("site_pages")

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> list[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def retrieve_relevant_documentation(user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query using LanceDB vector search.
    """
    try:
        query_embedding = await get_embedding(user_query, openai_client)

        print(table.schema)  # Print schema to verify column names
        
        # Search LanceDB for relevant documents
       # results = table.search(query_embedding).where("metadata LIKE '%pydantic_ai_docs%'").limit(5).to_pandas()
        # Ensure 'embedding' is the vector column name
        results = table.search(query_embedding, vector_column="embedding") \
            .where("metadata LIKE '%pydantic_ai_docs%'") \
            .limit(5) \
            .to_pandas()

        if results.empty:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for _, doc in results.iterrows():
            chunk_text = f"""
            # {doc['title']}

            {doc['content']}
            """
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps(
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with pydantic_ai_coder.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Store the final response in session state
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("Archon - Agent Builder")
    st.write("Describe to me an AI agent you want to build and I'll code it for you with Pydantic AI.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
