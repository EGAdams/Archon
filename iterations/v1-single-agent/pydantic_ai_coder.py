from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import json
import numpy as np
import pandas as pd
import lancedb

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from typing import List

# Load environment variables
load_dotenv()

# Initialize OpenAI model
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

# Configure logging
logfire.configure(send_to_logfire='if-token-present')

# Initialize LanceDB
DB_PATH = "site_pages_lancedb"
db = lancedb.connect(DB_PATH)
table = db.open_table("site_pages")

@dataclass
class PydanticAIDeps:
    openai_client: AsyncOpenAI

system_prompt = """
~~ CONTEXT: ~~
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

~~ GOAL: ~~
Your only job is to help the user create an AI agent with Pydantic AI.
The user will describe the AI agent they want to build, or if they don't, guide them towards doing so.
You will take their requirements, and then search through the Pydantic AI documentation with the tools provided
to find all the necessary information to create the AI agent with correct code.

It's important for you to search through multiple Pydantic AI documentation pages to get all the information you need.
Almost never stick to just one page - use RAG and the other documentation tools multiple times when you are creating
an AI agent from scratch for the user.

~~ STRUCTURE: ~~
When you build an AI agent from scratch, split the agent into this files and give the code for each:
- `agent.py`: The main agent file, which is where the Pydantic AI agent is defined.
- `agent_tools.py`: A tools file for the agent, which is where all the tool functions are defined. Use this for more complex agents.
- `agent_prompts.py`: A prompts file for the agent, which includes all system prompts and other prompts used by the agent. Use this when there are many prompts or large ones.
- `.env.example`: An example `.env` file - specify each variable that the user will need to fill in and a quick comment above each one for how to do so.
- `requirements.txt`: Don't include any versions, just the top level package names needed for the agent.

~~ INSTRUCTIONS: ~~
- Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before writing any code.
- When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.
- Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
- Helpful tip: when starting a new AI agent build, it's a good idea to look at the 'weather agent' in the docs as an example.
- When starting a new AI agent build, always produce the full code for the AI agent - never tell the user to finish a tool/function.
- When refining an existing AI agent build in a conversation, just share the code changes necessary.
"""

pydantic_ai_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
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

@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query using vector search.
    
    Args:
        ctx: The context including the OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the query embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        print(table.schema)  # Print schema to verify column names
        
        # Search LanceDB for relevant documents
        # results = table.search(query_embedding).where("metadata LIKE '%pydantic_ai_docs%'").limit(5).to_pandas()
        # Ensure 'embedding' is the vector column name
        # results = table.search(query_embedding, vector_column="embedding") \
        #     .where("metadata LIKE '%pydantic_ai_docs%'") \
        #     .limit(5) \
        #     .to_pandas()
        
        # Ensure query_embedding is a list of floats
        assert isinstance(query_embedding, list), "query_embedding must be a list"
        assert all(isinstance(x, float) for x in query_embedding), "query_embedding must contain only floats"

        print(lancedb.__version__)
        print(f"Vector columns: {table.vector_column_names}")

        
        # Run vector search
        results = table.search(query_embedding).where("metadata LIKE '%pydantic_ai_docs%'").limit(5).to_pandas()
        
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
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query LanceDB for unique URLs where source is 'pydantic_ai_docs'
        results = table.to_pandas()
        urls = sorted(set(results[results["metadata"].str.contains("pydantic_ai_docs")]["url"]))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the OpenAI client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query LanceDB for all chunks of this URL, ordered by chunk_number
        # results = table.to_pandas()
        # page_chunks = results[(results["url"] == url) & (results["metadata"].str.contains("pydantic_ai_docs"))]
        
        # Convert LanceDB results to a Pandas DataFrame
        results = table.to_pandas()

        # Ensure metadata is stored as a string for filtering
        results["metadata"] = results["metadata"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))

        # Debug: Print some metadata to check the format
        print(results[["url", "metadata"]].head())

        # Now filter the DataFrame
        page_chunks = results[
            (results["url"] == url) & 
            (results["metadata"].str.contains("pydantic_ai_docs", na=False))
        ]

        # Debug: Print filtered results
        print(f"Filtered {len(page_chunks)} chunks for URL: {url}")
        
        if page_chunks.empty:
            return f"No content found for URL: {url}"
        
        # Sort by chunk number and format the content
        page_chunks = page_chunks.sort_values("chunk_number")
        page_title = page_chunks.iloc[0]["title"].split(" - ")[0]  # Get main title
        formatted_content = [f"# {page_title}\n"] + list(page_chunks["content"])
        
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
    
# get user input and run
async def main():
    """
    Main function to run the agent and interact with the user.
    """
    user_input = "I want to build an agent that can tell me the weather."
    async with pydantic_ai_coder.run_stream(user_input, deps=PydanticAIDeps(openai_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))) as result:
        async for chunk in result.stream_text(delta=True):
            print(chunk, end="", flush=True)
        print("\n\n")
        print("New messages:")
        for msg in result.new_messages():
            print(msg)

if __name__ == "__main__":
    asyncio.run(main())
