from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import json
from typing import List

import lancedb
import pandas as pd
import numpy as np

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize models
llm = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')
model = OpenAIModel(llm, base_url=base_url, api_key=api_key)

embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

# Configure logging
logfire.configure(send_to_logfire='if-token-present')

# Initialize LanceDB
DB_PATH = "/home/adamsl/archon/iterations/v1-single-agent/site_pages_lancedb"  # <-- Update path to your LanceDB directory
db = lancedb.connect(DB_PATH)
table = db.open_table("site_pages")

@dataclass
class PydanticAIDeps:
    openai_client: AsyncOpenAI
    reasoner_output: str
    db: lancedb.LanceDBConnection
    table: lancedb.LanceTable

# Load system prompt
with open("system_prompt.md", "r") as file:
    system_prompt = file.read()

# Create the pydantic_ai agent
pydantic_ai_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

@pydantic_ai_coder.system_prompt  
def add_reasoner_output(ctx: RunContext[str]) -> str:
    """
    Additional reasoner logic appended to the system prompt.
    """
    return f"""
    \n\nAdditional thoughts/instructions from the reasoner LLM. 
    This scope includes documentation pages for you to search as well: 
    {ctx.deps.reasoner_output}
    """

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG, but now using LanceDB.
    
    Args:
        ctx: The context, including the LanceDB table and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Perform a vector search in LanceDB for top 5 matches, filtering by metadata
        results = (
            ctx.deps.table.search(query_embedding)
            .where("metadata LIKE '%pydantic_ai_docs%'")
            .limit(5)
            .to_pandas()
        )
        
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

async def list_documentation_pages_helper(table: lancedb.LanceTable) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages
    from LanceDB. This is called by the list_documentation_pages tool and also 
    externally to fetch documentation pages for the reasoner LLM.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Convert the table to a DataFrame
        df = table.to_pandas()

        # Convert metadata to string (for searching)
        df["metadata"] = df["metadata"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )

        # Filter to only those records with source = 'pydantic_ai_docs'
        docs = df[df["metadata"].str.contains("pydantic_ai_docs", na=False)]

        # Extract unique URLs
        urls = sorted(set(docs["url"]))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages, using 
    LanceDB for the data query.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_helper(ctx.deps.table)   

@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the LanceDB table
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Pull entire table into a Pandas DataFrame
        df = ctx.deps.table.to_pandas()
        
        # Ensure metadata is a string for searching
        df["metadata"] = df["metadata"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )
        
        # Filter to the page's chunks that match the pydantic_ai_docs source
        page_chunks = df[
            (df["url"] == url) &
            (df["metadata"].str.contains("pydantic_ai_docs", na=False))
        ]
        
        if page_chunks.empty:
            return f"No content found for URL: {url}"
        
        # Sort by chunk_number
        page_chunks = page_chunks.sort_values("chunk_number")
        
        # Format the page content
        page_title = page_chunks.iloc[0]["title"].split(" - ")[0]
        formatted_content = [f"# {page_title}\n"] + list(page_chunks["content"])
        
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
