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
# DB_PATH = "site_pages_lancedb"
DB_PATH = "/home/adamsl/archon/iterations/v1-single-agent/site_pages_lancedb"
db = lancedb.connect(DB_PATH)
table = db.open_table("site_pages")

@dataclass
class PydanticAIDeps:
    openai_client: AsyncOpenAI

with open("system_prompt.md", "r") as file:
    system_prompt = file.read()

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
        
        print(dir(table))

        # print(f"Vector columns: {table.vector_column_names}")

        
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
        # write formatted chunks to formatted_chunks file
        with open( "formatted_chunks.md", 'w') as file:
            file.write("\n\n".join(formatted_chunks))
        
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
        
        with open( "formatted_content.md", 'w') as file:
            file.write("\n\n".join(formatted_content))
        
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
