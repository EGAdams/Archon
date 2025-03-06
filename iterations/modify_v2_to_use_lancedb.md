I want you to analyze the first block of code to use as a guide to modify the second block of code.
The first block of code for an autonomous AI that writes code with the use of a lancedb vector database to build an Autonomous AI agent.  Here is the Python code:
# Python Source Code to use as a guide:
```python
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

# load the system prompt into the system_prompt string from the contents of system_prompt.md
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
        
        # Ensure query_embedding is a list of floats
        assert isinstance(query_embedding, list), "query_embedding must be a list"
        assert all(isinstance(x, float) for x in query_embedding), "query_embedding must contain only floats"
        print(lancedb.__version__)
        print(dir(table))
        
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
```

The following code is the code that I want you to modify to use lancedb.  Right now, it uses supabase which I don't care for at all.  The following code has been enhanced to use a reasoner llm for its planning phase.  I want to keep this ability.  So in summary, rewrite the code below because it is better and uses langraph, but replace all of the supabase parts with lancedb using the first block of code as a guide since we know that it works.

# Enhanced Reasoning Coder to modify for use with lancedb
```python
from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
api_key = os.getenv('LLM_API_KEY', 'no-llm-api-key-provided')
model = OpenAIModel(llm, base_url=base_url, api_key=api_key)
embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

logfire.configure(send_to_logfire='if-token-present')

is_ollama = "localhost" in base_url.lower()

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str

with open("system_prompt.md", "r") as file:
    system_prompt = file.read()

pydantic_ai_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

@pydantic_ai_coder.system_prompt  
def add_reasoner_output(ctx: RunContext[str]) -> str:
    return f"""
    \n\nAdditional thoughts/instructions from the reasoner LLM. 
    This scope includes documentation pages for you to search as well: 
    {ctx.deps.reasoner_output}
    """
    
    # Add this in to get some crazy tool calling:
    # You must get ALL documentation pages listed in the scope.

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model= embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
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

async def list_documentation_pages_helper(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages.
    This is called by the list_documentation_pages tool and also externally
    to fetch documentation pages for the reasoner LLM.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []        

@pydantic_ai_coder.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_helper(ctx.deps.supabase)

@pydantic_ai_coder.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
```
