#!/usr/bin/env python3
"""
Retrieve full text content from ChromaDB chunks using their IDs.

WHEN TO USE THIS:
- Get the complete content of specific chunks found through search.py
- Retrieve full context that was truncated in search results
- Access original document text for detailed analysis or content creation
- Follow up on interesting search results with complete information

HOW IT WORKS:
1. Takes chunk IDs (from search.py results) as input
2. Queries ChromaDB directly by ID to get full content
3. Returns complete text with metadata and source information
4. Preserves original formatting and context

USAGE EXAMPLES:
- python code/embeddings/retrieve.py chunk_id_123
- python code/embeddings/retrieve.py chunk_id_123 chunk_id_456 chunk_id_789
- python code/embeddings/retrieve.py --help

You can get chunk IDs from the search.py script results or by using the --show-ids flag.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import click

# Suppress ChromaDB telemetry error messages
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

console = Console()

class ChunkRetriever:
    def __init__(self, db_path: str = "./code/embeddings/chroma_db"):
        self.db_path = db_path
        try:
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_collection(name="music_promotion_docs")
        except Exception as e:
            console.print(f"[red]Error connecting to database: {e}[/red]")
            console.print("[yellow]Have you run the ingestion script yet? Try: python code/embeddings/ingest.py[/yellow]")
            sys.exit(1)
    
    def retrieve_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve full text content for the given chunk IDs."""
        
        try:
            results = self.collection.get(
                ids=chunk_ids,
                include=['documents', 'metadatas']
            )
            
            # Format results
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    formatted_results.append({
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    })
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Error retrieving chunks: {e}[/red]")
            return []
    
    def display_chunks(self, chunks: List[Dict[str, Any]]):
        """Display retrieved chunks in a formatted way."""
        
        if not chunks:
            console.print("[yellow]No chunks found for the provided IDs.[/yellow]")
            return
        
        console.print(f"\n[bold blue]Retrieved {len(chunks)} chunk(s):[/bold blue]\n")
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            content = chunk['content']
            chunk_id = chunk['id']
            
            # Create header with metadata
            source = metadata.get('source', 'Unknown')
            category = metadata.get('category', 'general')
            chunk_index = metadata.get('chunk_index', 0)
            
            header = f"[{i}] {source}"
            if chunk_index > 0:
                header += f" (chunk {chunk_index + 1})"
            header += f" | {category}"
            
            # Add ID info
            id_info = f"ID: {chunk_id}"
            
            # Create panel with full content
            panel_content = f"[dim]{id_info}[/dim]\n\n{content}"
            
            panel = Panel(
                panel_content,
                title=header,
                title_align="left",
                border_style="green" if i == 1 else "dim"
            )
            
            console.print(panel)
            console.print()

    def find_chunks_by_metadata(self, source_file: str, chunk_numbers: List[int] = None) -> List[Dict[str, Any]]:
        """Find chunks by source file and optionally by chunk numbers."""
        try:
            # Get all chunks from the collection
            results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            matching_chunks = []
            if results['documents']:
                for i, metadata in enumerate(results['metadatas']):
                    if metadata.get('source', '').endswith(source_file):
                        chunk_index = metadata.get('chunk_index', 0)
                        if chunk_numbers is None or (chunk_index in chunk_numbers):
                            matching_chunks.append({
                                'id': results['ids'][i],
                                'content': results['documents'][i],
                                'metadata': metadata
                            })
            
            return matching_chunks
            
        except Exception as e:
            console.print(f"[red]Error finding chunks: {e}[/red]")
            return []

@click.command()
@click.argument('chunk_ids', nargs=-1, required=False)
@click.option('--source', '-s', help='Find chunks by source file name')
@click.option('--chunks', '-c', help='Comma-separated chunk numbers (0-indexed) to retrieve from source')
def main(chunk_ids: tuple, source: str, chunks: str):
    """Retrieve full text content from ChromaDB chunks using their IDs or by source file.
    
    CHUNK_IDS: One or more chunk IDs to retrieve (space-separated)
    
    Examples:
    python retrieve.py chunk_id_123 chunk_id_456
    python retrieve.py --source thomson-management-marketing.pdf --chunks 75,79
    """
    
    retriever = ChunkRetriever()
    
    if source:
        chunk_numbers = None
        if chunks:
            try:
                chunk_numbers = [int(c.strip()) for c in chunks.split(',')]
            except ValueError:
                console.print("[red]Invalid chunk numbers. Use comma-separated integers.[/red]")
                return
        
        found_chunks = retriever.find_chunks_by_metadata(source, chunk_numbers)
        retriever.display_chunks(found_chunks)
        
    elif chunk_ids:
        chunks_result = retriever.retrieve_chunks(list(chunk_ids))
        retriever.display_chunks(chunks_result)
        
    else:
        console.print("[red]Please provide either chunk IDs or use --source option.[/red]")
        console.print("[yellow]Examples:[/yellow]")
        console.print("  python retrieve.py chunk_id_123")
        console.print("  python retrieve.py --source thomson-management-marketing.pdf --chunks 75,79")

if __name__ == "__main__":
    main()