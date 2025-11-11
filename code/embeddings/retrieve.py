#!/usr/bin/env python3
"""
Retrieve full text content from ChromaDB chunks using their IDs.

WHEN TO USE THIS:
- Get the complete content of specific chunks found through search.py
- Retrieve full context that was truncated in search results
- Access original document text for detailed analysis or content creation
- Follow up on interesting search results with complete information
- Retrieve continuous sections of text spanning multiple chunks without overlap

HOW IT WORKS:
1. Takes chunk IDs (from search.py results) as input
2. Queries ChromaDB directly by ID to get full content
3. Returns complete text with metadata and source information
4. Preserves original formatting and context
5. For --section: combines multiple chunks and removes overlapping text

USAGE EXAMPLES:
- python code/embeddings/retrieve.py chunk_id_123
- python code/embeddings/retrieve.py chunk_id_123 chunk_id_456 chunk_id_789
- python code/embeddings/retrieve.py --source filename.pdf --chunks 76,79
- python code/embeddings/retrieve.py --source filename.pdf --section 75,79
- python code/embeddings/retrieve.py --help

You can get chunk IDs from the search.py script results or by using the --show-ids flag.

NOTE: Chunk numbers are 1-indexed and match what you see in search results.
If search displays "(chunk 76)", use --chunks 76 to retrieve that exact chunk.

SECTION vs CHUNKS:
- --chunks: Retrieves individual chunks separately (with overlaps intact)
- --section: Combines chunks into continuous text (removing overlaps between chunks)
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

    def find_overlap(self, text1: str, text2: str, min_overlap: int = 50) -> int:
        """Find the length of overlapping text between the end of text1 and start of text2.

        Args:
            text1: First text chunk
            text2: Second text chunk
            min_overlap: Minimum overlap length to consider

        Returns:
            Length of the overlap (0 if no significant overlap found)
        """
        max_overlap = min(len(text1), len(text2), 200)  # Check up to 200 chars

        # Search from largest to smallest overlap
        for overlap_len in range(max_overlap, min_overlap - 1, -1):
            if text1[-overlap_len:] == text2[:overlap_len]:
                return overlap_len

        return 0

    def retrieve_section(self, source_file: str, start_chunk: int, end_chunk: int) -> Optional[Dict[str, Any]]:
        """Retrieve a continuous section from start_chunk to end_chunk (inclusive) without overlaps.

        Args:
            source_file: Source file name
            start_chunk: Starting chunk number (0-indexed)
            end_chunk: Ending chunk number (0-indexed, inclusive)

        Returns:
            Dictionary with combined text and metadata, or None if chunks not found
        """
        # Get all chunks in the range
        chunk_range = list(range(start_chunk, end_chunk + 1))
        chunks = self.find_chunks_by_metadata(source_file, chunk_range)

        if not chunks:
            return None

        # Sort by chunk_index to ensure proper order
        chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))

        # Check if we have all chunks in the range
        found_indices = [c['metadata'].get('chunk_index', 0) for c in chunks]
        missing = [i for i in chunk_range if i not in found_indices]

        if missing:
            console.print(f"[yellow]Warning: Missing chunks {[i+1 for i in missing]} (1-indexed)[/yellow]")

        # Combine chunks, removing overlaps
        combined_text = chunks[0]['content']

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]['content']
            overlap_len = self.find_overlap(combined_text, current_chunk)

            if overlap_len > 0:
                # Remove the overlapping portion from the beginning of current chunk
                combined_text += current_chunk[overlap_len:]
            else:
                # No overlap found, just concatenate (with a space separator)
                combined_text += " " + current_chunk

        return {
            'text': combined_text,
            'source': chunks[0]['metadata'].get('source', 'Unknown'),
            'start_chunk': start_chunk,
            'end_chunk': end_chunk,
            'num_chunks': len(chunks),
            'missing_chunks': missing
        }

    def display_section(self, section_data: Dict[str, Any]):
        """Display a combined section in a formatted way."""

        if not section_data:
            console.print("[yellow]No section data to display.[/yellow]")
            return

        # Create header with metadata
        header = f"{section_data['source']}"
        header += f" | Chunks {section_data['start_chunk'] + 1}-{section_data['end_chunk'] + 1}"
        header += f" ({section_data['num_chunks']} chunks)"

        if section_data['missing_chunks']:
            header += f" | [yellow]Missing: {[i+1 for i in section_data['missing_chunks']]}[/yellow]"

        # Stats
        text_length = len(section_data['text'])
        word_count = len(section_data['text'].split())
        stats = f"Length: {text_length:,} chars | Words: {word_count:,}"

        # Create panel with combined content
        panel_content = f"[dim]{stats}[/dim]\n\n{section_data['text']}"

        panel = Panel(
            panel_content,
            title=header,
            title_align="left",
            border_style="blue"
        )

        console.print()
        console.print(panel)
        console.print()

@click.command()
@click.argument('chunk_ids', nargs=-1, required=False)
@click.option('--source', '-s', help='Find chunks by source file name')
@click.option('--chunks', '-c', help='Comma-separated chunk numbers (1-indexed) to retrieve from source')
@click.option('--section', help='Retrieve a range of chunks as continuous text without overlaps. Format: "start,end" (1-indexed, inclusive). Example: "75,79"')
def main(chunk_ids: tuple, source: str, chunks: str, section: str):
    """Retrieve full text content from ChromaDB chunks using their IDs or by source file.

    CHUNK_IDS: One or more chunk IDs to retrieve (space-separated)

    Examples:
    python retrieve.py chunk_id_123 chunk_id_456
    python retrieve.py --source thomson-management-marketing.pdf --chunks 75,79
    python retrieve.py --source thomson-management-marketing.pdf --section 75,79
    """

    retriever = ChunkRetriever()

    if source:
        # Handle --section flag for continuous text retrieval
        if section:
            try:
                # Parse section range (1-indexed input)
                parts = section.replace('-', ',').split(',')
                if len(parts) != 2:
                    console.print("[red]Invalid section format. Use 'start,end' or 'start-end'.[/red]")
                    console.print("[yellow]Example: --section 75,79 or --section 75-79[/yellow]")
                    return

                start_chunk = int(parts[0].strip()) - 1  # Convert to 0-indexed
                end_chunk = int(parts[1].strip()) - 1    # Convert to 0-indexed

                if start_chunk < 0 or end_chunk < 0:
                    console.print("[red]Chunk numbers must be positive integers.[/red]")
                    return

                if start_chunk > end_chunk:
                    console.print("[red]Start chunk must be <= end chunk.[/red]")
                    return

                # Retrieve and display the section
                section_data = retriever.retrieve_section(source, start_chunk, end_chunk)

                if section_data:
                    retriever.display_section(section_data)
                else:
                    console.print(f"[yellow]No chunks found for {source} in range {start_chunk+1}-{end_chunk+1}[/yellow]")

            except ValueError:
                console.print("[red]Invalid section format. Use integers only.[/red]")
                console.print("[yellow]Example: --section 75,79 or --section 75-79[/yellow]")
                return

        # Handle --chunks flag for individual chunk retrieval
        elif chunks:
            chunk_numbers = None
            try:
                # Convert from 1-based user input to 0-based internal indexing
                chunk_numbers = [int(c.strip()) - 1 for c in chunks.split(',')]
            except ValueError:
                console.print("[red]Invalid chunk numbers. Use comma-separated integers.[/red]")
                return

            found_chunks = retriever.find_chunks_by_metadata(source, chunk_numbers)
            retriever.display_chunks(found_chunks)

        # Just --source with no other flags: show all chunks from that source
        else:
            found_chunks = retriever.find_chunks_by_metadata(source, None)
            retriever.display_chunks(found_chunks)

    elif chunk_ids:
        chunks_result = retriever.retrieve_chunks(list(chunk_ids))
        retriever.display_chunks(chunks_result)

    else:
        console.print("[red]Please provide either chunk IDs or use --source option.[/red]")
        console.print("[yellow]Examples:[/yellow]")
        console.print("  python retrieve.py chunk_id_123")
        console.print("  python retrieve.py --source thomson-management-marketing.pdf --chunks 75,79")
        console.print("  python retrieve.py --source thomson-management-marketing.pdf --section 75,79")

if __name__ == "__main__":
    main()