#!/usr/bin/env python3
"""
Search through embedded documents using semantic similarity.

WHEN TO USE THIS:
- Finding content related to specific themes, songs, or strategies
- Discovering connections between different parts of your promotional materials
- Generating ideas for social media posts based on song characteristics
- Researching industry practices from your PDF resources
- Cross-referencing lyrics with promotional strategies

HOW IT WORKS:
1. Takes your search query and converts it to an embedding
2. Applies pre-query filtering using hierarchical path metadata for maximum efficiency
3. Finds the most semantically similar document chunks in your filtered database
4. Returns results with source file information and relevance scores
5. Can filter by document category (strategy, content, reference, etc.) and specific paths

USAGE EXAMPLES:
- python code/embeddings/search.py "dark themes in songs"
- python code/embeddings/search.py "social media strategy" --category strategy
- python code/embeddings/search.py "album release timeline" --limit 10
- python code/embeddings/search.py "music business marketing" --category reference
- python code/embeddings/search.py "dark themes" --paths assets/
- python code/embeddings/search.py "marketing strategy" --paths references/resources/thomson-management-marketing.pdf
- python code/embeddings/search.py "content ideas" --paths assets/ --paths references/ --category content
- python code/embeddings/search.py "song analysis" --paths assets/audio/analysis/

The search uses semantic similarity with efficient pre-query path filtering, 
so it finds relevant content even when your exact keywords don't appear in the documents.
Path filtering is applied at the database level for maximum performance.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import click

# Suppress ChromaDB telemetry error messages
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

console = Console()

class DocumentSearcher:
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
    
    def search(self, query: str, limit: int = 5, category: Optional[str] = None, paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for documents similar to the given query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            category: Filter by document category (strategy, content, reference, etc.)
            paths: Filter by specific file paths or directories (e.g., ['assets/', 'references/file.pdf'])
        """
        
        # Build where clause for filtering
        where_clause = {}
        if category:
            where_clause["category"] = category
        
        # Add path filtering using hierarchical metadata
        if paths:
            path_conditions = self._build_path_conditions(paths)
            if path_conditions:
                if where_clause:
                    # Combine category and path conditions
                    where_clause = {"$and": [where_clause, path_conditions]}
                else:
                    where_clause = path_conditions
        
        # Convert empty dict to None for ChromaDB
        where_clause = where_clause if where_clause else None
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            console.print(f"[red]Error during search: {e}[/red]")
            return []
    
    def get_categories(self) -> List[str]:
        """Get all available document categories."""
        try:
            # Get a sample to see available categories
            results = self.collection.get(limit=1000)
            categories = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'category' in metadata:
                        categories.add(metadata['category'])
            return sorted(list(categories))
        except Exception as e:
            console.print(f"[red]Error getting categories: {e}[/red]")
            return []
    
    def _build_path_conditions(self, paths: List[str]) -> Dict[str, Any]:
        """Build ChromaDB where conditions for path filtering using hierarchical metadata."""
        if not paths:
            return {}
        
        # Separate directory paths from file paths
        dir_paths = []
        file_paths = []
        
        for path in paths:
            normalized_path = path.strip().strip('/')
            if not normalized_path:
                continue
                
            # Check if this is a directory (no file extension) or specific file
            if '.' not in normalized_path.split('/')[-1]:  # Directory path
                dir_paths.append(normalized_path)
            else:  # Specific file
                file_paths.append(normalized_path)
        
        conditions = []
        
        # Handle directory path filtering using exact path level matching
        if dir_paths:
            dir_conditions = []
            for dir_path in dir_paths:
                path_parts = dir_path.split('/')
                # Create conditions for each directory path by matching path levels
                level_conditions = []
                for i, part in enumerate(path_parts):
                    level_conditions.append({f'path_level_{i}': {"$eq": part}})
                
                if len(level_conditions) == 1:
                    dir_conditions.append(level_conditions[0])
                else:
                    dir_conditions.append({"$and": level_conditions})
            
            if len(dir_conditions) == 1:
                conditions.append(dir_conditions[0])
            else:
                conditions.append({"$or": dir_conditions})
        
        # Handle specific file path filtering using source
        if file_paths:
            if len(file_paths) == 1:
                conditions.append({"source": {"$eq": file_paths[0]}})
            else:
                conditions.append({"source": {"$in": file_paths}})
        
        # Combine directory and file conditions
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$or": conditions}
        
        return {}
    
    def display_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results in a formatted way."""
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        console.print(f"\n[bold blue]Search Results for: '{query}'[/bold blue]\n")
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            content = result['content']
            distance = result.get('distance', 0.0)
            
            # Create header with metadata
            source = metadata.get('source', 'Unknown')
            category = metadata.get('category', 'general')
            chunk_index = metadata.get('chunk_index', 0)
            
            similarity_score = max(0, (1 - distance) * 100)  # Convert distance to similarity percentage
            
            header = f"[{i}] {source}"
            header += f" (chunk {chunk_index + 1})"
            header += f" | {category} | {similarity_score:.1f}% match"

            # Create panel with result
            panel = Panel(
                content,
                title=header,
                title_align="left",
                border_style="blue" if i == 1 else "dim"
            )
            
            console.print(panel)
            console.print()

@click.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results to return')
@click.option('--category', '-c', help='Filter by document category')
@click.option('--paths', '-p', multiple=True, help='Filter by specific paths (can specify multiple times)')
@click.option('--list-categories', is_flag=True, help='Show available categories')
def main(query: str, limit: int, category: Optional[str], paths: tuple, list_categories: bool):
    """Search through embedded documents using semantic similarity."""
    
    searcher = DocumentSearcher()
    
    if list_categories:
        categories = searcher.get_categories()
        console.print("[bold]Available categories:[/bold]")
        for cat in categories:
            console.print(f"  • {cat}")
        return
    
    if not query:
        console.print("[red]Please provide a search query.[/red]")
        return
    
    # Convert tuple to list for paths
    paths_list = list(paths) if paths else None
    
    results = searcher.search(query, limit=limit, category=category, paths=paths_list)
    searcher.display_results(results, query)

if __name__ == "__main__":
    main()