# Claude Embeddings Toolkit

A portable semantic search system for intelligent content discovery in any project. This toolkit enables Claude Code to maintain context and continuity across conversations through semantic search of your files.

## Features

- **🔍 Semantic Search**: Find relevant content even without exact keyword matches
- **📁 Universal File Support**: PDFs, documents, markdown, code, data files, and more
- **🎯 Smart Categorization**: Auto-detects content types (strategy, content, reference, planning)
- **🔧 Path Filtering**: Search within specific directories or files
- **📋 Chunk Retrieval**: Get full context from search results
- **📖 Diary Integration**: Structured learning and insight capture
- **🚀 Local-First**: No external APIs or cloud dependencies

## Quick Start

### 1. Setup
```bash
# Create virtual environment
python -m venv code/virtual_env
source code/virtual_env/bin/activate  # Windows: code\virtual_env\Scripts\activate

# Install dependencies
pip install -r code/requirements.txt
```

### 2. Index Your Content
```bash
# Index all files in current directory
python code/embeddings/ingest.py

# Index specific directory
python code/embeddings/ingest.py --root-dir /path/to/content
```

### 3. Search
```bash
# Basic search
python code/embeddings/search.py "your search query"

# Advanced search with filtering
python code/embeddings/search.py "marketing strategies" --category strategy
python code/embeddings/search.py "technical docs" --paths docs/
```

### 4. Retrieve Full Context
```bash
# Get full text from specific chunks
python code/embeddings/retrieve.py --source filename.pdf --chunks 10,15,20

# Get continuous text from a section (removes overlaps)
python code/embeddings/retrieve.py --source filename.pdf --section 10,20
```

## Command Reference

### Search Commands
```bash
# Basic semantic search
python code/embeddings/search.py "query"

# Search by auto-detected category
python code/embeddings/search.py "query" --category [strategy|content|reference|planning|general]

# Search specific paths/files
python code/embeddings/search.py "query" --paths "directory/"
python code/embeddings/search.py "query" --paths "*.md"

# Multiple path filtering
python code/embeddings/search.py "query" --paths "docs/" --paths "references/"

# Show chunk IDs for retrieval
python code/embeddings/search.py "query" --show-ids

# Limit results
python code/embeddings/search.py "query" --limit 10
```

### Retrieval Commands
```bash
# Retrieve specific chunks by ID
python code/embeddings/retrieve.py chunk_id_1 chunk_id_2

# Retrieve individual chunks from source file (with overlaps intact)
python code/embeddings/retrieve.py --source filename.pdf --chunks 5,10,15

# Retrieve a continuous section without overlapping text
python code/embeddings/retrieve.py --source filename.pdf --section 5,10

# Retrieve all chunks from source
python code/embeddings/retrieve.py --source filename.pdf

# Show chunk metadata
python code/embeddings/retrieve.py --show-metadata chunk_id
```

> **Note**: Chunk numbers are **1-indexed** and match the chunk numbers displayed in search results. For example, if search shows "(chunk 76)", use `--chunks 76` to retrieve that exact chunk.

> **Tip**: Use `--section` instead of `--chunks` when you want continuous text without repeated overlap portions between chunks. This is ideal for reading long passages spanning multiple chunks.

### Ingestion Commands
```bash
# Index current directory
python code/embeddings/ingest.py

# Index specific directory
python code/embeddings/ingest.py --root-dir /path/to/content

# Watch for file changes (auto-update)
python code/embeddings/ingest.py --watch

# Force re-indexing (ignore cache)
python code/embeddings/ingest.py --force-reindex

# Clean up database
python code/embeddings/ingest.py --cleanup
```

## Supported File Types

| Category | Extensions |
|----------|------------|
| **Documents** | `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf` |
| **Text** | `.md`, `.txt`, `.rst`, `.org` |
| **Data** | `.json`, `.yaml`, `.yml`, `.csv`, `.tsv` |
| **Web** | `.html`, `.xml` |
| **Code** | `.py`, `.js`, `.ts`, `.css`, and any text file |
| **Presentations** | `.tex` (LaTeX) |

## Directory Structure

```
claude-embeddings-toolkit/
├── CLAUDE.md                    # Claude behavioral instructions
├── README.md                    # This file
├── code/
│   ├── requirements.txt         # Python dependencies
│   └── embeddings/
│       ├── ingest.py           # Document indexing
│       ├── search.py           # Semantic search
│       ├── retrieve.py         # Full text retrieval
│       ├── pytest.ini          # Test configuration
│       ├── chroma_db/          # Vector database (auto-created)
│       └── tests/              # Test suite
└── diaries/                    # Learning & insight capture
    ├── claude/                 # Claude's insights and learnings
    └── user/                   # User notes and reflections
```

## Configuration

### Database Location
- **Vector Database**: `code/embeddings/chroma_db/`
- **Ingestion Cache**: `code/embeddings/.ingestion_cache.json`

### Environment Variables
```bash
# Optional: Set custom chunk size (default: 1000)
export EMBEDDINGS_CHUNK_SIZE=800

# Optional: Set custom overlap (default: 200)
export EMBEDDINGS_CHUNK_OVERLAP=150
```

## Testing

```bash
cd code/embeddings

# Run full test suite
pytest

# Quick tests only (skip slow integration tests)
pytest -m "not slow"

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_search.py
```

## Best Practices

### Search Strategy
1. **Start broad** without filters
2. **Use multiple search terms** for comprehensive coverage
3. **Narrow gradually** with `--category` or `--paths` if needed
4. **Retrieve full chunks** for important findings

### Performance Tips
- **Re-index periodically** when adding many new files
- **Use specific paths** for faster targeted searches
- **Clean up database** if moving/renaming files frequently

### Diary Usage
- **Document insights immediately** when discovered
- **Use consistent formatting** for easy retrieval
- **Include source context** and strategic implications

## Troubleshooting

### Common Issues

**Search returns no results:**
- Check if files are indexed: `ls code/embeddings/chroma_db/`
- Re-run ingestion: `python code/embeddings/ingest.py`
- Try broader search terms

**Import errors:**
- Activate virtual environment: `source code/virtual_env/bin/activate`
- Install dependencies: `pip install -r code/requirements.txt`

**Performance issues:**
- Limit search results: `--limit 10`
- Use path filtering: `--paths specific/directory/`
- Clean up database: `python code/embeddings/ingest.py --cleanup`

### Database Management

```bash
# Check database size
du -sh code/embeddings/chroma_db/

# Reset database (clears all indexed content)
rm -rf code/embeddings/chroma_db/
python code/embeddings/ingest.py

# Clear ingestion cache
rm code/embeddings/.ingestion_cache.json
```

## Dependencies

Core dependencies (see `code/requirements.txt`):
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings model
- `rich` - Beautiful CLI output
- `click` - Command-line interface
- `pymupdf` - PDF processing
- `python-docx` - Word document support
- `beautifulsoup4` - HTML parsing
- `watchdog` - File watching

## License

This toolkit is designed for personal use with Claude Code. Modify and distribute as needed for your projects.