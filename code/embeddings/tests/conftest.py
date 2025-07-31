#!/usr/bin/env python3
"""
Shared pytest fixtures for embeddings system tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import hashlib
import chromadb
from unittest.mock import patch, MagicMock
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ingest import DocumentIngester
from search import DocumentSearcher  
from retrieve import ChunkRetriever


@pytest.fixture(scope="session")  
def test_data_dir():
    """Create a temporary directory for test data that persists across the session."""
    temp_dir = tempfile.mkdtemp(prefix="embeddings_test_data_")
    test_path = Path(temp_dir)
    
    # Copy sample files from the main data directory if they exist
    main_data_dir = Path(__file__).parent / "data"
    if main_data_dir.exists():
        for sample_file in main_data_dir.glob("*"):
            if sample_file.is_file():
                shutil.copy2(sample_file, test_path)
    
    yield test_path
    shutil.rmtree(temp_dir)


def cleanup_test_databases():
    """Clean up any leftover test database directories."""
    try:
        current_dir = Path.cwd()
        test_patterns = ["test_chroma_db", "test_quick", "*test*db*"]
        
        for pattern in test_patterns:
            for test_db in current_dir.glob(f"**/{pattern}"):
                if test_db.is_dir() and ("chroma" in str(test_db).lower() or 
                                        any(p in test_db.name.lower() for p in ["test_", "_test", "_db"])):
                    try:
                        shutil.rmtree(test_db)
                        logging.info(f"Cleaned up test database: {test_db}")
                    except Exception as e:
                        logging.warning(f"Failed to clean up {test_db}: {e}")
    except Exception as e:
        logging.warning(f"Error during test database cleanup: {e}")


@pytest.fixture(scope="function")
def temp_db_dir():
    """Create a temporary directory for test databases (clean for each test)."""
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="embeddings_test_db_")
        yield Path(temp_dir)
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.warning(f"Failed to clean up temp_db_dir {temp_dir}: {e}")


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return {
        "strategy.md": """# Marketing Strategy
        
        ## Social Media Approach
        Use Instagram and TikTok for engagement.
        Focus on authentic storytelling.
        
        ## Timeline
        - Week 1: Build anticipation
        - Week 2: Release single
        - Week 3: Follow-up content
        """,
        
        "song_analysis.md": """# Song Analysis: Dark Themes
        
        This song explores themes of melancholy and introspection.
        The lyrics delve into emotional depth and vulnerability.
        
        ## Musical Elements
        - Minor key progression
        - Sparse instrumentation
        - Intimate vocal delivery
        """,
        
        "reference.md": """# Industry Best Practices
        
        According to music industry research:
        - Release singles 4-6 weeks apart
        - Maintain consistent branding
        - Engage with fan communities
        
        ## Digital Marketing
        Streaming platforms prefer consistent releases.
        """
    }


@pytest.fixture
def sample_text_files(test_data_dir, sample_markdown_content):
    """Create sample text files in the test data directory."""
    files = {}
    
    # Create directory structure
    (test_data_dir / "strategy").mkdir(parents=True, exist_ok=True)
    (test_data_dir / "content" / "analysis").mkdir(parents=True, exist_ok=True)
    (test_data_dir / "references").mkdir(parents=True, exist_ok=True)
    
    # Write files
    strategy_file = test_data_dir / "strategy" / "marketing.md"
    strategy_file.write_text(sample_markdown_content["strategy.md"])
    files["strategy/marketing.md"] = strategy_file
    
    analysis_file = test_data_dir / "content" / "analysis" / "song_themes.md"
    analysis_file.write_text(sample_markdown_content["song_analysis.md"])
    files["content/analysis/song_themes.md"] = analysis_file
    
    reference_file = test_data_dir / "references" / "industry.md"
    reference_file.write_text(sample_markdown_content["reference.md"])
    files["references/industry.md"] = reference_file
    
    return files


@pytest.fixture
def sample_pdf_content():
    """Sample content that would be in a PDF."""
    return {
        "music_business.pdf": """Music Business Handbook
        
        Chapter 1: Marketing Fundamentals
        
        The music industry has evolved significantly with digital platforms.
        Artists must now think like entrepreneurs and marketers.
        
        Key principles:
        1. Know your audience
        2. Build authentic relationships
        3. Consistent content creation
        4. Data-driven decision making
        
        Chapter 2: Promotion Strategies
        
        Effective promotion requires multiple touchpoints:
        - Social media engagement
        - Email marketing campaigns  
        - Influencer partnerships
        - Live performance opportunities
        """,
        
        "industry_report.pdf": """Annual Music Industry Report
        
        Streaming Revenue Analysis:
        - 67% of industry revenue from streaming
        - TikTok drives 75% of new music discovery
        - Playlist placement crucial for success
        
        Artist Development Trends:
        - Independent artists growing 35% annually
        - Direct-to-fan platforms gaining traction
        - Authentic storytelling drives engagement
        """
    }


@pytest.fixture
def mock_pdf_files(test_data_dir, sample_pdf_content):
    """Create mock PDF files by patching the PDF extraction."""
    files = {}
    
    # Create actual files (they won't be real PDFs, but our mock will handle that)
    (test_data_dir / "references" / "resources").mkdir(parents=True, exist_ok=True)
    
    for filename, content in sample_pdf_content.items():
        file_path = test_data_dir / "references" / "resources" / filename
        file_path.write_text("dummy pdf content")  # Dummy content
        files[f"references/resources/{filename}"] = file_path
    
    # Mock the PDF extraction to return our sample content
    def mock_extract_pdf(pdf_path):
        filename = Path(pdf_path).name
        return sample_pdf_content.get(filename, "")
    
    with patch('ingest.DocumentIngester.extract_pdf_text', side_effect=mock_extract_pdf):
        yield files


@pytest.fixture
def document_ingester(temp_db_dir):
    """Create a DocumentIngester instance with temporary database."""
    db_path = temp_db_dir / "test_chroma_db"
    return DocumentIngester(str(db_path))


@pytest.fixture
def populated_ingester(document_ingester, test_data_dir, sample_text_files, mock_pdf_files):
    """Create an ingester with sample documents already loaded."""
    # Change to test data directory for ingestion
    original_cwd = os.getcwd()
    os.chdir(str(test_data_dir))
    
    try:
        document_ingester.ingest_directory(test_data_dir)
        yield document_ingester
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def document_searcher(populated_ingester):
    """Create a DocumentSearcher instance with populated database."""
    return DocumentSearcher(populated_ingester.db_path)


@pytest.fixture
def chunk_retriever(populated_ingester):
    """Create a ChunkRetriever instance with populated database."""
    return ChunkRetriever(populated_ingester.db_path)


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            'id': 'chunk_001',
            'content': 'This is a sample chunk about music marketing strategies.',
            'metadata': {
                'source': 'strategy/marketing.md',
                'filename': 'marketing.md',
                'chunk_index': 0,
                'file_type': '.md',
                'category': 'strategy',
                'path_depth': 1,
                'parent_dir': 'strategy',
                'path_level_0': 'strategy'
            }
        },
        {
            'id': 'chunk_002', 
            'content': 'This chunk discusses dark themes in song lyrics and emotional depth.',
            'metadata': {
                'source': 'content/analysis/song_themes.md',
                'filename': 'song_themes.md',
                'chunk_index': 0,
                'file_type': '.md',
                'category': 'content',
                'path_depth': 2,
                'parent_dir': 'analysis',
                'path_level_0': 'content',
                'path_level_1': 'analysis'
            }
        }
    ]


@pytest.fixture
def large_content():
    """Generate large content for performance testing."""
    base_content = "This is a test sentence that will be repeated many times. "
    return base_content * 1000  # ~60KB of content


@pytest.fixture
def mock_sentence_transformer():
    """Mock the sentence transformer to avoid downloading models in tests."""
    with patch('ingest.SentenceTransformer') as mock_transformer:
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def performance_metrics():
    """Helper fixture for tracking performance metrics."""
    metrics = {
        'ingestion_time': [],
        'search_time': [],
        'retrieval_time': [],
        'memory_usage': []
    }
    return metrics


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    cleanup_test_databases()


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    cleanup_test_databases()


def pytest_runtest_teardown(item, nextitem):
    """Called to tear down each test item."""
    cleanup_test_databases()