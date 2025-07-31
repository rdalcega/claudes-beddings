#!/usr/bin/env python3
"""
Utility functions for embeddings system tests.
"""

import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile
import hashlib
from contextlib import contextmanager


class PerformanceMonitor:
    """Monitor performance metrics during tests."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        """Start monitoring performance."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        
        return {
            'duration': end_time - self.start_time,
            'memory_delta': end_memory - self.start_memory,
            'peak_memory': end_memory
        }


@contextmanager
def performance_monitor():
    """Context manager for monitoring performance."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        # Store metrics for test analysis
        monitor.final_metrics = metrics


def create_test_files(base_dir: Path, file_specs: List[Dict[str, Any]]) -> List[Path]:
    """
    Create test files with specified content and structure.
    
    Args:
        base_dir: Base directory to create files in
        file_specs: List of file specifications with keys:
            - path: relative path to file
            - content: file content
            - file_type: optional file type (.md, .txt, etc.)
    
    Returns:
        List of created file paths
    """
    created_files = []
    
    for spec in file_specs:
        file_path = base_dir / spec['path']
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = spec['content']
        if spec.get('file_type') == '.pdf':
            # For PDF files, just create a placeholder - our tests will mock the extraction
            content = f"PDF_PLACEHOLDER_{hashlib.md5(content.encode()).hexdigest()}"
        
        file_path.write_text(content, encoding='utf-8')
        created_files.append(file_path)
    
    return created_files


def generate_test_content(content_type: str, size: str = "small") -> str:
    """
    Generate test content of various types and sizes.
    
    Args:
        content_type: Type of content (marketing, lyrics, analysis, reference)
        size: Size of content (small, medium, large)
    
    Returns:
        Generated content string
    """
    base_contents = {
        "marketing": """# Marketing Strategy
        
        ## Social Media Campaigns
        Develop engaging content for Instagram and TikTok platforms.
        Focus on authentic storytelling and behind-the-scenes content.
        
        ## Audience Engagement
        - Daily interaction with followers
        - Weekly live streaming sessions
        - Monthly exclusive content releases
        
        ## Performance Metrics
        Track engagement rates, reach, and conversion metrics.
        """,
        
        "lyrics": """# Song Lyrics: Emotional Journey
        
        Verse 1:
        Walking through the shadows of my mind
        Looking for the pieces left behind
        Every step reveals another scar
        Wonder if I've traveled way too far
        
        Chorus:
        But I keep moving forward
        Through the pain and through the doubt
        There's a light that's calling
        Showing me the way out
        """,
        
        "analysis": """# Song Analysis: Thematic Elements
        
        ## Lyrical Themes
        This composition explores themes of introspection and personal growth.
        The narrative arc follows a journey from uncertainty to resolution.
        
        ## Musical Structure
        - Verse-Chorus-Verse-Chorus-Bridge-Chorus format
        - Minor key progression in verses
        - Major key resolution in final chorus
        
        ## Production Notes
        Layered vocals create emotional depth and intimacy.
        """,
        
        "reference": """# Industry Research: Digital Marketing
        
        ## Current Trends
        According to recent industry reports, streaming platforms drive 70% of music discovery.
        Social media engagement correlates strongly with streaming success.
        
        ## Best Practices
        1. Consistent release schedule (every 4-6 weeks)
        2. Cross-platform content strategy
        3. Data-driven decision making
        4. Direct fan engagement
        
        ## Case Studies
        Independent artists using targeted social media campaigns have seen 300% growth in streams.
        """
    }
    
    base_content = base_contents.get(content_type, base_contents["marketing"])
    
    if size == "small":
        return base_content
    elif size == "medium":
        return base_content + "\n\n" + base_content + "\n\n" + base_content
    elif size == "large":
        return "\n\n".join([base_content] * 10)
    else:
        return base_content


def create_comprehensive_test_dataset(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Create a comprehensive test dataset with various file types and structures.
    
    Returns:
        Dictionary mapping category names to lists of created files
    """
    file_specs = [
        # Strategy files
        {
            'path': 'strategy/social_media.md',
            'content': generate_test_content('marketing', 'medium'),
        },
        {
            'path': 'strategy/release_timeline.md', 
            'content': generate_test_content('marketing', 'small'),
        },
        
        # Content files
        {
            'path': 'content/lyrics/song_01.md',
            'content': generate_test_content('lyrics', 'small'),
        },
        {
            'path': 'content/analysis/theme_analysis.md',
            'content': generate_test_content('analysis', 'medium'),
        },
        
        # Reference files
        {
            'path': 'references/industry_report.md',
            'content': generate_test_content('reference', 'large'),
        },
        {
            'path': 'references/resources/marketing_guide.pdf',
            'content': generate_test_content('reference', 'medium'),
            'file_type': '.pdf'
        },
        
        # Nested structure
        {
            'path': 'assets/audio/masters/analysis/song_breakdown.md',
            'content': generate_test_content('analysis', 'small'),
        }
    ]
    
    created_files = create_test_files(base_dir, file_specs)
    
    # Categorize files by type
    categorized = {
        'strategy': [],
        'content': [], 
        'reference': [],
        'analysis': []
    }
    
    for file_path in created_files:
        path_str = str(file_path)
        if 'strategy' in path_str:
            categorized['strategy'].append(file_path)
        elif 'content' in path_str or 'lyrics' in path_str:
            categorized['content'].append(file_path)
        elif 'references' in path_str:
            categorized['reference'].append(file_path)
        elif 'analysis' in path_str:
            categorized['analysis'].append(file_path)
    
    return categorized


def validate_search_results(results: List[Dict[str, Any]], expected_min: int = 1) -> Tuple[bool, str]:
    """
    Validate search results structure and content.
    
    Args:
        results: Search results to validate
        expected_min: Minimum number of expected results
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(results) < expected_min:
        return False, f"Expected at least {expected_min} results, got {len(results)}"
    
    for i, result in enumerate(results):
        # Check required fields
        required_fields = ['content', 'metadata']
        for field in required_fields:
            if field not in result:
                return False, f"Result {i} missing required field: {field}"
        
        # Check metadata structure
        metadata = result['metadata']
        required_metadata = ['source', 'category']
        for field in required_metadata:
            if field not in metadata:
                return False, f"Result {i} metadata missing field: {field}"
        
        # Check content is not empty
        if not result['content'].strip():
            return False, f"Result {i} has empty content"
    
    return True, ""


def validate_chunk_structure(chunk: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a single chunk structure.
    
    Args:
        chunk: Chunk to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['id', 'content', 'metadata']
    for field in required_fields:
        if field not in chunk:
            return False, f"Chunk missing required field: {field}"
    
    # Validate metadata
    metadata = chunk['metadata']
    required_metadata = ['source', 'filename', 'chunk_index', 'category']
    for field in required_metadata:
        if field not in metadata:
            return False, f"Chunk metadata missing field: {field}"
    
    return True, ""


def assert_performance_within_limits(metrics: Dict[str, float], 
                                    max_duration: float = 10.0,
                                    max_memory_mb: float = 100.0):
    """
    Assert that performance metrics are within acceptable limits.
    
    Args:
        metrics: Performance metrics from PerformanceMonitor
        max_duration: Maximum allowed duration in seconds
        max_memory_mb: Maximum allowed memory usage in MB
    """
    assert metrics['duration'] <= max_duration, f"Operation took {metrics['duration']:.2f}s, expected <= {max_duration}s"
    
    memory_mb = metrics['memory_delta'] / (1024 * 1024)
    assert memory_mb <= max_memory_mb, f"Memory usage {memory_mb:.2f}MB, expected <= {max_memory_mb}MB"


def cleanup_test_dbs(base_path: Path):
    """Clean up any test databases in the given directory."""
    import shutil
    
    for item in base_path.iterdir():
        if item.is_dir() and 'test' in item.name.lower() and 'db' in item.name.lower():
            shutil.rmtree(item, ignore_errors=True)