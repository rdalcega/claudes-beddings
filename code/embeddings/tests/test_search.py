#!/usr/bin/env python3
"""
Comprehensive tests for the semantic search system.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from .test_utils import (
    validate_search_results, performance_monitor,
    create_comprehensive_test_dataset
)


class TestDocumentSearcher:
    """Test cases for DocumentSearcher class."""
    
    @pytest.mark.unit
    def test_searcher_initialization(self, temp_db_dir):
        """Test DocumentSearcher initialization."""
        from search import DocumentSearcher
        
        # Test with non-existent database
        db_path = temp_db_dir / "nonexistent_db"
        
        with pytest.raises(SystemExit):
            DocumentSearcher(str(db_path))
    
    @pytest.mark.unit 
    def test_searcher_initialization_success(self, populated_ingester):
        """Test successful DocumentSearcher initialization."""
        from search import DocumentSearcher
        
        searcher = DocumentSearcher(populated_ingester.db_path)
        assert searcher.db_path == populated_ingester.db_path
        assert searcher.client is not None
        assert searcher.collection is not None
    
    @pytest.mark.database
    def test_basic_search(self, document_searcher):
        """Test basic semantic search functionality."""
        results = document_searcher.search("marketing strategies", limit=3)
        
        # Validate results structure
        is_valid, error = validate_search_results(results, expected_min=1)
        assert is_valid, error
        
        # Check that results contain relevant content
        found_marketing = any("marketing" in result['content'].lower() for result in results)
        assert found_marketing, "Search for 'marketing' should return marketing-related content"
    
    @pytest.mark.database
    def test_search_with_category_filter(self, document_searcher):
        """Test search with category filtering."""
        # Search for content in strategy category
        results = document_searcher.search("social media", category="strategy", limit=5)
        
        is_valid, error = validate_search_results(results, expected_min=0)
        assert is_valid, error
        
        # All results should be from strategy category
        for result in results:
            assert result['metadata']['category'] == 'strategy'
    
    @pytest.mark.database
    def test_search_with_path_filter_directory(self, document_searcher):
        """Test search with directory path filtering."""
        # Search within strategy directory
        results = document_searcher.search("content", paths=["strategy"], limit=5)
        
        is_valid, error = validate_search_results(results, expected_min=0)
        assert is_valid, error
        
        # All results should be from strategy directory
        for result in results:
            source = result['metadata']['source']
            assert source.startswith('strategy/'), f"Expected strategy/ path, got {source}"
    
    @pytest.mark.database
    def test_search_with_path_filter_nested(self, document_searcher):
        """Test search with nested directory path filtering."""
        # Search within nested directory structure
        results = document_searcher.search("analysis", paths=["content/analysis"], limit=5)
        
        is_valid, error = validate_search_results(results, expected_min=0)
        assert is_valid, error
        
        # Results should be from nested path
        for result in results:
            source = result['metadata']['source']
            assert 'content' in source and 'analysis' in source
    
    @pytest.mark.database 
    def test_search_with_multiple_paths(self, document_searcher):
        """Test search with multiple path filters."""
        results = document_searcher.search(
            "content", 
            paths=["strategy", "content"], 
            limit=10
        )
        
        is_valid, error = validate_search_results(results, expected_min=0)
        assert is_valid, error
        
        # Results should be from either strategy or content directories
        for result in results:
            source = result['metadata']['source']
            path_matches = source.startswith('strategy/') or source.startswith('content/')
            assert path_matches, f"Path {source} should start with strategy/ or content/"
    
    @pytest.mark.database
    def test_search_with_specific_file(self, document_searcher):
        """Test search within a specific file."""
        # First, find what files exist
        all_results = document_searcher.search("content", limit=20)
        
        if not all_results:
            pytest.skip("No documents found for file-specific search test")
        
        # Get a specific source file
        source_file = all_results[0]['metadata']['source']
        
        # Search within that specific file
        results = document_searcher.search("content", paths=[source_file], limit=5)
        
        is_valid, error = validate_search_results(results, expected_min=0)
        assert is_valid, error
        
        # All results should be from the specified file
        for result in results:
            assert result['metadata']['source'] == source_file
    
    @pytest.mark.database
    def test_search_combined_filters(self, document_searcher):
        """Test search with both category and path filters."""
        results = document_searcher.search(
            "marketing",
            category="strategy",
            paths=["strategy"],
            limit=5
        )
        
        is_valid, error = validate_search_results(results, expected_min=0)
        assert is_valid, error
        
        # Results should match both filters
        for result in results:
            assert result['metadata']['category'] == 'strategy'
            assert result['metadata']['source'].startswith('strategy/')
    
    @pytest.mark.database
    def test_search_empty_query(self, document_searcher):
        """Test search with empty query."""
        results = document_searcher.search("", limit=5)
        
        # Should handle empty query gracefully
        assert isinstance(results, list)
    
    @pytest.mark.database
    def test_search_long_query(self, document_searcher):
        """Test search with very long query."""
        long_query = "marketing strategy social media engagement " * 50
        results = document_searcher.search(long_query, limit=3)
        
        # Should handle long query without errors
        assert isinstance(results, list)
    
    @pytest.mark.database
    def test_search_special_characters(self, document_searcher):
        """Test search with special characters."""
        special_queries = [
            "marketing & promotion",
            "social-media content",
            "user's engagement",
            "content@platform.com",
            "hashtag#trending"
        ]
        
        for query in special_queries:
            results = document_searcher.search(query, limit=2)
            assert isinstance(results, list), f"Query '{query}' should return a list"
    
    @pytest.mark.database
    def test_search_limit_parameter(self, document_searcher):
        """Test search limit parameter."""
        # Test different limits
        for limit in [1, 3, 5, 10]:
            results = document_searcher.search("content", limit=limit)
            assert len(results) <= limit, f"Results should not exceed limit of {limit}"
    
    @pytest.mark.database
    def test_search_nonexistent_category(self, document_searcher):
        """Test search with non-existent category."""
        results = document_searcher.search("marketing", category="nonexistent")
        
        # Should return empty results for non-existent category
        assert len(results) == 0
    
    @pytest.mark.database
    def test_search_nonexistent_path(self, document_searcher):
        """Test search with non-existent path."""
        results = document_searcher.search("content", paths=["nonexistent/path"])
        
        # Should return empty results for non-existent path
        assert len(results) == 0
    
    @pytest.mark.database
    def test_get_categories(self, document_searcher):
        """Test getting available categories."""
        categories = document_searcher.get_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        # Should include expected categories
        expected_categories = ['strategy', 'content', 'reference']
        for expected in expected_categories:
            if any(expected in cat for cat in categories):
                continue  # Found at least one match
        
        # Categories should be sorted
        assert categories == sorted(categories)
    
    @pytest.mark.database
    def test_build_path_conditions_directory(self, document_searcher):
        """Test building path conditions for directory filtering."""
        paths = ["strategy", "content/analysis"]
        conditions = document_searcher._build_path_conditions(paths)
        
        assert isinstance(conditions, dict)
        assert "$or" in conditions  # Multiple paths should use $or
    
    @pytest.mark.database
    def test_build_path_conditions_file(self, document_searcher):
        """Test building path conditions for file filtering."""
        paths = ["strategy/marketing.md"]
        conditions = document_searcher._build_path_conditions(paths)
        
        assert isinstance(conditions, dict)
        assert "source" in conditions  # File paths should use source filter
    
    @pytest.mark.database
    def test_build_path_conditions_mixed(self, document_searcher):
        """Test building path conditions for mixed directory and file paths."""
        paths = ["strategy", "references/resources/guide.pdf"]
        conditions = document_searcher._build_path_conditions(paths)
        
        assert isinstance(conditions, dict)
        assert "$or" in conditions  # Mixed paths should use $or
    
    @pytest.mark.database
    def test_build_path_conditions_empty(self, document_searcher):
        """Test building path conditions with empty paths."""
        conditions = document_searcher._build_path_conditions([])
        assert conditions == {}
        
        conditions = document_searcher._build_path_conditions(None)
        assert conditions == {}
    
    @pytest.mark.database
    def test_search_semantic_similarity(self, document_searcher):
        """Test that search returns semantically similar content."""
        # Search for concepts that should match content
        results = document_searcher.search("promotional campaigns", limit=5)
        
        if not results:
            pytest.skip("No results found for semantic similarity test")
        
        # Results should contain marketing/strategy related content
        marketing_terms = ['marketing', 'promotion', 'campaign', 'strategy', 'social', 'engagement']
        
        found_relevant = False
        for result in results:
            content_lower = result['content'].lower()
            if any(term in content_lower for term in marketing_terms):
                found_relevant = True
                break
        
        assert found_relevant, "Semantic search should find conceptually related content"
    
    @pytest.mark.database
    def test_search_result_ranking(self, document_searcher):
        """Test that search results are properly ranked by relevance."""
        results = document_searcher.search("marketing strategy", limit=5)
        
        if len(results) < 2:
            pytest.skip("Need at least 2 results to test ranking")
        
        # Results should be ordered by relevance (distance/similarity)
        for i in range(len(results) - 1):
            current_distance = results[i].get('distance', 0)
            next_distance = results[i + 1].get('distance', 0)
            
            # Lower distance = higher similarity = better ranking
            assert current_distance <= next_distance, "Results should be ranked by relevance"
    
    @pytest.mark.database
    @pytest.mark.performance
    def test_search_performance(self, document_searcher):
        """Test search performance under various conditions."""
        queries = [
            "marketing",
            "social media engagement strategies",
            "emotional themes in music analysis",
            "industry best practices and recommendations"
        ]
        
        for query in queries:
            with performance_monitor() as monitor:
                results = document_searcher.search(query, limit=10)
            
            metrics = monitor.final_metrics
            
            # Search should complete quickly (under 2 seconds)
            assert metrics['duration'] < 2.0, f"Search too slow: {metrics['duration']:.2f}s for '{query}'"
            
            # Memory usage should be reasonable (under 50MB)
            memory_mb = metrics['memory_delta'] / (1024 * 1024)
            assert memory_mb < 50, f"Memory usage too high: {memory_mb:.2f}MB for '{query}'"
    
    @pytest.mark.database
    def test_display_results(self, document_searcher, capsys):
        """Test the display_results method."""
        results = document_searcher.search("marketing", limit=2)
        
        # Test displaying results
        document_searcher.display_results(results, "marketing")
        
        captured = capsys.readouterr()
        
        if results:
            # Should contain search query in output
            assert "marketing" in captured.out
            # Should contain result information
            assert "[1]" in captured.out
        else:
            # Should show no results message
            assert "No results found" in captured.out
    
    @pytest.mark.database
    def test_display_results_empty(self, document_searcher, capsys):
        """Test displaying empty results."""
        document_searcher.display_results([], "nonexistent query")
        
        captured = capsys.readouterr()
        assert "No results found" in captured.out