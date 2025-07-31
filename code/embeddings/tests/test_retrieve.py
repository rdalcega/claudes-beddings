#!/usr/bin/env python3
"""
Comprehensive tests for the chunk retrieval system.
"""

import pytest
from unittest.mock import patch
from pathlib import Path

from .test_utils import validate_chunk_structure, performance_monitor


class TestChunkRetriever:
    """Test cases for ChunkRetriever class."""
    
    @pytest.mark.unit
    def test_retriever_initialization(self, temp_db_dir):
        """Test ChunkRetriever initialization."""
        from retrieve import ChunkRetriever
        
        # Test with non-existent database
        db_path = temp_db_dir / "nonexistent_db"
        
        with pytest.raises(SystemExit):
            ChunkRetriever(str(db_path))
    
    @pytest.mark.unit
    def test_retriever_initialization_success(self, populated_ingester):
        """Test successful ChunkRetriever initialization."""
        from retrieve import ChunkRetriever
        
        retriever = ChunkRetriever(populated_ingester.db_path)
        assert retriever.db_path == populated_ingester.db_path
        assert retriever.client is not None
        assert retriever.collection is not None
    
    @pytest.mark.database
    def test_retrieve_chunks_by_id(self, chunk_retriever, document_searcher):
        """Test retrieving chunks by their IDs."""
        # First, search to get some chunk IDs
        search_results = document_searcher.search("content", limit=3)
        
        if not search_results:
            pytest.skip("No search results available for chunk retrieval test")
        
        # Extract chunk IDs
        chunk_ids = [result['id'] for result in search_results[:2]]
        
        # Retrieve chunks by ID
        retrieved_chunks = chunk_retriever.retrieve_chunks(chunk_ids)
        
        assert len(retrieved_chunks) == len(chunk_ids)
        
        # Validate chunk structure
        for chunk in retrieved_chunks:
            is_valid, error = validate_chunk_structure(chunk)
            assert is_valid, error
        
        # Verify IDs match
        retrieved_ids = [chunk['id'] for chunk in retrieved_chunks]
        assert set(retrieved_ids) == set(chunk_ids)
    
    @pytest.mark.database
    def test_retrieve_chunks_nonexistent_id(self, chunk_retriever):
        """Test retrieving chunks with non-existent IDs."""
        nonexistent_ids = ["fake_id_123", "another_fake_id"]
        
        retrieved_chunks = chunk_retriever.retrieve_chunks(nonexistent_ids)
        
        # Should return empty list for non-existent IDs
        assert len(retrieved_chunks) == 0
    
    @pytest.mark.database
    def test_retrieve_chunks_mixed_ids(self, chunk_retriever, document_searcher):
        """Test retrieving chunks with mix of valid and invalid IDs."""
        # Get one valid ID
        search_results = document_searcher.search("content", limit=1)
        
        if not search_results:
            pytest.skip("No search results available for mixed ID test")
        
        valid_id = search_results[0]['id']
        invalid_id = "fake_id_123"
        
        mixed_ids = [valid_id, invalid_id]
        retrieved_chunks = chunk_retriever.retrieve_chunks(mixed_ids)
        
        # Should return only the valid chunk
        assert len(retrieved_chunks) == 1
        assert retrieved_chunks[0]['id'] == valid_id
    
    @pytest.mark.database
    def test_find_chunks_by_metadata_source_only(self, chunk_retriever, document_searcher):
        """Test finding chunks by source file only."""
        # First, find what source files exist
        search_results = document_searcher.search("content", limit=5)
        
        if not search_results:
            pytest.skip("No search results available for metadata test")
        
        # Get a source file
        source_file = search_results[0]['metadata']['source']
        filename = Path(source_file).name
        
        # Find chunks by source filename
        found_chunks = chunk_retriever.find_chunks_by_metadata(filename)
        
        assert len(found_chunks) > 0
        
        # All chunks should be from the specified source
        for chunk in found_chunks:
            assert chunk['metadata']['source'].endswith(filename)
            
            # Validate chunk structure
            is_valid, error = validate_chunk_structure(chunk)
            assert is_valid, error
    
    @pytest.mark.database
    def test_find_chunks_by_metadata_with_chunk_numbers(self, chunk_retriever, document_searcher):
        """Test finding chunks by source file and specific chunk numbers."""
        # First, find a source with multiple chunks
        search_results = document_searcher.search("content", limit=10)
        
        if not search_results:
            pytest.skip("No search results available for chunk number test")
        
        # Group results by source to find one with multiple chunks
        by_source = {}
        for result in search_results:
            source = result['metadata']['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)
        
        # Find source with multiple chunks
        multi_chunk_source = None
        for source, chunks in by_source.items():
            if len(chunks) > 1:
                multi_chunk_source = source
                break
        
        if not multi_chunk_source:
            pytest.skip("No source with multiple chunks found")
        
        filename = Path(multi_chunk_source).name
        
        # Get available chunk indices
        available_chunks = by_source[multi_chunk_source]
        chunk_indices = [chunk['metadata']['chunk_index'] for chunk in available_chunks]
        
        # Request specific chunk numbers
        requested_indices = chunk_indices[:2] if len(chunk_indices) >= 2 else chunk_indices[:1]
        
        found_chunks = chunk_retriever.find_chunks_by_metadata(filename, requested_indices)
        
        assert len(found_chunks) == len(requested_indices)
        
        # Verify correct chunks were returned
        returned_indices = [chunk['metadata']['chunk_index'] for chunk in found_chunks]
        assert set(returned_indices) == set(requested_indices)
    
    @pytest.mark.database
    def test_find_chunks_by_metadata_nonexistent_source(self, chunk_retriever):
        """Test finding chunks with non-existent source file."""
        found_chunks = chunk_retriever.find_chunks_by_metadata("nonexistent.md")
        
        # Should return empty list
        assert len(found_chunks) == 0
    
    @pytest.mark.database
    def test_find_chunks_by_metadata_nonexistent_chunk_numbers(self, chunk_retriever, document_searcher):
        """Test finding chunks with non-existent chunk numbers."""
        # Get a valid source
        search_results = document_searcher.search("content", limit=1)
        
        if not search_results:
            pytest.skip("No search results available for nonexistent chunk test")
        
        source_file = search_results[0]['metadata']['source']
        filename = Path(source_file).name
        
        # Request non-existent chunk numbers
        nonexistent_chunks = [999, 1000]
        found_chunks = chunk_retriever.find_chunks_by_metadata(filename, nonexistent_chunks)
        
        # Should return empty list
        assert len(found_chunks) == 0
    
    @pytest.mark.database
    def test_display_chunks(self, chunk_retriever, document_searcher, capsys):
        """Test the display_chunks method."""
        # Get some chunks to display
        search_results = document_searcher.search("content", limit=2)
        
        if not search_results:
            pytest.skip("No search results for display test")
        
        chunk_ids = [result['id'] for result in search_results]
        chunks = chunk_retriever.retrieve_chunks(chunk_ids)
        
        # Test displaying chunks
        chunk_retriever.display_chunks(chunks)
        
        captured = capsys.readouterr()
        
        if chunks:
            # Should contain chunk information
            assert "[1]" in captured.out
            assert "ID:" in captured.out
        else:
            # Should show no chunks message
            assert "No chunks found" in captured.out
    
    @pytest.mark.database
    def test_display_chunks_empty(self, chunk_retriever, capsys):
        """Test displaying empty chunk list."""
        chunk_retriever.display_chunks([])
        
        captured = capsys.readouterr()
        assert "No chunks found" in captured.out
    
    @pytest.mark.database
    def test_chunk_content_preservation(self, chunk_retriever, document_searcher):
        """Test that chunk content is preserved during retrieval."""
        # Search for content
        search_results = document_searcher.search("content", limit=1)
        
        if not search_results:
            pytest.skip("No search results for content preservation test")
        
        original_result = search_results[0]
        chunk_id = original_result['id']
        original_content = original_result['content']
        
        # Retrieve the same chunk
        retrieved_chunks = chunk_retriever.retrieve_chunks([chunk_id])
        
        assert len(retrieved_chunks) == 1
        retrieved_chunk = retrieved_chunks[0]
        
        # Content should be identical
        assert retrieved_chunk['content'] == original_content
        
        # Metadata should be preserved
        original_metadata = original_result['metadata']
        retrieved_metadata = retrieved_chunk['metadata']
        
        # Key metadata fields should match
        key_fields = ['source', 'filename', 'chunk_index', 'category']
        for field in key_fields:
            if field in original_metadata and field in retrieved_metadata:
                assert original_metadata[field] == retrieved_metadata[field]
    
    @pytest.mark.database
    def test_chunk_retrieval_ordering(self, chunk_retriever, document_searcher):
        """Test that chunks are retrieved in the order requested."""
        # Get multiple chunk IDs
        search_results = document_searcher.search("content", limit=5)
        
        if len(search_results) < 3:
            pytest.skip("Need at least 3 search results for ordering test")
        
        # Request chunks in specific order
        requested_ids = [result['id'] for result in search_results[:3]]
        
        retrieved_chunks = chunk_retriever.retrieve_chunks(requested_ids)
        retrieved_ids = [chunk['id'] for chunk in retrieved_chunks]
        
        # Order should be preserved
        assert retrieved_ids == requested_ids
    
    @pytest.mark.database
    @pytest.mark.performance
    def test_retrieval_performance_single_chunk(self, chunk_retriever, document_searcher):
        """Test performance of single chunk retrieval."""
        # Get a chunk ID
        search_results = document_searcher.search("content", limit=1)
        
        if not search_results:
            pytest.skip("No search results for performance test")
        
        chunk_id = search_results[0]['id']
        
        with performance_monitor() as monitor:
            retrieved_chunks = chunk_retriever.retrieve_chunks([chunk_id])
        
        metrics = monitor.final_metrics
        
        # Single chunk retrieval should be very fast
        assert metrics['duration'] < 0.5, f"Single chunk retrieval too slow: {metrics['duration']:.3f}s"
        
        # Should successfully retrieve the chunk
        assert len(retrieved_chunks) == 1
    
    @pytest.mark.database
    @pytest.mark.performance
    def test_retrieval_performance_multiple_chunks(self, chunk_retriever, document_searcher):
        """Test performance of multiple chunk retrieval."""
        # Get multiple chunk IDs
        search_results = document_searcher.search("content", limit=10)
        
        if len(search_results) < 5:
            pytest.skip("Need at least 5 search results for multiple chunk performance test")
        
        chunk_ids = [result['id'] for result in search_results[:5]]
        
        with performance_monitor() as monitor:
            retrieved_chunks = chunk_retriever.retrieve_chunks(chunk_ids)
        
        metrics = monitor.final_metrics
        
        # Multiple chunk retrieval should still be fast
        assert metrics['duration'] < 1.0, f"Multiple chunk retrieval too slow: {metrics['duration']:.3f}s"
        
        # Should retrieve all requested chunks
        assert len(retrieved_chunks) == len(chunk_ids)
    
    @pytest.mark.database
    @pytest.mark.performance
    def test_metadata_search_performance(self, chunk_retriever, document_searcher):
        """Test performance of metadata-based chunk finding."""
        # Get a source file
        search_results = document_searcher.search("content", limit=1)
        
        if not search_results:
            pytest.skip("No search results for metadata search performance test")
        
        source_file = search_results[0]['metadata']['source']
        filename = Path(source_file).name
        
        with performance_monitor() as monitor:
            found_chunks = chunk_retriever.find_chunks_by_metadata(filename)
        
        metrics = monitor.final_metrics
        
        # Metadata search should be reasonably fast
        assert metrics['duration'] < 2.0, f"Metadata search too slow: {metrics['duration']:.3f}s"
        
        # Should find at least one chunk
        assert len(found_chunks) >= 1