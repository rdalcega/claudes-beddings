#!/usr/bin/env python3
"""
Integration tests for end-to-end embeddings system workflows.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from .test_utils import (
    create_comprehensive_test_dataset, performance_monitor,
    validate_search_results, validate_chunk_structure
)


class TestEmbeddingsIntegration:
    """Integration tests for complete embeddings workflows."""
    
    @pytest.mark.integration
    def test_complete_workflow_ingest_search_retrieve(self, temp_db_dir, test_data_dir):
        """Test complete workflow: ingest documents → search → retrieve chunks."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        from retrieve import ChunkRetriever
        
        # Step 1: Create test dataset
        dataset = create_comprehensive_test_dataset(test_data_dir)
        
        # Step 2: Ingest documents
        db_path = temp_db_dir / "integration_test_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            with performance_monitor() as ingest_monitor:
                ingester.ingest_directory(test_data_dir)
            
            # Verify ingestion
            collection_count = ingester.collection.count()
            assert collection_count > 0, "Documents should be ingested"
            
            # Step 3: Search documents
            searcher = DocumentSearcher(str(db_path))
            
            with performance_monitor() as search_monitor:
                search_results = searcher.search("marketing strategy", limit=5)
            
            # Verify search results
            is_valid, error = validate_search_results(search_results, expected_min=1)
            assert is_valid, f"Search results invalid: {error}"
            
            # Step 4: Retrieve chunks
            retriever = ChunkRetriever(str(db_path))
            
            chunk_ids = [result['id'] for result in search_results[:2]]
            
            with performance_monitor() as retrieve_monitor:
                retrieved_chunks = retriever.retrieve_chunks(chunk_ids)
            
            # Verify retrieval
            assert len(retrieved_chunks) == len(chunk_ids)
            
            for chunk in retrieved_chunks:
                is_valid, error = validate_chunk_structure(chunk)
                assert is_valid, f"Retrieved chunk invalid: {error}"
            
            # Verify performance
            ingest_metrics = ingest_monitor.final_metrics
            search_metrics = search_monitor.final_metrics
            retrieve_metrics = retrieve_monitor.final_metrics
            
            assert ingest_metrics['duration'] < 30.0, "Ingestion too slow"
            assert search_metrics['duration'] < 2.0, "Search too slow"
            assert retrieve_metrics['duration'] < 1.0, "Retrieval too slow"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_path_filtering_workflow(self, temp_db_dir, test_data_dir):
        """Test workflow with path-based filtering."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        
        # Create hierarchical test data
        test_files = [
            {
                'path': 'strategy/social_media.md',
                'content': 'Social media marketing strategies for music promotion.'
            },
            {
                'path': 'content/lyrics/song.md', 
                'content': 'Song lyrics about social themes and personal growth.'
            },
            {
                'path': 'references/industry.md',
                'content': 'Industry research on social media effectiveness.'
            }
        ]
        
        from .test_utils import create_test_files
        create_test_files(test_data_dir, test_files)
        
        # Ingest
        db_path = temp_db_dir / "path_filter_test_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            ingester.ingest_directory(test_data_dir)
            
            # Search with different path filters
            searcher = DocumentSearcher(str(db_path))
            
            # Test 1: Search in strategy directory only
            strategy_results = searcher.search("social", paths=["strategy"], limit=10)
            
            for result in strategy_results:
                assert result['metadata']['source'].startswith('strategy/')
            
            # Test 2: Search in content directory only
            content_results = searcher.search("social", paths=["content"], limit=10)
            
            for result in content_results:
                assert result['metadata']['source'].startswith('content/')
            
            # Test 3: Search in multiple directories
            multi_results = searcher.search("social", paths=["strategy", "references"], limit=10)
            
            for result in multi_results:
                source = result['metadata']['source']
                assert source.startswith('strategy/') or source.startswith('references/')
            
            # Test 4: Search specific file
            file_results = searcher.search("social", paths=["strategy/social_media.md"], limit=10)
            
            for result in file_results:
                assert result['metadata']['source'] == 'strategy/social_media.md'
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_category_filtering_workflow(self, temp_db_dir, test_data_dir):
        """Test workflow with category-based filtering."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        
        # Create test data with different categories
        test_files = [
            {
                'path': 'strategy/marketing.md',
                'content': 'Marketing strategies and promotional campaigns.'
            },
            {
                'path': 'content/analysis/themes.md',
                'content': 'Analysis of thematic elements in musical compositions.'
            },
            {
                'path': 'references/research.md', 
                'content': 'Research findings on music industry trends.'
            }
        ]
        
        from .test_utils import create_test_files
        create_test_files(test_data_dir, test_files)
        
        # Ingest
        db_path = temp_db_dir / "category_test_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            ingester.ingest_directory(test_data_dir)
            
            searcher = DocumentSearcher(str(db_path))
            
            # Test category filtering
            categories = ['strategy', 'content', 'reference']
            
            for category in categories:
                results = searcher.search("content", category=category, limit=10)
                
                for result in results:
                    assert result['metadata']['category'] == category
            
            # Test combined category and path filtering
            combined_results = searcher.search(
                "analysis",
                category="content",
                paths=["content"],
                limit=10
            )
            
            for result in combined_results:
                assert result['metadata']['category'] == 'content'
                assert result['metadata']['source'].startswith('content/')
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_rebuild_database_workflow(self, temp_db_dir, test_data_dir):
        """Test database rebuild functionality."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        
        # Create initial dataset
        initial_files = [
            {
                'path': 'strategy/initial.md',
                'content': 'Initial marketing strategy document.'
            }
        ]
        
        from .test_utils import create_test_files
        created_files = create_test_files(test_data_dir, initial_files)
        
        db_path = temp_db_dir / "rebuild_test_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            # Initial ingestion
            ingester.ingest_directory(test_data_dir)
            
            searcher = DocumentSearcher(str(db_path))
            initial_results = searcher.search("initial", limit=10)
            
            assert len(initial_results) > 0, "Should find initial content"
            initial_count = ingester.collection.count()
            
            # Add more files
            additional_files = [
                {
                    'path': 'strategy/additional.md',
                    'content': 'Additional marketing strategy content.'
                }
            ]
            
            create_test_files(test_data_dir, additional_files)
            
            # Incremental ingestion
            ingester.ingest_directory(test_data_dir)
            
            updated_count = ingester.collection.count()
            assert updated_count > initial_count, "Document count should increase"
            
            # Search for new content
            additional_results = searcher.search("additional", limit=10)
            assert len(additional_results) > 0, "Should find additional content"
            
            # Test rebuild functionality by simulating --rebuild flag
            # (In real usage, this would delete and recreate the database)
            old_count = ingester.collection.count()
            
            # Create new ingester instance to simulate rebuild
            rebuild_ingester = DocumentIngester(str(db_path))
            rebuild_ingester.ingest_directory(test_data_dir)
            
            rebuilt_count = rebuild_ingester.collection.count()
            
            # After rebuild, should still have all content
            assert rebuilt_count >= old_count
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_large_document_workflow(self, temp_db_dir, test_data_dir):
        """Test workflow with large documents that require chunking."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        from retrieve import ChunkRetriever
        
        # Create large document
        large_content = """
        # Large Marketing Strategy Document
        
        ## Introduction
        This is a comprehensive marketing strategy document that will be split into multiple chunks
        during the ingestion process to test the chunking functionality.
        
        ## Section 1: Social Media Strategy
        Social media platforms are crucial for music promotion in the modern era.
        Instagram, TikTok, and Twitter provide direct access to fan communities.
        Consistent posting schedules help maintain audience engagement.
        
        ## Section 2: Content Creation
        Creating engaging content requires understanding your audience demographics.
        Behind-the-scenes content performs well on most platforms.
        Live streaming creates intimate connections with fans.
        
        ## Section 3: Analytics and Metrics
        Tracking engagement metrics helps optimize content strategy.
        Reach, impressions, and conversion rates are key performance indicators.
        Monthly reports should analyze trends and adjust strategies accordingly.
        
        ## Section 4: Collaboration Strategies
        Collaborating with other artists expands audience reach.
        Cross-promotion benefits all parties involved.
        Guest appearances on podcasts and livestreams build credibility.
        
        ## Section 5: Long-term Planning
        Sustainable growth requires long-term strategic thinking.
        Building authentic relationships takes time and consistency.
        Success metrics should align with artistic goals and values.
        """ * 5  # Make it even larger
        
        large_file = [
            {
                'path': 'strategy/comprehensive_strategy.md',
                'content': large_content
            }
        ]
        
        from .test_utils import create_test_files
        create_test_files(test_data_dir, large_file)
        
        # Ingest
        db_path = temp_db_dir / "large_doc_test_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            with performance_monitor() as ingest_monitor:
                ingester.ingest_directory(test_data_dir)
            
            # Verify chunking occurred
            collection_count = ingester.collection.count()
            assert collection_count > 1, "Large document should be split into multiple chunks"
            
            # Search across chunks
            searcher = DocumentSearcher(str(db_path))
            
            # Search for content from different sections
            section_searches = [
                ("social media", "Section 1"),
                ("content creation", "Section 2"), 
                ("analytics", "Section 3"),
                ("collaboration", "Section 4"),
                ("long-term", "Section 5")
            ]
            
            retriever = ChunkRetriever(str(db_path))
            
            for query, expected_section in section_searches:
                results = searcher.search(query, limit=5)
                
                if results:
                    # Should find relevant content
                    found_section = any(expected_section.lower() in result['content'].lower() 
                                      for result in results)
                    
                    # Retrieve full chunks to verify content
                    chunk_ids = [result['id'] for result in results[:1]]
                    chunks = retriever.retrieve_chunks(chunk_ids)
                    
                    assert len(chunks) > 0, f"Should retrieve chunks for query: {query}"
            
            # Verify performance with large documents
            metrics = ingest_monitor.final_metrics
            assert metrics['duration'] < 60.0, "Large document ingestion should complete within 1 minute"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_error_recovery_workflow(self, temp_db_dir, test_data_dir):
        """Test system behavior with problematic files."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        
        # Create mix of good and problematic files
        test_files = [
            {
                'path': 'strategy/good_file.md',
                'content': 'This is a valid marketing strategy document.'
            },
            {
                'path': 'strategy/empty_file.md',
                'content': ''  # Empty file
            },
            {
                'path': 'strategy/unicode_file.md',
                'content': 'Document with unicode: café, naïve, résumé, 你好'
            }
        ]
        
        from .test_utils import create_test_files
        create_test_files(test_data_dir, test_files)
        
        # Also create a file that will cause PDF extraction to fail
        pdf_file = test_data_dir / "strategy" / "bad.pdf"
        pdf_file.write_bytes(b"This is not a valid PDF file")
        
        db_path = temp_db_dir / "error_recovery_test_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            # Ingestion should continue despite problematic files
            ingester.ingest_directory(test_data_dir)
            
            # Should have ingested the valid files
            collection_count = ingester.collection.count()
            assert collection_count > 0, "Should ingest valid files despite errors"
            
            # Search should work normally
            searcher = DocumentSearcher(str(db_path))
            results = searcher.search("marketing", limit=5)
            
            # Should find content from valid files
            assert len(results) > 0, "Should find content from valid files"
            
            # Unicode content should be handled properly
            unicode_results = searcher.search("café", limit=5)
            # May or may not find unicode content depending on encoding handling
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_persistence_across_sessions(self, temp_db_dir, test_data_dir):
        """Test that data persists across different sessions."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        from retrieve import ChunkRetriever
        
        # Create test data
        test_files = [
            {
                'path': 'strategy/persistent.md',
                'content': 'This content should persist across sessions.'
            }
        ]
        
        from .test_utils import create_test_files
        create_test_files(test_data_dir, test_files)
        
        db_path = temp_db_dir / "persistence_test_db"
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            # Session 1: Ingest data
            ingester1 = DocumentIngester(str(db_path))
            ingester1.ingest_directory(test_data_dir)
            
            initial_count = ingester1.collection.count()
            assert initial_count > 0
            
            # End session 1 (ingester goes out of scope)
            del ingester1
            
            # Session 2: Search data (new instances)
            searcher2 = DocumentSearcher(str(db_path))
            results = searcher2.search("persistent", limit=5)
            
            assert len(results) > 0, "Data should persist across sessions"
            
            # Session 3: Retrieve data (new instances)
            retriever3 = ChunkRetriever(str(db_path))
            chunk_ids = [result['id'] for result in results[:1]]
            chunks = retriever3.retrieve_chunks(chunk_ids)
            
            assert len(chunks) > 0, "Should retrieve data from previous sessions"
            assert "persistent" in chunks[0]['content'].lower()
            
        finally:
            os.chdir(original_cwd)