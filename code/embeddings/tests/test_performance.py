#!/usr/bin/env python3
"""
Performance and scalability tests for the embeddings system.
"""

import pytest
import os
import time
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .test_utils import (
    performance_monitor, create_test_files, generate_test_content,
    assert_performance_within_limits
)


class TestEmbeddingsPerformance:
    """Performance tests for embeddings system components."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_dataset_ingestion(self, temp_db_dir, test_data_dir):
        """Test ingestion performance with large datasets."""
        from ingest import DocumentIngester
        
        # Create large dataset (100 documents)
        file_specs = []
        for i in range(100):
            category = ['strategy', 'content', 'references'][i % 3]
            file_specs.append({
                'path': f'{category}/doc_{i:03d}.md',
                'content': generate_test_content('marketing', 'medium')
            })
        
        create_test_files(test_data_dir, file_specs)
        
        db_path = temp_db_dir / "large_dataset_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            with performance_monitor() as monitor:
                ingester.ingest_directory(test_data_dir)
            
            metrics = monitor.final_metrics
            
            # Performance assertions
            assert metrics['duration'] < 120.0, f"Large dataset ingestion too slow: {metrics['duration']:.2f}s"
            
            memory_mb = metrics['memory_delta'] / (1024 * 1024)
            assert memory_mb < 500, f"Memory usage too high: {memory_mb:.2f}MB"
            
            # Verify all documents were ingested
            collection_count = ingester.collection.count()
            assert collection_count >= 100, f"Expected at least 100 chunks, got {collection_count}"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.performance
    def test_search_response_time(self, populated_ingester):
        """Test search response times under various conditions."""
        from search import DocumentSearcher
        
        searcher = DocumentSearcher(populated_ingester.db_path)
        
        # Test queries with different complexities
        test_queries = [
            "music",  # Simple, common term
            "marketing strategy",  # Two terms
            "social media engagement and audience development",  # Complex query
            "emotional themes in musical composition analysis",  # Very specific query
            "promotion marketing social engagement strategy content"  # Many terms
        ]
        
        response_times = []
        
        for query in test_queries:
            with performance_monitor() as monitor:
                results = searcher.search(query, limit=10)
            
            metrics = monitor.final_metrics
            response_times.append(metrics['duration'])
            
            # Each search should complete quickly
            assert metrics['duration'] < 2.0, f"Search too slow for '{query}': {metrics['duration']:.3f}s"
            
            # Memory usage should be minimal
            memory_mb = metrics['memory_delta'] / (1024 * 1024)
            assert memory_mb < 50, f"Search memory usage too high: {memory_mb:.2f}MB"
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time:.3f}s"
    
    @pytest.mark.performance
    def test_concurrent_search_operations(self, populated_ingester):
        """Test system performance under concurrent search load."""
        from search import DocumentSearcher
        
        # Create multiple searcher instances (simulating concurrent users)
        searchers = [DocumentSearcher(populated_ingester.db_path) for _ in range(5)]
        
        queries = [
            "marketing strategy",
            "social media",
            "content creation", 
            "audience engagement",
            "promotional campaigns"
        ]
        
        def perform_search(searcher, query):
            """Perform a search and return timing info."""
            start_time = time.time()
            results = searcher.search(query, limit=5)
            end_time = time.time()
            
            return {
                'query': query,
                'duration': end_time - start_time,
                'result_count': len(results)
            }
        
        # Execute concurrent searches
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit search tasks
            futures = []
            for i, query in enumerate(queries):
                searcher = searchers[i]
                future = executor.submit(perform_search, searcher, query)
                futures.append(future)
            
            # Collect results
            search_results = []
            for future in as_completed(futures):
                search_results.append(future.result())
        
        total_time = time.time() - start_time
        
        # Concurrent searches should complete quickly
        assert total_time < 5.0, f"Concurrent searches too slow: {total_time:.2f}s"
        
        # Each individual search should still be fast
        for result in search_results:
            assert result['duration'] < 3.0, f"Individual search too slow: {result['duration']:.3f}s"
            assert result['result_count'] >= 0, "Should return valid results"
    
    @pytest.mark.performance
    def test_retrieval_scalability(self, populated_ingester):
        """Test chunk retrieval performance with varying numbers of chunks."""
        from search import DocumentSearcher
        from retrieve import ChunkRetriever
        
        searcher = DocumentSearcher(populated_ingester.db_path)
        retriever = ChunkRetriever(populated_ingester.db_path)
        
        # Get a large set of chunk IDs
        search_results = searcher.search("content", limit=50)
        
        if len(search_results) < 10:
            pytest.skip("Need at least 10 search results for scalability test")
        
        chunk_ids = [result['id'] for result in search_results]
        
        # Test retrieval with different batch sizes
        batch_sizes = [1, 5, 10, 20]
        if len(chunk_ids) >= 30:
            batch_sizes.append(30)
        
        for batch_size in batch_sizes:
            if batch_size > len(chunk_ids):
                continue
            
            test_ids = chunk_ids[:batch_size]
            
            with performance_monitor() as monitor:
                chunks = retriever.retrieve_chunks(test_ids)
            
            metrics = monitor.final_metrics
            
            # Retrieval should scale reasonably
            max_time = 0.1 * batch_size + 0.5  # Allow 0.1s per chunk + 0.5s overhead
            assert metrics['duration'] < max_time, f"Retrieval too slow for {batch_size} chunks: {metrics['duration']:.3f}s"
            
            # Should retrieve expected number of chunks
            assert len(chunks) == batch_size, f"Expected {batch_size} chunks, got {len(chunks)}"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_stability(self, temp_db_dir, test_data_dir):
        """Test memory usage stability during extended operations."""
        from ingest import DocumentIngester
        from search import DocumentSearcher
        
        # Create moderate dataset
        file_specs = []
        for i in range(50):
            file_specs.append({
                'path': f'strategy/doc_{i:02d}.md',
                'content': generate_test_content('marketing', 'small')
            })
        
        create_test_files(test_data_dir, file_specs)
        
        # Monitor memory throughout the process
        process = psutil.Process(os.getpid())
        memory_samples = []
        
        def record_memory():
            memory_samples.append(process.memory_info().rss / (1024 * 1024))  # MB
        
        record_memory()  # Initial measurement
        
        # Ingestion
        db_path = temp_db_dir / "memory_stability_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            ingester.ingest_directory(test_data_dir)
            record_memory()  # After ingestion
            
            # Multiple search operations
            searcher = DocumentSearcher(str(db_path))
            
            for i in range(20):
                searcher.search(f"marketing {i}", limit=5)
                if i % 5 == 0:
                    record_memory()
            
            # Memory should not grow indefinitely
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            memory_growth = final_memory - initial_memory
            
            # Allow some growth but not excessive
            assert memory_growth < 200, f"Memory growth too high: {memory_growth:.2f}MB"
            
            # No sample should be extremely high
            max_memory = max(memory_samples)
            assert max_memory < initial_memory + 300, f"Peak memory too high: {max_memory:.2f}MB"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.performance
    def test_database_size_efficiency(self, temp_db_dir, test_data_dir):
        """Test database storage efficiency."""
        from ingest import DocumentIngester
        
        # Create test dataset with known content size
        content_size_kb = 0
        file_specs = []
        
        for i in range(20):
            content = generate_test_content('marketing', 'medium')
            content_size_kb += len(content.encode('utf-8')) / 1024
            
            file_specs.append({
                'path': f'strategy/doc_{i:02d}.md',
                'content': content
            })
        
        create_test_files(test_data_dir, file_specs)
        
        db_path = temp_db_dir / "size_efficiency_db"
        ingester = DocumentIngester(str(db_path))
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            ingester.ingest_directory(test_data_dir)
            
            # Calculate database size
            db_size_mb = 0
            if db_path.exists():
                for item in db_path.rglob('*'):
                    if item.is_file():
                        db_size_mb += item.stat().st_size / (1024 * 1024)
            
            content_size_mb = content_size_kb / 1024
            
            # Database should not be excessively larger than content
            # Allow factor of 10 for embeddings, metadata, and indexing overhead
            max_expected_size = content_size_mb * 10
            
            assert db_size_mb < max_expected_size, f"Database too large: {db_size_mb:.2f}MB for {content_size_mb:.2f}MB content"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_test_large_queries(self, populated_ingester):
        """Stress test with many rapid queries."""
        from search import DocumentSearcher
        
        searcher = DocumentSearcher(populated_ingester.db_path)
        
        # Generate many varied queries
        query_templates = [
            "marketing strategy for {}",
            "social media {} engagement", 
            "{} content creation tips",
            "audience {} development",
            "{} promotional campaigns"
        ]
        
        terms = ["music", "artist", "band", "album", "song", "creative", "digital", "online"]
        
        queries = []
        for template in query_templates:
            for term in terms:
                queries.append(template.format(term))
        
        # Execute rapid queries
        start_time = time.time()
        successful_queries = 0
        total_results = 0
        
        with performance_monitor() as monitor:
            for query in queries[:50]:  # Limit to 50 queries for reasonable test time
                try:
                    results = searcher.search(query, limit=3)
                    successful_queries += 1
                    total_results += len(results)
                except Exception as e:
                    pytest.fail(f"Query failed: {query} - {e}")
        
        metrics = monitor.final_metrics
        
        # Stress test performance requirements
        assert metrics['duration'] < 30.0, f"Stress test too slow: {metrics['duration']:.2f}s"
        assert successful_queries == 50, f"Some queries failed: {successful_queries}/50"
        
        # Average response time should still be reasonable
        avg_response_time = metrics['duration'] / successful_queries
        assert avg_response_time < 1.0, f"Average response time under stress too high: {avg_response_time:.3f}s"
    
    @pytest.mark.performance
    def test_path_filtering_performance(self, populated_ingester):
        """Test performance impact of path filtering."""
        from search import DocumentSearcher
        
        searcher = DocumentSearcher(populated_ingester.db_path)
        
        # Test search without filtering
        with performance_monitor() as no_filter_monitor:
            no_filter_results = searcher.search("content", limit=10)
        
        # Test search with path filtering
        with performance_monitor() as path_filter_monitor:
            path_filter_results = searcher.search("content", paths=["strategy"], limit=10)
        
        # Test search with category filtering
        with performance_monitor() as category_filter_monitor:
            category_filter_results = searcher.search("content", category="strategy", limit=10)
        
        # Test search with combined filtering
        with performance_monitor() as combined_filter_monitor:
            combined_results = searcher.search("content", category="strategy", paths=["strategy"], limit=10)
        
        # Get metrics
        no_filter_time = no_filter_monitor.final_metrics['duration']
        path_filter_time = path_filter_monitor.final_metrics['duration']
        category_filter_time = category_filter_monitor.final_metrics['duration']
        combined_filter_time = combined_filter_monitor.final_metrics['duration']
        
        # All searches should complete quickly
        assert no_filter_time < 2.0, f"Unfiltered search too slow: {no_filter_time:.3f}s"
        assert path_filter_time < 2.0, f"Path filtered search too slow: {path_filter_time:.3f}s"
        assert category_filter_time < 2.0, f"Category filtered search too slow: {category_filter_time:.3f}s"
        assert combined_filter_time < 2.0, f"Combined filtered search too slow: {combined_filter_time:.3f}s"
        
        # Filtering should not add significant overhead (within 2x)
        max_acceptable_overhead = no_filter_time * 2
        assert path_filter_time < max_acceptable_overhead, "Path filtering adds too much overhead"
        assert category_filter_time < max_acceptable_overhead, "Category filtering adds too much overhead"
        assert combined_filter_time < max_acceptable_overhead, "Combined filtering adds too much overhead"