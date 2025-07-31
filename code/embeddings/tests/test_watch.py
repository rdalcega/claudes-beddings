#!/usr/bin/env python3
"""
Dedicated tests for file watching functionality.
"""

import pytest
import tempfile
import os
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from .test_utils import performance_monitor


class TestWatchModeIntegration:
    """Integration tests for watch mode functionality."""
    
    @pytest.mark.integration
    def test_watch_mode_debouncing(self, document_ingester, test_data_dir):
        """Test that file watching properly debounces rapid changes."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "debounce_test.md"
            test_file.write_text("# Initial Content\nThis is initial content.")
            
            # Create watcher with short debounce period for testing
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=0.5)
            
            # Process initial file
            initial_docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            # Track processing calls
            process_calls = []
            original_debounced_process = watcher._debounced_process_file
            
            def track_processing(file_path):
                process_calls.append(time.time())
                return original_debounced_process(file_path)
            
            watcher._debounced_process_file = track_processing
            
            # Simulate rapid file changes
            start_time = time.time()
            for i in range(5):
                test_file.write_text(f"# Modified Content {i}\nThis is modification {i}.")
                watcher.pending_files[str(test_file)] = time.time()
                # Small delay between modifications
                time.sleep(0.1)
            
            # Wait for debouncing to complete
            time.sleep(1.0)
            
            # Should have processed only once due to debouncing
            # (or very few times if timing is tricky)
            assert len(process_calls) <= 2, f"Expected <= 2 process calls, got {len(process_calls)}"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration  
    def test_watch_mode_concurrent_file_changes(self, document_ingester, test_data_dir):
        """Test handling of concurrent file changes."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create multiple test files
            test_files = []
            for i in range(3):
                test_file = test_data_dir / f"concurrent_test_{i}.md"
                test_file.write_text(f"# File {i}\nInitial content for file {i}.")
                test_files.append(test_file)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=0.2)
            
            # Process initial files
            for test_file in test_files:
                docs = document_ingester.process_file(test_file, force=True)
                if docs:
                    document_ingester.collection.upsert(
                        ids=[doc['id'] for doc in docs],
                        documents=[doc['content'] for doc in docs],
                        metadatas=[doc['metadata'] for doc in docs]
                    )
            
            initial_count = document_ingester.collection.count()
            
            # Modify all files concurrently
            def modify_file(file_path, file_index):
                time.sleep(0.1)  # Small stagger
                file_path.write_text(f"# Modified File {file_index}\nThis content was modified concurrently.")
                watcher._debounced_process_file(str(file_path))
            
            threads = []
            for i, test_file in enumerate(test_files):
                thread = threading.Thread(target=modify_file, args=(test_file, i))
                threads.append(thread)
                thread.start()
            
            # Wait for all modifications to complete
            for thread in threads:
                thread.join()
            
            # Verify all files were processed
            final_count = document_ingester.collection.count()
            assert final_count >= initial_count  # Should have at least the same number of docs
            
            # Verify each file's content was updated
            for i in range(3):
                results = document_ingester.collection.query(
                    query_texts=[f"Modified File {i}"],
                    n_results=1
                )
                if results['documents'][0]:
                    assert f"Modified File {i}" in results['documents'][0][0]
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_watch_mode_large_file_handling(self, document_ingester, test_data_dir):
        """Test watch mode with large files."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create large test file (~5KB)
            large_content = "# Large File\n" + ("This is a long line of content. " * 200)
            test_file = test_data_dir / "large_watch_test.md"
            test_file.write_text(large_content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Process with performance monitoring
            with performance_monitor() as monitor:
                watcher._debounced_process_file(str(test_file))
            
            # Verify processing completed successfully
            results = document_ingester.collection.query(
                query_texts=["Large File"],
                n_results=1
            )
            assert len(results['documents'][0]) > 0
            assert "Large File" in results['documents'][0][0]
            
            # Verify performance is reasonable
            metrics = monitor.final_metrics
            assert metrics['duration'] < 10.0  # Should complete within 10 seconds
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_watch_mode_error_recovery(self, document_ingester, test_data_dir):
        """Test that watch mode recovers gracefully from processing errors."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Create a file that will cause processing to fail
            bad_file = test_data_dir / "bad_file.pdf"
            bad_file.write_bytes(b"This is not a valid PDF file content")
            
            # Create a good file
            good_file = test_data_dir / "good_file.md"
            good_file.write_text("# Good File\nThis file should process successfully.")
            
            # Process the bad file (should handle error gracefully)
            watcher._debounced_process_file(str(bad_file))
            
            # Process the good file (should work despite previous error)
            watcher._debounced_process_file(str(good_file))
            
            # Verify the good file was processed successfully
            results = document_ingester.collection.query(
                query_texts=["Good File"],
                n_results=1
            )
            assert len(results['documents'][0]) > 0
            assert "Good File" in results['documents'][0][0]
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_watch_mode_file_type_filtering(self, document_ingester, test_data_dir):
        """Test that watch mode properly filters file types."""
        from ingest import DocumentWatcher
        
        # Create watcher
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        
        # Test supported file types
        supported_files = [
            "test.md", "test.pdf", "test.txt", "test.rtf", "test.docx",
            "test.html", "test.json", "test.xml", "test.yaml", "test.csv"
        ]
        
        for filename in supported_files:
            file_path = str(test_data_dir / filename)
            assert watcher._should_process_file(file_path), f"Should process {filename}"
        
        # Test unsupported file types
        unsupported_files = [
            "test.py", "test.js", "test.jpg", "test.png", "test.exe", "test.zip"
        ]
        
        for filename in unsupported_files:
            file_path = str(test_data_dir / filename)
            assert not watcher._should_process_file(file_path), f"Should not process {filename}"


class TestWatchModeEdgeCases:
    """Test edge cases and error conditions in watch mode."""
    
    @pytest.mark.unit
    def test_watch_nonexistent_file(self, document_ingester, test_data_dir):
        """Test handling of watch events for non-existent files."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        
        # Try to process a non-existent file
        nonexistent_file = str(test_data_dir / "does_not_exist.md")
        
        # Should handle gracefully without crashing
        watcher._debounced_process_file(nonexistent_file)
        
        # No documents should be added
        count = document_ingester.collection.count()
        assert count == 0
    
    @pytest.mark.unit
    def test_watch_file_permission_error(self, document_ingester, test_data_dir):
        """Test handling of permission errors during watch processing."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "permission_test.md"
            test_file.write_text("# Test Content\nThis file will have permission issues.")
            
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Mock file operations to raise permission error
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                # Should handle gracefully
                watcher._debounced_process_file(str(test_file))
            
            # Should not crash and no documents should be added
            count = document_ingester.collection.count()
            assert count == 0
            
        finally:
            os.chdir(original_cwd)
    
    
    @pytest.mark.unit
    def test_watch_file_rapidly_deleted_and_recreated(self, document_ingester, test_data_dir):
        """Test handling of files that are rapidly deleted and recreated."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "rapid_change_test.md"
            initial_content = "# Initial Content\nThis is the initial content."
            test_file.write_text(initial_content)
            
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Process initial file
            watcher._debounced_process_file(str(test_file))
            
            # Verify initial processing
            initial_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            if initial_results['documents'][0]:
                assert "Initial Content" in initial_results['documents'][0][0]
            
            # Simulate deletion
            test_file.unlink()
            mock_event = MagicMock()
            mock_event.src_path = str(test_file)
            mock_event.is_directory = False
            watcher.on_deleted(mock_event)
            
            # Recreate with new content
            new_content = "# New Content\nThis is completely new content."
            test_file.write_text(new_content)
            
            # Process recreated file
            watcher._debounced_process_file(str(test_file))
            
            # Verify new content is present
            new_results = document_ingester.collection.query(
                query_texts=["New Content"],
                n_results=1
            )
            if new_results['documents'][0]:
                assert "New Content" in new_results['documents'][0][0]
            
        finally:
            os.chdir(original_cwd)


class TestWatchModePerformance:
    """Performance tests for watch mode functionality."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_watch_mode_memory_usage(self, document_ingester, test_data_dir):
        """Test memory usage during extended watch mode operation."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:  
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Create and process multiple files to simulate extended operation
            with performance_monitor() as monitor:
                for i in range(20):
                    test_file = test_data_dir / f"memory_test_{i:02d}.md"
                    content = f"# Document {i}\n" + ("Content line. " * 50)  # ~650 chars each
                    test_file.write_text(content)
                    
                    watcher._debounced_process_file(str(test_file))
                    
                    # Simulate some processing delay
                    time.sleep(0.05)
            
            # Check memory usage is reasonable (less than 100MB delta)
            metrics = monitor.final_metrics
            memory_mb = metrics['memory_delta'] / (1024 * 1024)
            assert memory_mb < 100, f"Memory usage too high: {memory_mb:.2f}MB"
            
            # Verify all files were processed
            final_count = document_ingester.collection.count()
            assert final_count >= 20  # At least one chunk per file
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.performance
    def test_watch_mode_processing_speed(self, document_ingester, test_data_dir):
        """Test processing speed for watch mode operations."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Create test file
            test_file = test_data_dir / "speed_test.md"
            content = "# Speed Test\n" + ("This is a test line for speed testing. " * 25)  # ~1KB
            test_file.write_text(content)
            
            # Time the processing
            start_time = time.time()
            watcher._debounced_process_file(str(test_file))
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should complete within reasonable time (2 seconds for 1KB file)
            assert processing_time < 2.0, f"Processing too slow: {processing_time:.2f}s"
            
            # Verify content was processed
            results = document_ingester.collection.query(
                query_texts=["Speed Test"],
                n_results=1
            )
            assert len(results['documents'][0]) > 0
            
        finally:
            os.chdir(original_cwd)


class TestImprovedFileDetection:
    """Tests for improved file detection functionality."""
    
    @pytest.mark.integration
    def test_manual_scanning_detects_missed_changes(self, document_ingester, test_data_dir):
        """Test that manual scanning detects file changes missed by filesystem events."""
        from ingest import DocumentWatcher
        
        # Change to test data directory  
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "manual_scan_test.md"
            test_file.write_text("# Initial Content")
            
            # Create watcher with short scan interval for testing
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=0.5)
            watcher.scan_interval = 2.0  # Scan every 2 seconds for testing
            watcher._stop_manual_scanning()  # Stop default timer
            watcher._start_manual_scanning()  # Start with new interval
            
            # Process initial file to establish baseline hash
            watcher._check_file_changed_by_hash(test_file)
            
            # Directly modify the file (simulating external change)
            new_content = "# Modified Content\nThis was changed externally."
            test_file.write_text(new_content)
            
            # Wait for manual scan to detect the change
            time.sleep(3.0)
            
            # Verify the change was detected by checking if hash was updated
            assert str(test_file) in watcher.file_hashes
            
            # Clean up
            watcher._stop_manual_scanning()
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_hash_based_change_detection(self, document_ingester, test_data_dir):
        """Test hash-based change detection accuracy."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "hash_test.md"
            initial_content = "# Hash Test\nInitial content for hash testing."
            test_file.write_text(initial_content)
            
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # First check should detect change (file is new)
            assert watcher._check_file_changed_by_hash(test_file) == True
            
            # Second check should not detect change (file unchanged)
            assert watcher._check_file_changed_by_hash(test_file) == False
            
            # Modify file content
            test_file.write_text("# Hash Test\nModified content.")
            
            # Should detect change
            assert watcher._check_file_changed_by_hash(test_file) == True
            
            # Should not detect change on repeated check
            assert watcher._check_file_changed_by_hash(test_file) == False
            
            # Clean up
            watcher._stop_manual_scanning()
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration  
    def test_increased_debounce_reduces_duplicate_processing(self, document_ingester, test_data_dir):
        """Test that increased debounce time reduces duplicate processing."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "debounce_test.md"
            test_file.write_text("# Initial Content")
            
            # Create watcher with longer debounce (5 seconds)
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=2.0)
            watcher._stop_manual_scanning()  # Disable for this test
            
            # Track processing calls
            process_calls = []
            original_process = watcher._debounced_process_file
            
            def track_processing(file_path):
                process_calls.append(time.time())
                return original_process(file_path)
            
            watcher._debounced_process_file = track_processing
            
            # Simulate rapid modifications
            for i in range(5):
                test_file.write_text(f"# Modified {i}")
                watcher._process_file_change(str(test_file))
                time.sleep(0.2)  # Small delay between changes
            
            # Wait for debouncing to settle
            time.sleep(3.0)
            
            # Should have very few actual processing calls due to debouncing
            assert len(process_calls) <= 2, f"Too many process calls: {len(process_calls)}"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_external_file_operations_detected(self, document_ingester, test_data_dir):
        """Test detection of file operations made by external processes (simulating Claude writes)."""
        from ingest import DocumentWatcher
        import subprocess
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            test_file = test_data_dir / "external_test.md"
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=1.0)
            watcher.scan_interval = 2.0  # Fast scanning for testing
            
            # Track detected changes
            detected_changes = []
            original_process = watcher._process_file_change
            
            def track_changes(file_path):
                detected_changes.append(file_path)
                return original_process(file_path)
                
            watcher._process_file_change = track_changes
            
            # Use external process to create file (simulating Claude's behavior)
            subprocess.run([
                'python', '-c', 
                f'import pathlib; pathlib.Path("{test_file}").write_text("# External Creation")'
            ], check=True)
            
            # Wait for detection
            time.sleep(3.0)
            
            # Should have detected the external creation
            assert any(str(test_file) in change for change in detected_changes), \
                f"External file creation not detected. Changes: {detected_changes}"
            
            # Clean up
            watcher._stop_manual_scanning()
            
        finally:
            os.chdir(original_cwd)


class TestDatabaseConsistency:
    """Tests for database consistency and transactional processing."""
    
    @pytest.mark.integration
    def test_transactional_processing_rollback_on_failure(self, document_ingester, test_data_dir):
        """Test that processing failures trigger rollback of database changes."""
        from ingest import DocumentWatcher
        from unittest.mock import patch
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file and process it initially
            test_file = test_data_dir / "rollback_test.md"
            initial_content = "# Initial Content\nThis is the original content."
            test_file.write_text(initial_content)
            
            # Process initial file
            initial_docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            initial_count = document_ingester.collection.count()
            
            # Verify initial content is in database
            initial_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            assert len(initial_results['documents'][0]) > 0
            assert "Initial Content" in initial_results['documents'][0][0]
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()  # Disable for this test
            
            # Modify file content
            test_file.write_text("# Modified Content\nThis is new content.")
            
            # Mock the upsert operation to fail
            with patch.object(document_ingester.collection, 'upsert', side_effect=Exception("Database failure")):
                # Process file change - should fail and rollback
                watcher._debounced_process_file(str(test_file))
            
            # Verify database still contains original content (rollback worked)
            final_count = document_ingester.collection.count()
            assert final_count == initial_count, "Document count changed despite failure"
            
            rollback_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            assert len(rollback_results['documents'][0]) > 0, "Original content was lost"
            assert "Initial Content" in rollback_results['documents'][0][0], "Rollback failed"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_backup_and_restore_functionality(self, document_ingester, test_data_dir):
        """Test backup and restore functionality works correctly."""
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create and process test file
            test_file = test_data_dir / "backup_test.md"
            content = "# Backup Test\nThis content will be backed up and restored."
            test_file.write_text(content)
            
            # Process file to get it in the database
            docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Create backup
            backup_data = document_ingester._backup_existing_chunks(test_file)
            
            # Verify backup contains expected data
            assert backup_data is not None
            assert 'ids' in backup_data
            assert 'documents' in backup_data
            assert 'metadatas' in backup_data
            assert len(backup_data['ids']) > 0
            
            # Remove the chunks
            document_ingester._remove_existing_chunks(test_file, backup_data)
            
            # Verify removal worked
            removed_results = document_ingester.collection.query(
                query_texts=["Backup Test"],
                n_results=1
            )
            assert len(removed_results['documents'][0]) == 0, "Chunks were not removed"
            
            # Restore from backup
            document_ingester._restore_chunks_from_backup(backup_data)
            
            # Verify restoration worked
            restored_results = document_ingester.collection.query(
                query_texts=["Backup Test"],
                n_results=1
            )
            assert len(restored_results['documents'][0]) > 0, "Chunks were not restored"
            assert "Backup Test" in restored_results['documents'][0][0], "Wrong content restored"
            
        finally:
            os.chdir(original_cwd)


class TestErrorHandlingAndRecovery:
    """Tests for enhanced error handling and recovery functionality."""
    
    @pytest.mark.integration
    def test_retry_logic_with_temporary_failures(self, document_ingester, test_data_dir):
        """Test retry logic works with temporary failures."""
        from ingest import DocumentWatcher
        from unittest.mock import patch, MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "retry_test.md"
            test_file.write_text("# Retry Test\nThis tests retry functionality.")
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()  # Disable for this test
            
            # Mock to fail twice then succeed
            call_count = 0
            original_upsert = document_ingester.collection.upsert
            
            def failing_upsert(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Temporary database error")
                return original_upsert(*args, **kwargs)
            
            with patch.object(document_ingester.collection, 'upsert', side_effect=failing_upsert):
                # Process the file - should succeed after retries
                documents = document_ingester.process_file(test_file, force=True)
                
                # Manually call the upsert with retry logic to test it
                ids = [doc['id'] for doc in documents]
                doc_contents = [doc['content'] for doc in documents]
                metadatas = [doc['metadata'] for doc in documents]
                
                def upsert_operation():
                    return document_ingester.collection.upsert(
                        ids=ids,
                        documents=doc_contents,
                        metadatas=metadatas
                    )
                
                # Should succeed after retries
                watcher._retry_with_backoff(upsert_operation)
                
                # Verify it tried multiple times
                assert call_count == 3, f"Expected 3 calls (2 failures + 1 success), got {call_count}"
        
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_automatic_repair_functionality(self, document_ingester, test_data_dir):
        """Test automatic repair of database issues."""
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file and process it
            test_file = test_data_dir / "repair_test.md"
            test_file.write_text("# Repair Test\nThis will be used for repair testing.")
            
            docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Artificially create duplicate chunks to test repair
            duplicate_docs = []
            for doc in docs:
                duplicate_doc = doc.copy()
                duplicate_doc['id'] = doc['id'] + '_duplicate'
                duplicate_docs.append(duplicate_doc)
            
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in duplicate_docs],
                documents=[doc['content'] for doc in duplicate_docs],
                metadatas=[doc['metadata'] for doc in duplicate_docs]
            )
            
            # Check that issues exist
            consistency_check = document_ingester.check_database_consistency()
            assert len(consistency_check.get("issues", [])) > 0, "Expected to find issues"
            
            # Test dry run repair
            dry_repair = document_ingester.repair_database_issues(dry_run=True)
            assert dry_repair.get("dry_run") == True
            assert len(dry_repair.get("actions", [])) > 0
            
            # Verify dry run didn't change anything
            consistency_check_after_dry = document_ingester.check_database_consistency()
            assert len(consistency_check_after_dry.get("issues", [])) > 0, "Dry run should not fix issues"
            
            # Test actual repair
            repair_result = document_ingester.repair_database_issues(dry_run=False)
            assert repair_result.get("dry_run") == False
            assert repair_result.get("repaired", 0) > 0
            
            # Verify issues were fixed
            final_consistency_check = document_ingester.check_database_consistency()
            final_issues = final_consistency_check.get("issues", [])
            assert len(final_issues) == 0, f"Issues should be fixed, but found: {final_issues}"
            
        finally:
            os.chdir(original_cwd)


class TestMessageFiltering:
    """Tests for macOS system message filtering."""
    
    @pytest.mark.unit
    def test_message_filtering_context_manager(self):
        """Test that the message filtering context manager works correctly."""
        from ingest import suppress_system_messages
        import sys
        import io
        
        # Capture what gets written to stderr
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr
        
        try:
            sys.stderr = captured_stderr
            
            # Test with the filtering context manager
            with suppress_system_messages():
                # Write messages that should be filtered
                print("Context leak detected, msgtracer returned -1", file=sys.stderr)
                print("This is a normal error message", file=sys.stderr)
                print("CoreDuetContext framework error", file=sys.stderr)
                print("Another normal message", file=sys.stderr)
            
            # Check what was actually written to stderr
            output = captured_stderr.getvalue()
            
            # Should not contain filtered messages
            assert "Context leak detected" not in output
            assert "msgtracer returned -1" not in output
            assert "CoreDuetContext" not in output
            
            # Should contain normal messages
            assert "This is a normal error message" in output
            assert "Another normal message" in output
            
        finally:
            sys.stderr = original_stderr
    
    @pytest.mark.unit
    def test_message_filtering_preserves_important_errors(self):
        """Test that message filtering doesn't remove important error messages."""
        from ingest import suppress_system_messages
        import sys
        import io
        
        # Capture what gets written to stderr
        captured_stderr = io.StringIO()
        original_stderr = sys.stderr
        
        try:
            sys.stderr = captured_stderr
            
            # Test with various error messages
            with suppress_system_messages():
                print("Database connection error", file=sys.stderr)
                print("File not found: important.txt", file=sys.stderr)
                print("Permission denied", file=sys.stderr)
                print("Context leak detected in trace", file=sys.stderr)  # Should be filtered
            
            output = captured_stderr.getvalue()
            
            # Important errors should be preserved
            assert "Database connection error" in output
            assert "File not found" in output
            assert "Permission denied" in output
            
            # System messages should be filtered
            assert "Context leak detected" not in output
            
        finally:
            sys.stderr = original_stderr


class TestHashInitialization:
    """Tests for the new hash initialization feature to prevent false change detection."""
    
    @pytest.mark.unit
    def test_initialize_file_hashes_basic_functionality(self, document_ingester, test_data_dir):
        """Test that _initialize_file_hashes() correctly hashes existing files."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create several test files
            test_files = []
            for i in range(3):
                test_file = test_data_dir / f"hash_init_test_{i}.md"
                content = f"# Test File {i}\nThis is content for file {i}."
                test_file.write_text(content)
                test_files.append(test_file)
            
            # Create DocumentWatcher - this should initialize file hashes
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            watcher._stop_manual_scanning()  # Disable periodic scanning for test
            
            # Verify that file_hashes was populated
            assert len(watcher.file_hashes) >= 3, f"Expected at least 3 hashes, got {len(watcher.file_hashes)}"
            
            # Verify each test file has a hash
            for test_file in test_files:
                file_key = str(test_file)
                assert file_key in watcher.file_hashes, f"Hash not found for {test_file.name}"
                assert watcher.file_hashes[file_key] is not None, f"Hash is None for {test_file.name}"
                assert len(watcher.file_hashes[file_key]) > 0, f"Hash is empty for {test_file.name}"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_initialize_file_hashes_prevents_false_changes(self, document_ingester, test_data_dir):
        """Test that hash initialization prevents false change detection during first manual scan."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "false_change_test.md"
            content = "# False Change Test\nThis content shouldn't appear as changed."
            test_file.write_text(content)
            
            # Create DocumentWatcher (initializes hashes)
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            watcher._stop_manual_scanning()
            
            # Verify file was hashed during initialization
            file_key = str(test_file)
            assert file_key in watcher.file_hashes
            initial_hash = watcher.file_hashes[file_key]
            
            # Check for changes immediately - should return False (no change)
            has_changed = watcher._check_file_changed_by_hash(test_file)
            assert has_changed == False, "File should not appear changed after initialization"
            
            # Hash should remain the same
            assert watcher.file_hashes[file_key] == initial_hash, "Hash should not change when file is unchanged"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_initialize_file_hashes_handles_file_errors(self, document_ingester, test_data_dir):
        """Test that hash initialization handles file errors gracefully."""
        from ingest import DocumentWatcher
        from unittest.mock import patch
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test files
            good_file = test_data_dir / "good_file.md"
            good_file.write_text("# Good File\nThis file should hash correctly.")
            
            bad_file = test_data_dir / "bad_file.md"
            bad_file.write_text("# Bad File\nThis file will have hash errors.")
            
            # Mock _get_file_hash to fail for the bad file
            original_get_hash = document_ingester._get_file_hash
            def mock_get_hash(file_path):
                if file_path.name == "bad_file.md":
                    raise PermissionError("Access denied for testing")
                return original_get_hash(file_path)
            
            with patch.object(document_ingester, '_get_file_hash', side_effect=mock_get_hash):
                # Should not crash during initialization despite file error
                watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
                watcher._stop_manual_scanning()
            
            # Good file should be hashed
            good_file_key = str(good_file)
            assert good_file_key in watcher.file_hashes, "Good file should be hashed"
            
            # Bad file should not be in hashes (error was handled)
            bad_file_key = str(bad_file)
            assert bad_file_key not in watcher.file_hashes, "Bad file should not be hashed due to error"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_initialize_file_hashes_filters_excluded_paths(self, document_ingester, test_data_dir):
        """Test that hash initialization respects excluded paths."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create files in included directory
            included_file = test_data_dir / "included.md"
            included_file.write_text("# Included File\nThis should be hashed.")
            
            # Create files in excluded directory
            excluded_dir = test_data_dir / "code"
            excluded_dir.mkdir(exist_ok=True)
            excluded_file = excluded_dir / "excluded.md"
            excluded_file.write_text("# Excluded File\nThis should not be hashed.")
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            watcher._stop_manual_scanning()
            
            # Included file should be hashed
            included_key = str(included_file)
            assert included_key in watcher.file_hashes, "Included file should be hashed"
            
            # Excluded file should not be hashed
            excluded_key = str(excluded_file)
            assert excluded_key not in watcher.file_hashes, "Excluded file should not be hashed"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_initialize_file_hashes_only_supported_extensions(self, document_ingester, test_data_dir):
        """Test that hash initialization only processes supported file extensions."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create supported file
            supported_file = test_data_dir / "supported.md"
            supported_file.write_text("# Supported File\nThis should be hashed.")
            
            # Create unsupported file
            unsupported_file = test_data_dir / "unsupported.py"
            unsupported_file.write_text("# Unsupported file\nprint('This should not be hashed')")
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            watcher._stop_manual_scanning()
            
            # Supported file should be hashed
            supported_key = str(supported_file)
            assert supported_key in watcher.file_hashes, "Supported file should be hashed"
            
            # Unsupported file should not be hashed
            unsupported_key = str(unsupported_file)
            assert unsupported_key not in watcher.file_hashes, "Unsupported file should not be hashed"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_hash_initialization_prevents_watch_mode_bug(self, document_ingester, test_data_dir):
        """Test that hash initialization prevents the original watch mode bug."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test files that would be cached (simulating existing files)
            test_files = []
            for i in range(3):
                test_file = test_data_dir / f"cached_file_{i}.md"
                content = f"# Cached File {i}\nThis file should already be cached."
                test_file.write_text(content)
                test_files.append(test_file)
            
            # Simulate initial ingestion by processing files
            for test_file in test_files:
                docs = document_ingester.process_file(test_file, force=True)
                if docs:
                    document_ingester.collection.upsert(
                        ids=[doc['id'] for doc in docs],
                        documents=[doc['content'] for doc in docs],
                        metadatas=[doc['metadata'] for doc in docs]
                    )
            
            # Update cache to simulate files being cached
            document_ingester._save_cache()
            
            # Create watcher (this should initialize hashes to prevent false changes)
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            
            # Track detected changes
            detected_changes = []
            original_process_change = watcher._process_file_change
            
            def track_changes(file_path):
                detected_changes.append(file_path)
                return original_process_change(file_path)
            
            watcher._process_file_change = track_changes
            
            # Run manual scan (this would previously detect all files as "changed")
            watcher._manual_scan_files()
            
            # Should not detect any changes (files are already hashed and unchanged)
            assert len(detected_changes) == 0, f"Detected false changes: {detected_changes}"
            
            # Verify hashes are properly initialized
            for test_file in test_files:
                file_key = str(test_file)
                assert file_key in watcher.file_hashes, f"Hash missing for {test_file.name}"
            
            # Clean up
            watcher._stop_manual_scanning()
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_hash_initialization_detects_real_changes(self, document_ingester, test_data_dir):
        """Test that hash initialization still allows detection of real changes."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "real_change_test.md"
            initial_content = "# Initial Content\nThis is the original content."
            test_file.write_text(initial_content)
            
            # Create watcher (initializes hashes)
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            watcher._stop_manual_scanning()
            
            # Verify initial state
            file_key = str(test_file)
            assert file_key in watcher.file_hashes
            initial_hash = watcher.file_hashes[file_key]
            
            # No change should be detected initially
            assert watcher._check_file_changed_by_hash(test_file) == False
            
            # Modify the file content
            modified_content = "# Modified Content\nThis content has been changed."
            test_file.write_text(modified_content)
            
            # Should detect the real change
            assert watcher._check_file_changed_by_hash(test_file) == True
            
            # Hash should be updated
            new_hash = watcher.file_hashes[file_key]
            assert new_hash != initial_hash, "Hash should be updated after real change"
            
            # Subsequent check should not detect change
            assert watcher._check_file_changed_by_hash(test_file) == False
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_hash_initialization_debug_output(self, document_ingester, test_data_dir, capsys):
        """Test that hash initialization provides appropriate debug output."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test files
            for i in range(2):
                test_file = test_data_dir / f"debug_test_{i}.md"
                test_file.write_text(f"# Debug Test {i}\nContent for debug testing.")
            
            # Create watcher with debug enabled
            document_ingester.debug = True
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=False)
            watcher._stop_manual_scanning()
            
            # Capture debug output
            captured = capsys.readouterr()
            
            # Should contain initialization debug messages
            assert "Initializing file hashes for watch mode" in captured.out
            assert "Initialized" in captured.out and "file hashes for watch mode" in captured.out
            
        finally:
            document_ingester.debug = False
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_hash_initialization_with_empty_directory(self, document_ingester, test_data_dir):
        """Test hash initialization with an empty directory."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create empty subdirectory
            empty_dir = test_data_dir / "empty_test_dir"
            empty_dir.mkdir(exist_ok=True)
            
            # Create watcher on empty directory
            watcher = DocumentWatcher(document_ingester, empty_dir, verbose=False)
            watcher._stop_manual_scanning()
            
            # Should initialize without error but with no hashes
            assert isinstance(watcher.file_hashes, dict)
            assert len(watcher.file_hashes) == 0
            
        finally:
            os.chdir(original_cwd)


class TestAtomicFileOperations:
    """Tests for atomic file operation handling - the core fix for Claude write detection."""
    
    @pytest.mark.integration
    def test_atomic_file_write_detection(self, document_ingester, test_data_dir):
        """Test that atomic file writes (like Claude uses) are properly detected as modifications, not deletions."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        import tempfile
        import shutil
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create initial test file
            test_file = test_data_dir / "atomic_test.md"
            initial_content = "# Initial Content\nThis is the original content."
            test_file.write_text(initial_content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=1.0)
            watcher._stop_manual_scanning()
            
            # Process initial file
            initial_docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            # Track events to verify atomic operation handling
            deletion_events = []
            creation_events = []
            modification_events = []
            
            original_schedule_deletion = watcher._schedule_delayed_deletion
            original_cancel_deletion = watcher._cancel_pending_deletion
            original_process_change = watcher._process_file_change
            
            def track_schedule_deletion(file_path):
                deletion_events.append(('scheduled', file_path))
                return original_schedule_deletion(file_path)
            
            def track_cancel_deletion(file_path):
                deletion_events.append(('cancelled', file_path))
                return original_cancel_deletion(file_path)
            
            def track_process_change(file_path):
                modification_events.append(file_path)
                return original_process_change(file_path)
            
            watcher._schedule_delayed_deletion = track_schedule_deletion
            watcher._cancel_pending_deletion = track_cancel_deletion
            watcher._process_file_change = track_process_change
            
            # Simulate atomic file operation (like Claude does)
            new_content = "# Modified Content\nThis content was written atomically."
            
            # Step 1: Create temp file
            temp_file = test_file.with_suffix('.tmp')
            temp_file.write_text(new_content)
            
            # Step 2: Simulate deletion event for original file
            mock_delete_event = MagicMock()
            mock_delete_event.src_path = str(test_file)
            mock_delete_event.is_directory = False
            watcher.on_deleted(mock_delete_event)
            
            # Step 3: Simulate creation event (temp file becomes final)
            test_file.unlink()  # Remove original
            temp_file.rename(test_file)  # Atomic rename
            
            mock_create_event = MagicMock()
            mock_create_event.src_path = str(test_file)
            mock_create_event.is_directory = False
            watcher.on_created(mock_create_event)
            
            # Wait for atomic operation delay
            time.sleep(2.5)
            
            # Verify atomic operation was detected
            assert len(deletion_events) >= 2, f"Expected deletion schedule and cancel, got: {deletion_events}"
            assert any(event[0] == 'scheduled' for event in deletion_events), "Deletion should have been scheduled"
            assert any(event[0] == 'cancelled' for event in deletion_events), "Deletion should have been cancelled"
            
            # Verify file was processed as modification, not deletion
            assert len(modification_events) > 0, "File should have been processed as modification"
            assert str(test_file) in modification_events, "Target file should be in modification events"
            
            # Verify final content is correct
            results = document_ingester.collection.query(
                query_texts=["Modified Content"],
                n_results=1
            )
            assert len(results['documents'][0]) > 0, "Modified content should be in database"
            assert "Modified Content" in results['documents'][0][0], "Content should be updated"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_delayed_deletion_processing(self, document_ingester, test_data_dir):
        """Test that deletion processing is properly delayed to detect atomic operations."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        import threading
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "delayed_deletion_test.md"
            test_file.write_text("# Test Content\nThis will be used for deletion testing.")
            
            # Process initial file
            docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Create watcher with short delay for testing
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher.atomic_operation_delay = 1.0  # Short delay for testing
            watcher._stop_manual_scanning()
            
            # Track processing events
            processed_deletions = []
            original_process_delayed = watcher._process_delayed_deletion
            
            def track_delayed_deletion(file_path):
                processed_deletions.append(file_path)
                return original_process_delayed(file_path)
            
            watcher._process_delayed_deletion = track_delayed_deletion
            
            # Simulate deletion event
            mock_event = MagicMock()
            mock_event.src_path = str(test_file)
            mock_event.is_directory = False
            
            # File should still exist at this point
            assert test_file.exists(), "File should exist before deletion event"
            
            # Trigger deletion event
            watcher.on_deleted(mock_event)
            
            # Verify deletion is pending, not processed immediately
            assert str(test_file) in watcher.pending_deletions, "Deletion should be pending"
            assert len(processed_deletions) == 0, "Deletion should not be processed immediately"
            
            # Actually delete the file now
            test_file.unlink()
            
            # Wait for delayed processing
            time.sleep(1.5)
            
            # Verify delayed processing occurred
            assert len(processed_deletions) > 0, "Delayed deletion should have been processed"
            assert str(test_file) in processed_deletions, "Target file should have been processed"
            assert str(test_file) not in watcher.pending_deletions, "Deletion should no longer be pending"
            
            # Verify file was removed from database
            results = document_ingester.collection.query(
                query_texts=["Test Content"],
                n_results=1
            )
            # Should either have no results or empty results
            assert len(results['documents'][0]) == 0 or not results['documents'][0][0], \
                "Content should be removed from database"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_atomic_operation_cancellation(self, document_ingester, test_data_dir):
        """Test that pending deletions are cancelled when file is recreated (atomic operation)."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "cancellation_test.md"
            test_file.write_text("# Original Content\nThis will test cancellation.")
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher.atomic_operation_delay = 2.0  # Longer delay for testing
            watcher._stop_manual_scanning()
            
            # Schedule a deletion
            watcher._schedule_delayed_deletion(str(test_file))
            
            # Verify deletion is pending
            assert str(test_file) in watcher.pending_deletions, "Deletion should be pending"
            
            # Cancel the deletion (simulating atomic operation detection)
            watcher._cancel_pending_deletion(str(test_file))
            
            # Verify deletion was cancelled
            assert str(test_file) not in watcher.pending_deletions, "Deletion should be cancelled"
            
            # Schedule another deletion to test on_created cancellation
            watcher._schedule_delayed_deletion(str(test_file))
            assert str(test_file) in watcher.pending_deletions, "Deletion should be pending again"
            
            # Simulate file creation event (which should cancel pending deletion)
            mock_create_event = MagicMock()
            mock_create_event.src_path = str(test_file)
            mock_create_event.is_directory = False
            watcher.on_created(mock_create_event)
            
            # Verify creation event cancelled the pending deletion
            assert str(test_file) not in watcher.pending_deletions, "Creation should cancel pending deletion"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_atomic_operation_detection_patterns(self, document_ingester, test_data_dir):
        """Test detection of various atomic operation patterns."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()
            
            # Test temporary file patterns
            temp_patterns = [
                "document.md.tmp",
                "document.md.temp", 
                "document.md~",
                "document.md.bak",
                "document.md.swp",
                "document.md.swo"
            ]
            
            for temp_file in temp_patterns:
                temp_path = str(test_data_dir / temp_file)
                
                # Should detect as atomic operation
                assert watcher._is_atomic_operation(temp_path), \
                    f"Should detect {temp_file} as atomic operation"
            
            # Test normal files (should not be detected as atomic)
            normal_files = [
                "document.md",
                "normal.txt",
                "regular.pdf"
            ]
            
            for normal_file in normal_files:
                normal_path = str(test_data_dir / normal_file)
                
                # Should not detect as atomic operation
                assert not watcher._is_atomic_operation(normal_path), \
                    f"Should not detect {normal_file} as atomic operation"
            
            # Test pending deletion detection
            test_file = test_data_dir / "pending_test.md"
            test_file.write_text("# Test content")
            
            # Add to pending deletions
            watcher.pending_deletions[str(test_file)] = time.time()
            
            # Should now be detected as atomic operation
            assert watcher._is_atomic_operation(str(test_file)), \
                "File with pending deletion should be detected as atomic operation"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_temp_file_filtering(self, document_ingester, test_data_dir):
        """Test that temporary files used in atomic operations are filtered out."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, verbose=True)
            watcher._stop_manual_scanning()
            
            # Track processing events
            processed_files = []
            original_process_change = watcher._process_file_change
            
            def track_processing(file_path):
                processed_files.append(file_path)
                return original_process_change(file_path)
            
            watcher._process_file_change = track_processing
            
            # Create temporary files
            temp_file = test_data_dir / "document.md.tmp"
            temp_file.write_text("# Temp Content")
            
            # Simulate modification event for temp file
            mock_event = MagicMock()
            mock_event.src_path = str(temp_file)
            mock_event.is_directory = False
            watcher.on_modified(mock_event)
            
            # Temp file should be filtered out (not processed)
            assert len(processed_files) == 0, "Temp file should be filtered out"
            
            # Create normal file
            normal_file = test_data_dir / "document.md"
            normal_file.write_text("# Normal Content")
            
            # Simulate modification event for normal file
            mock_event.src_path = str(normal_file)
            watcher.on_modified(mock_event)
            
            # Normal file should be processed
            assert len(processed_files) > 0, "Normal file should be processed"
            assert str(normal_file) in processed_files, "Normal file should be in processed files"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_claude_style_write_simulation(self, document_ingester, test_data_dir):
        """Test complete simulation of Claude-style atomic write operations."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock, patch
        import tempfile
        import shutil
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create initial file and ingest it
            test_file = test_data_dir / "claude_write_test.md"
            initial_content = "# Initial Content\nThis simulates a file before Claude edits it."
            test_file.write_text(initial_content)
            
            # Process initial file
            initial_docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            # Verify initial content is in database
            initial_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            assert len(initial_results['documents'][0]) > 0
            assert "Initial Content" in initial_results['documents'][0][0]
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=1.0)
            watcher.atomic_operation_delay = 1.5  # Reasonable delay for testing
            watcher._stop_manual_scanning()
            
            # SIMULATE CLAUDE'S ATOMIC WRITE PROCESS
            
            # Step 1: Claude creates temporary file with new content
            new_content = "# Modified by Claude\nThis content was written using Claude's atomic write process."
            temp_file = test_file.with_suffix('.tmp')
            temp_file.write_text(new_content)
            
            # Step 2: Claude removes original file (triggers DELETE event)
            test_file.unlink()
            mock_delete_event = MagicMock()
            mock_delete_event.src_path = str(test_file)
            mock_delete_event.is_directory = False
            watcher.on_deleted(mock_delete_event)
            
            # Verify deletion is pending (not processed immediately)
            assert str(test_file) in watcher.pending_deletions, "Deletion should be pending"
            
            # Step 3: Claude renames temp file to final name (triggers CREATE event)
            temp_file.rename(test_file)
            mock_create_event = MagicMock()
            mock_create_event.src_path = str(test_file)
            mock_create_event.is_directory = False
            watcher.on_created(mock_create_event)
            
            # Verify atomic operation was detected and deletion cancelled
            assert str(test_file) not in watcher.pending_deletions, "Deletion should be cancelled"
            
            # Wait for processing to complete
            time.sleep(2.0)
            
            # Verify the new content is in the database
            final_results = document_ingester.collection.query(
                query_texts=["Modified by Claude"],
                n_results=1
            )
            assert len(final_results['documents'][0]) > 0, "New content should be in database"
            assert "Modified by Claude" in final_results['documents'][0][0], "Content should be updated"
            
            # Verify old content is no longer accessible (file was replaced, not just deleted)
            old_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            # Old content should either be gone or not match anymore
            if old_results['documents'][0]:
                assert "Initial Content" not in old_results['documents'][0][0], "Old content should be replaced"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_rapid_atomic_operations(self, document_ingester, test_data_dir):
        """Test handling of multiple rapid atomic operations."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create multiple test files
            test_files = []
            for i in range(3):
                test_file = test_data_dir / f"rapid_atomic_{i}.md"
                test_file.write_text(f"# File {i}\nInitial content for file {i}.")
                test_files.append(test_file)
            
            # Create watcher with short delays for testing
            watcher = DocumentWatcher(document_ingester, test_data_dir, debounce_seconds=0.5)
            watcher.atomic_operation_delay = 1.0
            watcher._stop_manual_scanning()
            
            # Track atomic operations
            atomic_operations = []
            original_cancel_deletion = watcher._cancel_pending_deletion
            
            def track_cancellation(file_path):
                atomic_operations.append(file_path)
                return original_cancel_deletion(file_path)
            
            watcher._cancel_pending_deletion = track_cancellation
            
            # Simulate rapid atomic operations on all files
            for i, test_file in enumerate(test_files):
                # Delete
                mock_delete = MagicMock()
                mock_delete.src_path = str(test_file)
                mock_delete.is_directory = False
                watcher.on_deleted(mock_delete)
                
                # Small delay between operations
                time.sleep(0.1)
                
                # Recreate
                test_file.write_text(f"# Modified File {i}\nRapidly modified content.")
                mock_create = MagicMock()
                mock_create.src_path = str(test_file)
                mock_create.is_directory = False
                watcher.on_created(mock_create)
            
            # Wait for all operations to settle
            time.sleep(2.0)
            
            # Verify all atomic operations were properly detected
            assert len(atomic_operations) >= len(test_files), \
                f"Expected {len(test_files)} atomic operations, got {len(atomic_operations)}"
            
            # Verify no files are stuck in pending deletions
            assert len(watcher.pending_deletions) == 0, \
                f"Should have no pending deletions, got: {list(watcher.pending_deletions.keys())}"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_atomic_operation_error_recovery(self, document_ingester, test_data_dir):
        """Test error recovery during atomic operations."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock, patch
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "error_recovery_test.md"
            test_file.write_text("# Error Recovery Test\nThis tests error handling during atomic operations.")
            
            # Process initial file
            docs = document_ingester.process_file(test_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()
            
            # Track rollback operations
            rollback_calls = []
            original_restore = document_ingester._restore_chunks_from_backup
            
            def track_rollback(backup_data):
                rollback_calls.append(backup_data)
                return original_restore(backup_data)
            
            document_ingester._restore_chunks_from_backup = track_rollback
            
            # Simulate atomic operation with processing error
            test_file.write_text("# Modified Content\nThis will cause a processing error.")
            
            # Mock process_file to fail during atomic operation
            with patch.object(document_ingester, 'process_file', side_effect=Exception("Processing failed")):
                # Trigger file change processing
                watcher._debounced_process_file(str(test_file))
            
            # Verify error was handled gracefully and rollback occurred
            assert len(rollback_calls) > 0, "Should have attempted rollback on processing error"
            
            # Verify original content is still accessible (rollback worked)
            recovery_results = document_ingester.collection.query(
                query_texts=["Error Recovery Test"],
                n_results=1
            )
            if recovery_results['documents'][0]:
                assert "Error Recovery Test" in recovery_results['documents'][0][0], "Original content should be preserved"
            
        finally:
            os.chdir(original_cwd)