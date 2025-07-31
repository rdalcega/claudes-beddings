# Embeddings System Test Suite

This is a comprehensive test suite for the embeddings system that covers all major functionality including document ingestion, semantic search, chunk retrieval, and integration workflows.

## Test Structure

### Core Test Files

- **`test_ingest.py`** - Tests for document ingestion system
  - File type handling (.md, .pdf, .txt, .rtf, .docx, .html, .json, .xml, .yaml, .rst, .tex, .csv, .tsv)
  - Text chunking algorithms
  - Metadata extraction and categorization
  - File change detection and caching
  - Batch processing and error handling

- **`test_search.py`** - Tests for semantic search functionality
  - Basic semantic similarity matching
  - Category and path-based filtering
  - Query edge cases and performance
  - Result ranking and validation

- **`test_retrieve.py`** - Tests for chunk retrieval system
  - Direct chunk ID retrieval
  - Source file + chunk number lookup
  - Multiple chunk retrieval
  - Error handling for missing chunks

- **`test_integration.py`** - End-to-end workflow tests
  - Complete ingest → search → retrieve workflows
  - Path and category filtering workflows
  - Database rebuild and persistence
  - Error recovery scenarios

- **`test_performance.py`** - Performance and scalability tests
  - Large dataset ingestion performance
  - Search response times under load
  - Concurrent operation handling
  - Memory usage and database efficiency

- **`test_watch.py`** - Tests for file watching functionality
  - Watch mode integration and edge cases
  - File change detection and debouncing
  - Hash-based change tracking
  - Error handling and recovery
  - Database consistency and transactional processing
  - Hash initialization and false change prevention

### Supporting Files

- **`conftest.py`** - Shared pytest fixtures and test setup
- **`test_utils.py`** - Utility functions for test data generation and validation
- **`pytest.ini`** - Pytest configuration and test markers

## Running Tests

### Prerequisites

Install testing dependencies:
```bash
pip install -r requirements.txt
```

### Basic Test Execution

Run all tests:
```bash
cd embeddings
pytest
```

Run specific test categories:
```bash
# Unit tests only
pytest -m unit

# Integration tests only  
pytest -m integration

# Performance tests only
pytest -m performance

# All tests except slow ones
pytest -m "not slow"
```

Run specific test files:
```bash
pytest tests/test_ingest.py
pytest tests/test_search.py
pytest tests/test_retrieve.py
pytest tests/test_integration.py
pytest tests/test_performance.py
pytest tests/test_watch.py
```

### Test Options

Run with coverage report:
```bash
pytest --cov=. --cov-report=html
```

Run with verbose output:
```bash
pytest -v
```

Run parallel tests (if pytest-xdist is installed):
```bash
pytest -n auto
```

## Test Markers

The test suite uses the following markers to categorize tests:

- **`unit`** - Fast unit tests for individual components
- **`integration`** - Integration tests for full workflows  
- **`performance`** - Performance and scalability tests
- **`slow`** - Tests that take a long time to run
- **`database`** - Tests that require database setup

## Test Data

Tests use a combination of:

- **Synthetic test data** - Generated markdown and text files with known content
- **Mock PDF files** - Using patched PDF extraction for consistent testing
- **Fixture-based data** - Reusable test document collections
- **Temporary databases** - Isolated test environments that are cleaned up automatically

## Writing New Tests

### Adding Tests to Existing Files

When adding new test methods to existing test classes:

1. Follow the naming convention: `test_descriptive_name`
2. Use appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
3. Use existing fixtures where possible
4. Add docstrings explaining what the test verifies

### Creating New Test Files

For new functionality, create new test files following this pattern:

```python
#!/usr/bin/env python3
"""
Tests for new functionality.
"""

import pytest
from test_utils import validate_results, performance_monitor

class TestNewFunctionality:
    """Test cases for new functionality."""
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        # Test implementation
        pass
    
    @pytest.mark.integration
    def test_integration_workflow(self):
        """Test integration with existing system."""
        # Test implementation
        pass
```

### Test Fixtures

Use the provided fixtures in `conftest.py`:

- `temp_db_dir` - Temporary database directory (cleaned up automatically)
- `test_data_dir` - Temporary directory for test files
- `document_ingester` - DocumentIngester instance with temp database
- `populated_ingester` - Ingester with sample documents loaded
- `document_searcher` - DocumentSearcher instance with populated database
- `chunk_retriever` - ChunkRetriever instance with populated database

### Performance Testing

For performance tests:

1. Use the `performance_monitor` context manager
2. Set reasonable performance expectations
3. Mark tests with `@pytest.mark.performance`
4. Add `@pytest.mark.slow` for tests that take >5 seconds

Example:
```python
@pytest.mark.performance
def test_search_performance(self, document_searcher):
    with performance_monitor() as monitor:
        results = document_searcher.search("query", limit=10)
    
    metrics = monitor.final_metrics
    assert metrics['duration'] < 1.0, "Search too slow"
```

## Extending the Test Suite

### Adding New File Types

When adding support for new file types:

1. Add test cases in `test_ingest.py` for the new file type
2. Create mock extraction functions in `conftest.py`
3. Add sample files to test fixtures
4. Update integration tests to include the new file type

### Adding New Search Features

When adding new search functionality:

1. Add unit tests in `test_search.py` for the new feature
2. Add integration tests showing the feature works end-to-end
3. Add performance tests if the feature might impact performance
4. Update test utilities to support validation of new result formats

### Adding New Metadata Fields

When adding new metadata fields:

1. Update `validate_chunk_structure` in `test_utils.py`
2. Add tests verifying the new metadata is extracted correctly
3. Add tests for filtering/searching using the new metadata
4. Update integration tests to verify metadata persistence

## Troubleshooting Tests

### Common Issues

**Tests fail with "No module named" errors:**
- Ensure you're running tests from the embeddings/ directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

**Database-related test failures:**
- Tests use temporary databases that should be cleaned up automatically
- If tests leave behind test databases, they may be in temp directories
- Check available disk space if tests fail during database creation

**Performance test failures:**
- Performance limits are set conservatively but may need adjustment on slower systems
- Use `pytest -m "not performance"` to skip performance tests
- Check system load when running performance tests

**Memory-related failures:**
- Performance tests monitor memory usage - ensure sufficient RAM is available
- Close other applications when running memory-intensive tests
- Some tests are marked as `slow` and can be skipped with `pytest -m "not slow"`

### Test Isolation

Each test uses isolated temporary databases and files, so tests should not interfere with each other. If you see test interdependency issues:

1. Check that tests are not modifying shared fixtures
2. Ensure test data is generated fresh for each test
3. Verify that database connections are properly closed

## Contributing

When contributing new tests:

1. Follow the existing code style and patterns
2. Add appropriate test markers
3. Include docstrings explaining what is being tested
4. Use existing fixtures and utilities where possible
5. Ensure tests are isolated and don't depend on external state
6. Add performance tests for any new functionality that might impact performance

## Recent Test Suite Enhancements

### Hash Initialization Tests (v2024.07)
A comprehensive new test class `TestHashInitialization` has been added to `test_watch.py` with 9 test cases:

- **`test_initialize_file_hashes_basic_functionality`** - Verifies hash initialization during DocumentWatcher creation
- **`test_initialize_file_hashes_prevents_false_changes`** - Tests prevention of false change detection after initialization  
- **`test_initialize_file_hashes_handles_file_errors`** - Tests graceful error handling during hash initialization
- **`test_initialize_file_hashes_filters_excluded_paths`** - Verifies exclusion of files in restricted directories
- **`test_initialize_file_hashes_only_supported_extensions`** - Tests filtering of unsupported file types
- **`test_hash_initialization_prevents_watch_mode_bug`** - **Key test** verifying the watch mode startup bug is fixed
- **`test_hash_initialization_detects_real_changes`** - Ensures real file changes are still detected correctly
- **`test_hash_initialization_debug_output`** - Tests debug message functionality
- **`test_hash_initialization_with_empty_directory`** - Tests behavior with empty directories

### Expanded File Format Coverage
Tests now cover all supported file formats:
- **Traditional formats**: .md, .pdf, .txt, .rtf
- **Office formats**: .docx
- **Web formats**: .html, .htm  
- **Data formats**: .json, .xml, .yaml, .yml, .csv, .tsv
- **Documentation formats**: .rst, .tex
- **Log formats**: .log

### Enhanced Test Utilities
- **Performance monitoring**: All test classes use consistent performance tracking
- **Error simulation**: Mock-based error injection for testing edge cases
- **Database consistency**: Comprehensive validation of database state
- **Transaction testing**: Verification of rollback and recovery functionality

## Test Statistics

### Current Test Coverage
- **Total test files**: 6
- **Total test classes**: 25+
- **Total test cases**: 140+
- **Test categories**: Unit (60+), Integration (50+), Performance (20+)

### Test Execution Time
- **Unit tests**: ~1-2 minutes
- **All tests (excluding performance)**: ~5-10 minutes
- **Full test suite**: ~15-30 minutes (includes slow performance tests)

### Test Reliability
- **Isolation**: All tests use temporary databases and files
- **Deterministic**: Tests produce consistent results across runs
- **Platform compatibility**: Tests work on macOS, Linux, and Windows
- **Error resilience**: Comprehensive error handling and recovery testing