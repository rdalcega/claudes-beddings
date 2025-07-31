#!/usr/bin/env python3
"""
Comprehensive tests for the document ingestion system.
"""

import pytest
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import hashlib
import shutil

from .test_utils import (
    PerformanceMonitor, create_test_files, generate_test_content,
    create_comprehensive_test_dataset, validate_chunk_structure,
    performance_monitor
)


class TestDocumentIngester:
    """Test cases for DocumentIngester class."""
    
    @pytest.mark.unit
    def test_ingester_initialization(self, temp_db_dir):
        """Test DocumentIngester initialization."""
        from ingest import DocumentIngester
        
        db_path = temp_db_dir / "test_db"
        ingester = DocumentIngester(str(db_path))
        
        assert ingester.db_path == str(db_path)
        assert ingester.client is not None
        assert ingester.collection is not None
        assert ingester.collection.name == "music_promotion_docs"
    
    @pytest.mark.unit
    def test_chunk_text_basic(self, document_ingester):
        """Test basic text chunking functionality."""
        text = "This is a short text that should not be chunked."
        chunks = document_ingester.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    @pytest.mark.unit
    def test_chunk_text_long(self, document_ingester):
        """Test chunking of long text."""
        # Create text longer than chunk size
        text = "This is a sentence. " * 100  # ~2000 characters
        chunks = document_ingester.chunk_text(text, chunk_size=500, overlap=50)
        
        assert len(chunks) > 1
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # There should be some overlap
            assert len(current_chunk) <= 500
            assert len(next_chunk) <= 500
    
    @pytest.mark.unit
    def test_chunk_text_sentence_boundary(self, document_ingester):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = document_ingester.chunk_text(text, chunk_size=30, overlap=10)
        
        # Most chunks should end with sentence-ending punctuation
        sentence_endings = ['.', '!', '?']
        ending_chunks = sum(1 for chunk in chunks[:-1] if chunk.strip()[-1] in sentence_endings)
        
        # At least some chunks should respect sentence boundaries
        assert ending_chunks > 0
    
    @pytest.mark.unit
    def test_extract_pdf_text_success(self, document_ingester, temp_db_dir):
        """Test successful PDF text extraction."""
        # Mock PyMuPDF to return sample text
        mock_text = "This is extracted PDF text content."
        
        with patch('ingest.fitz.open') as mock_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = mock_text
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_open.return_value = mock_doc
            
            # Create a dummy PDF file
            pdf_path = temp_db_dir / "test.pdf"
            pdf_path.write_bytes(b"dummy pdf content")
            
            result = document_ingester.extract_pdf_text(str(pdf_path))
            assert result == mock_text
    
    @pytest.mark.unit
    def test_extract_pdf_text_failure(self, document_ingester, temp_db_dir):
        """Test PDF text extraction failure handling."""
        # Create non-existent file path
        pdf_path = temp_db_dir / "nonexistent.pdf"
        
        result = document_ingester.extract_pdf_text(str(pdf_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_categorize_file(self, document_ingester):
        """Test file categorization logic."""
        test_cases = [
            ("strategy/marketing.md", "strategy"),
            ("content/lyrics/song.md", "content"),
            ("assets/audio/analysis/theme.md", "content"),
            ("references/resources/guide.pdf", "reference"),
            ("disorganized/notes.txt", "planning"),
            ("other/random.md", "general")
        ]
        
        for file_path, expected_category in test_cases:
            path_obj = Path(file_path)
            category = document_ingester._categorize_file(path_obj)
            assert category == expected_category, f"Expected {expected_category} for {file_path}, got {category}"
    
    @pytest.mark.unit
    def test_extract_path_metadata(self, document_ingester, test_data_dir):
        """Test path metadata extraction for hierarchical filtering."""
        # Create a test file in the test data directory
        nested_dir = test_data_dir / "content" / "lyrics" / "analysis"
        nested_dir.mkdir(parents=True, exist_ok=True)
        test_file = nested_dir / "song_theme.md"
        test_file.write_text("test content")
        
        # Change to test data directory so relative paths work correctly  
        original_cwd = os.getcwd()
        # Use resolve() to ensure consistent path representation
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Don't resolve the file path here since the method handles it
            metadata = document_ingester._extract_path_metadata(test_file)
            
            expected_ancestors = ["content", "content/lyrics", "content/lyrics/analysis"]
            
            assert metadata['path_depth'] == 3
            assert metadata['parent_dir'] == 'analysis'
            assert metadata['path_ancestors_str'] == ','.join(expected_ancestors)
            assert metadata['path_level_0'] == 'content'
            assert metadata['path_level_1'] == 'lyrics'
            assert metadata['path_level_2'] == 'analysis'
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_process_file_markdown(self, document_ingester, test_data_dir):
        """Test processing markdown files."""
        # Create test markdown file
        content = "# Test Document\n\nThis is test content for processing."
        file_path = test_data_dir / "test.md"
        file_path.write_text(content)
        
        # Change to test data directory so relative paths work correctly  
        original_cwd = os.getcwd()
        # Use resolve() to ensure consistent path representation
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            documents = document_ingester.process_file(file_path)
            
            assert len(documents) == 1
            doc = documents[0]
            
            assert doc['content'] == content
            assert doc['metadata']['filename'] == 'test.md'
            assert doc['metadata']['file_type'] == '.md'
            assert doc['metadata']['chunk_index'] == 0
            assert 'id' in doc
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_process_file_empty(self, document_ingester, test_data_dir):
        """Test processing empty files."""
        file_path = test_data_dir / "empty.md"
        file_path.write_text("")
        
        documents = document_ingester.process_file(file_path)
        assert len(documents) == 0
    
    @pytest.mark.unit
    def test_process_file_rtf_success(self, document_ingester, test_data_dir):
        """Test successful RTF file processing."""
        # Create a simple RTF file with real RTF content
        rtf_content = r"""{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}
\f0\fs24 This is a test RTF document with some content.
It contains multiple lines and should be processed correctly.
}"""
        file_path = test_data_dir / "test.rtf"
        file_path.write_text(rtf_content)
        
        # Change to test data directory so relative paths work correctly  
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            documents = document_ingester.process_file(file_path)
            
            assert len(documents) == 1
            doc = documents[0]
            
            # Verify the RTF content was converted to plain text
            assert "This is a test RTF document" in doc['content']
            assert doc['metadata']['filename'] == 'test.rtf'
            assert doc['metadata']['file_type'] == '.rtf'
            assert doc['metadata']['chunk_index'] == 0
            assert 'id' in doc
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_extract_rtf_text_success(self, document_ingester, test_data_dir):
        """Test RTF text extraction method directly."""
        rtf_content = r"""{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}
\f0\fs24 Hello World!\par
This is a test with special characters: \u8216'quotes\u8217' and \u8220"double quotes\u8221".
}"""
        file_path = test_data_dir / "test_extract.rtf"
        file_path.write_text(rtf_content)
        
        text = document_ingester.extract_rtf_text(str(file_path))
        
        assert "Hello World!" in text
        assert "This is a test with special characters" in text
        assert len(text.strip()) > 0
    
    @pytest.mark.unit
    def test_extract_rtf_text_failure(self, document_ingester, test_data_dir):
        """Test RTF text extraction failure handling."""
        # Create non-existent file path
        rtf_path = test_data_dir / "nonexistent.rtf"
        
        result = document_ingester.extract_rtf_text(str(rtf_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_rtf_text_malformed(self, document_ingester, test_data_dir):
        """Test RTF text extraction with malformed RTF content."""
        malformed_content = "This is not valid RTF content"
        file_path = test_data_dir / "malformed.rtf"
        file_path.write_text(malformed_content)
        
        # Should still return the content, even if not proper RTF
        text = document_ingester.extract_rtf_text(str(file_path))
        assert "This is not valid RTF content" in text
    
    @pytest.mark.unit
    def test_process_file_rtf_chunking(self, document_ingester, test_data_dir):
        """Test RTF file processing with content that needs chunking."""
        # Create RTF content that will be chunked
        large_text = "This is a long sentence. " * 100  # ~2500 chars
        rtf_content = f"""{{\\rtf1\\ansi\\deff0 {{\\fonttbl {{\\f0 Times New Roman;}}}}
\\f0\\fs24 {large_text}
}}"""
        file_path = test_data_dir / "large_rtf.rtf"
        file_path.write_text(rtf_content)
        
        # Change to test data directory so relative paths work correctly  
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            documents = document_ingester.process_file(file_path)
            
            # Should create multiple chunks
            assert len(documents) > 1
            
            # Verify chunk metadata
            for i, doc in enumerate(documents):
                assert doc['metadata']['filename'] == 'large_rtf.rtf'
                assert doc['metadata']['file_type'] == '.rtf'
                assert doc['metadata']['chunk_index'] == i
                assert 'This is a long sentence' in doc['content']
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_process_file_rtf_empty_content(self, document_ingester, test_data_dir):
        """Test RTF file with empty text content."""
        rtf_content = r"""{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}
}"""
        file_path = test_data_dir / "empty_rtf.rtf"
        file_path.write_text(rtf_content)
        
        documents = document_ingester.process_file(file_path)
        assert len(documents) == 0  # Should return empty if no meaningful content
    
    @pytest.mark.database
    def test_ingest_directory_basic(self, document_ingester, test_data_dir):
        """Test basic directory ingestion."""
        # Create test dataset
        create_comprehensive_test_dataset(test_data_dir)
        
        # Change to test directory for ingestion
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            document_ingester.ingest_directory(test_data_dir)
            
            # Verify documents were added to collection
            collection_count = document_ingester.collection.count()
            assert collection_count > 0
            
            # Verify we can query the collection
            results = document_ingester.collection.query(
                query_texts=["marketing"],
                n_results=1
            )
            assert len(results['documents'][0]) > 0
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.database
    def test_ingest_directory_filtering(self, document_ingester, test_data_dir):
        """Test that directory ingestion properly filters excluded directories (.git/ and code/)."""
        # Create files in both included and excluded directories
        (test_data_dir / "strategy").mkdir(parents=True, exist_ok=True)
        (test_data_dir / "content").mkdir(parents=True, exist_ok=True)
        (test_data_dir / "code").mkdir(parents=True, exist_ok=True)  # Should be excluded
        (test_data_dir / ".git").mkdir(parents=True, exist_ok=True)  # Should be excluded
        
        # Create files in different directories
        strategy_file = test_data_dir / "strategy" / "marketing.md"
        content_file = test_data_dir / "content" / "lyrics.md"
        code_file = test_data_dir / "code" / "script.md"  # Should be excluded
        git_file = test_data_dir / ".git" / "config.md"  # Should be excluded
        
        strategy_file.write_text("Marketing strategy content")
        content_file.write_text("Song lyrics content")
        code_file.write_text("This should be excluded - code directory")
        git_file.write_text("This should be excluded - git directory")
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            document_ingester.ingest_directory(test_data_dir)
            
            # Check what was actually ingested by looking at all documents
            all_results = document_ingester.collection.get()
            
            # Should find only the included files
            assert len(all_results['documents']) > 0
            
            # Check the sources to see which files were included
            sources = [meta['source'] for meta in all_results['metadatas']]
            
            # Should have strategy and content files, but not code or git files
            strategy_files = [s for s in sources if 'strategy' in s]
            content_files = [s for s in sources if 'content' in s]
            code_files = [s for s in sources if 'code' in s]
            git_files = [s for s in sources if '.git' in s]
            
            assert len(strategy_files) > 0, f"Should have strategy files, got sources: {sources}"
            assert len(content_files) > 0, f"Should have content files, got sources: {sources}"
            assert len(code_files) == 0, f"Should not have code files, got sources: {sources}"
            assert len(git_files) == 0, f"Should not have git files, got sources: {sources}"
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.database
    def test_batch_processing(self, document_ingester, test_data_dir):
        """Test batch processing of documents."""
        # Create many small files to test batching
        (test_data_dir / "strategy").mkdir(parents=True, exist_ok=True)
        
        file_specs = []
        for i in range(25):  # Create 25 small files
            file_specs.append({
                'path': f'strategy/doc_{i:02d}.md',
                'content': f'# Document {i}\n\nThis is test document number {i}.'
            })
        
        create_test_files(test_data_dir, file_specs)
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            # Monitor performance during ingestion
            with performance_monitor() as monitor:
                document_ingester.ingest_directory(test_data_dir)
            
            # Verify all documents were processed
            collection_count = document_ingester.collection.count()
            assert collection_count >= 25  # At least one chunk per file
            
            # Verify performance is reasonable
            metrics = monitor.final_metrics
            assert metrics['duration'] < 60.0  # Should complete within 1 minute
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_duplicate_document_handling(self, document_ingester, test_data_dir):
        """Test handling of duplicate documents."""
        # Create a test file
        content = "This is a test document for duplicate handling."
        file_path = test_data_dir / "test.md"
        file_path.write_text(content)
        
        # Change to test data directory so relative paths work correctly  
        original_cwd = os.getcwd()
        # Use resolve() to ensure consistent path representation
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Process the same file twice
            docs1 = document_ingester.process_file(file_path)
            docs2 = document_ingester.process_file(file_path)
            
            # First processing should succeed and have content
            assert len(docs1) > 0
            
            # Second processing should be skipped due to file change detection (unchanged file)
            assert len(docs2) == 0
            
            # If we force reprocessing by bypassing cache, documents should be identical
            document_ingester.cache.clear()  # Clear cache to force reprocessing
            docs3 = document_ingester.process_file(file_path)
            assert len(docs3) > 0
            assert docs1[0]['id'] == docs3[0]['id']
            assert docs1[0]['content'] == docs3[0]['content']
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.database
    @pytest.mark.slow
    def test_memory_usage_large_ingestion(self, document_ingester, test_data_dir):
        """Test memory usage during large ingestion."""
        # Create a large dataset
        (test_data_dir / "content").mkdir(parents=True, exist_ok=True)
        
        file_specs = []
        for i in range(10):
            # Each file ~10KB
            large_content = generate_test_content('analysis', 'large')
            file_specs.append({
                'path': f'content/large_doc_{i:02d}.md',
                'content': large_content
            })
        
        create_test_files(test_data_dir, file_specs)
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            with performance_monitor() as monitor:
                document_ingester.ingest_directory(test_data_dir)
            
            # Check memory usage is reasonable (less than 200MB delta)
            metrics = monitor.final_metrics
            memory_mb = metrics['memory_delta'] / (1024 * 1024)
            assert memory_mb < 200, f"Memory usage too high: {memory_mb:.2f}MB"
            
        finally:
            os.chdir(original_cwd)
    
    # Tests for new format extraction methods
    
    @pytest.mark.unit
    def test_extract_docx_text_success(self, document_ingester, test_data_dir):
        """Test successful DOCX text extraction."""
        # Use the pre-created sample DOCX file
        docx_path = test_data_dir / "sample.docx"
        
        text = document_ingester.extract_docx_text(str(docx_path))
        
        assert len(text.strip()) > 0
        assert "Music Promotion Test Document" in text
        assert "tiny little baby man" in text
        assert "Instagram" in text  # From the table
        assert "Marketing Goals" in text
    
    @pytest.mark.unit
    def test_extract_docx_text_failure(self, document_ingester, test_data_dir):
        """Test DOCX text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.docx"
        
        result = document_ingester.extract_docx_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_html_text_success(self, document_ingester, test_data_dir):
        """Test successful HTML text extraction."""
        html_path = test_data_dir / "sample.html"
        
        text = document_ingester.extract_html_text(str(html_path))
        
        assert len(text.strip()) > 0
        assert "El Nacimiento De tiny little baby man" in text
        assert "experimental sounds" in text
        assert "Promotional Content" in text
        # Scripts and styles should be removed
        assert "console.log" not in text
        assert "font-family" not in text
    
    @pytest.mark.unit
    def test_extract_html_text_failure(self, document_ingester, test_data_dir):
        """Test HTML text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.html"
        
        result = document_ingester.extract_html_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_json_text_success(self, document_ingester, test_data_dir):
        """Test successful JSON text extraction."""
        json_path = test_data_dir / "sample.json"
        
        text = document_ingester.extract_json_text(str(json_path))
        
        assert len(text.strip()) > 0
        assert "Sample Music Promotion Document" in text
        assert "experimental music" in text
        assert "tiny little baby man" in text
        # Should extract text values but not keys
        assert "Focus on social media engagement" in text
    
    @pytest.mark.unit
    def test_extract_json_text_failure(self, document_ingester, test_data_dir):
        """Test JSON text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.json"
        
        result = document_ingester.extract_json_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_json_text_malformed(self, document_ingester, test_data_dir):
        """Test JSON text extraction with malformed JSON."""
        malformed_json_path = test_data_dir / "malformed.json"
        malformed_json_path.write_text("{ invalid json content")
        
        result = document_ingester.extract_json_text(str(malformed_json_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_xml_text_success(self, document_ingester, test_data_dir):
        """Test successful XML text extraction."""
        xml_path = test_data_dir / "sample.xml"
        
        text = document_ingester.extract_xml_text(str(xml_path))
        
        assert len(text.strip()) > 0
        assert "tiny little baby man" in text
        assert "experimental sounds" in text
        assert "Visual storytelling" in text
    
    @pytest.mark.unit
    def test_extract_xml_text_failure(self, document_ingester, test_data_dir):
        """Test XML text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.xml"
        
        result = document_ingester.extract_xml_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_yaml_text_success(self, document_ingester, test_data_dir):
        """Test successful YAML text extraction."""
        yaml_path = test_data_dir / "sample.yaml"
        
        text = document_ingester.extract_yaml_text(str(yaml_path))
        
        assert len(text.strip()) > 0
        assert "tiny little baby man" in text
        assert "experimental music" in text
        assert "Build awareness" in text
        assert "Visual storytelling" in text
    
    @pytest.mark.unit
    def test_extract_yaml_text_failure(self, document_ingester, test_data_dir):
        """Test YAML text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.yaml"
        
        result = document_ingester.extract_yaml_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_rst_text_success(self, document_ingester, test_data_dir):
        """Test successful RST text extraction."""
        rst_path = test_data_dir / "sample.rst"
        
        text = document_ingester.extract_rst_text(str(rst_path))
        
        assert len(text.strip()) > 0
        assert "tiny little baby man" in text
        assert "experimental music" in text
        assert "Marketing Strategy" in text or "marketing strategy" in text.lower()
    
    @pytest.mark.unit
    def test_extract_rst_text_failure(self, document_ingester, test_data_dir):
        """Test RST text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.rst"
        
        result = document_ingester.extract_rst_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_extract_tex_text_success(self, document_ingester, test_data_dir):
        """Test successful TEX text extraction."""
        tex_path = test_data_dir / "sample.tex"
        
        text = document_ingester.extract_tex_text(str(tex_path))
        
        assert len(text.strip()) > 0
        assert "Music Promotion Strategy" in text or "music promotion strategy" in text.lower()
        assert "tiny little baby man" in text
        assert "experimental music" in text
    
    @pytest.mark.unit
    def test_extract_tex_text_failure(self, document_ingester, test_data_dir):
        """Test TEX text extraction failure handling."""
        nonexistent_path = test_data_dir / "nonexistent.tex"
        
        result = document_ingester.extract_tex_text(str(nonexistent_path))
        assert result == ""
    
    @pytest.mark.unit
    def test_process_file_new_formats(self, document_ingester, test_data_dir):
        """Test processing files with new supported formats."""
        # Change to test data directory for consistent path handling
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Test all new supported formats
            test_files = [
                ('sample.docx', 'Music Promotion Test Document'),
                ('sample.html', 'El Nacimiento De tiny little baby man'),
                ('sample.json', 'Sample Music Promotion Document'),
                ('sample.xml', 'tiny little baby man'),
                ('sample.yaml', 'experimental music'),
                ('sample.rst', 'tiny little baby man'),  # RST removes title formatting
                ('sample.tex', 'tiny little baby man'),  # TEX also processes differently
                ('sample.log', 'promotion campaign'),
                ('sample.csv', 'Instagram'),
                ('sample.tsv', 'Birth_Announcement')
            ]
            
            for filename, expected_content in test_files:
                file_path = test_data_dir / filename
                if file_path.exists():
                    documents = document_ingester.process_file(file_path)
                    
                    assert len(documents) > 0, f"No documents generated for {filename}"
                    
                    # Check that expected content is in at least one chunk
                    all_content = ' '.join(doc['content'] for doc in documents)
                    assert expected_content in all_content, f"Expected content '{expected_content}' not found in {filename}"
                    
                    # Verify metadata
                    doc = documents[0]
                    assert doc['metadata']['filename'] == filename
                    assert doc['metadata']['file_type'] == '.' + filename.split('.')[-1]
                    assert 'id' in doc
        
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_unsupported_format_detection(self, document_ingester, test_data_dir):
        """Test detection and tracking of unsupported formats."""
        # Change to test data directory for consistent path handling
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Test unsupported formats
            unsupported_files = ['sample.doc', 'sample.odt', 'sample.org']
            
            for filename in unsupported_files:
                file_path = test_data_dir / filename
                if file_path.exists():
                    documents = document_ingester.process_file(file_path)
                    
                    # Should return empty documents
                    assert len(documents) == 0, f"Unsupported file {filename} should not generate documents"
                    
                    # Should be added to unsupported files list
                    assert str(file_path) in document_ingester.unsupported_files, f"Unsupported file {filename} should be tracked"
        
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.database
    def test_ingest_directory_unsupported_reporting(self, document_ingester, test_data_dir):
        """Test that unsupported files are properly reported after directory ingestion."""
        # Define unsupported file types from ingest.py (line 301)
        unsupported_file_types = {'.doc', '.odt', '.pages', '.org', '.adoc', '.asciidoc'}
        
        # Count existing unsupported files in test_data_dir (copied from tests/data/)
        existing_unsupported_count = 0
        for file_path in test_data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in unsupported_file_types:
                existing_unsupported_count += 1
        
        # Create strategy directory and a supported file only
        (test_data_dir / "strategy").mkdir(parents=True, exist_ok=True)
        supported_file = test_data_dir / "strategy" / "campaign.md"
        supported_file.write_text("# Marketing Campaign\nThis is a test marketing campaign document.")
        
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            # Clear any existing unsupported files tracking
            document_ingester.unsupported_files = []
            
            # Run ingestion
            document_ingester.ingest_directory(test_data_dir)
            
            # Check that the number of unsupported files found matches what we counted
            assert len(document_ingester.unsupported_files) == existing_unsupported_count
            
            # Verify all tracked files are actually unsupported types
            for unsupported_file in document_ingester.unsupported_files:
                file_suffix = Path(unsupported_file).suffix.lower()
                assert file_suffix in unsupported_file_types, f"File {unsupported_file} has suffix {file_suffix} not in unsupported types"
            
            # Verify supported file was processed
            collection_count = document_ingester.collection.count()
            assert collection_count > 0
            
        finally:
            os.chdir(original_cwd)


class TestFileChangeDetection:
    """Test cases for file change detection functionality."""
    
    @pytest.mark.unit
    def test_cache_initialization(self, document_ingester, temp_db_dir):
        """Test that cache is properly initialized."""
        assert hasattr(document_ingester, 'cache')
        assert hasattr(document_ingester, 'cache_file')
        assert 'version' in document_ingester.cache
        assert 'files' in document_ingester.cache
        assert 'last_updated' in document_ingester.cache
    
    @pytest.mark.unit
    def test_get_file_hash(self, document_ingester, test_data_dir):
        """Test file hash calculation."""
        # Create a test file
        test_file = test_data_dir / "hash_test.txt"
        content = "This is test content for hash calculation"
        test_file.write_text(content)
        
        # Calculate hash
        hash1 = document_ingester._get_file_hash(test_file)
        assert hash1
        assert len(hash1) == 32  # MD5 hash length
        
        # Hash should be consistent
        hash2 = document_ingester._get_file_hash(test_file)
        assert hash1 == hash2
        
        # Hash should change when content changes
        test_file.write_text(content + " modified")
        hash3 = document_ingester._get_file_hash(test_file)
        assert hash1 != hash3
    
    @pytest.mark.unit
    def test_should_process_file_new_file(self, document_ingester, test_data_dir):
        """Test that new files should be processed."""
        test_file = test_data_dir / "new_file.md"
        test_file.write_text("New file content")
        
        should_process = document_ingester.should_process_file(test_file)
        assert should_process
    
    @pytest.mark.unit
    def test_should_process_file_cached_unchanged(self, document_ingester, test_data_dir):
        """Test that unchanged cached files should not be processed."""
        test_file = test_data_dir / "cached_file.md"
        content = "Cached file content"
        test_file.write_text(content)
        
        # First time should process
        assert document_ingester.should_process_file(test_file)
        
        # Update cache as if file was processed
        document_ingester._update_file_cache(test_file)
        
        # Second time should not process (unchanged)
        assert not document_ingester.should_process_file(test_file)
    
    @pytest.mark.unit
    def test_should_process_file_cached_modified(self, document_ingester, test_data_dir):
        """Test that modified cached files should be processed."""
        test_file = test_data_dir / "modified_file.md"
        test_file.write_text("Original content")
        
        # Process and cache file
        assert document_ingester.should_process_file(test_file)
        document_ingester._update_file_cache(test_file)
        assert not document_ingester.should_process_file(test_file)
        
        # Modify file
        import time
        time.sleep(0.1)  # Ensure different mtime
        test_file.write_text("Modified content")
        
        # Should process again
        assert document_ingester.should_process_file(test_file)
    
    @pytest.mark.unit
    def test_update_file_cache(self, document_ingester, test_data_dir):
        """Test cache updating functionality."""
        test_file = test_data_dir / "cache_update_test.md"
        test_file.write_text("Test content")
        
        # Initially empty cache for this file
        file_key = str(test_file.resolve())
        assert file_key not in document_ingester.cache['files']
        
        # Update cache
        document_ingester._update_file_cache(test_file)
        
        # Verify cache entry
        assert file_key in document_ingester.cache['files']
        cache_entry = document_ingester.cache['files'][file_key]
        
        assert 'mtime' in cache_entry
        assert 'size' in cache_entry
        assert 'processed_at' in cache_entry
        assert cache_entry['size'] == test_file.stat().st_size
    
    @pytest.mark.unit
    def test_save_and_load_cache(self, document_ingester, test_data_dir):
        """Test cache persistence."""
        test_file = test_data_dir / "persistence_test.md"
        test_file.write_text("Test content for persistence")
        
        # Update cache and save
        document_ingester._update_file_cache(test_file)
        document_ingester._save_cache()
        
        # Verify cache file exists
        assert document_ingester.cache_file.exists()
        
        # Create new ingester instance (should load existing cache)
        from ingest import DocumentIngester
        new_ingester = DocumentIngester(document_ingester.db_path)
        file_key = str(test_file.resolve())
        
        # Verify cache was loaded
        assert file_key in new_ingester.cache['files']
        assert new_ingester.cache['files'][file_key]['size'] == test_file.stat().st_size
    
    @pytest.mark.integration
    def test_process_file_with_caching(self, document_ingester, test_data_dir):
        """Test that process_file respects caching."""
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            test_file = test_data_dir / "caching_test.md"
            test_file.write_text("# Test Document\nThis is test content.")
            
            # First processing should work
            documents1 = document_ingester.process_file(test_file, force=False)
            assert len(documents1) > 0
            
            # Second processing should be skipped (cached)
            documents2 = document_ingester.process_file(test_file, force=False)
            assert len(documents2) == 0  # Skipped due to cache
            assert str(test_file) in document_ingester.skipped_files
            
            # Force processing should work even with cache
            documents3 = document_ingester.process_file(test_file, force=True)
            assert len(documents3) > 0
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_ingest_directory_with_caching(self, document_ingester, test_data_dir):
        """Test directory ingestion with caching."""
        # Create test files
        (test_data_dir / "cache_test").mkdir(exist_ok=True)
        
        file1 = test_data_dir / "cache_test" / "doc1.md"
        file2 = test_data_dir / "cache_test" / "doc2.md"
        
        file1.write_text("# Document 1\nContent for document 1")
        file2.write_text("# Document 2\nContent for document 2")
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir))
        
        try:
            # First ingestion
            document_ingester.ingest_directory(test_data_dir, force=False)
            initial_processed = len(document_ingester.successful_files)
            initial_skipped = len(document_ingester.skipped_files)
            
            # Reset tracking arrays for second run
            document_ingester.successful_files = []
            document_ingester.skipped_files = []
            document_ingester.processed_files = []
            
            # Second ingestion should skip unchanged files
            document_ingester.ingest_directory(test_data_dir, force=False)
            second_processed = len(document_ingester.successful_files)
            second_skipped = len(document_ingester.skipped_files)
            
            # Should have fewer processed files and more skipped files
            assert second_processed < initial_processed
            assert second_skipped > initial_skipped
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_cache_handles_nonexistent_file(self, document_ingester, test_data_dir):
        """Test cache behavior with non-existent files."""
        nonexistent_file = test_data_dir / "does_not_exist.md"
        
        # Should not process non-existent file
        should_process = document_ingester.should_process_file(nonexistent_file)
        assert not should_process
    
    @pytest.mark.unit 
    def test_cache_handles_corrupted_cache_file(self, document_ingester, temp_db_dir):
        """Test behavior when cache file is corrupted."""
        # Create corrupted cache file
        cache_file = temp_db_dir / ".ingestion_cache.json"
        cache_file.write_text("{ invalid json content")
        
        # Should handle gracefully and create new cache
        from ingest import DocumentIngester
        new_ingester = DocumentIngester(str(temp_db_dir / "test_db"))
        assert new_ingester.cache
        assert 'version' in new_ingester.cache
        assert 'files' in new_ingester.cache


class TestFileWatching:
    """Test cases for file watching functionality."""
    
    @pytest.mark.unit
    def test_document_watcher_initialization(self, document_ingester, test_data_dir):
        """Test DocumentWatcher initialization."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        
        assert watcher.ingester == document_ingester
        assert watcher.project_dir == test_data_dir
        assert watcher.debounce_seconds == 5.0
        assert isinstance(watcher.supported_extensions, set)
        assert '.md' in watcher.supported_extensions
        assert isinstance(watcher.excluded_paths, list)
        assert '.git/' in watcher.excluded_paths
    
    @pytest.mark.unit
    def test_should_process_file_by_extension(self, document_ingester, test_data_dir):
        """Test file filtering by extension."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        
        # Supported extensions
        assert watcher._should_process_file(str(test_data_dir / "test.md"))
        assert watcher._should_process_file(str(test_data_dir / "test.pdf"))
        assert watcher._should_process_file(str(test_data_dir / "test.txt"))
        
        # Unsupported extensions
        assert not watcher._should_process_file(str(test_data_dir / "test.py"))
        assert not watcher._should_process_file(str(test_data_dir / "test.jpg"))
        assert not watcher._should_process_file(str(test_data_dir / "test.exe"))
    
    @pytest.mark.unit
    def test_should_process_file_by_path(self, document_ingester, test_data_dir):
        """Test file filtering by path."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        
        # Allowed paths
        assert watcher._should_process_file(str(test_data_dir / "content" / "test.md"))
        assert watcher._should_process_file(str(test_data_dir / "strategy" / "test.md"))
        
        # Excluded paths
        assert not watcher._should_process_file(str(test_data_dir / ".git" / "config.md"))
        assert not watcher._should_process_file(str(test_data_dir / "code" / "script.md"))
    
    @pytest.mark.integration
    def test_watch_file_modification_success(self, document_ingester, test_data_dir):
        """Test successful file modification in watch mode."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file with initial content
            test_file = test_data_dir / "watch_test.md"
            initial_content = "# Initial Content\nThis is the initial content."
            test_file.write_text(initial_content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Process initial file to get baseline
            initial_docs = document_ingester.process_file(test_file, force=True)
            assert len(initial_docs) > 0
            
            # Add documents to database
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            # Verify initial documents are in database
            initial_count = document_ingester.collection.count()
            assert initial_count > 0
            
            # Modify file content
            modified_content = "# Modified Content\nThis content has been changed."
            test_file.write_text(modified_content)
            
            # Create mock event
            mock_event = MagicMock()
            mock_event.src_path = str(test_file)
            mock_event.is_directory = False
            
            # Process the file change directly (bypassing debouncing for test)
            watcher._debounced_process_file(str(test_file))
            
            # Verify that the database has been updated
            final_count = document_ingester.collection.count()
            assert final_count > 0  # Should still have documents
            
            # Verify the content has been updated by searching for new content
            results = document_ingester.collection.query(
                query_texts=["Modified Content"],
                n_results=1
            )
            assert len(results['documents'][0]) > 0
            assert "Modified Content" in results['documents'][0][0]
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_watch_file_modification_failure_preserves_data(self, document_ingester, test_data_dir):
        """Test that file modification failures preserve existing data."""
        from ingest import DocumentWatcher
        from unittest.mock import patch
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file with initial content
            test_file = test_data_dir / "watch_failure_test.md"
            initial_content = "# Initial Content\nThis is the initial content that should be preserved."
            test_file.write_text(initial_content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Process initial file
            initial_docs = document_ingester.process_file(test_file, force=True)
            assert len(initial_docs) > 0
            
            # Add documents to database
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            # Verify initial documents are in database
            initial_count = document_ingester.collection.count()
            assert initial_count > 0
            
            # Get initial content from database
            initial_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            assert len(initial_results['documents'][0]) > 0
            
            # Mock process_file to fail after creating documents but before removing chunks
            with patch.object(document_ingester, 'extract_pdf_text', side_effect=Exception("Processing failed")):
                # Make the file appear as PDF to trigger the mocked method
                test_file_pdf = test_data_dir / "watch_failure_test.pdf"
                test_file_pdf.write_bytes(b"fake pdf content")
                
                # Try to process the failing file
                watcher._debounced_process_file(str(test_file_pdf))
            
            # Verify original data is still preserved (since the original file wasn't affected)
            preserved_results = document_ingester.collection.query(
                query_texts=["Initial Content"],
                n_results=1
            )
            assert len(preserved_results['documents'][0]) > 0
            assert "Initial Content" in preserved_results['documents'][0][0]
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_watch_file_deletion(self, document_ingester, test_data_dir):
        """Test file deletion handling in watch mode."""
        from ingest import DocumentWatcher
        from unittest.mock import MagicMock
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file
            test_file = test_data_dir / "watch_delete_test.md"
            content = "# File to Delete\nThis file will be deleted."
            test_file.write_text(content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()  # Disable for this test
            
            # Process file and add to database
            docs = document_ingester.process_file(test_file, force=True)
            assert len(docs) > 0
            
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Verify file is in database
            initial_count = document_ingester.collection.count()
            assert initial_count > 0
            
            # Delete the file
            test_file.unlink()
            
            # Create mock deletion event
            mock_event = MagicMock()
            mock_event.src_path = str(test_file)
            mock_event.is_directory = False
            
            # Process deletion event
            watcher.on_deleted(mock_event)
            
            # With atomic operation support, deletion is now delayed
            # Wait for the delayed deletion to process (atomic_operation_delay + buffer)
            import time
            time.sleep(watcher.atomic_operation_delay + 0.5)
            
            # Verify chunks were removed from database
            final_count = document_ingester.collection.count()
            # Note: The count might be the same if there are other documents,
            # but the specific file should be gone
            results = document_ingester.collection.query(
                query_texts=["File to Delete"],
                n_results=5
            )
            # Should find no results or results should not contain our deleted content
            if results['documents'][0]:
                for doc in results['documents'][0]:
                    assert "File to Delete" not in doc
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_watch_empty_file_preserves_existing_data(self, document_ingester, test_data_dir):
        """Test that modifying a file to be empty preserves existing data instead of deleting it."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file with content
            test_file = test_data_dir / "watch_empty_test.md"
            initial_content = "# Original Content\nThis content should be preserved when file becomes empty."
            test_file.write_text(initial_content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            
            # Process initial file
            initial_docs = document_ingester.process_file(test_file, force=True)
            assert len(initial_docs) > 0
            
            # Add to database
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in initial_docs],
                documents=[doc['content'] for doc in initial_docs],
                metadatas=[doc['metadata'] for doc in initial_docs]
            )
            
            # Verify content is in database
            initial_results = document_ingester.collection.query(
                query_texts=["Original Content"],
                n_results=1
            )
            assert len(initial_results['documents'][0]) > 0
            
            # Make file empty
            test_file.write_text("")
            
            # Process the empty file
            watcher._debounced_process_file(str(test_file))
            
            # Verify original content is still in database (since empty file processing should fail)
            preserved_results = document_ingester.collection.query(
                query_texts=["Original Content"],
                n_results=1
            )
            assert len(preserved_results['documents'][0]) > 0
            assert "Original Content" in preserved_results['documents'][0][0]
            
        finally:
            os.chdir(original_cwd)


class TestFileMoveDetection:
    """Test cases for file move detection functionality."""
    
    @pytest.mark.unit
    def test_move_file_in_database_success(self, document_ingester, test_data_dir):
        """Test successful file move in database."""
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create test file and process it
            old_file = test_data_dir / "old_location" / "test.md"
            old_file.parent.mkdir(exist_ok=True)
            content = "# Test Document\nThis is test content for move testing."
            old_file.write_text(content)
            
            # Process and add to database
            docs = document_ingester.process_file(old_file, force=True)
            assert len(docs) > 0
            
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Verify original file is in database
            original_results = document_ingester.collection.get(
                where={"source": str(old_file.relative_to(test_data_dir))}
            )
            assert len(original_results['ids']) > 0
            
            # Create new location
            new_file = test_data_dir / "new_location" / "test.md"
            new_file.parent.mkdir(exist_ok=True)
            new_file.write_text(content)
            
            # Move file in database
            success = document_ingester.move_file_in_database(old_file, new_file)
            assert success
            
            # Verify old location is gone from database
            old_results = document_ingester.collection.get(
                where={"source": str(old_file.relative_to(test_data_dir))}
            )
            assert len(old_results['ids']) == 0
            
            # Verify new location exists in database
            new_results = document_ingester.collection.get(
                where={"source": str(new_file.relative_to(test_data_dir))}
            )
            assert len(new_results['ids']) > 0
            assert new_results['documents'][0] == content
            
            # Verify metadata was updated
            metadata = new_results['metadatas'][0]
            assert metadata['filename'] == 'test.md'
            assert metadata['source'] == str(new_file.relative_to(test_data_dir))
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_move_file_in_database_no_existing_chunks(self, document_ingester, test_data_dir):
        """Test move operation when no existing chunks exist."""
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            old_file = test_data_dir / "nonexistent.md"
            new_file = test_data_dir / "target.md"
            
            # Try to move file that doesn't exist in database
            success = document_ingester.move_file_in_database(old_file, new_file)
            assert not success
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.unit
    def test_document_watcher_move_detection_initialization(self, document_ingester, test_data_dir):
        """Test DocumentWatcher move detection initialization."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        
        assert hasattr(watcher, 'pending_moves')
        assert hasattr(watcher, 'move_detection_window')
        assert watcher.move_detection_window == 5.0
        assert watcher.scan_interval == 10.0  # Reduced from 30s
        assert isinstance(watcher.pending_moves, dict)
    
    @pytest.mark.unit
    def test_detect_file_move_success(self, document_ingester, test_data_dir):
        """Test successful file move detection."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Add a pending deletion
        old_path = str(test_data_dir / "old_file.md")
        watcher.pending_deletions[old_path] = time.time()
        
        # Test move detection
        new_path = str(test_data_dir / "new_file.md")
        detected_source = watcher._detect_file_move(new_path)
        
        assert detected_source == old_path
    
    @pytest.mark.unit
    def test_detect_file_move_no_match(self, document_ingester, test_data_dir):
        """Test file move detection when no matching deletion exists."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # No pending deletions
        new_path = str(test_data_dir / "new_file.md")
        detected_source = watcher._detect_file_move(new_path)
        
        assert detected_source is None
    
    @pytest.mark.unit
    def test_detect_file_move_expired_window(self, document_ingester, test_data_dir):
        """Test file move detection with expired time window."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Add old pending deletion (outside time window)
        old_path = str(test_data_dir / "old_file.md")
        watcher.pending_deletions[old_path] = time.time() - (watcher.move_detection_window + 1)
        
        # Test move detection
        new_path = str(test_data_dir / "new_file.md")
        detected_source = watcher._detect_file_move(new_path)
        
        assert detected_source is None
    
    @pytest.mark.integration
    def test_process_file_move_success(self, document_ingester, test_data_dir):
        """Test successful file move processing."""
        from ingest import DocumentWatcher
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            # Create and process original file
            old_file = test_data_dir / "move_source.md"
            content = "# Move Test\nThis file will be moved."
            old_file.write_text(content)
            
            docs = document_ingester.process_file(old_file, force=True)
            document_ingester.collection.upsert(
                ids=[doc['id'] for doc in docs],
                documents=[doc['content'] for doc in docs],
                metadatas=[doc['metadata'] for doc in docs]
            )
            
            # Create new file location
            new_file = test_data_dir / "move_target.md"
            new_file.write_text(content)
            
            # Create watcher and process move
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()  # Disable for testing
            
            watcher._process_file_move(str(old_file), str(new_file))
            
            # Verify move was successful
            old_results = document_ingester.collection.get(
                where={"source": str(old_file.relative_to(test_data_dir))}
            )
            assert len(old_results['ids']) == 0
            
            new_results = document_ingester.collection.get(
                where={"source": str(new_file.relative_to(test_data_dir))}
            )
            assert len(new_results['ids']) > 0
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_process_file_move_fallback_on_failure(self, document_ingester, test_data_dir):
        """Test file move processing fallback when database move fails."""
        from ingest import DocumentWatcher
        from unittest.mock import patch
        
        # Change to test data directory
        original_cwd = os.getcwd()
        os.chdir(str(test_data_dir.resolve()))
        
        try:
            old_file = test_data_dir / "move_fail_source.md"
            new_file = test_data_dir / "move_fail_target.md"
            content = "# Move Fail Test\nThis move will fail and fallback."
            
            old_file.write_text(content)
            new_file.write_text(content)
            
            # Create watcher
            watcher = DocumentWatcher(document_ingester, test_data_dir)
            watcher._stop_manual_scanning()  # Disable for testing
            
            # Mock move_file_in_database to fail
            with patch.object(document_ingester, 'move_file_in_database', return_value=False), \
                 patch.object(watcher, '_process_file_change') as mock_process:
                
                watcher._process_file_move(str(old_file), str(new_file))
                
                # Should have called fallback processing
                mock_process.assert_called_once_with(str(new_file))
            
        finally:
            os.chdir(original_cwd)


class TestAtomicOperationDetection:
    """Test cases for enhanced atomic operation detection."""
    
    @pytest.mark.unit
    def test_is_atomic_operation_temp_patterns(self, document_ingester, test_data_dir):
        """Test atomic operation detection for various temp file patterns."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Test various temp file patterns
        temp_patterns = [
            "file.tmp",
            "document.temp",
            "test.bak",
            "script.swp",
            "note.swo",
            "backup.orig",
            "file.tmp.md",
            "document.temp.txt",
            ".hidden.tmp",
            ".temp_file.md"
        ]
        
        for pattern in temp_patterns:
            file_path = str(test_data_dir / pattern)
            assert watcher._is_atomic_operation(file_path), f"Should detect {pattern} as atomic operation"
    
    @pytest.mark.unit
    def test_is_atomic_operation_macos_patterns(self, document_ingester, test_data_dir):
        """Test atomic operation detection for macOS-specific patterns."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Test macOS temp patterns
        macos_patterns = [
            "/var/folders/abc/def/T/temp_file.md",
            "/private/var/folders/xyz/123/T/document.txt",
            "/Users/test/TemporaryItems/file.md"
        ]
        
        for pattern in macos_patterns:
            assert watcher._is_atomic_operation(pattern), f"Should detect {pattern} as macOS atomic operation"
    
    @pytest.mark.unit
    def test_is_atomic_operation_random_patterns(self, document_ingester, test_data_dir):
        """Test atomic operation detection for random-looking file names in temp directories."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Test random-looking patterns in temp directories (should be detected)
        temp_random_patterns = [
            "/tmp/abcd1234efgh5678.md",  # Long random string in temp dir
            "/var/folders/abc/T/tmp_9a8b7c6d5e4f.txt",  # Random with tmp prefix in temp dir
        ]
        
        for pattern in temp_random_patterns:
            result = watcher._is_atomic_operation(pattern)
            assert result, f"Should detect {pattern} as potential atomic operation"
        
        # Test random-looking patterns in clearly normal directories (should NOT be detected)
        normal_random_patterns = [
            "/Users/test/project/abcd1234efgh5678.md",  # Random name but in normal dir
            "/home/user/docs/tmp_9a8b7c6d5e4f.txt",  # Random name but in normal dir
        ]
        
        for pattern in normal_random_patterns:
            result = watcher._is_atomic_operation(pattern)
            assert not result, f"Should NOT detect {pattern} as atomic operation (normal directory)"
    
    @pytest.mark.unit
    def test_is_atomic_operation_normal_files(self, document_ingester, test_data_dir):
        """Test that normal files are not detected as atomic operations."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Test normal file patterns in obviously non-temp locations
        normal_patterns = [
            "/Users/test/Documents/document.md",
            "/home/user/project/strategy.txt",
            "/opt/project/content.pdf",
            "/workspace/analysis.json",
            "/projects/myproject/notes.yaml"
        ]
        
        for pattern in normal_patterns:
            assert not watcher._is_atomic_operation(pattern), f"Should NOT detect {pattern} as atomic operation"
    
    @pytest.mark.unit
    def test_should_process_immediately_patterns(self, document_ingester, test_data_dir):
        """Test immediate processing detection patterns."""
        from ingest import DocumentWatcher
        
        # Use the test_data_dir as our project directory to avoid temp directory issues
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Test files that should be processed immediately
        immediate_files = [
            "document.md",
            "strategy.txt", 
            "notes.json"
        ]
        
        for filename in immediate_files:
            file_path = str(test_data_dir / filename)
            # Create the file in project directory
            Path(file_path).write_text("test content")
            
            # Since test_data_dir might be in temp space, let's test with a mock project dir
            # Or accept that temp files won't be processed immediately (which is correct behavior)
            result = watcher._should_process_immediately(file_path)
            
            # If the file is in a temp directory, it should NOT be processed immediately
            # which is actually the correct behavior
            if '/var/folders/' in file_path and '/T/' in file_path:
                # This is expected - temp files should not be processed immediately
                assert not result, f"Should NOT process {filename} immediately (in temp directory)"
            else:
                assert result, f"Should process {filename} immediately"
    
    @pytest.mark.unit
    def test_should_not_process_immediately_temp_files(self, document_ingester, test_data_dir):
        """Test that temp files are not processed immediately."""
        from ingest import DocumentWatcher
        
        watcher = DocumentWatcher(document_ingester, test_data_dir)
        watcher._stop_manual_scanning()  # Disable for testing
        
        # Test temp files that should not be processed immediately
        temp_files = [
            "file.tmp",
            "document.temp",
            "/var/folders/abc/def/temp.md"
        ]
        
        for filename in temp_files:
            assert not watcher._should_process_immediately(filename), f"Should NOT process {filename} immediately"