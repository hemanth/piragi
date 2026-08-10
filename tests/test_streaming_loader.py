import os
import types
import pytest
from unittest.mock import patch, MagicMock

from piragi.loader import DocumentLoader
from piragi.types import Document


def create_temp_files(tmp_path, num_files=3):
    files = []
    for i in range(num_files):
        f = tmp_path / "file_{}.txt".format(i)
        f.write_text("content {}".format(i))
        files.append(str(f))
    return files


def test_stream_yields_documents(tmp_path):
    create_temp_files(tmp_path)
    loader = DocumentLoader()
    stream_gen = loader.stream([str(tmp_path)])
    
    assert isinstance(stream_gen, types.GeneratorType)
    
    first_doc = next(stream_gen)
    assert isinstance(first_doc, Document)
    
    docs = [first_doc] + list(stream_gen)
    assert len(docs) == 3


def test_stream_matches_load(tmp_path):
    create_temp_files(tmp_path)
    loader = DocumentLoader()
    
    docs_load = loader.load(str(tmp_path))
    docs_stream = list(loader.stream(str(tmp_path)))
    
    assert len(docs_load) == len(docs_stream)
    for dl, ds in zip(docs_load, docs_stream):
        assert dl.content == ds.content
        assert dl.source == ds.source


def test_stream_single_file(tmp_path):
    files = create_temp_files(tmp_path, 1)
    loader = DocumentLoader()
    docs = list(loader.stream(files[0]))
    
    assert len(docs) == 1
    assert docs[0].source == files[0]


def test_stream_directory(tmp_path):
    create_temp_files(tmp_path, 5)
    loader = DocumentLoader()
    docs = list(loader.stream(str(tmp_path)))
    
    assert len(docs) == 5


@patch("piragi.loader.logger")
def test_stream_skips_unknown(mock_logger):
    loader = DocumentLoader()
    # A source that is not a directory, file, url, crawl url, remote uri, or glob pattern
    # For example, a custom scheme that is not remote
    unsupported = "ssh://my-server/file.txt"
    docs = list(loader.stream(unsupported))
    
    assert len(docs) == 0
    mock_logger.warning.assert_called_with("Skipping unknown source: %s", unsupported)


@patch.object(DocumentLoader, 'stream')
@patch.object(DocumentLoader, '_is_url')
def test_load_uses_stream(mock_is_url, mock_stream):
    mock_is_url.return_value = True
    doc = Document(content="test", source="test", metadata={})
    mock_stream.return_value = [doc]
    
    loader = DocumentLoader()
    docs = loader.load("test_source")
    
    mock_stream.assert_called_once_with(["test_source"])
    assert docs == [doc]
