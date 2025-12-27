"""Tests for memory system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestFinancialSituationMemory:
    """Tests for FinancialSituationMemory class."""

    @pytest.fixture
    def mock_config(self) -> dict:
        """Return a mock config for testing."""
        return {
            "backend_url": "https://api.openai.com/v1",
        }

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Return a mock OpenAI client."""
        mock = MagicMock()
        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536  # OpenAI embedding dimension
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock.embeddings.create.return_value = mock_response
        return mock

    @pytest.fixture
    def mock_chroma_client(self) -> MagicMock:
        """Return a mock ChromaDB client."""
        mock = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock.get_or_create_collection.return_value = mock_collection
        return mock

    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_init_with_openai_backend(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_config: dict,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Verify initialization with OpenAI backend."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        mock_chroma_class.return_value = mock_chroma_client

        memory = FinancialSituationMemory("test_memory", mock_config)

        assert memory.embedding == "text-embedding-3-small"
        mock_openai_class.assert_called_once_with(base_url=mock_config["backend_url"])
        mock_chroma_client.get_or_create_collection.assert_called_once_with(name="test_memory")

    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_init_with_ollama_backend(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Verify initialization with Ollama backend uses different embedding."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        mock_chroma_class.return_value = mock_chroma_client

        config = {"backend_url": "http://localhost:11434/v1"}
        memory = FinancialSituationMemory("test_memory", config)

        assert memory.embedding == "nomic-embed-text"

    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_get_embedding(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_config: dict,
        mock_openai_client: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Verify get_embedding calls OpenAI API correctly."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value = mock_chroma_client

        memory = FinancialSituationMemory("test_memory", mock_config)
        embedding = memory.get_embedding("test text")

        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )
        assert len(embedding) == 1536

    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_add_situations(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_config: dict,
        mock_openai_client: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Verify add_situations stores data in ChromaDB."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value = mock_chroma_client
        mock_collection = mock_chroma_client.get_or_create_collection.return_value

        memory = FinancialSituationMemory("test_memory", mock_config)

        situations = [
            ("Situation 1", "Advice 1"),
            ("Situation 2", "Advice 2"),
        ]
        memory.add_situations(situations)

        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]

        assert len(call_kwargs["documents"]) == 2
        assert call_kwargs["documents"] == ["Situation 1", "Situation 2"]
        assert call_kwargs["ids"] == ["0", "1"]
        assert len(call_kwargs["metadatas"]) == 2
        assert call_kwargs["metadatas"][0]["recommendation"] == "Advice 1"

    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_get_memories(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_config: dict,
        mock_openai_client: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Verify get_memories retrieves similar situations."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        mock_openai_class.return_value = mock_openai_client
        mock_chroma_class.return_value = mock_chroma_client
        mock_collection = mock_chroma_client.get_or_create_collection.return_value

        # Mock query results
        mock_collection.query.return_value = {
            "documents": [["Matched situation"]],
            "metadatas": [[{"recommendation": "Matched advice"}]],
            "distances": [[0.1]],
        }

        memory = FinancialSituationMemory("test_memory", mock_config)
        results = memory.get_memories("current situation", n_matches=1)

        assert len(results) == 1
        assert results[0]["matched_situation"] == "Matched situation"
        assert results[0]["recommendation"] == "Matched advice"
        assert results[0]["similarity_score"] == pytest.approx(0.9)  # 1 - 0.1

    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.PersistentClient")
    def test_init_with_persistent_storage(
        self,
        mock_persistent_client: MagicMock,
        mock_openai_class: MagicMock,
        mock_chroma_client: MagicMock,
    ) -> None:
        """Verify initialization with persistent storage creates PersistentClient."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        mock_persistent_client.return_value = mock_chroma_client

        config = {
            "backend_url": "https://api.openai.com/v1",
            "memory_persistence": True,
            "memory_dir": "/tmp/test_memory",
        }
        memory = FinancialSituationMemory("test_memory", config)

        assert memory.name == "test_memory"
        mock_persistent_client.assert_called_once()
        # Verify the path was passed
        call_kwargs = mock_persistent_client.call_args[1]
        assert call_kwargs["path"] == "/tmp/test_memory"
        mock_chroma_client.get_or_create_collection.assert_called_once_with(name="test_memory")
