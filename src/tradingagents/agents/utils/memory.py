from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI, OpenAI

from tradingagents.async_utils import to_async

logger = logging.getLogger(__name__)


class FinancialSituationMemory:
    def __init__(self, name: str, config: dict):
        if config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
        else:
            self.embedding = "text-embedding-3-small"
        self.client = OpenAI(base_url=config["backend_url"])
        self.async_client = AsyncOpenAI(base_url=config["backend_url"])
        self.name = name

        # Use persistent storage if configured
        if config.get("memory_persistence", False):
            memory_dir = Path(config.get("memory_dir", "./memory"))
            memory_dir.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=str(memory_dir),
                settings=Settings(allow_reset=True, anonymized_telemetry=False),
            )
            logger.info(f"Using persistent memory storage at {memory_dir}")
        else:
            self.chroma_client = chromadb.Client(Settings(allow_reset=True))
            logger.info("Using in-memory storage (memories will not persist)")

        # Get or create collection (allows persistence across runs)
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)
        existing_count = self.situation_collection.count()
        if existing_count > 0:
            logger.info(f"Loaded {existing_count} existing memories for {name}")

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""

        response = self.client.embeddings.create(model=self.embedding, input=text)
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results

    # Async methods
    async def get_embedding_async(self, text: str) -> list[float]:
        """Async version: Get OpenAI embedding for a text."""
        response = await self.async_client.embeddings.create(model=self.embedding, input=text)
        return response.data[0].embedding

    async def add_situations_async(self, situations_and_advice: list[tuple[str, str]]) -> None:
        """Async version: Add financial situations and their corresponding advice."""
        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        # Get all embeddings concurrently
        embedding_tasks = [
            self.get_embedding_async(situation) for situation, _ in situations_and_advice
        ]
        all_embeddings = await asyncio.gather(*embedding_tasks)

        for i, ((situation, recommendation), embedding) in enumerate(
            zip(situations_and_advice, all_embeddings, strict=True)
        ):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(embedding)

        # ChromaDB operations are sync, run in thread
        await to_async(
            self.situation_collection.add,
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    async def get_memories_async(self, current_situation: str, n_matches: int = 1) -> list[dict]:
        """Async version: Find matching recommendations using OpenAI embeddings."""
        query_embedding = await self.get_embedding_async(current_situation)

        # ChromaDB query is sync, run in thread
        results = await to_async(
            self.situation_collection.query,
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    example_config = {
        "backend_url": "https://api.openai.com/v1",
        "memory_persistence": False,  # Use in-memory for example
    }
    matcher = FinancialSituationMemory("example_memory", example_config)

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
