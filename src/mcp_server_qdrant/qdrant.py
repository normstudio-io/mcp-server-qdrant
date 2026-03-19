import hashlib
import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
        check_compatibility: bool = True,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url,
            api_key=qdrant_api_key,
            path=qdrant_local_path,
            check_compatibility=check_compatibility,
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def store_many(
        self,
        entries: list[Entry],
        *,
        collection_name: str | None = None,
        deterministic_ids: bool = False,
    ) -> None:
        """
        Store multiple entries in one batch (single embed call + single upsert).
        :param entries: List of entries to store.
        :param collection_name: Collection to use; defaults to default collection.
        :param deterministic_ids: If True, use hash(source_url + chunk_index) as point id for dedup.
        """
        if not entries:
            return
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        texts = [e.content for e in entries]
        embeddings = await self._embedding_provider.embed_documents(texts)
        vector_name = self._embedding_provider.get_vector_name()

        points = []
        for i, (entry, vector) in enumerate(zip(entries, embeddings)):
            if deterministic_ids and entry.metadata:
                src = entry.metadata.get("source_url", "")
                idx = entry.metadata.get("chunk_index", i)
                # Qdrant requires string IDs to be valid UUIDs.
                # Generate a deterministic UUID from the source and index.
                hash_hex = hashlib.sha256(f"{src}|{idx}".encode()).hexdigest()
                point_id = str(uuid.UUID(hex=hash_hex[:32]))
            else:
                point_id = str(uuid.uuid4())
            payload = {"document": entry.content, METADATA_PATH: entry.metadata or {}}
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={vector_name: vector},
                    payload=payload,
                )
            )
        await self._client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info("Upserted %d points to collection %s", len(points), collection_name)

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes if configured
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
        else:
            # Collection already exists: ensure payload indexes exist (e.g. after adding new fields)
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    try:
                        await self._client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                        )
                    except Exception as e:
                        # Index may already exist; log and continue
                        logger.debug("Payload index %s: %s", field_name, e)
