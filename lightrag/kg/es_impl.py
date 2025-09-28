import os
import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Union, final, Dict, List, Set
from ..base import (
    BaseKVStorage,
    DocStatusStorage,
    DocProcessingStatus,
    DocStatus,
    BaseVectorStorage,
    BaseGraphStorage,
)

from ..utils import logger, compute_mdhash_id
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
import pipmaster as pm

if not pm.is_installed("opensearch-py"):
    pm.install("opensearch-py[async]")

from opensearchpy import AsyncOpenSearch, NotFoundError
from opensearchpy.helpers import async_bulk, BulkIndexError
from opensearchpy.exceptions import ConnectionError, TransportError, RequestError


class ESClientManager:
    """
    Manages singleton instance of AsyncOpenSearch client with thread-safe operations.
    Handles client initialization, release, index name sanitization, and index creation.
    """

    _client: AsyncOpenSearch | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncOpenSearch:
        """
        Get a singleton instance of AsyncOpenSearch client.
        Creates a new client if it doesn't exist, using environment variables for authentication.

        Returns:
            AsyncOpenSearch: An instance of the OpenSearch async client.
        """
        async with cls._lock:
            if cls._client is None:
                es_user = os.environ.get("ES_USERNAME")
                es_pass = os.environ.get("ES_PASSWORD")
                auth = (es_user, es_pass) if es_user and es_pass else None

                # Configuration pour bypasser SSL
                es_host = os.environ.get("ES_HOST", "http://localhost:9200")
                verify_certs = os.environ.get("ES_VERIFY_CERTS", "true").lower() == "true"

                # Si verify_certs est False et que l'URL est HTTPS, la convertir en HTTP
                if not verify_certs and es_host.startswith("https://"):
                    es_host = es_host.replace("https://", "http://")
                    logger.info(f"SSL bypassed: Converting URL to HTTP: {es_host}")

                # Configuration du client OpenSearch pour version 2.19.3
                client_config = {
                    "hosts": [es_host],
                    "http_auth": auth,
                    "timeout": 60,
                    "use_ssl": es_host.startswith("https://"),
                    "verify_certs": verify_certs if es_host.startswith("https://") else False,
                    "ssl_show_warn": False
                }

                cls._client = AsyncOpenSearch(**client_config)
            return cls._client

    @classmethod
    async def release_client(cls):
        """
        Release the OpenSearch client by closing the connection and resetting the singleton instance.
        Uses a lock to ensure thread-safe operation.
        """
        async with cls._lock:
            if cls._client:
                await cls._client.close()
                cls._client = None

    @classmethod
    def _sanitize_index_name(cls, name: str) -> str:
        """
        Sanitize index name to comply with OpenSearch naming restrictions.
        Replaces invalid characters with underscores and converts to lowercase.

        Args:
            name: Original index name to sanitize.

        Returns:
            Sanitized index name suitable for OpenSearch.
        """
        sanitized = name.lower()
        for char in ["/", "\\", "*", "?", '"', "<", ">", "|", " ", ","]:
            sanitized = sanitized.replace(char, "_")
        return sanitized

    @classmethod
    async def create_index_if_not_exist(
        cls, index_name: str, mapping: Dict[str, Any], max_retries: int = 3, retry_delay: float = 5.0
    ) -> None:
        """
        Asynchronously create an OpenSearch index if it doesn't exist.
        Includes retry logic for connection issues.

        Args:
            index_name: Name of the index to create.
            mapping: Dictionary defining the index mapping (schema).
            max_retries: Maximum number of connection retry attempts.
            retry_delay: Delay between retry attempts in seconds.
        """
        safe_index_name = cls._sanitize_index_name(index_name)
        
        for attempt in range(max_retries + 1):
            try:
                client = await cls.get_client()

                # Check if the index exists asynchronously
                exists = await client.indices.exists(index=safe_index_name)
                if not exists:
                    # Create the index asynchronously if it does not exist
                    await client.indices.create(index=safe_index_name, body=mapping)
                    logger.info(f"Created index: {index_name}")
                return  # Success, exit the retry loop
                
            except (ConnectionError, TransportError) as e:
                if attempt == max_retries:
                    logger.error(f"Failed to connect to OpenSearch after {max_retries + 1} attempts: {str(e)}")
                    raise ConnectionError(f"Cannot establish connection to OpenSearch at {os.environ.get('ES_HOST', 'http://localhost:9200')}. "
                                        f"Please ensure OpenSearch is running and accessible. Original error: {str(e)}")
                else:
                    logger.warning(f"Connection attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                                 f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error during index creation: {str(e)}")
                raise


@final
@dataclass
class ESKVStorage(BaseKVStorage):
    """
    OpenSearch-based implementation of the BaseKVStorage interface.
    Provides key-value storage functionality using OpenSearch indices.
    """

    es_client: AsyncOpenSearch | None = field(default=None)
    index_name: str = field(default=None)

    def _ensure_client(self) -> AsyncOpenSearch:
        """Ensure es_client is not None, raise error if it is."""
        if self.es_client is None:
            raise RuntimeError("ESKVStorage not initialized - es_client is None")
        return self.es_client

    def __post_init__(self):
        """
        Post-initialization setup. Constructs the final namespace with workspace prefix if provided,
        and sets the index name based on the namespace.
        """
        es_workspace = os.environ.get("ES_WORKSPACE")
        if es_workspace and es_workspace.strip():
            effective_workspace = es_workspace.strip()
        else:
            effective_workspace = self.workspace

        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        self.index_name = self.namespace

    async def initialize(self):
        """
        Initialize the KV storage. Retrieves the OpenSearch client and creates the index
        with appropriate mapping if it doesn't exist.
        Implements robust error handling with connection retries.
        """
        try:
            if self.es_client is None:
                self.es_client = await ESClientManager.get_client()

            kv_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(self.index_name, kv_mapping)
            logger.info(f"Successfully initialized ESKVStorage for index: {self.index_name}")
            
        except (ConnectionError, TransportError) as e:
            error_msg = (f"Failed to initialize ESKVStorage: Cannot connect to OpenSearch at "
                        f"{os.environ.get('ES_HOST', 'http://localhost:9200')}. "
                        f"Please verify that OpenSearch is running and accessible. "
                        f"Error: {str(e)}")
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during ESKVStorage initialization: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def finalize(self):
        """
        Clean up resources by releasing the OpenSearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    def _flatten_es_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten an OpenSearch document response into a simplified dictionary.
        Mirrors PostgreSQL structure where content and other fields are at root level,
        with metadata in separate 'meta' field.

        Args:
            doc: OpenSearch document response (including '_id' and '_source').

        Returns:
            Flattened dictionary containing '_id', 'id', 'content', timestamps, and metadata fields at root level.
        """
        source = doc["_source"]
        result = {
            "_id": doc["_id"],  # Add _id explicitly for consistency with JsonKVStorage
            "id": doc["_id"],   # Keep id for backward compatibility
            "workspace": source.get("workspace", ""),  # Include workspace like PostgreSQL
            "content": source.get("content", ""),  # Content at root level like PostgreSQL
            "create_time": source.get("create_time", 0),  # Default value for old data
            "update_time": source.get("update_time", 0),  # Default value for old data
            **source.get("meta", {}),  # Flatten metadata at root level
        }
        return result

    async def get_by_id(self, id: str) -> Union[Dict[str, Any], None]:
        """
        Retrieve a document by its ID from the KV storage.

        Args:
            id: Document ID to retrieve.

        Returns:
            Flattened document data if found; None if the document does not exist.
        """
        try:
            assert self.es_client is not None, "ESKVStorage not initialized"
            doc_response = await self.es_client.get(index=self.index_name, id=id)
            # Convert OpenSearch response to proper dict format
            doc = {
                "_id": doc_response["_id"],
                "_source": doc_response["_source"]
            }
            result = self._flatten_es_doc(doc)
            # Ensure _id field is present for consistency with JsonKVStorage
            result["_id"] = id
            return result
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple documents by their IDs from the KV storage.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of flattened document data for found IDs (excludes non-existent IDs).
        """
        if not ids:
            return []

        body = {"ids": ids}
        response = await self.es_client.mget(index=self.index_name, body=body)
        docs = []
        for hit in response["docs"]:
            if hit.get("found", False) or (hit.get("_source") is not None):
                doc = self._flatten_es_doc(hit)
                # Ensure _id field is present for consistency with JsonKVStorage
                doc["_id"] = hit["_id"]
                docs.append(doc)
        return docs

    async def filter_keys(self, keys: Set[str]) -> Set[str]:
        """
        Filter a set of keys to identify those that do NOT exist in the storage.

        Args:
            keys: Set of keys to check for existence.

        Returns:
            Subset of keys that are not found in the storage.
        """
        if not keys:
            return set()

        body = {"ids": list(keys)}
        res = await self.es_client.mget(index=self.index_name, body=body)
        found_ids = {doc["_id"] for doc in res["docs"] if doc.get("found", False) or (doc.get("_source") is not None)}
        return keys - found_ids

    async def get_all(self) -> dict[str, Any]:
        """
        Retrieve all documents from the KV storage using scroll API for large result sets.

        Returns:
            Dictionary mapping document IDs to their flattened data.
        """
        result = {}
        scroll = "2m"  # Maintain search context for 2 minutes
        response = await self.es_client.search(
            index=self.index_name,
            body={"query": {"match_all": {}}},
            scroll=scroll,
            size=1000,
        )

        scroll_id = response.get("_scroll_id")
        while scroll_id:
            for hit in response["hits"]["hits"]:
                doc_id = hit["_id"]
                doc = self._flatten_es_doc(hit)
                # Ensure _id field is present for consistency with JsonKVStorage
                doc["_id"] = doc_id
                result[doc_id] = doc

            response = await self.es_client.scroll(scroll_id=scroll_id, scroll=scroll)
            scroll_id = response.get("_scroll_id")

        # Clear the scroll context to free resources
        if scroll_id:
            await self.es_client.clear_scroll(scroll_id=scroll_id)

        return result

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Insert or update multiple documents in bulk. Handles both new documents (insert)
        and existing documents (update) with timestamp tracking similar to JsonKVStorage.

        Args:
            data: Dictionary where keys are document IDs and values are metadata to store.
        """
        if not data:
            return

        current_time = int(time.time())
        actions = []

        # Check which documents already exist to implement conditional timestamp logic
        existing_docs = await self.get_by_ids(list(data.keys()))
        existing_ids = {doc["_id"] for doc in existing_docs if doc}

        for k, v in data.items():
            # Ensure 'llm_cache_list' exists for text_chunks namespace
            if self.namespace.endswith("text_chunks"):
                if "llm_cache_list" not in v:
                    v["llm_cache_list"] = []

            # Conditional timestamp logic like JsonKVStorage
            if k in existing_ids:  # Key exists, only update update_time
                v["update_time"] = current_time
            else:  # New key, set both create_time and update_time
                v["create_time"] = current_time
                v["update_time"] = current_time

            # Add _id field like JsonKVStorage
            v["_id"] = k

            # Extract content and metadata like PostgreSQL structure
            content = v.get("content", "")  # content at root level

            # Extract metadata (exclude reserved fields and content)
            meta_data = {
                key: value
                for key, value in v.items()
                if key not in ["_id", "id", "workspace", "content", "create_time", "update_time"]
            }

            # Prepare bulk action: update if exists, insert (upsert) if not
            action = {
                "_op_type": "update",
                "_index": self.index_name,
                "_id": k,
                "doc": {
                    "workspace": self.workspace,  # Add workspace like PostgreSQL
                    "content": content,  # Content at root level like PostgreSQL
                    "update_time": v["update_time"],
                    "meta": meta_data,
                },
                "upsert": {
                    "id": k,
                    "workspace": self.workspace,  # Add workspace like PostgreSQL
                    "content": content,  # Content at root level like PostgreSQL
                    "create_time": v.get("create_time", current_time),
                    "update_time": v["update_time"],
                    "meta": meta_data,
                },
            }

            actions.append(action)

        # Execute bulk operation
        try:
            await async_bulk(self.es_client, actions, refresh="wait_for")
        except Exception as e:
            logger.error(f"Unexpected error during bulk upsert: {e}")

    async def index_done_callback(self):
        """
        Callback invoked after indexing completes. No specific operation implemented.
        """
        pass

    async def delete(self, ids: List[str]) -> None:
        """
        Delete multiple documents by their IDs from the KV storage.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        # Prepare bulk delete actions
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]

        try:
            result = await async_bulk(
                self.es_client, actions, refresh="wait_for"
            )
            # Handle different return formats from OpenSearch async_bulk
            if isinstance(result, tuple):
                successes, errors = result
                success_count = len(successes) if hasattr(successes, '__len__') else successes
                error_count = len(errors) if hasattr(errors, '__len__') else (errors if errors else 0)
            else:
                success_count = result if isinstance(result, int) else len(result)
                error_count = 0
                errors = []

            logger.info(f"Deleted {success_count} documents, {error_count} failed")
            if errors:
                logger.error(f"Delete errors: {errors}")
        except Exception as e:
            logger.error(f"Error deleting documents from {self.index_name}: {e}")

    async def drop_cache_by_modes(self, modes: List[str] = None) -> bool:
        """
        Delete documents associated with specific modes (for LLM response cache).
        Matches documents using regex pattern on document IDs.

        Args:
            modes: List of modes to filter documents for deletion.

        Returns:
            True if deletion is successful; False if modes are not provided.
        """
        if not modes:
            return False

        try:
            # Regex pattern: match IDs starting with any mode in the list
            pattern = f"({'|'.join(modes)}):.*"
            response = await self.es_client.delete_by_query(
                index=self.index_name, body={"query": {"regexp": {"_id": pattern}}}
            )
            logger.info(f"Deleted {response['deleted']} documents by modes: {modes}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache by modes {modes}: {e}")
            return False

    async def drop(self) -> Dict[str, str]:
        """
        Delete all documents in the KV storage index.

        Returns:
            Dictionary with 'status' (success/error) and 'message' describing the result.
        """
        try:
            response = await self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                wait_for_completion=True,
            )
            return {
                "status": "success",
                "message": f"Deleted {response['deleted']} documents",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Define the OpenSearch index mapping for the KV storage.
        Mirrors PostgreSQL table structure with workspace, content at root level.

        Returns:
            Dictionary specifying the index mapping.
        """
        return {
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "id": {"type": "keyword"},
                    "workspace": {"type": "keyword"},  # Add workspace field like PostgreSQL
                    "content": {"type": "text"},  # Add content field at root level like PostgreSQL
                    "create_time": {"type": "long"},
                    "update_time": {"type": "long"},
                    "meta": {"type": "object", "dynamic": True},
                },
            }
        }


@final
@dataclass
class ESDocStatusStorage(DocStatusStorage):
    """
    OpenSearch-based implementation of the DocStatusStorage interface.
    Tracks document processing status (e.g., indexing state, chunk counts) using OpenSearch.
    """

    es_client: AsyncOpenSearch | None = field(default=None)
    index_name: str = field(default=None)

    def __post_init__(self):
        """
        Post-initialization setup. Constructs the final namespace with workspace prefix if provided,
        and sets the index name based on the namespace.
        """
        es_workspace = os.environ.get("ES_WORKSPACE")
        if es_workspace and es_workspace.strip():
            effective_workspace = es_workspace.strip()
        else:
            effective_workspace = self.workspace

        # Apply workspace prefix to namespace
        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        # Set index name
        self.index_name = self.namespace

    async def initialize(self):
        """
        Initialize the document status storage. Retrieves the OpenSearch client and creates
        the index with appropriate mapping if it doesn't exist.
        Implements robust error handling with connection retries.
        """
        try:
            if self.es_client is None:
                self.es_client = await ESClientManager.get_client()

            doc_status_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(
                self.index_name, doc_status_mapping
            )
            logger.info(f"Successfully initialized ESDocStatusStorage for index: {self.index_name}")
            
        except (ConnectionError, TransportError) as e:
            error_msg = (f"Failed to initialize ESDocStatusStorage: Cannot connect to OpenSearch at "
                        f"{os.environ.get('ES_HOST', 'http://localhost:9200')}. "
                        f"Please verify that OpenSearch is running and accessible. "
                        f"Error: {str(e)}")
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during ESDocStatusStorage initialization: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def finalize(self):
        """
        Clean up resources by releasing the OpenSearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    async def get_by_id(self, id: str) -> Union[Dict[str, Any], None]:
        """
        Retrieve a document's status by its ID.

        Args:
            id: Document ID to retrieve status for.

        Returns:
            Status data if found; None if the document status does not exist.
        """
        try:
            res = await self.es_client.get(index=self.index_name, id=id)
            doc = res["_source"]
            return doc
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve status data for multiple documents by their IDs.

        Args:
            ids: List of document IDs.

        Returns:
            List of status data dictionaries for found documents.
        """
        if not ids:
            return []

        body = {"ids": ids}
        res = await self.es_client.mget(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["docs"] if hit.get("found", False) or (hit.get("_source") is not None)]

    async def filter_keys(self, keys: Set[str]) -> Set[str]:
        """
        Filter a set of keys to identify those that do NOT exist in the storage.

        Args:
            keys: Set of keys to check for existence.

        Returns:
            Subset of keys that are not found in the storage.
        """
        if not keys:
            return set()

        response = await self.es_client.mget(
            index=self.index_name, body={"ids": list(keys)}
        )
        existing_ids = set()
        for doc in response["docs"]:
            # OpenSearch/Elasticsearch can have different response format
            if doc.get("found", False) or (doc.get("_source") is not None):
                existing_ids.add(doc["_id"])
        return keys - existing_ids

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Insert or update document status data in bulk. Ensures 'chunks_list' is a list of strings.
        Adds workspace field like PostgreSQL implementation.

        Args:
            data: Dictionary where keys are document IDs and values are status metadata.
        """
        if not data:
            return

        actions = []
        for doc_id, doc_data in data.items():
            # Ensure 'chunks_list' is a list (normalize input)
            if "chunks_list" not in doc_data or doc_data["chunks_list"] is None:
                doc_data["chunks_list"] = []
            elif not isinstance(doc_data["chunks_list"], list):
                if isinstance(doc_data["chunks_list"], (str, int)):
                    doc_data["chunks_list"] = [str(doc_data["chunks_list"])]
                else:
                    doc_data["chunks_list"] = []

            # Add workspace field like PostgreSQL
            doc_data["workspace"] = self.workspace
            doc_data["id"] = doc_id  # Ensure id field is present

            # logger.info(f"Upserting doc {doc_id}: {doc_data}")

            # Prepare bulk action: update if exists, insert if not
            actions.append(
                {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": doc_id,
                    "doc": doc_data,
                    "doc_as_upsert": True,  # Insert as new document if not exists
                }
            )

        # Execute bulk operation
        try:
            await async_bulk(self.es_client, actions, refresh="wait_for")
        except BulkIndexError as e:
            logger.error(
                f"BulkIndexError: {len(e.errors)} document(s) failed to index."
            )
            for err in e.errors:
                logger.error(f"Indexing error detail: {err}")
            raise
        except (ConnectionError, TransportError, RequestError) as e:
            logger.error(f"OpenSearch error: {e}")
            raise
        except Exception:
            logger.exception("Unexpected exception during OpenSearch bulk upsert.")
            raise

    async def get_status_counts(self) -> Dict[str, int]:
        """
        Get the count of documents grouped by their processing status.

        Returns:
            Dictionary with status values as keys and their respective counts as values.
        """
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "size": 0,  # Do not return actual documents
                "aggs": {
                    "status_counts": {
                        "terms": {
                            "field": "status.keyword",  # Use keyword sub-field for exact matches
                            "size": 100,  # Support up to 100 distinct statuses
                        }
                    }
                },
            },
        )

        counts = {}
        for bucket in response["aggregations"]["status_counts"]["buckets"]:
            counts[bucket["key"]] = bucket["doc_count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> Dict[str, DocProcessingStatus]:
        """
        Retrieve documents with a specific processing status.

        Args:
            status: Target document status to filter by (from DocStatus enum).

        Returns:
            Dictionary mapping document IDs to DocProcessingStatus objects.
        """
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {"term": {"status": status.value}},  # Match status enum value
                "size": 1000,  # Adjust based on expected result size
            },
        )

        result = {}
        for hit in response["hits"]["hits"]:
            doc_id = hit["_id"]
            doc_data = hit["_source"]

            result[doc_id] = DocProcessingStatus(
                content_summary=doc_data.get("content_summary", ""),
                content_length=doc_data.get("content_length", 0),
                file_path=doc_data.get("file_path", doc_id),
                status=doc_data.get("status", status.value),
                created_at=doc_data.get("created_at", ""),
                updated_at=doc_data.get("updated_at", ""),
                chunks_count=doc_data.get("chunks_count", -1),
                chunks_list=doc_data.get("chunks_list", []),
            )
        return result

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts
        """
        # This is the same as get_status_counts for single workspace
        return await self.get_status_counts()

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id

        Args:
            track_id: The track ID to filter by

        Returns:
            Dictionary mapping document IDs to DocProcessingStatus objects
        """
        try:
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"track_id": track_id}},
                    "size": 10000,  # Adjust based on expected results
                },
            )

            result = {}
            for hit in response["hits"]["hits"]:
                doc_id = hit["_id"]
                doc_data = hit["_source"]

                result[doc_id] = DocProcessingStatus(
                    content_summary=doc_data.get("content_summary", ""),
                    content_length=doc_data.get("content_length", 0),
                    file_path=doc_data.get("file_path", doc_id),
                    status=doc_data.get("status", ""),
                    created_at=doc_data.get("created_at", ""),
                    updated_at=doc_data.get("updated_at", ""),
                    chunks_count=doc_data.get("chunks_count", -1),
                    chunks_list=doc_data.get("chunks_list", []),
                )
            return result
        except Exception as e:
            logger.error(f"Error getting docs by track_id {track_id}: {e}")
            return {}

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc"
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination

        Args:
            status_filter: Optional status to filter by
            page: Page number (1-based)
            page_size: Number of documents per page
            sort_field: Field to sort by
            sort_direction: Sort direction ("asc" or "desc")

        Returns:
            Tuple of (documents list, total count)
        """
        try:
            # Calculate offset
            offset = (page - 1) * page_size

            # Build query
            if status_filter:
                query = {"term": {"status": status_filter.value}}
            else:
                query = {"match_all": {}}

            # Build search body
            search_body = {
                "query": query,
                "from": offset,
                "size": page_size,
                "sort": [{sort_field: {"order": sort_direction}}]
            }

            # Execute search
            response = await self.es_client.search(
                index=self.index_name,
                body=search_body
            )

            # Get total count
            total_count = response["hits"]["total"]["value"]

            # Process results as list of tuples
            result = []
            for hit in response["hits"]["hits"]:
                doc_id = hit["_id"]
                doc_data = hit["_source"]

                doc_status = DocProcessingStatus(
                    content_summary=doc_data.get("content_summary", ""),
                    content_length=doc_data.get("content_length", 0),
                    file_path=doc_data.get("file_path", doc_id),
                    status=doc_data.get("status", ""),
                    created_at=doc_data.get("created_at", ""),
                    updated_at=doc_data.get("updated_at", ""),
                    chunks_count=doc_data.get("chunks_count", -1),
                    chunks_list=doc_data.get("chunks_list", []),
                )
                result.append((doc_id, doc_status))

            return result, total_count

        except Exception as e:
            logger.error(f"Error getting paginated docs: {e}")
            return [], 0

    async def index_done_callback(self):
        """
        Callback invoked after indexing completes. No specific operation implemented.
        """
        pass

    async def drop(self) -> Dict[str, str]:
        """
        Delete all documents in the status storage index.

        Returns:
            Dictionary with 'status' (success/error) and 'message' describing the result.
        """
        try:
            response = await self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                wait_for_completion=True,
            )
            return {
                "status": "success",
                "message": f"Deleted {response['deleted']} documents",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete(self, ids: list[str]) -> None:
        """
        Delete status records for multiple documents by their IDs.

        Args:
            ids: List of document IDs to delete status records for.
        """
        if not ids:
            return

        # Prepare bulk delete actions
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]

        try:
            await async_bulk(
                self.es_client, actions, refresh="wait_for", raise_on_error=False
            )
            logger.debug(f"Deleted {len(ids)} doc statuses from {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting doc statuses: {e}")

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Define the OpenSearch index mapping for document status storage.
        Mirrors PostgreSQL table structure with workspace field.

        Returns:
            Dictionary specifying the index mapping.
        """
        return {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "workspace": {"type": "keyword"},  # Add workspace field like PostgreSQL
                    "status": {
                        "type": "keyword"  # Exact matches for status filtering
                    },
                    "content": {
                        "type": "text"  # Full-text searchable content
                    },
                    "content_summary": {
                        "type": "text"  # Summary of content
                    },
                    "content_length": {
                        "type": "integer"  # Length of content
                    },
                    "created_at": {
                        "type": "date"  # Timestamp of creation
                    },
                    "updated_at": {
                        "type": "date"  # Timestamp of last update
                    },
                    "chunks_count": {
                        "type": "integer"  # Number of chunks in the document
                    },
                    "chunks_list": {
                        "type": "keyword",  # List of chunk IDs (as keywords)
                    },
                    "file_path": {
                        "type": "keyword"  # Path to source file (exact matches)
                    },
                    "track_id": {
                        "type": "keyword"  # Track ID for grouping documents
                    },
                }
            }
        }


@final
@dataclass
class ESVectorDBStorage(BaseVectorStorage):
    """
    OpenSearch-based implementation of the BaseVectorStorage interface.
    Stores and queries vector embeddings using OpenSearch's dense vector support,
    enabling similarity search for embeddings (e.g., text embeddings).
    """

    es_client: AsyncOpenSearch | None = field(default=None)
    index_name: str = field(default="", init=False)
    embedding_dim: int = field(default=0, init=False)

    def __post_init__(self):
        """
        Post-initialization setup. Configures workspace, index name, embedding dimension,
        and similarity threshold from environment variables and global config.
        """
        # Handle workspace prefix for namespace
        es_workspace = os.environ.get("ES_WORKSPACE")
        if es_workspace and es_workspace.strip():
            effective_workspace = es_workspace.strip()
        else:
            effective_workspace = self.workspace

        # Apply workspace prefix to namespace
        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        # Set index name for vector storage
        self.index_name = f"vector_{self.namespace}"

        # Get embedding dimension from the embedding function
        self.embedding_dim = self.embedding_func.embedding_dim

        # Get cosine similarity threshold from global config
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in global config"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Set batch size for embedding generation
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        """
        Initialize the vector storage. Retrieves the OpenSearch client and creates
        the vector index with dense vector mapping if it doesn't exist.
        Implements robust error handling with connection retries.
        """
        try:
            if self.es_client is None:
                self.es_client = await ESClientManager.get_client()

            # Check if index exists and has correct mapping
            try:
                exists = await self.es_client.indices.exists(index=self.index_name)
                if exists:
                    # Check if vector field has correct knn_vector type
                    mapping = await self.es_client.indices.get_mapping(index=self.index_name)
                    current_mapping = mapping[self.index_name]["mappings"]["properties"]
                    vector_field = current_mapping.get("vector", {})

                    # If vector field is not knn_vector type, recreate index
                    if vector_field.get("type") != "knn_vector":
                        logger.warning(f"Index {self.index_name} has incorrect vector mapping. Recreating...")
                        await self.drop()
                        return  # drop() already recreates the index

            except (ConnectionError, TransportError):
                # Let connection errors bubble up to be handled by the outer try-catch
                raise
            except Exception as e:
                logger.warning(f"Could not check index mapping: {e}")

            vector_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(self.index_name, vector_mapping)
            logger.info(f"Successfully initialized ESVectorDBStorage for index: {self.index_name}")
            
        except (ConnectionError, TransportError) as e:
            error_msg = (f"Failed to initialize ESVectorDBStorage: Cannot connect to OpenSearch at "
                        f"{os.environ.get('ES_HOST', 'http://localhost:9200')}. "
                        f"Please verify that OpenSearch is running and accessible. "
                        f"Error: {str(e)}")
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during ESVectorDBStorage initialization: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def finalize(self):
        """
        Clean up resources by releasing the OpenSearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Insert or update vector documents in bulk. Generates embeddings for content using
        the configured embedding function and stores them with metadata.

        Args:
            data: Dictionary where keys are document IDs and values contain 'content' and metadata.
        """
        logger.info(f"Inserting {len(data)} documents to {self.index_name}")
        if not data:
            return

        current_time = int(time.time())

        # Extract content for embedding generation
        contents = [v["content"] for v in data.values()]
        # Split into batches to avoid overwhelming the embedding function
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        # Generate embeddings for all batches (async)
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        # Concatenate batch embeddings into a single array
        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) != len(data):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) does not match data count ({len(data)})"
            )

        # Prepare bulk index actions
        actions = []
        for i, (doc_id, doc_data) in enumerate(data.items()):
            # Handle source_id splitting like PostgreSQL implementation
            source_id = doc_data.get("source_id", "")
            if isinstance(source_id, str) and GRAPH_FIELD_SEP in source_id:
                chunk_ids = source_id.split(GRAPH_FIELD_SEP)
            else:
                chunk_ids = [source_id] if source_id else []

            # Handle file_path splitting like PostgreSQL implementation
            file_path = doc_data.get("file_path", "")
            if isinstance(file_path, str) and GRAPH_FIELD_SEP in file_path:
                file_paths = file_path.split(GRAPH_FIELD_SEP)
            else:
                file_paths = [file_path] if file_path else []

            # Construct document with vector, timestamps, workspace, and allowed metadata
            doc = {
                "id": doc_id,
                "workspace": self.workspace,  # Add workspace like PostgreSQL
                "vector": embeddings[i].tolist(),  # Convert numpy array to list
                "created_at": current_time,
                "chunk_ids": chunk_ids,  # Store as array like PostgreSQL
                "file_paths": file_paths,  # Store as array like PostgreSQL
                "meta": {k: v for k, v in doc_data.items() if k in self.meta_fields},
            }
            actions.append(
                {
                    "_op_type": "index",  # Overwrite if exists (idempotent)
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": doc,
                }
            )

        # Execute bulk insertion with refresh to make data immediately searchable
        try:
            result = await async_bulk(
                self.es_client, actions, refresh="wait_for"
            )
            # Handle different return formats from OpenSearch async_bulk
            if isinstance(result, tuple):
                success, errors = result
                error_count = len(errors) if hasattr(errors, '__len__') else (errors if errors else 0)
            else:
                errors = []
                error_count = 0

            if errors:
                logger.error(f"Upsert failed for {error_count} documents: {errors}")
        except BulkIndexError as e:
            logger.error(f"Bulk index error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in upsert: {e}")

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """
        Perform a vector similarity search using a query text or provided embedding.
        Finds the top-k most similar vectors in the storage.

        Args:
            query: Input text to generate a query vector from (used if query_embedding is None).
            top_k: Number of top matching results to return.
            query_embedding: Pre-computed query embedding (optional, will compute if None).

        Returns:
            List of matching documents with metadata, IDs, distances, and timestamps,
            filtered by the cosine similarity threshold.
        """
        # Use provided embedding or generate one from query text
        if query_embedding is not None:
            query_vector = query_embedding
        else:
            # Generate embedding for the query text
            embedding = await self.embedding_func([query], _priority=5)
            query_vector = embedding[0].tolist()

        # Configure OpenSearch KNN query - using the correct structure
        es_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            }
        }

        # Execute the search
        logger.debug(f"ESVectorDBStorage query: index={self.index_name}, query_vector_len={len(query_vector)}, top_k={top_k}")
        response = await self.es_client.search(index=self.index_name, body=es_query)
        hits = response["hits"]["hits"]
        logger.debug(f"ESVectorDBStorage query: found {len(hits)} raw hits, threshold={self.cosine_better_than_threshold}")

        # Format results, filtering by similarity threshold
        results = [
            {
                "id": hit["_id"],
                "distance": hit.get("_score"),  # Cosine similarity score
                "created_at": hit["_source"].get("created_at"),
                **hit["_source"]["meta"],  # Include metadata fields
            }
            for hit in hits
            if hit.get("_score") > self.cosine_better_than_threshold  # Apply threshold
        ]
        logger.debug(f"ESVectorDBStorage query: returning {len(results)} results after threshold filtering")
        return results

    async def index_done_callback(self) -> None:
        """
        Callback after indexing completes. No specific operation required for OpenSearch.
        """
        pass

    async def delete(self, ids: list[str]) -> None:
        """
        Delete multiple vector documents by their IDs from the storage.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        max_batch_size = 100
        ids = list(ids)

        ids_list = list(ids) if not isinstance(ids, list) else ids
        batches = [
            ids_list[i : i + max_batch_size] for i in range(0, len(ids_list), max_batch_size)
        ]

        for batch_index, batch_ids in enumerate(batches, start=1):
            # Prepare bulk delete actions
            actions = [
                {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
                for doc_id in batch_ids
            ]

            try:
                result = await async_bulk(
                    self.es_client, actions, refresh="wait_for", raise_on_error=False
                )

                # Handle different return formats from OpenSearch async_bulk
                if isinstance(result, tuple):
                    success, failed = result
                else:
                    failed = []

                if failed:
                    for item in failed:
                        # Ignore 404 errors (document not found)
                        if (
                            "result" in item.get("delete", {})
                            and item["delete"]["result"] == "not_found"
                        ):
                            continue
                            # logger.info(f"Document {item['delete']['_id']} not found, skipping deletion.")
                        else:
                            logger.error(f"Failure details: {item}")
                else:
                    logger.info(
                        f"Successfully deleted {success} documents in batch {batch_index}."
                    )

            except Exception as e:
                logger.error(f"Batch delete failed for batch {batch_index}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """
        Delete a vector document associated with a specific entity name.
        The entity ID is generated using a hash of the entity name.

        Args:
            entity_name: Name of the entity to delete (e.g., a named entity from text).
        """
        try:
            # Generate entity ID using consistent hashing
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            # Delete the document, ignoring 404 (not found) errors
            try:
                response = await self.es_client.delete(
                    index=self.index_name, id=entity_id
                )
            except NotFoundError:
                response = {"result": "not_found"}
            if response["result"] == "deleted":
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Delete all vector documents representing relations (edges) involving a specific entity.
        Matches documents where the entity is either the source or target in the metadata.

        Args:
            entity_name: Name of the entity whose relations to delete.
        """
        try:
            # Query to match relations where entity is source or target
            query = {
                "query": {
                    "bool": {
                        "should": [  # Logical OR
                            {
                                "term": {"meta.src_id.keyword": entity_name}
                            },  # Entity is source
                            {
                                "term": {"meta.tgt_id.keyword": entity_name}
                            },  # Entity is target
                        ]
                    }
                }
            }
            # Delete all matching documents
            await self.es_client.delete_by_query(
                index=self.index_name,
                body=query,
                refresh=True,
                wait_for_completion=True,
            )
            logger.debug(f"Deleted relations for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """
        Retrieve a vector document by its ID (excluding the raw vector to save bandwidth).

        Args:
            id: Document ID to retrieve.

        Returns:
            Document data with metadata, ID, and timestamps if found; None otherwise.
        """
        try:
            # Get document without vector field for efficiency
            result = await self.es_client.get(
                index=self.index_name, id=id
            )
            if not (result.get("found", False) or (result.get("_source") is not None)):
                return None

            return {
                "id": result["_id"],
                "workspace": result["_source"].get("workspace", ""),  # Include workspace
                "created_at": result["_source"].get("created_at"),
                **result["_source"]["meta"],  # Include metadata
            }

        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve multiple vector documents by their IDs (excluding raw vectors).

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            List of document data for found IDs (excludes non-existent IDs).
        """
        if not ids:
            return []

        try:
            # Get documents without vector field for efficiency
            res = await self.es_client.mget(
                index=self.index_name, body={"ids": ids}
            )
            docs = res["docs"]
            results = []
            for doc in docs:
                if doc.get("found", False) or (doc.get("_source") is not None):
                    results.append(
                        {
                            "id": doc["_id"],
                            "workspace": doc["_source"].get("workspace", ""),
                            "created_at": doc["_source"].get("created_at"),
                            **doc["_source"]["meta"],
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Error retrieving multiple docs: {e}")
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
        """
        if not ids:
            return {}

        try:
            res = await self.es_client.mget(
                index=self.index_name, body={"ids": ids}
            )
            docs = res["docs"]
            vectors = {}
            for doc in docs:
                if (doc.get("found", False) or (doc.get("_source") is not None)) and "vector" in doc.get("_source", {}):
                    vectors[doc["_id"]] = doc["_source"]["vector"]
            return vectors
        except Exception as e:
            logger.error(f"Error retrieving vectors: {e}")
            return {}

    async def drop(self) -> dict[str, str]:
        """
        Delete the entire vector index and recreate it with the same mapping.
        Useful for resetting the vector storage.

        Returns:
            Dictionary with 'status' (success/error) and 'message' describing the result.
        """
        try:
            # Check if the index exists
            exists = await self.es_client.indices.exists(index=self.index_name)
            if exists:
                # Delete the index if it exists
                await self.es_client.indices.delete(index=self.index_name)
                logger.info(f"Dropped index {self.index_name}")

            # Recreate the index with the correct mapping
            vector_mapping = self.get_index_mapping()
            await ESClientManager.create_index_if_not_exist(
                self.index_name, vector_mapping
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping index {self.index_name}: {e}")
            return {"status": "error", "message": str(e)}

    def get_index_mapping(self) -> Dict[str, Any]:
        """
        Define the OpenSearch index mapping for vector storage.
        Includes a dense vector field with cosine similarity and metadata fields.

        Returns:
            Dictionary specifying the index mapping.
        """
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "dynamic": "strict",  # Prevent dynamic addition of new fields
                "properties": {
                    "id": {"type": "keyword"},  # Document ID (exact matches)
                    "workspace": {"type": "keyword"},  # Add workspace field like PostgreSQL
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,  # Dimension of the vector
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "created_at": {"type": "date"},  # Timestamp of creation
                    "chunk_ids": {"type": "keyword"},  # Array of chunk IDs
                    "file_paths": {"type": "keyword"},  # Array of file paths
                    "meta": {
                        "type": "object",
                        "dynamic": True,
                    },  # Metadata (dynamic fields)
                },
            }
        }


@final
@dataclass
class ESGraphStorage(BaseGraphStorage):
    """
    OpenSearch-based implementation of the BaseGraphStorage interface.
    Stores graph nodes and edges in separate OpenSearch indices with native graph capabilities.
    Supports efficient graph traversal, node/edge queries, and batch operations.
    """

    es_client: AsyncOpenSearch | None = field(default=None)
    nodes_index_name: str = field(default="", init=False)
    edges_index_name: str = field(default="", init=False)

    def __post_init__(self):
        """
        Post-initialization setup. Configures workspace and index names for nodes and edges.
        """
        # Handle workspace prefix for namespace
        es_workspace = os.environ.get("ES_WORKSPACE")
        if es_workspace and es_workspace.strip():
            effective_workspace = es_workspace.strip()
        else:
            effective_workspace = self.workspace

        # Apply workspace prefix to namespace
        if effective_workspace:
            self.namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(f"Final namespace with workspace prefix: '{self.namespace}'")

        # Set index names for nodes and edges storage
        self.nodes_index_name = f"graph_nodes_{self.namespace}"
        self.edges_index_name = f"graph_edges_{self.namespace}"

    async def initialize(self):
        """
        Initialize the graph storage. Retrieves the OpenSearch client and creates
        the nodes and edges indices with appropriate mappings if they don't exist.
        Implements robust error handling with connection retries.
        """
        try:
            if self.es_client is None:
                self.es_client = await ESClientManager.get_client()

            # Create mappings for nodes and edges indices
            nodes_mapping = self._get_nodes_index_mapping()
            edges_mapping = self._get_edges_index_mapping()

            await ESClientManager.create_index_if_not_exist(self.nodes_index_name, nodes_mapping)
            await ESClientManager.create_index_if_not_exist(self.edges_index_name, edges_mapping)
            logger.info(f"Successfully initialized ESGraphStorage for indices: {self.nodes_index_name}, {self.edges_index_name}")
            
        except (ConnectionError, TransportError) as e:
            error_msg = (f"Failed to initialize ESGraphStorage: Cannot connect to OpenSearch at "
                        f"{os.environ.get('ES_HOST', 'http://localhost:9200')}. "
                        f"Please verify that OpenSearch is running and accessible. "
                        f"Error: {str(e)}")
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during ESGraphStorage initialization: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def finalize(self):
        """
        Clean up resources by releasing the OpenSearch client.
        """
        if self.es_client is not None:
            await ESClientManager.release_client()
            self.es_client = None

    def _get_nodes_index_mapping(self) -> Dict[str, Any]:
        """
        Define the OpenSearch index mapping for graph nodes.
        Mirrors PostgreSQL structure with workspace field.
        """
        return {
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "entity_id": {"type": "keyword"},  # Node unique identifier
                    "workspace": {"type": "keyword"},  # Add workspace field like PostgreSQL
                    "entity_name": {"type": "text", "analyzer": "standard"},  # Searchable name
                    "entity_type": {"type": "keyword"},  # Node type/label
                    "description": {"type": "text", "analyzer": "standard"},  # Node description
                    "source_id": {"type": "keyword"},  # Source document/chunk ID
                    "created_at": {"type": "date"},  # Creation timestamp
                    "properties": {"type": "object", "dynamic": True},  # Additional node properties
                    "degree": {"type": "integer"},  # Node degree (cached for performance)
                },
            }
        }

    def _get_edges_index_mapping(self) -> Dict[str, Any]:
        """
        Define the OpenSearch index mapping for graph edges.
        Mirrors PostgreSQL structure with workspace field.
        """
        return {
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "source_entity_id": {"type": "keyword"},  # Source node ID
                    "target_entity_id": {"type": "keyword"},  # Target node ID
                    "workspace": {"type": "keyword"},  # Add workspace field like PostgreSQL
                    "relation": {"type": "text", "analyzer": "standard"},  # Relationship description
                    "keywords": {"type": "text", "analyzer": "standard"},  # Relationship keywords
                    "weight": {"type": "float"},  # Edge weight/strength
                    "source_id": {"type": "keyword"},  # Source document/chunk ID
                    "created_at": {"type": "date"},  # Creation timestamp
                    "properties": {"type": "object", "dynamic": True},  # Additional edge properties
                    "edge_id": {"type": "keyword"},  # Composite edge identifier
                },
            }
        }

    def _generate_edge_id(self, source_node_id: str, target_node_id: str) -> str:
        """Generate a consistent edge ID from source and target node IDs."""
        # Sort IDs to ensure undirected edge consistency (A-B same as B-A)
        sorted_ids = sorted([source_node_id, target_node_id])
        return f"{sorted_ids[0]}--{sorted_ids[1]}"

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return False
            result = await self.es_client.exists(index=self.nodes_index_name, id=node_id)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking node existence for {node_id}: {e}")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return False
            edge_id = self._generate_edge_id(source_node_id, target_node_id)
            result = await self.es_client.exists(index=self.edges_index_name, id=edge_id)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking edge existence for {source_node_id}-{target_node_id}: {e}")
            return False

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return 0
            # Query edges where node is either source or target
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source_entity_id": node_id}},
                            {"term": {"target_entity_id": node_id}}
                        ]
                    }
                }
            }

            response = await self.es_client.count(index=self.edges_index_name, body=query)
            return response["count"]
        except Exception as e:
            logger.error(f"Error calculating node degree for {node_id}: {e}")
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes)."""
        try:
            src_degree = await self.node_degree(src_id)
            tgt_degree = await self.node_degree(tgt_id)
            return src_degree + tgt_degree
        except Exception as e:
            logger.error(f"Error calculating edge degree for {src_id}-{tgt_id}: {e}")
            return 0

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return None
            result = await self.es_client.get(index=self.nodes_index_name, id=node_id)
            if result.get("found", False) or (result.get("_source") is not None):
                source = result["_source"]
                # Return flattened node properties like other implementations
                return {
                    "entity_id": source.get("entity_id", node_id),
                    "entity_name": source.get("entity_name", ""),
                    "entity_type": source.get("entity_type", ""),
                    "description": source.get("description", ""),
                    "source_id": source.get("source_id", ""),
                    "created_at": source.get("created_at", ""),
                    **source.get("properties", {}),
                }
            return None
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {e}")
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        """Get edge properties between two nodes."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return None
            edge_id = self._generate_edge_id(source_node_id, target_node_id)
            result = await self.es_client.get(index=self.edges_index_name, id=edge_id)
            if result.get("found", False) or (result.get("_source") is not None):
                source = result["_source"]
                return {
                    "source_entity_id": source.get("source_entity_id", ""),
                    "target_entity_id": source.get("target_entity_id", ""),
                    "relation": source.get("relation", ""),
                    "keywords": source.get("keywords", ""),
                    "weight": float(source.get("weight", 0.0)),
                    "source_id": source.get("source_id", ""),
                    **source.get("properties", {}),
                }
            return None
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting edge {source_node_id}-{target_node_id}: {e}")
            return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return None
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source_entity_id": source_node_id}},
                            {"term": {"target_entity_id": source_node_id}}
                        ]
                    }
                },
                "size": 10000,  # Adjust based on expected max edges per node
                "_source": ["source_entity_id", "target_entity_id"]
            }

            response = await self.es_client.search(index=self.edges_index_name, body=query)
            edges = []

            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                src_id = source["source_entity_id"]
                tgt_id = source["target_entity_id"]

                # Return edge as tuple, ensuring source_node_id is always first
                if src_id == source_node_id:
                    edges.append((src_id, tgt_id))
                else:
                    edges.append((tgt_id, src_id))

            return edges if edges else None
        except Exception as e:
            logger.error(f"Error getting node edges for {source_node_id}: {e}")
            return None

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all nodes that are associated with the given chunk_ids."""
        if not chunk_ids:
            return []

        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return []
            query = {
                "query": {
                    "terms": {"source_id": chunk_ids}
                },
                "size": 10000,  # Adjust based on expected results
            }

            response = await self.es_client.search(index=self.nodes_index_name, body=query)
            nodes = []

            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                node_data = {
                    "entity_id": source.get("entity_id", ""),
                    "entity_name": source.get("entity_name", ""),
                    "entity_type": source.get("entity_type", ""),
                    "description": source.get("description", ""),
                    "source_id": source.get("source_id", ""),
                    **source.get("properties", {}),
                }
                nodes.append(node_data)

            return nodes
        except Exception as e:
            logger.error(f"Error getting nodes by chunk IDs: {e}")
            return []

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all edges that are associated with the given chunk_ids."""
        if not chunk_ids:
            return []

        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return []
            query = {
                "query": {
                    "terms": {"source_id": chunk_ids}
                },
                "size": 10000,  # Adjust based on expected results
            }

            response = await self.es_client.search(index=self.edges_index_name, body=query)
            edges = []

            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                edge_data = {
                    "source_entity_id": source.get("source_entity_id", ""),
                    "target_entity_id": source.get("target_entity_id", ""),
                    "relation": source.get("relation", ""),
                    "keywords": source.get("keywords", ""),
                    "weight": source.get("weight", 0.0),
                    "source_id": source.get("source_id", ""),
                    **source.get("properties", {}),
                }
                edges.append(edge_data)

            return edges
        except Exception as e:
            logger.error(f"Error getting edges by chunk IDs: {e}")
            return []

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return
            current_time = int(time.time())

            # Prepare node document with required fields including workspace
            doc = {
                "entity_id": node_id,
                "workspace": self.workspace,  # Add workspace like PostgreSQL
                "entity_name": node_data.get("entity_name", ""),
                "entity_type": node_data.get("entity_type", ""),
                "description": node_data.get("description", ""),
                "source_id": node_data.get("source_id", ""),
                "created_at": current_time,
                "properties": {k: v for k, v in node_data.items()
                             if k not in ["entity_id", "entity_name", "entity_type",
                                        "description", "source_id", "created_at", "workspace"]},
                "degree": 0,  # Will be updated when edges are added
            }

            # Upsert the node document
            await self.es_client.index(
                index=self.nodes_index_name,
                id=node_id,
                body=doc,
                refresh="wait_for"
            )

        except Exception as e:
            logger.error(f"Error upserting node {node_id}: {e}")
            raise

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return
            current_time = int(time.time())
            edge_id = self._generate_edge_id(source_node_id, target_node_id)

            # Prepare edge document with required fields including workspace
            doc = {
                "source_entity_id": source_node_id,
                "target_entity_id": target_node_id,
                "workspace": self.workspace,  # Add workspace like PostgreSQL
                "relation": edge_data.get("relation", ""),
                "keywords": edge_data.get("keywords", ""),
                "weight": float(edge_data.get("weight", 1.0)),
                "source_id": edge_data.get("source_id", ""),
                "created_at": current_time,
                "edge_id": edge_id,
                "properties": {k: v for k, v in edge_data.items()
                             if k not in ["source_entity_id", "target_entity_id",
                                        "relation", "keywords", "weight", "source_id",
                                        "created_at", "edge_id", "workspace"]},
            }

            # Upsert the edge document
            await self.es_client.index(
                index=self.edges_index_name,
                id=edge_id,
                body=doc,
                refresh="wait_for"
            )

            # Update degree count for both nodes (async)
            await self._update_node_degrees([source_node_id, target_node_id])

        except Exception as e:
            logger.error(f"Error upserting edge {source_node_id}-{target_node_id}: {e}")
            raise

    async def _update_node_degrees(self, node_ids: list[str]) -> None:
        """Update the degree count for specified nodes."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return
            for node_id in node_ids:
                degree = await self.node_degree(node_id)
                try:
                    await self.es_client.update(
                        index=self.nodes_index_name,
                        id=node_id,
                        body={"doc": {"degree": degree}}
                    )
                except NotFoundError:
                    # Node doesn't exist, ignore
                    pass
        except Exception as e:
            logger.error(f"Error updating node degrees: {e}")

    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        try:
            # First, get all edges connected to this node
            edges = await self.get_node_edges(node_id)
            if edges:
                # Delete all connected edges
                edge_ids = [self._generate_edge_id(src, tgt) for src, tgt in edges]
                await self._delete_edges_by_ids(edge_ids)

            # Delete the node
            try:
                await self.es_client.delete(
                    index=self.nodes_index_name,
                    id=node_id,
                    refresh="wait_for"
                )
            except NotFoundError:
                # Node doesn't exist, ignore
                pass

        except Exception as e:
            logger.error(f"Error deleting node {node_id}: {e}")
            raise

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Delete multiple nodes from the graph."""
        for node_id in nodes:
            await self.delete_node(node_id)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Delete multiple edges from the graph."""
        try:
            edge_ids = [self._generate_edge_id(src, tgt) for src, tgt in edges]
            await self._delete_edges_by_ids(edge_ids)

            # Update degrees for affected nodes
            affected_nodes = set()
            for src, tgt in edges:
                affected_nodes.update([src, tgt])
            await self._update_node_degrees(list(affected_nodes))

        except Exception as e:
            logger.error(f"Error removing edges: {e}")
            raise

    async def _delete_edges_by_ids(self, edge_ids: list[str]) -> None:
        """Delete edges by their IDs."""
        if not edge_ids:
            return

        try:
            actions = [
                {"_op_type": "delete", "_index": self.edges_index_name, "_id": edge_id}
                for edge_id in edge_ids
            ]

            await async_bulk(
                self.es_client, actions, refresh="wait_for", raise_on_error=False
            )
        except Exception as e:
            logger.error(f"Error deleting edges by IDs: {e}")

    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph."""
        try:
            query = {
                "size": 0,
                "aggs": {
                    "entity_ids": {
                        "terms": {
                            "field": "entity_id",
                            "size": 10000
                        }
                    }
                }
            }

            response = await self.es_client.search(index=self.nodes_index_name, body=query)

            labels = []
            for bucket in response["aggregations"]["entity_ids"]["buckets"]:
                if bucket["key"]:  # Only include non-empty labels
                    labels.append(bucket["key"])

            return sorted(labels)
        except Exception as e:
            logger.error(f"Error getting all labels: {e}")
            return []

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph of nodes where the label includes the specified node_label."""
        try:
            if self.es_client is None:
                logger.error("ESGraphStorage not initialized - es_client is None")
                return KnowledgeGraph(nodes=[], edges=[])

            # Find nodes with matching label
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {"wildcard": {"entity_name": f"*{node_label}*"}},
                            {"wildcard": {"entity_type": f"*{node_label}*"}},
                            {"wildcard": {"description": f"*{node_label}*"}}
                        ]
                    }
                },
                "size": max_nodes
            }

            nodes_response = await self.es_client.search(index=self.nodes_index_name, body=query)

            nodes = []
            node_ids = set()
            # Map entity_id to internal id for consistency with PostgreSQL
            entity_to_internal_id = {}
            internal_id_counter = 1

            # Process found nodes - create structure similar to PostgreSQL
            for hit in nodes_response["hits"]["hits"]:
                source = hit["_source"]
                entity_id = source.get("entity_id", "")
                node_ids.add(entity_id)

                # Create an internal ID like PostgreSQL does
                internal_id = str(internal_id_counter)
                entity_to_internal_id[entity_id] = internal_id
                internal_id_counter += 1

                # Structure properties like PostgreSQL - all fields in properties
                properties = {
                    "entity_id": entity_id,
                    "entity_name": source.get("entity_name", ""),
                    "entity_type": source.get("entity_type", ""),
                    "description": source.get("description", ""),
                    "source_id": source.get("source_id", ""),
                    "created_at": source.get("created_at", 0),
                    "file_path": source.get("source_id", "").replace("<SEP>", "<SEP>"),  # Keep format similar
                    **source.get("properties", {})
                }

                nodes.append(KnowledgeGraphNode(
                    id=internal_id,  # Use internal ID like PostgreSQL
                    labels=[entity_id],  # Use entity_id as label
                    properties=properties
                ))

            # Find edges between these nodes
            edges_query = {
                "query": {
                    "bool": {
                        "should": [
                            {"terms": {"source_entity_id": list(node_ids)}},
                            {"terms": {"target_entity_id": list(node_ids)}}
                        ]
                    }
                },
                "size": max_nodes * 2  # Edges might be more numerous
            }

            edges_response = await self.es_client.search(index=self.edges_index_name, body=edges_query)

            edges = []
            edge_id_counter = 1

            for hit in edges_response["hits"]["hits"]:
                source = hit["_source"]
                src_entity_id = source.get("source_entity_id", "")
                tgt_entity_id = source.get("target_entity_id", "")

                # Only include edges between our selected nodes
                if src_entity_id in node_ids and tgt_entity_id in node_ids:
                    # Use internal IDs for source and target like PostgreSQL
                    src_internal_id = entity_to_internal_id.get(src_entity_id, src_entity_id)
                    tgt_internal_id = entity_to_internal_id.get(tgt_entity_id, tgt_entity_id)

                    edge_internal_id = str(edge_id_counter)
                    edge_id_counter += 1

                    # Structure properties like PostgreSQL
                    properties = {
                        "relation": source.get("relation", ""),
                        "keywords": source.get("keywords", ""),
                        "weight": source.get("weight", 1.0),
                        "source_id": source.get("source_id", ""),
                        **source.get("properties", {})
                    }

                    edges.append(KnowledgeGraphEdge(
                        id=edge_internal_id,  # Use internal ID like PostgreSQL
                        type=source.get("relation", ""),
                        source=src_internal_id,  # Use internal ID
                        target=tgt_internal_id,  # Use internal ID
                        properties=properties
                    ))

            return KnowledgeGraph(nodes=nodes, edges=edges)

        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{node_label}': {e}")
            return KnowledgeGraph(nodes=[], edges=[])

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph."""
        try:
            all_nodes = []
            scroll = "2m"

            query = {"query": {"match_all": {}}}
            response = await self.es_client.search(
                index=self.nodes_index_name,
                body=query,
                scroll=scroll,
                size=1000,
            )

            scroll_id = response.get("_scroll_id")
            while scroll_id:
                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    node_data = {
                        "entity_id": source.get("entity_id", ""),
                        "entity_name": source.get("entity_name", ""),
                        "entity_type": source.get("entity_type", ""),
                        "description": source.get("description", ""),
                        "source_id": source.get("source_id", ""),
                        **source.get("properties", {}),
                    }
                    all_nodes.append(node_data)

                if self.es_client is not None:
                    response = await self.es_client.scroll(scroll_id=scroll_id, scroll=scroll)
                    scroll_id = response.get("_scroll_id")
                else:
                    break

            # Clear scroll context
            if scroll_id and self.es_client is not None:
                await self.es_client.clear_scroll(scroll_id=scroll_id)

            return all_nodes
        except Exception as e:
            logger.error(f"Error getting all nodes: {e}")
            return []

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph."""
        try:
            all_edges = []
            scroll = "2m"

            query = {"query": {"match_all": {}}}
            response = await self.es_client.search(
                index=self.edges_index_name,
                body=query,
                scroll=scroll,
                size=1000,
            )

            scroll_id = response.get("_scroll_id")
            while scroll_id:
                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    edge_data = {
                        "source_entity_id": source.get("source_entity_id", ""),
                        "target_entity_id": source.get("target_entity_id", ""),
                        "relation": source.get("relation", ""),
                        "keywords": source.get("keywords", ""),
                        "weight": source.get("weight", 1.0),
                        "source_id": source.get("source_id", ""),
                        **source.get("properties", {}),
                    }
                    all_edges.append(edge_data)

                if self.es_client is not None:
                    response = await self.es_client.scroll(scroll_id=scroll_id, scroll=scroll)
                    scroll_id = response.get("_scroll_id")
                else:
                    break

            # Clear scroll context
            if scroll_id and self.es_client is not None:
                await self.es_client.clear_scroll(scroll_id=scroll_id)

            return all_edges
        except Exception as e:
            logger.error(f"Error getting all edges: {e}")
            return []

    async def index_done_callback(self) -> None:
        """Callback invoked after indexing completes. No specific operation implemented for graph storage."""
        pass

    async def drop(self) -> dict[str, str]:
        """Drop all graph data and recreate indices."""
        try:
            # Check if indices exist and delete them
            for index_name in [self.nodes_index_name, self.edges_index_name]:
                exists = await self.es_client.indices.exists(index=index_name)
                if exists:
                    await self.es_client.indices.delete(index=index_name)
                    logger.info(f"Dropped graph index {index_name}")

            # Recreate indices with correct mappings
            nodes_mapping = self._get_nodes_index_mapping()
            edges_mapping = self._get_edges_index_mapping()

            await ESClientManager.create_index_if_not_exist(self.nodes_index_name, nodes_mapping)
            await ESClientManager.create_index_if_not_exist(self.edges_index_name, edges_mapping)

            return {"status": "success", "message": "Graph data dropped and indices recreated"}
        except Exception as e:
            logger.error(f"Error dropping graph data: {e}")
            return {"status": "error", "message": str(e)}