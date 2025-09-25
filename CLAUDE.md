# LightRAG Project Documentation

## Overview

LightRAG is a sophisticated Retrieval-Augmented Generation (RAG) system that combines traditional vector search with knowledge graph construction and querying. The system extracts entities and relationships from documents using Large Language Models (LLMs), builds a knowledge graph, and provides advanced query capabilities.

## Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI       │    │   Core Engine   │
│   (SPA)         │◄──►│   Server        │◄──►│   (LightRAG)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                       ▼                                 ▼                                 ▼
              ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
              │   Vector Store  │              │   Graph Store   │              │   KV Store      │
              │   (Embeddings)  │              │   (Entities &   │              │   (Documents &  │
              │                 │              │   Relations)    │              │   Metadata)     │
              └─────────────────┘              └─────────────────┘              └─────────────────┘
```

### Data Flow

#### Document Ingestion
```
Document Input → Text Chunking → Entity Extraction → Knowledge Graph Construction → Vector Indexing → Storage
```

#### Query Processing
```
Query Input → Keyword Extraction → Context Building → Retrieval (KG + Vector) → Reranking → Response Generation
```

## Key Features

### 1. Multi-Modal Storage Support
- **Vector Storage**: NanoVectorDB, Faiss, Milvus, Qdrant, PostgreSQL, Elasticsearch
- **Graph Storage**: NetworkX, PostgreSQL, Elasticsearch
- **Key-Value Storage**: JSON files, Redis, PostgreSQL, Elasticsearch
- **Document Status**: Tracking processing state and metadata

### 2. Advanced Query Modes
- **Local**: Context-dependent retrieval
- **Global**: Global knowledge retrieval
- **Hybrid**: Combined local/global approach
- **Naive**: Basic vector similarity search
- **Mix**: Knowledge graph + vector retrieval
- **Bypass**: Direct LLM processing

### 3. LLM Provider Support
- OpenAI, Anthropic Claude, Azure OpenAI, AWS Bedrock
- Hugging Face, Google Gemini, X.AI Grok
- Configurable via environment variables

### 4. Knowledge Graph Construction
- Automatic entity and relationship extraction
- Entity deduplication and merging
- Relationship consolidation
- Configurable entity types

## Configuration

### Environment Variables

#### Storage Configuration
```bash
# Storage Backend Selection
LIGHTRAG_KV_STORAGE=ESKVStorage
LIGHTRAG_VECTOR_STORAGE=ESVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=ESGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=ESDocStatusStorage

# Elasticsearch Configuration
ES_HOST=https://localhost:9200
ES_USERNAME=admin
ES_PASSWORD=password
ES_WORKSPACE=rag
ES_VERIFY_CERTS=false
ES_USE_SSL=true
```

#### LLM Configuration
```bash
# LLM Provider
LLM_BINDING=openai
LLM_MODEL=gpt-4
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

# Embedding Provider
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

#### Processing Configuration
```bash
# Document Processing
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=300
ENTITY_TYPES='["Organization", "Person", "Location", "Event"]'
SUMMARY_LANGUAGE=English

# Query Configuration
TOP_K=40
CHUNK_TOP_K=20
MAX_ENTITY_TOKENS=6000
MAX_RELATION_TOKENS=8000
MAX_TOTAL_TOKENS=30000
COSINE_THRESHOLD=0.2
```

## API Endpoints

### Document Management
- `POST /document/text` - Insert text document
- `GET /document/status` - Get processing status
- `GET /document/list` - List documents with pagination
- `DELETE /document/{doc_id}` - Delete document

### Query Processing
- `POST /query/stream` - Streaming query with WebSocket
- `POST /query/data` - Query with raw data response
- `POST /query` - Standard query processing

### Knowledge Graph
- `POST /graph/data` - Get graph data for visualization
- `GET /graph/export` - Export graph data


## Development

### Project Structure
```
lightrag/
├── __init__.py              # Package entry point
├── lightrag.py              # Core orchestrator class
├── base.py                  # Abstract storage interfaces
├── operate.py               # Core operations (extraction, querying)
├── api/                     # FastAPI server and REST API
│   ├── lightrag_server.py   # Main server
│   ├── routers/             # API route handlers
│   └── webui/               # Web interface
├── kg/                      # Storage implementations
│   ├── json_kv_impl.py      # JSON file storage
│   ├── postgres_impl.py     # PostgreSQL storage
│   ├── es_impl.py           # Elasticsearch storage
│   └── ...                  # Other storage backends
├── llm/                     # LLM provider integrations
└── tools/                   # Utilities and visualization
```

### Key Classes
- `LightRAG`: Main orchestrator class
- `BaseVectorStorage`, `BaseKVStorage`, `BaseGraphStorage`: Storage abstractions
- `QueryParam`: Query configuration model
- `DocProcessingStatus`: Document status tracking

### Adding New Storage Backends

1. Implement the base storage interfaces:
```python
class CustomVectorStorage(BaseVectorStorage):
    async def query(self, query: str, top_k: int) -> List[Dict]:
        # Implementation
        pass

    async def upsert(self, data: Dict[str, Dict]) -> None:
        # Implementation
        pass
```

2. Register in configuration:
```bash
LIGHTRAG_VECTOR_STORAGE=CustomVectorStorage
```

### Testing
- Use environment variables for configuration
- Start with JSON storage for development
- Use PostgreSQL or Elasticsearch for production

## Deployment

### Docker Support
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access web interface
http://localhost:9621
```

### Production Considerations
- Use production-ready storage backends (PostgreSQL, Elasticsearch)
- Configure authentication and CORS properly
- Set appropriate worker counts and timeouts
- Monitor memory usage for large knowledge graphs
- Use Redis for distributed caching

## Troubleshooting

### Common Issues

1. **Storage Connection Errors**
   - Verify connection strings and credentials
   - Check network connectivity to storage backends
   - Ensure proper SSL configuration

2. **Memory Issues**
   - Reduce chunk sizes and batch sizes
   - Use external storage backends instead of in-memory
   - Configure appropriate worker limits

3. **LLM Rate Limits**
   - Configure appropriate timeouts and retries
   - Use caching to reduce API calls
   - Consider local models with Ollama

4. **Graph Construction Issues**
   - Review entity extraction prompts
   - Adjust entity types and extraction parameters
   - Monitor LLM response quality

### Debugging
- Enable DEBUG logging: `LOG_LEVEL=DEBUG`
- Use verbose mode: `VERBOSE=True`
- Check processing status via API endpoints
- Monitor storage backend logs

## Performance Optimization

### Storage Optimization
- Use appropriate vector database for scale
- Configure proper indexing for graph queries
- Optimize chunk sizes for your use case
- Use connection pooling for databases

### Query Optimization
- Adjust similarity thresholds
- Use reranking for better relevance
- Configure appropriate top-k values
- Cache frequent queries

### Resource Management
- Monitor memory usage during processing
- Configure appropriate worker counts
- Use async processing for I/O operations
- Implement proper error handling and retries