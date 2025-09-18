# Exemples de requêtes pour l'insertion de métadonnées dans LightRAG

## 1. Insertion d'un texte unique avec métadonnées

```bash
curl -X POST "http://localhost:8020/documents/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ce document explique les principes fondamentaux de l'\''intelligence artificielle et du machine learning. Il couvre les algorithmes de base, les réseaux de neurones et les applications pratiques.",
    "file_source": "guide_ia_fundamentaux.md",
    "metadata": {
      "author": "Dr. Jean Dupont",
      "language": "fr",
      "category": "documentation_technique", 
      "created_date": "2025-01-15T09:30:00Z",
      "tags": ["IA", "machine learning", "algorithmes"],
      "version": "2.1",
      "department": "R&D",
      "level": "intermediate"
    }
  }'
```

## 2. Insertion de plusieurs textes avec métadonnées

```bash
curl -X POST "http://localhost:8020/documents/texts" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Introduction aux réseaux de neurones artificiels et leur fonctionnement.",
      "Guide pratique de l'\''apprentissage par renforcement avec des exemples concrets.",
      "Techniques avancées de traitement du langage naturel utilisant les transformers."
    ],
    "file_sources": [
      "neural_networks_intro.md",
      "reinforcement_learning_practical.md", 
      "nlp_transformers_advanced.md"
    ],
    "metadata_list": [
      {
        "author": "Marie Curie",
        "language": "fr",
        "category": "neural_networks",
        "created_date": "2025-01-15T10:00:00Z",
        "tags": ["réseaux de neurones", "apprentissage"],
        "difficulty": "beginner"
      },
      {
        "author": "Alan Turing", 
        "language": "fr",
        "category": "reinforcement_learning",
        "created_date": "2025-01-15T11:00:00Z",
        "tags": ["apprentissage par renforcement", "pratique"],
        "difficulty": "intermediate"
      },
      {
        "author": "Ada Lovelace",
        "language": "fr", 
        "category": "nlp",
        "created_date": "2025-01-15T12:00:00Z",
        "tags": ["NLP", "transformers", "technique avancée"],
        "difficulty": "advanced"
      }
    ]
  }'
```

## 3. Exemples de métadonnées courantes

### Pour de la documentation technique
```json
{
  "author": "Nom de l'auteur",
  "language": "fr|en|es|de...",
  "category": "documentation|tutorial|reference|faq",
  "created_date": "2025-01-15T09:30:00Z",
  "updated_date": "2025-01-15T15:45:00Z",
  "tags": ["mot-clé1", "mot-clé2", "mot-clé3"],
  "version": "1.2.3",
  "department": "Engineering|Sales|Support|R&D",
  "priority": "low|medium|high|critical",
  "difficulty": "beginner|intermediate|advanced|expert"
}
```

### Pour du contenu marketing
```json
{
  "author": "Équipe Marketing",
  "language": "fr", 
  "category": "marketing",
  "created_date": "2025-01-15T14:20:00Z",
  "campaign": "Q1_2025_Product_Launch",
  "target_audience": "developers|managers|executives",
  "content_type": "blog_post|whitepaper|case_study|press_release",
  "product": "LightRAG",
  "status": "draft|review|published|archived"
}
```

### Pour du contenu support client
```json
{
  "author": "Support Team",
  "language": "fr",
  "category": "support",
  "created_date": "2025-01-15T16:10:00Z", 
  "issue_type": "bug|feature_request|how_to|troubleshooting",
  "product_version": "2.1.0",
  "severity": "low|medium|high|critical",
  "resolution_status": "open|in_progress|resolved|closed",
  "customer_tier": "free|premium|enterprise"
}
```

## 4. Récupération des métadonnées lors des requêtes

Une fois les documents insérés avec métadonnées, vous pouvez les récupérer lors des requêtes :

```bash
curl -X POST "http://localhost:8020/query/data" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Comment fonctionnent les réseaux de neurones?",
    "mode": "hybrid",
    "stream": false,
    "only_need_context": false
  }'
```

La réponse contiendra des chunks avec leurs métadonnées dans le champ `document_metadata` :

```json
{
  "data": {
    "chunks": [
      {
        "content": "Introduction aux réseaux de neurones...",
        "file_path": "neural_networks_intro.md",
        "chunk_id": "chunk_123",
        "document_metadata": {
          "author": "Marie Curie",
          "language": "fr",
          "category": "neural_networks",
          "created_date": "2025-01-15T10:00:00Z",
          "tags": ["réseaux de neurones", "apprentissage"],
          "difficulty": "beginner"
        },
        "document_name": "neural_networks_intro.md",
        "full_doc_id": "doc_456"
      }
    ]
  }
}
```

## Notes importantes

1. **Optionnel**: Le champ `metadata` est entièrement optionnel
2. **Flexibilité**: Vous pouvez utiliser n'importe quelles clés dans les métadonnées
3. **Cohérence**: Pour les requêtes multiples (`/texts`), la longueur de `metadata_list` doit correspondre à celle de `texts`
4. **Types supportés**: Les valeurs peuvent être des chaînes, nombres, booléens, listes ou objets JSON
5. **Recherche**: Les métadonnées sont conservées et retournées lors des requêtes mais ne participent pas directement à la recherche vectorielle