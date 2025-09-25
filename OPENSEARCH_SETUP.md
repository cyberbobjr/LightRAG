# Configuration OpenSearch pour LightRAG

Ce guide explique comment résoudre les problèmes de connexion OpenSearch avec LightRAG et comment configurer correctement l'environnement Docker.

## Problème

L'erreur que vous rencontrez :
```
ConnectionError: Cannot connect to host opensearch:9200 ssl:default [Connect call failed ('10.89.5.84', 9200)]
```

Indique que LightRAG ne peut pas se connecter au serveur OpenSearch. Cela peut être dû à :

1. **OpenSearch n'est pas configuré dans docker-compose** 
2. **OpenSearch n'est pas encore prêt au démarrage de LightRAG**
3. **Configuration réseau incorrecte**

## Solutions

### Solution 1: Utiliser le docker-compose avec OpenSearch intégré

Utilisez le fichier `docker-compose.with-opensearch.yml` qui inclut OpenSearch :

```bash
# Arrêter les services existants
docker-compose down

# Utiliser le nouveau docker-compose avec OpenSearch
docker-compose -f docker-compose.with-opensearch.yml up -d
```

### Solution 2: Configuration manuelle

1. **Copiez la configuration d'environnement** :
```bash
cp .env.opensearch.example .env
```

2. **Éditez le fichier `.env`** pour ajouter vos clés API et ajuster la configuration.

3. **Utilisez le docker-compose avec OpenSearch** :
```bash
docker-compose -f docker-compose.with-opensearch.yml up -d
```

### Solution 3: Configuration avec OpenSearch externe

Si vous avez déjà un serveur OpenSearch externe :

1. Modifiez les variables d'environnement dans `.env` :
```bash
ES_HOST=http://your-opensearch-host:9200
ES_USERNAME=your_username  # si requis
ES_PASSWORD=your_password  # si requis
```

2. Assurez-vous que LightRAG peut accéder à votre serveur OpenSearch.

## Améliorations du code

Le code a été amélioré pour :

1. **Gestion robuste des erreurs de connexion** avec retry automatique
2. **Messages d'erreur plus informatifs** indiquant le problème exact
3. **Vérification de la santé d'OpenSearch** avant le démarrage de LightRAG

## Configuration des backends de stockage

Pour utiliser OpenSearch comme backend, assurez-vous que ces variables sont définies :

```bash
# Backends de stockage OpenSearch
LIGHTRAG_KV_STORAGE=ESKVStorage
LIGHTRAG_VECTOR_STORAGE=ESVectorDBStorage
LIGHTRAG_DOC_STATUS_STORAGE=ESDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=ESGraphStorage

# Configuration de connexion OpenSearch
ES_HOST=http://opensearch:9200
ES_VERIFY_CERTS=false
ES_USE_SSL=false
ES_WORKSPACE=rag
```

## Vérification

Une fois OpenSearch démarré, vous pouvez vérifier la connexion :

```bash
# Vérifier l'état d'OpenSearch
curl http://localhost:9200/_cluster/health

# Voir les logs de LightRAG
docker logs lightrag

# Voir les logs d'OpenSearch
docker logs opensearch
```

## Dépannage

- **OpenSearch lent à démarrer** : Attendez que le health check passe avant de démarrer LightRAG
- **Mémoire insuffisante** : Ajustez les paramètres Java d'OpenSearch dans docker-compose
- **Problèmes de réseau** : Vérifiez que les conteneurs sont sur le même réseau Docker

## Passage d'un backend à l'autre

Si vous voulez revenir aux backends par défaut (fichiers JSON), supprimez ou commentez les variables `LIGHTRAG_*_STORAGE` dans votre fichier `.env`.