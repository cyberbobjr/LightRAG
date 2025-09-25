#!/bin/bash

# Script d'attente pour OpenSearch
# Utilisé pour s'assurer qu'OpenSearch est prêt avant de démarrer LightRAG

# Configuration par défaut
ES_HOST="${ES_HOST:-http://localhost:9200}"
TIMEOUT="${OPENSEARCH_TIMEOUT:-120}"  # 2 minutes
RETRY_INTERVAL=5

echo "🔍 Waiting for OpenSearch at ${ES_HOST}..."
echo "⏱️  Timeout set to ${TIMEOUT} seconds"

start_time=$(date +%s)
end_time=$((start_time + TIMEOUT))

while [ $(date +%s) -lt $end_time ]; do
    # Test de connexion simple
    if curl -s -f "${ES_HOST}/_cluster/health" >/dev/null 2>&1; then
        echo "✅ OpenSearch is ready!"
        
        # Test plus détaillé du statut du cluster
        health=$(curl -s "${ES_HOST}/_cluster/health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        echo "📊 Cluster status: $health"
        
        if [ "$health" = "green" ] || [ "$health" = "yellow" ]; then
            echo "🚀 OpenSearch cluster is healthy, starting application..."
            exit 0
        else
            echo "⚠️  Cluster status is $health, but continuing anyway..."
            exit 0
        fi
    fi
    
    remaining=$((end_time - $(date +%s)))
    echo "⏳ Still waiting for OpenSearch... (${remaining}s remaining)"
    sleep $RETRY_INTERVAL
done

echo "❌ Timeout waiting for OpenSearch after ${TIMEOUT} seconds"
echo "💡 Please check:"
echo "   - OpenSearch container is running: docker ps"
echo "   - OpenSearch logs: docker logs opensearch"
echo "   - Network connectivity between containers"
echo "   - ES_HOST configuration: ${ES_HOST}"

exit 1