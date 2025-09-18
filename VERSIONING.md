# Stratégie de Versioning pour LightRAG Fork

## Vue d'ensemble

Ce fork de LightRAG utilise un système de versioning personnalisé pour maintenir la traçabilité avec le projet original tout en permettant un développement indépendant.

## Format de Version

```
[MAJOR].[MINOR].[PATCH]-cyberbobjr.[FORK_VERSION]
```

### Exemples
- `1.4.9-cyberbobjr.1` - Première version du fork basée sur 1.4.9
- `1.4.9-cyberbobjr.2` - Deuxième version avec corrections/améliorations
- `1.5.0-cyberbobjr.1` - Fork basé sur une nouvelle version mineure
- `2.0.0-cyberbobjr.1` - Fork basé sur une nouvelle version majeure

## Types de Changements

### Fork Version (`fork`)
Incrémente uniquement le numéro de fork. Utilisé pour :
- Bug fixes spécifiques au fork
- Petites améliorations
- Ajustements de configuration
- Documentation

### Patch Version (`patch`)
Incrémente la version patch et remet le fork à 1. Utilisé pour :
- Bug fixes importants
- Corrections de sécurité
- Compatibilité avec le projet upstream

### Minor Version (`minor`)
Incrémente la version mineure. Utilisé pour :
- Nouvelles fonctionnalités compatibles
- Améliorations significatives
- Nouvelles APIs

### Major Version (`major`)
Incrémente la version majeure. Utilisé pour :
- Changements incompatibles
- Refonte architecture
- Nouveaux paradigmes

## Utilisation

### Manuel
```bash
# Modifier directement dans lightrag/__init__.py
__version__ = "1.4.9-cyberbobjr.2"
```

### Automatique
```bash
# Incrémenter le numéro de fork
python version_manager.py fork

# Incrémenter patch
python version_manager.py patch

# Incrémenter version mineure  
python version_manager.py minor

# Incrémenter version majeure
python version_manager.py major
```

## Synchronisation avec Upstream

### Quand une nouvelle version upstream est disponible :

1. **Analyser les changements** dans le projet original
2. **Décider du type de merge** :
   - Merge simple : garder votre numéro de version actuel + incrément fork
   - Breaking changes : nouvelle version avec suffix `-cyberbobjr.1`

3. **Exemple de workflow** :
   ```bash
   # Si upstream passe de 1.4.9 à 1.5.0
   git fetch upstream
   git merge upstream/main
   
   # Mettre à jour la version
   python version_manager.py minor  # Résultat: 1.5.0-cyberbobjr.1
   ```

## Métadonnées du Fork

### Fichiers modifiés
- `lightrag/__init__.py` : version, auteur, URL
- `lightrag/api/__init__.py` : version API (timestamp)
- `pyproject.toml` : nom du package, URLs, métadonnées

### Identification
- **Nom du package** : `lightrag-hku-cyberbobjr`
- **Auteur** : `Zirui Guo (forked by cyberbobjr)`  
- **URL** : `https://github.com/cyberbobjr/LightRAG`

## Bonnes Pratiques

1. **Toujours documenter** les changements dans CHANGELOG.md
2. **Tagger** chaque version avec git
3. **Maintenir** la compatibilité avec l'API upstream quand possible
4. **Contribution** : considérer proposer des améliorations au projet original
5. **Tests** : s'assurer que les tests passent avant chaque version

## Commandes Git Recommandées

```bash
# Après changement de version
git add .
git commit -m "chore: bump version to X.Y.Z-cyberbobjr.N"
git tag vX.Y.Z-cyberbobjr.N  
git push origin HEAD --tags

# Pour synchroniser avec upstream
git remote add upstream https://github.com/HKUDS/LightRAG.git
git fetch upstream
git merge upstream/main
```