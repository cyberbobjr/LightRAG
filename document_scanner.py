#!/usr/bin/env python3
"""
Document Scanner pour LightRAG
Scanne un répertoire et insère automatiquement tous les documents dans LightRAG
avec métadonnées configurables via l'API REST.
"""

import json
import os
import requests
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration de logging basique
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DocumentScanner:
    """Scanner de documents pour insertion automatique dans LightRAG via API REST"""
    
    def __init__(self, config_path: str = "scanner_config.json"):
        """
        Initialize le scanner avec un fichier de configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        self._setup_api_client()
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier JSON"""
        if not os.path.exists(self.config_path):
            # Créer un fichier de configuration exemple
            self._create_default_config()
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_default_config(self):
        """Crée un fichier de configuration par défaut"""
        default_config = {
            "api": {
                "base_url": "http://localhost:9621",
                "api_key": None,  # Sera lu depuis la variable d'environnement LIGHTRAG_API_KEY
                "timeout": 300,
                "max_retries": 3
            },
            "scanner": {
                "source_directory": "./documents",
                "file_extensions": [".txt", ".md", ".pdf", ".docx", ".json"],
                "recursive": True,
                "exclude_patterns": ["__pycache__", ".git", ".env", "*.pyc"],
                "max_file_size_mb": 50,
                "batch_size": 10  # Nombre de documents à traiter par batch
            },
            "metadata": {
                "global_metadata": {
                    "source": "document_scanner",
                    "language": "fr",
                    "ingestion_date": None,
                    "scanner_version": "1.0"
                },
                "file_type_metadata": {
                    ".txt": {
                        "document_type": "text",
                        "format": "plain_text"
                    },
                    ".md": {
                        "document_type": "markdown",
                        "format": "markdown"
                    },
                    ".pdf": {
                        "document_type": "document",
                        "format": "pdf"
                    },
                    ".json": {
                        "document_type": "data",
                        "format": "json"
                    }
                },
                "path_based_metadata": {
                    "rules": [
                        {
                            "pattern": "**/legal/**",
                            "metadata": {
                                "category": "legal",
                                "confidentiality": "restricted"
                            }
                        },
                        {
                            "pattern": "**/public/**",
                            "metadata": {
                                "category": "public",
                                "confidentiality": "public"
                            }
                        }
                    ]
                }
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "file": None
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
            
        print(f"Configuration par défaut créée: {self.config_path}")
        print("Veuillez éditer cette configuration avant de continuer.")
        
    def _setup_logging(self):
        """Configure le système de logging"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())
        
        format_type = log_config.get("format", "text")
        if format_type == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler si spécifié
        log_file = log_config.get("file")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def _setup_api_client(self):
        """Configure le client API"""
        api_config = self.config.get("api", {})
        self.base_url = api_config.get("base_url", "http://localhost:9621")
        
        # Récupération de la clé API depuis la config ou l'environnement
        self.api_key = api_config.get("api_key") or os.getenv("LIGHTRAG_API_KEY")
        
        self.timeout = api_config.get("timeout", 300)
        self.max_retries = api_config.get("max_retries", 3)
        
        # Configuration des headers HTTP
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Obtenir un token d'authentication
        self._authenticate()
            
        logging.info(f"Client API configuré pour {self.base_url}")
    
    def _authenticate(self) -> None:
        """Obtient un token d'authentication"""
        try:
            # Essayer d'abord avec les credentials de l'environnement
            auth_data = {
                "username": "admin",  # TODO: rendre configurable
                "password": "admin123"  # TODO: rendre configurable
            }
            
            response = requests.post(
                f"{self.base_url}/login",
                data=auth_data,  # Form data, pas JSON
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data.get("access_token")
                if token:
                    self.headers["Authorization"] = f"Bearer {token}"
                    logging.info("Authentication réussie")
                else:
                    logging.warning("Token non trouvé dans la réponse")
            else:
                logging.warning(f"Authentication échouée: {response.status_code}")
                
        except Exception as e:
            logging.warning(f"Erreur d'authentication: {e}. Tentative sans token...")
    
    def _make_api_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Effectue une requête API avec retry"""
        url = f"{self.base_url}{endpoint}"
        response = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    timeout=self.timeout,
                    **kwargs
                )
                
                if response.status_code == 200:
                    return response
                else:
                    logging.warning(f"Tentative {attempt + 1}: Status {response.status_code} pour {url}")
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                        
            except requests.exceptions.RequestException as e:
                logging.warning(f"Tentative {attempt + 1}: Erreur {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        # Si nous arrivons ici, quelque chose s'est mal passé
        if response is not None:
            return response
        else:
            raise Exception("Toutes les tentatives ont échoué sans réponse")
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Détermine si un fichier doit être traité"""
        scanner_config = self.config.get("scanner", {})
        
        # Vérifier l'extension
        extensions = scanner_config.get("file_extensions", [])
        if extensions and file_path.suffix.lower() not in extensions:
            return False
        
        # Vérifier la taille du fichier
        max_size_mb = scanner_config.get("max_file_size_mb", 50)
        if file_path.stat().st_size > max_size_mb * 1024 * 1024:
            logging.warning(f"Fichier trop volumineux ignoré: {file_path}")
            return False
        
        # Vérifier les patterns d'exclusion
        exclude_patterns = scanner_config.get("exclude_patterns", [])
        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return False
        
        return True
    
    def _generate_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Génère les métadonnées pour un fichier"""
        metadata_config = self.config.get("metadata", {})
        
        # Métadonnées globales
        metadata = metadata_config.get("global_metadata", {}).copy()
        
        # Ajouter la date d'ingestion
        metadata["ingestion_date"] = datetime.now().isoformat()
        
        # Métadonnées basées sur le type de fichier
        file_type_meta = metadata_config.get("file_type_metadata", {})
        ext_metadata = file_type_meta.get(file_path.suffix.lower(), {})
        metadata.update(ext_metadata)
        
        # Métadonnées basées sur le chemin
        path_rules = metadata_config.get("path_based_metadata", {}).get("rules", [])
        for rule in path_rules:
            pattern = rule.get("pattern", "")
            if file_path.match(pattern):
                rule_metadata = rule.get("metadata", {})
                metadata.update(rule_metadata)
        
        # Métadonnées du fichier
        metadata.update({
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": file_path.stat().st_size,
            "relative_path": str(file_path.relative_to(self.config["scanner"]["source_directory"])),
            "absolute_path": str(file_path.absolute()),
            "modification_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        })
        
        return metadata
    
    def _read_file_content(self, file_path: Path) -> str:
        """Lit le contenu d'un fichier"""
        try:
            # Pour les fichiers texte simples
            if file_path.suffix.lower() in ['.txt', '.md', '.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Pour les PDF (nécessite PyPDF2 ou pdfplumber)
            elif file_path.suffix.lower() == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\\n"
                        return text
                except ImportError:
                    logging.warning(f"PyPDF2 non installé, PDF ignoré: {file_path}")
                    return ""
            
            # Pour les DOCX (nécessite python-docx)
            elif file_path.suffix.lower() == '.docx':
                try:
                    from docx import Document
                    doc = Document(str(file_path))
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\\n"
                    return text
                except ImportError:
                    logging.warning(f"python-docx non installé, DOCX ignoré: {file_path}")
                    return ""
            
            else:
                logging.warning(f"Type de fichier non supporté: {file_path}")
                return ""
                
        except Exception as e:
            logging.error(f"Erreur lecture fichier {file_path}: {e}")
            return ""
    
    def scan_directory(self) -> List[Dict[str, Any]]:
        """Scanne le répertoire et retourne la liste des documents à traiter"""
        scanner_config = self.config.get("scanner", {})
        source_dir = Path(scanner_config.get("source_directory", "./documents"))
        
        if not source_dir.exists():
            raise FileNotFoundError(f"Répertoire source non trouvé: {source_dir}")
        
        documents = []
        recursive = scanner_config.get("recursive", True)
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in source_dir.glob(pattern):
            if file_path.is_file() and self._should_process_file(file_path):
                content = self._read_file_content(file_path)
                if content.strip():  # Ignorer les fichiers vides
                    metadata = self._generate_metadata(file_path)
                    
                    documents.append({
                        "path": file_path,
                        "content": content,
                        "metadata": metadata
                    })
                    
                    logging.info(f"Document préparé: {file_path}")
        
        logging.info(f"Total documents trouvés: {len(documents)}")
        return documents
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Insère les documents dans LightRAG"""
        if not documents:
            logging.warning("Aucun document à insérer")
            return ""
        
        # Préparer les données pour l'insertion via API
        texts = []
        file_sources = []
        metadata_list = []
        
        for doc in documents:
            texts.append(doc["content"])
            file_sources.append(str(doc["path"]))
            if doc["metadata"]:
                metadata_list.append(doc["metadata"])
            else:
                metadata_list.append({})
        
        # Insérer via API REST
        try:
            endpoint = "/documents/texts"
            payload = {
                "texts": texts,
                "file_sources": file_sources,
                "metadata_list": metadata_list
            }
            
            response = self._make_api_request("POST", endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Insertion terminée avec succès. Résultat: {result}")
                return str(result)
            else:
                error_msg = f"Erreur API {response.status_code}: {response.text}"
                logging.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            logging.error(f"Erreur lors de l'insertion: {e}")
            raise
    
    def run(self) -> str:
        """Exécute le scan et l'insertion des documents"""
        # Scanner les documents
        documents = self.scan_directory()
        
        if not documents:
            logging.info("Aucun document trouvé à traiter")
            return ""
        
        # Insérer les documents via API
        result = self.insert_documents(documents)
        logging.info(f"Traitement terminé avec succès. Résultat: {result}")
        return result


def main():
    """Point d'entrée principal"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Scanner de documents pour LightRAG")
    parser.add_argument(
        "--config", 
        default="scanner_config.json",
        help="Chemin vers le fichier de configuration JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (scan seulement, pas d'insertion)"
    )
    
    args = parser.parse_args()
    
    try:
        scanner = DocumentScanner(args.config)
        if args.dry_run:
            # Mode simulation
            documents = scanner.scan_directory()
            print(f"Mode simulation: {len(documents)} documents seraient traités")
            for doc in documents[:5]:  # Afficher les 5 premiers
                print(f"  - {doc['path']}")
                print(f"    Métadonnées: {doc['metadata']}")
            if len(documents) > 5:
                print(f"    ... et {len(documents) - 5} autres")
        else:
            # Exécution réelle
            result = scanner.run()
            print(f"Traitement terminé: {result}")
    except Exception as e:
        logging.error(f"Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()