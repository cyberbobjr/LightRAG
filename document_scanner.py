#!/usr/bin/env python3
"""
Document Scanner pour LightRAG
Scanne un répertoire et insère automatiquement tous les documents dans LightRAG
avec métadonnées configurables.
"""

import json
import os
import asyncio
import logging
from lightrag.french_chunking import create_french_chunking_func
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv  # Ajouté pour charger .env

load_dotenv()

# Import LightRAG
import sys
sys.path.append(str(Path(__file__).parent / "LightRag"))

try:
    from lightrag import LightRAG
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import EmbeddingFunc
    from lightrag.constants import *
except ImportError as e:
    print(f"Erreur d'import LightRAG: {e}")
    print("Assurez-vous que LightRAG est installé et accessible")
    sys.exit(1)


class DocumentScanner:
    """Scanner de documents pour insertion automatique dans LightRAG"""
    
    def __init__(self, config_path: str = "scanner_config.json"):
        """
        Initialize le scanner avec un fichier de configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.rag = self._initialize_rag()
        self._setup_logging()
    
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
            "scanner": {
                "source_directory": "./documents",
                "file_extensions": [".txt", ".md", ".pdf", ".docx", ".json"],
                "recursive": True,
                "exclude_patterns": ["__pycache__", ".git", ".env", "*.pyc"],
                "max_file_size_mb": 50
            },
            "lightrag": {
                "working_dir": "./rag_storage",
                "chunk_token_size": 1200,
                "model_name": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "api_key_env": "OPENAI_API_KEY"
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
    
    def _initialize_rag(self) -> LightRAG:
        """Initialise l'instance LightRAG avec la configuration"""
        rag_config = self.config.get("lightrag", {})
        
        # Configuration par défaut pour les LLM
        async def llm_model_func(
            prompt, system_prompt=None, history_messages=[], **kwargs
        ) -> str:
            return await gpt_4o_mini_complete(
                prompt, system_prompt, history_messages, **kwargs
            )
        
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await openai_embed(
                texts, model=rag_config.get("embedding_model", "text-embedding-3-small")
            )
        
        # Créer l'instance LightRAG
        rag = LightRAG(
            working_dir=rag_config.get("working_dir", "./rag_storage"),
            chunk_token_size=rag_config.get("chunk_token_size", 1200),
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=rag_config.get("embedding_dim", 1536),
                max_token_size=rag_config.get("embedding_max_tokens", 8192),
                func=embedding_func,
            ),
            chunking_func=create_french_chunking_func(),
        )

        return rag
    
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
    
    async def insert_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Insère les documents dans LightRAG"""
        if not documents:
            logging.warning("Aucun document à insérer")
            return ""
        
        # Préparer les données pour l'insertion
        contents = [doc["content"] for doc in documents]
        file_paths = [str(doc["path"]) for doc in documents]
        metadata = [doc["metadata"] for doc in documents]
        
        # Insérer via LightRAG
        try:
            track_id = await self.rag.ainsert(
                input=contents,
                file_paths=file_paths,
                metadata=metadata
            )
            
            logging.info(f"Insertion terminée. Track ID: {track_id}")
            return track_id
            
        except Exception as e:
            logging.error(f"Erreur lors de l'insertion: {e}")
            raise
    
    async def run(self) -> str:
        """Exécute le scan et l'insertion des documents"""
        # Scanner les documents
        documents = self.scan_directory()
        
        if not documents:
            logging.info("Aucun document trouvé à traiter")
            return ""
        
        # Insérer les documents
        track_id = await self.insert_documents(documents)
        logging.info(f"Traitement terminé avec succès. Track ID: {track_id}")
        return track_id

    async def initialize(self):
        """Initialise les stockages et le pipeline status de LightRAG"""
        await self.rag.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()


def main():
    """Point d'entrée principal"""
    import argparse
    
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
            asyncio.run(run_with_init(scanner))
    except Exception as e:
        logging.error(f"Erreur fatale: {e}")
        sys.exit(1)


async def run_with_init(scanner):
    await scanner.initialize()
    await scanner.run()


if __name__ == "__main__":
    main()