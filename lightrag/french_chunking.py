"""
Fonction de chunking français optimisée utilisant NLTK
Découpe par taille de tokens avec respect des fins de phrases en français
"""

from typing import List, Dict, Any, Optional
import re

# Import NLTK avec lazy loading pour éviter les erreurs d'initialisation
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    print("NLTK disponible. Chargement effectué.")
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK n'est pas disponible. Installation nécessaire: pip install nltk")

def ensure_nltk_data():
    """Assure que les données NLTK nécessaires sont téléchargées"""
    if not NLTK_AVAILABLE:
        return False
    
    try:
        # Vérifier si les ressources sont déjà disponibles
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        try:
            # Télécharger les ressources nécessaires
            nltk.download('punkt', quiet=True)
            return True
        except Exception as e:
            print(f"Erreur lors du téléchargement des données NLTK: {e}")
            return False

def chunking_by_token_size_with_sentence_control(
    tokenizer,
    content: str,
    split_by_character: Optional[str] = None,
    split_by_character_only: bool = False,
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Découpe le texte par taille de tokens avec contrôle des fins de phrase en français.
    
    Args:
        tokenizer: Instance du tokenizer pour compter les tokens
        content: Texte à découper
        split_by_character: Caractère de division (ignoré dans cette implémentation)
        split_by_character_only: Flag de division uniquement par caractère (ignoré)
        chunk_token_size: Taille maximale d'un chunk en tokens
        chunk_overlap_token_size: Taille de chevauchement en tokens
    
    Returns:
        Liste de dictionnaires contenant les chunks avec leurs métadonnées
    """
    
    # Nettoyage du contenu
    content = content.strip()
    if not content:
        return []
    
    # Si NLTK n'est pas disponible, fallback sur la méthode par défaut
    if not NLTK_AVAILABLE or not ensure_nltk_data():
        return _fallback_chunking(
            tokenizer, content, chunk_token_size, chunk_overlap_token_size
        )
    
    try:
        # Utiliser NLTK pour découper en phrases (supporte le français)
        sentences = sent_tokenize(content, language='french')
    except Exception:
        # Si la tokenization échoue, utiliser le fallback
        return _fallback_chunking(
            tokenizer, content, chunk_token_size, chunk_overlap_token_size
        )
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    previous_chunk_sentences = []
    chunk_index = 0  # Ajout d'un compteur de chunks
    
    def create_overlap_from_sentences(sentences_list: List[str], target_tokens: int) -> str:
        """Crée le texte de chevauchement à partir des dernières phrases"""
        if not sentences_list or target_tokens <= 0:
            return ""
        
        overlap_text = ""
        overlap_tokens = 0
        
        # Prendre les dernières phrases jusqu'à atteindre la taille cible
        for sentence in reversed(sentences_list):
            sentence_tokens = len(tokenizer.encode(sentence))
            if overlap_tokens + sentence_tokens <= target_tokens:
                overlap_text = sentence + " " + overlap_text
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_text.strip()
    
    def finalize_chunk(sentences_list: List[str], overlap_text: str = "", index: int = 0) -> Dict[str, Any]:
        """Finalise un chunk à partir d'une liste de phrases"""
        if not sentences_list:
            return {"content": "", "tokens": 0, "chunk_order_index": index}
        
        chunk_content = " ".join(sentences_list)
        if overlap_text:
            chunk_content = overlap_text + " " + chunk_content
        
        chunk_content = chunk_content.strip()
        token_count = len(tokenizer.encode(chunk_content))
        
        return {
            "content": chunk_content,
            "tokens": token_count,
            "chunk_order_index": index,
        }
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # Si une phrase unique dépasse la taille maximale, la découper
        if sentence_tokens > chunk_token_size:
            # Finaliser le chunk actuel s'il existe
            if current_chunk_sentences:
                overlap_text = create_overlap_from_sentences(
                    previous_chunk_sentences, chunk_overlap_token_size
                ) if previous_chunk_sentences else ""
                
                chunk = finalize_chunk(current_chunk_sentences, overlap_text, chunk_index)
                if chunk and chunk["content"]:
                    chunks.append(chunk)
                    chunk_index += 1
                
                previous_chunk_sentences = current_chunk_sentences[:]
                current_chunk_sentences = []
                current_chunk_tokens = 0
            
            # Découper la phrase longue
            long_sentence_chunks = _split_long_sentence(
                sentence, tokenizer, chunk_token_size, chunk_overlap_token_size, chunk_index
            )
            
            for long_chunk in long_sentence_chunks:
                overlap_text = create_overlap_from_sentences(
                    previous_chunk_sentences, chunk_overlap_token_size
                ) if previous_chunk_sentences else ""
                
                if overlap_text:
                    final_content = overlap_text + " " + long_chunk["content"]
                    final_tokens = len(tokenizer.encode(final_content))
                    chunk = {
                        "content": final_content,
                        "tokens": final_tokens,
                        "chunk_order_index": long_chunk["chunk_order_index"],  # Utiliser l'index du sous-chunk
                    }
                else:
                    chunk = long_chunk
                
                chunks.append(chunk)
                previous_chunk_sentences = [long_chunk["content"]]  # Pour le prochain overlap
            
            # Mettre à jour chunk_index pour continuer après les sous-chunks
            chunk_index += len(long_sentence_chunks)
            
            continue
        
        # Vérifier si on peut ajouter cette phrase au chunk actuel
        projected_tokens = current_chunk_tokens + sentence_tokens
        
        # Ajuster pour le chevauchement si ce n'est pas le premier chunk
        if chunks and chunk_overlap_token_size > 0:
            overlap_text = create_overlap_from_sentences(
                previous_chunk_sentences, chunk_overlap_token_size
            )
            overlap_tokens = len(tokenizer.encode(overlap_text)) if overlap_text else 0
            effective_limit = chunk_token_size - overlap_tokens
        else:
            effective_limit = chunk_token_size
        
        if projected_tokens <= effective_limit:
            # Ajouter la phrase au chunk actuel
            current_chunk_sentences.append(sentence)
            current_chunk_tokens = projected_tokens
        else:
            # Finaliser le chunk actuel
            if current_chunk_sentences:
                overlap_text = create_overlap_from_sentences(
                    previous_chunk_sentences, chunk_overlap_token_size
                ) if previous_chunk_sentences else ""
                
                chunk = finalize_chunk(current_chunk_sentences, overlap_text, chunk_index)
                if chunk and chunk["content"]:
                    chunks.append(chunk)
                    chunk_index += 1
                
                previous_chunk_sentences = current_chunk_sentences[:]
            
            # Commencer un nouveau chunk avec la phrase actuelle
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
    
    # Finaliser le dernier chunk s'il existe
    if current_chunk_sentences:
        overlap_text = create_overlap_from_sentences(
            previous_chunk_sentences, chunk_overlap_token_size
        ) if previous_chunk_sentences else ""
        
        chunk = finalize_chunk(current_chunk_sentences, overlap_text, chunk_index)
        if chunk and chunk["content"]:
            chunks.append(chunk)
    
    return chunks

def _split_long_sentence(
    sentence: str, 
    tokenizer, 
    chunk_token_size: int, 
    chunk_overlap_token_size: int,
    start_index: int = 0
) -> List[Dict[str, Any]]:
    """Découpe une phrase trop longue en fragments"""
    
    # Points de découpage préférés en français
    split_patterns = [
        r',\s+',      # Virgules
        r';\s+',      # Points-virgules  
        r':\s+',      # Deux-points
        r'\s+et\s+',  # "et"
        r'\s+ou\s+',  # "ou"
        r'\s+mais\s+', # "mais"
        r'\s+car\s+',  # "car"
        r'\s+donc\s+', # "donc"
        r'\s+',       # Espaces (dernier recours)
    ]
    
    # Essayer chaque pattern de découpage
    for pattern in split_patterns:
        parts = re.split(f'({pattern})', sentence)
        if len(parts) > 1:
            # Reconstituer les fragments avec les séparateurs
            fragments = []
            for i in range(0, len(parts), 2):
                fragment = parts[i]
                if i + 1 < len(parts):
                    fragment += parts[i + 1]  # Ajouter le séparateur
                if fragment.strip():
                    fragments.append(fragment.strip())
            
            # Si on a des fragments, les traiter
            if len(fragments) > 1:
                chunks = []
                current_fragment = ""
                current_index = start_index
                
                for fragment in fragments:
                    test_content = current_fragment + (" " if current_fragment else "") + fragment
                    test_tokens = len(tokenizer.encode(test_content))
                    
                    if test_tokens <= chunk_token_size:
                        current_fragment = test_content
                    else:
                        # Finaliser le fragment actuel
                        if current_fragment:
                            chunks.append({
                                "content": current_fragment,
                                "tokens": len(tokenizer.encode(current_fragment)),
                                "chunk_order_index": current_index,
                            })
                            current_index += 1
                        current_fragment = fragment
                
                # Ajouter le dernier fragment
                if current_fragment:
                    chunks.append({
                        "content": current_fragment,
                        "tokens": len(tokenizer.encode(current_fragment)),
                        "chunk_order_index": current_index,
                    })
                
                return chunks
    
    # Si aucun découpage n'a fonctionné, découpage brutal
    # Estimation du nombre de caractères par token
    estimated_chars_per_token = len(sentence) / len(tokenizer.encode(sentence))
    chunk_chars = int(chunk_token_size * estimated_chars_per_token * 0.8)  # 80% pour la sécurité
    
    chunks = []
    pos = 0
    current_index = start_index
    while pos < len(sentence):
        end_pos = min(pos + chunk_chars, len(sentence))
        
        # Essayer de couper à un espace
        if end_pos < len(sentence):
            space_pos = sentence.rfind(' ', pos, end_pos)
            if space_pos > pos + chunk_chars * 0.5:  # Si l'espace n'est pas trop loin
                end_pos = space_pos
        
        sub_fragment = sentence[pos:end_pos].strip()
        if sub_fragment:
            chunks.append({
                "content": sub_fragment,
                "tokens": len(tokenizer.encode(sub_fragment)),
                "chunk_order_index": current_index,
            })
            current_index += 1
        
        pos = end_pos
        if sentence[pos:pos+1] == ' ':  # Ignorer l'espace de coupure
            pos += 1
    
    return chunks

def _fallback_chunking(
    tokenizer, 
    content: str, 
    chunk_token_size: int, 
    chunk_overlap_token_size: int
) -> List[Dict[str, Any]]:
    """Méthode de fallback si NLTK n'est pas disponible"""
    
    # Découpage simple par phrases avec regex français
    sentences = re.split(r'[.!?]+\s+', content)
    
    # Nettoyer et filtrer les phrases vides
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for sentence in sentences:
        # Ajouter la ponctuation manquante
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        test_tokens = len(tokenizer.encode(test_chunk))
        
        if test_tokens <= chunk_token_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "tokens": len(tokenizer.encode(current_chunk)),
                    "chunk_order_index": chunk_index,
                })
                chunk_index += 1
            current_chunk = sentence
    
    # Ajouter le dernier chunk
    if current_chunk:
        chunks.append({
            "content": current_chunk,
            "tokens": len(tokenizer.encode(current_chunk)),
            "chunk_order_index": chunk_index,
        })
    
    return chunks

def create_french_chunking_func():
    """Crée une fonction de chunking compatible avec LightRAG pour le français"""
    return chunking_by_token_size_with_sentence_control