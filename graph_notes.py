import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import tempfile
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
import collections
import re
from collections import Counter
import spacy
from wikipedia2vec import Wikipedia2Vec
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Streamlit page config MUST be first ---
st.set_page_config(layout="wide")

# --- Load Wikipedia2Vec Model with error handling ---
w2v_path = "enwiki_20180420_100d.pkl"
wikipedia2vec = None
nlp = None
status_messages = []

try:
    if os.path.exists(w2v_path):
        wikipedia2vec = Wikipedia2Vec.load(w2v_path)
        status_messages.append(("success", "Wikipedia2Vec model loaded successfully"))
    else:
        status_messages.append(("warning", f"Wikipedia2Vec model not found at {w2v_path}. Entity features will be disabled."))
except Exception as e:
    status_messages.append(("warning", f"Failed to load Wikipedia2Vec model: {e}. Entity features will be disabled."))

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    status_messages.append(("warning", f"Failed to load spaCy model: {e}. Entity extraction will be disabled."))

# --- Embedding model: bge-base-en or fallback ---
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
embed_model = None
embedding_fn = None

def load_embedding_model(model_name, force_cpu=False):
    """Load embedding model with proper device handling"""
    try:
        # Force CPU to avoid device issues
        device = 'cpu' if force_cpu else None
        model = SentenceTransformer(model_name, device=device)
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'} if force_cpu else {}
        )
        return model, hf_embeddings, None
    except Exception as e:
        return None, None, str(e)

# Try loading the primary model
embed_model, embedding_fn, error = load_embedding_model(EMBED_MODEL_NAME, force_cpu=True)

if embed_model is None:
    status_messages.append(("warning", f"Failed to load {EMBED_MODEL_NAME}: {error}"))
    # Try fallback model
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
    embed_model, embedding_fn, error = load_embedding_model(EMBED_MODEL_NAME, force_cpu=True)
    
    if embed_model is None:
        status_messages.append(("error", f"Failed to load fallback model {EMBED_MODEL_NAME}: {error}"))
        # Try even simpler approach
        try:
            from sentence_transformers import SentenceTransformer
            try:
                import torch
                # Clear any cached models and force CPU
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except ImportError:
                pass  # torch not available, continue anyway
            embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            embedding_fn = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
            status_messages.append(("success", "Loaded fallback embedding model on CPU"))
        except Exception as final_error:
            status_messages.append(("error", f"Complete failure to load any embedding model: {final_error}"))
    else:
        status_messages.append(("success", f"Loaded fallback model {EMBED_MODEL_NAME}"))
else:
    status_messages.append(("success", f"Loaded primary model {EMBED_MODEL_NAME}"))

# --- Load and deduplicate notes with error handling ---
notes = []
try:
    with open("notes.txt", "r") as f:
        text = f.read()
    notes_raw = re.findall(r'~(.*?)~', text, re.DOTALL)
    notes = list(dict.fromkeys([n.strip() for n in notes_raw if n.strip()]))
    if not notes:
        status_messages.append(("error", "No notes found in notes.txt. Please check the file format."))
except FileNotFoundError:
    status_messages.append(("error", "notes.txt file not found. Please create the file with your notes."))
except Exception as e:
    status_messages.append(("error", f"Error reading notes.txt: {e}"))

# --- Chroma vector DB ---
vector_db = None
if notes and embedding_fn is not None:
    try:
        vector_db = Chroma.from_texts(notes, embedding=embedding_fn)
    except Exception as e:
        status_messages.append(("error", f"Failed to create vector database: {e}"))
elif embedding_fn is None:
    status_messages.append(("error", "Cannot create vector database without embedding model"))

# --- Entity-aware hybrid embeddings with improved error handling ---
def extract_entities(note):
    if not nlp or not wikipedia2vec:
        return []
    
    try:
        doc = nlp(note)
        entities = []
        for ent in doc.ents:
            try:
                # Try to get the entity to see if it exists
                wikipedia2vec.get_entity(ent.text)
                entities.append(ent.text)
            except (KeyError, AttributeError):
                continue
        return entities
    except Exception:
        return []

def extract_keywords(note, top_k=10):
    """Extract important keywords from a note"""
    if not nlp:
        # Fallback: simple word extraction
        import string
        words = note.lower().translate(str.maketrans('', '', string.punctuation)).split()
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        keywords = [w for w in words if w not in common_words and len(w) > 2]
        return list(set(keywords))[:top_k]
    
    try:
        doc = nlp(note)
        # Extract nouns, proper nouns, and technical terms
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Add named entities as keywords
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                keywords.append(ent.text.lower())
        
        return list(set(keywords))[:top_k]
    except Exception:
        return []

def get_entity_similarity(entities1, entities2):
    """Calculate similarity based on shared entities"""
    if not entities1 or not entities2:
        return 0.0
    
    set1, set2 = set(entities1), set(entities2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    # Jaccard similarity with boost for multiple shared entities
    jaccard = intersection / union
    # Boost for having multiple shared entities
    entity_boost = min(intersection * 0.2, 0.5)
    return min(jaccard + entity_boost, 1.0)

def get_keyword_similarity(keywords1, keywords2):
    """Calculate similarity based on shared keywords"""
    if not keywords1 or not keywords2:
        return 0.0
    
    set1, set2 = set(keywords1), set(keywords2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def get_wikipedia_link_similarity(entities1, entities2):
    """Calculate similarity based on Wikipedia entity connections"""
    if not wikipedia2vec or not entities1 or not entities2:
        return 0.0
    
    # This is a simplified version - in practice you'd need Wikipedia's link graph
    # For now, we'll use a heuristic based on entity embedding similarity
    try:
        entity_pairs = []
        for e1 in entities1:
            for e2 in entities2:
                try:
                    vec1 = wikipedia2vec.get_entity_vector(e1)
                    vec2 = wikipedia2vec.get_entity_vector(e2)
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    entity_pairs.append(similarity)
                except (KeyError, AttributeError):
                    continue
        
        if entity_pairs:
            return max(entity_pairs)  # Take the strongest entity connection
        return 0.0
    except Exception:
        return 0.0

def compute_multi_signal_similarity(note1, note2, semantic_sim):
    """Compute comprehensive similarity using multiple signals"""
    
    # Extract features from both notes
    entities1 = extract_entities(note1)
    entities2 = extract_entities(note2)
    keywords1 = extract_keywords(note1)
    keywords2 = extract_keywords(note2)
    
    # Calculate different similarity signals
    entity_sim = get_entity_similarity(entities1, entities2)
    keyword_sim = get_keyword_similarity(keywords1, keywords2)
    wiki_link_sim = get_wikipedia_link_similarity(entities1, entities2)
    
    # Weight the different signals
    weights = {
        'semantic': 0.4,    # Base semantic similarity
        'entity': 0.25,     # Shared entities (people, places, concepts)
        'keyword': 0.20,    # Shared important terms
        'wiki_link': 0.15   # Wikipedia connectivity
    }
    
    # Combine signals
    combined_score = (
        weights['semantic'] * semantic_sim +
        weights['entity'] * entity_sim +
        weights['keyword'] * keyword_sim +
        weights['wiki_link'] * wiki_link_sim
    )
    
    # Apply bonus for multiple strong signals
    strong_signals = sum([
        semantic_sim > 0.5,
        entity_sim > 0.3,
        keyword_sim > 0.3,
        wiki_link_sim > 0.4
    ])
    
    if strong_signals >= 2:
        combined_score *= 1.2  # 20% boost for multiple strong signals
    
    return min(combined_score, 1.0), {
        'semantic': semantic_sim,
        'entity': entity_sim,
        'keyword': keyword_sim,
        'wiki_link': wiki_link_sim,
        'entities1': entities1,
        'entities2': entities2,
        'keywords1': keywords1,
        'keywords2': keywords2,
        'strong_signals': strong_signals
    }

def get_entity_embedding(note):
    if not wikipedia2vec:
        return np.zeros(100)  # Default dimension when Wikipedia2Vec is not available
    
    entities = extract_entities(note)
    
    # Determine vector dimension safely
    try:
        # Try to get dimension from the model
        if hasattr(wikipedia2vec, 'syn0'):
            vector_dim = wikipedia2vec.syn0.shape[1]
        else:
            # Fallback to getting dimension from any entity
            sample_entities = ['the', 'and', 'or', 'a', 'an']  # Common entities likely to exist
            vector_dim = 100  # Default from filename
            for sample_ent in sample_entities:
                try:
                    sample_vec = wikipedia2vec.get_entity_vector(sample_ent)
                    vector_dim = len(sample_vec)
                    break
                except (KeyError, AttributeError):
                    continue
    except Exception:
        vector_dim = 100  # Fallback dimension from model filename
    
    if not entities:
        return np.zeros(vector_dim)
    
    vectors = []
    for ent in entities:
        try:
            vectors.append(wikipedia2vec.get_entity_vector(ent))
        except (KeyError, AttributeError):
            continue
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_dim)

def get_hybrid_embedding(note, mode="hybrid"):
    if embed_model is None:
        # Fallback to simple text processing if no embedding model
        return np.random.rand(384)  # Default dimension for simple fallback
    
    try:
        sent_vec = embed_model.encode(note)
        
        if mode in ["sentence_only", "sentence_enhanced", "simple_semantic", "smart_multi"]:
            return sent_vec  # Multi-signal enhancement happens later in graph construction
        elif mode == "entity_only":
            entity_vec = get_entity_embedding(note)
            return entity_vec if entity_vec.size > 0 else np.random.rand(100)
        else:  # hybrid mode
            entity_vec = get_entity_embedding(note)
            return np.concatenate([sent_vec, entity_vec])
    except Exception as e:
        # Fallback to just sentence embedding on error
        try:
            return embed_model.encode(note)
        except Exception:
            # Ultimate fallback - random vector (for debugging)
            return np.random.rand(384)

# --- Streamlit UI ---
st.title("🕸️ Semantic Note Graph")

# Display status messages
for msg_type, msg in status_messages:
    if msg_type == "success":
        st.success(msg)
    elif msg_type == "warning":
        st.warning(msg)
    elif msg_type == "error":
        st.error(msg)

# Stop execution if critical errors occurred
if not notes or embed_model is None:
    if not notes:
        st.error("Cannot proceed without notes.")
    if embed_model is None:
        st.error("Cannot proceed without embedding model.")
    st.stop()

st.markdown("""
    <style>
    div.block-container{padding-top:2rem;}
    iframe{min-height:900px !important;}
    .stButton>button{font-size:1.1rem;}
    .note-popup{background:#222;padding:1em;border-radius:8px;}
    </style>
""", unsafe_allow_html=True)

user_prompt = st.text_input("Ask for a subgraph or focus:")
use_default_nodes = st.checkbox("Show all notes (ignore prompt filtering)", value=True)

# Add embedding mode selector
embedding_mode = st.radio(
    "Connection Algorithm:",
    ["Simple Semantic", "Smart Multi-Signal", "Sentence + Entity Enhancement", "Entity Only"],
    help="'Smart Multi-Signal' uses multiple factors: semantic similarity, entities, keywords, topics, and Wikipedia connections"
)

# --- Hybrid Search ---
def hybrid_search(user_prompt, notes, k):
    try:
        prompt_words = user_prompt.lower().split()
        keyword_hits = [i for i, n in enumerate(notes) if all(w in n.lower() for w in prompt_words)]
        semantic_hits = []
        if len(keyword_hits) < k and vector_db is not None:
            results = vector_db.similarity_search(user_prompt, k=k*2)
            indices = []
            for r in results:
                try:
                    idx = notes.index(r.page_content)
                except ValueError:
                    idx = -1
                indices.append(idx)
            semantic_hits = [idx for idx in indices if idx not in keyword_hits and idx != -1]
            if len(keyword_hits) + len(semantic_hits) > k:
                semantic_hits = semantic_hits[:k - len(keyword_hits)]
        return keyword_hits + semantic_hits
    except Exception as e:
        # Return default indices on error
        return list(range(min(k, len(notes))))

# --- Node filtering ---
if use_default_nodes or not user_prompt.strip():
    relevant_notes = notes
else:
    indices = hybrid_search(user_prompt, notes, k=len(notes))
    relevant_notes = [notes[i] for i in indices]

if not relevant_notes:
    st.error("No relevant notes found.")
    st.stop()

# --- Embedding relevant notes with progress bar ---
embeddings = None
with st.spinner("Creating embeddings..."):
    try:
        mode_map = {
            "Simple Semantic": "simple_semantic",
            "Smart Multi-Signal": "smart_multi",
            "Sentence + Entity Enhancement": "sentence_enhanced", 
            "Entity Only": "entity_only"
        }
        selected_mode = mode_map[embedding_mode]
        embeddings = np.array([get_hybrid_embedding(note, mode=selected_mode) for note in relevant_notes])
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.stop()

# --- Clustering ---
clustering_method = st.selectbox("Clustering method", ["HDBSCAN (default)", "KMeans"])

clustering_info = None
try:
    if clustering_method == "HDBSCAN (default)":
        max_val = max(2, len(relevant_notes) // 2)
        if max_val > 2:
            min_cluster_size = st.slider("Min cluster size (HDBSCAN)", 2, max_val, 2)
        else:
            min_cluster_size = 2
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)
        if hasattr(labels, "all") and np.all(labels == -1):
            labels = np.zeros(len(embeddings), dtype=int)
    else:
        def extract_kmeans_n_clusters():
            match = re.search(r"group(?: into| in)?\s*(\d+)", user_prompt.lower())
            if match:
                return int(match.group(1))
            return min(max(2, int((len(relevant_notes)/2)**0.5)), 10)

        n_clusters = extract_kmeans_n_clusters()
        clustering_info = f"KMeans using {n_clusters} clusters"
        st.info(clustering_info)
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
except Exception as e:
    clustering_warning = f"Clustering failed: {e}. Using single cluster."
    st.warning(clustering_warning)
    labels = np.zeros(len(embeddings), dtype=int)

# --- Create human-readable cluster topics ---
def generate_smart_cluster_name(cluster_notes, cluster_id):
    """Generate intelligent cluster names using multiple signals"""
    if not cluster_notes:
        return f"Cluster {cluster_id}"
    
    # Collect all entities and keywords from cluster notes
    all_entities = []
    all_keywords = []
    all_text = " ".join(cluster_notes)
    
    for note in cluster_notes:
        all_entities.extend(extract_entities(note))
        all_keywords.extend(extract_keywords(note, top_k=15))
    
    # Count frequencies
    entity_counts = Counter(all_entities)
    keyword_counts = Counter(all_keywords)
    
    # Get top entities and keywords
    top_entities = [e for e, c in entity_counts.most_common(3) if c > 1 or len(cluster_notes) == 1]
    top_keywords = [k for k, c in keyword_counts.most_common(5) if c > 1 or len(cluster_notes) == 1]
    
    # Try to create a meaningful name
    name_parts = []
    
    # Strategy 1: Use prominent entities (people, places, organizations)
    if top_entities:
        # Filter for meaningful entities (not too generic)
        meaningful_entities = [e for e in top_entities if len(e) > 2 and not e.lower() in ['the', 'and', 'or']]
        if meaningful_entities:
            name_parts.extend(meaningful_entities[:2])
    
    # Strategy 2: Use distinctive keywords  
    if top_keywords and len(name_parts) < 2:
        # Filter out generic terms
        generic_terms = {'note', 'notes', 'text', 'information', 'data', 'content', 'thing', 'things', 'way', 'ways', 'time', 'times', 'people', 'person'}
        distinctive_keywords = [k for k in top_keywords if k.lower() not in generic_terms and len(k) > 2]
        
        needed = 3 - len(name_parts)
        name_parts.extend(distinctive_keywords[:needed])
    
    # Strategy 3: Use theme detection based on common terms
    if not name_parts:
        # Look for thematic patterns
        text_lower = all_text.lower()
        themes = {
            'AI/ML': ['neural', 'model', 'training', 'algorithm', 'embedding', 'machine', 'learning', 'artificial', 'intelligence'],
            'Technology': ['software', 'code', 'programming', 'computer', 'digital', 'tech', 'system'],
            'Science': ['research', 'study', 'analysis', 'experiment', 'theory', 'scientific'],
            'Business': ['market', 'business', 'company', 'financial', 'strategy', 'revenue'],
            'Politics': ['government', 'political', 'policy', 'election', 'president', 'congress'],
            'Chemistry': ['chemical', 'reaction', 'formula', 'compound', 'molecular'],
            'Physics': ['physics', 'quantum', 'energy', 'particle', 'force'],
            'Biology': ['biological', 'cell', 'organism', 'gene', 'protein']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in text_lower for keyword in keywords):
                name_parts = [theme]
                break
    
    # Create final name
    if name_parts:
        # Clean up and format
        clean_parts = []
        for part in name_parts[:3]:  # Max 3 parts
            # Capitalize properly
            if len(part) > 1:
                clean_part = part.title() if part.islower() else part
                clean_parts.append(clean_part)
        
        return " & ".join(clean_parts) if len(clean_parts) > 1 else clean_parts[0]
    else:
        return f"Topic {cluster_id + 1}"

cluster_topics = {}
for cluster_id in set(labels):
    cluster_id = int(cluster_id)  # Convert numpy int64 to Python int
    cluster_notes = [relevant_notes[i] for i, l in enumerate(labels) if int(l) == cluster_id]
    cluster_topics[cluster_id] = generate_smart_cluster_name(cluster_notes, cluster_id)

# --- Graph Construction ---
similarities = cosine_similarity(embeddings)
connection_details = {}  # Initialize connection details

# --- Smart Multi-Signal Similarity (if enabled) ---
if selected_mode == "smart_multi":
    st.info("🧠 Computing smart multi-signal similarities...")
    
    # Create enhanced similarity matrix
    enhanced_similarities = np.zeros_like(similarities)
    
    for i in range(len(relevant_notes)):
        for j in range(len(relevant_notes)):
            if i != j:
                semantic_sim = similarities[i][j]
                enhanced_sim, details = compute_multi_signal_similarity(
                    relevant_notes[i], relevant_notes[j], semantic_sim
                )
                enhanced_similarities[i][j] = enhanced_sim
                connection_details[(i, j)] = details
            else:
                enhanced_similarities[i][j] = 1.0
    
    # Use enhanced similarities for connections
    similarities = enhanced_similarities
    st.success("✨ Enhanced similarities computed with multi-signal analysis")

G = nx.Graph()
for i, note in enumerate(relevant_notes):
    cluster_id = int(labels[i])  # Convert numpy int64 to Python int
    cluster_name = cluster_topics.get(cluster_id, f"Cluster {cluster_id}")
    
    # Enhanced node tooltip with cluster information
    node_tooltip = f"Note {i+1}\nCluster: {cluster_name}\n\n{note[:300]}{'...' if len(note) > 300 else ''}"
    
    G.add_node(i, 
               label=f"Note {i+1}", 
               title=node_tooltip,
               group=str(cluster_id), 
               cluster_topic=cluster_name,
               cluster_id=int(cluster_id),  # Ensure it's a Python int
               full_note=note)

# --- Connection controls ---
col1, col2 = st.columns(2)
with col1:
    if selected_mode == "smart_multi":
        similarity_threshold = st.slider(
            "Smart similarity threshold", 
            0.0, 1.0, 0.25, 0.05,
            help="Multi-signal threshold (considers semantic + entities + keywords + Wikipedia links)"
        )
    else:
        similarity_threshold = st.slider(
            "Similarity threshold (minimum to connect)", 
            0.0, 1.0, 0.3, 0.05,
            help="Only connect notes with similarity above this threshold. Higher = fewer connections."
        )
with col2:
    top_k = st.slider(
        "Max edges per node", 
        1, min(10, len(relevant_notes)-1), 
        max(2, min(3, len(relevant_notes)//3)),
        help="Maximum number of connections per note (among those above threshold)"
    )

# --- Graph layout buttons ---
col1, col2 = st.columns(2)
semantic_mode = col1.button("Semantic Graph")
sort_mode = col2.button("Central Node + Cluster Sort")

if sort_mode and len(G.nodes) > 0:
    central_idx = max(nx.degree_centrality(G), key=lambda i: nx.degree_centrality(G)[i])
    cluster_centers = {}
    for cluster_id in set(labels):
        cluster_id = int(cluster_id)  # Convert numpy int64 to Python int
        cluster_indices = [i for i, l in enumerate(labels) if int(l) == cluster_id]
        if cluster_indices:
            cluster_sims = similarities[cluster_indices][:, cluster_indices]
            center = cluster_indices[np.argmax(cluster_sims.mean(axis=1))]
            cluster_centers[cluster_id] = center
    
    # Connect cluster centers to central node (if above threshold)
    for cid, center in cluster_centers.items():
        if center != central_idx and similarities[central_idx][center] >= similarity_threshold:
            G.add_edge(central_idx, center, weight=float(similarities[central_idx][center]), edge_type="semantic")
    
    # Connect notes to their cluster centers (if above threshold)
    for i in range(len(labels)):
        cluster_id = int(labels[i])  # Convert numpy int64 to Python int
        cluster_center = cluster_centers.get(cluster_id)
        if cluster_center is not None and i != cluster_center:
            if similarities[i][cluster_center] >= similarity_threshold:
                G.add_edge(i, cluster_center, weight=float(similarities[i][cluster_center]), edge_type="semantic")
else:
    # Graph construction with smart similarity
    total_possible_edges = 0
    edges_created = 0
    
    for i in range(len(relevant_notes)):
        sims = similarities[i].copy()
        sims[i] = -1  # Don't connect to self
        
        # First, filter by similarity threshold
        valid_indices = np.where(sims >= similarity_threshold)[0]
        total_possible_edges += len(valid_indices)
        
        if len(valid_indices) > 0:
            # Among valid candidates, take top-K
            valid_sims = sims[valid_indices]
            sorted_indices = valid_indices[np.argsort(valid_sims)[::-1]]
            top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
            
            for j in top_indices:
                if not G.has_edge(i, j):
                    edge_weight = float(similarities[i][j])
                    
                    # Add connection details for smart multi-signal mode
                    if selected_mode == "smart_multi" and (i, j) in connection_details:
                        details = connection_details[(i, j)]
                        
                        # Determine dominant connection type
                        signal_scores = {
                            'semantic': details['semantic'],
                            'entity': details['entity'],
                            'keyword': details['keyword'],
                            'wiki_link': details['wiki_link']
                        }
                        dominant_signal = max(signal_scores, key=signal_scores.get)
                        
                        G.add_edge(i, j, 
                                 weight=float(edge_weight),
                                 edge_type="smart_multi",
                                 dominant_signal=dominant_signal,
                                 signal_scores={k: float(v) for k, v in signal_scores.items()},  # Convert to Python floats
                                 strong_signals=int(details['strong_signals']),  # Convert to Python int
                                 shared_entities=list(set(details['entities1']).intersection(set(details['entities2']))),
                                 shared_keywords=list(set(details['keywords1']).intersection(set(details['keywords2'])))
                        )
                    else:
                        G.add_edge(i, j, weight=edge_weight, edge_type="semantic")
                    
                    edges_created += 1
    
    # Show connection statistics
    if selected_mode == "smart_multi":
        st.info(f"🔗 Created {edges_created} smart connections out of {total_possible_edges} possible above threshold")
    else:
        st.info(f"Created {edges_created} semantic connections out of {total_possible_edges} possible above threshold")
    
    if edges_created == 0:
        st.warning("⚠️ No connections created! Try lowering the similarity threshold.")

# --- Entity Enhancement (if enabled for legacy mode) ---
entity_edges_created = 0
if selected_mode == "sentence_enhanced" and wikipedia2vec is not None and nlp is not None:
    entity_threshold = st.slider(
        "Entity connection threshold", 
        1, 5, 2,
        help="Minimum number of shared entities to create a connection"
    )
    
    st.info("🔗 Adding entity-based secondary connections...")
    
    for i in range(len(relevant_notes)):
        entities_i = set(extract_entities(relevant_notes[i]))
        
        for j in range(i+1, len(relevant_notes)):
            entities_j = set(extract_entities(relevant_notes[j]))
            shared_entities = entities_i.intersection(entities_j)
            
            if len(shared_entities) >= entity_threshold:
                # Add entity connection if not already connected or strengthen existing
                if G.has_edge(i, j):
                    # Strengthen existing semantic connection
                    current_weight = G[i][j]['weight']
                    entity_boost = min(0.2, len(shared_entities) * 0.05)  # Cap boost
                    G[i][j]['weight'] = float(min(1.0, current_weight + entity_boost))
                    G[i][j]['edge_type'] = "semantic+entity"
                    G[i][j]['shared_entities'] = list(shared_entities)
                else:
                    # Create new entity-based connection
                    entity_strength = min(0.8, len(shared_entities) * 0.15)  # Cap at 0.8
                    G.add_edge(i, j, weight=float(entity_strength), edge_type="entity", shared_entities=list(shared_entities))
                    entity_edges_created += 1
    
    if entity_edges_created > 0:
        st.success(f"✨ Added {entity_edges_created} entity-based connections")
    else:
        st.info("No entity connections met the threshold")

# --- Highlight central node ---
if len(G.nodes) > 0:
    central_idx = max(nx.degree_centrality(G), key=lambda i: nx.degree_centrality(G)[i])
    for i in G.nodes:
        if i == central_idx:
            G.nodes[i]['color'] = 'gold'
            G.nodes[i]['size'] = 55
            G.nodes[i]['borderWidth'] = 5
        else:
            G.nodes[i]['size'] = 26

# --- Visualize with PyVis ---
if len(G.nodes) > 0:
    net = Network(height="950px", width="100%", bgcolor="#181818", font_color="white", cdn_resources="in_line")
    G = nx.relabel_nodes(G, lambda x: str(x))
    
    # Style edges based on type and similarity strength
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        edge_type = edge[2].get('edge_type', 'semantic')
        
        if edge_type == 'entity':
            # Entity connections - blue/purple tones
            G.edges[edge[0], edge[1]]['color'] = '#9966ff'  # Purple for entity connections
            G.edges[edge[0], edge[1]]['width'] = 3
            G.edges[edge[0], edge[1]]['dashes'] = True  # Dashed for entity
            shared_entities = edge[2].get('shared_entities', [])
            G.edges[edge[0], edge[1]]['title'] = f"Entity connection (shared: {', '.join(shared_entities[:3])})"
        elif edge_type == 'semantic+entity':
            # Enhanced semantic connections - gold/orange
            G.edges[edge[0], edge[1]]['color'] = '#ff9900'  # Orange for enhanced
            G.edges[edge[0], edge[1]]['width'] = 4
            shared_entities = edge[2].get('shared_entities', [])
            G.edges[edge[0], edge[1]]['title'] = f"Enhanced connection (entities: {', '.join(shared_entities[:3])})"
        else:
            # Pure semantic connections - original color scheme
            if weight >= 0.7:
                G.edges[edge[0], edge[1]]['color'] = '#00ff00'  # Green for strong connections
                G.edges[edge[0], edge[1]]['width'] = 4
            elif weight >= 0.5:
                G.edges[edge[0], edge[1]]['color'] = '#ffff00'  # Yellow for medium connections  
                G.edges[edge[0], edge[1]]['width'] = 2
            else:
                G.edges[edge[0], edge[1]]['color'] = '#ff6666'  # Light red for weak connections
                G.edges[edge[0], edge[1]]['width'] = 1
            G.edges[edge[0], edge[1]]['title'] = f"Semantic similarity: {weight:.3f}"
    
    net.from_nx(G)
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.write_html(tmp_file.name)
            st.components.v1.html(open(tmp_file.name).read(), height=950, scrolling=True)
            
        # Add enhanced legend
        if selected_mode == "sentence_enhanced":
            st.markdown("""
            **Enhanced Connection Legend:**
            - 🟢 **Green edges**: Strong semantic similarity (≥0.7)
            - 🟡 **Yellow edges**: Medium semantic similarity (0.5-0.7)  
            - 🔴 **Red edges**: Weak semantic similarity (below 0.5)
            - 🟣 **Purple dashed**: Entity-only connections (shared entities)
            - 🟠 **Orange thick**: Enhanced semantic + entity connections
            """)
        else:
            st.markdown("""
            **Connection Legend:**
            - 🟢 **Green edges**: Very strong similarity (≥0.7)
            - 🟡 **Yellow edges**: Medium similarity (0.5-0.7)  
            - 🔴 **Red edges**: Weak similarity (below 0.5)
            """)
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
else:
    st.warning("No nodes to display in the graph.")

# --- Interactive Note Explorer ---
if relevant_notes:
    st.header("📄 Interactive Note Explorer")
    
    # Enhanced note selector with cluster info
    note_options = []
    for i in range(len(relevant_notes)):
        cluster_id = int(labels[i])  # Convert numpy int64 to Python int
        cluster_name = cluster_topics.get(cluster_id, f"Cluster {cluster_id}")
        note_options.append((i, f"Note {i+1} - 📋 {cluster_name}"))
    
    selected_node = st.selectbox("Select a note to explore", options=note_options, format_func=lambda x: x[1])
    node_idx = selected_node[0]
    note = relevant_notes[node_idx]
    cluster_id = int(labels[node_idx])  # Convert numpy int64 to Python int
    cluster_name = cluster_topics.get(cluster_id, f"Cluster {cluster_id}")
    
    st.subheader(f"Note {node_idx+1} - 📋 {cluster_name}")
    edited_note = st.text_area("Edit note:", value=note, height=200, key=f"edit_{node_idx}")

    if st.button("Save changes to note", key=f"save_{node_idx}"):
        try:
            original_note_idx = notes.index(note)
            notes[original_note_idx] = edited_note
            with open("notes.txt", "w") as f:
                for n in notes:
                    f.write(f"~\n{n.strip()}\n~\n")
            st.success("Note updated! Please rerun the app to refresh the graph and cluster names.")
        except Exception as e:
            st.error(f"Error saving note: {e}")

    st.markdown("**Connected Notes:**")
    if len(G.nodes) > 0:
        connected = list(G.neighbors(str(node_idx)))
        if connected:
            st.write(f"Found {len(connected)} connections above threshold ({similarity_threshold:.2f}):")
            for j in connected:
                j = int(j)
                sim_score = similarities[node_idx][j]
                
                # Check for connection info
                edge_data = G.get_edge_data(str(node_idx), str(j), {})
                edge_type = edge_data.get('edge_type', 'semantic')
                
                connection_info = f"**Note {j+1}** (Score: {sim_score:.3f}"
                
                if edge_type == 'smart_multi':
                    dominant_signal = edge_data.get('dominant_signal', 'unknown')
                    signal_scores = edge_data.get('signal_scores', {})
                    strong_signals = edge_data.get('strong_signals', 0)
                    shared_entities = edge_data.get('shared_entities', [])
                    shared_keywords = edge_data.get('shared_keywords', [])
                    
                    connection_info += f", Dominant: {dominant_signal.title()}, Strong signals: {strong_signals}/4"
                    if shared_entities:
                        connection_info += f", Entities: {', '.join(shared_entities[:2])}"
                    if shared_keywords:
                        connection_info += f", Keywords: {', '.join(shared_keywords[:2])}"
                        
                elif edge_type == 'entity':
                    shared_entities = edge_data.get('shared_entities', [])
                    connection_info += f", Entity connection: {', '.join(shared_entities[:3])}"
                elif edge_type == 'semantic+entity':
                    shared_entities = edge_data.get('shared_entities', [])
                    connection_info += f", Enhanced by entities: {', '.join(shared_entities[:3])}"
                
                connection_info += ")"
                
                st.markdown(f"- {connection_info}<br>{relevant_notes[j][:200]}{'...' if len(relevant_notes[j])>200 else ''}", unsafe_allow_html=True)
        else:
            st.write(f"No connections above similarity threshold ({similarity_threshold:.2f})")
            
            # Show the closest notes even if below threshold
            sims = similarities[node_idx].copy()
            sims[node_idx] = -1
            closest_indices = np.argsort(sims)[::-1][:3]
            st.write("**Closest notes (below threshold):**")
            for j in closest_indices:
                sim_score = similarities[node_idx][j]
                st.markdown(f"- **Note {j+1}** (Score: {sim_score:.3f}) - *Too weak to connect*<br>{relevant_notes[j][:150]}{'...' if len(relevant_notes[j])>150 else ''}", unsafe_allow_html=True)
    else:
        st.write("Graph not available.")

with st.expander("🔍 Debug: Multi-Signal Analysis" if selected_mode == "smart_multi" else "🔍 Debug: Entity Extraction & Similarities"):
    if selected_mode == "smart_multi":
        st.subheader("🧠 Multi-Signal Breakdown:")
        
        # Show signal analysis for each note pair above threshold
        high_sim_pairs = []
        for i in range(len(relevant_notes)):
            for j in range(i+1, len(relevant_notes)):
                if similarities[i][j] > similarity_threshold and (i, j) in connection_details:
                    high_sim_pairs.append((i, j, similarities[i][j]))
        
        high_sim_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        
        for i, j, sim_score in high_sim_pairs[:10]:  # Show top 10
            details = connection_details[(i, j)]
            st.write(f"**Note {i+1} ↔ Note {j+1}** (Combined: {sim_score:.3f})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Semantic", f"{details['semantic']:.3f}")
            col2.metric("Entity", f"{details['entity']:.3f}")
            col3.metric("Keyword", f"{details['keyword']:.3f}")
            col4.metric("Wiki-Link", f"{details['wiki_link']:.3f}")
            
            if details['entities1'] or details['entities2']:
                st.write(f"*Entities:* {details['entities1']} | {details['entities2']}")
            if details['keywords1'] or details['keywords2']:
                st.write(f"*Keywords:* {details['keywords1'][:5]} | {details['keywords2'][:5]}")
            
            st.write(f"*Previews:* {relevant_notes[i][:60]}... | {relevant_notes[j][:60]}...")
            st.write("---")
    else:
        st.subheader("Entities extracted from each note:")
        for i, note in enumerate(relevant_notes):
            entities = extract_entities(note)
            keywords = extract_keywords(note, top_k=5)
            entity_str = ", ".join(entities) if entities else "No entities found"
            keyword_str = ", ".join(keywords) if keywords else "No keywords found"
            st.write(f"**Note {i+1}:**")
            st.write(f"- *Entities:* {entity_str}")
            st.write(f"- *Keywords:* {keyword_str}")
            st.write(f"- *Preview:* {note[:100]}{'...' if len(note) > 100 else ''}")
            st.write("---")
    
    st.subheader("Similarity Matrix:")
    st.write("Values show similarity between each pair of notes (0=different, 1=identical)")
    
    # Create a readable similarity matrix
    import pandas as pd
    sim_df = pd.DataFrame(similarities)
    sim_df.index = [f"Note {i+1}" for i in range(len(relevant_notes))]
    sim_df.columns = [f"Note {i+1}" for i in range(len(relevant_notes))]
    
    # Format to 3 decimal places
    sim_df_formatted = sim_df.round(3)
    st.dataframe(sim_df_formatted, use_container_width=True)

with st.expander("📊 Cluster Overview"):
    st.subheader("🏷️ Smart Cluster Topics")
    
    cluster_stats = []
    for cluster_id in set(labels):
        cluster_id = int(cluster_id)  # Convert numpy int64 to Python int
        cluster_name = cluster_topics[cluster_id]
        notes_in_cluster = [i for i, l in enumerate(labels) if int(l) == cluster_id]
        cluster_stats.append((cluster_id, cluster_name, len(notes_in_cluster), notes_in_cluster))
    
    # Sort by cluster size (largest first)
    cluster_stats.sort(key=lambda x: x[2], reverse=True)
    
    for cluster_id, cluster_name, size, note_indices in cluster_stats:
        st.markdown(f"**📋 {cluster_name}** ({size} notes)")
        
        # Show cluster composition
        col1, col2 = st.columns([3, 1])
        
        with col1:
            note_previews = []
            for i in note_indices:
                preview = relevant_notes[i][:80].replace('\n', ' ')
                note_previews.append(f"• Note {i+1}: {preview}{'...' if len(relevant_notes[i]) > 80 else ''}")
            st.write("\n".join(note_previews))
        
        with col2:
            # Show cluster keywords/entities if available
            cluster_notes_text = [relevant_notes[i] for i in note_indices]
            all_keywords = []
            all_entities = []
            
            for note_text in cluster_notes_text:
                all_keywords.extend(extract_keywords(note_text, top_k=10))
                all_entities.extend(extract_entities(note_text))
            
            if all_keywords:
                top_keywords = [k for k, _ in Counter(all_keywords).most_common(3)]
                st.write(f"**Keywords:** {', '.join(top_keywords)}")
            
            if all_entities:
                top_entities = [e for e, _ in Counter(all_entities).most_common(3)]
                st.write(f"**Entities:** {', '.join(top_entities)}")
        
        st.write("---")

with st.expander("📊 Similarity Analysis"):
    # Show similarity distribution
    all_similarities = []
    for i in range(len(similarities)):
        for j in range(i+1, len(similarities)):
            all_similarities.append(similarities[i][j])
    
    if all_similarities:
        # Create histogram data using numpy and display with Streamlit
        hist_data, bin_edges = np.histogram(all_similarities, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create a simple dataframe for Streamlit's bar chart
        import pandas as pd
        chart_data = pd.DataFrame({
            'Similarity Score': bin_centers,
            'Frequency': hist_data
        })
        
        st.bar_chart(chart_data.set_index('Similarity Score'))
        st.markdown(f"🔴 **Red line shows current threshold: {similarity_threshold:.2f}**")
        
        avg_sim = np.mean(all_similarities)
        max_sim = np.max(all_similarities)
        min_sim = np.min(all_similarities)
        above_threshold = np.sum(np.array(all_similarities) >= similarity_threshold)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Similarity", f"{avg_sim:.3f}")
        col2.metric("Max Similarity", f"{max_sim:.3f}")
        col3.metric("Min Similarity", f"{min_sim:.3f}")
        col4.metric("Above Threshold", f"{above_threshold}/{len(all_similarities)}")

with st.expander("Show clusters and topics"):
        for cluster_id in set(labels):
            notes_in_cluster = [f"Note {i+1}" for i, l in enumerate(labels) if l == cluster_id]
            st.markdown(f"**Cluster {cluster_id}** — Topic: *{cluster_topics.get(cluster_id, 'No topic')}*")
            st.write(", ".join(notes_in_cluster))

st.markdown("""
<style>
iframe { min-height: 950px !important; }
.stTextArea textarea { background: #222; color: #fff; font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)