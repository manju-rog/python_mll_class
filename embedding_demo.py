#!/usr/bin/env python3
"""
Interactive demonstration of how embeddings work in the routing system
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
from app.router import load_catalog, build_app_corpus, get_embedder_and_index, score_embed
import json

def demonstrate_embedding_creation():
    """Show how app corpora are built and converted to embeddings"""
    
    print("=" * 80)
    print("EMBEDDING CREATION DEMONSTRATION")
    print("=" * 80)
    
    # Load the catalog
    catalog = load_catalog(Path('apps.yaml'))
    
    print("\n1. BUILDING APP CORPORA")
    print("-" * 50)
    
    for app in catalog['apps']:
        print(f"\n{app['id'].upper()} APP:")
        print(f"Description: {app.get('description', '')}")
        print(f"Keywords: {app.get('keywords', [])}")
        print(f"Intents: {len(app.get('intents', []))} intents")
        
        # Build the corpus
        corpus = build_app_corpus(app)
        print(f"\nFull Corpus:")
        print("-" * 30)
        print(corpus)
        print(f"\nCorpus length: {len(corpus)} characters")
        print(f"Corpus lines: {len(corpus.split())} words")

def demonstrate_embedding_encoding():
    """Show the actual embedding creation process"""
    
    print("\n\n" + "=" * 80)
    print("EMBEDDING ENCODING DEMONSTRATION")
    print("=" * 80)
    
    # Load embedder and create embeddings
    catalog = load_catalog(Path('apps.yaml'))
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    
    print(f"\n2. EMBEDDING MODEL DETAILS")
    print("-" * 50)
    print(f"Model: {embedder}")
    print(f"Embedding dimensions: {embs.shape[1]}")
    print(f"Number of apps: {embs.shape[0]}")
    print(f"Embedding data type: {embs.dtype}")
    print(f"Memory usage: {embs.nbytes / 1024:.1f} KB")
    
    print(f"\n3. APP EMBEDDINGS (first 10 dimensions)")
    print("-" * 50)
    
    for i, app_id in enumerate(app_ids):
        embedding = embs[i]
        print(f"\n{app_id}:")
        print(f"  First 10 dims: {embedding[:10]}")
        print(f"  Vector length: {np.linalg.norm(embedding):.6f}")  # Should be 1.0
        print(f"  Min value: {embedding.min():.6f}")
        print(f"  Max value: {embedding.max():.6f}")
        print(f"  Mean value: {embedding.mean():.6f}")

def demonstrate_similarity_calculation():
    """Show how similarity scores are calculated"""
    
    print("\n\n" + "=" * 80)
    print("SIMILARITY CALCULATION DEMONSTRATION")
    print("=" * 80)
    
    catalog = load_catalog(Path('apps.yaml'))
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    
    # Test messages
    test_messages = [
        "who was absent last week",
        "draft a TDO for microservice", 
        "estimate cost for new API",
        "what's the weather today",
        "hello there"
    ]
    
    print(f"\n4. SIMILARITY CALCULATIONS")
    print("-" * 50)
    
    for msg in test_messages:
        print(f"\nMessage: '{msg}'")
        print("-" * 40)
        
        # Get user message embedding
        user_emb = embedder.encode([msg], normalize_embeddings=True)[0]
        print(f"User embedding shape: {user_emb.shape}")
        print(f"User embedding (first 5): {user_emb[:5]}")
        
        # Calculate similarities manually to show the math
        print(f"\nManual similarity calculations:")
        manual_scores = {}
        for i, app_id in enumerate(app_ids):
            app_emb = embs[i]
            # Dot product (since both are normalized)
            similarity = np.dot(user_emb, app_emb)
            manual_scores[app_id] = similarity
            print(f"  {app_id}: {similarity:.6f}")
        
        # Compare with system function
        system_scores = score_embed(msg, embedder, embs, app_ids)
        print(f"\nSystem function results:")
        for app_id in app_ids:
            print(f"  {app_id}: {system_scores[app_id]:.6f}")
        
        # Verify they match
        print(f"\nVerification (should be very close):")
        for app_id in app_ids:
            diff = abs(manual_scores[app_id] - system_scores[app_id])
            print(f"  {app_id} difference: {diff:.10f}")

def demonstrate_semantic_understanding():
    """Show how embeddings capture semantic similarity"""
    
    print("\n\n" + "=" * 80)
    print("SEMANTIC UNDERSTANDING DEMONSTRATION")
    print("=" * 80)
    
    catalog = load_catalog(Path('apps.yaml'))
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    
    print(f"\n5. SEMANTIC SIMILARITY EXAMPLES")
    print("-" * 50)
    
    # Groups of similar messages
    similarity_groups = {
        "Absence Queries": [
            "who was absent last week",
            "show me yesterday's absences", 
            "which employees were out",
            "attendance report for Monday",
            "who took PTO recently"
        ],
        "TDO Requests": [
            "draft a TDO for microservice",
            "create technical design outline",
            "prepare architecture document", 
            "design spec for new API",
            "HLD for payment service"
        ],
        "Cost Questions": [
            "estimate cost for new API",
            "what's the budget needed",
            "predict infrastructure costs",
            "forecast expenses for project",
            "pricing for cloud services"
        ],
        "Out of Scope": [
            "what's the weather today",
            "tell me a joke",
            "how to cook pasta",
            "latest news updates",
            "play some music"
        ]
    }
    
    for group_name, messages in similarity_groups.items():
        print(f"\n{group_name.upper()}:")
        print("-" * 30)
        
        for msg in messages:
            scores = score_embed(msg, embedder, embs, app_ids)
            best_app = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_app]
            
            print(f"'{msg}'")
            print(f"  → Best match: {best_app} ({best_score:.3f})")
            print(f"  → All scores: {', '.join([f'{k}:{v:.3f}' for k,v in scores.items()])}")

def demonstrate_embedding_math():
    """Show the mathematical operations behind embeddings"""
    
    print("\n\n" + "=" * 80)
    print("EMBEDDING MATHEMATICS DEMONSTRATION")
    print("=" * 80)
    
    catalog = load_catalog(Path('apps.yaml'))
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    
    print(f"\n6. MATHEMATICAL OPERATIONS")
    print("-" * 50)
    
    # Take a simple example
    msg = "who was absent"
    user_emb = embedder.encode([msg], normalize_embeddings=True)[0]
    
    print(f"Message: '{msg}'")
    print(f"User embedding dimensions: {len(user_emb)}")
    print(f"User embedding is normalized: {np.linalg.norm(user_emb):.6f} (should be 1.0)")
    
    print(f"\nDot Product Calculation (manual):")
    print("-" * 40)
    
    for i, app_id in enumerate(app_ids):
        app_emb = embs[i]
        
        print(f"\n{app_id.upper()}:")
        print(f"  App embedding length: {np.linalg.norm(app_emb):.6f}")
        
        # Show first few terms of dot product
        print(f"  Dot product terms (first 5):")
        for j in range(5):
            term = user_emb[j] * app_emb[j]
            print(f"    {user_emb[j]:.6f} × {app_emb[j]:.6f} = {term:.6f}")
        
        # Full dot product
        full_dot = np.dot(user_emb, app_emb)
        print(f"  Full dot product (sum of all {len(user_emb)} terms): {full_dot:.6f}")
        
        # This is the similarity score
        print(f"  Final similarity score: {full_dot:.6f}")

def demonstrate_why_normalization_matters():
    """Show why vector normalization is important"""
    
    print("\n\n" + "=" * 80)
    print("NORMALIZATION IMPORTANCE DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n7. WHY NORMALIZATION MATTERS")
    print("-" * 50)
    
    # Create example vectors
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([2.0, 4.0, 6.0])  # Same direction, different length
    vec3 = np.array([1.0, 0.0, 0.0])  # Different direction
    
    print("Example vectors:")
    print(f"  vec1: {vec1}")
    print(f"  vec2: {vec2} (same direction as vec1, but longer)")
    print(f"  vec3: {vec3} (different direction)")
    
    print(f"\nWithout normalization:")
    print(f"  vec1 · vec2 = {np.dot(vec1, vec2):.3f}")
    print(f"  vec1 · vec3 = {np.dot(vec1, vec3):.3f}")
    print("  Problem: vec2 gets higher score just because it's longer!")
    
    # Normalize
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    vec3_norm = vec3 / np.linalg.norm(vec3)
    
    print(f"\nAfter normalization:")
    print(f"  vec1_norm: {vec1_norm}")
    print(f"  vec2_norm: {vec2_norm}")
    print(f"  vec3_norm: {vec3_norm}")
    
    print(f"\nWith normalization:")
    print(f"  vec1_norm · vec2_norm = {np.dot(vec1_norm, vec2_norm):.3f}")
    print(f"  vec1_norm · vec3_norm = {np.dot(vec1_norm, vec3_norm):.3f}")
    print("  Better: Same direction vectors get same score regardless of length!")

def main():
    """Run all demonstrations"""
    
    print("COMPLETE EMBEDDING SYSTEM DEMONSTRATION")
    print("This will show you exactly how embeddings work in our router system.")
    print("\nPress Enter to start...")
    input()
    
    try:
        demonstrate_embedding_creation()
        input("\nPress Enter to continue to encoding demonstration...")
        
        demonstrate_embedding_encoding()
        input("\nPress Enter to continue to similarity calculation...")
        
        demonstrate_similarity_calculation()
        input("\nPress Enter to continue to semantic understanding...")
        
        demonstrate_semantic_understanding()
        input("\nPress Enter to continue to mathematics...")
        
        demonstrate_embedding_math()
        input("\nPress Enter to continue to normalization...")
        
        demonstrate_why_normalization_matters()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nYou've seen:")
        print("1. How app corpora are built from YAML config")
        print("2. How text is converted to 384-dimensional vectors")
        print("3. How similarity is calculated using dot products")
        print("4. How semantic understanding works across variations")
        print("5. The mathematical operations behind the scenes")
        print("6. Why vector normalization is crucial")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()