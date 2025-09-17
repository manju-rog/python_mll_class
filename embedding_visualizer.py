#!/usr/bin/env python3
"""
Visual representation of embeddings (simplified to 2D for understanding)
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.router import load_catalog, get_embedder_and_index, score_embed
from sklearn.decomposition import PCA

def visualize_embeddings_2d():
    """
    Reduce 384-dimensional embeddings to 2D for visualization
    """
    
    print("=" * 80)
    print("EMBEDDING VISUALIZATION (2D Projection)")
    print("=" * 80)
    
    # Load system
    catalog = load_catalog(Path('apps.yaml'))
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    
    # Test messages
    test_messages = [
        "who was absent last week",
        "show yesterday's absences", 
        "mark John absent today",
        "draft a TDO for microservice",
        "create technical design outline",
        "architecture document needed",
        "estimate cost for new API",
        "what's the budget required",
        "predict infrastructure costs",
        "what's the weather today",
        "tell me a joke",
        "how to cook pasta"
    ]
    
    # Get embeddings for test messages
    test_embeddings = embedder.encode(test_messages, normalize_embeddings=True)
    
    # Combine app and test embeddings
    all_embeddings = np.vstack([embs, test_embeddings])
    all_labels = app_ids + [f"msg_{i}" for i in range(len(test_messages))]
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot app embeddings (larger, different shapes)
    app_colors = ['red', 'blue', 'green']
    app_markers = ['s', '^', 'D']  # square, triangle, diamond
    
    for i, app_id in enumerate(app_ids):
        x, y = embeddings_2d[i]
        plt.scatter(x, y, c=app_colors[i], marker=app_markers[i], s=200, 
                   label=f'{app_id} (app)', alpha=0.8, edgecolors='black')
        plt.annotate(app_id, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    # Plot test messages (smaller, circles)
    message_colors = []
    for i, msg in enumerate(test_messages):
        # Color based on which app it's closest to
        scores = score_embed(msg, embedder, embs, app_ids)
        best_app_idx = app_ids.index(max(scores.keys(), key=lambda k: scores[k]))
        message_colors.append(app_colors[best_app_idx])
    
    for i, msg in enumerate(test_messages):
        x, y = embeddings_2d[len(app_ids) + i]
        plt.scatter(x, y, c=message_colors[i], marker='o', s=50, alpha=0.6)
        
        # Annotate with shortened message
        short_msg = msg[:20] + "..." if len(msg) > 20 else msg
        plt.annotate(short_msg, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.title('Embedding Space Visualization (384D → 2D projection)\nApps are large shapes, messages are small circles', 
              fontsize=14, fontweight='bold')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('embedding_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'embedding_visualization.png'")
    
    # Show clustering analysis
    print(f"\nClustering Analysis:")
    print("-" * 50)
    
    for i, msg in enumerate(test_messages):
        scores = score_embed(msg, embedder, embs, app_ids)
        best_app = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_app]
        
        print(f"'{msg[:30]}...'")
        print(f"  → Closest to: {best_app} (similarity: {best_score:.3f})")
        
        # Show distance in 2D space (for illustration)
        msg_pos = embeddings_2d[len(app_ids) + i]
        app_idx = app_ids.index(best_app)
        app_pos = embeddings_2d[app_idx]
        distance_2d = np.linalg.norm(msg_pos - app_pos)
        print(f"  → 2D distance: {distance_2d:.3f}")
        print()

def demonstrate_similarity_matrix():
    """
    Show similarity matrix between all messages and apps
    """
    
    print("\n" + "=" * 80)
    print("SIMILARITY MATRIX DEMONSTRATION")
    print("=" * 80)
    
    catalog = load_catalog(Path('apps.yaml'))
    embedder, index, app_ids, embs = get_embedder_and_index(catalog)
    
    # Test messages grouped by expected category
    test_groups = {
        'Absence': [
            "who was absent last week",
            "show yesterday's absences",
            "mark John absent today"
        ],
        'TDO': [
            "draft a TDO for microservice", 
            "create technical design outline",
            "architecture document needed"
        ],
        'Cost': [
            "estimate cost for new API",
            "what's the budget required", 
            "predict infrastructure costs"
        ],
        'Out-of-Scope': [
            "what's the weather today",
            "tell me a joke",
            "how to cook pasta"
        ]
    }
    
    # Create similarity matrix
    all_messages = []
    group_labels = []
    
    for group, messages in test_groups.items():
        all_messages.extend(messages)
        group_labels.extend([group] * len(messages))
    
    # Calculate similarities
    similarity_matrix = []
    for msg in all_messages:
        scores = score_embed(msg, embedder, embs, app_ids)
        similarity_matrix.append([scores[app_id] for app_id in app_ids])
    
    similarity_matrix = np.array(similarity_matrix)
    
    # Create heatmap
    plt.figure(figsize=(10, 12))
    
    # Create custom labels for y-axis
    y_labels = []
    for i, msg in enumerate(all_messages):
        short_msg = msg[:25] + "..." if len(msg) > 25 else msg
        y_labels.append(f"{group_labels[i]}: {short_msg}")
    
    im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    plt.xticks(range(len(app_ids)), app_ids, rotation=45)
    plt.yticks(range(len(all_messages)), y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Similarity Score', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(all_messages)):
        for j in range(len(app_ids)):
            text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.title('Similarity Matrix: Messages vs Apps\n(Higher scores = better matches)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Applications')
    plt.ylabel('Test Messages')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('similarity_matrix.png', dpi=300, bbox_inches='tight')
    print("Similarity matrix saved as 'similarity_matrix.png'")
    
    # Print analysis
    print(f"\nMatrix Analysis:")
    print("-" * 50)
    
    for i, msg in enumerate(all_messages):
        best_app_idx = np.argmax(similarity_matrix[i])
        best_app = app_ids[best_app_idx]
        best_score = similarity_matrix[i, best_app_idx]
        
        expected_group = group_labels[i]
        print(f"{expected_group}: '{msg[:30]}...'")
        print(f"  → Best match: {best_app} ({best_score:.3f})")
        
        # Check if it matches expectation
        expected_matches = {
            'Absence': 'absence',
            'TDO': 'tdo_drafting', 
            'Cost': 'cost_estimation',
            'Out-of-Scope': None  # Should have low scores for all
        }
        
        expected = expected_matches.get(expected_group)
        if expected and best_app == expected:
            print(f"  ✓ Correct match!")
        elif expected_group == 'Out-of-Scope' and best_score < 0.4:
            print(f"  ✓ Correctly low scores (likely out-of-scope)")
        else:
            print(f"  ⚠ Unexpected match (expected {expected})")
        print()

def main():
    """Run visualization demonstrations"""
    
    print("EMBEDDING VISUALIZATION DEMONSTRATION")
    print("This will create visual representations of how embeddings work.")
    print("\nNote: This requires matplotlib. Install with: pip install matplotlib")
    print("\nPress Enter to start...")
    input()
    
    try:
        visualize_embeddings_2d()
        input("\nPress Enter to continue to similarity matrix...")
        
        demonstrate_similarity_matrix()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE!")
        print("=" * 80)
        print("\nFiles created:")
        print("1. embedding_visualization.png - 2D projection of embedding space")
        print("2. similarity_matrix.png - Heatmap of message-app similarities")
        print("\nThese visualizations show:")
        print("- How similar messages cluster together in embedding space")
        print("- How apps and messages relate in the semantic space")
        print("- Which messages are correctly matched to which apps")
        
    except ImportError:
        print("Error: matplotlib not installed. Run: pip install matplotlib")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()