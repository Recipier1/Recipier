"""
Test script to verify ImageBind is working correctly.

This script tests:
1. Text embeddings
2. Image embeddings  
3. Cross-modal similarity (text ↔ image)
4. ChromaDB integration

Usage:
    python test_imagebind.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_text_embeddings():
    """Test that text embedding works and produces correct dimensions."""
    print("\n" + "="*60)
    print("Test 1: Text Embeddings")
    print("="*60)
    
    from backend.imagebind_embeddings import ImageBindEmbedder
    
    embedder = ImageBindEmbedder()
    
    texts = [
        "A delicious chicken pasta recipe",
        "Chocolate cake with frosting",
        "Vegetarian stir fry with tofu"
    ]
    
    embeddings = embedder.embed_text(texts)
    
    print(f"   Input: {len(texts)} texts")
    print(f"   Output: {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    
    assert len(embeddings) == len(texts), "Embedding count mismatch"
    assert len(embeddings[0]) == 1024, f"Expected 1024 dims, got {len(embeddings[0])}"
    
    print("   ✅ Text embedding test passed!")
    return True


def test_image_embeddings():
    """Test that image embedding works."""
    print("\n" + "="*60)
    print("Test 2: Image Embeddings")
    print("="*60)
    
    from backend.imagebind_embeddings import ImageBindEmbedder
    
    # Use sample images from ImageBind repo
    sample_images = [
        "ImageBind/.assets/dog_image.jpg",
        "ImageBind/.assets/car_image.jpg",
        "ImageBind/.assets/bird_image.jpg"
    ]
    
    # Check if files exist
    existing_images = [img for img in sample_images if Path(img).exists()]
    
    if not existing_images:
        print("   ⚠️  No sample images found. Skipping image test.")
        print("   Expected images in: ImageBind/.assets/")
        return True
    
    embedder = ImageBindEmbedder()
    embeddings = embedder.embed_image(existing_images)
    
    print(f"   Input: {len(existing_images)} images")
    print(f"   Output: {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    
    assert len(embeddings) == len(existing_images), "Embedding count mismatch"
    assert len(embeddings[0]) == 1024, f"Expected 1024 dims, got {len(embeddings[0])}"
    
    print("   ✅ Image embedding test passed!")
    return True


def test_cross_modal_similarity():
    """Test that cross-modal similarity works (text should match related images)."""
    print("\n" + "="*60)
    print("Test 3: Cross-Modal Similarity")
    print("="*60)
    
    from backend.imagebind_embeddings import ImageBindEmbedder
    
    # Check if sample images exist
    dog_image = "ImageBind/.assets/dog_image.jpg"
    if not Path(dog_image).exists():
        print("   ⚠️  Sample image not found. Skipping cross-modal test.")
        return True
    
    embedder = ImageBindEmbedder()
    
    # Embed image of a dog
    dog_img_emb = embedder.embed_image([dog_image])[0]
    
    # Embed text descriptions
    texts = ["A photo of a dog", "A photo of a car", "A photo of a bird"]
    text_embs = embedder.embed_text(texts)
    
    # Calculate similarities (dot product since embeddings are normalized)
    similarities = []
    for i, (text, text_emb) in enumerate(zip(texts, text_embs)):
        sim = np.dot(dog_img_emb, text_emb)
        similarities.append((text, sim))
        print(f"   '{text}' ↔ dog_image.jpg: {sim:.4f}")
    
    # Dog text should have highest similarity to dog image
    sorted_by_sim = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    if sorted_by_sim[0][0] == "A photo of a dog":
        print("   ✅ Cross-modal similarity test passed!")
        print("      (Dog text correctly matched dog image best)")
        return True
    else:
        print("   ⚠️  Unexpected result: dog text didn't match dog image best")
        print(f"      Best match was: {sorted_by_sim[0][0]}")
        return False


def test_chromadb_integration():
    """Test that ChromaDB integration works with ImageBind embeddings."""
    print("\n" + "="*60)
    print("Test 4: ChromaDB Integration")
    print("="*60)
    
    import chromadb
    from backend.imagebind_embeddings import ImageBindEmbedder, ImageBindEmbeddingFunction
    
    # Use in-memory client for testing
    client = chromadb.Client()
    
    embedder = ImageBindEmbedder()
    embedding_fn = ImageBindEmbeddingFunction(embedder)
    
    # Create test collection
    collection = client.create_collection(
        name="test_imagebind",
        embedding_function=embedding_fn
    )
    
    # Add some test documents
    test_docs = [
        "Spaghetti carbonara with bacon and egg",
        "Grilled chicken with lemon and herbs",
        "Chocolate brownie with walnuts",
        "Caesar salad with croutons"
    ]
    
    test_ids = [f"doc_{i}" for i in range(len(test_docs))]
    test_metadata = [{"type": "recipe", "index": i} for i in range(len(test_docs))]
    
    collection.add(
        documents=test_docs,
        ids=test_ids,
        metadatas=test_metadata
    )
    
    print(f"   Added {len(test_docs)} documents to test collection")
    
    # Test query
    query = "pasta dish"
    results = collection.query(
        query_texts=[query],
        n_results=2,
        include=["documents", "distances"]
    )
    
    print(f"   Query: '{query}'")
    print(f"   Top 2 results:")
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"      {i+1}. {doc[:50]}... (distance: {dist:.4f})")
    
    # Clean up
    client.delete_collection("test_imagebind")
    
    print("   ✅ ChromaDB integration test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("  ImageBind Integration Tests")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Text Embeddings", test_text_embeddings()))
    except Exception as e:
        print(f"   ❌ Text embedding test failed: {e}")
        results.append(("Text Embeddings", False))
    
    try:
        results.append(("Image Embeddings", test_image_embeddings()))
    except Exception as e:
        print(f"   ❌ Image embedding test failed: {e}")
        results.append(("Image Embeddings", False))
    
    try:
        results.append(("Cross-Modal Similarity", test_cross_modal_similarity()))
    except Exception as e:
        print(f"   ❌ Cross-modal test failed: {e}")
        results.append(("Cross-Modal Similarity", False))
    
    try:
        results.append(("ChromaDB Integration", test_chromadb_integration()))
    except Exception as e:
        print(f"   ❌ ChromaDB integration test failed: {e}")
        results.append(("ChromaDB Integration", False))
    
    # Summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n   {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! ImageBind is ready to use.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
