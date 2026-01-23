"""
Migration script: Re-embed all existing recipes using ImageBind.

This script:
1. Exports all documents + metadata from the old 'recipes' collection
2. Creates a new 'recipes_imagebind' collection with ImageBind embeddings
3. Re-embeds and inserts documents in batches
4. Verifies migration was successful

Usage:
    python migrate_to_imagebind.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.database import get_chromadb_client
from backend.imagebind_embeddings import ImageBindEmbedder, ImageBindEmbeddingFunction


def migrate_to_imagebind(
    old_collection_name: str = "recipes",
    new_collection_name: str = "recipes_imagebind",
    batch_size: int = 32
):
    """
    Migrate existing ChromaDB data to use ImageBind embeddings.
    
    Args:
        old_collection_name: Name of existing collection to migrate from
        new_collection_name: Name of new collection to create
        batch_size: Number of documents to process at once
    """
    
    # Initialize
    client = get_chromadb_client()
    embedder = ImageBindEmbedder()
    embedding_fn = ImageBindEmbeddingFunction(embedder)
    
    # Step 1: Export from old collection
    print(f"📤 Exporting data from '{old_collection_name}' collection...")
    
    try:
        old_collection = client.get_collection(old_collection_name)
    except Exception as e:
        print(f"❌ Error: Could not find collection '{old_collection_name}'")
        print(f"   Make sure you have existing data to migrate.")
        print(f"   Error: {e}")
        return False
    
    all_data = old_collection.get(include=["documents", "metadatas"])
    
    total_docs = len(all_data["ids"])
    print(f"   Found {total_docs} documents to migrate")
    
    if total_docs == 0:
        print("❌ No documents found in the old collection. Nothing to migrate.")
        return False
    
    # Step 2: Create new collection with ImageBind embeddings
    print(f"🆕 Creating new collection '{new_collection_name}' with ImageBind embeddings...")
    
    # Delete if exists (for re-running migration)
    try:
        client.delete_collection(new_collection_name)
        print(f"   Deleted existing '{new_collection_name}' collection")
    except:
        pass
    
    new_collection = client.create_collection(
        name=new_collection_name,
        embedding_function=embedding_fn,
        metadata={
            "embedding_model": "imagebind_huge",
            "embedding_dim": "1024",
            "migrated_from": old_collection_name
        }
    )
    
    # Step 3: Re-embed and insert in batches
    print(f"🔄 Migrating {total_docs} documents in batches of {batch_size}...")
    
    failed_batches = []
    
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        
        batch_ids = all_data["ids"][i:batch_end]
        batch_docs = all_data["documents"][i:batch_end]
        batch_meta = all_data["metadatas"][i:batch_end]
        
        try:
            # Generate embeddings explicitly
            batch_embeddings = embedder.embed_text(batch_docs)
            
            new_collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embeddings
            )
            
            print(f"   ✅ Migrated {batch_end}/{total_docs} documents")
            
        except Exception as e:
            print(f"   ❌ Error in batch {i}-{batch_end}: {e}")
            failed_batches.append((i, batch_end))
    
    if failed_batches:
        print(f"\n⚠️  {len(failed_batches)} batches failed. You may need to retry.")
        return False
    
    # Step 4: Verify
    print("\n" + "="*60)
    print("✅ Migration complete! Verifying...")
    print("="*60)
    
    return verify_migration(old_collection, new_collection, embedder)


def verify_migration(old_collection, new_collection, embedder: ImageBindEmbedder) -> bool:
    """
    Verify migration was successful.
    
    Returns:
        True if verification passed, False otherwise
    """
    
    old_count = old_collection.count()
    new_count = new_collection.count()
    
    print(f"\n📊 Document counts:")
    print(f"   Old collection: {old_count} documents")
    print(f"   New collection: {new_count} documents")
    
    if old_count != new_count:
        print("   ⚠️  Document count mismatch!")
        return False
    else:
        print("   ✅ Document counts match!")
    
    # Test a sample query
    test_queries = [
        "chicken pasta recipe",
        "vegetarian dinner ideas",
        "chocolate cake dessert"
    ]
    
    print(f"\n🧪 Testing sample queries...")
    
    for test_query in test_queries:
        print(f"\n   Query: '{test_query}'")
        
        # Query new collection with ImageBind embedding
        query_embedding = embedder.embed_text([test_query])
        
        results = new_collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"][0]:
            print("   ⚠️  No results returned")
            continue
            
        print("   Top 3 results:")
        for j, (doc_id, meta, dist) in enumerate(zip(
            results["ids"][0], 
            results["metadatas"][0],
            results["distances"][0]
        )):
            recipe = meta.get("recipe", "Unknown")
            chunk_type = meta.get("type", "unknown")
            print(f"      {j+1}. [{chunk_type}] {recipe} (distance: {dist:.4f})")
    
    print("\n" + "="*60)
    print("🎉 Migration verification complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Update your code to use collection '{new_collection.name}'")
    print(f"  2. Test your application thoroughly")
    print(f"  3. Once confirmed working, you can delete the old collection")
    
    return True


def show_collection_info():
    """Show information about existing collections."""
    client = get_chromadb_client()
    collections = client.list_collections()
    
    print("\n📚 Existing collections:")
    for col in collections:
        count = client.get_collection(col.name).count()
        print(f"   - {col.name}: {count} documents")
        if col.metadata:
            for key, value in col.metadata.items():
                print(f"     {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate ChromaDB to ImageBind embeddings")
    parser.add_argument("--old", default="recipes", help="Old collection name")
    parser.add_argument("--new", default="recipes_imagebind", help="New collection name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for migration")
    parser.add_argument("--info", action="store_true", help="Show collection info only")
    
    args = parser.parse_args()
    
    if args.info:
        show_collection_info()
    else:
        print("="*60)
        print("  ImageBind Migration Script")
        print("="*60)
        show_collection_info()
        print()
        
        success = migrate_to_imagebind(
            old_collection_name=args.old,
            new_collection_name=args.new,
            batch_size=args.batch_size
        )
        
        if success:
            print("\n✅ Migration successful!")
        else:
            print("\n❌ Migration failed or incomplete.")
            sys.exit(1)
