#!/usr/bin/env python3
"""
Quick test script to verify hybrid search setup.
Run this after setting up the environment to ensure everything works.
"""
import os
import sys


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Checking imports...")
    print("="*70)
    
    try:
        import chromadb
        print("✅ chromadb imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import chromadb: {e}")
        return False
    
    try:
        from rank_bm25 import BM25Okapi
        print("✅ rank_bm25 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import rank_bm25: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import openai: {e}")
        return False
    
    try:
        from backend.database import get_chromadb_client
        print("✅ backend.database imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import backend.database: {e}")
        return False
    
    try:
        from backend.search import HybridRecipeSearch
        print("✅ backend.search imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import backend.search: {e}")
        return False
    
    return True


def test_environment():
    """Test environment variables."""
    print("\n" + "="*70)
    print("TEST 2: Checking environment variables...")
    print("="*70)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.startswith("sk-"):
        print("✅ OPENAI_API_KEY is set")
        has_openai = True
    else:
        print("⚠️  OPENAI_API_KEY not set (LLM features will be disabled)")
        has_openai = False
    
    chroma_path = os.getenv("CHROMA_DB_PATH", "./recipe_db")
    print(f"✅ CHROMA_DB_PATH: {chroma_path}")
    
    return has_openai


def test_chromadb_connection():
    """Test ChromaDB connection and collection."""
    print("\n" + "="*70)
    print("TEST 3: Checking ChromaDB connection...")
    print("="*70)
    
    try:
        from backend.database import get_chromadb_client
        
        client = get_chromadb_client()
        print("✅ ChromaDB client created successfully")
        
        # Try to get or create collection
        collection = client.get_or_create_collection(name="recipes")
        print("✅ Connected to 'recipes' collection")
        
        # Get collection stats
        count = collection.count()
        print(f"📊 Collection contains {count} documents")
        
        if count == 0:
            print("⚠️  Collection is empty. Run the ingestion pipeline first:")
            print("   python -m backend.main")
            return False
        else:
            print("✅ Collection has data")
            return True
    
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_search():
    """Test hybrid search functionality."""
    print("\n" + "="*70)
    print("TEST 4: Testing hybrid search...")
    print("="*70)
    
    try:
        from backend.search import HybridRecipeSearch
        
        searcher = HybridRecipeSearch()
        print("✅ HybridRecipeSearch initialized")
        
        # Test search
        query = "chicken"
        print(f"🔍 Searching for: '{query}'")
        
        results = searcher.hybrid_search(query, top_k=3)
        print(f"✅ Search completed, found {len(results['documents'])} results")
        
        # Display results
        for i, doc in enumerate(results['documents'], 1):
            metadata = results['metadatas'][i-1]
            score = results['scores'][i-1]
            recipe = metadata.get('recipe', 'Unknown')
            chunk_type = metadata.get('type', 'unknown')
            
            print(f"\n  {i}. [{chunk_type}] {recipe}")
            print(f"     Score: {score:.4f}")
            print(f"     Preview: {doc[:80]}...")
        
        return True
    
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_integration(has_openai):
    """Test LLM integration if API key is available."""
    print("\n" + "="*70)
    print("TEST 5: Testing LLM integration...")
    print("="*70)
    
    if not has_openai:
        print("⚠️  Skipping LLM test (OPENAI_API_KEY not set)")
        return None
    
    try:
        from backend.search import HybridRecipeSearch
        
        searcher = HybridRecipeSearch()
        
        query = "What's a simple chicken recipe?"
        print(f"🤖 Asking: '{query}'")
        
        result = searcher.search_and_generate(query, top_k=5)
        
        if result.get('error'):
            print(f"❌ LLM generation failed: {result['error']}")
            return False
        
        print(f"✅ LLM response generated")
        print(f"📊 Used {result.get('tokens_used', 'unknown')} tokens")
        print(f"📚 Based on {len(result['sources']['documents'])} recipe chunks")
        print(f"\n💬 Response preview:")
        print(f"   {result['answer'][:200]}...")
        
        return True
    
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 HYBRID SEARCH SETUP VERIFICATION")
    print("="*70)
    print("\nThis script verifies that your hybrid search system is set up correctly.")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n❌ FAILED: Missing dependencies. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test 2: Environment
    has_openai = test_environment()
    results['environment'] = True  # Always pass, OpenAI is optional
    
    # Test 3: ChromaDB
    results['chromadb'] = test_chromadb_connection()
    if not results['chromadb']:
        print("\n⚠️  WARNING: ChromaDB collection is empty or inaccessible")
        print("Run the ingestion pipeline to add recipes:")
        print("  python -m backend.main")
        
        # Ask if user wants to continue
        response = input("\nContinue with remaining tests? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Test 4: Hybrid Search
    if results['chromadb']:
        results['search'] = test_hybrid_search()
    else:
        print("\n⚠️  Skipping search test (no data in collection)")
        results['search'] = None
    
    # Test 5: LLM
    results['llm'] = test_llm_integration(has_openai)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    status_emoji = {True: "✅", False: "❌", None: "⚠️ "}
    
    print(f"{status_emoji[results['imports']]} Imports: {'PASS' if results['imports'] else 'FAIL'}")
    print(f"{status_emoji[results['environment']]} Environment: PASS")
    print(f"{status_emoji[results['chromadb']]} ChromaDB: {'PASS' if results['chromadb'] else 'EMPTY/FAIL'}")
    print(f"{status_emoji[results['search']]} Hybrid Search: {'PASS' if results['search'] else 'SKIPPED/FAIL' if results['search'] is not None else 'SKIPPED'}")
    print(f"{status_emoji[results['llm']]} LLM Integration: {'PASS' if results['llm'] else 'SKIPPED/FAIL' if results['llm'] is not None else 'SKIPPED'}")
    
    # Overall status
    critical_tests = [results['imports'], results['chromadb'], results['search']]
    if all(t for t in critical_tests if t is not None):
        print("\n✅ ALL CRITICAL TESTS PASSED!")
        print("\nYour hybrid search system is ready to use.")
        print("\nNext steps:")
        print("  1. Run demo: python backend/example_search.py")
        print("  2. Start API: uvicorn backend.api:app --reload --port 8000")
        print("  3. See guide: HYBRID_SEARCH_GUIDE.md")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before using the system.")
        sys.exit(1)


if __name__ == "__main__":
    main()
