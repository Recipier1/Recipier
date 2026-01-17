"""
Example usage of the HybridRecipeSearch system.
Run this script to test the hybrid search functionality.
"""
from search import HybridRecipeSearch, quick_search, quick_ask


def demo_basic_search():
    """Demonstrate basic hybrid search."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Hybrid Search")
    print("="*70)
    
    searcher = HybridRecipeSearch()
    
    # Get collection stats
    stats = searcher.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Unique recipes: {stats['unique_recipes']}")
    print(f"   - Avg chunks per recipe: {stats['avg_chunks_per_recipe']:.1f}")
    print(f"   - Chunk types: {stats['chunk_types']}")
    
    # Search example
    query = "chicken pasta recipes"
    print(f"\n🔍 Searching for: '{query}'")
    
    results = searcher.hybrid_search(
        query=query,
        top_k=5,
        semantic_weight=0.7,  # 70% semantic, 30% keyword
        keyword_weight=0.3
    )
    
    print(f"\n📝 Top {len(results['documents'])} Results:")
    for i, doc in enumerate(results['documents'], 1):
        metadata = results['metadatas'][i-1]
        score = results['scores'][i-1]
        recipe = metadata.get('recipe', 'Unknown')
        chunk_type = metadata.get('type', 'unknown')
        
        print(f"\n{i}. [{chunk_type.upper()}] {recipe}")
        print(f"   Score: {score:.4f}")
        print(f"   Text: {doc[:150]}..." if len(doc) > 150 else f"   Text: {doc}")


def demo_llm_integration():
    """Demonstrate LLM-powered question answering."""
    print("\n" + "="*70)
    print("DEMO 2: LLM-Powered Question Answering")
    print("="*70)
    
    searcher = HybridRecipeSearch()
    
    # Example questions
    questions = [
        "How do I make chicken alfredo?",
        "What are some healthy breakfast recipes?",
        "Show me a recipe with spinach and chicken"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 70)
        
        result = searcher.search_and_generate(
            query=question,
            top_k=10,
            model="gpt-4o-mini"  # Cost-effective model
        )
        
        if result.get('answer'):
            print(f"\n🤖 Answer:")
            print(result['answer'])
            print(f"\n📊 Tokens used: {result.get('tokens_used', 'N/A')}")
            print(f"📚 Sources: {len(result['sources']['documents'])} recipe chunks")
        else:
            print(f"\n❌ Error: {result.get('error')}")
        
        print("\n" + "="*70)
        
        # Only show one example to avoid excessive API calls
        break


def demo_weight_comparison():
    """Compare different weight configurations."""
    print("\n" + "="*70)
    print("DEMO 3: Comparing Semantic vs Keyword Weights")
    print("="*70)
    
    searcher = HybridRecipeSearch()
    query = "spicy chicken"
    
    weight_configs = [
        (1.0, 0.0, "Pure Semantic"),
        (0.7, 0.3, "Balanced (70/30)"),
        (0.5, 0.5, "Equal Weight"),
        (0.3, 0.7, "Keyword Focused"),
        (0.0, 1.0, "Pure Keyword (BM25)")
    ]
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📊 Comparing top 3 results with different weight configurations:\n")
    
    for sem_weight, kw_weight, label in weight_configs:
        print(f"\n{label} (semantic={sem_weight}, keyword={kw_weight}):")
        print("-" * 50)
        
        results = searcher.hybrid_search(
            query=query,
            top_k=3,
            semantic_weight=sem_weight,
            keyword_weight=kw_weight
        )
        
        for i, doc in enumerate(results['documents'], 1):
            metadata = results['metadatas'][i-1]
            score = results['scores'][i-1]
            recipe = metadata.get('recipe', 'Unknown')
            
            print(f"  {i}. {recipe[:50]}... (score: {score:.4f})")


def demo_quick_functions():
    """Demonstrate convenience functions."""
    print("\n" + "="*70)
    print("DEMO 4: Quick Convenience Functions")
    print("="*70)
    
    # Quick search
    print("\n🔍 Using quick_search():")
    results = quick_search("pasta recipes", top_k=3)
    for i, doc in enumerate(results['documents'], 1):
        recipe = results['metadatas'][i-1].get('recipe', 'Unknown')
        print(f"  {i}. {recipe}")
    
    # Quick ask (only if OpenAI is configured)
    print("\n🤖 Using quick_ask():")
    answer = quick_ask("What's a simple pasta recipe?", top_k=5)
    if "error" not in answer.lower() or "not configured" not in answer.lower():
        print(f"  {answer[:50]}...")
    else:
        print(f"  (Skipped - {answer})")


if __name__ == "__main__":
    print("\n🍳 Recipe Hybrid Search Demo")
    print("="*70)
    
    try:
        # Run demos
        demo_basic_search()
        demo_weight_comparison()
        demo_quick_functions()
        
        # Only run LLM demo if API key is available
        import os
        if os.getenv("OPENAI_API_KEY"):
            demo_llm_integration()
        else:
            print("\n⚠️  Skipping LLM demo - OPENAI_API_KEY not set in .env")
        
        print("\n✅ All demos completed!")
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
