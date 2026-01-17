"""
Hybrid search implementation using semantic search (ChromaDB) + keyword search (BM25).
Optimized for large collections (2000+ recipes, ~20,000 chunks).
"""
import time
import os
from rank_bm25 import BM25Okapi
from openai import OpenAI
from .database import get_chromadb_client


class HybridRecipeSearch:
    """
    Hybrid search combining semantic and keyword matching.
    
    Strategy for large collections:
    1. Use semantic search to get top candidates (fast - uses vector index)
    2. Re-rank candidates with BM25 keyword scoring
    3. Combine scores with weighted fusion
    
    Features:
    - Smart BM25 index caching (rebuilds only when needed)
    - Optimized for 2000+ recipes (~20k chunks)
    - LLM integration for natural language responses
    """
    
    def __init__(self, collection_name: str = "recipes"):
        """
        Initialize hybrid search engine.
        
        Args:
            collection_name: Name of ChromaDB collection to search
        """
        self.client = get_chromadb_client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # BM25 index caching
        self.bm25_index = None
        self.bm25_doc_ids = None
        self.index_timestamp = None
        
        # OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("⚠️  Warning: OPENAI_API_KEY not set. LLM features will be disabled.")
    
    def _ensure_bm25_index(self, max_age_seconds: int = 3600):
        """
        Build or refresh BM25 index if needed.
        
        Args:
            max_age_seconds: Maximum age of cached index before rebuilding (default: 1 hour)
        """
        now = time.time()
        
        # Build index if doesn't exist or is stale
        if (self.bm25_index is None or 
            self.index_timestamp is None or 
            now - self.index_timestamp > max_age_seconds):
            
            print("🔄 Building BM25 index...")
            start_time = time.time()
            
            all_docs = self.collection.get(include=["documents"])
            tokenized_docs = [doc.lower().split() for doc in all_docs['documents']]
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_doc_ids = all_docs['ids']
            self.index_timestamp = now
            
            elapsed = time.time() - start_time
            print(f"✅ Built BM25 index for {len(tokenized_docs)} documents in {elapsed:.2f}s")
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        candidate_pool: int = 50
    ) -> dict:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            candidate_pool: Number of semantic candidates to consider for BM25 re-ranking
        
        Returns:
            Dict with keys: ids, documents, metadatas, scores
        
        Example:
            >>> searcher = HybridRecipeSearch()
            >>> results = searcher.hybrid_search("chicken pasta recipes", top_k=5)
            >>> for i, doc in enumerate(results['documents']):
            ...     print(f"{i+1}. {doc} (score: {results['scores'][i]:.3f})")
        """
        self._ensure_bm25_index()
        
        # Step 1: Semantic search (fast vector lookup)
        candidate_pool = min(candidate_pool, max(top_k * 10, 50))
        
        print(f"🔍 Semantic search: retrieving top {candidate_pool} candidates...")
        semantic_results = self.collection.query(
            query_texts=[query],
            n_results=candidate_pool,
            include=["documents", "metadatas", "distances"]
        )
        
        # Step 2: BM25 re-ranking on ONLY the semantic candidates
        candidate_ids = semantic_results['ids'][0]
        candidate_docs = semantic_results['documents'][0]
        tokenized_query = query.lower().split()
        
        print(f"📊 BM25 re-ranking {len(candidate_docs)} candidates...")
        # Build mini BM25 index for just these candidates (fast!)
        tokenized_candidates = [doc.lower().split() for doc in candidate_docs]
        mini_bm25 = BM25Okapi(tokenized_candidates)
        bm25_scores = mini_bm25.get_scores(tokenized_query)
        
        # Step 3: Combine scores with weighted fusion
        semantic_distances = semantic_results['distances'][0]
        max_distance = max(semantic_distances) if semantic_distances else 1
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1
        
        combined_scores = {}
        for i, doc_id in enumerate(candidate_ids):
            # Normalize semantic score (inverse of distance)
            semantic_score = 1 - (semantic_distances[i] / max_distance)
            # Normalize BM25 score
            bm25_score = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0
            
            # Weighted combination
            combined_scores[doc_id] = (
                semantic_weight * semantic_score + 
                keyword_weight * bm25_score
            )
        
        # Get top K results
        top_ids = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Format results
        results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'scores': []
        }
        
        for doc_id, score in top_ids:
            idx = candidate_ids.index(doc_id)
            results['ids'].append(doc_id)
            results['documents'].append(candidate_docs[idx])
            results['metadatas'].append(semantic_results['metadatas'][0][idx])
            results['scores'].append(float(score))
        
        print(f"✅ Retrieved top {len(results['ids'])} results")
        return results
    
    def search_and_generate(
        self, 
        query: str, 
        top_k: int = 10,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 3000  # Increased from 1000 to allow longer responses
    ) -> dict:
        """
        Perform hybrid search and generate answer using LLM.
        
        Args:
            query: User's recipe query
            top_k: Number of context chunks to retrieve
            model: OpenAI model to use (gpt-4o-mini is cost-effective)
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with keys:
                - answer: Generated text response
                - sources: Retrieved chunks used as context
                - context_used: Formatted context string
        
        Example:
            >>> searcher = HybridRecipeSearch()
            >>> result = searcher.search_and_generate("How do I make chicken alfredo?")
            >>> print(result['answer'])
        """
        if not self.openai_client:
            return {
                "error": "OpenAI API key not configured",
                "answer": None,
                "sources": None,
                "context_used": None
            }
        
        # 1. Hybrid search to get relevant chunks
        print(f"🔍 Searching for: {query}")
        search_results = self.hybrid_search(query, top_k=top_k)
        
        # 2. Format context from retrieved chunks
        context = self._format_context(search_results)
        
        # 3. Create prompt for LLM
        prompt = self._create_prompt(query, context)
        
        # 4. Call OpenAI API
        print(f"🤖 Generating response with {model}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful cooking assistant that provides recipe advice based on the given context. Always cite specific recipes when answering."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": search_results,
                "context_used": context,
                "model": model,
                "tokens_used": response.usage.total_tokens
            }
        
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            return {
                "error": str(e),
                "answer": None,
                "sources": search_results,
                "context_used": context
            }
    
    def _format_context(self, search_results: dict) -> str:
        """
        Format search results into context string for LLM.
        Groups chunks by recipe for better readability.
        
        Args:
            search_results: Results from hybrid_search()
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Group by recipe
        recipes = {}
        for i, doc in enumerate(search_results['documents']):
            metadata = search_results['metadatas'][i]
            recipe_name = metadata.get('recipe', 'Unknown Recipe')
            
            if recipe_name not in recipes:
                recipes[recipe_name] = []
            
            recipes[recipe_name].append({
                'text': doc,
                'type': metadata.get('type', 'unknown'),
                'metadata': metadata,
                'score': search_results['scores'][i]
            })
        
        # Format each recipe's chunks
        for recipe_name, chunks in recipes.items():
            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"Recipe: {recipe_name}")
            context_parts.append('='*60)
            
            # Sort chunks by type for better organization
            type_order = {'title': 0, 'ingredients': 1, 'directions': 2, 'nutrition': 3}
            chunks.sort(key=lambda x: type_order.get(x['type'], 99))
            
            for chunk in chunks:
                chunk_type = chunk['type'].upper()
                text = chunk['text']
                score = chunk['score']
                context_parts.append(f"\n[{chunk_type}] (relevance: {score:.3f})")
                context_parts.append(text)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM with query and context.
        
        Args:
            query: User's question
            context: Formatted context from search results
        
        Returns:
            Complete prompt string
        """
        return f"""You are "Chef AI," a culinary expert and mentor. You don't just find recipes; you guide users through the preparation process with professional tips and clear instructions.

### Context from Database
{context}

### User's Request
"{query}"

### Instructions
1. Source Grounding: Only use the provided context. If instructions are missing for a specific dish, state that you can only provide the ingredients and a general preparation method.
2. Comprehensive Guidance: - Preparation: List exactly what needs to be prepped before turning on the heat (chopping, marinating, etc.).
   - Step-by-Step: Break down the cooking process into logical, numbered steps.
   - Chef's Tips: Include at least two "Pro-Tips" for each dish (e.g., "Don't overcrowd the pan," "Let the meat rest," or "Substitute X for a spicier kick").
3. Safety & Health: Highlight any common allergens found in the context (nuts, dairy, etc.) and ensure instructions mention food safety (like internal temperatures for meat).

### Output Format for Each Dish
---
## 🍳 [Recipe Name from Context]

**Why This Matches**: Explain in one clear sentence how this recipe relates to the user's query and what makes it a good fit.

### 🛒 Ingredients & Preparation
**Ingredients:**
- List each ingredient with its quantity (e.g., "2 cups all-purpose flour")
- Group by type if helpful (proteins, vegetables, spices, liquids)

**Prep Work Before Cooking:**
- Detail all prep tasks (dice 1 onion, mince 3 garlic cloves, marinate chicken for 30 minutes)
- Mention any equipment needed

### 👨‍🍳 Cooking Instructions
1. First major step with specific actions, timing, and temperature
2. Next step with visual or textural cues to know it's ready
3. Continue with sequential numbered steps
4. Include final plating or serving suggestions

### 💡 Chef's Secret Tips
**Pro Tip #1**: Share a specific technique tip that improves results (e.g., "Toast spices in a dry pan for 30 seconds before adding to release their essential oils and deepen flavor")

**Pro Tip #2**: Offer a practical substitution or variation (e.g., "For a lighter version, swap heavy cream with Greek yogurt added off-heat to prevent curdling")

**Allergen Alert**: Note any allergens present (e.g., "Contains: dairy, eggs, tree nuts")

---

**Important**: Present ALL relevant recipes from the context. If multiple recipes match the query, show each one in the format above.
"""

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the recipe collection.
        
        Returns:
            Dict with collection statistics
        """
        all_data = self.collection.get(include=["metadatas"])
        
        # Count recipes and chunk types
        recipes = set()
        chunk_types = {}
        
        for metadata in all_data['metadatas']:
            recipe_name = metadata.get('recipe', 'Unknown')
            recipes.add(recipe_name)
            
            chunk_type = metadata.get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'total_chunks': len(all_data['ids']),
            'unique_recipes': len(recipes),
            'chunk_types': chunk_types,
            'avg_chunks_per_recipe': len(all_data['ids']) / len(recipes) if recipes else 0
        }


# Convenience function for quick searches
def quick_search(query: str, top_k: int = 5) -> dict:
    """
    Quick hybrid search without creating a class instance.
    
    Args:
        query: Search query
        top_k: Number of results
    
    Returns:
        Search results dict
    """
    searcher = HybridRecipeSearch()
    return searcher.hybrid_search(query, top_k=top_k)


def quick_ask(query: str, top_k: int = 10) -> str:
    """
    Quick LLM query without creating a class instance.
    
    Args:
        query: Question about recipes
        top_k: Number of context chunks
    
    Returns:
        Generated answer text
    """
    searcher = HybridRecipeSearch()
    result = searcher.search_and_generate(query, top_k=top_k)
    return result.get('answer', result.get('error', 'No response generated'))
