"""
Hybrid search implementation using semantic search (ChromaDB) + keyword search (BM25).
Optimized for large collections (2000+ recipes, ~20,000 chunks).

Supports multimodal search with ImageBind embeddings (text, image, video).
"""
import time
import os
from typing import Optional

import numpy as np
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
    - Multimodal search with ImageBind (text, image, video queries)
    """

    def __init__(
        self,
        collection_name: str = "recipes",
        use_imagebind: bool = False
    ):
        """
        Initialize hybrid search engine.

        Args:
            collection_name: Name of ChromaDB collection to search
            use_imagebind: If True, use ImageBind for embeddings (enables image/video search)
        """
        self.client = get_chromadb_client()
        self.use_imagebind = use_imagebind
        self.embedder = None
        
        # Initialize ImageBind if requested
        if use_imagebind:
            try:
                from .imagebind_embeddings import ImageBindEmbedder, ImageBindEmbeddingFunction
                self.embedder = ImageBindEmbedder()
                embedding_fn = ImageBindEmbeddingFunction(self.embedder)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_fn
                )
                print(f"✅ Using ImageBind embeddings for collection '{collection_name}'")
            except ImportError as e:
                print(f"⚠️  ImageBind not available: {e}")
                print("   Falling back to default embeddings.")
                self.use_imagebind = False
                self.collection = self.client.get_or_create_collection(
                    name=collection_name)
        else:
            self.collection = self.client.get_or_create_collection(
                name=collection_name)

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
            tokenized_docs = [doc.lower().split()
                              for doc in all_docs['documents']]
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_doc_ids = all_docs['ids']
            self.index_timestamp = now

            elapsed = time.time() - start_time
            print(
                f"✅ Built BM25 index for {len(tokenized_docs)} documents in {elapsed:.2f}s")

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

        print(
            f"🔍 Semantic search: retrieving top {candidate_pool} candidates...")
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

        # This step will create a frequency table to store the TF (term frequency). 
        # It also calculates how "rare" each word is. Frequent terms like "and", "or" etc will be penalised heavier, while less frequent terms will be given heavier weightage (IDF - Inverse Document Frequency)
        # Average document length is also stored to ensure that long wordy, documents don't have an unfair advantage over short, concise ones just because it has more words
        mini_bm25 = BM25Okapi(tokenized_candidates)
        
        # score individual query words for each document and return a list of scores
        bm25_scores = mini_bm25.get_scores(tokenized_query)

        # Step 3: Combine scores with weighted fusion
        semantic_distances = semantic_results['distances'][0]
        max_distance = max(semantic_distances) if semantic_distances else 1
        max_bm25 = max(bm25_scores) if len(
            bm25_scores) > 0 and max(bm25_scores) > 0 else 1

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
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
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

    def search_and_generate_stream(
        self,
        query: str,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Perform hybrid search and generate answer using LLM with streaming.

        Args:
            query: User's recipe query
            top_k: Number of context chunks to retrieve
            model: OpenAI model to use (gpt-4o-mini is cost-effective)
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens in response

        Yields:
            Dict with keys:
                - type: "chunk" | "sources" | "error" | "done"
                - content: Text chunk (for type="chunk")
                - sources: Retrieved chunks (for type="sources")
                - context_used: Formatted context (for type="sources")
                - error: Error message (for type="error")
                - model: Model name (for type="done")
                - tokens_used: Total tokens (for type="done")

        Example:
            >>> searcher = HybridRecipeSearch()
            >>> for event in searcher.search_and_generate_stream("How do I make chicken alfredo?"):
            ...     if event['type'] == 'chunk':
            ...         print(event['content'], end='', flush=True)
        """
        if not self.openai_client:
            yield {
                "type": "error",
                "error": "OpenAI API key not configured"
            }
            return

        # 1. Hybrid search to get relevant chunks
        print(f"🔍 Searching for: {query}")
        search_results = self.hybrid_search(query, top_k=top_k)

        # 2. Format context from retrieved chunks
        context = self._format_context(search_results)

        # Yield sources immediately
        yield {
            "type": "sources",
            "sources": search_results,
            "context_used": context
        }

        # 3. Create prompt for LLM
        prompt = self._create_prompt(query, context)

        # 4. Call OpenAI API with streaming
        print(f"🤖 Generating response with {model}...")
        try:
            stream = self.openai_client.chat.completions.create(
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
                max_tokens=max_tokens,
                stream=True
            )

            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield {
                        "type": "chunk",
                        "content": content
                    }

            # Yield completion event
            yield {
                "type": "done",
                "model": model,
                "answer": full_answer
            }

        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            yield {
                "type": "error",
                "error": str(e)
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
            type_order = {'title': 0, 'ingredients': 1,
                          'directions': 2, 'nutrition': 3}
            chunks.sort(key=lambda x: type_order.get(x['type'], 99))

            for chunk in chunks:
                chunk_type = chunk['type'].upper()
                text = chunk['text']
                score = chunk['score']
                context_parts.append(
                    f"\n[{chunk_type}] (relevance: {score:.3f})")
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

    # ==================== ImageBind Multimodal Search Methods ====================

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> dict:
        """
        Search recipes using an image query.
        
        Requires ImageBind to be enabled (use_imagebind=True in constructor).
        
        Args:
            image_path: Path to query image file
            top_k: Number of results to return
        
        Returns:
            Dict with keys: ids, documents, metadatas, distances
            
        Example:
            >>> searcher = HybridRecipeSearch(collection_name="recipes_imagebind", use_imagebind=True)
            >>> results = searcher.search_by_image("pasta_photo.jpg", top_k=5)
        """
        if not self.use_imagebind or self.embedder is None:
            raise ValueError(
                "ImageBind not enabled. Initialize with use_imagebind=True "
                "and ensure you're using an ImageBind-embedded collection."
            )
        
        print(f"🖼️  Searching by image: {image_path}")
        
        # Generate image embedding
        image_embedding = self.embedder.embed_image([image_path])[0]
        
        # Query ChromaDB with the image embedding
        results = self.collection.query(
            query_embeddings=[image_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"✅ Found {len(results['ids'][0])} results")
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }

    def search_by_video(
        self,
        video_path: str,
        top_k: int = 10,
    ) -> dict:
        """
        Search recipes using a video query.
        
        Requires ImageBind to be enabled (use_imagebind=True in constructor).
        
        Args:
            video_path: Path to query video file
            top_k: Number of results to return
        
        Returns:
            Dict with keys: ids, documents, metadatas, distances
            
        Example:
            >>> searcher = HybridRecipeSearch(collection_name="recipes_imagebind", use_imagebind=True)
            >>> results = searcher.search_by_video("cooking_video.mp4", top_k=5)
        """
        if not self.use_imagebind or self.embedder is None:
            raise ValueError(
                "ImageBind not enabled. Initialize with use_imagebind=True "
                "and ensure you're using an ImageBind-embedded collection."
            )
        
        print(f"🎬 Searching by video: {video_path}")
        
        # Generate video embedding
        video_embedding = self.embedder.embed_video([video_path])[0]
        
        # Query ChromaDB with the video embedding
        results = self.collection.query(
            query_embeddings=[video_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"✅ Found {len(results['ids'][0])} results")
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }

    def multimodal_search(
        self,
        query_text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        top_k: int = 10,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        video_weight: float = 0.5,
    ) -> dict:
        """
        Combined multimodal search using text, image, and/or video queries.
        
        Embeddings from different modalities are combined using weighted averaging.
        Since ImageBind maps all modalities to the same vector space, this allows
        for powerful cross-modal search.
        
        Args:
            query_text: Optional text query
            image_path: Optional path to query image
            video_path: Optional path to query video
            top_k: Number of results to return
            text_weight: Weight for text embedding (0-1)
            image_weight: Weight for image embedding (0-1)
            video_weight: Weight for video embedding (0-1)
        
        Returns:
            Dict with keys: ids, documents, metadatas, distances
            
        Example:
            >>> searcher = HybridRecipeSearch(collection_name="recipes_imagebind", use_imagebind=True)
            >>> # Search with both text and image
            >>> results = searcher.multimodal_search(
            ...     query_text="pasta dish",
            ...     image_path="tomato_sauce.jpg",
            ...     text_weight=0.3,
            ...     image_weight=0.7
            ... )
        """
        if not self.use_imagebind or self.embedder is None:
            raise ValueError(
                "ImageBind not enabled. Initialize with use_imagebind=True "
                "and ensure you're using an ImageBind-embedded collection."
            )
        
        if not any([query_text, image_path, video_path]):
            raise ValueError("At least one of query_text, image_path, or video_path must be provided")
        
        embeddings = []
        weights = []
        modalities_used = []
        
        if query_text:
            text_emb = self.embedder.embed_text([query_text])[0]
            embeddings.append(text_emb)
            weights.append(text_weight)
            modalities_used.append("text")
        
        if image_path:
            img_emb = self.embedder.embed_image([image_path])[0]
            embeddings.append(img_emb)
            weights.append(image_weight)
            modalities_used.append("image")
        
        if video_path:
            vid_emb = self.embedder.embed_video([video_path])[0]
            embeddings.append(vid_emb)
            weights.append(video_weight)
            modalities_used.append("video")
        
        print(f"🔀 Multimodal search using: {', '.join(modalities_used)}")
        
        # Weighted average of embeddings (they're in the same space!)
        combined = np.average(embeddings, axis=0, weights=weights)
        # Normalize to unit length
        combined = combined / np.linalg.norm(combined)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[combined.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"✅ Found {len(results['ids'][0])} results")
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }

    def search_by_image_and_generate(
        self,
        image_path: str,
        query_text: Optional[str] = None,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> dict:
        """
        Search by image and generate a response using LLM.
        
        Args:
            image_path: Path to query image
            query_text: Optional additional text query/context
            top_k: Number of context chunks to retrieve
            model: OpenAI model to use
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with answer, sources, context_used
        """
        if not self.openai_client:
            return {
                "error": "OpenAI API key not configured",
                "answer": None,
                "sources": None,
                "context_used": None
            }
        
        # Search by image (or multimodal if text also provided)
        if query_text:
            search_results = self.multimodal_search(
                query_text=query_text,
                image_path=image_path,
                top_k=top_k,
                text_weight=0.3,
                image_weight=0.7
            )
            user_query = f"Based on this image and the query '{query_text}'"
        else:
            search_results = self.search_by_image(image_path, top_k=top_k)
            user_query = "Based on this food image, suggest relevant recipes"
        
        # Convert distances to scores for context formatting
        if 'distances' in search_results and 'scores' not in search_results:
            max_dist = max(search_results['distances']) if search_results['distances'] else 1
            search_results['scores'] = [
                1 - (d / max_dist) for d in search_results['distances']
            ]
        
        # Format context and generate response
        context = self._format_context(search_results)
        prompt = self._create_prompt(user_query, context)
        
        print(f"🤖 Generating response with {model}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful cooking assistant that provides recipe advice based on the given context. The user has uploaded an image of food, and you should suggest relevant recipes from the context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "answer": response.choices[0].message.content,
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


# ==================== ImageBind Convenience Functions ====================

def quick_image_search(
    image_path: str,
    top_k: int = 5,
    collection_name: str = "recipes_imagebind"
) -> dict:
    """
    Quick image search using ImageBind embeddings.

    Args:
        image_path: Path to query image
        top_k: Number of results
        collection_name: Name of ImageBind-embedded collection

    Returns:
        Search results dict
    """
    searcher = HybridRecipeSearch(
        collection_name=collection_name,
        use_imagebind=True
    )
    return searcher.search_by_image(image_path, top_k=top_k)


def quick_multimodal_search(
    query_text: str = None,
    image_path: str = None,
    top_k: int = 5,
    collection_name: str = "recipes_imagebind"
) -> dict:
    """
    Quick multimodal search combining text and/or image.

    Args:
        query_text: Optional text query
        image_path: Optional image path
        top_k: Number of results
        collection_name: Name of ImageBind-embedded collection

    Returns:
        Search results dict
    """
    searcher = HybridRecipeSearch(
        collection_name=collection_name,
        use_imagebind=True
    )
    return searcher.multimodal_search(
        query_text=query_text,
        image_path=image_path,
        top_k=top_k
    )
