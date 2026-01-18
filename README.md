# Recipier

Your AI-powered "Chef in Your Pocket"

## Inspiration

Finding the right recipe online can be overwhelming. We often have to sift through lengthy blog posts, compare multiple sources, and piece together cooking instructions. For students living overseas, this pain point is amplified when we are unfamiliar with local ingredients.

We wanted to create an AI-powered **"Chef in Your Pocket"** that understands natural language queries like:

- "What's a good weeknight dinner with chicken?"
- "What can I make with zucchini and eggplant?"

Recipier aims to provide home cooks with context-aware, intelligent culinary guidance through natural language understanding and AI-powered recommendations.

---

## What It Does

**Recipier** is an intelligent recipe search engine that combines web scraping, vector databases, and AI to deliver personalised cooking guidance.

### Key Features

- **Natural Language Search**  
  Ask questions in plain English such as:
    - "easy Indonesian recipes"
    - "what can I make with spinach and chicken?"

- **Hybrid Search Engine**  
  Combines semantic search using **ChromaDB** with keyword matching using **BM25** to retrieve the most relevant recipes from a database of over 10,000 scraped recipes.

- **AI Chef Assistant**  
  Powered by **GPT-4o-mini**, providing:
    - Step-by-step cooking instructions
    - Pro tips
    - Ingredient substitutions
    - Allergen alerts

- **Smart Web Crawler**  
  Automatically scrapes and indexes recipes from cooking websites while:
    - Respecting `robots.txt`
    - Tracking seen URLs to avoid duplicates

- **Streaming Responses**  
  Real-time AI responses that appear as they are generated, creating a conversational experience.

- **Search History**  
  Saves past queries so users can easily revisit recipes.

---

## How We Built It

### Backend Architecture

- **Web Scraping**  
  Custom Python crawler using `BeautifulSoup` and `requests`, with specialised parsers for AllRecipes.com.

- **Vector Database**  
  ChromaDB for semantic search using embeddings, storing approximately 100,000 recipe chunks across ingredients, directions, nutrition information and metadata.

- **Hybrid Search**  
  Combined semantic similarity (vector search) with BM25 keyword scoring for optimal relevance.

- **RAG Pipeline**  
  Retrieval-Augmented Generation using OpenAI’s GPT-4o-mini to generate contextual responses from retrieved recipes.

---

### Frontend

- **Streamlit**  
  Responsive web interface with dark theme styling.

- **State Management**  
  Custom state manager for handling search history, loading states, and cached responses.

- **Database**  
  SQLite for persisting search history and user sessions.

---

### Key Optimisations

- BM25 index caching with smart refresh logic for large collections
- Seen URL tracking to avoid re-scraping recipes
- Streaming API responses for improved user experience
- Chunked recipe storage (ingredients, directions, nutrition) for precise retrieval

---

## Challenges We Ran Into

- **Search Quality**  
  Pure semantic search sometimes missed keyword-specific queries (e.g., searching for "Indonesian recipes").  
  This was solved by implementing hybrid search combining semantic understanding with keyword matching.

- **Scalability**  
  With over 2,000 recipes (approximately 20,000 chunks), BM25 indexing became slow.  
  Smart caching was implemented to rebuild the index only when necessary, reducing latency from seconds to milliseconds.

- **Web Scraping Ethics**  
  Ensuring respect for `robots.txt` and implementing proper delays between requests while still gathering sufficient data.  
  A robust crawler with configurable delays and robots.txt parsing was built.

- **Context Management**  
  Formatting retrieved recipe chunks for coherent, structured LLM responses required careful design.  
  A custom context formatter was developed to group chunks by recipe and content type.

- **Streaming UX**  
  Making real-time streaming responses work smoothly with Streamlit’s state management without race conditions or flickering.

---

## Accomplishments We’re Proud Of

- Successfully implemented a production-grade hybrid search system that outperforms simple vector search.
- Built a scalable architecture capable of handling over 10,000 recipes efficiently.
- Delivered genuinely helpful natural language cooking guidance with pro tips, not just regurgitated recipe text.
- Maintained a clean, extensible codebase with clear separation of concerns across scraper, backend, and frontend.
- Created a polished, production-ready user experience with streaming responses and search history.

---

## What We Learned

- **RAG is more than vector search**  
  Combining semantic and keyword-based retrieval significantly improves search quality.

- **Prompt engineering matters**  
  A well-crafted system prompt transformed generic responses into professional chef-level guidance.

- **Optimisation at scale**  
  Caching strategies and smart indexing are essential when working with thousands of documents.

- **User experience matters**  
  Streaming responses and proper loading states greatly improve perceived performance.

- **Web scraping best practices**  
  Respecting robots.txt, handling errors gracefully, and implementing request delays are critical for ethical scraping.

---

## What’s Next for Recipier

- Multi-source scraping beyond AllRecipes, including Food Network, Bon Appétit, and international recipe sites
- Image generation using AI or scraped photos to improve visual appeal
- User preferences for dietary restrictions, allergies, and cuisine types
- Recipe collections for saving favourites and creating custom lists
- Mobile app development using React Native
- Community features such as ratings, reviews, and shared variations
- Voice interface for hands-free cooking assistance
- Camera integration using computer vision (e.g. OpenCV) to recognise ingredients via user cameras
