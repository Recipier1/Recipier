from scraper.WebScraper import WebScraper
from scraper.AllRecipesWebCrawler import AllRecipesCrawler
from scraper.RecipeTransformer import RecipeTransformer
from .database import get_chromadb_client
from backend.search import HybridRecipeSearch


def run_recipe_pipeline(seed_url, max_recipes=5, debug=False):
    # init tools
    crawler = AllRecipesCrawler(delay=0.8)
    scraper = WebScraper()
    client = get_chromadb_client()

    # Create fresh collection
    collection = client.get_or_create_collection(name="recipes")

    # Determine if seed URL is a recipe page (leaf node)
    is_recipe_page = '/recipe/' in seed_url.lower() or '-recipe-' in seed_url.lower()

    # Set fallback URL to a category page if starting from a recipe page
    fallback_url = None
    if is_recipe_page:
        fallback_url = "https://www.allrecipes.com/recipes/"
        print(
            f"📌 Starting from recipe page. Fallback URL set to: {fallback_url}")

    # Crawl pages (includes both recipes and category pages)
    # But we want to scrape MORE than max_recipes pages to find enough recipes
    crawl_results = crawler.crawl(seed_url, max_pages=max_recipes)

    # Filter to only actual recipe pages (not category pages)
    recipe_urls = {url: info for url, info in crawl_results.items()
                   if '/recipe/' in url.lower()}

    print(
        f"\n🎯 Found {len(recipe_urls)} recipe pages out of {len(crawl_results)} crawled pages")

    # Limit to max_recipes
    recipe_urls = dict(list(recipe_urls.items())[:max_recipes])

    for url, info in recipe_urls.items():
        print(f"📖 Scraping recipe: {info['title']}")

        # Get the BeautifulSoup object for the specific recipe
        soup = scraper.get_data(url)

        # Extract the structured recipe data (the list of dicts)
        try:
            raw_recipe_data = scraper.extract_data(soup)

            # Transform the data for ChromaDB
            transformer = RecipeTransformer(raw_recipe_data)
            chroma_data = transformer.transform_for_chroma()

            # 4. Step 4: Load into ChromaDB
            collection.add(
                documents=chroma_data["documents"],
                metadatas=chroma_data["metadatas"],
                ids=chroma_data["ids"]
            )
            print(
                f"✅ Indexed {len(chroma_data['ids'])} chunks for {info['title']}")

        except Exception as e:
            print(
                f"⚠️  Skipping {url} - possibly not a recipe page. Error: {e}")

    print("\n✨ Ingestion Complete! Your RAG database is ready.")
    # results = collection.query(
    #     query_texts=["spinach chicken"],
    #     n_results=5,
    #     include=["metadatas", "documents", "distances"]
    # )

    searcher = HybridRecipeSearch()

    # Test search
    query = "seaweed"
    results = searcher.hybrid_search(query, top_k=3)

    # Display results
    for i, doc in enumerate(results['documents'], 1):
        metadata = results['metadatas'][i-1]
        score = results['scores'][i-1]
        recipe = metadata.get('recipe', 'Unknown')
        chunk_type = metadata.get('type', 'unknown')

        print(f"\n  {i}. [{chunk_type}] {recipe}")
        print(f"     Score: {score:.4f}")
        print(f"     Preview: {doc[:80]}...")

    print("queried:")
    print(results)

    print(searcher.search_and_generate(
        query="",
    ))


if __name__ == "__main__":
    START_URL = "https://www.allrecipes.com/recipes/698/world-cuisine/asian/indonesian/"

    # Enable debug mode to see link discovery details
    run_recipe_pipeline(START_URL, max_recipes=1000)
