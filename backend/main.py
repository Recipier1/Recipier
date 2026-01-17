from pdb import run
from scraper.WebScraper import WebScraper
from scraper.AllRecipesWebCrawler import AllRecipesCrawler, WebCrawler
from scraper.RecipeTransformer import RecipeTransformer
from .database import get_chromadb_client


def run_recipe_pipeline(seed_url, max_recipes=5):
    # init tools
    crawler = AllRecipesCrawler(delay=1.0)
    scraper = WebScraper()
    client = get_chromadb_client()
    # Delete existing collection if it exists
    try:
        client.delete_collection(name="recipes")
        print("🗑️  Deleted existing 'recipes' collection")
    except Exception as e:
        print(f"ℹ️  No existing collection to delete (or error: {e})")

    # Create fresh collection
    collection = client.get_or_create_collection(name="recipes")
    # client.delete_collection(name="recipes")

    # for url, info
    crawl_results = crawler.crawl(seed_url, max_pages=max_recipes)

    for url, info in crawl_results.items():
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
                f" Skipping {url} - possibly not a recipe page. Error: {e}")

    print("\n✨ Ingestion Complete! Your RAG database is ready.")
    results = collection.query(
        query_texts=["spinach chicken"],
        n_results=5,
        include=["metadatas", "documents", "distances"]
    )

    print("queried:")
    print(results)


if __name__ == "__main__":
    # Example Seed URL
    START_URL = "https://www.allrecipes.com/recipe/125658/chicken-enchiladas-with-creamy-green-chile-sauce/"
    run_recipe_pipeline(START_URL, max_recipes=20)
