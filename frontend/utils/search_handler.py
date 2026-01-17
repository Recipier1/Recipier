"""Backend integration for recipe search."""
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for backend import
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend import HybridRecipeSearch
from ..database import HistoryDB, SearchEntry


class SearchHandler:
    """Handles recipe search requests and caching."""

    def __init__(self):
        """Initialize search handler with backend and database."""
        self._searcher: Optional[HybridRecipeSearch] = None
        self.db = HistoryDB()

    @property
    def searcher(self) -> HybridRecipeSearch:
        """Lazy-load the search engine."""
        if self._searcher is None:
            self._searcher = HybridRecipeSearch()
        return self._searcher

    def search(self, query: str) -> dict:
        """Perform a recipe search and cache the result.

        Args:
            query: Natural language recipe query

        Returns:
            Dict with keys:
                - success: bool
                - answer: str (AI response) or None
                - error: str or None
                - tokens_used: int or None
                - entry_id: int (database ID for caching)
        """
        if not query or not query.strip():
            return {
                'success': False,
                'answer': None,
                'error': 'Please enter a search query',
                'tokens_used': None,
                'entry_id': None
            }

        query = query.strip()

        try:
            # Call backend
            result = self.searcher.search_and_generate(query)

            if result.get('error'):
                return {
                    'success': False,
                    'answer': None,
                    'error': result['error'],
                    'tokens_used': None,
                    'entry_id': None
                }

            answer = result.get('answer', '')
            tokens_used = result.get('tokens_used')

            # Save to history
            entry = SearchEntry.create(
                query=query,
                full_response=answer,
                tokens_used=tokens_used
            )
            entry_id = self.db.add_search(entry)

            return {
                'success': True,
                'answer': answer,
                'error': None,
                'tokens_used': tokens_used,
                'entry_id': entry_id
            }

        except Exception as e:
            return {
                'success': False,
                'answer': None,
                'error': f'Search failed: {str(e)}',
                'tokens_used': None,
                'entry_id': None
            }

    def get_cached_response(self, search_id: int) -> Optional[str]:
        """Get a cached response from history.

        Args:
            search_id: Database ID of the search entry

        Returns:
            Cached response text or None
        """
        entry = self.db.get_search_by_id(search_id)
        return entry.full_response if entry else None
