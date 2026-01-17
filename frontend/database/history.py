"""SQLite database operations for search history and settings."""
import sqlite3
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from .models import SearchEntry, UserSettings


class HistoryDB:
    """SQLite database for search history and user settings."""

    def __init__(self, db_path: str = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database. Defaults to data/recipier.db
        """
        if db_path is None:
            # Default to data/recipier.db relative to project root
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / 'data' / 'recipier.db'

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Search history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    response_preview TEXT,
                    full_response TEXT,
                    tokens_used INTEGER
                )
            ''')

            # User settings table (single row)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    theme TEXT DEFAULT 'light' CHECK (theme IN ('light', 'dark'))
                )
            ''')

            # Ensure settings row exists
            cursor.execute('''
                INSERT OR IGNORE INTO user_settings (id, theme) VALUES (1, 'light')
            ''')

    # --- Search History Operations ---

    def add_search(self, entry: SearchEntry) -> int:
        """Add a search entry to history.

        Args:
            entry: SearchEntry to save

        Returns:
            ID of the inserted row
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_history (query, timestamp, response_preview, full_response, tokens_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                entry.query,
                entry.timestamp.isoformat(),
                entry.response_preview,
                entry.full_response,
                entry.tokens_used
            ))
            return cursor.lastrowid

    def get_history(self, limit: int = 50) -> List[SearchEntry]:
        """Get recent search history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of SearchEntry objects, most recent first
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, query, timestamp, response_preview, full_response, tokens_used
                FROM search_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            return [SearchEntry.from_row(tuple(row)) for row in cursor.fetchall()]

    def get_search_by_id(self, search_id: int) -> Optional[SearchEntry]:
        """Get a specific search entry by ID.

        Args:
            search_id: ID of the search entry

        Returns:
            SearchEntry or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, query, timestamp, response_preview, full_response, tokens_used
                FROM search_history
                WHERE id = ?
            ''', (search_id,))

            row = cursor.fetchone()
            return SearchEntry.from_row(tuple(row)) if row else None

    def delete_search(self, search_id: int) -> bool:
        """Delete a search entry.

        Args:
            search_id: ID of the search entry to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM search_history WHERE id = ?', (search_id,))
            return cursor.rowcount > 0

    def clear_history(self):
        """Delete all search history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM search_history')

    # --- User Settings Operations ---

    def get_settings(self) -> UserSettings:
        """Get user settings.

        Returns:
            UserSettings object
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, theme FROM user_settings WHERE id = 1')
            row = cursor.fetchone()
            return UserSettings.from_row(tuple(row)) if row else UserSettings()

    def set_theme(self, theme: str):
        """Set the UI theme.

        Args:
            theme: 'light' or 'dark'
        """
        if theme not in ('light', 'dark'):
            raise ValueError("Theme must be 'light' or 'dark'")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_settings (id, theme) VALUES (1, ?)
            ''', (theme,))
