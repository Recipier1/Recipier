"""Database package for search history and settings."""
from .models import SearchEntry, UserSettings
from .history import HistoryDB

__all__ = ['SearchEntry', 'UserSettings', 'HistoryDB']
