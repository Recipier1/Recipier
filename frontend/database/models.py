"""Data models for frontend storage."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SearchEntry:
    """Represents a search history entry."""
    id: Optional[int]
    query: str
    timestamp: datetime
    response_preview: str  # First 200 chars for sidebar display
    full_response: str     # Complete AI response
    tokens_used: Optional[int] = None

    @classmethod
    def from_row(cls, row: tuple) -> 'SearchEntry':
        """Create SearchEntry from database row."""
        return cls(
            id=row[0],
            query=row[1],
            timestamp=datetime.fromisoformat(row[2]) if isinstance(row[2], str) else row[2],
            response_preview=row[3] or '',
            full_response=row[4] or '',
            tokens_used=row[5]
        )

    @classmethod
    def create(cls, query: str, full_response: str, tokens_used: Optional[int] = None) -> 'SearchEntry':
        """Create a new SearchEntry from query and response."""
        preview = full_response[:200] + '...' if len(full_response) > 200 else full_response
        return cls(
            id=None,
            query=query,
            timestamp=datetime.now(),
            response_preview=preview,
            full_response=full_response,
            tokens_used=tokens_used
        )


@dataclass
class UserSettings:
    """User preferences storage."""
    theme: str = 'light'  # 'light' or 'dark'

    @classmethod
    def from_row(cls, row: tuple) -> 'UserSettings':
        """Create UserSettings from database row."""
        return cls(theme=row[1] if len(row) > 1 else 'light')
