"""Session state management for Streamlit."""
import streamlit as st
from typing import Optional, Any
from ..database import HistoryDB


class StateManager:
    """Manages Streamlit session state for the app."""

    # State keys
    CURRENT_QUERY = 'current_query'
    CURRENT_RESPONSE = 'current_response'
    SELECTED_HISTORY_ID = 'selected_history_id'
    IS_LOADING = 'is_loading'
    ERROR_MESSAGE = 'error_message'

    @classmethod
    def initialize(cls):
        """Initialize all session state variables with defaults."""
        defaults = {
            cls.CURRENT_QUERY: '',
            cls.CURRENT_RESPONSE: None,
            cls.SELECTED_HISTORY_ID: None,
            cls.IS_LOADING: False,
            cls.ERROR_MESSAGE: None,
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a session state value."""
        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any):
        """Set a session state value."""
        st.session_state[key] = value

    @classmethod
    def get_current_query(cls) -> str:
        """Get current search query."""
        return cls.get(cls.CURRENT_QUERY, '')

    @classmethod
    def set_current_query(cls, query: str):
        """Set current search query."""
        cls.set(cls.CURRENT_QUERY, query)

    @classmethod
    def get_current_response(cls) -> Optional[str]:
        """Get current response text."""
        return cls.get(cls.CURRENT_RESPONSE)

    @classmethod
    def set_current_response(cls, response: Optional[str]):
        """Set current response text."""
        cls.set(cls.CURRENT_RESPONSE, response)

    @classmethod
    def is_loading(cls) -> bool:
        """Check if a search is in progress."""
        return cls.get(cls.IS_LOADING, False)

    @classmethod
    def set_loading(cls, loading: bool):
        """Set loading state."""
        cls.set(cls.IS_LOADING, loading)

    @classmethod
    def get_error(cls) -> Optional[str]:
        """Get current error message."""
        return cls.get(cls.ERROR_MESSAGE)

    @classmethod
    def set_error(cls, message: Optional[str]):
        """Set error message."""
        cls.set(cls.ERROR_MESSAGE, message)

    @classmethod
    def clear_error(cls):
        """Clear error message."""
        cls.set(cls.ERROR_MESSAGE, None)

    @classmethod
    def select_history(cls, search_id: int):
        """Select a history entry and load its cached response."""
        db = HistoryDB()
        entry = db.get_search_by_id(search_id)
        if entry:
            cls.set(cls.SELECTED_HISTORY_ID, search_id)
            cls.set(cls.CURRENT_QUERY, entry.query)
            cls.set(cls.CURRENT_RESPONSE, entry.full_response)
            cls.clear_error()

    @classmethod
    def clear_selection(cls):
        """Clear current selection and response."""
        cls.set(cls.SELECTED_HISTORY_ID, None)
        cls.set(cls.CURRENT_QUERY, '')
        cls.set(cls.CURRENT_RESPONSE, None)
        cls.clear_error()
