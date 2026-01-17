"""Sidebar component with search history (Claude-style)."""
import streamlit as st
from datetime import datetime
from typing import Callable, Optional
from ..database import HistoryDB, SearchEntry


def render_sidebar(
    on_select: Callable[[int], None],
    on_clear: Optional[Callable[[], None]] = None,
    selected_id: Optional[int] = None
):
    """Render the sidebar with search history in Claude-style contiguous list.

    Args:
        on_select: Callback when a history item is clicked (receives search ID)
        on_clear: Callback when clear history is clicked
        selected_id: Currently selected history entry ID
    """
    db = HistoryDB()
    history = db.get_history(limit=50)

    if not history:
        st.markdown("""<div style="text-align: center; padding: 2rem 1rem; opacity: 0.6;">
    <p style="font-size: 0.875rem; color: #A8A29E;">No searches yet</p>
</div>""", unsafe_allow_html=True)
        return

    # Build the contiguous history list HTML
    html_items = []
    for entry in history:
        time_str = _format_time(entry.timestamp)
        display_query = entry.query[:50] + '...' if len(entry.query) > 50 else entry.query
        selected_class = 'selected' if entry.id == selected_id else ''

        html_items.append(f'''<div class="history-item {selected_class}" data-id="{entry.id}">
    <div class="history-query">{_escape_html(display_query)}</div>
    <div class="history-time">{time_str}</div>
</div>''')

    # Render as a single HTML block for contiguous styling
    # IMPORTANT: No leading spaces in the HTML string (markdown treats indented text as code)
    st.markdown(f'''<div class="history-container">
{''.join(html_items)}
</div>''', unsafe_allow_html=True)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def _format_time(timestamp: datetime) -> str:
    """Format timestamp for display.

    Args:
        timestamp: datetime to format

    Returns:
        Human-readable time string
    """
    now = datetime.now()
    diff = now - timestamp

    if diff.days == 0:
        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            mins = diff.seconds // 60
            return f"{mins}m ago"
        else:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
    elif diff.days == 1:
        return "Yesterday"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    else:
        return timestamp.strftime("%b %d")
