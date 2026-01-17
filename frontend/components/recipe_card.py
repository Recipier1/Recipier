"""Recipe card component for displaying AI responses."""
import streamlit as st
from typing import Optional


def render_recipe_card(
    response: Optional[str],
    query: Optional[str] = None,
    is_loading: bool = False,
    error: Optional[str] = None
):
    """Render the recipe response card.

    Args:
        response: AI-generated response text (markdown)
        query: The query that generated this response
        is_loading: Whether a search is in progress
        error: Error message to display, if any
    """
    # Use a container with a key to ensure it replaces previous content
    container = st.container()

    with container:
        if is_loading:
            _render_loading_state()
            return

        if error:
            _render_error_state(error)
            return

        if not response:
            _render_empty_state()
            return

        _render_response(response, query)


def _render_loading_state():
    """Show cooking animation loader."""
    st.markdown("""<div class="cooking-loader">
    <div class="cooking-animation">🍳</div>
    <div class="cooking-text">
        Chef AI is preparing your recipes<span class="cooking-dots"></span>
    </div>
</div>""", unsafe_allow_html=True)


def _render_error_state(error: str):
    """Show error message."""
    st.error(f"**Oops!** {error}")
    st.markdown("""<div class="empty-state">
    <div class="empty-state-icon">😕</div>
    <p>Something went wrong. Please try again.</p>
</div>""", unsafe_allow_html=True)


def _render_empty_state():
    """Show empty state when no search has been performed."""
    st.markdown("""<div class="empty-state">
    <div class="empty-state-icon">🍽️</div>
    <h3 style="margin-bottom: 0.5rem; color: #FAFAF9;">What would you like to cook?</h3>
    <p>Search for recipes by ingredients, cuisine, or dish name.</p>
    <p style="font-size: 0.875rem; opacity: 0.7;">
        Try: "quick weeknight pasta" or "healthy chicken recipes"
    </p>
</div>""", unsafe_allow_html=True)


def _render_response(response: str, query: Optional[str] = None):
    """Render the AI response with proper formatting."""
    # Show the query as context
    if query:
        st.markdown(f"""<div style="margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #44403C;">
    <span style="opacity: 0.6; font-size: 0.875rem; color: #A8A29E;">Results for:</span>
    <span style="font-weight: 600; margin-left: 0.5rem; color: #FAFAF9;">"{query}"</span>
</div>""", unsafe_allow_html=True)

    # Render the markdown response directly (no wrapper div that causes stacking)
    st.markdown(response)
