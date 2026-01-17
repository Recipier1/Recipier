"""Search bar component."""
import streamlit as st
from typing import Callable, Optional


def render_search_bar(
    on_search: Callable[[str], None],
    default_value: str = '',
    placeholder: str = 'Search for recipes... (e.g., "quick chicken dinner" or "vegan desserts")'
) -> Optional[str]:
    """Render the main search bar.

    Args:
        on_search: Callback function when search is submitted
        default_value: Default value for the input
        placeholder: Placeholder text

    Returns:
        The submitted query if form was submitted, None otherwise
    """
    # Use a form for proper enter-key submission
    with st.form(key='search_form', clear_on_submit=False):
        col1, col2 = st.columns([6, 1])

        with col1:
            query = st.text_input(
                label='Search',
                value=default_value,
                placeholder=placeholder,
                label_visibility='collapsed',
                key='search_input'
            )

        with col2:
            submitted = st.form_submit_button(
                label='Search',
                use_container_width=True,
                type='primary'
            )

    if submitted and query:
        on_search(query)
        return query

    return None
