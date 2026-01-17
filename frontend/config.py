"""Design tokens and configuration for Recipier frontend."""
from dataclasses import dataclass


@dataclass
class ThemeColors:
    """Color palette for the theme."""
    background: str
    surface: str
    text: str
    text_secondary: str
    accent: str
    accent_hover: str
    border: str
    shadow: str


# Dark Mode - Warm dark aesthetic
DARK_THEME = ThemeColors(
    background='#1C1917',        # Warm dark (stone-900)
    surface='#292524',           # Card surface (stone-800)
    text='#FAFAF9',             # Off-white (stone-50)
    text_secondary='#A8A29E',    # Gray (stone-400)
    accent='#F59E0B',            # Amber
    accent_hover='#FBBF24',      # Lighter amber
    border='#44403C',            # Border (stone-700)
    shadow='rgba(0, 0, 0, 0.3)'
)


# Typography
FONTS = {
    'family': "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif",
    'mono': "'SF Mono', 'Fira Code', Menlo, Monaco, monospace",
    'size_xs': '0.75rem',
    'size_sm': '0.875rem',
    'size_base': '1rem',
    'size_lg': '1.125rem',
    'size_xl': '1.25rem',
    'size_2xl': '1.5rem',
    'size_3xl': '1.875rem',
}

# Spacing
SPACING = {
    'xs': '0.25rem',
    'sm': '0.5rem',
    'md': '1rem',
    'lg': '1.5rem',
    'xl': '2rem',
    '2xl': '3rem',
}

# Border radius
RADIUS = {
    'sm': '0.375rem',
    'md': '0.5rem',
    'lg': '0.75rem',
    'xl': '1rem',
    'full': '9999px',
}

# App configuration
APP_CONFIG = {
    'title': 'Recipier',
    'icon': '🍳',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'page_title': 'Recipier - AI Recipe Search',
}
