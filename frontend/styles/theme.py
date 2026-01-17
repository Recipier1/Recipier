"""CSS theme generator for Streamlit styling (Dark mode only)."""
from ..config import DARK_THEME, FONTS, SPACING, RADIUS


def get_theme_css() -> str:
    """Generate complete CSS for dark theme.

    Returns:
        CSS string to inject into Streamlit
    """
    colors = DARK_THEME

    return f"""
    <style>
    /* ===== Global Styles ===== */
    .stApp {{
        background-color: {colors.background};
        font-family: {FONTS['family']};
    }}

    /* Hide Streamlit branding and form hints */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Hide "Press Enter to submit form" */
    .stForm [data-testid="stFormSubmitButton"] + div,
    .stForm > div:last-child > div:has(small),
    div[data-testid="InputInstructions"],
    .stTextInput small,
    [data-testid="InputInstructions"] {{
        display: none !important;
    }}

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {{
        background-color: {colors.surface};
        border-right: 1px solid {colors.border};
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: {colors.text};
    }}

    /* Hide all buttons in sidebar (only show HTML history) */
    [data-testid="stSidebar"] .stButton {{
        display: none !important;
    }}

    /* ===== Headers ===== */
    h1, h2, h3, h4, h5, h6 {{
        color: {colors.text} !important;
        font-family: {FONTS['family']};
    }}

    h1 {{
        font-size: {FONTS['size_3xl']} !important;
        font-weight: 700 !important;
    }}

    /* ===== Text ===== */
    p, span, label, .stMarkdown {{
        color: {colors.text};
    }}

    /* ===== Search Input ===== */
    .stTextInput > div > div > input {{
        background-color: {colors.surface};
        color: {colors.text};
        border: 2px solid {colors.border};
        border-radius: {RADIUS['lg']};
        padding: {SPACING['md']} {SPACING['lg']};
        font-size: {FONTS['size_lg']};
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {colors.accent};
        box-shadow: 0 0 0 3px {colors.accent}33;
        outline: none;
    }}

    .stTextInput > div > div > input::placeholder {{
        color: {colors.text_secondary};
    }}

    /* ===== Buttons ===== */
    .stButton > button {{
        background-color: {colors.accent};
        color: white;
        border: none;
        border-radius: {RADIUS['md']};
        padding: {SPACING['sm']} {SPACING['lg']};
        font-weight: 600;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }}

    .stButton > button:hover {{
        background-color: {colors.accent_hover};
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ===== History List (Claude-style) ===== */
    .history-container {{
        display: flex;
        flex-direction: column;
        gap: 2px;
    }}

    .history-item {{
        padding: 10px 12px;
        border-radius: {RADIUS['md']};
        cursor: pointer;
        transition: background-color 0.15s ease;
        border: none;
        background: transparent;
    }}

    .history-item:hover {{
        background-color: {colors.background};
    }}

    .history-item.selected {{
        background-color: {colors.background};
    }}

    .history-query {{
        color: {colors.text};
        font-size: {FONTS['size_sm']};
        line-height: 1.4;
        margin: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }}

    .history-time {{
        color: {colors.text_secondary};
        font-size: {FONTS['size_xs']};
        margin-top: 4px;
    }}

    /* Hide default button styling in sidebar history */
    [data-testid="stSidebar"] .history-btn {{
        all: unset;
        cursor: pointer;
        width: 100%;
        display: block;
    }}

    /* ===== Cooking Loader ===== */
    .cooking-loader {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        text-align: center;
    }}

    .cooking-animation {{
        font-size: 4rem;
        animation: cooking 1.5s ease-in-out infinite;
    }}

    @keyframes cooking {{
        0%, 100% {{ transform: rotate(-10deg) scale(1); }}
        25% {{ transform: rotate(10deg) scale(1.1); }}
        50% {{ transform: rotate(-5deg) scale(1); }}
        75% {{ transform: rotate(5deg) scale(1.05); }}
    }}

    .cooking-text {{
        margin-top: 1rem;
        color: {colors.text_secondary};
        font-size: {FONTS['size_base']};
    }}

    .cooking-dots::after {{
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
    }}

    @keyframes dots {{
        0% {{ content: ''; }}
        25% {{ content: '.'; }}
        50% {{ content: '..'; }}
        75% {{ content: '...'; }}
        100% {{ content: ''; }}
    }}

    /* ===== Recipe Card ===== */
    .recipe-card {{
        background-color: {colors.surface};
        border-radius: {RADIUS['xl']};
        padding: {SPACING['xl']};
        margin: {SPACING['md']} 0;
        box-shadow: 0 4px 6px -1px {colors.shadow}, 0 2px 4px -2px {colors.shadow};
        border: 1px solid {colors.border};
    }}

    /* ===== Empty State ===== */
    .empty-state {{
        text-align: center;
        padding: {SPACING['2xl']};
        color: {colors.text_secondary};
    }}

    .empty-state-icon {{
        font-size: 4rem;
        margin-bottom: {SPACING['md']};
        opacity: 0.5;
    }}

    /* ===== Recipe Content Styling ===== */
    .recipe-content h2 {{
        color: {colors.accent} !important;
        border-bottom: 2px solid {colors.border};
        padding-bottom: {SPACING['sm']};
        margin-top: {SPACING['lg']};
    }}

    .recipe-content h3 {{
        color: {colors.text} !important;
        margin-top: {SPACING['md']};
    }}

    .recipe-content ul, .recipe-content ol {{
        color: {colors.text};
        padding-left: {SPACING['lg']};
    }}

    .recipe-content li {{
        margin-bottom: {SPACING['xs']};
    }}

    .recipe-content strong {{
        color: {colors.accent};
    }}

    /* ===== Divider ===== */
    hr {{
        border-color: {colors.border};
    }}

    /* ===== Toast/Notifications ===== */
    .stAlert {{
        border-radius: {RADIUS['md']};
    }}

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {colors.background};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {colors.border};
        border-radius: {RADIUS['full']};
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {colors.text_secondary};
    }}
    </style>
    """
