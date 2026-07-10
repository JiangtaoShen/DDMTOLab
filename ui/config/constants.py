"""UI constants, colors, and sizes."""

# Window dimensions
WINDOW_WIDTH = 1550
WINDOW_HEIGHT = 900
MIN_WINDOW_WIDTH = 1400
MIN_WINDOW_HEIGHT = 800

# Panel widths
LEFT_PANEL_WIDTH = 280
MIDDLE_PANEL_WIDTH = 300

# Refined palette (claude.ai/design "DDMTOLab GUI (Refined)")
# -- dark chrome
COLOR_APP_BG = (19, 22, 28, 255)         # #13161c window background
COLOR_PANEL_BG = (26, 30, 38, 255)       # #1a1e26 panel cards
COLOR_PANEL_BORDER = (43, 51, 64, 255)   # #2b3340
COLOR_WELL_BG = (16, 19, 26, 255)        # #10131a inner list wells
COLOR_WELL_BORDER = (38, 46, 58, 255)    # #262e3a
COLOR_FIELD_BG = (35, 42, 53, 255)       # #232a35 inputs / combos
COLOR_FIELD_BORDER = (49, 58, 71, 255)   # #313a47
COLOR_TOOLBAR_BG = (20, 25, 34, 255)     # #141922 toolbar strip
COLOR_CARD_BG = (30, 37, 48, 255)        # #1e2530 selected-item cards
COLOR_CARD_HOVER = (38, 48, 62, 255)
# -- text
COLOR_TEXT = (223, 228, 236, 255)        # #dfe4ec body text
COLOR_HEADING = (234, 238, 244, 255)     # #eaeef4 section headings
COLOR_LABEL = (139, 148, 164, 255)       # #8b94a4 muted field labels
COLOR_LIST_TEXT = (199, 206, 222, 255)   # #c7cede list rows
# -- accents
COLOR_ACCENT = (63, 149, 230, 255)       # #3f95e6 primary blue
COLOR_ACCENT2 = (55, 196, 232, 255)      # #37c4e8 cyan
COLOR_CARET = (87, 176, 244, 255)        # #57b0f4 combo carets / links
# -- light results surface
COLOR_RESULTS_BG = (251, 251, 252, 255)  # #fbfbfc
COLOR_RESULTS_TEXT = (42, 47, 56, 255)   # #2a2f38
COLOR_RESULTS_TITLE = (30, 37, 48, 255)  # #1e2530 headings on white
COLOR_RESULTS_MUTED = (152, 160, 171, 255)  # #98a0ab
COLOR_TABLE_HEADER = (43, 51, 64, 255)   # #2b3340 table header on white
# -- semantic (legacy names kept for compatibility)
COLOR_TITLE = (47, 143, 224, 255)        # #2f8fe0 best cells / info accents
COLOR_SECTION = (53, 119, 184, 255)      # #3577b8 sub-labels on white
COLOR_SUCCESS = (47, 158, 86, 255)       # #2f9e56
COLOR_ERROR = (209, 84, 79, 255)         # #d1544f
COLOR_WARNING = (201, 138, 58, 255)      # #c98a3a
COLOR_HIGHLIGHT = (255, 255, 100, 255)

# Font sizes
FONT_SIZE_DEFAULT = 18
FONT_SIZE_TITLE = 24
FONT_SIZE_SECTION = 20

# Categories
CATEGORIES = ["STSO", "STMO", "MTSO", "MTMO", "RWO"]
ALGO_CATEGORIES = ["STSO", "STMO", "MTSO", "MTMO"]

# Metrics for multi-objective problems
METRICS = ["IGD", "HV", "IGDp", "GD", "DeltaP", "Spacing", "Spread"]

# Metric descriptions
METRIC_INFO = {
    "IGD": {"direction": "minimize", "requires_ref": True, "description": "Inverted Generational Distance"},
    "HV": {"direction": "maximize", "requires_ref": True, "description": "Hypervolume"},
    "IGDp": {"direction": "minimize", "requires_ref": True, "description": "IGD Plus"},
    "GD": {"direction": "minimize", "requires_ref": True, "description": "Generational Distance"},
    "DeltaP": {"direction": "minimize", "requires_ref": True, "description": "Delta_p"},
    "Spacing": {"direction": "minimize", "requires_ref": False, "description": "Spacing"},
    "Spread": {"direction": "minimize", "requires_ref": True, "description": "Spread"},
}

# Table formats
TABLE_FORMATS = ["excel", "latex"]

# Figure formats
FIGURE_FORMATS = ["png", "pdf", "svg"]

# Statistic types
STATISTIC_TYPES = ["mean", "median", "max", "min"]

# Paths (relative to tests folder)
DEFAULT_DATA_PATH = "Data"
DEFAULT_RESULTS_PATH = "Results"
DEFAULT_BACKUP_PATH = "backup"
