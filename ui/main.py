"""DDMTOLab UI - Main entry point."""

import re
import sys
from pathlib import Path

# Ensure 'from main import ...' resolves to this module (not a duplicate)
# when this file is run as __main__
sys.modules.setdefault('main', sys.modules[__name__])

# Analysis figures are rendered in background threads; interactive matplotlib
# backends (TkAgg) are not thread-safe, so force the non-interactive Agg.
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

# Setup paths
_ui_dir = Path(__file__).resolve().parent
_project_root = _ui_dir.parent

if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))
if str(_ui_dir) not in sys.path:
    sys.path.insert(0, str(_ui_dir))

import dearpygui.dearpygui as dpg
from components.dpg_helpers import get_texture_registry, load_image_to_texture
from pages import test_mode, batch_mode
from config.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    COLOR_APP_BG, COLOR_PANEL_BG, COLOR_PANEL_BORDER, COLOR_FIELD_BG,
    COLOR_FIELD_BORDER, COLOR_TOOLBAR_BG, COLOR_CARD_BG,
    COLOR_TEXT, COLOR_LABEL, COLOR_ACCENT, COLOR_ACCENT2, COLOR_TITLE,
)

# Store fonts globally
_fonts = {"default": None, "title": None, "logo": None, "header": None, "section": None, "tab": None, "bold": None, "header_large": None}


def get_fonts():
    """Return the fonts dictionary for use by other modules."""
    return _fonts

def _create_dark_theme():
    """Global refined dark theme (claude.ai/design 'DDMTOLab GUI (Refined)')."""
    with dpg.theme() as dark_theme:
        with dpg.theme_component(dpg.mvAll):
            # Surfaces
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLOR_APP_BG)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLOR_PANEL_BG)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (28, 33, 43))
            dpg.add_theme_color(dpg.mvThemeCol_Border, COLOR_FIELD_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, COLOR_TOOLBAR_BG)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, COLOR_PANEL_BG)
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, COLOR_TOOLBAR_BG)
            # Fields (inputs / combos)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, COLOR_FIELD_BG)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (42, 51, 66))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (49, 60, 78))
            # Headers (collapsing headers, combo selection)
            dpg.add_theme_color(dpg.mvThemeCol_Header, COLOR_CARD_BG)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (38, 48, 62))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (44, 56, 73))
            # Generic buttons (specific buttons get dedicated themes)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (35, 44, 55))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (44, 55, 69))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (54, 67, 85))
            # Tabs (pill-like)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (26, 33, 44))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (36, 50, 68))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (36, 50, 68))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (26, 33, 44))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (36, 50, 68))
            # Text and accents
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLOR_TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (90, 99, 115))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, COLOR_TITLE)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, COLOR_TITLE)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (87, 176, 244))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, COLOR_TITLE)  # progress fill
            # Scrollbars
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (56, 66, 79))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (70, 82, 100))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (84, 98, 118))
            # Shape language
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 7)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 7)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
    return dark_theme


def _add_logo_wordmark(parent):
    """Add the 'D²MTOLab' wordmark next to the logo mark."""
    with dpg.group(parent=parent):
        # Nudge the text down so it centers against the 54px mark
        dpg.add_spacer(height=8)
        logo_text = dpg.add_text("D²MTOLab", color=(255, 255, 255, 255))
        if _fonts["logo"]:
            dpg.bind_item_font(logo_text, _fonts["logo"])


def _load_logo(parent):
    """Load and display the logo mark + wordmark (design: mark then D²MTOLab)."""
    from PIL import Image
    import numpy as np
    import io

    img = None

    # Try loading from SVG first (first existing candidate wins)
    for svg_name in ("logo_new.svg", "logo.svg"):
        svg_logo = _ui_dir / "assets" / svg_name
        if not svg_logo.exists():
            continue
        try:
            import cairosvg
            # The exported SVGs carry a baked-in "DDMTOLab" wordmark. The header
            # draws its own D²MTOLab wordmark next to the mark, so strip any text
            # from the asset first; otherwise both appear side by side. The PNG
            # fallback is already mark-only, so this keeps the two paths identical.
            svg_source = re.sub(r'<text\b.*?</text>', '',
                                svg_logo.read_text(encoding='utf-8'), flags=re.S)
            png_data = cairosvg.svg2png(bytestring=svg_source.encode('utf-8'), scale=3)
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")
            # Crop to content area
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            break
        except Exception:
            img = None

    # Fallback to PNG if SVG fails
    if img is None:
        png_logo = _ui_dir / "assets" / "logo.png"
        if png_logo.exists():
            try:
                img = Image.open(str(png_logo)).convert("RGBA")
            except Exception:
                img = None

    if img is not None:
        try:
            w, h = img.size

            target_height = 54
            if h > target_height:
                scale = target_height / h
                w = int(w * scale)
                h = target_height
                img = img.resize((w, h), Image.Resampling.LANCZOS)

            data = np.array(img).astype(np.float32) / 255.0
            flat = data.flatten().tolist()

            tex_reg = get_texture_registry()
            if not dpg.does_item_exist(tex_reg):
                raise RuntimeError("Texture registry does not exist")

            tex_tag = dpg.add_static_texture(
                width=w, height=h, default_value=flat,
                parent=tex_reg
            )

            if not dpg.does_item_exist(parent):
                raise RuntimeError(f"Parent {parent} does not exist")

            dpg.add_image(tex_tag, width=w, height=h, parent=parent)
            _add_logo_wordmark(parent)
            return

        except Exception:
            pass

    # Text-only fallback when no mark image is available
    _add_logo_wordmark(parent)


def main(smoke_frames: int = 0, on_ready=None):
    """Main application entry point.

    Parameters
    ----------
    smoke_frames : int, optional
        If > 0, render only this many frames and exit (for automated smoke
        tests). Can also be set via the DDMTOLAB_UI_SMOKE_FRAMES env var.
    on_ready : callable, optional
        Called once after the UI is fully built, before the render loop.
    """
    import os
    smoke_frames = smoke_frames or int(os.environ.get("DDMTOLAB_UI_SMOKE_FRAMES", "0") or 0)

    dpg.create_context()
    get_texture_registry()

    # Set base path for data storage
    base_path = str(_project_root / "tests")

    # Create dark theme
    dark_theme = _create_dark_theme()

    # Load fonts
    with dpg.font_registry():
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei (Chinese support)
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    _fonts["default"] = dpg.add_font(fp, 18)
                    _fonts["section"] = dpg.add_font(fp, 20)
                    _fonts["tab"] = dpg.add_font(fp, 20)
                    _fonts["title"] = dpg.add_font(fp, 24)
                    _fonts["header"] = dpg.add_font(fp, 28)
                    _fonts["logo"] = dpg.add_font(fp, 32)
                    _fonts["header_large"] = dpg.add_font(fp, 34)
                    # Try to load bold font variant
                    bold_paths = [
                        "C:/Windows/Fonts/msyhbd.ttc",    # Microsoft YaHei Bold
                        "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
                        "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Bold
                    ]
                    for bp in bold_paths:
                        if Path(bp).exists():
                            try:
                                _fonts["bold"] = dpg.add_font(bp, 18)
                                break
                            except Exception:
                                continue
                    # Try to load a stylish title font
                    title_font_paths = [
                        "C:/Windows/Fonts/GOTHIC.TTF",     # Century Gothic
                        "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI
                        "C:/Windows/Fonts/georgia.ttf",    # Georgia
                        "C:/Windows/Fonts/cambria.ttc",    # Cambria
                        "C:/Windows/Fonts/arial.ttf",      # Arial
                    ]
                    for tp in title_font_paths:
                        if Path(tp).exists():
                            try:
                                _fonts["title_stylish"] = dpg.add_font(tp, 36)
                                break
                            except Exception:
                                continue
                    break
                except Exception:
                    continue

    if _fonts["default"]:
        dpg.bind_font(_fonts["default"])

    # Apply default dark theme
    dpg.bind_theme(dark_theme)

    dpg.create_viewport(title="DDMTOLab - Data-Driven Multitask Optimization Laboratory",
                        width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    with dpg.window(tag="primary_window"):
        # Header with logo and title
        with dpg.group(horizontal=True, tag="header_group"):
            dpg.add_spacer(width=2)
            _load_logo("header_group")
            dpg.add_spacer(width=10)
            # Vertical divider between logo and title
            with dpg.drawlist(width=2, height=40):
                dpg.draw_line((0, 4), (0, 38), color=(43, 51, 64, 255), thickness=1)
            dpg.add_spacer(width=10)
            with dpg.group():
                dpg.add_spacer(height=1)
                # Gradient title: accent blue #3f95e6 -> cyan #37c4e8
                _title_words = ["Data-Driven", " Multitask", " Optimization", " Laboratory"]
                _c_start = (63, 149, 230)
                _c_end = (55, 196, 232)
                _title_font = _fonts.get("title_stylish") or _fonts.get("header_large") or _fonts.get("header")
                # Zero-spacing theme for tight word packing
                with dpg.theme() as _title_spacing_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0)
                with dpg.group(horizontal=True) as _title_grp:
                    for idx, word in enumerate(_title_words):
                        t = idx / max(len(_title_words) - 1, 1)
                        r = int(_c_start[0] + (_c_end[0] - _c_start[0]) * t)
                        g = int(_c_start[1] + (_c_end[1] - _c_start[1]) * t)
                        b = int(_c_start[2] + (_c_end[2] - _c_start[2]) * t)
                        tw = dpg.add_text(word, color=(r, g, b, 255))
                        if _title_font:
                            dpg.bind_item_font(tw, _title_font)
                dpg.bind_item_theme(_title_grp, _title_spacing_theme)

        # Accent line under the header: blue -> cyan fading out (design: 2px)
        accent_w = 1500
        accent_h = 2
        with dpg.drawlist(width=accent_w, height=accent_h, tag="header_accent"):
            segments = 60
            seg_w = accent_w / segments
            for i in range(segments):
                t = i / segments
                # alpha .9 at left, 0 by 70% of the width
                alpha = int(230 * max(0.0, 1 - t / 0.7))
                cr = int(63 + (55 - 63) * t)
                cg = int(149 + (196 - 149) * t)
                cb = int(230 + (232 - 230) * t)
                dpg.draw_rectangle(
                    pmin=(i * seg_w, 0), pmax=((i + 1) * seg_w, accent_h),
                    color=(cr, cg, cb, alpha), fill=(cr, cg, cb, alpha),
                )
        dpg.add_spacer(height=6)

        # Main tab bar - use section font (20pt) for tab labels only
        _tab_font = _fonts.get("section")
        with dpg.tab_bar(tag="main_tabs"):
            with dpg.tab(label="  Test Mode  ", tag="test_tab") as _tab1:
                test_mode.create(parent="test_tab", base_path=base_path)
            with dpg.tab(label="  Batch Experiment  ", tag="batch_tab") as _tab2:
                batch_mode.create(parent="batch_tab", base_path=base_path)

        # Bind font to tab buttons after content is created with default font
        if _tab_font and _fonts.get("default"):
            dpg.bind_item_font(_tab1, _tab_font)
            dpg.bind_item_font(_tab2, _tab_font)
            # Restore default font on content containers
            for child in dpg.get_item_children(_tab1, 1) or []:
                dpg.bind_item_font(child, _fonts["default"])
            for child in dpg.get_item_children(_tab2, 1) or []:
                dpg.bind_item_font(child, _fonts["default"])

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    if on_ready is not None:
        on_ready()

    # Main loop
    frame_count = 0
    while dpg.is_dearpygui_running():
        test_mode.update()
        batch_mode.update()
        dpg.render_dearpygui_frame()
        if smoke_frames:
            frame_count += 1
            if frame_count >= smoke_frames:
                break

    dpg.destroy_context()


if __name__ == "__main__":
    main()
