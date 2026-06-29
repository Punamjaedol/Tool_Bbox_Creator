# ui_theme.py - Modern Minimal Design

import tkinter as tk
from tkinter import ttk

# -------------------------------------------------
# Color Palette (Modern Minimal - Almost Black + Neon)
# -------------------------------------------------

BG = "#070B0F"          # 메인 배경 (거의 검정)
PANEL = "#0D1117"       # 패널 (약간 밝은 검정)
PANEL2 = "#0E1218"      # 버튼/헤더 (더 검게)
SURFACE = "#1A1F26"     # 표면/호버 상태

TEXT = "#FAFBFC"        # 텍스트 (밝은 흰색)
TEXT_DIM = "#9CA3AF"    # 보조 텍스트 (부드러운 회색)

ACCENT = "#10B981"      # 네온 초록 (현대적이고 대담함)
ACCENT_HOVER = "#34D399" # 초록 호버 (더 밝은 톤)

BUTTON_BG = "#10B981"   # 버튼 배경 (네온 초록)
BUTTON_TEXT = "#070B0F" # 버튼 텍스트 (검정)

BORDER = "#1A1F26"      # 테두리

LIST_BG = "#0D1117"     # 리스트 배경
LIST_SELECTED = "#10B981" # 리스트 선택 (네온 초록)

# -------------------------------------------------
# Typography & Sizing (Modern Design)
# -------------------------------------------------

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_LABEL = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_BUTTON = ("Segoe UI", 10, )

# Modern spacing & sizing
BUTTON_PADDING = (18, 7)
ENTRY_PADDING = (8, 8)
TREEVIEW_ROWHEIGHT = 32


def set_all_frame_bg(widget):
    try:
        if widget.winfo_class() == "Frame":
            widget.configure(bg=BG)
        if widget.winfo_class() == "PanedWindow":
            widget.configure(bg=BG)
    except tk.TclError:
        pass
    
    for child in widget.winfo_children():
        set_all_frame_bg(child)
 

def apply(root):

    root.configure(bg=BG)
    set_all_frame_bg(root)
    
    style = ttk.Style(root)
    style.configure("Background.TFrame", background=BG)
    style.configure("Background.TLabel", background=BG, foreground=TEXT, font=FONT)
    
    # -------------------------------------------------
    # TButton - Modern flat design
    # -------------------------------------------------
    style.configure(
        "TButton",
        background=BUTTON_BG,
        foreground=BUTTON_TEXT,
        relief="raised",
        borderwidth=0,
        font=FONT_BUTTON,
        cursor="hand2",
        padding=BUTTON_PADDING,
        focuscolor="none",
    )
    style.map(
        "TButton",
        background=[("active", ACCENT_HOVER), ("pressed", "#0D8B5C")],
        foreground=[("active", BUTTON_TEXT)]
    )

    # -------------------------------------------------
    # TEntry - Larger, better spacing
    # -------------------------------------------------
    style.configure(
        "TEntry",
        fieldbackground=PANEL,
        foreground=TEXT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        insertcolor=ACCENT,
        relief="solid",
        borderwidth=1,
        font=FONT,
        padding=ENTRY_PADDING,
    )
    
    # -------------------------------------------------
    # Treeview - Spacious modern design
    # -------------------------------------------------

    style.configure(
        "Treeview",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=PANEL,
        borderwidth=0,
        rowheight=TREEVIEW_ROWHEIGHT,
        font=FONT,
    )

    style.map(
        "Treeview",
        background=[("selected", LIST_SELECTED)],
        foreground=[("selected", "white")],
    )

    style.configure(
        "Treeview.Heading",
        background=PANEL,
        foreground=TEXT,
        relief="flat",
        font=FONT_BOLD,
        borderwidth=1,
    )

    style.map(
        "Treeview.Heading",
        background=[
            ("active", SURFACE),
        ],
    )

    # -------------------------------------------------
    # TLabelFrame - Clean modern look
    # -------------------------------------------------

    style.configure(
        "TLabelframe",
        background=BG,
        relief="flat",
        borderwidth=0,
    )

    style.configure(
        "TLabelframe.Label",
        background=BG,
        foreground=TEXT,
        font=FONT_TITLE,
    )

    # -------------------------------------------------
    # TLabel - Better typography
    # -------------------------------------------------
    style.configure(
        "TLabel",
        background=BG,
        foreground=TEXT,
        font=FONT,
    )

    # -------------------------------------------------
    # Vertical.TScrollbar - Modern style
    # -------------------------------------------------

    style.configure(
        "Vertical.TScrollbar",
        background=PANEL,
        troughcolor=BG,
        bordercolor=BG,
        arrowcolor=TEXT,
        darkcolor=PANEL,
        lightcolor=PANEL,
    )

    # -------------------------------------------------


def paint(widget):

    """
    Only style tk widgets.
    ttk widgets are handled by ttk.Style().
    """
    global BG, PANEL, PANEL2, SURFACE, TEXT, TEXT_DIM, ACCENT, ACCENT_HOVER, BUTTON_BG, BUTTON_TEXT, BORDER, LIST_BG, LIST_SELECTED, FONT, FONT_BOLD

    for child in widget.winfo_children():

        cls = child.winfo_class()

        try:

            if cls == "Frame":
                child.configure(bg=BG)

            elif cls == "Label":
                child.configure(
                    bg=BG,
                    fg=TEXT,
                    font=FONT_BOLD,
                )

            elif cls == "Listbox":
                child.configure(
                    bg=LIST_BG,
                    fg=TEXT,
                    selectbackground=LIST_SELECTED,
                    selectforeground="white",
                    relief="solid",
                    bd=1,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                    font=FONT,
                )

            elif cls == "Canvas":
                child.configure(
                    bg=BG,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                )
                
            elif cls == "PanedWindow":
                child.configure(
                    bg=BG,
                    sashrelief="flat",
                    sashwidth=8,
                )
                
            elif cls == "Text":
                child.configure(
                    bg=PANEL,
                    fg=TEXT,
                    insertbackground=ACCENT,
                    relief="solid",
                    bd=1,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                    font=FONT,
                )
                
        except tk.TclError as e:
            pass

        paint(child)