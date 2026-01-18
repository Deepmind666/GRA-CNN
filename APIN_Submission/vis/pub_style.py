"""
Professional Visualization Utilities for GRA-CNN APIN Paper
============================================================
Implements publication-quality figure styling based on 2024-2025 Q1/Q2 journal standards.
Reference: Nature Machine Intelligence, IEEE TPAMI, Applied Intelligence

Key Features:
- Nature/Science professional color palette (vibrant, distinguishable)
- Error bar support for statistical robustness
- Multi-panel subplot layouts
- 600 DPI vector PDF output
- Times New Roman typography
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import os

# =============================================================================
# PROFESSIONAL COLOR PALETTES
# =============================================================================

# Nature-inspired vibrant palette (NOT black/white)
NATURE_PALETTE = {
    'GRA-CNN': '#E64B35',    # Vibrant Red (Our method - stands out)
    'GRA': '#E64B35',        # Alias
    'L1-Norm': '#4DBBD5',    # Cyan Blue
    'L1': '#4DBBD5',         # Alias
    'FPGM': '#00A087',       # Teal Green
    'HRank': '#3C5488',      # Navy Blue
    'ACP': '#F39B7F',        # Salmon
    'DepGraph': '#8491B4',   # Lavender
    'Baseline': '#7E6148',   # Brown
}

# Alternative: Cell/Science palette
SCIENCE_PALETTE = {
    'GRA-CNN': '#DC0000',    # Deep Red
    'GRA': '#DC0000',
    'L1-Norm': '#3C8DAD',    # Steel Blue  
    'L1': '#3C8DAD',
    'FPGM': '#00A651',       # Emerald
    'HRank': '#9558B2',      # Purple
    'ACP': '#E69F00',        # Orange
    'DepGraph': '#56B4E9',   # Sky Blue
    'Baseline': '#666666',   # Dark Gray
}

# Markers for different methods
MARKERS = {
    'GRA-CNN': 'o',  # Circle (filled)
    'GRA': 'o',
    'L1-Norm': 's',  # Square
    'L1': 's',
    'FPGM': '^',     # Triangle up
    'HRank': 'D',    # Diamond
    'ACP': 'v',      # Triangle down
    'DepGraph': 'p', # Pentagon
    'Baseline': '_', # Horizontal line
}

# Line styles
LINESTYLES = {
    'GRA-CNN': '-',   # Solid (our method)
    'GRA': '-',
    'L1-Norm': '--',  # Dashed
    'L1': '--',
    'FPGM': '-.',     # Dash-dot
    'HRank': ':',     # Dotted
    'ACP': '--',
    'DepGraph': '-.',
    'Baseline': ':',
}

def get_palette(style='nature'):
    """Get color palette. Default is Nature-style vibrant colors."""
    if style == 'science':
        return SCIENCE_PALETTE
    return NATURE_PALETTE

def get_markers():
    return MARKERS

def get_linestyles():
    return LINESTYLES

# =============================================================================
# PUBLICATION STYLE SETUP
# =============================================================================

def set_publication_style():
    """
    Configure matplotlib for APIN/Nature-quality publications.
    - Times New Roman font
    - Appropriate sizes for single/double column figures
    - Professional axis styling
    """
    plt.style.use('default')  # Reset first
    
    # Use a style close to publication quality
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 600,  # High-res for publication
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Axes settings
        'axes.linewidth': 1.0,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        
        # Line settings
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.0,
        
        # Tick settings
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': True,
        
        # Error bar settings
        'errorbar.capsize': 3,
    })

# =============================================================================
# FIGURE SIZE STANDARDS (APIN/Springer)
# =============================================================================

# Single column: ~88mm = 3.46in
# Double column: ~180mm = 7.09in
FIGURE_SIZES = {
    'single_column': (3.5, 2.8),
    'single_column_tall': (3.5, 4.0),
    'double_column': (7.0, 3.5),
    'double_column_tall': (7.0, 5.0),
    'square': (4.0, 4.0),
    'wide': (7.0, 2.5),
}

def get_figure_size(style='single_column'):
    return FIGURE_SIZES.get(style, FIGURE_SIZES['single_column'])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig, filename, formats=['pdf']):
    """
    Save figure in publication-ready format(s).
    Creates directory if needed.
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    base, _ = os.path.splitext(filename)
    
    for fmt in formats:
        output_path = f"{base}.{fmt}"
        fig.savefig(output_path, format=fmt, dpi=600, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

def add_significance_annotation(ax, x1, x2, y, text='*', h=0.02):
    """Add significance annotation (bracket with stars) between two bars."""
    y_max = y + h * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y_max, y_max, y], 'k-', lw=0.8)
    ax.text((x1 + x2) / 2, y_max, text, ha='center', va='bottom', fontsize=9)

def create_grouped_bar_positions(n_groups, n_bars, bar_width=0.25, group_gap=0.3):
    """Calculate positions for grouped bar chart."""
    positions = []
    for i in range(n_bars):
        offset = (i - n_bars/2 + 0.5) * bar_width
        pos = np.arange(n_groups) * (n_bars * bar_width + group_gap) + offset
        positions.append(pos)
    return positions

# =============================================================================
# INITIALIZATION
# =============================================================================

# Auto-apply publication style when imported
set_publication_style()

if __name__ == '__main__':
    # Demo: Show color palette
    print("Nature Palette Colors:")
    for name, color in NATURE_PALETTE.items():
        print(f"  {name}: {color}")
