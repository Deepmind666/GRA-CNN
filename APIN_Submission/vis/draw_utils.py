import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    """
    Sets the APIN journal style for matplotlib.
    """
    plt.style.use('default')
    
    # Font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    
    # Line and Marker settings
    plt.rcParams['lines.linewidth'] = 2.2
    plt.rcParams['lines.markersize'] = 7
    plt.rcParams['axes.linewidth'] = 1.5
    
    # Ticks
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    
    # Grid
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

def get_palette():
    """
    Returns the fixed color palette for the paper.
    """
    return {
        'L1-Norm': '#1f77b4',  # Blue
        'L1': '#1f77b4',       # Alias
        'FPGM': '#2ca02c',     # Green
        'HRank': '#ff7f0e',    # Orange
        'GRA-CNN': '#d62728',  # Red
        'GRA': '#d62728',      # Alias
        'Baseline': 'black'
    }

def get_markers():
    """
    Returns the fixed markers for the paper.
    """
    return {
        'L1-Norm': 'o',
        'L1': 'o',
        'FPGM': 's',
        'HRank': '^',
        'GRA-CNN': 'D',
        'GRA': 'D',
        'Baseline': '_'
    }

def save_fig(fig, filename):
    """
    Saves the figure to the specified path in PDF format.
    Ensures the directory exists.
    """
    # Ensure directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save as vector PDF
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved figure to {filename}")
