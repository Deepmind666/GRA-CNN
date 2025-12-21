import os
import subprocess
import sys

scripts = [
    'draw_fig2.py',
    'draw_fig3.py',
    'draw_flops.py',
    'draw_convergence.py',
    'draw_correlation.py',
    'draw_rho_detailed.py',
    'draw_c100_r56.py', # Fig 8
    'draw_r110_c10.py', # Fig 9
    'draw_vgg.py'       # Fig 10
]

def main():
    vis_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in scripts:
        print(f"Running {script}...")
        script_path = os.path.join(vis_dir, script)
        try:
            subprocess.run([sys.executable, script_path], check=True, cwd=vis_dir)
            print(f"Successfully ran {script}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
        except Exception as e:
            print(f"Unexpected error running {script}: {e}")

if __name__ == "__main__":
    main()
