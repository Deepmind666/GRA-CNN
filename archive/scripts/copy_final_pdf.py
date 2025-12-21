import shutil
import os

src = r'C:\GRA-CNN\APIN_Submission\manuscript_apin.pdf'
dst = r'C:\GRA-CNN\GRA-CNN_Final_Revision_v2.pdf'

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"Copied to {dst}")
else:
    print("Source not found")
