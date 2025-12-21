import shutil
import os
import glob

src_dir = r'C:\GRA-CNN\vis'
dst_dir = r'C:\GRA-CNN\APIN_Submission\vis'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for file in glob.glob(os.path.join(src_dir, '*')):
    if os.path.isfile(file):
        shutil.copy(file, dst_dir)
        print(f"Copied {file}")
