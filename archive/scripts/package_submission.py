import shutil
import os

src_dir = r'C:\GRA-CNN\APIN_Submission'
dst_dir = r'C:\GRA-CNN\Final_Submission_Package'

# Copy PDF
shutil.copy(os.path.join(src_dir, 'manuscript_apin.pdf'), os.path.join(dst_dir, 'GRA-CNN_Final_Revision.pdf'))
print(f"Copied PDF to {dst_dir}\\GRA-CNN_Final_Revision.pdf")

# Copy TeX
shutil.copy(os.path.join(src_dir, 'manuscript_apin.tex'), os.path.join(dst_dir, 'manuscript.tex'))
shutil.copy(os.path.join(src_dir, 'manuscript_fixed.bib'), os.path.join(dst_dir, 'references.bib'))

# Copy Vis
if not os.path.exists(os.path.join(dst_dir, 'vis')):
    shutil.copytree(os.path.join(src_dir, 'vis'), os.path.join(dst_dir, 'vis'))
else:
    # If exists, copy contents
    for file in os.listdir(os.path.join(src_dir, 'vis')):
        src_file = os.path.join(src_dir, 'vis', file)
        dst_file = os.path.join(dst_dir, 'vis', file)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)

# Copy root figures (fig1.pdf etc)
for file in os.listdir(src_dir):
    if file.endswith('.pdf') and file.startswith('fig'):
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
    if file.endswith('.cls') or file.endswith('.bst') or file.endswith('.sty'):
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

print("Package creation complete.")
