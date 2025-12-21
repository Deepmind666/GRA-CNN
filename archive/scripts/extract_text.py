
import PyPDF2
import os

pdf_path = r"C:\GRA-CNN\PRLETTERS-D-25-01152 (2).pdf"
output_path = r"C:\GRA-CNN\draft_content.txt"

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Successfully extracted {len(text)} characters to {output_path}")
except Exception as e:
    print(f"Error: {e}")
