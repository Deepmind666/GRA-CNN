import os
import xml.etree.ElementTree as ET
import base64
from PIL import Image
from io import BytesIO

def convert_svg_wrapper_to_pdf(svg_path, output_pdf_path):
    try:
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Namespaces
        ns = {'svg': 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink'}
        
        # Find image tag
        # Note: ElementTree might not handle namespaces gracefully in find/findall without explicit map
        # We'll try to find any tag that ends with 'image' or use the namespace
        
        image_tag = root.find('.//{http://www.w3.org/2000/svg}image')
        if image_tag is None:
            # Try without namespace if it fails
            image_tag = root.find('.//image')
            
        if image_tag is None:
            print("No image tag found in SVG.")
            return False

        # Get href
        href = image_tag.get('{http://www.w3.org/1999/xlink}href')
        if not href:
            href = image_tag.get('href') # try standard href
            
        if not href or not href.startswith('data:image/png;base64,'):
            print("Image href is not a base64 png.")
            return False
            
        # Extract base64 string
        base64_str = href.split('data:image/png;base64,')[1]
        
        # Decode
        image_data = base64.b64decode(base64_str)
        
        # Open image with PIL
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB (remove alpha if present, as PDF doesn't always like it, or keep it)
        # Usually for papers, RGB is fine. PNG might be RGBA.
        if img.mode == 'RGBA':
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save as PDF
        img.save(output_pdf_path, "PDF", resolution=100.0)
        print(f"Successfully converted {svg_path} to {output_pdf_path}")
        return True
        
    except Exception as e:
        print(f"Error converting SVG to PDF: {e}")
        return False

if __name__ == "__main__":
    svg_file = r"C:\GRA-CNN\APIN_Submission\GRA-CNN流程图.svg"
    pdf_file = r"C:\GRA-CNN\APIN_Submission\fig1.pdf"
    
    if os.path.exists(svg_file):
        convert_svg_wrapper_to_pdf(svg_file, pdf_file)
    else:
        print(f"File not found: {svg_file}")
