
import requests
import os

base_url = "https://raw.githubusercontent.com/godkingjay/springer-nature-latex-template/master/"
files = ["sn-jnl.cls", "sn-article.tex", "sn-basic.bst", "sn-mathphys.bst"]
dest_dir = r"C:\GRA-CNN\APIN_Submission"

os.makedirs(dest_dir, exist_ok=True)

for file in files:
    url = base_url + file
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(dest_dir, file), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file}")
        else:
            print(f"Failed to download {file}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {file}: {e}")
