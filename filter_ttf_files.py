"""
This file was used to extract ttf files from a github repository.
This is seperate from the OCR program.
"""

import os

master_folder = "googleFonts\\ufl"
save_folder = "12SEN CharacterGenerator\\Fonts\\Google"

for folder in os.listdir(master_folder):
    subpath = os.path.join(master_folder, folder)
    
    for file in os.listdir(subpath):
        if file.endswith("-Regular.ttf") or file.endswith("-Bold.ttf") or file.endswith("-Italic.ttf") or file.endswith("-BoldItalic.ttf"):
            ttf_filepath = os.path.join(subpath, file)
        
            with open(ttf_filepath, "rb") as f:
                content = f.read()
            
            save_filepath = os.path.join(save_folder, file)
            with open(save_filepath, "wb") as f:
                f.write(content)

print("done")