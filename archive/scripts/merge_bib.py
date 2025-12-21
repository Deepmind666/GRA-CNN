import re

def parse_bib(filename):
    entries = {}
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split by @
    parts = content.split('@')
    for part in parts:
        if not part.strip(): continue
        if part.strip().startswith('%'): continue
        
        # Extract key
        key_match = re.search(r'\{([^,]+),', part)
        if key_match:
            key = key_match.group(1).strip()
            entries[key] = '@' + part
    return entries

def main():
    base_dir = r'C:\GRA-CNN\APIN_Submission'
    fixed_entries = parse_bib(os.path.join(base_dir, 'manuscript_fixed.bib'))
    updated_entries = parse_bib(os.path.join(base_dir, 'manuscript_updated.bib'))
    
    # Merge: Updated overwrites Fixed if duplicate? 
    # Or just combine. Keys should be unique.
    final_entries = fixed_entries.copy()
    final_entries.update(updated_entries)
    
    print(f"Fixed: {len(fixed_entries)}, Updated: {len(updated_entries)}")
    print(f"Total unique: {len(final_entries)}")
    
    with open(os.path.join(base_dir, 'manuscript_final.bib'), 'w', encoding='utf-8') as f:
        for key in sorted(final_entries.keys()):
            f.write(final_entries[key])
            f.write('\n')
            
    print("Merged to manuscript_final.bib")

if __name__ == "__main__":
    import os
    main()
