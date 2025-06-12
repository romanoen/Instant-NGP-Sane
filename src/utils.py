# utils.py
import os

def get_filenames(root_folder):
    """
    Durchläuft bis zu 4 Ebenen von Unterordnern in root_folder
    und liefert sortierte, eindeutige Model-IDs zurück.
    """
    unique_names = set()
    for dirpath, _, filenames in os.walk(root_folder):
        rel = os.path.relpath(dirpath, root_folder)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 4: 
            continue
        for f in filenames:
            name = os.path.splitext(f)[0]
            cut = name.rsplit("_", 1)[0] if "_" in name else name
            unique_names.add(cut)
    return sorted(unique_names)[:-6]
