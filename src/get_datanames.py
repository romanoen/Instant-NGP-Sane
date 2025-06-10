import os

def get_filenames(root_folder):
    """
    Durchläuft bis zu 4 Ebenen von Unterordnern in root_folder,
    schneidet jeden Dateinamen vor dem letzten Unterstrich ab,
    sammelt nur eindeutige Werte und gibt sie sortiert zurück.
    """
    unique_names = set()

    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Berechne die Tiefe relativ zum root_folder
        rel_path = os.path.relpath(dirpath, root_folder)
        if rel_path == ".":
            depth = 0
        else:
            # Anzahl der Verzeichnisse zwischen root_folder und dirpath
            depth = rel_path.count(os.sep) + 1

        # Nur Unterordner bis einschl. Tiefe 4 betrachten
        if depth <= 4:
            for f in filenames:
                # Dateiname ohne Erweiterung
                name_no_ext = os.path.splitext(f)[0]
                # Vor dem letzten '_' abschneiden
                if "_" in name_no_ext:
                    cut_name = name_no_ext.rsplit("_", 1)[0]
                else:
                    cut_name = name_no_ext
                unique_names.add(cut_name)

    #return sorted(unique_names)[6:-48]
    return sorted(unique_names)[:-6]