from pathlib import Path
from os.path import join as ospj
import os
from collections import defaultdict

def count_pdfs_by_folder(root: Path | str = Path("/")) -> dict[str, int]:
    """
    Walk through `root` and return a mapping:
        {absolute_folder_path: number_of_pdf_files_in_that_folder_and_descendants}
    """
    root = Path(root).expanduser().resolve()
    counts: defaultdict[str, int] = defaultdict(int)

    for path in root.rglob("*.pdf"):
        # path.parent is the directory containing this PDF;
        # path.parents includes all ancestor directories up to `root`
        for parent in path.parents:
            if parent == root.parent:  # stop once we move above the chosen root
                break
            counts[str(parent)] += 1

    return dict(counts)

if __name__ == "__main__":
    # Example: change this to any directory you want as the starting point
    root_directory = ospj(os.getcwd(),"output",'singleagent')         # or Path("C:/") on Windows
    pdf_counts = count_pdfs_by_folder(root_directory)

    for folder, n in sorted(pdf_counts.items()):
        print(f"{folder}: {n} PDF(s)")
