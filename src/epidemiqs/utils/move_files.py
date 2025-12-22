import os
import shutil
from os.path import join as ospj
import re
from datetime import datetime
from epidemiqs.config import get_settings
from typing import List
def create_experiment_folder(base_path: str, experiment_name: str) -> str:
    """
    Create a unique experiment folder inside the given base_path.
    Example: base_path/experiment-11051432
    Returns the full path to the created folder.
    """

    safe_name = re.sub(r'[^A-Za-z0-9_\- ]+', '', experiment_name).strip().replace(' ', '_')

    if not safe_name:
        safe_name = "experiment"
    os.makedirs(base_path, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%H%M")
    folder_name = f"{safe_name}-{timestamp}"
    folder_path = os.path.join(base_path, folder_name)

    counter = 1
    while os.path.exists(folder_path):
        folder_path = os.path.join(base_path, f"{safe_name}-{timestamp}_{counter}")
        counter += 1
        
    os.makedirs(folder_path)
    return folder_path

def move_to_destination(
    source_dir: str,
    destination: str,
    move: bool = True,
    force_move: bool = False,
    ignore_names: List[str] = [ "token.csv", "tokens_by_phase.csv"]
) -> List[str]:
    """
    Move or copy files from source_dir to destination with rules:

    Modes:
    - force_move=True:
         Move EVERYTHING (ignore ignore_names)
    - move=False:
         Copy EVERYTHING
    - move=True:
         Apply rules:
              * Python files (*.py): ALWAYS MOVE
              * ignore_names: COPY ONLY
              * everything else: MOVE

    Returns:
        List of new full paths in destination.
    """
    try:
        os.makedirs(destination, exist_ok=True)
        ignore_set = set(ignore_names or [])
        new_paths = []
        if os.path.isdir(source_dir):
                for filename in os.listdir(source_dir):
                    src = os.path.join(source_dir, filename)
                    dst = os.path.join(destination, filename)

                    if not os.path.isfile(src):
                        continue  # ignore subdirectories

                    # FORCE MOVE EVERYTHING
                    if force_move:
                        shutil.move(src, dst)
                        new_paths.append(dst)
                        continue
                    
                    
                    # MOVE MODE (with rules)
                    # Always move Python files
                    if filename.endswith(".py"):
                        shutil.move(src, dst)
                        new_paths.append(dst)
                        continue
                    
                    if filename.endswith(".jsonl"):
                        shutil.move(src, dst)
                        new_paths.append(dst)
                        continue
                    # COPY EVERYTHING
                    if not move:
                        shutil.copy2(src, dst)
                        new_paths.append(dst)
                        continue



                    # If file is in ignore list → COPY instead of move
                    if filename in ignore_set:
                        shutil.copy2(src, dst)
                        new_paths.append(dst)
                        continue

                    # Otherwise: MOVE
                    shutil.move(src, dst)
                    new_paths.append(dst)

                return new_paths
        elif os.path.isfile(source_dir):
            filename = os.path.basename(source_dir)
            dst = os.path.join(destination, filename)

            # FORCE MOVE EVERYTHING
            if force_move:
                shutil.move(source_dir, dst)
                new_paths.append(dst)
                return new_paths
            
            if not move:
                shutil.copy2(source_dir, dst)
                new_paths.append(dst)
                return new_paths
            else:
                shutil.move(source_dir, dst)
                new_paths.append(dst)
                return new_paths
    except Exception as e:
        print(f"Error moving/copying files: {str(e)}")
        return None

if __name__ == "__main__":
    cfg=get_settings(config_path="config.yaml")
    print(cfg.paths.data_path)
    source_dest=ospj(cfg.paths.data_path)
    print(source_dest)
    destination=ospj(os.getcwd(),f"output")
    print(move_to_destination(source_dest,destination,move=False))
    # move files to destination
    #move_to_destination(source_dest, destination)