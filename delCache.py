import os
import shutil

def remove_unwanted_dirs(root_dir):
    """
    Remove all '__pycache__' directories and 'logs/tensorboard' directories
    within the specified root directory and its subdirectories.

    :param root_dir: The root directory to search
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check and remove '__pycache__' directories
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            shutil.rmtree(pycache_path)
            print(f"Deleted: {pycache_path}")

        # Check and remove 'logs/tensorboard' directories
        logs_path = os.path.join(dirpath, 'logs', 'tensorboard')
        if os.path.exists(logs_path) and os.path.isdir(logs_path):
            shutil.rmtree(logs_path)
            print(f"Deleted: {logs_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    remove_unwanted_dirs(project_root)
    print("All Done!")