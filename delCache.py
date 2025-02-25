import os
import shutil

def remove_pycache_dirs(root_dir):
    """
    删除指定目录及其子目录中的所有 __pycache__ 文件夹。

    :param root_dir: 要搜索的根目录
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            shutil.rmtree(pycache_path)
            print(f"已删除: {pycache_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    remove_pycache_dirs(project_root)
