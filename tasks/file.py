import shutil
from pathlib import Path


def get_safe_path(path: Path) -> Path:

    copy_num = 0
    while True:
        if copy_num:
            save_location = path.parent.absolute() / (path.stem + "(" + str(copy_num) + ")" + path.suffix)
        else:
            save_location = path

        if not save_location.is_file():
            return save_location
        copy_num += 1


def safe_move(fldr: Path, file: Path):
    """
    Move file, if the same file already exists add a  valid copy number (ex. image(#).png)
    :param fldr: Folder Path to move the file to.
    :param file: Current Path of file to move.
    :return:
    """
    shutil.move(file, get_safe_path(fldr / file.name))
