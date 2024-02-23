import shutil


def safe_move(file, fldr):
    #  move file, if the same file already exists add a  valid copy number (ex. image(#).png)
    fail = True
    copy_num = 0
    while fail:
        if copy_num:
            save_location = fldr / (file.stem + "(" + str(copy_num) + ")" + file.suffix)
        else:
            save_location = fldr / file.name

        if not save_location.is_file():
            shutil.move(file, save_location)
            fail = False
        else:
            copy_num += 1
    return
