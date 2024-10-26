import os
import platform
import subprocess


def rm_file_ext(filename: str) -> str:
    extension_len = len(filename.split(sep='.')[-1])
    return filename[:-(extension_len+1)]


def get_filename_ext(img_name) -> tuple[str, str]:
    split = img_name.split('.')
    return ''.join(split[:-1]), split[-1]


def get_filename(path) -> str:
    return os.path.split(path)[1]


def open_file_default(path):
    if platform.system() == 'Darwin':    # macOS
        subprocess.call(('open', path))
    elif platform.system() == 'Windows':
        os.startfile(path)
    else:  # linux variants
        subprocess.call(('xdg-open', path))


def get_dir_name(path, steps=1) -> str:
    dir_name = path
    for i in range(steps):
        dir_name = os.path.dirname(dir_name)
    return dir_name
