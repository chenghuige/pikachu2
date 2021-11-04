import atexit
import os
import shutil
import uuid

GLOBAL_TMP_DIR = os.path.join(os.environ['HOME'], ".cache/lichee")
GLOBAL_TMP_FILE_PATHS = []
GLOBAL_TMP_LOCK_DIR = os.path.join(os.environ['HOME'], ".cache/lichee/lock")
GLOBAL_TMP_DIRS = [GLOBAL_TMP_LOCK_DIR]


def get_global_temp_dir():
    """get global unique temp dir

    """
    if not os.path.exists(GLOBAL_TMP_DIR):
        os.makedirs(GLOBAL_TMP_DIR)
    return GLOBAL_TMP_DIR


def get_temp_file_path_once():
    """get empty temp file path which will be cleaned upon exit

    """
    global_tmp_dir = get_global_temp_dir()
    tmp_file_path = os.path.join(global_tmp_dir, uuid.uuid4().hex)
    GLOBAL_TMP_FILE_PATHS.append(tmp_file_path)
    return tmp_file_path


def get_temp_dir_once():
    """get empty temp dir which will be cleaned upon exit

    """
    global_tmp_dir = get_global_temp_dir()
    tmp_dir = os.path.join(global_tmp_dir, uuid.uuid4().hex)
    os.makedirs(tmp_dir)
    GLOBAL_TMP_DIRS.append(tmp_dir)
    return tmp_dir


def get_global_temp_lock_dir():
    if not os.path.exists(GLOBAL_TMP_LOCK_DIR):
        os.makedirs(GLOBAL_TMP_LOCK_DIR)
    return GLOBAL_TMP_LOCK_DIR


def create_temp_lock_file(filename: str):
    global_tmp_lock_dir = get_global_temp_lock_dir()
    tmp_lock_file_path = os.path.join(global_tmp_lock_dir, filename + ".lock")
    with open(tmp_lock_file_path, mode="w+"):
        pass
    return tmp_lock_file_path


def exist_temp_lock_file(filename: str):
    global_tmp_lock_dir = get_global_temp_lock_dir()
    tmp_lock_file_path = os.path.join(global_tmp_lock_dir, filename + ".lock")
    return os.path.exists(tmp_lock_file_path)


def clear_tmp_files():
    for file_path in GLOBAL_TMP_FILE_PATHS:
        try:
            os.remove(file_path)
        except:
            pass

    for dir_path in GLOBAL_TMP_DIRS:
        try:
            shutil.rmtree(dir_path)
        except:
            pass


atexit.register(clear_tmp_files)
