from time import strftime
from datetime import datetime
from os import remove, getcwd, makedirs
from glob import glob
from shutil import rmtree
from os.path import join, isdir, abspath


# TODO: used only once, do we need this as a function
def init_folder(keep_files: bool) -> str:
    """Initialize an output folder.

    Parameters:
    -----------
    keep_files: bool
        If False, folder previously built is deleted.

    Returns:
    --------
    path: str
        Relative path of the folder.

    """

    date = strftime("%Y%m%d")
    time = strftime("%H%M%S")

    dir_name = "../../output/resite/"
    if not isdir(dir_name):
        makedirs(abspath(dir_name))

    path = abspath(dir_name + str(date) + '_' + str(time))
    makedirs(path)

    custom_log(' Folder path is: {}'.format(str(path)))

    if not keep_files:
        custom_log(' WARNING! Files will be deleted at the end of the run.')

    return path


# TODO: do we need a function ?
def remove_garbage(keep_files: bool, output_folder: str, lp: bool = True, script: bool = True, sol: bool = True):
    """Remove different files after the run.

    Parameters:
    -----------
    keep_files: bool
        If False, folder previously built is deleted.
    output_folder: str
        Path of output folder.
    lp: bool (default: True)
        Whether to remove .lp file
    script: bool (default: True)
        Whether to remove .script file
    sol: bool (default: True)
        Whether to remove .sol file
    """

    if not keep_files:
        rmtree(output_folder)

    directory = getcwd()

    if lp:
        files = glob(join(directory, '*.lp'))
        for f in files:
            remove(f)

    if script:
        files = glob(join(directory, '*.script'))
        for f in files:
            remove(f)

    if sol:
        files = glob(join(directory, '*.sol'))
        for f in files:
            remove(f)


def custom_log(message):
    """Prints a given message preceded by current time."""
    print(datetime.now().strftime('%H:%M:%S')+' --- '+str(message))
