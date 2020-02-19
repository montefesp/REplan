import yaml
from time import strftime
from datetime import datetime
from os import remove, getcwd, makedirs
from glob import glob
from shutil import rmtree
from os.path import join, isdir, abspath


# TODO: do we really need a function for this? can be done in one line
def read_inputs(inputs):
    """

    Parameters
    ----------
    inputs :

    Returns
    -------

    """
    with open(inputs) as infile:
        data = yaml.safe_load(infile)

    return data


# TODO: used only once, do we need this as a function
def init_folder(keepfiles):
    """Initiliaze an output folder.

    Parameters:

    ------------

    keepfiles : boolean
        If False, folder previously built is deleted.

    Returns:

    ------------

    path : str
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

    if not keepfiles:
        custom_log(' WARNING! Files will be deleted at the end of the run.')

    return path


# TODO: do we need a function ?
def remove_garbage(keepfiles, output_folder, lp=True, script=True, sol=True):

    """Remove different files after the run.

    Parameters:

    ------------

    keepfiles : boolean
        If False, folder previously built is deleted.

    output_folder : str
        Path of output folder.

    """

    if not keepfiles:
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
    """
    Parameters
    ----------
    message : str

    """
    print(datetime.now().strftime('%H:%M:%S')+' --- '+str(message))
