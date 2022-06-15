from ast import Str
from pathlib import Path
from typing import (
    Callable
)
from xmlrpc.client import Boolean

from grpc import Call

def make_dir_function(
    dir_name= '',
    workspace=''
) -> Callable[..., Path]:
    """Generate a function that converts a string or iterable of strings into
    a path relative to the project directory.

    Args:
        dirname: Name of the subdirectories to extend the path of the main
            project.
            If an iterable of strings is passed as an argument, then it is
            collapsed to a single steing with anchors dependent on the
            operating system.
        
        workspace: Path of the workspace. If it is none, the folder in which the 
            file that is being executed is located is taken.

    Returns:
        A function that returns the path relative to a directory that can
        receive `n` number of arguments for expansion.
    """

    if workspace:
        workspace_path = Path(workspace).resolve()
    else:
        workspace_path = Path('.').resolve()

    def dir_path(*args) -> Path:
        if isinstance(dir_name, str):
            return workspace_path.joinpath(dir_name, *args)
        else:
            return workspace_path.joinpath(*dir_name, *args)

    return dir_path

project_dir = make_dir_function("")

for dir_type in [
        ["data"],
        ["data", "raw"],
        ["data", "processed"],
        ["data", "interim"],
        ["data", "external"],
        ["models"],
        ["notebooks"],
        ["references"],
        ["reports"],
        ["reports", "figures"]
    ]:
    dir_var = '_'.join(dir_type) + "_dir"
    exec(f"{dir_var} = make_dir_function({dir_type})")

from os import listdir
from os.path import (exists, isfile)

def is_valid(
    path: Path
) -> bool:
    """
    Function to Check if the path specified specified is an existent
    non empty directory
    """
    if exists(path) and not isfile(path):

        # Checking if the directory is empty or not
        if  len(listdir(path))!=0:
            return True
        else:
            return False
    
    else: 
        return False



from typing import (Callable, Sequence)
from distutils.dir_util import copy_tree
from os import listdir
from shutil import copy


def make_remote_copy_of_workspace_functions(
    local_path = '',
    remote_path = '', 
    notebook_path = '',
) -> Sequence[Callable[... , None]]:
    '''
    Generates functions to generate updates to files from another folder on the 
    same machine to the folder where this notebook is located, following the 
    project structure. 
    A different path can be specified for the current notebook folder.

    Args:
        remote_path:
        notebook_path:
        
    '''



    local_dir = make_dir_function(workspace=local_path)
    remote_dir = make_dir_function(workspace = remote_path)

    def update_from_remote(): 
        if local_dir() != remote_dir() and is_valid(remote_dir()):
            copy_tree(str(remote_dir()), str(local_dir()))
            print('The remote files have been copied to the local repository.')
        
        else:
            print('There is no need to copy the remote files, '+
                  'as the remote repository does not exist or is empty.')

    def update_to_remote():
        if local_dir() == remote_dir():
            return None

        exception_list = ['notebooks',  '.config', '.git', 'sample_data' ]
       
        local_file_names = listdir(local_dir())
        for name in local_file_names:
            if (local_dir(name).is_dir() and name not in exception_list):
                copy_tree(str(local_dir(name)),
                        str(remote_dir(name)))
    

    def update_notebook(to_remote=False):
        if not notebook_path:
            return None
    
        file_path = Path(notebook_path).resolve()

        if to_remote:
            copy(file_path, remote_dir('notebooks', file_path.name))
        else:
            copy(local_dir('notebooks', file_path.name), file_path)            
        
        print(f'The notebook {file_path.name} has been updated in the' + 
                ' remote folder')

    
    return local_dir, update_from_remote, update_to_remote, update_notebook






        


