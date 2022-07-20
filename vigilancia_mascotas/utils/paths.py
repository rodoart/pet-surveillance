from pathlib import Path
from typing import (
    Callable
)

from shutil import copyfile
from distutils.dir_util import copy_tree, mkpath
from os import name
from pathlib import Path, WindowsPath, PosixPath

from warnings import warn




def make_dir_function(
    dir_name:str  = '',
    workspace:str =''
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
        
        files = listdir(path)

        # Checking if the directory is empty or not
        if  len(files)!=0:
            files = [file_name for file_name in files if not file_name.endswith('.ini')]

            if len(files)!=0:
                return True
            else:
                return False
        else:
            return False
    
    else: 
        return False


if name == 'posix':
    _base = PosixPath
else:
    _base = WindowsPath

class RelativePath(_base):
    def __init__(self, workspace, *args) -> None:                             
        super().__init__()
        self.workspace = workspace

    def is_valid(self):
        return is_valid(self)
        

class TwoWorkspacePath():     

    def __init__(self, *args, local_workspace, remote_workspace='') -> None:
        self.__args = args
        self.local = local_workspace   
        self.remote = remote_workspace

    @property
    def local(self):
        return self.__local        

    @local.setter
    def local(self, workspace):
        if type(workspace) == str:
            workspace =  Path(workspace).resolve()   

        self.__local = RelativePath(workspace, *self.__args)


    @property
    def remote(self):
        
        return self.__remote

    @remote.setter
    def remote(self, workspace):
        
        if not workspace:
            self.__remote = ''
            return None

        
        if type(workspace) == str:
            workspace =  Path(workspace).resolve()   

        self.__remote = RelativePath(workspace, *self.__args)

        return None

    @property
    def relative(self):
        return self.local.relative_to(self.local.workspace)


    def copy_to(self, destiny):
        if not self.remote:
            warn('No remote directory provided so files cannot be copied.')
            return None

        if destiny == 'remote':
            src = self.local
            dst = self.remote
        else:
            src = self.remote
            dst = self.local

        if not dst.parent.is_dir():
            mkpath(str(dst.parent))

        if src.is_file():          
            copyfile(str(src), str(dst))
        elif src.is_dir():
            copy_tree(src = str(src), dst = str(dst))

    def upload(self):
        self.copy_to('remote')
    
    def download(self):
        self.copy_to('local')

    def copy(self):
        return TwoWorkspacePath(
            local_workspace = self.local.workspace,
            remote_workspace = self.remote.workspace,
            *self.__args
        )

    def joinpath(self, *path_labels):
        twp = self.copy()
        twp.__local = twp.__local.joinpath(*path_labels)
        twp.__remote = twp.__remote.joinpath(*path_labels)
        
        return twp

    def __str__(self):
        return str(self.relative)

def make_two_dir_function(local_workspace, remote_workspace=''):
    def fun(*args):
        return TwoWorkspacePath(*args, local_workspace=local_workspace,
                         remote_workspace=remote_workspace)
    
    return fun

        


