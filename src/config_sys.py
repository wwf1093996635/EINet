import sys
import re
def get_sys_type():
    if re.match(r'win', sys.platform) is not None:
        sys_type = 'windows'
    elif re.match(r"linux", sys.platform) is not None:
        sys_type = 'linux'
    else:
        sys_type = 'unknown'
    return sys_type

def get_libs_path():
    return {
        'WWF-PC': 'A:/Software_Projects/Libs/',
        'srthu2': '/data4/wangweifan/Libs/',
    }

def init():
    # import paths of environment modules
    sys_type = get_sys_type()
    libs_path = get_libs_path()
    if sys_type in ['windows']:
        sys.path.append(libs_path["WWF-PC"])
    elif sys_type in ['linux']:
        sys.path.append(libs_path["srthu2"])
    else:
        raise Exception("Cannot add Libs path. Unknown system type.")
    
    # import paths.
    paths = []
    '''
    paths.append("./Models/")
    paths.append("./Optimizers")
    '''
    for path in paths:
        sys.path.append(path)
init()