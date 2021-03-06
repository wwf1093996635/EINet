# useful commands

# python cmd.py --task copy --path ./Instances/TP-1  //move necessary files for training and analysis to path.

import os
import shutil
import argparse
import warnings

parser = argparse.ArgumentParser(description='Parse args.')
parser.add_argument('--device', type=str, default="None", help='device')
parser.add_argument('--task', type=str, default=None, help='task to do')
parser.add_argument('--path', type=str, default=None, help='dest to copy files')
args = parser.parse_args()


file_list = [
    'cmd.py',
    'Models',
    'Agent.py',
    'Arenas.py',
    'Trainers.py',
    'Optimizers',
    'Analyzer.py',
    'config.py',
    'main.py',
    'config.py',
    'utils_agent.py',
    'utils_arena.py',
    'utils_anal.py',
    'config_sys.py',
]

def copy_files(file_list, path, sys_type='linux'):
    if not dest.endswith('/'):
        dest += '/'

    if sys_type in ['linux']:
        for file in file_list:
            #shutil.copy2(file, dest + file)
            if os.path.exists(path+file):
                os.system('rm -r %s'%(path+file))
            os.system('cp -r %s %s'%(file, dest+file))
    elif sys_type in ['windows']:
        # to be implemented 
        pass
    else:
        raise Exception('copy_files: Invalid sys_type: '%str(sys_type))

if __name__ == '__main__':
    if args.task in ['copy files', 'copy']:
        copy_files(args)
    elif args.task is None:
        warnings.warn('warning: task is None. do nothing.')
    else:
        raise Exception('Invalid task: %s'%str(args.task))

