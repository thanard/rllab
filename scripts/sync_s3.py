import sys
sys.path.append('.')
from rllab import config
import os
import argparse
import ast

def sync(folder, local_dir, remote_dir):
    if '?' in folder:
        print('?', folder)
        for i in range(0,10):
            sync(folder.replace('?', str(i), 1), local_dir, remote_dir)
    else:
        print('#',folder)
        if folder:
            remote_dir = os.path.join(remote_dir, folder)
            local_dir = os.path.join(local_dir, folder)
        # If exist it returns 0
        if not os.system('aws s3 ls {remote_dir}'.format(remote_dir=remote_dir)):
            if args.bare:
                command = ("""
                    aws s3 sync {remote_dir} {local_dir} --exclude '*' --include '*.csv' --include '*.json' --content-type "UTF-8"
                """.format(local_dir=local_dir, remote_dir=remote_dir))
            else:
                command = ("""
                    aws s3 sync {remote_dir} {local_dir} --exclude '*stdout.log' --exclude '*stdouterr.log' --content-type "UTF-8"
                """.format(local_dir=local_dir, remote_dir=remote_dir))
            if args.dry:
                print(command)
            else:
                os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    args = parser.parse_args()
    remote_dir = config.AWS_S3_PATH
    local_dir = os.path.join(config.LOG_DIR, "s3")

    sync(args.folder, local_dir, remote_dir)
