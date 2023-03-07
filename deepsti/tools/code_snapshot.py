import os
import shutil

def not_in(s, l):
    for x in l:
        if x in s: return False
    return True

def code_snapshot(dstDir, argv):
    os.makedirs(dstDir, exist_ok=True)
    with open(os.path.join(dstDir, 'cmd_snapshot.txt'), 'w') as f:
        f.writelines(' '.join(argv))
    
    extension = '.py'
    exclude = ['.ipynb_checkpoints']
    srcDir = '.'
    for root, dirs, files in os.walk(srcDir):
        for file_ in files:
            if file_.endswith(extension) and not_in(root, exclude) and (len(root.split('/')) == 1 or root.split('/')[1] != 'snapshots'):
                relpath = os.path.relpath(root, srcDir)
                os.makedirs(os.path.join(dstDir, relpath), exist_ok=True)
                shutil.copy(os.path.join(root, file_), os.path.join(dstDir, relpath, file_))


