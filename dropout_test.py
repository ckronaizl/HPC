"""
This file is for testing various settings with the dropout.py file
1) What is an optimal epoch?
2) What is an optimal number of layers?
3) What is an optimal number of nodes per layer?
"""

import subprocess
from subprocess import PIPE

def modify_epoch(epoch):
    file_path = "./dropout.py"
    subprocess.call(["python3", f"{file_path}", f"{epoch}"], stdout=PIPE)


modify_epoch(60)
