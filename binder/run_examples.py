from subprocess import run
from glob import glob
import os.path as op

scripts = glob(op.join(op.dirname(op.abspath(__file__)), '..', 'examples', '*.py'))
for script in scripts:
    # Run each script, this will ensure data is downloaded in the built image and that the scripts work.
    call = ['python', script]
    print('Running script: {}'.format(script))
    run(call, check=True)
