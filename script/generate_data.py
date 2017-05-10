import subprocess
import time
import cPickle as pickle
import sys


# Setup logging
import logging
LOG_FILENAME = 'lifespan.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
        
        
sample_size = 32

for sample in range(sample_size):
  fname = "neural_predictions500_99276_" + str(sample+20) + ".p"
  cmd = ["python neural_model.py " + fname]
  p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=sys.stdout, shell=True)
  out, err = p.communicate() 
  logging.debug(time.strftime('%a %H:%M:%S') + "::  " + 'Done with sample ' +  str(sample))

