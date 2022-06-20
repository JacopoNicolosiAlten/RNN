# parameters
overwrite_dataset=False

import sys
pipeline_path = r"C:\Users\jnicolosi\Desktop\tesi\codes"
sys.path.append(pipeline_path)

# import pipeline steps
import generate_dataset
import RNN_preprocess as preprocess

if(overwrite_dataset):
    print('\ngenerating dataset...\n')
    #generate_dataset.help()
    generate_dataset.main()

print('\npreprocessing...\n')
#preprocess.help()
preprocess.main()

print('\ndone.')