"""Fine_Tune_MMS_on_Common_Voice.

Original file is located at
    https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_MMS_on_Common_Voice.ipynb
"""

from finetune_setup_example.job_utils import run_local_job
from finetune_setup_example.mains.mms_common_voice_adaptune import main

if __name__ == "__main__":
    job_func = main

    run_local_job(main)
