"""Fine_Tune_MMS_on_Common_Voice.

Original file is located at
    https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_MMS_on_Common_Voice.ipynb
"""

from pathlib import Path

import dotenv

from finetune_setup_example.local_job_utils import run_local_job
from finetune_setup_example.mains.mms_common_voice_adaptune import main

if __name__ == "__main__":
    current_file = Path(__file__)
    env_file = current_file.parent / f"{current_file.stem}.env"
    if env_file:
        print(f"Loading environment variables from {env_file}")
        dotenv.load_dotenv(env_file)
    run_local_job(main)
