"""An example script to try with launch options."""

import time

import numpy as np

if __name__ == "__main__":
    print("Counting...")
    for i in np.arange(10):
        time.sleep(1)
        print(f"{i}...")
    print("Working!")
