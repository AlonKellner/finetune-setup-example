"""An example script to try with launch options."""

import time

if __name__ == "__main__":
    print("Counting...")
    for i in range(10):
        time.sleep(1)
        print(f"{i}...")
    print("Working!")
