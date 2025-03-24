"""Will be an entrypoint for job training."""

from datetime import datetime, timedelta

start = datetime.now()
counter = 0
while datetime.now() - start < timedelta(seconds=10):
    counter += 1
print(counter)
