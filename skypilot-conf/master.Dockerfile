FROM berkeleyskypilot/skypilot-nightly:latest AS sky-nightly
FROM sky-nightly AS update
COPY . /root/.ssh
WORKDIR /skypilot-original
RUN chmod -R 600 /root/.ssh && \
    git config --global user.name "AlonKellner" && \
    git config --global user.email "me@alonkellner.com" && \
    git clone https://github.com/skypilot-org/skypilot.git . && \
    git fetch --all --tags --prune && \
    git checkout tags/v0.9.2 -b stable-master && \
    git remote rm origin && \
    git remote add origin git@github.com:AlonKellner/skypilot-beta.git && \
    git push --set-upstream origin stable-master:stable-master --force
FROM berkeleyskypilot/skypilot-api-test:latest AS sky-api
FROM sky-api AS image
WORKDIR /skypilot
RUN git remote rm origin && \
    git remote add origin https://github.com/AlonKellner/skypilot-beta.git && \
    git fetch && git branch -v -a && git switch stable-master && \
    pip install .
WORKDIR /
