FROM berkeleyskypilot/skypilot-nightly:latest AS sky
FROM sky AS update
COPY . /root/.ssh
WORKDIR /skypilot-original
RUN chmod -R 600 /root/.ssh && \
    git config --global user.name "AlonKellner" && \
    git config --global user.email "me@alonkellner.com" && \
    git clone https://github.com/skypilot-org/skypilot.git . && \
    git fetch --all --tags --prune && \
    git checkout tags/v0.9.3 -b stable-master && \
    git remote rm origin && \
    git remote add origin git@github.com:AlonKellner/skypilot-beta.git && \
    git push --set-upstream origin stable-master:stable-master --force
FROM sky AS image
RUN pip uninstall -y skypilot-nightly && \
    pip install "skypilot[kubernetes,runpod,gcp] @ git+https://github.com/AlonKellner/skypilot-beta.git@custom-stable"
