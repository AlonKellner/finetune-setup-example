FROM berkeleyskypilot/skypilot-api-test:latest AS sky
FROM sky AS update
COPY .ssh /root/.ssh
WORKDIR /skypilot-original
RUN chmod -R 600 /root/.ssh && \
    git config --global user.name "AlonKellner" && \
    git config --global user.email "me@alonkellner.com" && \
    git clone https://github.com/skypilot-org/skypilot.git . && \
    git remote rm origin && \
    git remote add origin github.com-skypilot:AlonKellner/skypilot-beta.git && \
    git push --set-upstream origin master
WORKDIR /skypilot
RUN git remote rm origin && \
    git remote add origin github.com-skypilot:AlonKellner/skypilot-beta.git && \
    git push --set-upstream origin restapi
FROM sky AS image
WORKDIR /skypilot
RUN git remote rm origin && \
    git remote add origin https://github.com/AlonKellner/skypilot-beta.git && \
    git fetch && git branch -v -a && git switch custom && \
    pip install runpod
WORKDIR /
