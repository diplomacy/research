FROM pcpaquette/tensorflow-serving:20190226

# To build and push:
# sudo -s
# apt-get update -y
# apt-get install -y docker.io
# docker build -t gcr.io/ppaquette-diplomacy/web-deploy:7473ed4 --build-arg COMMIT_ID=7473ed4798a65c957b8b1a5eadbd876a307e00cf .
# gcloud auth configure-docker
# gcloud docker -- push gcr.io/ppaquette-diplomacy/web-deploy:7473ed4

ENV PYTHONIOENCODING=utf-8
ENV LANG=en_CA.UTF-8
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
ENV PYTHONUNBUFFERED=1
ENV WORKING_DIR=/work_dir
ENV DEBIAN_FRONTEND=noninteractive

ARG GITHUB_TOKEN
ARG REPO
ARG COMMIT_ID
USER root
RUN echo "------------------------------------------------------" && \
    echo "Installing dependencies (psmisc)" && \
    echo "------------------------------------------------------" && \
    apt-get -y update && \
    apt-get install -y psmisc software-properties-common wget curl && \
    \
    echo "------------------------------------------------------" && \
    echo "Installing Nginx" && \
    echo "------------------------------------------------------" && \
    add-apt-repository -y ppa:nginx/stable && \
    apt-get update && \
    apt-get install -y nginx && \
    echo "\ndaemon off;" >> /etc/nginx/nginx.conf && \
    chown -R www-data:www-data /var/lib/nginx && \
    \
    apt-get clean && \
    apt-get autoclean && \
    rm -Rf /tmp/pip_build_root && \
    rm -rf /var/lib/apt/lists/*

RUN echo "------------------------------------------------------" && \
    echo "Installing Diplomacy" && \
    echo "------------------------------------------------------" && \
    apt-get update -y && \
    mkdir -p /work_dir && \
    mkdir -p /data && \
    mkdir -p /data/certs && \
    rm -Rf /data/diplomacy && \
    rm -Rf /data/env3.7 && \
    cd /data && \
    git clone "https://$GITHUB_TOKEN:x-oauth-basic@github.com/$REPO.git" diplomacy && \
    wget -nv https://storage.googleapis.com/ppaquette-diplomacy/files/Miniconda3-py37.sh -O miniconda.sh && \
    bash ./miniconda.sh -b -p /data/env3.7 && \
    export PATH="/data/env3.7/bin:$PATH" && \
    pip install --upgrade pip && \
    cd diplomacy && \
    git checkout $COMMIT_ID && \
    .jenkins/expand_vars.sh && \
    pip install -r requirements.txt && \
    pip install -r requirements_dev.txt && \
    git checkout -- requirements.txt requirements_dev.txt && \
    ./run_install_nvm.sh && \
    cd /data && \
    \
    chmod -R 777 /data && \
    chmod -R 400 /data/certs && \
    chmod -R 777 /work_dir && \
    \
    echo "------------------------------------------------------" && \
    echo "Building launch script" && \
    echo "------------------------------------------------------" && \
    echo '#!/bin/bash' > /data/entry.sh && \
    echo 'export PATH="/data/env3.7/bin:$PATH"' >> /data/entry.sh && \
    echo "export DIPLOMACY_ROOT=`python -c 'import diplomacy; print(diplomacy.__path__[0])'`" >> /data/entry.sh && \
    echo 'cd /data/diplomacy' >> /data/entry.sh && \
    echo 'mkdir -p "$HOME/.cache/diplomacy"' >> /data/entry.sh && \
    echo 'wget -nv https://storage.googleapis.com/ppaquette-diplomacy/cache-jenkins/convoy_paths_cache.pkl -O "$HOME/.cache/diplomacy/convoy_paths_cache.pkl"' >> /data/entry.sh && \
    echo 'cd /work_dir' >> /data/entry.sh && \
    echo 'rm -f /work_dir/data/log_*.txt' >> /data/entry.sh && \
    echo 'python -m diplomacy.server.run |& tee -a /work_dir/data/log_server.txt &' >> /data/entry.sh && \
    echo 'python /data/diplomacy/diplomacy_research/scripts/launch_bot.py |& tee -a /work_dir/data/log_bot.txt &' >> /data/entry.sh && \
    echo 'cd $DIPLOMACY_ROOT/web/' >> /data/entry.sh && \
    echo 'export NVM_DIR="$HOME/.nvm"' >> /data/entry.sh && \
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> /data/entry.sh && \
    echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> /data/entry.sh && \
    echo 'npm run-script build' >> /data/entry.sh && \
    echo 'cd -' >> /data/entry.sh && \
    echo '/usr/sbin/nginx' >> /data/entry.sh && \
    \
    echo "------------------------------------------------------" && \
    echo "Cleaning up" && \
    echo "------------------------------------------------------" && \
    apt-get clean && \
    apt-get autoclean && \
    rm -Rf /tmp/pip_build_root && \
    rm -rf /var/lib/apt/lists/*

COPY default /etc/nginx/sites-available/default
WORKDIR /data/diplomacy
VOLUME ["/work_dir", "/data/certs"]
ENTRYPOINT ["/bin/bash", "/data/entry.sh"]

# Expose ports.
EXPOSE 80
EXPOSE 443
EXPOSE 8432
EXPOSE 8433
