FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install Docker CLI (purple agent runs containers to inspect repos)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg gosu && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password agent
RUN groupadd -f docker && usermod -aG docker agent
RUN chmod 777 /var/run

# Stay as root for entrypoint.sh (it drops to 'agent' via gosu)
WORKDIR /home/agent

COPY --chown=agent pyproject.toml uv.lock README.md ./
COPY --chown=agent src src
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
# uv sync needs to run as agent to put cache in the right place
USER agent
RUN uv sync --locked
USER root

ENTRYPOINT ["/entrypoint.sh", "uv", "run", "src/purple/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9022"]
EXPOSE 9022
