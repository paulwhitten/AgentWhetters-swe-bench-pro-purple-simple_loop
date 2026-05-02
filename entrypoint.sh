#!/bin/sh
# Match the container's docker group GID to the mounted socket's GID
# so the non-root 'agent' user can access Docker.
SOCK=/var/run/docker.sock
if [ -S "$SOCK" ]; then
    SOCK_GID=$(stat -c '%g' "$SOCK")
    groupmod -g "$SOCK_GID" docker 2>/dev/null || true
fi

exec gosu agent "$@"
