FROM ghcr.io/prefix-dev/pixi:0.82.0

WORKDIR /app

# Copy everything needed to install (see .dockerignore for what is excluded)
COPY . .

# Install the lean test environment exactly as locked; cache downloads between builds
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --locked -e test

# Default: regenerate all paper outputs into /app/outputs
CMD ["pixi", "run", "-e", "test", "paper"]
