# syntax=docker/dockerfile:1.7

FROM rust:1.75-bookworm AS build

WORKDIR /src
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bins -p tla-cli -p pnml-tools \
    && install -Dm755 target/release/ty /out/ty \
    && install -Dm755 target/release/tla /out/tla \
    && install -Dm755 target/release/pnml-tools /out/pnml-tools

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY --from=build /out/ty /usr/local/bin/ty
COPY --from=build /out/tla /usr/local/bin/tla
COPY --from=build /out/pnml-tools /usr/local/bin/pnml-tools

ENTRYPOINT ["/usr/local/bin/ty"]
CMD ["--help"]
