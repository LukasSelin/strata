# Herbie at a pinned tag, for amd64 and arm64. This follows Herbie's own
# Dockerfile, except that Racket comes from its native installer rather
# than racket/racket: that image and uwplse/herbie are amd64-only, and
# Racket CS does not start under Rosetta on Apple silicon.
ARG HERBIE_TAG=v2.3
ARG RACKET_VERSION=8.18

FROM rust:1.88.0 AS builder
ARG HERBIE_TAG
WORKDIR /herbie
RUN git clone --depth 1 --branch "$HERBIE_TAG" https://github.com/herbie-fp/herbie.git .
RUN cargo build --release --manifest-path=egg-herbie/Cargo.toml
RUN cargo install --locked --git https://github.com/egraphs-good/egglog-experimental egglog-experimental

FROM debian:bookworm-slim
ARG RACKET_VERSION
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgmp10 libmpfr6 libssl3 sqlite3 \
 && rm -rf /var/lib/apt/lists/*
RUN arch=$(uname -m) \
 && curl -fsSL -o /tmp/racket.sh \
      "https://download.racket-lang.org/installers/${RACKET_VERSION}/racket-${RACKET_VERSION}-${arch}-linux-buster-cs.sh" \
 && sh /tmp/racket.sh --create-dir --unix-style --dest /usr/ \
 && rm /tmp/racket.sh
COPY --from=builder /herbie/egg-herbie /src/egg-herbie
RUN raco pkg install --no-docs /src/egg-herbie
COPY --from=builder /usr/local/cargo/bin/egglog-experimental /usr/local/bin/egglog-experimental
COPY --from=builder /herbie/src /src/herbie
RUN raco pkg install --no-docs --auto /src/herbie
ENTRYPOINT ["racket", "/src/herbie/main.rkt"]
