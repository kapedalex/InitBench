FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y gcc
COPY fakesleep.c /tmp/fakesleep.c
RUN gcc -shared -fPIC -o /usr/lib/libfakesleep.so /tmp/fakesleep.c \
    && strip /usr/lib/libfakesleep.so

FROM python:3.12-slim
RUN apt-get update && apt-get install -y faketime
COPY --from=builder /usr/lib/libfakesleep.so /usr/lib/libfakesleep.so
RUN echo -n "@2026-03-21 12:00:00" > /etc/faketimerc
# fakesleep must be first so its nanosleep takes priority over libfaketime's
RUN echo /usr/lib/libfakesleep.so > /etc/ld.so.preload \
    && find /usr/lib -name libfaketime.so.1 >> /etc/ld.so.preload

CMD ["sleep", "infinity"]
