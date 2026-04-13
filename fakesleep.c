#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <string.h>

#define FAKETIMERC "/etc/faketimerc"

static void advance_time(long seconds) {
    if (seconds <= 0) return;

    FILE *f = fopen(FAKETIMERC, "r");
    if (!f) return;

    char buf[64] = {0};
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return; }
    fclose(f);

    buf[strcspn(buf, "\n")] = 0;
    if (buf[0] != '@') return;

    struct tm t = {0};
    if (sscanf(buf + 1, "%d-%d-%d %d:%d:%d",
               &t.tm_year, &t.tm_mon, &t.tm_mday,
               &t.tm_hour, &t.tm_min, &t.tm_sec) != 6) return;

    t.tm_year -= 1900;
    t.tm_mon  -= 1;
    t.tm_sec  += (int)seconds;
    t.tm_isdst = -1;

    time_t epoch = timegm(&t);
    struct tm *nt = gmtime(&epoch);
    if (!nt) return;

    f = fopen(FAKETIMERC, "w");
    if (!f) return;
    fprintf(f, "@%04d-%02d-%02d %02d:%02d:%02d",
            nt->tm_year + 1900, nt->tm_mon + 1, nt->tm_mday,
            nt->tm_hour, nt->tm_min, nt->tm_sec);
    fclose(f);
}

int nanosleep(const struct timespec *req, struct timespec *rem) {
    if (req) {
        long secs = req->tv_sec + (req->tv_nsec >= 500000000L ? 1 : 0);
        advance_time(secs);
    }
    if (rem) { rem->tv_sec = 0; rem->tv_nsec = 0; }
    return 0;
}

int clock_nanosleep(clockid_t clockid, int flags,
                    const struct timespec *req, struct timespec *rem) {
    (void)clockid; (void)flags;
    return nanosleep(req, rem);
}
