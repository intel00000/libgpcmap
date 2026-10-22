/* gpcmap: print and save the GPC map of a device.
 *
 *   gpcmap [--device N] [--smctrl PATH] [--method auto|rm|cluster|corun]
 *          [--json FILE] [--no-validate] [--no-complement] [--timeout-ms T]
 *
 * After the cluster method, each TPC the clusters left unplaced is tried
 * against the GPCs found, one child process per try: a cluster of 8 CTAs on
 * the TPC plus three TPCs of a GPC runs only when they share the GPC, and a
 * try that finds no room pends for good, so it runs in a child that is
 * killed on timeout. Candidates are ordered by the id stride of each GPC. */
#include "gpcmap.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static const char *g_smctrl = NULL;
static int g_device = 0;

static int usage(void)
{
    fprintf(stderr, "usage: gpcmap [--device N] [--smctrl PATH] [--method auto|rm|cluster|corun] [--json FILE]\n"
                    "              [--no-validate] [--no-complement] [--timeout-ms T]\n");
    return 2;
}

static int child_fits(int s, const int *g, int n_g, int timeout_ms)
{
    /* 1 fits, 0 pends, -1 error */
    pid_t pid = fork();
    if (pid < 0)
        return -1;
    if (pid == 0)
    {
        char dev[16], to[16], set[64];
        snprintf(dev, sizeof(dev), "%d", g_device);
        snprintf(to, sizeof(to), "%d", timeout_ms);
        snprintf(set, sizeof(set), "%d,%d,%d,%d", s, g[0], g[1], g[2]);
        (void)n_g;
        execl("/proc/self/exe", "gpcmap", "--device", dev, "--smctrl", g_smctrl ? g_smctrl : "", "--timeout-ms", to,
              "--fits", set, (char *)NULL);
        _exit(127);
    }
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int status = 0;
    for (;;)
    {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid)
            break;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long ms = (now.tv_sec - t0.tv_sec) * 1000L + (now.tv_nsec - t0.tv_nsec) / 1000000L;
        if (ms > timeout_ms + 15000)
        {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            return 0;
        }
        usleep(5000);
    }
    if (!WIFEXITED(status))
        return 0;
    int code = WEXITSTATUS(status);
    return code == 0 ? 1 : code == 3 ? 0
                                     : -1;
}

static int stride_of(const gpcmap_t *m, int g)
{
    /* the common difference between consecutive member ids */
    int counts[GPCMAP_MAX_TPCS] = {0}, prev = -1, best = 1, best_n = 0;
    for (int t = 0; t < m->num_tpcs; ++t)
        if (m->gpc_of_tpc[t] == g)
        {
            if (prev >= 0)
                counts[t - prev]++;
            prev = t;
        }
    for (int d = 1; d < GPCMAP_MAX_TPCS; ++d)
        if (counts[d] > best_n)
        {
            best_n = counts[d];
            best = d;
        }
    return best;
}

static void complement(gpcmap_t *m, int timeout_ms)
{
    for (int s = 0; s < m->num_tpcs; ++s)
    {
        if (m->gpc_of_tpc[s] >= 0)
            continue;
        /* candidate GPCs, best id-pattern fit first */
        int order[GPCMAP_MAX_GPCS], score[GPCMAP_MAX_GPCS];
        for (int g = 0; g < m->num_gpcs; ++g)
        {
            int d = stride_of(m, g), nearest = 1 << 20, phase = 1;
            for (int t = 0; t < m->num_tpcs; ++t)
                if (m->gpc_of_tpc[t] == g)
                {
                    int dist = abs(t - s);
                    if (dist < nearest)
                        nearest = dist;
                    if (dist % d == 0)
                        phase = 0;
                }
            order[g] = g;
            score[g] = phase * 100000 + nearest;
        }
        for (int i = 0; i < m->num_gpcs; ++i)
            for (int j = i + 1; j < m->num_gpcs; ++j)
                if (score[order[j]] < score[order[i]])
                {
                    int tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
        for (int i = 0; i < m->num_gpcs && m->gpc_of_tpc[s] < 0; ++i)
        {
            int g = order[i], three[3], n = 0;
            for (int t = 0; t < m->num_tpcs && n < 3; ++t)
                if (m->gpc_of_tpc[t] == g && t != s)
                    three[n++] = t;
            if (n < 3)
                continue;
            int r = child_fits(s, three, 3, timeout_ms);
            fprintf(stderr, "gpcmap: TPC %d with GPC %d (TPCs %d, %d, %d): %s\n", s, g, three[0], three[1], three[2],
                    r == 1 ? "a cluster runs: same GPC" : r == 0 ? "pends: not this GPC"
                                                                 : "error");
            if (r == 1)
                m->gpc_of_tpc[s] = g;
        }
    }
}

static void print_map(const gpcmap_t *m, FILE *f)
{
    fprintf(f, "%s (cuda %d, uuid %s): %d SMs, %d TPCs, %d GPCs by %s%s\n", m->name, m->cuda_device, m->uuid,
            m->num_sms, m->num_tpcs, m->num_gpcs, m->method, m->cluster_capacity ? "" : "");
    if (m->cluster_capacity)
        fprintf(f, "clusters of 8 CTAs held at once: %d\n", m->cluster_capacity);
    for (int g = 0; g < m->num_gpcs; ++g)
    {
        int n = 0;
        fprintf(f, "  GPC %2d:", g);
        for (int t = 0; t < m->num_tpcs; ++t)
            if (m->gpc_of_tpc[t] == g)
            {
                fprintf(f, " %d", t);
                n++;
            }
        fprintf(f, "   (%d TPCs)\n", n);
    }
    int unplaced = 0;
    for (int t = 0; t < m->num_tpcs; ++t)
        unplaced += m->gpc_of_tpc[t] < 0;
    if (unplaced)
    {
        fprintf(f, "  unplaced:");
        for (int t = 0; t < m->num_tpcs; ++t)
            if (m->gpc_of_tpc[t] < 0)
                fprintf(f, " %d", t);
        fprintf(f, "\n");
    }
}

int main(int argc, char **argv)
{
    const char *method = "auto", *json = NULL, *fits = NULL;
    int validate = 1, do_complement = 1, timeout_ms = 3000;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--device") && i + 1 < argc)
            g_device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--smctrl") && i + 1 < argc)
            g_smctrl = argv[++i];
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
            method = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json = argv[++i];
        else if (!strcmp(argv[i], "--timeout-ms") && i + 1 < argc)
            timeout_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fits") && i + 1 < argc)
            fits = argv[++i];
        else if (!strcmp(argv[i], "--no-validate"))
            validate = 0;
        else if (!strcmp(argv[i], "--no-complement"))
            do_complement = 0;
        else
            return usage();
    }
    if (g_smctrl && !*g_smctrl)
        g_smctrl = NULL;
    int rc = gpcmap_init(g_smctrl);
    if (rc)
    {
        fprintf(stderr, "gpcmap: libsmctrl %s: %s\n", g_smctrl ? g_smctrl : "(preloaded)", strerror(rc));
        return 1;
    }
    if (fits)
    { /* the child: 0 ran, 3 pending */
        int tpcs[4], n = 0;
        char *copy = strdup(fits);
        for (char *tok = strtok(copy, ","); tok && n < 4; tok = strtok(NULL, ","))
            tpcs[n++] = atoi(tok);
        int r = gpcmap_cluster_fits(g_device, tpcs, n, timeout_ms);
        _exit(r == 1 ? 0 : r == 0 ? 3
                                  : 1);
    }
    gpcmap_t map;
    rc = gpcmap_query(g_device, method, &map);
    if (rc)
    {
        gpcmap_describe(g_device, &map);
        fprintf(stderr, "gpcmap: %s on %s: %s\n", method, map.name, strerror(rc));
        return 1;
    }
    if (do_complement && !strcmp(map.method, "cluster"))
        complement(&map, timeout_ms);
    print_map(&map, stdout);
    if (validate)
    {
        int failed[16];
        int bad = gpcmap_validate(g_device, failed, 16);
        if (bad < 0)
        {
            fprintf(stderr, "gpcmap: the single-TPC check could not run\n");
            return 1;
        }
        printf("single-TPC check: %d of %d TPCs map to exactly their two SMs%s\n", map.num_tpcs - bad, map.num_tpcs,
               bad ? " (FAILED)" : "");
        if (bad)
        {
            printf("  failing TPCs:");
            for (int i = 0; i < bad && i < 16; ++i)
                printf(" %d", failed[i]);
            printf("\n");
        }
        if (bad)
            rc = 1;
    }
    if (json)
    {
        int w = gpcmap_write_json(&map, json);
        if (w)
        {
            fprintf(stderr, "gpcmap: %s: %s\n", json, strerror(w));
            return 1;
        }
        printf("wrote %s\n", json);
    }
    return rc;
}
