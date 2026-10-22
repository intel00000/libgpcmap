/* libgpcmap: which TPC ids share a GPC on this device.
 *
 * A mask bit of libsmctrl is one TPC (two SMs: 2t and 2t+1). The global TPC
 * id is the driver's numbering of the enabled TPCs; which ids sit in one GPC
 * differs per product and per chip and has to be read from the device:
 *   rm       the driver's own table through two non-privileged RM controls
 *            (built with RM=1 against the open-gpu-kernel-modules SDK of the
 *            installed driver; refused on MIG slices)
 *   cluster  thread-block clusters (Hopper and later): the CTAs of one
 *            cluster of 8 are co-resident in one GPC; the clusters seen
 *            refine the table (a cluster never spans two GPCs) and may leave
 *            TPCs unplaced, which gpcmap_cluster_fits() settles one by one
 *   corun    a streaming kernel slows a victim more inside its own GPC
 *            (Ampere; accepted only with a clear two-mode split)
 * Every answer is checked with the single-TPC %smid test (gpcmap_validate).
 * Stream masks come from the libsmctrl fork, loaded by path. */
#ifndef GPCMAP_H
#define GPCMAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define GPCMAP_MAX_TPCS 128
#define GPCMAP_MAX_GPCS 32

    typedef struct
    {
        int cuda_device;
        char name[96];
        char uuid[40];
        int num_sms;
        int num_tpcs;
        int num_gpcs;
        int gpc_of_tpc[GPCMAP_MAX_TPCS]; /* -1: unplaced */
        char method[16];                 /* rm | cluster | corun */
        int cluster_capacity;            /* clusters of 8 CTAs the device holds at once (cluster method) */
    } gpcmap_t;

    /* libsmctrl's stream masks: the library at smctrl_path, or the preloaded one
     * when NULL. 0, or errno (ENOENT: not found, ENOSYS: no stream-mask symbol). */
    int gpcmap_init(const char *smctrl_path);

    /* Confine a CUDA stream (a cudaStream_t) to the given TPCs; n == 0 clears. */
    void gpcmap_mask_stream(void *stream, const int *tpcs, int n);

    /* The map by "rm", "cluster", "corun" or "auto" (that order, the first that
     * answers). 0, or errno: ENOSYS (method not built or not supported here),
     * EAGAIN (no clear answer), EIO (CUDA error). */
    int gpcmap_query(int cuda_device, const char *method, gpcmap_t *out);

    /* Does one cluster of 8 CTAs run with only these TPCs enabled? 1 it ran,
     * 0 still pending after timeout_ms, <0 a CUDA error. A pending launch stays
     * pending in this CUDA context for good: call this in a child process. */
    int gpcmap_cluster_fits(int cuda_device, const int *tpcs, int n, int timeout_ms);

    /* With TPC t alone enabled a kernel lands on SMs 2t and 2t+1, both and no
     * other, for every t. Returns the number of failing TPCs, the first
     * max_failed of them in failed[]. */
    int gpcmap_validate(int cuda_device, int *failed, int max_failed);

    /* The device's name, uuid and SM count into *out (no probe). */
    int gpcmap_describe(int cuda_device, gpcmap_t *out);

    /* The map as JSON in the gpc_to_tpcs form. */
    int gpcmap_write_json(const gpcmap_t *map, const char *path);

    /* The RM method alone (gpcmap_rm.c); ENOSYS unless built with RM=1. */
    int gpcmap_rm_query(const char *uuid, gpcmap_t *out);

#ifdef __cplusplus
}
#endif
#endif
