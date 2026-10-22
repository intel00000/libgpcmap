/* libgpcmap: which TPC ids share a GPC on this device.
 *
 * A libsmctrl mask bit is one TPC (SMs 2t and 2t+1). The global TPC id is the
 * driver's numbering of the enabled TPCs; which ids share a GPC is read from
 * the device:
 *   rm       the driver's table through two RM controls (build with RM=1
 *            against the open-gpu-kernel-modules SDK; refused on MIG)
 *   cluster  thread-block clusters (Hopper and later): the CTAs of one
 *            cluster are co-resident in one GPC; TPCs the clusters leave
 *            unplaced are settled by gpcmap_cluster_fits()
 *   corun    streaming kernels slow each other more inside one GPC
 * Every answer is checked by the single-TPC %smid test (gpcmap_validate). */
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
        int corun_conflicts;             /* co-run merges refused for exceeding a GPC; the command exits 2 */
    } gpcmap_t;

    /* Load libsmctrl (NULL: the preloaded one). 0, or ENOENT / ENOSYS. */
    int gpcmap_init(const char *smctrl_path);

    /* Confine a cudaStream_t to these TPCs; n == 0 clears the mask. */
    void gpcmap_mask_stream(void *stream, const int *tpcs, int n);

    /* The map by "rm", "cluster", "corun" or "auto" (that order). 0, or ENOSYS
     * (not built or not supported here), EAGAIN (no clear answer), EIO. */
    int gpcmap_query(int cuda_device, const char *method, gpcmap_t *out);

    /* Does one cluster of 8 CTAs run with only these TPCs enabled? 1 ran,
     * 0 pending after timeout_ms (and pending for good: call it in a child
     * process), <0 a CUDA error. */
    int gpcmap_cluster_fits(int cuda_device, const int *tpcs, int n, int timeout_ms);

    /* For every t, a kernel on TPC t alone lands on SMs 2t and 2t+1, both and
     * no other. Returns the number of failing TPCs, the first ones in failed[]. */
    int gpcmap_validate(int cuda_device, int *failed, int max_failed);

    /* Name, uuid and SM count, no probe. */
    int gpcmap_describe(int cuda_device, gpcmap_t *out);

    /* The map as JSON in the gpc_to_tpcs form. */
    int gpcmap_write_json(const gpcmap_t *map, const char *path);

    /* gpcmap_rm.c; ENOSYS unless built with RM=1. */
    int gpcmap_rm_query(const char *uuid, gpcmap_t *out);

#ifdef __cplusplus
}
#endif
#endif
