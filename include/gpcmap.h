/* libgpcmap: which TPC ids share a GPC on this device.
 *
 * A libsmctrl mask bit is one TPC (SMs 2t and 2t+1). The global TPC id is the
 * driver's numbering of the enabled TPCs; which ids share a GPC is read from
 * the device:
 *   rm       the driver's table through two RM controls (build with RM=1
 *            against the open-gpu-kernel-modules SDK; refused on MIG)
 *   nvdebug  the same table from the work distributor's registers through
 *            the nvdebug kernel module's /proc/gpuN files, when loaded
 *   cluster  thread-block clusters (Hopper and later): the CTAs of one
 *            cluster are co-resident in one GPC; TPCs the clusters leave
 *            unplaced are settled by gpcmap_cluster_fits()
 *   corun    the groups of TPCs that share a path to memory (the GPCs on
 *            GB202, sub-groups of a GPC on GA100): a map in doubt, exit 2
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
        int gpc_of_tpc[GPCMAP_MAX_TPCS];         /* -1: unplaced */
        int num_virtual_gpcs;                    /* 0: the method does not know the virtual grouping */
        int virtual_gpc_of_tpc[GPCMAP_MAX_TPCS]; /* the virtual GPC: where clusters are placed; -1: unknown */
        char method[16];                         /* rm | cluster | corun */
        int cluster_capacity;                    /* clusters of 8 CTAs the device holds at once (cluster method) */
        int corun_conflicts;                     /* co-run merges refused for exceeding a GPC; the command exits 2 */
    } gpcmap_t;

    /* Load libsmctrl (NULL: the preloaded one). 0, or ENOENT / ENOSYS. */
    int gpcmap_init(const char *smctrl_path);

    /* Confine a cudaStream_t to these TPCs; n == 0 clears the mask. */
    void gpcmap_mask_stream(void *stream, const int *tpcs, int n);

    /* The map by "rm", "nvdebug", "cluster", "corun" or "auto" (that order). 0,
     * or ENOSYS (not built or not supported here), ENOENT (nvdebug not loaded),
     * EBUSY (MPS is running: %smid is emulated per client, no probe can run),
     * EAGAIN (no clear answer), EIO. */
    int gpcmap_query(int cuda_device, const char *method, gpcmap_t *out);

    /* Launch one cluster of 8 CTAs with only these TPCs enabled and poll it
     * for timeout_ms. 1: it ran, so the TPCs share a virtual GPC. 0: still
     * pending. A cluster with no room is never scheduled and never fails:
     * CUDA does not time it out and the launch cannot be cancelled, so the
     * calling context keeps it for good; call this from a process you can
     * discard. <0: a CUDA error. */
    int gpcmap_cluster_fits(int cuda_device, const int *tpcs, int n, int timeout_ms);

    /* For every t, a kernel on TPC t alone lands on SMs 2t and 2t+1, both and
     * no other. Returns the number of failing TPCs, the first ones in failed[]. */
    int gpcmap_validate(int cuda_device, int *failed, int max_failed);

    /* Name, uuid and SM count, no probe. */
    int gpcmap_describe(int cuda_device, gpcmap_t *out);

    /* The map as JSON in the gpc_to_tpcs form. */
    int gpcmap_write_json(const gpcmap_t *map, const char *path);

    /* What gpc_to_tpcs holds: "gpc" (physical GPCs, the memory path), "virtual_gpc"
     * (the cluster method: where clusters are placed) or "shared_path" (co-run). */
    const char *gpcmap_groups(const gpcmap_t *map);

    /* gpcmap_rm.c; ENOSYS unless built with RM=1. */
    int gpcmap_rm_query(const char *uuid, gpcmap_t *out);

    /* nvdebug's /proc/gpuN files (the module numbers GPUs in PCI order). */
    int gpcmap_nvdebug_query(int cuda_device, gpcmap_t *out);

    /* Is an MPS control daemon reachable (the way nvidia-cuda-mps-control checks)? */
    int gpcmap_mps_running(void);

    int gpcmap_rm_query(const char *uuid, gpcmap_t *out);

#ifdef __cplusplus
}
#endif
#endif
