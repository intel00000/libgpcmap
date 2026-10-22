/* The driver's SM topology through two RM controls
 * (NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER, ..._GET_SM_TO_GPC_TPC_MAPPINGS).
 * The ABI is the open-gpu-kernel-modules tree of the installed driver:
 *   make RM=1 NV=/path/to/open-gpu-kernel-modules
 * Refused on MIG slices. */
#include "gpcmap.h"

#include <errno.h>

#ifndef GPCMAP_RM
int gpcmap_rm_query(const char *uuid, gpcmap_t *out)
{
    (void)uuid;
    (void)out;
    return ENOSYS;
}
#else
#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvos.h"
#include "nv_escape.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl2080/ctrl2080gr.h"
#include "nv-ioctl.h"
#include "nv-ioctl-numbers.h"

/* /proc/driver/nvidia/gpus/<pci>/information: the device minor of the GPU
 * whose uuid (without the GPU- prefix) start like the given uuid. */
static int device_minor(const char *uuid)
{
    DIR *d = opendir("/proc/driver/nvidia/gpus");
    if (!d)
        return -1;
    struct dirent *e;
    int minor = -1;
    while (minor < 0 && (e = readdir(d)))
    {
        if (e->d_name[0] == '.')
            continue;
        char path[512];
        snprintf(path, sizeof(path), "/proc/driver/nvidia/gpus/%s/information", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        char line[256], seen[64] = {0};
        int m = -1;
        while (fgets(line, sizeof(line), f))
        {
            // first 8 characters of the line to "GPU UUID"
            if (!strncmp(line, "GPU UUID", 8))
            {
                char *c = strchr(line, ':');
                if (c)
                {
                    c++;
                    while (*c == ' ' || *c == '\t')
                        c++;
                    if (!strncmp(c, "GPU-", 4))
                        c += 4;
                    snprintf(seen, sizeof(seen), "%s", c);
                    seen[strcspn(seen, "\r\n ")] = 0;
                }
            }
            else if (!strncmp(line, "Device Minor", 12))
            {
                char *c = strchr(line, ':');
                if (c)
                    m = atoi(c + 1);
            }
        }
        fclose(f);
        if (m >= 0 && seen[0] && !strncasecmp(seen, uuid, 8))
            minor = m;
    }
    closedir(d);
    return minor;
}

static int rm_alloc_root(int fd, NvHandle *client)
{
    NVOS21_PARAMETERS p = {0};
    p.hClass = NV01_ROOT;
    int rc = ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21_PARAMETERS), &p);
    if (rc || p.status)
        return -1;
    *client = p.hObjectNew;
    return 0;
}

static int rm_alloc64(int fd, NvHandle client, NvHandle parent, NvHandle object, NvU32 class_id, void *params)
{
    NVOS64_PARAMETERS p = {0};
    p.hRoot = client;
    p.hObjectParent = parent;
    p.hObjectNew = object;
    p.hClass = class_id;
    p.pAllocParms = (NvP64)(uintptr_t)params;
    int rc = ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS64_PARAMETERS), &p);
    return (rc || p.status) ? -1 : 0;
}

static int rm_control(int fd, NvHandle client, NvHandle object, NvU32 cmd, void *params, NvU32 size)
{
    NVOS54_PARAMETERS p = {0};
    p.hClient = client;
    p.hObject = object;
    p.cmd = cmd;
    p.params = (NvP64)(uintptr_t)params;
    p.paramsSize = size;
    int rc = ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS), &p);
    return (rc || p.status) ? -1 : 0;
}

int gpcmap_rm_query(const char *uuid, gpcmap_t *out)
{
    int minor = device_minor(uuid);
    if (minor < 0)
        return ENODEV;
    char dev[32];
    snprintf(dev, sizeof(dev), "/dev/nvidia%d", minor);
    int gpu_fd = open(dev, O_RDWR | O_CLOEXEC); /* must stay open for the controls */
    if (gpu_fd < 0)
        return errno;
    int ctl_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (ctl_fd < 0)
    {
        close(gpu_fd);
        return errno;
    }
    int rc = EIO;
    nv_ioctl_card_info_t cards[32] = {0};
    if (ioctl(ctl_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_CARD_INFO, cards), cards))
        goto done;
    nv_ioctl_card_info_t card = {0};
    int found = 0;
    for (unsigned i = 0; i < 32; ++i)
        if (cards[i].valid && cards[i].minor_number == (unsigned)minor)
        {
            card = cards[i];
            found = 1;
        }
    if (!found)
        goto done;
    NvHandle client, device = 0x5c000003U, subdevice = 0x5c000004U;
    if (rm_alloc_root(ctl_fd, &client))
        goto done;
    NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS id = {0};
    id.gpuId = card.gpu_id;
    if (rm_control(ctl_fd, client, client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, &id, sizeof(id)))
        goto done;
    NV0080_ALLOC_PARAMETERS dev_params = {0};
    dev_params.deviceId = id.deviceInstance;
    dev_params.hClientShare = client;
    dev_params.vaMode = NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES;
    if (rm_alloc64(ctl_fd, client, client, device, NV01_DEVICE_0, &dev_params))
        goto done;
    NV2080_ALLOC_PARAMETERS sub_params = {0};
    if (rm_alloc64(ctl_fd, client, device, subdevice, NV20_SUBDEVICE_0, &sub_params))
        goto done;
    static NV2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS order;
    memset(&order, 0, sizeof(order));
    if (rm_control(ctl_fd, client, subdevice, NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER, &order, sizeof(order)))
    {
        rc = EACCES; /* refused: a MIG slice */
        goto done;
    }
    if ((int)order.numSm != out->num_sms)
    {
        rc = ERANGE;
        goto done;
    }
    /* per SM; both SMs of a TPC must agree */
    int gpc_ids[GPCMAP_MAX_GPCS * 4];
    int n_ids = 0;
    for (unsigned i = 0; i < order.numSm; ++i)
    {
        int t = (int)order.globalSmId[i].globalTpcId;
        int g = (int)order.globalSmId[i].gpcId;
        if (t != (int)(i / 2) || t >= GPCMAP_MAX_TPCS)
        {
            rc = ERANGE;
            goto done;
        }
        int k;
        for (k = 0; k < n_ids && gpc_ids[k] != g; ++k)
        {
        }
        if (k == n_ids)
        {
            if (n_ids == GPCMAP_MAX_GPCS * 4)
            {
                rc = ERANGE;
                goto done;
            }
            gpc_ids[n_ids++] = g;
        }
        if (out->gpc_of_tpc[t] >= 0 && out->gpc_of_tpc[t] != k)
        {
            rc = ERANGE;
            goto done;
        }
        out->gpc_of_tpc[t] = k;
    }
    out->num_gpcs = n_ids;
    snprintf(out->method, sizeof(out->method), "rm");
    rc = 0;
done:
    close(ctl_fd);
    close(gpu_fd);
    return rc;
}
#endif
