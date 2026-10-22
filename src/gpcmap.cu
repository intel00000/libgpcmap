// libgpcmap: the probes (see gpcmap.h).
#include "gpcmap.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

namespace cg = cooperative_groups;

// libsmctrl
static void (*g_set_mask)(void *, uint64_t, uint64_t) = nullptr;

int gpcmap_init(const char *smctrl_path)
{
    void *lib = dlopen(smctrl_path, RTLD_NOW | RTLD_GLOBAL);
    if (!lib)
        return ENOENT;
    g_set_mask = (void (*)(void *, uint64_t, uint64_t))dlsym(lib, "libsmctrl_set_stream_mask_ext");
    return g_set_mask ? 0 : ENOSYS;
}

void gpcmap_mask_stream(void *stream, const int *tpcs, int n)
{
    // libsmctrl: a set bit DISABLES the TPC; 128 bits; zero means unrestricted.
    uint64_t lo = 0, hi = 0;
    if (n > 0)
    {
        uint64_t on_lo = 0, on_hi = 0;
        for (int i = 0; i < n; ++i)
        {
            if (tpcs[i] < 64)
                on_lo |= 1ull << tpcs[i];
            else
                on_hi |= 1ull << (tpcs[i] - 64);
        }
        lo = ~on_lo;
        hi = ~on_hi;
    }
    g_set_mask(stream, lo, hi);
}

// kernels
__device__ __forceinline__ unsigned smid_reg()
{
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

__global__ void k_smid(int *out, long long spin_cycles)
{
    long long t0 = clock64();
    if (threadIdx.x == 0)
        out[blockIdx.x] = smid_reg();
    while (clock64() - t0 < spin_cycles)
    {
    }
}

// The CTAs of one cluster are resident together in one GPC: each records its
// SM after the cluster has assembled. Dynamic shared memory sized by the
// caller so that one CTA fills an SM.
__global__ void k_cluster(int *sm_of_block, long long spin_cycles)
{
    extern __shared__ unsigned char big[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    long long t0 = clock64();
    if (threadIdx.x == 0)
    {
        sm_of_block[blockIdx.x] = smid_reg();
        big[0] = 1;
    }
    while (clock64() - t0 < spin_cycles)
    {
    }
    cl.sync();
#else
    if (threadIdx.x == 0)
        sm_of_block[blockIdx.x] = -1;
#endif
}

// Stream read kernel: each thread reads a float4 from the input array, sums the x,y,z,w components,
// and accumulates into a local variable.
__global__ void k_stream_read(const float4 *__restrict__ p, size_t n4, float *sink, int reps)
{
    float acc = 0.f;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t start = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = 0; r < reps; ++r)
    {
        size_t i = start;
        for (; i + 3 * stride < n4; i += 4 * stride)
        {
            float4 a = __ldcs(p + i), b = __ldcs(p + i + stride);
            float4 c = __ldcs(p + i + 2 * stride), d = __ldcs(p + i + 3 * stride);
            acc += a.x + b.y + c.z + d.w;
        }
        for (; i < n4; i += stride)
            acc += __ldcs(p + i).x;
    }
    if (acc == 123456.789f)
        sink[0] = acc;
}

// helpers
#define CK(call)                 \
    do                           \
    {                            \
        cudaError_t e_ = (call); \
        if (e_ != cudaSuccess)   \
            return -(int)e_;     \
    } while (0)

// fill the gpcmap_t structure
static int describe(int dev, gpcmap_t *out)
{
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    memset(out, 0, sizeof(*out));
    out->cuda_device = dev;
    snprintf(out->name, sizeof(out->name), "%.95s", prop.name);
    const unsigned char *u = (const unsigned char *)prop.uuid.bytes;
    snprintf(out->uuid, sizeof(out->uuid),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3], u[4],
             u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    out->num_sms = prop.multiProcessorCount;
    out->num_tpcs = prop.multiProcessorCount / 2;
    for (int t = 0; t < GPCMAP_MAX_TPCS; ++t)
        out->gpc_of_tpc[t] = -1;
    return 0;
}

int gpcmap_describe(int dev, gpcmap_t *out)
{
    int rc = describe(dev, out);
    return rc < 0 ? EIO : 0;
}

// Calculate the maximum dynamic shared memory that can be allocated to one CTA given the device's opt-in limit and the per-SM limit.
//  - The opt-in limit is the maximum shared memory per block that can be requested by a kernel launch,
//  - The per-SM limit is the total shared memory available on each SM.
//
// returns the maximum shared memory that can be allocated to one CTA.
static int one_cta_smem(int dev)
{
    int optin = 0, per_sm = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    int smem = optin - 1024;
    if (smem < per_sm / 2 + 1024)
        smem = per_sm / 2 + 1024;
    return smem > optin ? optin : smem;
}

// here we launch a cluster of CTAs, each of which records its SM after the cluster has assembled. The
// cluster is sized to fill the GPC, and the SMs are returned to the host
static cudaError_t launch_clusters(cudaStream_t s, int *d_sm, int n_clusters, int smem)
{
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 8;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_clusters * 8);
    cfg.blockDim = dim3(64);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, k_cluster, d_sm, 20000LL);
}

// Determine the maximum number of clusters that can be resident on the device at once, given the dynamic shared memory size.
static int cluster_capacity(int dev, int smem)
{
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 8;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8);
    cfg.blockDim = dim3(64);
    cfg.dynamicSmemBytes = smem;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, (void *)k_cluster, &cfg) != cudaSuccess)
        return -1;
    return n;
}

struct UnionFind
{
    std::vector<int> p;
    explicit UnionFind(int n) : p(n)
    {
        for (int i = 0; i < n; ++i)
            p[i] = i;
    }
    int find(int a)
    {
        while (p[a] != a)
            a = p[a] = p[p[a]];
        return a;
    }
    void unite(int a, int b) { p[find(a)] = find(b); }
};

static void groups_to_map(UnionFind &uf, gpcmap_t *out)
{
    // components of more than one TPC become GPCs, numbered by their lowest TPC id
    std::vector<int> root_of_tpc(out->num_tpcs);
    for (int t = 0; t < out->num_tpcs; ++t)
        root_of_tpc[t] = uf.find(2 * t);
    std::vector<std::pair<int, int>> firsts; // (lowest tpc, root)
    for (int t = 0; t < out->num_tpcs; ++t)
    {
        int r = root_of_tpc[t];
        bool seen = false;
        for (auto &f : firsts)
            seen |= f.second == r;
        if (!seen)
            firsts.push_back({t, r});
    }
    out->num_gpcs = 0;
    for (auto &f : firsts)
    {
        int members = 0;
        for (int t = 0; t < out->num_tpcs; ++t)
            members += root_of_tpc[t] == f.second;
        if (members < 2)
            continue;
        for (int t = 0; t < out->num_tpcs; ++t)
            if (root_of_tpc[t] == f.second)
                out->gpc_of_tpc[t] = out->num_gpcs;
        out->num_gpcs++;
    }
}

// the cluster method
static int method_cluster(int dev, gpcmap_t *out)
{
    int supported = 0;
    cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, dev);
    if (!supported)
        return ENOSYS;
    int smem = one_cta_smem(dev);
    cudaFuncSetAttribute(k_cluster, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    // determine the the max # of clusters that can be resident on the device at once
    int cap = cluster_capacity(dev, smem);
    if (cap < 1)
        return EIO;
    out->cluster_capacity = cap;

    const int rounds = cap + cap / 2;
    cudaStream_t s;
    if (cudaStreamCreate(&s) != cudaSuccess)
        return EIO;
    int *d_sm = nullptr;
    const int max_ctas = rounds * 8;
    if (cudaMalloc(&d_sm, max_ctas * sizeof(int)) != cudaSuccess)
        return EIO;
    std::vector<int> h_sm(max_ctas);
    UnionFind uf(out->num_sms);
    auto run = [&](int n_clusters) -> int
    {
        cudaMemsetAsync(d_sm, 0xff, n_clusters * 8 * sizeof(int), s);
        if (launch_clusters(s, d_sm, n_clusters, smem) != cudaSuccess)
            return EIO;
        if (cudaStreamSynchronize(s) != cudaSuccess)
            return EIO;
        cudaMemcpy(h_sm.data(), d_sm, n_clusters * 8 * sizeof(int), cudaMemcpyDeviceToHost);
        for (int c = 0; c < n_clusters; ++c)
        {
            const int *m = &h_sm[c * 8];
            for (int k = 0; k < 8; ++k)
                if (m[k] < 0 || m[k] >= out->num_sms)
                    return EIO;
            for (int k = 1; k < 8; ++k)
                uf.unite(m[0], m[k]);
        }
        return 0;
    };
    int rc = 0;
    for (int i = 0; i < 10 && !rc; ++i)
        rc = run(cap);
    // one TPC excluded at a time: clusters move around it (always feasible,
    // every GPC keeps the rest of its TPCs)
    std::vector<int> all(out->num_tpcs);
    for (int t = 0; t < out->num_tpcs; ++t)
        all[t] = t;
    for (int t = 0; t < out->num_tpcs && !rc; ++t)
    {
        std::vector<int> keep;
        for (int u : all)
            if (u != t)
                keep.push_back(u);
        gpcmap_mask_stream(s, keep.data(), (int)keep.size());
        rc = run(rounds);
    }
    gpcmap_mask_stream(s, nullptr, 0);
    cudaFree(d_sm);
    cudaStreamDestroy(s);
    if (rc)
        return rc;
    // SMs 2t and 2t+1 are one TPC: a component that splits a TPC is no GPC
    for (int t = 0; t < out->num_tpcs; ++t)
    {
        int a = uf.find(2 * t), b = uf.find(2 * t + 1);
        if (a != b)
        {
            // one of them was never seen in a cluster: join it to its sibling
            uf.unite(2 * t, 2 * t + 1);
        }
    }
    groups_to_map(uf, out);
    snprintf(out->method, sizeof(out->method), "cluster");
    return 0;
}

int gpcmap_cluster_fits(int dev, const int *tpcs, int n, int timeout_ms)
{
    gpcmap_t info;
    if (describe(dev, &info) < 0)
        return -1;
    CK(cudaSetDevice(dev));
    int smem = one_cta_smem(dev);
    cudaFuncSetAttribute(k_cluster, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int *d_sm = nullptr;
    CK(cudaMalloc(&d_sm, 8 * sizeof(int)));
    cudaEvent_t done;
    CK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    gpcmap_mask_stream(s, tpcs, n);
    cudaError_t e = launch_clusters(s, d_sm, 1, smem);
    if (e != cudaSuccess)
        return -(int)e;
    CK(cudaEventRecord(done, s));
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;)
    {
        cudaError_t q = cudaEventQuery(done);
        if (q == cudaSuccess)
            return 1;
        if (q != cudaErrorNotReady)
            return -(int)q;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long ms = (now.tv_sec - t0.tv_sec) * 1000L + (now.tv_nsec - t0.tv_nsec) / 1000000L;
        if (ms > timeout_ms)
            return 0;
        usleep(1000);
    }
}

// the co-run method, see README.md
// tldr: We assume that TPCs that share a GPC will slow each other down when they run together,
// and that TPCs that do not share a GPC will has less of a impact.
//
// Start with one TPC, then add the slowest TPC to the anchor until a gap appears in the slowdowns.
// The slow TPCs are assigned to the same GPC as the anchor.
static float median_of(std::vector<float> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int method_corun(int dev, gpcmap_t *out)
{
    const size_t bytes = (size_t)1 << 30;
    float *buf_a = nullptr, *buf_v = nullptr, *sink = nullptr;
    if (cudaMalloc(&buf_a, bytes) != cudaSuccess)
        return EIO;
    if (cudaMalloc(&buf_v, bytes) != cudaSuccess)
        return EIO;
    if (cudaMalloc(&sink, 8) != cudaSuccess)
        return EIO;
    cudaMemset(buf_a, 0, bytes);
    cudaMemset(buf_v, 0, bytes);
    cudaStream_t agg, vic;
    cudaStreamCreate(&agg);
    cudaStreamCreate(&vic);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const size_t n4 = bytes / 16;
    auto victim_ms = [&](bool with_aggressor, int agg_blocks) -> float
    {
        std::vector<float> ts;
        for (int i = 0; i < 6; ++i)
        {
            if (with_aggressor)
                k_stream_read<<<agg_blocks, 256, 0, agg>>>((const float4 *)buf_a, n4, sink, 4);
            cudaEventRecord(e0, vic);
            k_stream_read<<<16, 256, 0, vic>>>((const float4 *)buf_v, n4, sink, 1);
            cudaEventRecord(e1, vic);
            cudaStreamSynchronize(vic);
            cudaStreamSynchronize(agg);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i)
                ts.push_back(ms); // the first run warms
        }
        return median_of(ts);
    };
    std::vector<int> unassigned;
    for (int t = 0; t < out->num_tpcs; ++t)
        unassigned.push_back(t);
    out->num_gpcs = 0;
    int rc = 0;
    // The anchor starts as one TPC (consecutive ids rarely share a GPC) and
    // grows by the slowest victim while the slowdowns show no clear split.
    while (!unassigned.empty() && !rc)
    {
        std::vector<int> anchor = {unassigned[0]};
        std::vector<int> high;
        for (int attempt = 0;; ++attempt)
        {
            gpcmap_mask_stream(agg, anchor.data(), (int)anchor.size());
            std::vector<std::pair<int, float>> slow; // (tpc, co / solo)
            for (int t : unassigned)
            {
                if (std::find(anchor.begin(), anchor.end(), t) != anchor.end())
                    continue;
                gpcmap_mask_stream(vic, &t, 1);
                float solo = victim_ms(false, 0);
                float co = victim_ms(true, 32 * (int)anchor.size());
                slow.push_back({t, co / solo});
            }
            if (slow.empty())
                break;
            std::vector<float> xs;
            for (auto &p : slow)
                xs.push_back(p.second);
            std::sort(xs.begin(), xs.end());
            float gap = 0, cut = xs.back();
            for (size_t i = 0; i + 1 < xs.size(); ++i)
                if (xs[i + 1] - xs[i] > gap)
                {
                    gap = xs[i + 1] - xs[i];
                    cut = xs[i];
                }
            high.clear();
            if (gap >= 0.03f)
                for (auto &p : slow)
                    if (p.second > cut)
                        high.push_back(p.first);
            fprintf(stderr, "gpcmap: corun anchor of %zu from TPC %d: best gap %.3f, %zu TPCs in the slow mode\n",
                    anchor.size(), anchor[0], gap, high.size());
            if (gap >= 0.03f && !high.empty() && high.size() <= 12)
                break;
            // the last GPC: every TPC left is a mate of the anchor, all slowed alike
            if (gap < 0.03f && out->num_gpcs > 0 && slow.size() + anchor.size() <= 12 && xs[xs.size() / 2] >= 1.03f)
            {
                for (auto &p : slow)
                    high.push_back(p.first);
                break;
            }
            if (attempt == 3)
            {
                rc = EAGAIN;
                break;
            }
            auto slowest = std::max_element(slow.begin(), slow.end(),
                                            [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                                            {
                                                return a.second < b.second;
                                            });
            anchor.push_back(slowest->first);
        }
        if (rc)
            break;
        for (int u : anchor)
            out->gpc_of_tpc[u] = out->num_gpcs;
        for (int u : high)
            out->gpc_of_tpc[u] = out->num_gpcs;
        out->num_gpcs++;
        std::vector<int> rest;
        for (int u : unassigned)
            if (out->gpc_of_tpc[u] < 0)
                rest.push_back(u);
        unassigned = rest;
    }
    gpcmap_mask_stream(agg, nullptr, 0);
    gpcmap_mask_stream(vic, nullptr, 0);
    cudaFree(buf_a);
    cudaFree(buf_v);
    cudaFree(sink);
    cudaStreamDestroy(agg);
    cudaStreamDestroy(vic);
    if (!rc)
        snprintf(out->method, sizeof(out->method), "corun");
    return rc;
}

// validation
int gpcmap_validate(int dev, int *failed, int max_failed)
{
    gpcmap_t info;
    if (describe(dev, &info) < 0)
        return -1;
    if (cudaSetDevice(dev) != cudaSuccess)
        return -1;
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int blocks = 4 * info.num_sms;
    int *d_out = nullptr;
    cudaMalloc(&d_out, blocks * sizeof(int));
    std::vector<int> h(blocks);
    int bad = 0;
    for (int t = 0; t < info.num_tpcs; ++t)
    {
        gpcmap_mask_stream(s, &t, 1);
        cudaMemsetAsync(d_out, 0xff, blocks * sizeof(int), s);
        k_smid<<<blocks, 256, 0, s>>>(d_out, 200000LL);
        cudaStreamSynchronize(s);
        cudaMemcpy(h.data(), d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost);
        bool seen_lo = false, seen_hi = false, leak = false;
        for (int b = 0; b < blocks; ++b)
        {
            if (h[b] == 2 * t)
                seen_lo = true;
            else if (h[b] == 2 * t + 1)
                seen_hi = true;
            else if (h[b] >= 0)
                leak = true;
        }
        if (leak || !seen_lo || !seen_hi)
        {
            if (bad < max_failed)
                failed[bad] = t;
            bad++;
        }
    }
    gpcmap_mask_stream(s, nullptr, 0);
    cudaFree(d_out);
    cudaStreamDestroy(s);
    return bad;
}

// the query
int gpcmap_query(int dev, const char *method, gpcmap_t *out)
{
    if (!g_set_mask)
        return ENOSYS;
    if (describe(dev, out) < 0)
        return EIO;
    if (cudaSetDevice(dev) != cudaSuccess)
        return EIO;
    bool automatic = !method || !strcmp(method, "auto");
    int rc = ENOSYS;
    if (automatic || !strcmp(method, "rm"))
    {
        rc = gpcmap_rm_query(out->uuid, out);
        if (rc == 0 || !automatic)
            return rc;
    }
    if (automatic || !strcmp(method, "cluster"))
    {
        rc = method_cluster(dev, out);
        if (rc == 0 || !automatic)
            return rc;
    }
    if (automatic || !strcmp(method, "corun"))
    {
        rc = method_corun(dev, out);
        return rc;
    }
    return EINVAL;
}

int gpcmap_write_json(const gpcmap_t *m, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return errno;
    fprintf(f, "{\n \"device\": \"%s\",\n \"uuid\": \"%s\",\n \"cuda_device\": %d,\n \"num_sms\": %d,\n \"num_tpcs\": %d,\n"
               " \"method\": \"%s\",\n \"cluster_capacity\": %d,\n \"gpc_to_tpcs\": {",
            m->name, m->uuid, m->cuda_device, m->num_sms, m->num_tpcs, m->method, m->cluster_capacity);
    for (int g = 0; g < m->num_gpcs; ++g)
    {
        fprintf(f, "%s\n  \"%d\": [", g ? "," : "", g);
        bool first = true;
        for (int t = 0; t < m->num_tpcs; ++t)
            if (m->gpc_of_tpc[t] == g)
            {
                fprintf(f, "%s%d", first ? "" : ", ", t);
                first = false;
            }
        fprintf(f, "]");
    }
    fprintf(f, "\n },\n \"unplaced\": [");
    bool first = true;
    for (int t = 0; t < m->num_tpcs; ++t)
        if (m->gpc_of_tpc[t] < 0)
        {
            fprintf(f, "%s%d", first ? "" : ", ", t);
            first = false;
        }
    fprintf(f, "]\n}\n");
    fclose(f);
    return 0;
}
