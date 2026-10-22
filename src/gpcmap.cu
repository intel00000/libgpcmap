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
    // libsmctrl: a set bit disables the TPC; zero means unrestricted
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

// each CTA of a cluster records its SM; the dynamic shared memory leaves one CTA per SM
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
    memset(out, 0, sizeof(*out));
    for (int t = 0; t < GPCMAP_MAX_TPCS; ++t)
        out->gpc_of_tpc[t] = -1;
    CK(cudaGetDeviceProperties(&prop, dev));
    out->cuda_device = dev;
    snprintf(out->name, sizeof(out->name), "%.95s", prop.name);
    const unsigned char *u = (const unsigned char *)prop.uuid.bytes;
    snprintf(out->uuid, sizeof(out->uuid),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3], u[4],
             u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    out->num_sms = prop.multiProcessorCount;
    out->num_tpcs = prop.multiProcessorCount / 2;
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
    // components of two or more TPCs are GPCs, numbered by their lowest TPC
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
    // one TPC excluded at a time
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
    for (int t = 0; t < out->num_tpcs; ++t)
        uf.unite(2 * t, 2 * t + 1); // the two SMs of a TPC
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
// tldr: TPCs that share a GPC slow each other down when they stream together;
// TPCs in different GPCs have less of an impact on each other.
//
// Two stages. The atoms: every TPC streams against every other, the victim's
// time beside the aggressor over its time alone (back to back, so clock drift
// cancels); a row's plateau of near-equal slowdowns is its closest kin and
// mutually slow TPCs merge. The joining: two clusters stream together, and
// clusters in one GPC share its path to memory and stream slower than their
// rates add up to.
static float median_of(std::vector<float> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static const int MAX_GPC_TPCS = 10;

struct Corun
{
    cudaStream_t agg = nullptr, vic = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    float *buf_a = nullptr, *buf_v = nullptr, *sink = nullptr;
    size_t n4 = 0;

    int init()
    {
        const size_t bytes = (size_t)1 << 29;
        if (cudaMalloc(&buf_a, bytes) != cudaSuccess || cudaMalloc(&buf_v, bytes) != cudaSuccess)
            return EIO;
        if (cudaMalloc(&sink, 8) != cudaSuccess)
            return EIO;
        cudaMemset(buf_a, 0, bytes);
        cudaMemset(buf_v, 0, bytes);
        cudaStreamCreate(&agg);
        cudaStreamCreate(&vic);
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        n4 = bytes / 16;
        return 0;
    }

    float run(const std::vector<int> &victim, const std::vector<int> *aggressor)
    {
        if (aggressor)
            k_stream_read<<<32 * (int)aggressor->size(), 256, 0, agg>>>((const float4 *)buf_a, n4, sink,
                                                                        2 * (int)aggressor->size());
        cudaEventRecord(e0, vic);
        k_stream_read<<<16 * (int)victim.size(), 256, 0, vic>>>((const float4 *)buf_v, n4, sink, 1);
        cudaEventRecord(e1, vic);
        cudaStreamSynchronize(vic);
        cudaStreamSynchronize(agg);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms;
    }

    // the victim's slowdown beside the aggressor: alone and together, three
    // times each, back to back
    float ratio(const std::vector<int> &victim, const std::vector<int> &aggressor)
    {
        gpcmap_mask_stream(vic, victim.data(), (int)victim.size());
        gpcmap_mask_stream(agg, aggressor.data(), (int)aggressor.size());
        run(victim, nullptr);
        std::vector<float> r;
        for (int i = 0; i < 3; ++i)
        {
            float solo = run(victim, nullptr);
            float co = run(victim, &aggressor);
            r.push_back(co / solo);
        }
        return median_of(r);
    }

    // the time to stream the buffer on these TPCs together, median of three
    float stream_ms(const std::vector<int> &tpcs)
    {
        gpcmap_mask_stream(agg, tpcs.data(), (int)tpcs.size());
        std::vector<float> ts;
        for (int i = 0; i <= 3; ++i)
        {
            cudaEventRecord(e0, agg);
            k_stream_read<<<32 * (int)tpcs.size(), 256, 0, agg>>>((const float4 *)buf_a, n4, sink, 2);
            cudaEventRecord(e1, agg);
            cudaStreamSynchronize(agg);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i)
                ts.push_back(ms);
        }
        return median_of(ts);
    }

    // how much slower two clusters stream together than their rates add up
    // to: above 1 when they share a path to memory
    float cap(const std::vector<int> &a, const std::vector<int> &b)
    {
        std::vector<int> both(a);
        both.insert(both.end(), b.begin(), b.end());
        std::vector<float> r;
        for (int i = 0; i < 3; ++i)
        {
            float ta = stream_ms(a), tb = stream_ms(b), tab = stream_ms(both);
            r.push_back(tab * (1.0f / ta + 1.0f / tb));
        }
        return median_of(r);
    }

    void release()
    {
        gpcmap_mask_stream(agg, nullptr, 0);
        gpcmap_mask_stream(vic, nullptr, 0);
        cudaFree(buf_a);
        cudaFree(buf_v);
        cudaFree(sink);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(agg);
        cudaStreamDestroy(vic);
    }
};

static int method_corun(int dev, gpcmap_t *out)
{
    Corun c;
    if (c.init())
        return EIO;
    std::vector<std::vector<int>> cl(out->num_tpcs);
    for (int t = 0; t < out->num_tpcs; ++t)
        cl[t] = {t};
    // the victim rows once, for the atoms: later rounds with bigger aggressors
    // measure the device's bandwidth floor, not the GPC
    for (int round = 0; round < 1; ++round)
    {
        const size_t k = cl.size();
        if (k < 2)
            break;
        std::vector<std::vector<float>> ratio(k, std::vector<float>(k, 1.0f));
        for (size_t a = 0; a < k; ++a)
            for (size_t b = 0; b < k; ++b)
                if (a != b)
                    ratio[a][b] = c.ratio(cl[b], cl[a]);
        int remaining = 0;
        for (auto &g : cl)
            remaining += (int)g.size();
        // a row's slow mode: the largest plateau (within 3%) of at least two
        // clusters slowed 3% or more over the row's quiet half, no wider than
        // a GPC; or all slowed alike when the rest fits one GPC; a row with
        // neither is blank, keeping its strongest victim as a pair candidate
        std::vector<std::vector<char>> slow(k, std::vector<char>(k, 0));
        std::vector<char> blank(k, 1);
        std::vector<int> lone(k, -1);
        int split_rows = 0;
        for (size_t a = 0; a < k; ++a)
        {
            std::vector<std::pair<float, int>> xs;
            for (size_t b = 0; b < k; ++b)
                if (b != a)
                    xs.push_back({ratio[a][b], (int)b});
            std::sort(xs.begin(), xs.end());
            float base = xs[(xs.size() / 2) / 2].first;
            std::vector<std::pair<float, int>> cand;
            for (auto &x : xs)
                if (x.first >= base + 0.03f)
                    cand.push_back(x);
            size_t best_i = 0, best_n = 0;
            for (size_t i = 0; i < cand.size(); ++i)
            {
                size_t j = i;
                while (j + 1 < cand.size() && cand[j + 1].first - cand[i].first <= 0.03f)
                    j++;
                if (j - i + 1 > best_n)
                {
                    best_n = j - i + 1;
                    best_i = i;
                }
            }
            if (best_n >= 2 && (int)best_n <= MAX_GPC_TPCS - (int)cl[a].size())
            {
                for (size_t i = best_i; i < best_i + best_n; ++i)
                    slow[a][cand[i].second] = 1;
                blank[a] = 0;
                split_rows++;
            }
            else if (xs.front().first >= 1.03f && remaining <= MAX_GPC_TPCS)
            {
                for (size_t b = 0; b < k; ++b)
                    slow[a][b] = b != a;
                blank[a] = 0;
            }
            else if (xs.back().first >= base + 0.2f)
            {
                lone[a] = xs.back().second;
            }
        }
        // merges by strength: plateaus naming each other, a plateau naming a
        // blank smaller cluster, or two blank rows whose strongest victims are
        // each other; none beyond a GPC
        struct Link
        {
            float strength;
            int a, b;
        };
        std::vector<Link> links;
        for (size_t a = 0; a < k; ++a)
            for (size_t b = a + 1; b < k; ++b)
            {
                bool mutual = slow[a][b] && slow[b][a];
                bool defer = (slow[a][b] && blank[b] && cl[a].size() > cl[b].size()) ||
                             (slow[b][a] && blank[a] && cl[b].size() > cl[a].size());
                bool pair = lone[a] == (int)b && lone[b] == (int)a;
                if (mutual || defer || pair)
                    links.push_back({std::min(ratio[a][b], ratio[b][a]), (int)a, (int)b});
            }
        std::sort(links.begin(), links.end(), [](const Link &x, const Link &y)
                  { return x.strength > y.strength; });
        UnionFind uf((int)k);
        std::vector<int> size(k);
        for (size_t a = 0; a < k; ++a)
            size[a] = (int)cl[a].size();
        int merges = 0, refused = 0;
        for (const Link &l : links)
        {
            int ra = uf.find(l.a), rb = uf.find(l.b);
            if (ra == rb)
                continue;
            if (size[ra] + size[rb] > MAX_GPC_TPCS)
            {
                refused++;
                continue;
            }
            uf.unite(ra, rb);
            size[uf.find(ra)] = size[ra] + size[rb];
            merges++;
        }
        out->corun_conflicts += refused;
        fprintf(stderr, "gpcmap: corun round %d: %zu clusters, %d rows with a plateau, %d merges, %d refused\n",
                round + 1, k, split_rows, merges, refused);
        if (const char *dump = getenv("GPCMAP_DUMP"))
        {
            char path[512];
            snprintf(path, sizeof(path), "%s.round%d.csv", dump, round + 1);
            if (FILE *f = fopen(path, "w"))
            {
                for (size_t a = 0; a < k; ++a)
                {
                    fprintf(f, "%d", cl[a][0]);
                    for (size_t b = 0; b < k; ++b)
                        fprintf(f, ",%.4f", ratio[a][b]);
                    fprintf(f, "\n");
                }
                fclose(f);
            }
        }
        if (!merges)
            break;
        std::vector<std::vector<int>> next;
        std::vector<int> index(k, -1);
        for (size_t a = 0; a < k; ++a)
        {
            int r = uf.find((int)a);
            if (index[r] < 0)
            {
                index[r] = (int)next.size();
                next.push_back({});
            }
            next[index[r]].insert(next[index[r]].end(), cl[a].begin(), cl[a].end());
        }
        for (auto &g : next)
            std::sort(g.begin(), g.end());
        cl = next;
    }
    if (getenv("GPCMAP_CORUN_SPLIT"))
    { // a check of the joining stage: halve every cluster first
        std::vector<std::vector<int>> halves;
        for (auto &g : cl)
        {
            if (g.size() < 2)
            {
                halves.push_back(g);
                continue;
            }
            halves.push_back(std::vector<int>(g.begin(), g.begin() + g.size() / 2));
            halves.push_back(std::vector<int>(g.begin() + g.size() / 2, g.end()));
        }
        cl = halves;
    }
    // Joining: clusters on the same GPC share a memory path, so they run
    // together more slowly than the sum of their individual rates. Pairs on
    // different GPCs with the same sizes are limited by device-wide capacity;
    // a row's excess over same-sized clusters identifies its GPC-mates.
    for (int pass = 0; pass < 8; ++pass)
    {
        const size_t k = cl.size();
        std::vector<std::vector<float>> capv(k, std::vector<float>(k, 0.0f));
        int measured = 0;
        for (size_t a = 0; a < k; ++a)
            for (size_t b = a + 1; b < k; ++b)
                if ((int)(cl[a].size() + cl[b].size()) <= MAX_GPC_TPCS)
                {
                    capv[a][b] = capv[b][a] = c.cap(cl[a], cl[b]);
                    measured++;
                }
        if (!measured)
            break;
        // the bar for a pair: the same union with a stand-in of the partner's
        // size cut from each of the three largest other clusters (foreign in
        // all but at most one), the median of their caps
        auto control = [&](size_t a, size_t b) -> float
        {
            std::vector<size_t> others;
            for (size_t o = 0; o < k; ++o)
                if (o != a && o != b && cl[o].size() >= cl[b].size())
                    others.push_back(o);
            std::sort(others.begin(), others.end(), [&](size_t x, size_t y)
                      { return cl[x].size() > cl[y].size(); });
            std::vector<float> vals;
            for (size_t i = 0; i < others.size() && i < 3; ++i)
            {
                std::vector<int> stand(cl[others[i]].begin(), cl[others[i]].begin() + cl[b].size());
                vals.push_back(c.cap(cl[a], stand));
            }
            return vals.empty() ? 1.0f : median_of(vals);
        };
        std::vector<std::vector<float>> excess(k, std::vector<float>(k, 0.0f));
        for (size_t a = 0; a < k; ++a)
            for (size_t b = a + 1; b < k; ++b)
                if (capv[a][b] >= 1.10f)
                {
                    excess[a][b] = capv[a][b] - control(a, b);
                    excess[b][a] = capv[a][b] - control(b, a);
                    if (getenv("GPCMAP_DEBUG"))
                        fprintf(stderr, "  join %d(%zu)+%d(%zu): cap %.3f, over the stand-ins %.3f / %.3f\n", cl[a][0],
                                cl[a].size(), cl[b][0], cl[b].size(), capv[a][b], excess[a][b], excess[b][a]);
                }
        struct Link
        {
            float strength;
            int a, b;
        };
        std::vector<Link> links;
        for (size_t a = 0; a < k; ++a)
            for (size_t b = a + 1; b < k; ++b)
                if (capv[a][b] >= 1.10f && excess[a][b] >= 0.05f && excess[b][a] >= 0.05f)
                    links.push_back({std::min(excess[a][b], excess[b][a]), (int)a, (int)b});
        std::sort(links.begin(), links.end(), [](const Link &x, const Link &y)
                  { return x.strength > y.strength; });
        UnionFind uf((int)k);
        std::vector<int> size(k);
        for (size_t a = 0; a < k; ++a)
            size[a] = (int)cl[a].size();
        int merges = 0, refused = 0;
        for (const Link &l : links)
        {
            int ra = uf.find(l.a), rb = uf.find(l.b);
            if (ra == rb)
                continue;
            if (size[ra] + size[rb] > MAX_GPC_TPCS)
            {
                refused++;
                continue;
            }
            uf.unite(ra, rb);
            size[uf.find(ra)] = size[ra] + size[rb];
            merges++;
        }
        out->corun_conflicts += refused;
        fprintf(stderr, "gpcmap: corun join %d: %zu clusters, %d pairs streamed together, %d merges, %d refused\n",
                pass + 1, k, measured, merges, refused);
        if (const char *dump = getenv("GPCMAP_DUMP"))
        {
            char path[512];
            snprintf(path, sizeof(path), "%s.join%d.csv", dump, pass + 1);
            if (FILE *f = fopen(path, "w"))
            {
                for (size_t a = 0; a < k; ++a)
                {
                    fprintf(f, "%d", cl[a][0]);
                    for (size_t b = 0; b < k; ++b)
                        fprintf(f, ",%.4f", capv[a][b]);
                    fprintf(f, "\n");
                }
                fclose(f);
            }
        }
        if (!merges)
            break;
        std::vector<std::vector<int>> next;
        std::vector<int> index(k, -1);
        for (size_t a = 0; a < k; ++a)
        {
            int r = uf.find((int)a);
            if (index[r] < 0)
            {
                index[r] = (int)next.size();
                next.push_back({});
            }
            next[index[r]].insert(next[index[r]].end(), cl[a].begin(), cl[a].end());
        }
        for (auto &g : next)
            std::sort(g.begin(), g.end());
        cl = next;
    }
    c.release();
    std::sort(cl.begin(), cl.end(), [](const std::vector<int> &x, const std::vector<int> &y)
              { return x[0] < y[0]; });
    // clusters of two or more TPCs are the GPCs; the rest stay unplaced
    out->num_gpcs = 0;
    for (auto &g : cl)
    {
        if (g.size() < 2)
            continue;
        for (int t : g)
            out->gpc_of_tpc[t] = out->num_gpcs;
        out->num_gpcs++;
    }
    snprintf(out->method, sizeof(out->method), "corun");
    return 0;
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
        describe(dev, out);
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
    fprintf(f, "{\n \"device\": \"%s\",\n \"uuid\": \"%s\",\n \"cuda_device\": %d,\n \"sms\": %d,\n \"tpcs\": %d,\n"
               " \"method\": \"%s\",\n \"cluster_capacity\": %d,\n \"corun_conflicts\": %d,\n \"gpc_to_tpcs\": {",
            m->name, m->uuid, m->cuda_device, m->num_sms, m->num_tpcs, m->method, m->cluster_capacity,
            m->corun_conflicts);
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
