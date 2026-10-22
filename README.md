# libgpcmap - Mapping TPCs to GPCs

Test which TPC ids share a GPC on this device.

Why it exists: each mask bit represents one TPC (two SMs), and global TPC
ids follow the driver's numbering of enabled TPCs. The TPCs that share a GPC
vary by product and chip, so layouts that reserve a GPC—or SMs within one
GPC for a clustered decode kernel—must read this map from the device.

## Methods

- `rm`: reads the driver's own table through two non-privileged RM controls
  (`NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER` and
  `..._GET_SM_TO_GPC_TPC_MAPPINGS`). It is exact but unavailable on MIG
  slices, and its ABI is tied to the installed driver; build it with
  `make RM=1 NV=/path/to/open-gpu-kernel-modules` from that driver.

  The table identifies both the physical GPC for each TPC (its memory path)
  and the virtual GPC where the work distributor places clusters. Both are
  reported, with the virtual grouping in `virtual_gpc_to_tpcs`.

- `nvdebug`: reads the same table from the hardware when the `nvdebug` kernel
  module is loaded. The files `/proc/gpuN/CWD_GPC_TPC_ID*` expose the work
  distributor's registers, which map logical TPC ids to GPCs using one byte per
  TPC; the GPC id is in the upper nibble. **Requires root, untested**.

- `cluster`: uses thread-block clusters (Hopper and later). The hardware
  places all 8 CTAs of a cluster inside one virtual GPC, so SMs that host
  the same cluster share a GPC.

  - Probing: a kernel is launched as clusters of 8 CTAs, each CTA sized to
    fill one SM, so a cluster occupies 8 SMs (4 TPCs) of one GPC. Every CTA
    records its SM id, and the SMs of each cluster are unioned.

  - Coverage: the device holds only a limited number of clusters at once,
    so one launch does not reach every SM.
    The launch is repeated, and then repeated with one TPC excluded at
    a time through the stream mask, which moves the clusters around until
    every TPC has shared a cluster with its GPC-mates.

  - Settling: a TPC still unplaced is tried against each GPC found: one
    cluster on that TPC plus three TPCs of the GPC runs only if the four
    share a GPC. A cluster that finds no room does not fail, it pends
    forever and times out eventually.

  - Limits: a TPC that forms a virtual GPC of its own (the tail TPCs on
    Hopper) can never share a cluster with anything, so it stays unplaced;
    the command then exits 2, and the driver's table is the complete answer.
    The groups are reported as `virtual_gpc`.

- `corun`: the groups of TPCs that share a path to memory, by timing, in
  two stages.

  - First, each TPC streams against every other TPC. The victim's runtime
    with an aggressor is compared with its solo runtime. In each row, a
    plateau of nearly equal slowdowns identifies the closest peers, and
    mutually slowing TPCs are merged.

  - Next, pairs of clusters stream together, and their runtime is compared with
    the sum of their separate rates, against the same union with foreign
    stand-ins of the partner's size.
    Clusters on one path run slower together; unrelated pairs show only the
    device-wide effect, which the stand-ins measure.
    A join requires at least 10% overhead over the additive time and 5% overhead
    over the stand-ins in both directions.
    Merges are selected by strength, subject to the ten-TPC limit, and repeated
    until no more merges are possible.

## Methods comparison

| | `rm` | `nvdebug` | `cluster` | `corun` |
| --- | --- | --- | --- | --- |
| source | the driver's SM order table, two RM controls: the physical and the virtual GPC of every TPC | the same table in hardware, the work distributor's `CWD_GPC_TPC_ID` registers through the nvdebug module | where clusters of 8 CTAs land: co-resident in one GPC | timing: streams that share a path to memory slow each other |
| nature | read, exact | read, exact | measured placement, exact where it covers | measured contention, inferred |
| needs | the open-kernel-module SDK headers at build time (`RM=1`); no privilege | the module loaded as root; reading is unprivileged; a context active | Hopper or later; libsmctrl stream masks | libsmctrl stream masks; a quiet GPU |
| refused or absent | on MIG slices | where the module cannot be loaded | Ampere and older | never; finds nothing on Hopper |
| answer | the physical and the virtual GPC of every TPC | the GPC of every TPC | the virtual GPC of every TPC the clusters reach; the rest settled one by one; single-TPC virtual GPCs stay unplaced | the groups of TPCs that share a path: the GPCs on GB202, sub-groups of three or four inside a GPC on GA100 |
| cost | milliseconds | milliseconds | 0.6 s on an RTX 5090, 109 s on an H200 | about 4 min on an RTX 5090, 2.5 min on an A100 |
| exit, `groups` | 0, `gpc` | 0, `gpc` | 0, `virtual_gpc`; 2 with TPCs left unplaced | 2, `shared_path`; a refused join is counted |
| hazards | ABI tied to the installed driver | PCI-order numbering; zero while the GPU is off | a cluster that finds no room pends for good, hence child processes | drift, size-dependent floors, one-way partner spikes, all handled; still inference |

`rm` is the ground truth where it builds, `nvdebug` the same table where
the module is loaded; `cluster` recovers the virtual GPCs, less the
single-TPC ones, without privileges or headers on Hopper and later;

`corun` is what remains on an Ampere MIG slice and finds the sub-group
structure there. `auto` tries them in order and takes the first
answer; every answer is followed by the single-TPC check.

No probe runs under MPS: MPS emulates `%smid` per client (every client's
blocks report SM ids from zero), so the command refuses the probing methods
and the check when an MPS control daemon is reachable.

Every answer is followed by the single-TPC check: with TPC t alone enabled a
kernel lands on SMs 2t and 2t+1, both and no other.

## Layout

`include/gpcmap.h` is the C API; `src/gpcmap.cu` the probes and the check,
`src/gpcmap_rm.c` the driver-table method, `src/gpcmap_cli.c` the command.

## Use

    make
    ./gpcmap --smctrl /path/to/libsmctrl.so --json gpc_map.json

prints the GPCs with their TPC ids and writes the JSON that the geometry
suite and slackserve's mask layouts read: `gpc_to_tpcs` with `groups` saying
what its groups are (`gpc`, `virtual_gpc`, `shared_path`), and
`virtual_gpc_to_tpcs` when the method knows the virtual grouping. `--method` picks one
method, `--no-complement` skips the per-TPC settlement, `--no-validate` the
check; `GPCMAP_DUMP=<prefix>` writes the co-run matrices as CSV. Exit 0 is
a checked GPC map, 2 a map in doubt (every co-run map, a refused join, or TPCs left unplaced), 1 no answer. The C API (`gpcmap_init`, `gpcmap_query`,
`gpcmap_cluster_fits`, `gpcmap_validate`, `gpcmap_groups`, `gpcmap_write_json`).

## Measured so far

| GPU | SMs | GPC layout | TPC numbering |
| --- | --- | --- | --- |
| A100 SXM4 80GB | 108 | 7 physical GPCs: 2×7 TPCs and 5×8 TPCs; no virtual grouping | Round-robin numbering, with the 7-TPC GPCs last. A second A100 matched. `corun` finds two groups per GPC. |
| H100 SXM 80GB | 132 | 8 physical GPCs: 2×9 TPCs and 6×8 TPCs; virtual layout matches the H200 | TPCs 0–31 use 8 GPCs, 32–59 use 7 GPCs, and 60–65 are leftovers. |
| H200 SXM 141GB | 132 | 8 physical GPCs with 6–9 TPCs each; 8 virtual GPCs with 4, 8, 8, 8, 8, 8, 9, and 9 TPCs, plus singleton virtual GPCs for TPCs 62–65 | TPCs 0–31 are round-robin across 8 virtual GPCs; 32–59 across virtual GPCs 1–7. TPCs 60–61 are the ninth TPCs in GPCs 6–7; 62–65 are singletons. |
| RTX 5090 | 170 | 11 GPCs: 3×7 TPCs and 8×8 TPCs; physical and virtual groups match | Round-robin for TPCs 0–76; TPCs 77–84 fill the full GPCs. |

Physical and virtual GPCs. The driver groups TPCs by physical and virtual
GPC. TPCs in a physical GPC share a memory path, so this grouping determines
bandwidth isolation. The work distributor also places clusters within virtual
GPCs. On Hopper, a virtual GPC is usually a physical GPC without a few tail
TPCs; each tail TPC forms its own virtual GPC.

An H200 probe confirmed that an eight-CTA cluster runs only when its four
TPCs are in the same virtual GPC. All 126 tested four-TPC combinations in a
nine-TPC GPC worked; none spanning two GPCs or including a tail TPC did.
Therefore, cluster-shaped kernels need eight free SMs in one virtual GPC;
free SMs on a tail TPC do not count.

The co-run finds nothing on Hopper: two TPCs of one GPC stream as fast
together as apart, and the GPC's path only saturates from four TPCs on.

## Sources

- Stream masks: the libsmctrl fork at `github.com/intel00000/libsmctrl`,
  from libsmctrl (`rtsrv.cs.unc.edu/public/libsmctrl.git`; J. Bakita and
  J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs", RTAS 2023).
- The `nvdebug` method follows `libsmctrl_get_gpc_info_ext()` from libsmctrl
  `ecrts25-ae` branch and reads the files of Bakita's nvdebug kernel module.
  The MPS check follows `libsmctrl_is_mps_running()`.
  (J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs
  for Composable Systems", ECRTS 2025 (DOI 10.4230/LIPIcs.ECRTS.2025.21),
  "Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management", RTAS 2024)
