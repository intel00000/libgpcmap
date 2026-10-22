# libgpcmap

Which TPC ids share a GPC on this device. A small library and command beside
the libsmctrl fork: libsmctrl confines a stream to a set of TPCs, libgpcmap
says which of them sit together.

Why it exists: a mask bit is one TPC (two SMs), the global TPC id is the
driver's numbering of the enabled TPCs, and which ids share a GPC differs
per product and per chip (round robin over the enabled GPCs with a
product-specific tail on every part measured so far). Mask layouts that keep
a GPC free, or free SMs inside one GPC for a clustered decode kernel, need
the map; it has to be read from the device.

## Methods

- `rm`: the driver's own table through two non-privileged RM controls
  (`NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER`, `..._GET_SM_TO_GPC_TPC_MAPPINGS`).
  Exact, refused on MIG slices, and ABI-bound: build with
  `make RM=1 NV=/path/to/open-gpu-kernel-modules` of the installed driver.
- `nvdebug`: the same table from the hardware, when `nvdebug` kernel module
  is loaded: `/proc/gpuN/CWD_GPC_TPC_ID*` are the work distributor's registers
  the driver programs with its striping of logical
  TPC ids over the GPCs, one byte per TPC with the GPC in the upper nibble.
  Loading the module needs root, reading the files does not; the table reads as
  zero while the GPU is off.
- `cluster`: thread-block clusters (Hopper and later). The 8 CTAs of one
  cluster are resident in one GPC, so the SMs they land on share a GPC;
  clusters are launched with one TPC excluded at a time and the SMs unioned.
  The device holds only so many clusters at once (15 on an H200 of 132 SMs,
  19 on an RTX 5090 of 170), so the clusters seen refine the table rather
  than partition every SM: TPCs left unplaced are settled afterwards, one
  cluster on the TPC plus three TPCs of a candidate GPC, which runs only when
  they share the GPC. A cluster that finds no room pends for good, so the
  command runs each of those tries in a child process with a timeout. A try
  can also pend inside the table's GPC (seen on an H200 for the tail TPCs of
  a GPC: cluster placement is constrained below the GPC), so TPCs may stay
  unplaced; the command then exits 2 and the driver's table is the complete
  answer.
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

  What the groups are depends on the part: on an RTX 5090 the path is the
  GPC's (one streaming TPC slows its whole GPC by 14%, two halves of a GPC
  stream together at 1.4 to 2.0 times their additive time) and the groups are
  the eleven GPCs; on an A100 the path belongs to a sub-group of three or four
  TPCs (pairs couple at 1.52, the sub-group at 1.14, the two halves of a GPC
  not at all, and a same-GPC triple and quad stream together no slower than
  foreign ones), so the groups are the sub-groups and the GPC does not show.
  The command therefore reports a co-run map as shared-path groups, in doubt
  as a GPC map, and exits 2 (a join refused for exceeding the limit is
  reported as well); the JSON says `"groups": "shared_path"`. On Ampere the
  GPC comes from the driver's table or not at all.

## Methods comparison

| | `rm` | `nvdebug` | `cluster` | `corun` |
| --- | --- | --- | --- | --- |
| source | the driver's SM order table, two RM controls | the same table in hardware, the work distributor's `CWD_GPC_TPC_ID` registers through the nvdebug module | where clusters of 8 CTAs land: co-resident in one GPC | timing: streams that share a path to memory slow each other |
| nature | read, exact | read, exact | measured placement, exact where it covers | measured contention, inferred |
| needs | the open-kernel-module SDK headers at build time (`RM=1`); no privilege | the module loaded as root; reading is unprivileged; a context active | Hopper or later; libsmctrl stream masks | libsmctrl stream masks; a quiet GPU |
| refused or absent | on MIG slices | where the module cannot be loaded | Ampere and older | never; finds nothing on Hopper |
| answer | the GPC of every TPC | the GPC of every TPC | the GPC of every TPC the clusters reach; the rest settled one by one, or left unplaced | the groups of TPCs that share a path: the GPCs on GB202, sub-groups of three or four inside a GPC on GA100 |
| cost | milliseconds | milliseconds | 0.6 s on an RTX 5090, 109 s on an H200 | about 4 min on an RTX 5090, 2.5 min on an A100 |
| exit, `groups` | 0, `gpc` | 0, `gpc` | 0, `gpc`; 2 with TPCs left unplaced | 2, `shared_path`; a refused join is counted |
| hazards | ABI tied to the installed driver | PCI-order numbering; zero while the GPU is off | a cluster that finds no room pends for good, hence child processes | drift, size-dependent floors, one-way partner spikes, all handled; still inference |

`rm` and `nvdebug` are the ground truth where available and identical in
content; `cluster` recovers the same map, less the tail TPCs its clusters
never reach, without privileges or headers on Hopper and later;

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

prints the GPCs with their TPC ids and writes the JSON (`gpc_to_tpcs`) that
the geometry suite and slackserve's mask layouts read. `--method` picks one
method, `--no-complement` skips the per-TPC settlement, `--no-validate` the
check; `GPCMAP_DUMP=<prefix>` writes the co-run matrices as CSV. Exit 0 is
a checked GPC map, 2 a map in doubt (every co-run map, a refused join, or TPCs left unplaced), 1 no answer. The C API (`gpcmap_init`, `gpcmap_query`,
`gpcmap_cluster_fits`, `gpcmap_validate`, `gpcmap_write_json`).

## Measured so far

| part | SMs | GPCs enabled (TPCs each) | ids |
| --- | --- | --- | --- |
| A100 SXM4 80GB | 108 | 7: 7, 7, 8, 8, 8, 8, 8 | round robin, the 7-TPC GPCs last; a second A100 identical; `corun` gives the two halves of every GPC |
| H100 SXM 80GB | 132 | 8: 9, 8, 8, 8, 8, 8, 8, 9 | round robin for ids 0-31, seven GPCs for 32-59, leftovers 60-65 |
| H200 SXM 141GB, three chips | 132 | 8: 6, 8, 8, 8, 9, 9, 9, 9 / 7, 8, 8, 8, 8, 9, 9, 9 / 8, 8, 8, 8, 8, 8, 9, 9 | as the H100 for ids 0-59; 60-65 differ per chip, so the map is per device |
| RTX 5090 | 170 | 11 of 12: three of 7, eight of 8 | round robin for ids 0-76, 77-84 to the full GPCs |

On one H200 the cluster method left the four tail TPCs of a GPC unplaced:
a cluster of 8 pends on them with three TPCs of their own GPC, so cluster
placement is constrained below the GPC and the table is the complete
answer there. The co-run finds nothing on Hopper: two TPCs of one GPC
stream as fast together as apart, the GPC's path binds only from four.

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
