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
- `cluster`: thread-block clusters (Hopper and later). The 8 CTAs of one
  cluster are resident in one GPC, so the SMs they land on share a GPC;
  clusters are launched with one TPC excluded at a time and the SMs unioned.
  The device holds only so many clusters at once (15 on an H200 of 132 SMs,
  19 on an RTX 5090 of 170), so the clusters seen refine the table rather
  than partition every SM: TPCs left unplaced are settled afterwards, one
  cluster on the TPC plus three TPCs of a candidate GPC, which runs only when
  they share the GPC. A cluster that finds no room pends for good, so the
  command runs each of those tries in a child process with a timeout.
- `corun`: because streaming kernels share a GPC's path to memory, this
  method proceeds in two stages.
  
  - First, each TPC streams against every other TPC. The victim's runtime
    with an aggressor is compared with its solo runtime. In each row, a
    plateau of nearly equal slowdowns identifies the closest peers, and
    mutually slowing TPCs are merged.

  - Next, pairs of clusters stream together, and their runtime is compared with
    the sum of their separate rates.
    Clusters in the same GPC share a path and therefore run slower together;
    unrelated pairs show only the device-wide effect, measured using foreign
    stand-ins of the partner's size.
    A join requires at least 10% overhead over the additive time and 5% overhead
    over the stand-ins in both directions.

    Merges are selected by strength, subject to the ten-TPC limit, and repeated
    until no more merges are possible. If a join exceed a GPC's capacity, the map
    is considered uncertain; the command reports this and exits 2. This is a
    timing-only method and serves as the fallback when clusters are unavailable
    (Ampere) or the driver table is refused (MIG slices).

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
a checked answer, 2 an answer in doubt, 1 no answer. The C API (`gpcmap_init`, `gpcmap_query`,
`gpcmap_cluster_fits`, `gpcmap_validate`, `gpcmap_write_json`).

## Measured so far

| part | SMs | GPCs enabled (TPCs each) | ids |
| --- | --- | --- | --- |
| A100 SXM4 80GB | 108 | 7: 7, 7, 8, 8, 8, 8, 8 | round robin, the 7-TPC GPCs last |
| H100 SXM 80GB | 132 | 8: 9, 8, 8, 8, 8, 8, 8, 9 | round robin for ids 0-31, seven GPCs for 32-59, leftovers 60-65 |
| RTX 5090 | 170 | 11 of 12: three of 7, eight of 8 | round robin for ids 0-76, 77-84 to the full GPCs |
