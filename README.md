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
- `corun`: streaming kernels contend for the GPC's path to memory, so a
  victim slows more beside an aggressor in its own GPC. The method is
  agglomerative: clusters start as single TPCs; each round every cluster
  streams as the aggressor against every other cluster as the victim, a row's
  slowdowns split into a slow mode (gap of 3% or more) and the rest, mutually
  slow clusters merge (a row with no split defers to a bigger aggressor's
  row), and the merged clusters press harder in the next round; a merge
  beyond ten TPCs ends the method with no answer. Timing only, so the
  fallback where clusters do not exist (Ampere) or the table is refused (MIG
  slices). On the RTX 5090 one streaming TPC already slows its whole GPC
  (12 to 13%) and the first round yields the map; on an A100 one TPC couples
  to one partner only and the pairs find the rest of their GPC in later
  rounds.

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
check. The C API (`gpcmap_init`, `gpcmap_query`,
`gpcmap_cluster_fits`, `gpcmap_validate`, `gpcmap_write_json`).

## Measured so far

| part | SMs | GPCs enabled (TPCs each) | ids |
| --- | --- | --- | --- |
| A100 SXM4 80GB | 108 | 7: 7, 7, 8, 8, 8, 8, 8 | round robin, the 7-TPC GPCs last |
| H100 SXM 80GB | 132 | 8: 9, 8, 8, 8, 8, 8, 8, 9 | round robin for ids 0-31, seven GPCs for 32-59, leftovers 60-65 |
| RTX 5090 | 170 | 11 of 12: three of 7, eight of 8 | round robin for ids 0-76, 77-84 to the full GPCs |
