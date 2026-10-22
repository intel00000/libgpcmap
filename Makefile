# libgpcmap: construct GPC map of a device (see gpcmap.h).
#   make                 libgpcmap.so and the gpcmap command (CUDA probes)
#   make RM=1 NV=<dir>   also the driver's own table, built against the
#                        open-gpu-kernel-modules tree of the installed driver
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCHS ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 \
         -gencode arch=compute_120,code=compute_120
CFLAGS ?= -O2 -Wall
RM ?= 0
NV ?= /usr/src/open-gpu-kernel-modules
ifeq ($(RM),1)
RMFLAGS := -DGPCMAP_RM -I$(NV)/src/common/sdk/nvidia/inc -I$(NV)/src/nvidia/arch/nvalloc/unix/include \
           -I$(NV)/kernel-open/common/inc
endif

all: libgpcmap.so gpcmap

libgpcmap.so: src/gpcmap.cu src/gpcmap_rm.c include/gpcmap.h rmflags
	$(NVCC) $(ARCHS) -O2 -std=c++17 -Iinclude -Xcompiler -fPIC -shared -o $@ src/gpcmap.cu src/gpcmap_rm.c $(RMFLAGS) -ldl -lcuda

rmflags: FORCE
	@echo '$(RMFLAGS)' | cmp -s - $@ || echo '$(RMFLAGS)' > $@

gpcmap: src/gpcmap_cli.c libgpcmap.so include/gpcmap.h
	$(CC) $(CFLAGS) -Iinclude -o $@ src/gpcmap_cli.c -L. -lgpcmap -Wl,-rpath,'$$ORIGIN'

clean:
	rm -f libgpcmap.so gpcmap rmflags

.PHONY: all clean FORCE
