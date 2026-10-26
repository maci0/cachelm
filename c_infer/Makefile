# Bare-metal LL3M inference
#
# Cache-minded build defaults:
#   -O3 + native AVX-512/VNNI
#   LTO + gc-sections  → tighter L1i hot path
#   align functions/loops to 64/32 → fewer I-cache splits
#   optional PGO: make pgo MODEL=... N=200
#
# Secondary packs (opt-in; default remains external -m model.ll3m):
#   make embed EMBED_MODEL=../checkpoints/v76_dense_c/best.ll3m
#   make static              # fully static libc link (no embedded weights)
#   make static-embed EMBED_MODEL=...
#
CC      ?= gcc
# CLM_QUANT=2 ternary Linears (2-bit); CLM_EMBED_QUANT=1 int8 embeds (product)
# mlock ON by default; disable with: make CFLAGS='... -DCLM_NO_MLOCK'

# x86: AVX-512/VNNI defaults. aarch64 (e.g. spark GB10): native only (no -mavx*).
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCH_ISA ?= -mavx512f -mavx512vnni -mfma
else
  ARCH_ISA ?=
endif

OPTFLAGS ?= -O3 -march=native -mtune=native $(ARCH_ISA) \
	-ffast-math -fno-math-errno -fno-trapping-math

# I-cache / binary size: LTO + dead code drop + hot entry alignment
ICACHEFLAGS ?= -flto=auto \
	-ffunction-sections -fdata-sections \
	-falign-functions=64 -falign-loops=32 -falign-jumps=32 \
	-fvisibility=hidden -fno-semantic-interposition

CFLAGS  ?= $(OPTFLAGS) $(ICACHEFLAGS) -std=c11 -Wall -Wextra -pthread \
	-DCLM_QUANT=2 -DCLM_EMBED_QUANT=1
LDFLAGS ?= -lm -pthread -flto=auto -Wl,--gc-sections -Wl,-O1

# Fully static link (secondary). Needs static libc/libm on the host.
STATIC_LDFLAGS ?= -static -lm -pthread -flto=auto -Wl,--gc-sections -Wl,-O1

# mlock is ON by default; disable with: make CFLAGS='... -DCLM_NO_MLOCK'
# Prefetch knobs (ll3m_v2.c / docs/READAHEAD_DESIGN.md):
#   -DCLM_W_PREFETCH_AHEAD=32  -DCLM_W_PREFETCH_HINT=0|1|2  -DCLM_NEXT_LAYER_PREFETCH=1
# Grid: bash scripts/prefetch_grid.sh

BIN = ll3m_c
MULTI = multi_bench
SRC = ll3m.c ll3m_v2.c main.c
HDR = ll3m.h
LIBSRC = ll3m.c ll3m_v2.c

# Embedded pack (secondary). Copy EMBED_MODEL → clm_pack.ll3m then objcopy to .o
EMBED_MODEL ?=
PACK_BIN = clm_pack.ll3m
PACK_OBJ = clm_pack.o
OBJCOPY ?= objcopy

# PGO paths
PROF_DIR ?= .pgo
PROF_DATA ?= $(PROF_DIR)/default.profdata
MODEL ?= ../checkpoints/v76_dense_c/best.ll3m
PGO_N ?= 200
PGO_TOKEN_IDS ?= 12,34,56,78,90,100,110,120,130,140,150,160

.PHONY: all clean run export bench cachestat l3 pgo pgo-gen pgo-run pgo-use size \
	embed static static-embed pack-obj multi

all: $(BIN) $(MULTI)

multi: $(MULTI)

$(MULTI): multi_bench.c $(LIBSRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ multi_bench.c $(LIBSRC) $(LDFLAGS)

$(BIN): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# --- Secondary: embed .ll3m into the binary (still dynamically linked by default) ---
#   make embed EMBED_MODEL=../checkpoints/v76_dense_c/best.ll3m
# Run without -m, or with -m :embedded:
pack-obj: $(PACK_OBJ)

$(PACK_BIN):
	@test -n "$(EMBED_MODEL)" || (echo "set EMBED_MODEL=path/to/model.ll3m"; exit 1)
	@test -f "$(EMBED_MODEL)" || (echo "missing EMBED_MODEL=$(EMBED_MODEL)"; exit 1)
	cp -f "$(EMBED_MODEL)" $(PACK_BIN)
	@ls -la $(PACK_BIN)

# GNU objcopy binary → ELF .o with symbols:
#   _binary_clm_pack_ll3m_start / _binary_clm_pack_ll3m_end
$(PACK_OBJ): $(PACK_BIN)
	$(OBJCOPY) -I binary -O elf64-x86-64 -B i386:x86-64 \
		--rename-section .data=.rodata,alloc,load,readonly,data,contents \
		$(PACK_BIN) $(PACK_OBJ)
	@nm $(PACK_OBJ) | head -5 || true

embed: $(SRC) $(HDR) $(PACK_OBJ)
	$(CC) $(CFLAGS) -DCLM_EMBED_WEIGHTS -o $(BIN) $(SRC) $(PACK_OBJ) $(LDFLAGS)
	@echo "built embedded $(BIN) from EMBED_MODEL=$(EMBED_MODEL)"
	@ls -la $(BIN)
	@size $(BIN) 2>/dev/null || true

# Fully static binary, external -m model.ll3m still required
static: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(STATIC_LDFLAGS)
	@echo "built static $(BIN)"
	@file $(BIN)
	@ls -la $(BIN)

# Fully static + embedded weights (single-file ship)
static-embed: $(SRC) $(HDR) $(PACK_OBJ)
	$(CC) $(CFLAGS) -DCLM_EMBED_WEIGHTS -o $(BIN) $(SRC) $(PACK_OBJ) $(STATIC_LDFLAGS)
	@echo "built static+embedded $(BIN) from EMBED_MODEL=$(EMBED_MODEL)"
	@file $(BIN)
	@ls -la $(BIN)
	@size $(BIN) 2>/dev/null || true

# --- Profile-guided optimization (best L1i packing for real decode) ---
#   make pgo MODEL=../checkpoints/best_L6.clm PGO_N=200
# Use the same -o name for gen/use so GCC .gcda basenames match.
pgo: clean-prof pgo-instrument pgo-train pgo-apply size

clean-prof:
	rm -f $(BIN)-*.gcda *.gcda *.gcno $(BIN).pgo-gen

pgo-instrument:
	$(CC) $(CFLAGS) -fprofile-generate -o $(BIN) $(SRC) $(LDFLAGS) -fprofile-generate
	@echo "instrumented $(BIN) — next trains on MODEL=$(MODEL)"

pgo-train: $(BIN)
	@test -f $(MODEL) || (echo "missing MODEL=$(MODEL)"; exit 1)
	# Product path: Python BPE → --token-ids (no C encode). Profile the hard decode path.
	taskset -c 8 ./$(BIN) -m $(MODEL) --token-ids $(PGO_TOKEN_IDS) -n $(PGO_N) \
		--greedy --residency-hard --ccd 1 --warm 16 --steady-warm 32 --no-cache-stats \
		>/dev/null || true
	@ls -la $(BIN)-*.gcda 2>/dev/null || ls -la *.gcda 2>/dev/null || true
	@echo "profile trained"

pgo-apply:
	$(CC) $(CFLAGS) -fprofile-use -fprofile-correction \
		-Wno-missing-profile \
		-o $(BIN) $(SRC) $(LDFLAGS) -fprofile-use
	@echo "built PGO $(BIN)"

size: $(BIN)
	@size $(BIN) 2>/dev/null || true
	@ls -la $(BIN)
	@echo "sections:"
	@objdump -h $(BIN) 2>/dev/null | grep -E 'text|hot|rodata' || true

export:
	cd .. && uv run python scripts/export_weights.py \
		--checkpoint checkpoints/best.pt \
		--out checkpoints/best.clm

run: $(BIN)
	./$(BIN) -m ../checkpoints/best_L6.clm -p "ROMEO:\n" -n 200 --seed 1 --residency-hard

bench: $(BIN)
	./$(BIN) -m ../checkpoints/best_L6.clm -p "ROMEO:\n" -n 1 --bench --greedy --residency-hard

# Hardware proof: DRAM share of L2 fills near 0 + WS fits CCD
cachestat: $(BIN)
	./measure_cache.sh ../checkpoints/best_L6.clm 200

l3: cachestat

clean: clean-prof
	rm -f $(BIN) $(MULTI) $(PACK_BIN) $(PACK_OBJ)
	rm -rf $(PROF_DIR)
