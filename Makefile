##############################################################################
# Paths – edit CUDA_HOME if nvcc lives elsewhere
##############################################################################
CUDA_HOME_RAW := /usr
CUDA_HOME     := $(strip $(CUDA_HOME_RAW))

##############################################################################
# Compilers and flags
##############################################################################
# ---‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ add this line ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
MAX_REGS_PER_THREAD := 96
# ------------------------------------------------------------
CXX        := g++
NVCC       := $(CUDA_HOME)/bin/nvcc

GEN80      := -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_80,code=compute_80

CXXFLAGS := -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA_HOME)/include


NVCCFLAGS := $(GEN80) -O3 --use_fast_math -lineinfo \
             -Xptxas -O3,-dlcm=ca,-v \
             -maxrregcount=$(MAX_REGS_PER_THREAD)
# NVCCFLAGS += -maxrregcount=128     # uncomment while tuning

LFLAGS     := -lgmp -lpthread -L$(CUDA_HOME)/lib64 -lcudart

##############################################################################
# Sources
##############################################################################
CPPSRC := CudaBrainSecp.cpp \
          CPU/Point.cpp CPU/Int.cpp CPU/IntMod.cpp CPU/SECP256K1.cpp

CUSRC  := GPU/GPUSecp.cu seq_gpu.cu

##############################################################################
# Derived paths
##############################################################################
OBJDIR  := obj
CPPOBJ  := $(addprefix $(OBJDIR)/,$(CPPSRC:.cpp=.o))
CUOBJ   := $(addprefix $(OBJDIR)/,$(CUSRC:.cu=.o))

##############################################################################
# Top‑level targets
##############################################################################
.PHONY: all
all: CudaBrainSecp SeqGPU

##############################################################################
# Ensure obj directories exist
##############################################################################
$(OBJDIR):
	@mkdir -p $(OBJDIR) $(OBJDIR)/CPU $(OBJDIR)/GPU

##############################################################################
# *** CUDA compile rule BEFORE generic C++ rule ***
##############################################################################
$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	$(NVCC) -rdc=true $(NVCCFLAGS) -I. -c $< -o $@ -maxrregcount=$(MAX_REGS_PER_THREAD)

# Generic C++ rule (will no longer match seq_gpu.cu)
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

##############################################################################
# Link
##############################################################################
CudaBrainSecp: $(CPPOBJ) $(OBJDIR)/GPU/GPUSecp.o
	$(NVCC) $(NVCCFLAGS) $^ $(LFLAGS) -o $@ -maxrregcount=$(MAX_REGS_PER_THREAD)

SeqGPU: $(OBJDIR)/seq_gpu.o \
        $(OBJDIR)/GPU/GPUSecp.o \
        $(OBJDIR)/CPU/Point.o \
        $(OBJDIR)/CPU/Int.o \
        $(OBJDIR)/CPU/IntMod.o \
        $(OBJDIR)/CPU/SECP256K1.o
	$(NVCC) $(NVCCFLAGS) $^ $(LFLAGS) -o $@ -maxrregcount=$(MAX_REGS_PER_THREAD)

##############################################################################
# Clean
##############################################################################
.PHONY: clean
clean:
	@echo "Cleaning…"
	@rm -rf $(OBJDIR) SeqGPU CudaBrainSecp gtable_le.bin \
	        checkpoint.txt seq_gpu.o || true
