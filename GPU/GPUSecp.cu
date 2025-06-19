#pragma diag_suppress 177           // (optional) silence "unused" warnings
#ifndef MAX_REGS_PER_THREAD
#define MAX_REGS_PER_THREAD 96       // 96 regs → 4 blocks/SM on GA100
#endif

#include "GPUSecp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gmp.h>
#include "GPUMath.h"
#include "GPUHash.h"

extern "C" __device__
void _PointMulti4w_inl(uint64_t* __restrict__ qx,
                       uint64_t* __restrict__ qy,
                       uint64_t              k,
                       const uint8_t* __restrict__ gTableX,
                       const uint8_t* __restrict__ gTableY);

// GPU/GPUSecp.cu
// ---- single definitions for the __constant__ tables -----------------
__device__ __constant__ int CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536*0,  65536*1,  65536*2,  65536*3,
  65536*4,  65536*5,  65536*6,  65536*7,
  65536*8,  65536*9,  65536*10, 65536*11,
  65536*12, 65536*13, 65536*14, 65536*15,
};

__device__ __constant__ int MULTI_EIGHT[65] = { 0,
    0 + 8,   0 + 16,   0 + 24,   0 + 32,   0 + 40,   0 + 48,   0 + 56,   0 + 64,
   64 + 8,  64 + 16,  64 + 24,  64 + 32,  64 + 40,  64 + 48,  64 + 56,  64 + 64,
  128 + 8, 128 + 16, 128 + 24, 128 + 32, 128 + 40, 128 + 48, 128 + 56, 128 + 64,
  192 + 8, 192 + 16, 192 + 24, 192 + 32, 192 + 40, 192 + 48, 192 + 56, 192 + 64,
  256 + 8, 256 + 16, 256 + 24, 256 + 32, 256 + 40, 256 + 48, 256 + 56, 256 + 64,
  320 + 8, 320 + 16, 320 + 24, 320 + 32, 320 + 40, 320 + 48, 320 + 56, 320 + 64,
  384 + 8, 384 + 16, 384 + 24, 384 + 32, 384 + 40, 384 + 48, 384 + 56, 384 + 64,
  448 + 8, 448 + 16, 448 + 24, 448 + 32, 448 + 40, 448 + 48, 448 + 56, 448 + 64,
};

__device__ __constant__ uint8_t COMBO_SYMBOLS[COUNT_COMBO_SYMBOLS] = {
  // 0–9
  0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,
  // punctuation
  0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,
  0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,0x40,0x5B,0x5C,0x5D,0x5E,0x5F,0x60,0x7B,0x7C,0x7D,0x7E,
  // A–Z
  0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x4B,0x4C,0x4D,0x4E,0x4F,0x50,
  0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,
  // a–z
  0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x6B,0x6C,0x6D,0x6E,0x6F,0x70,
  0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,
  // special
  0x00,0x7F,0xFF,0x09,0x0D
};

/* ------------------------------------------------------------------ */
/*  Fetch one pre‑computed point from the new tile‑major G‑table.      */
/*  idx            : 0 … COUNT_GTABLE_POINTS‑1                         */
/*  outX/outY[4]   : little‑endian words (ready for field ops)         */
/* ------------------------------------------------------------------ */
static __device__ __forceinline__
void loadGTablePoint(uint32_t            idx,
                     uint64_t            outX[4],
                     uint64_t            outY[4],
                     const uint8_t* __restrict__ gX,
                     const uint8_t* __restrict__ gY)
{
    const uint32_t tile      = idx >> 6;     // /64
    const uint32_t posInside = idx & 63;     // %64
    constexpr uint32_t TILE_BYTES = 2048;    // 4 limbs × 64 × 8 B

    const uint8_t* baseX = gX + tile * TILE_BYTES;
    const uint8_t* baseY = gY + tile * TILE_BYTES;

#pragma unroll
    for (int limb = 0; limb < 4; ++limb) {
        const uint32_t offs = limb * 64 + posInside;    // limb‑major
        // 16‑byte vec4 helps the HW combine neighbouring lanes
        const ulonglong2* vX = reinterpret_cast<const ulonglong2*>(baseX);
        const ulonglong2* vY = reinterpret_cast<const ulonglong2*>(baseY);
        ulonglong2 vx = __ldg(vX + (limb*64 + posInside)/2);
        ulonglong2 vy = __ldg(vY + (limb*64 + posInside)/2);
        const int hi = (offs & 1);              // even/odd element
        outX[limb]   = (hi ? vx.y : vx.x);
        outY[limb]   = (hi ? vy.y : vy.x);
    }
}


using namespace std;

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    printf("cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

// ----------------------------------------------------------------------------
// Read one point from the limb‑major table written by loadGTable().
//
// Layout recap (TILE = 64):
//   tile 0 :  limb0‑P0 … limb0‑P63 | limb1‑P0 … limb1‑P63 | limb2‑ … | limb3‑ …
//   tile 1 :  (next 64 points)                                              etc.
//
// A "point index" therefore maps to:
//   tile  = idx >> 6          ( /64 )
//   row   = idx & 63          ( %64 )
//   limb0 = base + row
//   limb1 = base + 64 + row
//   limb2 = base + 128 + row
//   limb3 = base + 192 + row
// ----------------------------------------------------------------------------
static __device__ __forceinline__
void _gtableLoad4Limbs( const uint8_t *tbl,
                        int            pointIdx,
                        uint64_t       lim[4] )
{
    constexpr int TILE = 64;                     // 64‑point tile
    const int tile = pointIdx >> 6;              // idx / 64
    const int row  = pointIdx & (TILE - 1);      // idx % 64

    // Pointer to limb0 of P0 in this tile, already 8‑byte aligned
    const uint64_t *base = reinterpret_cast<const uint64_t*>(tbl) +
                           static_cast<size_t>(tile) * TILE * 4;

#pragma unroll
    for (int l = 0; l < 4; ++l)
        lim[l] = __ldg(base + l*TILE + row);     // limb‑l, same row
}


//Cuda Secp256k1 Point Multiplication
//Takes 32-byte privKey + gTable and outputs 64-byte public key [qx,qy]
// after  (add const & restrict; pointers become read‑only)
static __device__ __forceinline__
void _PointMultiSecp256k1_inl(uint64_t* __restrict__ qx,
                          uint64_t* __restrict__ qy,
                          const uint16_t* __restrict__ privKey,
                          const uint8_t*  __restrict__ gTableX,
                          const uint8_t*  __restrict__ gTableY)
{
    int      chunk = 0;
    uint64_t qz_buf[5] = {1, 0, 0, 0, 0};
    uint64_t __restrict__ *qz = qz_buf;
    uint64_t __restrict__ *qxLoc = qx;
    uint64_t __restrict__ *qyLoc = qy;

    // ── 1) first non‑zero window → initialise (qx,qy) ─────────────────
    for (; chunk < NUM_GTABLE_CHUNK; ++chunk)
    {
        uint16_t w = privKey[chunk];
        if (w == 0) continue;

        int idx = CHUNK_FIRST_ELEMENT[chunk] + (w - 1);

        _gtableLoad4Limbs(gTableX, idx, qx);
        _gtableLoad4Limbs(gTableY, idx, qy);

        ++chunk;                              // start adding with next chunk
        break;
    }

    // ── 2) add the remaining non‑zero windows ─────────────────────────
    for (; chunk < NUM_GTABLE_CHUNK; ++chunk)
    {
        uint16_t w = privKey[chunk];
        if (w == 0) continue;

        int idx = CHUNK_FIRST_ELEMENT[chunk] + (w - 1);

        uint64_t gx[4], gy[4];
        _gtableLoad4Limbs(gTableX, idx, gx);
        _gtableLoad4Limbs(gTableY, idx, gy);

        _PointAddSecp256k1(qxLoc, qyLoc, qz, gx, gy);
    }

    // ── 3) convert from Jacobian (qx,qy,qz) to affine ─────────────────
    _ModInv(qz);
    _ModMult(qxLoc, qz);
    _ModMult(qyLoc, qz);
}


// Computes pub-key X for sequential priv-keys: start + thread_id
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void CudaRunSecp256k1Sequential(
        uint64_t start,
        uint8_t *gTableXGPU, uint8_t *gTableYGPU,
        uint8_t *outPrivGPU, uint8_t *outPubXGPU)
{
    const uint64_t k = start + IDX_CUDA_THREAD;   // 64‑bit private key

    /* ---- scalar multiply (NEW 4‑window path) ----------------------- */
    uint64_t qx[4], qy[4];
    _PointMulti4w_inl(qx, qy, k, gTableXGPU, gTableYGPU);

    /* ---- serialise k (big‑endian, 32 bytes) for host output -------- */
    uint8_t privKey[SIZE_PRIV_KEY] = {0};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        privKey[31 - i] = (k >> (8 * i)) & 0xFF;

    /* ---- write results to global memory ---------------------------- */
    const int tid = IDX_CUDA_THREAD;
#pragma unroll
    for (int i = 0; i < SIZE_PRIV_KEY; ++i)
        outPrivGPU[tid * SIZE_PRIV_KEY + i] = privKey[i];

    const uint8_t *qxBytes = reinterpret_cast<uint8_t*>(qx);
#pragma unroll
    for (int i = 0; i < 32; ++i)
        outPubXGPU[tid * 32 + i] = qxBytes[i];
}


GPUSecp::GPUSecp(
  	int countPrime, 
		int countAffix,
    const uint8_t *gTableXCPU,
    const uint8_t *gTableYCPU,
		const uint8_t * inputBookPrimeCPU, 
		const uint8_t * inputBookAffixCPU, 
    const uint64_t *inputHashBufferCPU
    )
{
  printf("GPUSecp Starting\n");

  int gpuId = 0; // FOR MULTIPLE GPUS EDIT THIS
  CudaSafeCall(cudaSetDevice(gpuId));

  cudaDeviceProp deviceProp;
  CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

  printf("GPU.gpuId: #%d \n", gpuId);
  printf("GPU.deviceProp.name: %s \n", deviceProp.name);
  printf("GPU.multiProcessorCount: %d \n", deviceProp.multiProcessorCount);
  printf("GPU.BLOCKS_PER_GRID: %d \n", BLOCKS_PER_GRID);
  printf("GPU.THREADS_PER_BLOCK: %d \n", THREADS_PER_BLOCK);
  printf("GPU.CUDA_THREAD_COUNT: %d \n", COUNT_CUDA_THREADS);
  printf("GPU.countHash160: %d \n", COUNT_INPUT_HASH);
  printf("GPU.countPrime: %d \n", countPrime);
  printf("GPU.countAffix: %d \n", countAffix);

  if (countPrime > 0 && countPrime != COUNT_INPUT_PRIME) {
    printf("ERROR: countPrime must be equal to COUNT_INPUT_PRIME \n");
    printf("Please edit GPUSecp.h configuration and set COUNT_INPUT_PRIME to %d \n", countPrime);
    exit(-1);
  }

  CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, SIZE_CUDA_STACK));

  size_t limit = 0;
  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
  printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

  if (countPrime > 0) {
    printf("Allocating inputBookPrime \n");
    CudaSafeCall(cudaMalloc((void **)&inputBookPrimeGPU, countPrime * MAX_LEN_WORD_PRIME));
    CudaSafeCall(cudaMemcpy(inputBookPrimeGPU, inputBookPrimeCPU, countPrime * MAX_LEN_WORD_PRIME, cudaMemcpyHostToDevice));

    printf("Allocating inputBookAffix \n");
    CudaSafeCall(cudaMalloc((void **)&inputBookAffixGPU, countAffix * MAX_LEN_WORD_AFFIX));
    CudaSafeCall(cudaMemcpy(inputBookAffixGPU, inputBookAffixCPU, countAffix * MAX_LEN_WORD_AFFIX, cudaMemcpyHostToDevice));
  } else {
    printf("Allocating inputCombo buffer \n");
    CudaSafeCall(cudaMalloc((void **)&inputComboGPU, SIZE_COMBO_MULTI));
  }
  
  printf("Allocating inputHashBuffer \n");
  CudaSafeCall(cudaMalloc((void **)&inputHashBufferGPU, COUNT_INPUT_HASH * SIZE_LONG));
  CudaSafeCall(cudaMemcpy(inputHashBufferGPU, inputHashBufferCPU, COUNT_INPUT_HASH * SIZE_LONG, cudaMemcpyHostToDevice));

  printf("Allocating gTableX \n");
  CudaSafeCall(cudaMalloc((void **)&gTableXGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemset(gTableXGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemcpy(gTableXGPU, gTableXCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));

  printf("Allocating gTableY \n");
  CudaSafeCall(cudaMalloc((void **)&gTableYGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemset(gTableYGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemcpy(gTableYGPU, gTableYCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));

  printf("Allocating outputBuffer \n");
  CudaSafeCall(cudaMalloc((void **)&outputBufferGPU, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaHostAlloc(&outputBufferCPU, COUNT_CUDA_THREADS, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("Allocating outputHashes \n");
  CudaSafeCall(cudaMalloc((void **)&outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaHostAlloc(&outputHashesCPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("Allocating outputPrivKeys \n");
  CudaSafeCall(cudaMalloc((void **)&outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));
  CudaSafeCall(cudaHostAlloc(&outputPrivKeysCPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("Allocation Complete \n");
  CudaSafeCall(cudaGetLastError());
}

// ────────────────────────────────────────────────────────────────────
// 4‑window specialisation for k < 2⁶⁴
//   – uses the last four 16‑bit windows (chunks 12‑15)
//   – no 32‑byte temp key, no loop, no branches inside the hot path
//   – keeps all temporaries (qx,qy,gx,gy,qz) in registers
// ────────────────────────────────────────────────────────────────────
extern "C" __device__
void _PointMulti4w_inl(uint64_t* __restrict__ qx,
                       uint64_t* __restrict__ qy,
                       uint64_t              k,
                       const uint8_t* __restrict__ gTableX,
                       const uint8_t* __restrict__ gTableY)
{
    /* split k into 4 little‑endian 16‑bit windows */
    const uint16_t w0 =  k        & 0xFFFF;          // chunk 15
    const uint16_t w1 = (k >> 16) & 0xFFFF;          // chunk 14
    const uint16_t w2 = (k >> 32) & 0xFFFF;          // chunk 13
    const uint16_t w3 = (k >> 48) & 0xFFFF;          // chunk 12

    /* pre‑computed chunk bases:  (15‑i)*65536  for i = {0,1,2,3}       */

    const int base[4] = { 983040, 917504, 851968, 786432 };
    const uint16_t win[4] = { w0, w1, w2, w3 };

    uint64_t qz[5] = {1,0,0,0,0};                    // Jacobian Z
    bool     havePoint = false;

#pragma unroll
    for (int i = 3; i >= 0; --i)                     // chunks 12 → 15
    {
        const uint16_t w = win[i];
        if (w == 0) continue;

        uint64_t gx[4], gy[4];
        loadGTablePoint(base[i] + (w - 1), gx, gy, gTableX, gTableY);
        if (!havePoint) {                            // first non‑zero window
#pragma unroll
            for (int limb = 0; limb < 4; ++limb) {
                qx[limb] = gx[limb];
                qy[limb] = gy[limb];
            }
            havePoint = true;
        } else {
            _PointAddSecp256k1(qx, qy, qz, gx, gy);  // Jacobian add
        }
    }

    if (!havePoint) {                                // k == 0
#pragma unroll
        for (int limb = 0; limb < 4; ++limb) {
            qx[limb] = qy[limb] = 0;
        }
        return;
    }

    /* convert (qx,qy,qz) → affine */
    _ModInv(qz);
    _ModMult(qx, qz);
    _ModMult(qy, qz);
}



extern "C" __device__ __noinline__
void _PointMultiSecp256k1(uint64_t*        qx,
                          uint64_t*        qy,
                          const uint16_t*  privKey,
                          const uint8_t* __restrict__ gTableX,
                          const uint8_t* __restrict__ gTableY)
{
    _PointMultiSecp256k1_inl(qx, qy, privKey, gTableX, gTableY);
}

//GPU kernel function for computing Secp256k1 public key from input books
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void
CudaRunSecp256k1Books(
    int iteration, uint8_t * gTableXGPU, uint8_t * gTableYGPU,
    uint8_t *inputBookPrimeGPU, uint8_t *inputBookAffixGPU, uint64_t *inputHashBufferGPU,
    uint8_t *outputBufferGPU, uint8_t *outputHashesGPU, uint8_t *outputPrivKeysGPU) {

  //Load affix word from global memory based on thread index
  uint32_t offsetAffix = (COUNT_CUDA_THREADS * iteration * MAX_LEN_WORD_AFFIX) + (IDX_CUDA_THREAD * MAX_LEN_WORD_AFFIX);
  uint8_t wordAffix[MAX_LEN_WORD_AFFIX];
  uint8_t privKey[SIZE_PRIV_KEY];
  uint8_t sizeAffix = inputBookAffixGPU[offsetAffix];
  for (uint8_t i = 0; i < sizeAffix; i++) {
    wordAffix[i] = inputBookAffixGPU[offsetAffix + i + 1];
  }
  
  for (int idxPrime = 0; idxPrime < COUNT_INPUT_PRIME; idxPrime++) {
  
  _SHA256Books((uint32_t *)privKey, inputBookPrimeGPU, wordAffix, sizeAffix, idxPrime);

    uint64_t qx[4];
    uint64_t qy[4];

    _PointMultiSecp256k1_inl(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

    uint8_t hash160[SIZE_HASH160];
    uint64_t hash160Last8Bytes;

    _GetHash160Comp(qx, (uint8_t)(qy[0] & 1), hash160);
    GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

    if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
      int idxCudaThread = IDX_CUDA_THREAD;
      outputBufferGPU[idxCudaThread] += 1;
      for (int i = 0; i < SIZE_HASH160; i++) {
        outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
      }
      for (int i = 0; i < SIZE_PRIV_KEY; i++) {
        outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
      }
    }
    
    _GetHash160(qx, qy, hash160);
    GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

    if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
      int idxCudaThread = IDX_CUDA_THREAD;
      outputBufferGPU[idxCudaThread] += 1;
      for (int i = 0; i < SIZE_HASH160; i++) {
        outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
      }
      for (int i = 0; i < SIZE_PRIV_KEY; i++) {
        outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
      }
    }
  }
}

__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void CudaRunSecp256k1Combo(
    int8_t * inputComboGPU, uint8_t * gTableXGPU, uint8_t * gTableYGPU, uint64_t *inputHashBufferGPU,
    uint8_t *outputBufferGPU, uint8_t *outputHashesGPU, uint8_t *outputPrivKeysGPU) {

  int8_t combo[SIZE_COMBO_MULTI] = {};
  _FindComboStart(inputComboGPU, combo);

  for (combo[0] = 0; combo[0] < COUNT_COMBO_SYMBOLS; combo[0]++) {
    for (combo[1] = 0; combo[1] < COUNT_COMBO_SYMBOLS; combo[1]++) {

      uint8_t privKey[SIZE_PRIV_KEY];
      _SHA256Combo((uint32_t *)privKey, combo);

      uint64_t qx[4];
      uint64_t qy[4];

      _PointMultiSecp256k1_inl(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

      uint8_t hash160[SIZE_HASH160];
      uint64_t hash160Last8Bytes;

      _GetHash160Comp(qx, (uint8_t)(qy[0] & 1), hash160);
      GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

      if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
        int idxCudaThread = IDX_CUDA_THREAD;
        outputBufferGPU[idxCudaThread] += 1;
        for (int i = 0; i < SIZE_HASH160; i++) {
          outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
        }
        for (int i = 0; i < SIZE_PRIV_KEY; i++) {
          outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
        }
      }
      
      _GetHash160(qx, qy, hash160);
      GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

      if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {
        int idxCudaThread = IDX_CUDA_THREAD;
        outputBufferGPU[idxCudaThread] += 1;
        for (int i = 0; i < SIZE_HASH160; i++) {
          outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
        }
        for (int i = 0; i < SIZE_PRIV_KEY; i++) {
          outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
        }
      }
    }
  }
}


void GPUSecp::doIterationSecp256k1Books(int iteration) {
  CudaSafeCall(cudaMemset(outputBufferGPU, 0, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaMemset(outputHashesGPU, 0, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaMemset(outputPrivKeysGPU, 0, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));

  CudaRunSecp256k1Books<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    iteration, gTableXGPU, gTableYGPU,
    inputBookPrimeGPU, inputBookAffixGPU, inputHashBufferGPU,
    outputBufferGPU, outputHashesGPU, outputPrivKeysGPU);

  CudaSafeCall(cudaMemcpy(outputBufferCPU, outputBufferGPU, COUNT_CUDA_THREADS, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputHashesCPU, outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputPrivKeysCPU, outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaGetLastError());
}

void GPUSecp::doIterationSecp256k1Combo(int8_t * inputComboCPU) {
  CudaSafeCall(cudaMemset(outputBufferGPU, 0, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaMemset(outputHashesGPU, 0, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaMemset(outputPrivKeysGPU, 0, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));

  CudaSafeCall(cudaMemcpy(inputComboGPU, inputComboCPU, SIZE_COMBO_MULTI, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaGetLastError());

  CudaRunSecp256k1Combo<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    inputComboGPU, gTableXGPU, gTableYGPU, inputHashBufferGPU,
    outputBufferGPU, outputHashesGPU, outputPrivKeysGPU);

  CudaSafeCall(cudaMemcpy(outputBufferCPU, outputBufferGPU, COUNT_CUDA_THREADS, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputHashesCPU, outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputPrivKeysCPU, outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaGetLastError());
}


// ─── GPU/GPUSecp.cu ────────────────────────────────────────────────────
void GPUSecp::doPrintOutput()
{
    for (int idxThread = 0; idxThread < COUNT_CUDA_THREADS; ++idxThread)
    {
        if (outputBufferCPU[idxThread] == 0)        // no hit for this slot
            continue;

        // Pointer to the 32-byte private key for this thread
        const uint8_t *pkBytes = outputPrivKeysCPU + (idxThread * SIZE_PRIV_KEY);

        // Convert to a big integer using GMP
        mpz_t pkInt;
        mpz_init(pkInt);
        mpz_import(pkInt,             // destination
                   SIZE_PRIV_KEY,     // 32 bytes
                   1,                 // most-significant word first
                   1,                 // word size = 1 byte
                   0, 0,              // big-endian, no nails
                   pkBytes);          // source buffer

        char *decStr = mpz_get_str(nullptr, 10, pkInt);   // base-10 string

        // Print to the console
        printf("%s\n", decStr);

        // Append to the output file
        FILE *file = fopen(NAME_FILE_OUTPUT, "a");
        if (file)
        {
            fprintf(file, "%s\n", decStr);
            fclose(file);
        }

        // Clean up
        free(decStr);
        mpz_clear(pkInt);
    }
}


void GPUSecp::doFreeMemory() {
  printf("\nGPUSecp Freeing memory... ");

  CudaSafeCall(cudaFree(inputComboGPU));
  CudaSafeCall(cudaFree(inputBookPrimeGPU));
  CudaSafeCall(cudaFree(inputBookAffixGPU));
  CudaSafeCall(cudaFree(inputHashBufferGPU));

  CudaSafeCall(cudaFree(gTableXGPU));
  CudaSafeCall(cudaFree(gTableYGPU));

  CudaSafeCall(cudaFreeHost(outputBufferCPU));
  CudaSafeCall(cudaFree(outputBufferGPU));

  CudaSafeCall(cudaFreeHost(outputHashesCPU));
  CudaSafeCall(cudaFree(outputHashesGPU));

  CudaSafeCall(cudaFreeHost(outputPrivKeysCPU));
  CudaSafeCall(cudaFree(outputPrivKeysGPU));

  printf("Done \n");
}
