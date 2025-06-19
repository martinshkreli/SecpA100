#pragma diag_suppress 177           // (optional) silence "unused" warnings
#ifndef MAX_REGS_PER_THREAD
#define MAX_REGS_PER_THREAD 96       // 96 regs → 4 blocks/SM on GA100
#endif

// seq_gpu.cu  ───────────────────────────────────────────────────────────────
// GPU tool (A100 sm_80): two modes—
//   1) Throughput mode: ./SeqGPU N [--continue] [--verbose]
//      - Measures how many k → (k·G) operations per second for k = 1..N.
//      - If you pass --continue, it reads/writes "checkpoint.txt" to resume.
//      - If you pass --verbose, it prints the first 10 (k, pubX) so you can verify.
//   2) Print mode:      ./SeqGPU print k1 [k2 k3 ...]
//      - Computes and prints "PRIV k   PUB_X <decimal>" for each specified k.
//───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <array>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include "GPU/GPUSecp.h"      // NUM_GTABLE_CHUNK, NUM_GTABLE_VALUE, THREADS_PER_BLOCK, etc.
#include "CPU/SECP256k1.h"    // CPU class that builds secp.GTable under the hood

//───────────────────────────────────────────────────────────────────────────
// loadGTable(...): copy exactly TOTAL_POINTS = NUM_GTABLE_CHUNK×(NUM_GTABLE_VALUE−1)
// from secp->GTable[] into gX/gY, reversing each 32 bytes so GPU sees
// little‐endian 4×uint64_t (instead of big‐endian).
//───────────────────────────────────────────────────────────────────────────
extern "C" __device__ __noinline__
void _PointMulti4w_inl(uint64_t* __restrict__ qx,
                       uint64_t* __restrict__ qy,
                       uint64_t              k,
                       const uint8_t* __restrict__ gTableX,
                       const uint8_t* __restrict__ gTableY);

static void loadGTable(uint8_t* gX, uint8_t* gY)
{
    Secp256K1* secp = new Secp256K1();
    secp->Init();

    /* --------------------------------------------------------------------
     *  Limb‑major “64‑point tiles”
     *  – Each tile: 64 points × 4 limbs × 8 B  = 2048 B for X, same for Y
     *  – Gives fully‑coalesced LDGSTS.64 loads on the GPU.
     * ------------------------------------------------------------------ */

    constexpr int POINTS_PER_CHUNK = (NUM_GTABLE_VALUE - 1);     // 65 535
    constexpr int TOTAL_POINTS     = NUM_GTABLE_CHUNK * POINTS_PER_CHUNK;
    constexpr int TILE             = 64;

    std::array< std::array<uint64_t,4>, TILE > bufX;
    std::array< std::array<uint64_t,4>, TILE > bufY;

    for (int base = 0; base < TOTAL_POINTS; base += TILE) {

        /* a) copy 64     “point‑major”  entries →   two temporary buffers  */
        for (int i = 0; i < TILE; ++i) {
            /* **MUST be a mutable object** because GetByte64() is non‑const */
            Point p = secp->GTable[base + i];

            for (int limb = 0; limb < 4; ++limb) {
                uint64_t wX = 0, wY = 0;
                for (int b = 0; b < 8; ++b) {          // BE → LE
                    wX = (wX << 8) | p.x.GetByte64(limb*8 + b);
                    wY = (wY << 8) | p.y.GetByte64(limb*8 + b);
                }
                bufX[i][3 - limb] = wX;                // little‑endian limb order
                bufY[i][3 - limb] = wY;
            }
        }

        /* b) write    “limb‑major”   ordering into the final table         */
        uint8_t* dstX = gX + static_cast<size_t>(base) * SIZE_GTABLE_POINT;
        uint8_t* dstY = gY + static_cast<size_t>(base) * SIZE_GTABLE_POINT;

        for (int limb = 0; limb < 4; ++limb) {
            for (int i = 0; i < TILE; ++i) {
                std::memcpy(dstX, &bufX[i][limb], 8);  dstX += 8;
            }
        }
        for (int limb = 0; limb < 4; ++limb) {
            for (int i = 0; i < TILE; ++i) {
                std::memcpy(dstY, &bufY[i][limb], 8);  dstY += 8;
            }
        }
    }
    delete secp;
}

static constexpr uint32_t GTABLE_MAGIC = 0x4C4D4254;
static void loadGTableFast(uint8_t* gX, uint8_t* gY, size_t bytes)
{
    std::ifstream fin("gtable_le.bin", std::ios::binary);
    uint32_t hdr = 0;
    if (fin && fin.read(reinterpret_cast<char*>(&hdr), 4) && hdr == GTABLE_MAGIC)
    {
        fin.read(reinterpret_cast<char*>(gX), bytes);
        fin.read(reinterpret_cast<char*>(gY), bytes);
        return;                             // ← new‑layout cache hit
    }
    // Build once, then save
    printf("Building G‑table (first run)…\n");
    loadGTable(gX, gY);                     // your existing builder

    std::ofstream fout("gtable_le.bin", std::ios::binary | std::ios::trunc);
    fout.write(reinterpret_cast<const char*>(&GTABLE_MAGIC), 4);
    fout.write(reinterpret_cast<const char*>(gX), bytes);
    fout.write(reinterpret_cast<const char*>(gY), bytes);
    printf("G‑table cached to gtable_le.bin (%.1f MiB)\n",
           bytes / (1024.0*1024.0));
}


static void buildAndUploadTargetTable(const char* filename,
                                      uint64_t** d_targetX,
                                      int*      numTargets);

//───────────────────────────────────────────────────────────────────────────
// Device helper (from GPU/GPUSecp.cu).  Expects privKey16[NUM_GTABLE_CHUNK],
// each in little‐endian 16‐bit windows; G‐table is also little‐endian.
// Computes (k·G) into qx[], qy[].
//───────────────────────────────────────────────────────────────────────────
// extern "C" __device__
// void _PointMultiSecp256k1(uint64_t* qx,
//                           uint64_t* qy,
//                           uint16_t* privKey,
//                           uint8_t* gTableX,
//                           uint8_t* gTableY);

//───────────────────────────────────────────────────────────────────────────
// Kernel A: Throughput mode—compute k = start+tid for tid< N_total,
// decompose k into 4 nonzero 16‐bit windows, call _PointMultiSecp256k1.
// Pure compute (no writes except guard).
//───────────────────────────────────────────────────────────────────────────
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void CudaSeqComputeOnly(uint64_t start,
                                   uint64_t N_total,
                                   const uint8_t* __restrict__ gX,
                                   const uint8_t* __restrict__ gY)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)threadIdx.x;
    if (tid >= N_total) return;

    uint64_t k = start + tid;

    uint16_t privKey16[NUM_GTABLE_CHUNK] = {0};
    #pragma unroll
    for (int c = 0; c < NUM_GTABLE_CHUNK_ACTIVE; ++c) {
        int shift = 16 * c;
        if (shift < 64) {
            privKey16[c] = static_cast<uint16_t>((k >> shift) & 0xFFFFULL);
        } else {
            privKey16[c] = 0;
        }
    }

    uint64_t qx[4], qy[4];
    _PointMultiSecp256k1(qx, qy, privKey16,
                         (uint8_t*)gX, (uint8_t*)gY);
}

//───────────────────────────────────────────────────────────────────────────
// Kernel B: Print mode—each thread i < M loads k = ks[i],
// decomposes into 4 windows, calls _PointMultiSecp256k1, writes 32 bytes
// of little‐endian X‐coordinate into pubXOut[i*32..i*32+31].
//───────────────────────────────────────────────────────────────────────────
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void CudaComputeSpecificKeys(const uint64_t* __restrict__ ks,
                                        int M,
                                        const uint8_t* __restrict__ gX,
                                        const uint8_t* __restrict__ gY,
                                        uint8_t* __restrict__ pubXOut)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    uint64_t k = ks[tid];

    
    uint16_t privKey16[NUM_GTABLE_CHUNK] = {0};

    #pragma unroll
    for (int c = 0; c < NUM_GTABLE_CHUNK_ACTIVE; ++c) {
        int shift = 16 * c;
        if (shift < 64) {
            privKey16[c] = static_cast<uint16_t>((k >> shift) & 0xFFFFULL);
        } else {
            privKey16[c] = 0;
        }
    }

    uint64_t qx[4], qy[4];
    _PointMultiSecp256k1(qx, qy, privKey16,
                         (uint8_t*)gX, (uint8_t*)gY);

    const uint8_t* qxBytes = reinterpret_cast<const uint8_t*>(qx);
    #pragma unroll
    for (int b = 0; b < 32; ++b) {
        pubXOut[tid * 32 + b] = qxBytes[b];
    }
}

//─────────────────────────────────────────────────────────────────────────────
// 256‐bit compare helper: compares two big‐endian 256‐bit values stored as
// four uint64_t words each. Returns -1 if A < B, 0 if A == B, +1 if A > B.
// ──────────────────────────────────────────────────────────────────────────────
static __device__ inline int cmp256_be(const uint64_t A[4], const uint64_t B[4]) {
    if (A[0] < B[0]) return -1;
    if (A[0] > B[0]) return +1;
    if (A[1] < B[1]) return -1;
    if (A[1] > B[1]) return +1;
    if (A[2] < B[2]) return -1;
    if (A[2] > B[2]) return +1;
    if (A[3] < B[3]) return -1;
    if (A[3] > B[3]) return +1;
    return 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Device‐side binary search over a sorted array of big‐endian 256‐bit keys.
//   - arr: pointer to an array of shape [numTargets][4], each row is a BE key.
//   - numTargets: how many entries are in arr.
//   - key: the 256‐bit value we’re searching for (also BE).
//   - outIndex: if found, set to the matching index; otherwise left unchanged.
// Returns true if found, false otherwise.
// ──────────────────────────────────────────────────────────────────────────────
static __device__ bool binarySearch256_be(
    const uint64_t (*arr)[4],
    int numTargets,
    const uint64_t key[4],
    int &outIndex
) {
    int lo = 0, hi = numTargets - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c = cmp256_be(arr[mid], key);
        if (c == 0) {
            outIndex = mid;
            return true;
        }
        else if (c < 0) {
            lo = mid + 1;
        }
        else {
            hi = mid - 1;
        }
    }
    return false;
}

//─────────────────────────────────────────────────────────────────────────────
// Kernel C: Throughput+Check mode—compute k = start+tid, decompose, multiply,
// then convert qx to big‐endian and binary‐search against d_targetX[]. If hit,
// record (k, index) via atomicAdd.
//───────────────────────────────────────────────────────────────────────────
__global__ void CudaSeqComputeWithCheck(
    uint64_t start,
    uint64_t N_total,
    const uint8_t* __restrict__ gX,
    const uint8_t* __restrict__ gY,

    // Sorted array of [numTargets][4] uint64_t, each row is a BE‐encoded 256‐bit X‐coordinate
    const uint64_t (*d_targetX)[4],
    int         numTargets,

    // Buffers for recording matches (must be pre‐allocated on the device)
    uint64_t*   d_hitKList,   // length ≥ maxHits
    int*        d_hitIdxList, // length ≥ maxHits
    int*        d_hitCount    // single‐int counter (initialized to 0)
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)threadIdx.x;
    if (tid >= N_total) return;

    // 1) Compute the private key (k)
    uint64_t k = start + tid;

    // 3) Call the point‐multiply helper (_PointMultiSecp256k1).
    //    This fills qx_le[] and qy_le[] with the result in “little‐endian per‐word” form.
    uint64_t qx_le[4], qy_le[4];

    #if 1   // fast 4‑window path for k < 2^64
    _PointMulti4w_inl(qx_le, qy_le, k, gX, gY);
    #else   // fallback (rarely used in your scan)
    /* 2) Decompose k into 16‑bit windows — only needed here */
    uint16_t privKey16[NUM_GTABLE_CHUNK] = {0};
    #pragma unroll
    for (int c = 0; c < NUM_GTABLE_CHUNK_ACTIVE; ++c) {
        int shift = 16 * c;
        privKey16[c] = (shift < 64)
                     ? static_cast<uint16_t>((k >> shift) & 0xFFFFULL)
                     : 0;
    }

    _PointMultiSecp256k1(qx_le, qy_le, privKey16,
                        (uint8_t*)gX, (uint8_t*)gY);
    #endif

    // 4) Convert each 64‐bit limb from LE → BE, so we can do a lexicographic compare.
    uint64_t qx_be[4];
    #pragma unroll
    for (int w = 0; w < 4; w++) {
        uint64_t x = qx_le[w];
        x = ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
        x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
        x = ((x & 0x00FF00FF00FF00FFULL) <<  8) | ((x & 0xFF00FF00FF00FF00ULL) >>  8);
        qx_be[w] = x;
    }

    // 5) Binary‐search the 256‐bit qx_be against d_targetX[0..numTargets-1]
    int foundIdx = -1;
    bool isHit = binarySearch256_be(d_targetX, numTargets, qx_be, foundIdx);

    if (isHit) {
        // 6) If it matches, use atomicAdd to get a unique slot and store (k, foundIdx)
        int slot = atomicAdd(d_hitCount, 1);
        d_hitKList[slot]   = k;
        d_hitIdxList[slot] = foundIdx;
    }
    // Done. Non‐hit threads perform no further global writes.
}

//───────────────────────────────────────────────────────────────────────────
// Helper: check CUDA errors and exit on failure
//───────────────────────────────────────────────────────────────────────────
static void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: %s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main(int argc, char** argv)
{
    // ─────────────────────────────────────────────────────────────────
    // 0) Revised command‐line parsing:
    //    - Remove --continue entirely (we always continue by default).
    //    - Add a new --reset flag to clear any existing checkpoint.
    // ─────────────────────────────────────────────────────────────────
    bool printMode   = false;
    bool resetFlag   = false;    // <<< NEW >>>
    const std::string checkpointFile = "checkpoint.txt";

    // Collect arguments except "--reset"
    std::vector<char*> args;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--reset") == 0) {
            resetFlag = true;       // <<< NEW >>>
        }
        else {
            args.push_back(argv[i]);
        }
    }

    // Check if we're in print mode:
    std::vector<uint64_t> keyList;
    if (args.size() >= 1 && std::strcmp(args[0], "print") == 0) {
        printMode = true;
        if (args.size() < 2) {
            fprintf(stderr, "Usage: %s print k1 [k2 k3 ...]\n", argv[0]);
            return 1;
        }
        for (size_t i = 1; i < args.size(); i++) {
            long long tmp = atoll(args[i]);
            if (tmp <= 0) {
                fprintf(stderr, "ERROR: invalid key '%s'\n", args[i]);
                return 1;
            }
            keyList.push_back(static_cast<uint64_t>(tmp));
        }
    }

    // For throughput mode (i.e. not printMode), determine N_totalArg from args (or default to 1,000,000)
    uint64_t N_totalArg = 1000000ULL;
    if (!printMode) {
        if (args.size() >= 1) {
            long long tmp = atoll(args[0]);
            if (tmp > 0) {
                N_totalArg = static_cast<uint64_t>(tmp);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 1) Always try to read “checkpoint.txt” (unless --reset was passed)
    // ─────────────────────────────────────────────────────────────────
    uint64_t prevDone = 0;
    if (!printMode && !resetFlag) {
        std::ifstream chkIn(checkpointFile);
        if (chkIn.is_open()) {
            chkIn >> prevDone;
            chkIn.close();
        }
    }

    // If --reset was given, or file didn’t exist, prevDone stays 0,
    // but if resetFlag==true, we also delete any old file so we start clean.
    if (resetFlag) {
        std::remove(checkpointFile.c_str());
        prevDone = 0;
    }

    // ─────────────────────────────────────────────────────────────────
    // 2) Compute new “endAt” based on the old checkpoint + user’s argument
    //    (always “continue” semantics).  
    //    If N_totalArg ≤ prevDone, we mean “prevDone + N_totalArg.”
    //    Otherwise we mean “absolute up to N_totalArg.”
    // ─────────────────────────────────────────────────────────────────
    uint64_t endAt = N_totalArg;
    if (!printMode && N_totalArg <= prevDone) {
        endAt = prevDone + N_totalArg;
    }

    // Now our scan range is [prevDone+1 .. endAt], so:
    uint64_t startOffset       = prevDone;
    uint64_t N_total_effective = (endAt > prevDone) 
                                ? (endAt - prevDone) 
                                : 0ULL;

    if (!printMode) {
        printf("Resuming from key %llu (previous run processed %llu). Will scan up to %llu.\n",
               (unsigned long long)(startOffset + 1),
               (unsigned long long)startOffset,
               (unsigned long long)endAt);
    }

// ─────────────────────────────────────────────────────────────────
// 3) PRINT mode: compute (k·G).X for the user‑supplied key list
// ─────────────────────────────────────────────────────────────────
if (printMode) {
    int M = static_cast<int>(keyList.size());
    printf("PRINT mode: computing %d specified key(s)\n", M);

    //----------------------------------------------------------------
    // 3a) Build & upload the (shrunk) G‑table exactly once
    //----------------------------------------------------------------
    size_t tblBytes = static_cast<size_t>(COUNT_GTABLE_POINTS) *
                      SIZE_GTABLE_POINT;           // 32 MiB after step 1.3
    uint8_t *gXCPU = (uint8_t*)std::malloc(tblBytes);
    uint8_t *gYCPU = (uint8_t*)std::malloc(tblBytes);
    loadGTableFast(gXCPU, gYCPU, tblBytes);

    uint8_t *gXGPU = nullptr, *gYGPU = nullptr;
    cudaCheck(cudaMalloc(&gXGPU, tblBytes), "cudaMalloc gXGPU");
    cudaCheck(cudaMalloc(&gYGPU, tblBytes), "cudaMalloc gYGPU");
    cudaCheck(cudaMemcpy(gXGPU, gXCPU, tblBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy gXGPU");
    cudaCheck(cudaMemcpy(gYGPU, gYCPU, tblBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy gYGPU");
    std::free(gXCPU);
    std::free(gYCPU);

    //----------------------------------------------------------------
    // 3b) Copy the key list to the GPU
    //----------------------------------------------------------------
    uint64_t *d_keys     = nullptr;
    uint8_t  *d_pubX     = nullptr;     // M × 32 bytes, little‑endian words
    cudaCheck(cudaMalloc(&d_keys,  sizeof(uint64_t) * M), "cudaMalloc d_keys");
    cudaCheck(cudaMalloc(&d_pubX,  32 * M),               "cudaMalloc d_pubX");
    cudaCheck(cudaMemcpy(d_keys, keyList.data(),
                         sizeof(uint64_t) * M,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy d_keys");

    //----------------------------------------------------------------
    // 3c) Launch one block per 256 threads (plenty for small M)
    //----------------------------------------------------------------
    int threadsPerBlock = 256;
    int blocks = (M + threadsPerBlock - 1) / threadsPerBlock;
    CudaComputeSpecificKeys<<<blocks, threadsPerBlock>>>(
        d_keys, M, gXGPU, gYGPU, d_pubX);
    cudaCheck(cudaGetLastError(),      "kernel launch (print mode)");
    cudaCheck(cudaDeviceSynchronize(), "kernel sync (print mode)");

    //----------------------------------------------------------------
    // 3d) Copy results back
    //----------------------------------------------------------------
    std::vector<uint8_t> h_pubX(32 * M);
    cudaCheck(cudaMemcpy(h_pubX.data(), d_pubX, 32 * M,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy h_pubX");

    //----------------------------------------------------------------
    // 3e) Format & print
    //----------------------------------------------------------------
    for (int i = 0; i < M; ++i) {
        const uint8_t* be = h_pubX.data() + (i * 32);   // already big‑endian
        mpz_t xInt;  mpz_init(xInt);
        mpz_import(xInt,          /* rop */
                32,            /* 32 bytes */
                1,             /* most‑significant word first  */
                1,             /* word size = 1 byte          */
                0, 0,          /* big‑endian, no nails        */
                be);           /* source buffer               */
        char* decStr = mpz_get_str(nullptr, 10, xInt);

        printf("PRIV %llu   PUB_X %s\n",
            (unsigned long long)keyList[i], decStr);

        free(decStr);
        mpz_clear(xInt);
    }

    //----------------------------------------------------------------
    // 3f) Cleanup and exit
    //----------------------------------------------------------------
    cudaFree(d_keys);
    cudaFree(d_pubX);
    cudaFree(gXGPU);
    cudaFree(gYGPU);
    return 0;
}

    // ─────────────────────────────────────────────────────────────────
    // 4) Now we’re in “throughput+check” mode, scanning [startOffset+1..endAt]
    // ─────────────────────────────────────────────────────────────────
    if (N_total_effective == 0) {
        printf("No new keys to process (endAt == prevDone). Exiting.\n");
        return 0;
    }

    printf("Throughput+Check mode: will scan %llu keys (resuming)\n",
           (unsigned long long)endAt);

    // 4a) Build & upload G‐table as before
    size_t tblBytes = static_cast<size_t>(COUNT_GTABLE_POINTS) * SIZE_GTABLE_POINT;
    uint8_t* gXCPU = (uint8_t*)std::malloc(tblBytes);
    uint8_t* gYCPU = (uint8_t*)std::malloc(tblBytes);
    if (!gXCPU || !gYCPU) {
        fprintf(stderr, "ERROR: malloc G‐table CPU buffers failed\n");
        return 1;
    }
    loadGTableFast(gXCPU, gYCPU, tblBytes);

    uint8_t *gXGPU = nullptr, *gYGPU = nullptr;
    cudaCheck(cudaMalloc(&gXGPU, tblBytes), "cudaMalloc gXGPU");
    cudaCheck(cudaMalloc(&gYGPU, tblBytes), "cudaMalloc gYGPU");
    cudaCheck(cudaMemcpy(gXGPU, gXCPU, tblBytes, cudaMemcpyHostToDevice), "cudaMemcpy gXGPU");
    cudaCheck(cudaMemcpy(gYGPU, gYCPU, tblBytes, cudaMemcpyHostToDevice), "cudaMemcpy gYGPU");
    std::free(gXCPU);
    std::free(gYCPU);

    // 4b) Build & upload target table, allocate hit‐lists, etc.
    uint64_t (*d_targetX)[4] = nullptr;
    int       numTargets    = 0;
    buildAndUploadTargetTable("target.csv", (uint64_t**)&d_targetX, &numTargets);

    int maxHits = numTargets + 16;
    uint64_t* d_hitKList   = nullptr;
    int*      d_hitIdxList = nullptr;
    int*      d_hitCount   = nullptr;
    cudaCheck(cudaMalloc(&d_hitKList,   sizeof(uint64_t) * maxHits),   "cudaMalloc d_hitKList");
    cudaCheck(cudaMalloc(&d_hitIdxList, sizeof(int)      * maxHits),   "cudaMalloc d_hitIdxList");
    cudaCheck(cudaMalloc(&d_hitCount,   sizeof(int)),                 "cudaMalloc d_hitCount");
    cudaCheck(cudaMemset(d_hitCount, 0, sizeof(int)),                  "cudaMemset d_hitCount");

    // 4c) Determine how many blocks/launches are needed to cover [startOffset+1..endAt]
    const int threadsPerBlock = THREADS_PER_BLOCK;
    uint64_t fullBlocks  = (N_total_effective + threadsPerBlock - 1ULL) / threadsPerBlock;
    const uint64_t maxBlocks = 0x7FFFFFFFULL;
    uint64_t numLaunches = (fullBlocks + maxBlocks - 1ULL) / maxBlocks;

    // 4d) Launch each chunk, check for early hits, do periodic checkpoints
    auto t_runStart       = std::chrono::high_resolution_clock::now();
    auto t_lastCheckpoint = t_runStart;
    const double checkpointIntervalSec = 300.0;     // every 300 seconds

    uint64_t keysLaunched   = 0;
    uint64_t blocksLaunched = 0;
    bool earlyExit = false;

    for (uint64_t launchIdx = 0; launchIdx < numLaunches; ++launchIdx) {
        uint64_t blocksLeft = fullBlocks - blocksLaunched;
        uint64_t thisBlocks = (blocksLeft < maxBlocks ? blocksLeft : maxBlocks);

        uint64_t keysThisLaunch = thisBlocks * (uint64_t)threadsPerBlock;
        if (keysLaunched + keysThisLaunch > N_total_effective) {
            keysThisLaunch = N_total_effective - keysLaunched;
            thisBlocks    = (keysThisLaunch + threadsPerBlock - 1ULL) / threadsPerBlock;
        }

        dim3 gridDim(static_cast<uint32_t>(thisBlocks));
        dim3 blockDim(threadsPerBlock);

        uint64_t startForKernel = (uint64_t)1 + startOffset + keysLaunched;
        CudaSeqComputeWithCheck<<<gridDim, blockDim>>>(
            /*start=*/       startForKernel,
            /*N_total=*/     keysThisLaunch,
            /*gX=*/          gXGPU,
            /*gY=*/          gYGPU,
            /*d_targetX=*/   d_targetX,
            /*numTargets=*/  numTargets,
            /*d_hitKList=*/  d_hitKList,
            /*d_hitIdxList=*/d_hitIdxList,
            /*d_hitCount=*/  d_hitCount
        );
        cudaCheck(cudaGetLastError(), "Kernel launch (throughput+check)");
        cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize (throughput+check)");

        keysLaunched   += keysThisLaunch;
        blocksLaunched += thisBlocks;

        // ────────────────────────────────────────────────────────────────────────────
        // Check for any hits immediately (early‐exit):
        int h_hitCount = 0;
        cudaMemcpy(&h_hitCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_hitCount > 0) {
            printf("== EARLY MATCH FOUND: %d hit(s) ==\n", h_hitCount);
            std::vector<uint64_t> hK(h_hitCount);
            std::vector<int>     hIdx(h_hitCount);
            cudaMemcpy(hK.data(),   d_hitKList,   sizeof(uint64_t)*h_hitCount, cudaMemcpyDeviceToHost);
            cudaMemcpy(hIdx.data(), d_hitIdxList, sizeof(int)*h_hitCount,        cudaMemcpyDeviceToHost);

            // Print to screen
            for (int i = 0; i < h_hitCount; i++) {
                printf("  k = %llu  matched targetIndex = %d\n",
                       (unsigned long long)hK[i],
                       hIdx[i]);
            }
            // Dump to file
            std::ofstream fout("match_found.txt", std::ios::trunc);
            if (fout.is_open()) {
                for (int i = 0; i < h_hitCount; i++) {
                    fout << "k = " << hK[i]
                         << "  matched targetIndex = " << hIdx[i]
                         << "\n";
                }
                fout.close();
                printf("Match details written to match_found.txt\n");
            } else {
                fprintf(stderr, "WARNING: could not open match_found.txt for writing\n");
            }

            earlyExit = true;
        }
        if (earlyExit) break;

        // ────────────────────────────────────────────────────────────────────────────
        // Periodic checkpoint (every ~5 minutes):
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsedSinceLast = std::chrono::duration<double>(t_now - t_lastCheckpoint).count();
        if (elapsedSinceLast >= checkpointIntervalSec) {
            uint64_t totalProcessed = startOffset + keysLaunched;
            double elapsedSinceRunStart = std::chrono::duration<double>(t_now - t_runStart).count();
            double runThroughput = (double)keysLaunched / elapsedSinceRunStart; // keys/sec
            double runThroughputM = runThroughput / 1e6;

            printf(
                "Checkpoint: processed %llu/%llu keys. Throughput: %.2f M keys/sec\n",
                (unsigned long long)totalProcessed,
                (unsigned long long)endAt,
                runThroughputM
            );

            // Overwrite checkpoint file with “totalProcessed”
            std::ofstream chkOut(checkpointFile, std::ios::trunc);
            if (chkOut.is_open()) {
                chkOut << totalProcessed;
                chkOut.close();
            } else {
                fprintf(stderr, "WARNING: could not write checkpoint to %s\n", checkpointFile.c_str());
            }
            t_lastCheckpoint = t_now;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 5) Final summary (unless we already early‐exited)
    // ─────────────────────────────────────────────────────────────────
    if (!earlyExit) {
        auto t_runEnd      = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_runEnd - t_runStart).count();
        printf("Elapsed time:  %.3f ms\n", elapsed_ms);
        double keys_per_sec  = (double)N_total_effective / (elapsed_ms / 1000.0);
        double mkeys_per_sec = keys_per_sec / 1e6;
        printf("Throughput:    %.0f keys/sec  (%.2f M keys/sec)\n",
               keys_per_sec, mkeys_per_sec);

        int h_hitCount_final = 0;
        cudaMemcpy(&h_hitCount_final, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_hitCount_final > 0) {
            printf("== FOUND %d MATCHES ==\n", h_hitCount_final);
            std::vector<uint64_t> hKf(h_hitCount_final);
            std::vector<int>     hIdxf(h_hitCount_final);
            cudaMemcpy(hKf.data(),   d_hitKList,   sizeof(uint64_t)*h_hitCount_final, cudaMemcpyDeviceToHost);
            cudaMemcpy(hIdxf.data(), d_hitIdxList, sizeof(int)*h_hitCount_final,        cudaMemcpyDeviceToHost);
            for (int i = 0; i < h_hitCount_final; i++) {
                printf("  k = %llu  matched targetIndex = %d\n",
                       (unsigned long long)hKf[i],
                       hIdxf[i]);
            }
        } else {
            printf("No matches found.\n");
        }
    } else {
        printf("Terminated early due to a match. Stopping throughput scan.\n");
    }

    // ─────────────────────────────────────────────────────────────────
    // 6) Final checkpoint: write “endAt” so future runs resume past this
    // ─────────────────────────────────────────────────────────────────
    {
        std::ofstream chkOut(checkpointFile, std::ios::trunc);
        if (chkOut.is_open()) {
            chkOut << endAt;
            chkOut.close();
        } else {
            fprintf(stderr, "WARNING: could not write final checkpoint to %s\n", checkpointFile.c_str());
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 7) Cleanup
    // ─────────────────────────────────────────────────────────────────
    cudaFree(d_targetX);
    cudaFree(d_hitKList);
    cudaFree(d_hitIdxList);
    cudaFree(d_hitCount);

    cudaFree(gXGPU);
    cudaFree(gYGPU);
    return 0;
}


//───────────────────────────────────────────────────────────────────────────
// buildAndUploadTargetTable(...) definition (exactly as before):
//───────────────────────────────────────────────────────────────────────────
static void buildAndUploadTargetTable(const char* filename,
                                      uint64_t** d_targetX,
                                      int*      numTargets)
{
    // 1) Open the CSV (one column, no header, unknown N)
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        fprintf(stderr, "ERROR: could not open %s\n", filename);
        std::exit(1);
    }

    std::vector<std::string> lines;
    lines.reserve(6000);
    std::string line;
    while (std::getline(ifs, line)) {
        // Trim leading/trailing whitespace:
        size_t a = line.find_first_not_of(" \t\r\n");
        size_t b = line.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) {
            // empty or all‐whitespace line → skip
            continue;
        }
        std::string decStr = line.substr(a, b - a + 1);
        lines.push_back(decStr);
    }
    ifs.close();

    int N = static_cast<int>(lines.size());
    if (N <= 0) {
        fprintf(stderr, "ERROR: %s contained no valid lines\n", filename);
        std::exit(1);
    }
    // We won’t enforce a strict upper bound here; if you expect ~5000, this will handle any N ≥ 1.

    // 2) Parse each decimal string into a 32-byte big-endian buffer via GMP
    std::vector< std::array<uint8_t,32> > beBytes(N);
    for (int i = 0; i < N; i++) {
        mpz_t tmp;
        mpz_init(tmp);
        if (mpz_set_str(tmp, lines[i].c_str(), 10) != 0) {
            fprintf(stderr, "ERROR: invalid decimal in %s: \"%s\"\n", filename, lines[i].c_str());
            mpz_clear(tmp);
            std::exit(1);
        }
        // Zero out all 32 bytes first:
        for (int b = 0; b < 32; b++) {
            beBytes[i][b] = 0;
        }
        // Export into big-endian bytes; mpz_export writes as many bytes as needed into the rightmost portion
        size_t countp = 0;
        mpz_export(beBytes[i].data(),
                   &countp,
                   1,    // most-significant word first
                   1,    // each word = 1 byte
                   0, 0, // big-endian, no nails
                   tmp);
        mpz_clear(tmp);
    }

    // 3) Pack each 32-byte BE buffer → 4 × uint64_t (big-endian order)
    std::vector< std::array<uint64_t,4> > hostTable(N);
    for (int i = 0; i < N; i++) {
        for (int w = 0; w < 4; w++) {
            uint64_t accum = 0ULL;
            for (int b = 0; b < 8; b++) {
                accum <<= 8;
                accum |= static_cast<uint64_t>(beBytes[i][w*8 + b]);
            }
            hostTable[i][w] = accum;
        }
    }

    // 4) Sort lexicographically by (word0, word1, word2, word3) ascending
    std::sort(hostTable.begin(), hostTable.end(),
        [](auto &A, auto &B) {
            if (A[0] < B[0]) return true;
            if (A[0] > B[0]) return false;
            if (A[1] < B[1]) return true;
            if (A[1] > B[1]) return false;
            if (A[2] < B[2]) return true;
            if (A[2] > B[2]) return false;
            return (A[3] < B[3]);
        }
    );

    // 5) Flatten the sorted array into a contiguous uint64_t[N*4] buffer
    size_t bytes = sizeof(uint64_t) * 4 * static_cast<size_t>(N);
    uint64_t* h_flat = static_cast<uint64_t*>(std::malloc(bytes));
    if (!h_flat) {
        fprintf(stderr, "ERROR: malloc hostTable[%zu] returned NULL\n", bytes);
        std::exit(1);
    }
    for (int i = 0; i < N; i++) {
        h_flat[ (size_t)i*4 + 0 ] = hostTable[i][0];
        h_flat[ (size_t)i*4 + 1 ] = hostTable[i][1];
        h_flat[ (size_t)i*4 + 2 ] = hostTable[i][2];
        h_flat[ (size_t)i*4 + 3 ] = hostTable[i][3];
    }

    // 6) cudaMalloc & cudaMemcpy into device memory
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d_targetX), bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_targetX failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
    err = cudaMemcpy(*d_targetX, h_flat, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy to d_targetX failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
    std::free(h_flat);

    // 7) Return the number of targets
    *numTargets = N;
    printf("Host: uploaded %d target X‐coordinates (sorted) to device.\n", N);
}
