//GPUSecp.h

#ifndef GPUSECP
#define GPUSECP

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define NAME_HASH_FOLDER "TestHash"
#define NAME_SEED_FOLDER "TestBook"
#define NAME_HASH_BUFFER "merged-sorted-unique-8-byte-hashes"
#define NAME_INPUT_PRIME NAME_SEED_FOLDER "/list_prime"
#define NAME_INPUT_AFFIX NAME_SEED_FOLDER "/list_affix"
#define NAME_FILE_OUTPUT "TEST_OUTPUT"

//CUDA-specific parameters that determine occupancy and thread-count
#define BLOCKS_PER_GRID 30
#define THREADS_PER_BLOCK 256

//Maximum length of each Prime word (+1 because first byte contains word length) 
#define MAX_LEN_WORD_PRIME 20

//Maximum length of each Affix word (+1 because first byte contains word length) 
#define MAX_LEN_WORD_AFFIX 4

//Determines if book Affix words will be added as prefix or as suffix to Prime words.
#define AFFIX_IS_SUFFIX true

//This is how many hashes are in NAME_HASH_FOLDER, Defined as constant to save one register in device kernel
#define COUNT_INPUT_HASH 204

//This is how many prime words are in NAME_INPUT_PRIME file, Defined as constant to save one register in device kernel
#define COUNT_INPUT_PRIME 100

//Combo symbol count - how many unique symbols exist in the COMBO_SYMBOLS array
#define COUNT_COMBO_SYMBOLS 100

//Combo multiplication / buffer size - how many times symbols will be multiplied with each-other (maximum supported size is 8)
#define SIZE_COMBO_MULTI 4

//CPU stack size in bytes that will be allocated to this program - needs to fit GTable / InputBooks / InputHashes 
#define SIZE_CPU_STACK 1024 * 1024 * 1024

//GPU stack size in bytes that will be allocated to each thread - has complex functionality - please read cuda docs about this
#define SIZE_CUDA_STACK 32768

//---------------------------------------------------------------------------------------------------------------------------
// Don't edit configuration below this line
//---------------------------------------------------------------------------------------------------------------------------

#define SIZE_LONG 8            // Each Long is 8 bytes
#define SIZE_HASH160 20        // Each Hash160 is 20 bytes
#define SIZE_PRIV_KEY 32 	   // Length of the private key that is generated from input seed (in bytes)
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (BLOCKS_PER_GRID * THREADS_PER_BLOCK)
// -------------------------------------------------------------------
//  NEW: how many 16‑bit windows the sequential scanner actually uses
// -------------------------------------------------------------------
#define NUM_GTABLE_CHUNK_ACTIVE 4                    // ≤ NUM_GTABLE_CHUNK
#define COUNT_GTABLE_POINTS_ACTIVE  (NUM_GTABLE_CHUNK_ACTIVE * NUM_GTABLE_VALUE)


// GPU/GPUSecp.h
// …
extern __device__ __constant__ int      CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK];
extern __device__ __constant__ int      MULTI_EIGHT[65];
extern __device__ __constant__ uint8_t  COMBO_SYMBOLS[COUNT_COMBO_SYMBOLS];
// Forward declaration so every translation unit sees the same signature
extern "C" __device__
void _PointMultiSecp256k1(uint64_t*       qx,
                          uint64_t*       qy,
                          const uint16_t* privKey,
                          const uint8_t* __restrict__ gTableX,
                          const uint8_t* __restrict__ gTableY);


#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

class GPUSecp
{

public:
	GPUSecp(
		int primeCount, 
		int affixCount,
		const uint8_t * gTableXCPU,
		const uint8_t * gTableYCPU,
		const uint8_t * inputBookPrimeCPU, 
		const uint8_t * inputBookAffixCPU, 
		const uint64_t * inputHashBufferCPU
		);

	void doIterationSecp256k1Books(int iteration);
	void doIterationSecp256k1Combo(int8_t * inputComboCPU);
	void doPrintOutput();
	void doFreeMemory();

private:
	//Input combo buffer, used only in Combo Mode, defines the starting position for each thread
	int8_t * inputComboGPU;

	//GTable buffer containing ~1 million pre-computed points for Secp256k1 point multiplication
	uint8_t * gTableXGPU;
	uint8_t * gTableYGPU;

	//Input buffer that holds Prime wordlist in global memory of the GPU device
	uint8_t * inputBookPrimeGPU;

	//Input buffer that holds Affix wordlist in global memory of the GPU device
	uint8_t * inputBookAffixGPU;

	//Input buffer that holds merged-sorted-unique-8-byte-hashes in global memory of the GPU device
	uint64_t * inputHashBufferGPU;

	//Output buffer containing result of single iteration
	//If seed created a known Hash160 then outputBufferGPU for that affix will be 1
	uint8_t * outputBufferGPU;
	uint8_t * outputBufferCPU;

	//Output buffer containing result of succesful hash160
	//Each hash160 is 20 bytes long, total size is N * 20 bytes
	uint8_t * outputHashesGPU;
	uint8_t * outputHashesCPU;

	//Output buffer containing private keys that were used in succesful hash160
	//Each private key is 32-byte number that was the output of SHA256
	uint8_t * outputPrivKeysGPU;
	uint8_t * outputPrivKeysCPU;
};



#endif // GPUSecpH
