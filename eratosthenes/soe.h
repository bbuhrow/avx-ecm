/*----------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Ben Buhrow. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

Some parts of the code (and also this header), included in this 
distribution have been reused from other sources. In particular I 
have benefitted greatly from the work of Jason Papadopoulos's msieve @ 
www.boo.net/~jasonp, Scott Contini's mpqs implementation, and Tom St. 
Denis Tom's Fast Math library.  Many thanks to their kind donation of 
code to the public domain.
       				   --bbuhrow@gmail.com 7/28/10
----------------------------------------------------------------------*/

#ifndef YAFU_SOE_H
#define YAFU_SOE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#ifndef NO_THREADS
#include <pthread.h>
#endif
#include "gmp.h"
#include "util.h"
#include <math.h>
#include <string.h>

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <intrin.h>	
//#include <malloc.h>
#include <windows.h>
#include <process.h>
#endif

#ifndef NO_THREADS
#define USE_SOE_THREADPOOL
#endif

#ifdef _MSC_VER
// optionally define this or not depending on whether your hardware supports it.
// if defined, compile the sse41 functions into the fat binary.  the global
// flag HAS_SSE41 is set at runtime on compatible hardware to enable the functions
// to be used.  For gcc and mingw64 builds, USE_SSE41 is enabled in the makefile.
//#define USE_SSE41 1
//#define TARGET_KNL 1
#endif

#define BLOCKSIZE 32768
#define FLAGSIZE 262144
#define FLAGSIZEm1 262143
#define FLAGBITS 18
#define BUCKETSTARTP 393216 
#define BUCKETSTARTI 33335
#define BITSINBYTE 8
#define MAXSIEVEPRIMECOUNT 100000000	//# primes less than ~2e9: limit of 2e9^2 = 4e18
//#define INPLACE_BUCKET 1
//#define DO_SPECIAL_COUNT


enum soe_command {
	SOE_COMMAND_INIT,
	SOE_COMMAND_WAIT,
	SOE_COMMAND_SIEVE_AND_COUNT,
	SOE_COMMAND_SIEVE_AND_COMPUTE,
	SOE_COMPUTE_ROOTS,
	SOE_COMPUTE_PRIMES,
	SOE_COMPUTE_PRPS,
	SOE_COMMAND_END
};

#if defined (TARGET_KNL) || defined(SKYLAKEX)
// for storage of presieving lists from prime index 24 to 40 (97 to 173 inclusive)
#ifdef __GNUC__
__attribute__((aligned(64))) uint64_t presieve_largemasks[16][173][4];
__attribute__((aligned(64))) uint32_t presieve_steps[32];
__attribute__((aligned(64))) uint32_t presieve_primes[32];
__attribute__((aligned(64))) uint32_t presieve_p1[32];
#else
__declspec(align(64)) uint64_t presieve_largemasks[16][173][4];
__declspec(align(64)) uint32_t presieve_steps[32];
__declspec(align(64)) uint32_t presieve_primes[32];
__declspec(align(64)) uint32_t presieve_p1[32];
#endif
#endif

typedef struct
{
	uint16_t loc;
	uint8_t mask;
	uint8_t bnum;
} soe_bucket_t_old;

//typedef struct
//{
//	uint32_t root;
//	uint32_t prime;
//} soe_bucket_t;

typedef struct
{
	//uint32_t prime;		// the prime, so that we don't have to also look in the
						// main prime array
	uint32_t bitloc;		// bit location of the current hit
	uint32_t next_pid;	// index of the next prime that hits in the current sieve
	uint32_t p_div;		// prime / prodN
	uint8_t p_mod;		// prime % prodN
	uint8_t eacc;			// accumulated error
} soe_inplace_p;

typedef struct
{
    int sync_count;
	uint32_t *sieve_p;
	int *root;
	uint32_t *lower_mod_prime;
	uint64_t blk_r;
	uint64_t blocks;
	uint64_t partial_block_b;
	uint64_t prodN;
	uint64_t startprime;
	uint64_t orig_hlimit;
	uint64_t orig_llimit;
	uint64_t pbound;
	uint64_t pboundi;

	uint32_t bucket_start_id;
	uint32_t large_bucket_start_prime;
	uint32_t num_bucket_primes;
	uint32_t inplace_start_id;
	uint32_t num_inplace_primes;

	uint64_t lowlimit;
	uint64_t highlimit;
	uint64_t numlinebytes;
	uint32_t numclasses;
	uint32_t *rclass;
	uint32_t *special_count;
	uint32_t num_special_bins;
	uint8_t **lines;
	uint32_t bucket_alloc;
	uint32_t large_bucket_alloc;
	uint64_t num_found;
#if defined(INPLACE_BUCKET)
	soe_inplace_p *inplace_data;
	int **inplace_ptrs;
#endif
	int only_count;
	mpz_t *offset;
	int sieve_range;
	uint64_t min_sieved_val;

    // presieving stuff
    int presieve_max_id;

} soe_staticdata_t;

typedef struct
{
	uint64_t *pbounds;
	uint32_t *offsets;
	uint64_t lblk_b;
	uint64_t ublk_b;
	uint64_t blk_b_sqrt;
	uint32_t bucket_depth;

	uint32_t bucket_alloc;
	uint32_t *bucket_hits;
    uint64_t **sieve_buckets;
	
	uint32_t *special_count;
	uint32_t num_special_bins;

	uint32_t **large_sieve_buckets;
	uint32_t *large_bucket_hits;
	uint32_t bucket_alloc_large;
	
	uint64_t *primes;
	uint32_t largep_offset;
	uint64_t min_sieved_val;

    // presieving stuff
    uint32_t *presieve_scratch;

} soe_dynamicdata_t;

typedef struct {
	soe_dynamicdata_t ddata;
	soe_staticdata_t sdata;
	uint64_t linecount;
	uint32_t current_line;

    int tindex;
    int tstartup;

	// start and stop for computing roots
	uint32_t startid, stopid;

	// stuff for computing PRPs
	mpz_t offset, lowlimit, highlimit, tmpz;

	/* fields for thread pool synchronization */
	volatile enum soe_command command;

#ifndef NO_THREADS
#ifdef USE_SOE_THREADPOOL
    /* fields for thread pool synchronization */
    volatile int *thread_queue, *threads_waiting;

#if defined(WIN32) || defined(_WIN64)
    HANDLE thread_id;
    HANDLE run_event;

    HANDLE finish_event;
    HANDLE *queue_event;
    HANDLE *queue_lock;

#else
    pthread_t thread_id;
    pthread_mutex_t run_lock;
    pthread_cond_t run_cond;

    pthread_mutex_t *queue_lock;
    pthread_cond_t *queue_cond;
#endif

#else

#if defined(WIN32) || defined(_WIN64)
    HANDLE thread_id;
    HANDLE run_event;
    HANDLE finish_event;
#else
    pthread_t thread_id;
    pthread_mutex_t run_lock;
    pthread_cond_t run_cond;
#endif
#endif

#endif

} thread_soedata_t;

// for use with threadpool
typedef struct
{
    soe_staticdata_t *sdata;
    thread_soedata_t *ddata;
} soe_userdata_t;

// top level sieving code
uint64_t spSOE(uint32_t *sieve_p, uint32_t num_sp,
	uint64_t lowlimit, uint64_t *highlimit, int count, uint64_t *primes);

// thread ready sieving functions
void sieve_line(thread_soedata_t *thread_data);
uint64_t count_line(soe_staticdata_t *sdata, uint32_t current_line);
void count_line_special(thread_soedata_t *thread_data);
uint32_t compute_32_bytes(soe_staticdata_t *sdata,
    uint32_t pcount, uint64_t *primes, uint64_t byte_offset);
uint64_t primes_from_lineflags(soe_staticdata_t *sdata, thread_soedata_t *thread_data,
	uint32_t start_count, uint64_t *primes);
void get_offsets(thread_soedata_t *thread_data);
void getRoots(soe_staticdata_t *sdata, thread_soedata_t *thread_data);
void stop_soe_worker_thread(thread_soedata_t *t);
void start_soe_worker_thread(thread_soedata_t *t);
#if defined(WIN32) || defined(_WIN64)
DWORD WINAPI soe_worker_thread_main(LPVOID thread_data);
#else
void *soe_worker_thread_main(void *thread_data);
#endif

// routines for finding small numbers of primes; seed primes for main SOE
uint32_t tiny_soe(uint32_t limit, uint32_t *primes);

void test_soe(int upper);

// interface functions
uint64_t *GetPRIMESRange(uint32_t *sieve_p, uint32_t num_sp, 
    uint64_t lowlimit, uint64_t highlimit, uint64_t *num_p);
uint64_t *soe_wrapper(uint32_t *sieve_p, uint32_t num_sp, 
	uint64_t lowlimit, uint64_t highlimit, int count, uint64_t *num_p);
uint64_t *sieve_to_depth(uint32_t *seed_p, uint32_t num_sp, 
    mpz_t lowlimit, mpz_t highlimit, int count, int num_witnesses, uint64_t *num_p);

// misc and helper functions
uint64_t estimate_primes_in_range(uint64_t lowlimit, uint64_t highlimit);
void get_numclasses(uint64_t highlimit, uint64_t lowlimit, soe_staticdata_t *sdata);
int check_input(uint64_t highlimit, uint64_t lowlimit, uint32_t num_sp, uint32_t *sieve_p,
    soe_staticdata_t *sdata);
uint64_t init_sieve(soe_staticdata_t *sdata);
void set_bucket_depth(soe_staticdata_t *sdata);
uint64_t alloc_threaddata(soe_staticdata_t *sdata, thread_soedata_t *thread_data);
void do_soe_sieving(soe_staticdata_t *sdata, thread_soedata_t *thread_data, int count);
void finalize_sieve(soe_staticdata_t *sdata,
	thread_soedata_t *thread_data, int count, uint64_t *primes);

void pre_sieve(soe_dynamicdata_t *ddata, soe_staticdata_t *sdata, uint8_t *flagblock);
void pre_sieve_avx2(soe_dynamicdata_t *ddata, soe_staticdata_t *sdata, uint8_t *flagblock);
void (*pre_sieve_ptr)(soe_dynamicdata_t *, soe_staticdata_t *, uint8_t *);

uint32_t compute_8_bytes(soe_staticdata_t *sdata,
    uint32_t pcount, uint64_t *primes, uint64_t byte_offset);
uint32_t compute_8_bytes_avx2(soe_staticdata_t *sdata,
    uint32_t pcount, uint64_t *primes, uint64_t byte_offset);
uint32_t (*compute_8_bytes_ptr)(soe_staticdata_t *, uint32_t, uint64_t *, uint64_t);

uint32_t modinv_1(uint32_t a, uint32_t p);
uint32_t modinv_1c(uint32_t a, uint32_t p);
uint64_t mpz_get_64(mpz_t src);

//masks for removing or reading single bits in a byte.  nmasks are simply
//the negation of these masks, and are filled in within the spSOE function.
static const uint8_t masks[8] = {0xfe, 0xfd, 0xfb, 0xf7, 0xef, 0xdf, 0xbf, 0x7f};
uint8_t nmasks[8];
uint32_t max_bucket_usage;
uint64_t GLOBAL_OFFSET;
int NO_STORE;
int PRIMES_TO_FILE;
int PRIMES_TO_SCREEN;
int SOE_THREADS;
int SOE_VFLAG;
//this array holds NUM_P primes in the range P_MIN to P_MAX, and
//can change as needed - always check the range and size to see
//if the primes you need are in there before using it
uint64_t *PRIMES;
uint64_t NUM_P;
uint64_t P_MIN;
uint64_t P_MAX;

//this array holds a global store of prime numbers
uint32_t *spSOEprimes;	//the primes	
uint32_t szSOEp;			//count of primes

#endif // YAFU_SOE_H
