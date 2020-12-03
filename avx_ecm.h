/*
Copyright (c) 2019, Ben Buhrow
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <malloc.h>
#include <sys/time.h>	//for gettimeofday using gcc
#include <time.h>
#include "util.h"
#include "gmp.h"
#include "queue.h"
#include <stdint.h>

#define INV_2_POW_64 5.4210108624275221700372640043497e-20

#if defined (__INTEL_COMPILER)
#define ALIGNED_MEM __declspec(align(64))
#else
#define ALIGNED_MEM __attribute__((aligned(64)))
#endif

#define DEFINED 1
#define MAX_WINSIZE 8
#define BLOCKWORDS 4
#define INLINE __inline
#define strto_uint64_t strtoull
#define DEC 10
#define HEX 16

//#ifndef MAXBITS
//#define MAXBITS 1040
//#endif
//#define DIGITBITS 32

#ifndef DIGITBITS
#define DIGITBITS 52
#endif

#if DIGITBITS==52

#define base_t uint64_t
#define base_signed_t int64_t

#define HALFBITS 26
#define HALFMASK 0x3ffffff
#define MAXDIGIT 0xfffffffffffffULL
#define HIBITMASK 0x8000000000000ULL
#define VECLEN 8

#elif DIGITBITS==32

#define base_t uint32_t
#define base_signed_t int32_t

#define HALFBITS 16
#define HALFMASK 0xffff
#define MAXDIGIT 0xffffffff
#define HIBITMASK 0x80000000
#define VECLEN 16

#else
#error "DIGITBITS must be either 52 or 32"
#endif

#ifdef WIN64
#define PRId64 "lld"
#define PRIu64 "llu"
#define PRIx64 "llx"
#define mpz_get_ui(x) ( ((x)->_mp_d[0]) )
#define mpz_set_ui(x, y) ( ((x)->_mp_d[0]) = (y) , (x)->_mp_size = 1 )
#else
#define PRId64 "ld"
#define PRIu64 "lu"
#define PRIx64 "lx"
#endif

uint32_t MAXBITS;
uint32_t NWORDS;
uint32_t NBLOCKS;

typedef struct
{
    base_t *data;
    int size;
    uint32_t signmask;
} bignum;

uint64_t *lcg_state;

typedef struct {
	long		secs;
	long		usecs;
} TIME_DIFF;

// a vector math library for montgomery arithmetic using AVX-512
typedef struct
{
    mpz_t nhat;
    mpz_t rhat;
    mpz_t gmp_t1;
    mpz_t gmp_t2;
    bignum *r;
    bignum *n;
    bignum *vnhat;
    bignum *vrhat;
    bignum *rmask;
    bignum *one;
    bignum *mtmp1;
    bignum *mtmp2;
    bignum *mtmp3;
    bignum *mtmp4;
    bignum **g;             // storage for windowed method precomputation
    base_t *vrho;
    base_t rho;
    int nbits;
    int isMersenne;
} monty;

void print_vechexbignum(bignum* a, const char* pre);
void print_vechexbignum52(bignum* a, const char* pre);
void print_hexbignum(bignum *a, const char *pre);
void print_hex(bignum *a, const char *pre);
void print_vechex(base_t *a, int v, int n, const char *pre);
monty * monty_alloc(void);
void monty_free(monty *mdata);
void copy_vec_lane(bignum *src, bignum *dest, int num, int size);
void vecCopy(bignum * src, bignum * dest);
void vecCopyn(bignum * src, bignum * dest, int size);
void vecClear(bignum *n);
bignum * vecInit(void);
void vecFree(bignum *);

// 52-BIT functions
void vecmulmod52_1(bignum *a, base_t *b, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecredc52_base(bignum *a, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecmulmod52(bignum *a, bignum *b, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsqrmod52(bignum *a, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsubmod52(bignum* a, bignum* b, bignum* c, monty* mdata);
void vecaddmod52(bignum* a, bignum* b, bignum* c, monty* mdata);
void vec_simul_addsub52(bignum* a, bignum* b, bignum* sum, bignum* diff, monty* mdata);
void vecmulmod52_mersenne(bignum* a, bignum* b, bignum* c, bignum* n, bignum* s, monty* mdata);
void vecsqrmod52_mersenne(bignum* a, bignum* c, bignum* n, bignum* s, monty* mdata);
void vecsubmod52_mersenne(bignum* a, bignum* b, bignum* c, monty* mdata);
void vecaddmod52_mersenne(bignum* a, bignum* b, bignum* c, monty* mdata);
void vec_simul_addsub52_mersenne(bignum* a, bignum* b, bignum* sum, bignum* diff, monty* mdata);
void vec_bignum52_mask_sub(bignum *a, bignum *b, bignum *c, uint32_t wmask);
void vec_bignum52_mask_rshift_1(bignum * u, uint32_t wmask);
uint32_t vec_bignum52_mask_lshift_1(bignum * u, uint32_t wmask);
uint32_t vec_bignum52_mask_lshift_n(bignum * u, int n, uint32_t wmask);
uint32_t vec_eq52(base_t * u, base_t * v, int sz);
uint32_t vec_gte52(bignum * u, bignum * v);
void vec_bignum52_add_1(bignum *a, base_t *b, bignum *c);

// 32-BIT functions
void vecmulmod(bignum *a, bignum *b, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsqrmod(bignum *a, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsubmod(bignum *a, bignum *b, bignum *c, monty *mdata);
void vecaddmod(bignum *a, bignum *b, bignum *c, monty *mdata);
void vec_simul_addsub(bignum *a, bignum *b, bignum *sum, bignum *diff, monty* mdata);
void vecmulmod_mersenne(bignum* a, bignum* b, bignum* c, bignum* n, bignum* s, monty* mdata);
void vecsqrmod_mersenne(bignum* a, bignum* c, bignum* n, bignum* s, monty* mdata);
void vecsubmod_mersenne(bignum* a, bignum* b, bignum* c, monty* mdata);
void vecaddmod_mersenne(bignum* a, bignum* b, bignum* c, monty* mdata);
void vec_simul_addsub_mersenne(bignum* a, bignum* b, bignum* sum, bignum* diff, monty* mdata);
void vec_bignum_mask_sub(bignum *a, bignum *b, bignum *c, uint32_t wmask);
void vec_bignum_mask_rshift_1(bignum * u, uint32_t wmask);
uint32_t vec_bignum_mask_lshift_1(bignum * u, uint32_t wmask);
uint32_t vec_eq(base_t * u, base_t * v, int sz);
uint32_t vec_gte(bignum * u, bignum * v);

void extract_bignum_from_vec_to_mpz(mpz_t dest, bignum *vec_src, int num, int sz);
void broadcast_mpz_to_vec(bignum *vec_dest, mpz_t src);
void insert_mpz_to_vec(bignum *vec_dest, mpz_t src, int lane);

void(*vecmulmod_ptr)(bignum *, bignum *, bignum *, bignum *, bignum *, monty *);
void(*vecsqrmod_ptr)(bignum *, bignum *, bignum *, bignum *, monty *);
void(*vecaddmod_ptr)(bignum *, bignum *, bignum *, monty*);
void(*vecsubmod_ptr)(bignum *, bignum *, bignum *, monty*);
void(*vecaddsubmod_ptr)(bignum *, bignum *, bignum *, bignum *, monty*);

// ecm stuff
typedef struct 
{
	bignum *X;
	bignum *Z;
} ecm_pt;

typedef struct 
{
	bignum *sum1;
	bignum *diff1;
	bignum *sum2;
	bignum *diff2;
	bignum *tt1;
	bignum *tt2;
	bignum *tt3;
	bignum *tt4;
	bignum *tt5;
	bignum *s;
	bignum *n;
	ecm_pt pt1;
	ecm_pt pt2;
	ecm_pt pt3;
	ecm_pt pt4;
	ecm_pt pt5;
	uint64_t sigma;

	uint32_t *map;
	ecm_pt *Pa;
	ecm_pt *Pd;
	ecm_pt *Pad;
    bignum** Paprod;
    bignum** Pa_inv;
	bignum **Pbprod;
	ecm_pt *Pb;
    ecm_pt *Pdnorm;
	bignum *stg2acc;
    uint32_t paired;
	uint32_t ptadds;
    uint32_t ptdups;
    uint32_t numinv;
	uint64_t numprimes;
    uint64_t A;
    uint32_t last_pid;
	uint32_t amin;

	uint32_t U;
	uint32_t L;
	uint32_t D;
	uint32_t R;

} ecm_work;


typedef struct
{
    mpz_t factor;
    uint64_t *sigma;
    monty *mdata;
    ecm_work *work;
    uint32_t curves;
    uint32_t b1;
    uint32_t b2;
    ecm_pt *P;
    uint64_t lcg_state;
    uint32_t tid;
    uint32_t total_threads;
    uint32_t phase_done;
    uint32_t ecm_phase;     // 0 == build curve, 1 == stage 1, 2 == stage 2
    uint32_t *pairmap_v;
    uint32_t *pairmap_u;
    uint32_t pairmap_steps;
    uint32_t* Qmap;
    uint32_t* Qrmap;
    Queue_t** Q;
	int save_b1;
} thread_data_t;

typedef struct
{
    thread_data_t *tdata;
    
} process_data_t;

void vececm(thread_data_t *tdata);
void ecm_pt_init(ecm_pt *pt);
void ecm_pt_free(ecm_pt *pt);
void ecm_work_init(ecm_work *work);
void ecm_work_free(ecm_work *work);

// global array of primes
base_t nump;

// global limits
uint64_t STAGE1_MAX;
uint64_t STAGE2_MAX;
uint32_t PRIME_RANGE;
int DO_STAGE2;

