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

//#define HALF_VEC
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

#endif

//#define NWORDS (MAXBITS / DIGITBITS)
#define PRId64 "ld"
#define PRIu64 "lu"
#define PRIx64 "lx"

//#if (MAXBITS % DIGITBITS) != 0
//#error "MAXBITS must be divisble by DIGITBITS"
//#endif

uint32_t MAXBITS;
uint32_t NWORDS;
uint32_t NBLOCKS;


typedef struct
{
    base_t *data;
    int size;
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
} monty;

void print_vechexbignum(bignum *a, const char *pre);
void print_hexbignum(bignum *a, const char *pre);
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
void vecmulmod52(bignum *a, bignum *b, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsqrmod52(bignum *a, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsubmod52(bignum *a, bignum *b, bignum *c, bignum *n);
void vecaddmod52(bignum *a, bignum *b, bignum *c, bignum *n);
void vec_simul_addsub52(bignum *a, bignum *b, bignum *sum, bignum *diff, bignum *n);
void vec_bignum52_mask_sub(bignum *a, bignum *b, bignum *c, uint32_t wmask);
void vec_bignum52_mask_rshift_1(bignum * u, uint32_t wmask);
uint32_t vec_bignum52_mask_lshift_1(bignum * u, uint32_t wmask);
uint32_t vec_eq52(base_t * u, base_t * v, int sz);
uint32_t vec_gte52(bignum * u, bignum * v);

// 32-BIT functions
void vecmulmod(bignum *a, bignum *b, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsqrmod(bignum *a, bignum *c, bignum *n, bignum *s, monty *mdata);
void vecsubmod(bignum *a, bignum *b, bignum *c, bignum *n);
void vecaddmod(bignum *a, bignum *b, bignum *c, bignum *n);
void vec_simul_addsub(bignum *a, bignum *b, bignum *sum, bignum *diff, bignum *n);
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
void(*vecaddmod_ptr)(bignum *, bignum *, bignum *, bignum *);
void(*vecsubmod_ptr)(bignum *, bignum *, bignum *, bignum *);
void(*vecaddsubmod_ptr)(bignum *, bignum *, bignum *, bignum *, bignum *);

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
	base_t sigma;

	uint8_t *marks;
	uint8_t *nmarks;
	uint32_t *map;
	ecm_pt *Pa;
	ecm_pt *Pd;
	ecm_pt *Pad;
	bignum **Paprod;	
	bignum **Pbprod;
	ecm_pt *Pb;
	bignum *stg2acc;
	uint32_t stg1Add;
	uint32_t stg1Doub;
    uint32_t paired;
	uint32_t ptadds;
	uint64_t numprimes;
    uint64_t A;
    uint32_t last_pid;
	uint32_t amin;

	uint32_t U;
	uint32_t L;
	uint32_t D;
	uint32_t R;

	uint32_t *Qmap;
	uint32_t *Qrmap;
	Queue_t **Q;

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
base_t STAGE1_MAX;
uint64_t STAGE2_MAX;
uint32_t PRIME_RANGE;
int DO_STAGE2;

