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


#include "avx_ecm.h"
#include "omp.h"
#include "eratosthenes/soe.h"
#include <unistd.h> 
#include "queue.h"
#include "gmp.h"
#include "calc.h"

// performance comparison
// http://www.mersenneforum.org/showthread.php?t=16480&page=20
// http://www.mersenneforum.org/showthread.php?t=5722&page=122

// t-level estimate
// http://mersenneforum.org/showpost.php?p=427989&postcount=2429


uint32_t spRand(uint64_t *lcg_state, uint32_t lower, uint32_t upper);
void thread_init(thread_data_t *tdata, monty *mdata);

static int debugctr = 0;

#ifdef WIN64
//#define mpz_get_ui(x) ( (((x)->_mp_d[1]) << 32) | ((x)->_mp_d[0]) )

#endif

void extract_bignum_from_vec_to_mpz(mpz_t dest, bignum *vec_src, int num, int sz)
{
    int j;

    if (dest == NULL)
    {
        printf("invalid dest address in extract_vec_bignum_from_vec_to_mpz\n");
    }

    mpz_set_ui(dest, 0);
    for (j = sz - 1; j >= 0; j--)
    {
        
#if defined(WIN64) && (DIGITBITS == 52)
        mpz_mul_2exp(dest, dest, 20);
        mpz_add_ui(dest, dest, vec_src->data[num + j * VECLEN] >> 32);
        mpz_mul_2exp(dest, dest, 32);
        mpz_add_ui(dest, dest, (vec_src->data[num + j * VECLEN]) & 0xffffffff);
#else
        mpz_mul_2exp(dest, dest, DIGITBITS);
        mpz_add_ui(dest, dest, vec_src->data[num + j * VECLEN]);
#endif
        //gmp_printf("word is %016llx, dest is now %Zx\n", vec_src->data[num + j * VECLEN], dest);
    }

    return;
}

void broadcast_mpz_to_vec(bignum *vec_dest, mpz_t src)
{
    mpz_t src_cp;
    int i, j;

    mpz_init(src_cp);
    mpz_set(src_cp, src);

    i = 0;
    vec_dest->size = 0;
    while (mpz_cmp_ui(src_cp, 0) > 0)
    {
        base_t thisword = mpz_get_ui(src_cp) & MAXDIGIT;
        for (j = 0; j < VECLEN; j++)
        {
            vec_dest->data[j + i * VECLEN] = thisword;
        }
        vec_dest->size++;
        i++;
        mpz_tdiv_q_2exp(src_cp, src_cp, DIGITBITS);
    }

    mpz_clear(src_cp);
    return;
}

void insert_mpz_to_vec(bignum *vec_dest, mpz_t src, int lane)
{
    mpz_t src_cp;
    int i;

    mpz_init(src_cp);
    mpz_set(src_cp, src);

    i = 0;
    vec_dest->size = 0;
    while (mpz_cmp_ui(src_cp, 0) > 0)
    {
        base_t thisword = mpz_get_ui(src_cp) & MAXDIGIT;
        vec_dest->data[lane + i * VECLEN] = thisword;
        i++;
        mpz_tdiv_q_2exp(src_cp, src_cp, DIGITBITS);
    }

    vec_dest->size = MAX(vec_dest->size, i);
    mpz_clear(src_cp);
    return;
}


int main(int argc, char **argv)
{
    thread_data_t *tdata;
	bignum **f, *n;
    mpz_t gmpn, g, r;
	uint32_t *siglist;
	uint32_t numcurves;
	uint32_t numcurves_per_thread;
	uint32_t b1;
	uint32_t i, j;
	char **nextptr;
	monty *montyconst;
	int threads;    
	int pid = getpid();
    uint64_t limit;
    int size_n;
    str_t input;

    // primes
    uint32_t seed_p[6542];
    uint32_t numSOEp;

	// timing variables
	struct timeval stopt;	// stop time of this job
	struct timeval startt;	// start time of this job
	double t_time;

    if (argc < 4)
    {
        printf("usage: avx-ecm $input $numcurves $B1 [$threads] [$B2]\n");
        exit(1);
    }
    else
    {
        
    }
	
    //printf("ECM has been configured with MAXBITS = %d, NWORDS = %d, "
    //    "VECLEN = %d\n", 
    //    MAXBITS, NWORDS, VECLEN);

	gettimeofday(&startt, NULL);

	if (pid <= 0)
		pid = startt.tv_usec;

	printf("starting process %d\n", pid);

    mpz_init(gmpn);
    mpz_init(g);
    mpz_init(r);

    sInit(&input);
    calc_init();
    toStr(argv[1], &input);
    calc(&input);
    calc_finalize();

    mpz_set_str(gmpn, input.s, 10);
    gmp_printf("commencing parallel ecm on %Zd\n", gmpn);
    size_n = mpz_sizeinbase(gmpn, 2);
	numcurves = strtoul(argv[2], NULL, 10);
	b1 = strtoul(argv[3], NULL, 10);	
	STAGE1_MAX = b1;
	STAGE2_MAX = 100ULL * (uint64_t)b1;

    if (DIGITBITS == 52)
    {
        MAXBITS = 208;
        while (MAXBITS <= size_n)
        {
            MAXBITS += 208;
        }
    }
    else
    {
        MAXBITS = 128;
        while (MAXBITS <= size_n)
        {
            MAXBITS += 128;
        }
    }

    NWORDS = MAXBITS / DIGITBITS;
    NBLOCKS = NWORDS / BLOCKWORDS;

    printf("ECM has been configured with DIGITBITS = %u, VECLEN = %d, GMP_LIMB_BITS = %d\n",
        DIGITBITS, VECLEN, GMP_LIMB_BITS);

    printf("Choosing MAXBITS = %u, NWORDS = %d, NBLOCKS = %d based on input size %d\n",
        MAXBITS, NWORDS, NBLOCKS, size_n);

    SOE_VFLAG = 0;
	threads = SOE_THREADS = 1;
	if (argc >= 5)
        threads = atoi(argv[4]);
    SOE_THREADS = 2;

    DO_STAGE2 = 1;
    if (argc == 6)
	{
		STAGE2_MAX = strtoull(argv[5], NULL, 10);

		if (STAGE2_MAX <= STAGE1_MAX)
		{
			DO_STAGE2 = 0;
			STAGE2_MAX = STAGE1_MAX;
		}
	}
        

    if (STAGE1_MAX < 1000)
    {
        printf("stage 1 too small\n");
		exit(0);
    }

    szSOEp = 50000000;
    numSOEp = tiny_soe(65537, seed_p);
	PRIMES = soe_wrapper(seed_p, numSOEp, 0, szSOEp, 0, &limit);

    //save a batch of sieve primes too.
    spSOEprimes = (uint32_t *)malloc((size_t)(limit * sizeof(uint32_t)));
    for (i = 0; i < limit; i++)
    {
        spSOEprimes[i] = (uint32_t)PRIMES[i];
    }

    szSOEp = limit;
    PRIME_RANGE = 100000000;

    printf("cached %u primes < %u\n", szSOEp, spSOEprimes[limit - 1]);

	if (numcurves < threads)
		numcurves = threads;
	
	numcurves_per_thread = numcurves / threads + (numcurves % threads != 0);
	numcurves = numcurves_per_thread * threads;

    printf("Input has %d bits, using %d threads (%d curves/thread)\n", 
        mpz_sizeinbase(gmpn, 2), threads, numcurves_per_thread);
    printf("Processing in batches of %u primes\n", PRIME_RANGE);

    tdata = (thread_data_t *)malloc(threads * sizeof(thread_data_t));
    // expects n to be in packed 64-bit form
    montyconst = monty_alloc();

    mpz_set_ui(r, 1);
    mpz_mul_2exp(r, r, DIGITBITS * NWORDS);
    //gmp_printf("r = %Zd\n", r);
    mpz_invert(montyconst->nhat, gmpn, r);
    mpz_sub(montyconst->nhat, r, montyconst->nhat);
    mpz_invert(montyconst->rhat, r, gmpn);
    broadcast_mpz_to_vec(montyconst->n, gmpn);
    broadcast_mpz_to_vec(montyconst->r, r);
    broadcast_mpz_to_vec(montyconst->vrhat, montyconst->rhat);
    broadcast_mpz_to_vec(montyconst->vnhat, montyconst->nhat);
    mpz_tdiv_r(r, r, gmpn);
    broadcast_mpz_to_vec(montyconst->one, r);
    //gmp_printf("n = %Zx\n", gmpn);
    //gmp_printf("rhat = %Zx\n", montyconst->rhat);
    //gmp_printf("nhat = %Zx\n", montyconst->nhat);
    //gmp_printf("one = %Zx\n", r);
    //printf("rho = %016llx\n", mpz_get_ui(montyconst->nhat) & MAXDIGIT);
    for (i = 0; i < VECLEN; i++)
    {
        montyconst->vrho[i] = mpz_get_ui(montyconst->nhat) & MAXDIGIT;
    }
    
    if (DIGITBITS == 52)
    {
        vecmulmod_ptr = &vecmulmod52;
        vecsqrmod_ptr = &vecsqrmod52;
        vecaddmod_ptr = &vecaddmod52;
        vecsubmod_ptr = &vecsubmod52;
        vecaddsubmod_ptr = &vec_simul_addsub52;
    }
    else
    {
        vecmulmod_ptr = &vecmulmod;
        vecsqrmod_ptr = &vecsqrmod;
        vecaddmod_ptr = &vecaddmod;
        vecsubmod_ptr = &vecsubmod;
        vecaddsubmod_ptr = &vec_simul_addsub;
    }

	gettimeofday(&stopt, NULL);

    for (i = 0; i < threads; i++)
    {
        int j;
        thread_init(&tdata[i], montyconst);
        tdata[i].curves = numcurves_per_thread;
        tdata[i].tid = i;
		tdata[i].lcg_state = hash64(stopt.tv_usec + i) + hash64(pid); // 
        tdata[i].total_threads = threads;
    }

	gettimeofday(&stopt, NULL);
    t_time = my_difftime(&startt, &stopt);

    printf("Initialization took %1.4f seconds.\n", t_time);

    // top level ECM 
    vececm(tdata);    
	printf("\n");

    // clean up thread data
	for (i = 0; i < threads; i++)
	{
        mpz_clear(tdata[i].factor);
        ecm_work_free(tdata[i].work);
        ecm_pt_free(tdata[i].P);
        monty_free(tdata[i].mdata);
        free(tdata[i].mdata);
        free(tdata[i].work);
        free(tdata[i].P);
        free(tdata[i].sigma);
	}

    // clean up local/global data
    mpz_clear(gmpn);
    mpz_clear(g);
    mpz_clear(r);
    monty_free(montyconst);
	free(montyconst);
    free(tdata);
    sFree(&input);

	return 0;
}

void thread_init(thread_data_t *tdata, monty *mdata)
{    
    mpz_init(tdata->factor);
    tdata->work = (ecm_work *)malloc(sizeof(ecm_work));
    tdata->P = (ecm_pt *)malloc(sizeof(ecm_pt));
    tdata->sigma = (uint64_t *)malloc(VECLEN * sizeof(uint64_t));
	uint32_t D = tdata->work->D = 1155;
	tdata->work->R = 480 + 3;

	// decide on the stage 2 parameters.  Larger U means
	// more memory and more setup overhead, but more prime pairs.
	// Smaller U means the opposite.  find a good balance.
	static double pairing[4] = { 0.7283, 0.6446, 0.5794, 0.5401 };
	static double adds[4];
	double best = 99999999999.;
	int bestU = 4;

	if (STAGE1_MAX <= 8192)
	{
		D = tdata->work->D = 210;
		tdata->work->R = 48 + 3;
	}

	adds[0] = (double)D * 1.0;
	adds[1] = (double)D * 2.0;
	adds[2] = (double)D * 4.0;
	adds[3] = (double)D * 8.0;

	int i;
	for (i = 1; i < 4; i++)
	{
		int numadds = (STAGE2_MAX - STAGE1_MAX) / (2 * D);
		double addcost = 6.0 * ((double)numadds + adds[i]);
		double paircost = ((double)STAGE2_MAX / log((double)STAGE2_MAX) -
			(double)STAGE1_MAX / log((double)STAGE1_MAX)) * pairing[i] * 2.0;

		//printf("estimating %u primes paired\n", (uint32_t)((double)STAGE2_MAX / log((double)STAGE2_MAX) -
		//	(double)STAGE1_MAX / log((double)STAGE1_MAX)));
		//printf("%d adds + %d setup adds\n", numadds, (int)adds[i]);
		//printf("addcost = %f\n", addcost);
		//printf("paircost = %f\n", paircost);
		//printf("totalcost = %f\n", addcost + paircost);

		if ((addcost + paircost) < best)
		{
			best = addcost + paircost;
			bestU = 1 << i;
		}
	}

	tdata->work->U = bestU;
	tdata->work->L = bestU * 2;

    ecm_work_init(tdata->work);
    ecm_pt_init(tdata->P);

    // allocate and then copy some constants over to this thread's mdata structure.
    tdata->mdata = monty_alloc();
    mpz_set(tdata->mdata->nhat, mdata->nhat);
    mpz_set(tdata->mdata->rhat, mdata->rhat);
    vecCopy(mdata->n, tdata->mdata->n);
    vecCopy(mdata->n, tdata->work->n);
    vecCopy(mdata->one, tdata->mdata->one);
    vecCopy(mdata->vnhat, tdata->mdata->vnhat);
    vecCopy(mdata->vrhat, tdata->mdata->vrhat);
    memcpy(tdata->mdata->vrho, mdata->vrho, VECLEN * sizeof(base_t));

    return;
}

double my_difftime (struct timeval * start, struct timeval * end)
{
	TIME_DIFF diff;

	if (start->tv_sec == end->tv_sec) {
		diff.secs = 0;
		diff.usecs = end->tv_usec - start->tv_usec;
	}
	else {
		diff.usecs = 1000000 - start->tv_usec;
		diff.secs = end->tv_sec - (start->tv_sec + 1);
		diff.usecs += end->tv_usec;
		if (diff.usecs >= 1000000) {
			diff.usecs -= 1000000;
			diff.secs += 1;
		}
	}
	
    return ((double)diff.secs + (double)diff.usecs / 1000000.);
}

uint64_t lcg_rand(uint64_t *lcg_state)
{
	// Knuth's MMIX LCG
	*lcg_state = 6364136223846793005ULL * *lcg_state + 1442695040888963407ULL;
	return *lcg_state;
}

#define INV_2_POW_32 2.3283064365386962890625e-10
// Knuth's 64 bit MMIX LCG, using a global 64 bit state variable.
uint32_t spRand(uint64_t *lcg_state, uint32_t lower, uint32_t upper)
{
	// advance the state of the LCG and return the appropriate result
	*lcg_state = 6364136223846793005ULL * (*lcg_state) + 1442695040888963407ULL;
	return lower + (uint32_t)(
		(double)(upper - lower) * (double)(*lcg_state >> 32) * INV_2_POW_32);
}

// =============== 64-bit hashing ================ //
// FNV-1 hash algorithm:
// http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
uint64_t hash64(uint64_t in)
{
	uint64_t hash = 14695981039346656037ULL;
	uint64_t prime = 1099511628211ULL;
	uint64_t hash_mask;
	uint64_t xor;
	
	hash = hash * prime;
	hash_mask = 0xffffffffffffff00ULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0xffffffffffff00ffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0xffffffffff00ffffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0xffffffff00ffffffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0xffffff00ffffffffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0xffff00ffffffffffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0xff00ffffffffffffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	hash = hash * prime;
	hash_mask = 0x00ffffffffffffffULL;
	xor = hash ^ in;
	hash = (hash & hash_mask) | (xor & (~hash_mask));

	return hash;
}
