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

// todo:
// IFMA and benchmarking
// proper command line flags
// smarter use of primes array when stage 2 needs multiple ranges
// test script - other inputs, other B1/B2, check for correct factors
// options to control memory use (KNL)
// option to run gmp-ecm stage2 hybrid plan
// option -one (otherwise reduce and continue)
// -power 2
// -power N


uint32_t spRand(uint64_t *lcg_state, uint32_t lower, uint32_t upper);
void thread_init(thread_data_t *tdata, monty *mdata);

static int debugctr = 0;

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

#define NUM_SMALL_P 168
int small_p[NUM_SMALL_P] = { 2,3,5,7,11,13,17,19,23,29,
    31,37,41,43,47,53,59,61,67,71,
    73,79,83,89,97,101,103,107,109,
    113,127,131,137,139,149,151,157,
    163,167,173,179,181,191,193,197,
    199,211,223,227,229,233,239,241,
    251,257,263,269,271,277,281,283,
    293,307,311,313,317,331,337,347,
    349,353,359,367,373,379,383,389,
    397,401,409,419,421,431,433,439,
    443,449,457,461,463,467,479,487,
    491,499,503,509,521,523,541,547,
    557,563,569,571,577,587,593,599,
    601,607,613,617,619,631,641,643,
    647,653,659,661,673,677,683,691,
    701,709,719,727,733,739,743,751,
    757,761,769,773,787,797,809,811,
    821,823,827,829,839,853,857,859,
    863,877,881,883,887,907,911,919,
    929,937,941,947,953,967,971,977,
    983,991,997};

int tdiv_int(int x, int* factors)
{
    int numf = 0;
    int xx = x;
    int i;

    i = 0;
    while ((xx > 1) && (i < NUM_SMALL_P))
    {
        int q = (int)small_p[i];

        if (xx % q != 0)
            i++;
        else
        {
            xx /= q;
            factors[numf++] = q;
        }
    }

    return numf;
}

// see: http://home.earthlink.net/~elevensmooth/MathFAQ.html#PrimDistinct
void find_primitive_factor(mpz_t primitive, int exp1, int base1, int64_t coeff2)
{
    // factor the exponent.  The algebraic reductions yafu knows how to handle are
    // for cunningham and homogenous cunningham inputs where the exponent is in
    // the exp1 field.
    int e = exp1;
    int f[32];
    int nf, i, j, k, m, mult;
    // ranks of factors - we support up to 3 distinct odd factors of e
    int franks[4][32];		// and beans
    // and counts of the factors in each rank
    int cranks[4];			// doesn't that make you happy?   <-- bonus points if you get this...
    int nr, mrank;
    mpz_t n, term, t;
    int VFLAG = 3;

    nf = tdiv_int(e, f);

    for (i = 0; i < 4; i++)
        cranks[i] = 0;

    // now arrange the factors into ranks of combinations of unique, distinct, and odd factors.
    // rank 0 is always 1
    franks[0][0] = 1;
    cranks[0] = 1;

    // rank 1 is a list of the distinct odd factors.
    j = 0;
    if (VFLAG > 1) printf("gen: rank 1 terms: ");
    for (i = 0; i < nf; i++)
    {
        if (f[i] & 0x1)
        {
            // odd
            if (j == 0 || f[i] != franks[1][j - 1])
            {
                // distinct
                franks[1][j++] = f[i];
                if (VFLAG > 1) printf("%d ", f[i]);
            }
        }
    }
    if (VFLAG > 1) printf("\n");
    cranks[1] = j;
    nr = j + 1;

    if (j > 3)
    {
        printf("gen: too many distinct odd factors in exponent!\n");
        exit(1);
    }

    // ranks 2...nf build on the first rank combinatorially.
    // knuth, of course, has a lot to say on enumerating combinations:
    // http://www.cs.utsa.edu/~wagner/knuth/fasc3a.pdf
    // in which algorithm T might be sufficient since e shouldn't have too many
    // factors.
    // but since e shouldn't have too many factors and I don't feel like implementing
    // algorithm T from that reference right now, I will proceed to hardcode a bunch of
    // simple loops.
    // here is the second rank, if necessary
    if (cranks[1] == 2)
    {
        franks[2][0] = franks[1][0] * franks[1][1];
        cranks[2] = 1;
        if (VFLAG > 1) printf("gen: rank 2 term: %d\n", franks[2][0]);
    }
    else if (cranks[1] == 3)
    {
        // combinations of 2 primes
        m = 0;
        if (VFLAG > 1) printf("gen: rank 2 terms: ");
        for (j = 0; j < cranks[1] - 1; j++)
        {
            for (k = j + 1; k < cranks[1]; k++)
            {
                franks[2][m++] = franks[1][j] * franks[1][k];
                if (VFLAG > 1) printf("%d ", franks[2][m - 1]);
            }
        }
        cranks[2] = m;
        if (VFLAG > 1) printf("\n");

        // combinations of 3 primes
        franks[3][0] = franks[1][0] * franks[1][1] * franks[1][2];
        cranks[3] = 1;
        if (VFLAG > 1) printf("gen: rank 3 term: %d\n", franks[3][0]);
    }

    // for exponents with repeated or even factors, find the multiplier
    mult = e;
    for (i = 0; i < cranks[1]; i++)
        mult /= franks[1][i];
    if (VFLAG > 1) printf("gen: base exponent multiplier: %d\n", mult);

    // form the primitive factor, following the rank system of
    // http://home.earthlink.net/~elevensmooth/MathFAQ.html#PrimDistinct
    mpz_init(n);
    mpz_set_ui(n, 1);
    mpz_init(term);
    mpz_init(t);
    if ((nr & 0x1) == 1)
        mrank = 0;
    else
        mrank = 1;
    for (i = nr - 1; i >= 0; i--)
    {
        char c;
        if ((i & 0x1) == mrank)
        {
            // multiply by every other rank - do this before the division
            for (j = 0; j < cranks[i]; j++)
            {
                mpz_set_ui(term, base1);
                mpz_pow_ui(term, term, franks[i][j] * mult);
                if (coeff2 < 0) {
                    mpz_sub_ui(term, term, 1); c = '-';
                }
                else {
                    mpz_add_ui(term, term, 1); c = '+';
                }
                if (VFLAG > 1) gmp_printf("gen: multiplying by %d^%d %c 1 = %Zd\n",
                    base1, franks[i][j] * mult, c, term);

                mpz_mul(n, n, term);
            }
        }
    }
    for (i = nr - 1; i >= 0; i--)
    {
        char c;
        if ((i & 0x1) == (!mrank))
        {
            // divide by every other rank
            for (j = 0; j < cranks[i]; j++)
            {
                mpz_set_ui(term, base1);
                mpz_pow_ui(term, term, franks[i][j] * mult);
                if (coeff2 < 0) {
                    mpz_sub_ui(term, term, 1); c = '-';
                }
                else {
                    mpz_add_ui(term, term, 1); c = '+';
                }
                if (VFLAG > 1) gmp_printf("gen: dividing by %d^%d %c 1 = %Zd\n",
                    base1, franks[i][j] * mult, c, term);

                mpz_mod(t, n, term);
                if (mpz_cmp_ui(t, 0) != 0)
                {
                    printf("gen: error, term doesn't divide n!\n");
                }
                else
                {
                    mpz_tdiv_q(n, n, term);
                }
            }
        }
    }

    mpz_set(primitive, n);

    mpz_clear(n);
    mpz_clear(term);
    mpz_clear(t);
    return;
}

int main(int argc, char **argv)
{
    thread_data_t *tdata;
    mpz_t gmpn, g, r;
	uint32_t numcurves;
	uint32_t numcurves_per_thread;
	uint64_t b1;
	uint32_t i;
	monty *montyconst;
	int threads;    
	int pid = getpid();
    uint64_t limit, sigma = 0;
    int size_n;
    str_t input;
	int isMersenne = 0, forceNoMersenne = 0;

    // primes
    uint32_t seed_p[6542];
    uint32_t numSOEp;

	// timing variables
	struct timeval stopt;	// stop time of this job
	struct timeval startt;	// start time of this job
	double t_time;

    if (argc < 4)
    {
        printf("usage: avx-ecm $input $numcurves $B1 [$threads] [$B2] [$sigma]\n");
        exit(1);
    }
	
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
    
	// check for Mersenne inputs
    size_n = mpz_sizeinbase(gmpn, 2);

    for (i = size_n - 1; i < 2048; i++)
    {
        mpz_set_ui(r, 1);
        mpz_mul_2exp(r, r, i);
        mpz_sub_ui(r, r, 1);
        mpz_mod(g, r, gmpn);
        if (mpz_cmp_ui(g, 0) == 0)
        {
            size_n = i;
            isMersenne = 1;
            break;
        }

        mpz_set_ui(r, 1);
        mpz_mul_2exp(r, r, i);
        mpz_add_ui(r, r, 1);
        mpz_mod(g, r, gmpn);
        if (mpz_cmp_ui(g, 0) == 0)
        {
            size_n = i;
            isMersenne = -1;
            break;
        }

        // detect pseudo-Mersennes
        mpz_set_ui(r, 1);
        mpz_mul_2exp(r, r, i);
        mpz_mod(g, r, gmpn);
        if (mpz_sizeinbase(g, 2) < DIGITBITS)
        {
            size_n = i;
            isMersenne = mpz_get_ui(g);
            break;
        }
    }

    // if the input is Mersenne and still contains algebraic factors, remove them.
    if (abs(isMersenne) == 1)
    {
        char ftype[8];
        find_primitive_factor(g, size_n, 2, -isMersenne);
        mpz_tdiv_q(r, gmpn, g);
        if (mpz_probab_prime_p(g, 3))
            strcpy(ftype, "PRP");
        else
            strcpy(ftype, "C");

        gmp_printf("removing algebraic %s%d factor %Zd\n", ftype, (int)mpz_sizeinbase(r, 10), r);
        mpz_gcd(gmpn, gmpn, g);
    }
		
	numcurves = strtoul(argv[2], NULL, 10);
	b1 = strtoull(argv[3], NULL, 10);	
	STAGE1_MAX = b1;
	STAGE2_MAX = 100ULL * b1;
	
	// compute NBLOCKS if using the actual size of the input (non-Mersenne)
    if (DIGITBITS == 52)
    {
        MAXBITS = 208;
        while (MAXBITS <= mpz_sizeinbase(gmpn, 2))
        {
            MAXBITS += 208;
        }
    }
    else
    {
        MAXBITS = 128;
        while (MAXBITS <= mpz_sizeinbase(gmpn, 2))
        {
            MAXBITS += 128;
        }
    }

    NWORDS = MAXBITS / DIGITBITS;
    NBLOCKS = NWORDS / BLOCKWORDS;

    // and compute NBLOCKS if using Mersenne mod
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

    gmp_printf("commencing parallel ecm on %Zd\n", gmpn);

    if (isMersenne && ((double)NWORDS / ((double)MAXBITS / (double)DIGITBITS) < 0.7))
    {
        char c;
        if (isMersenne > 0)
            c = '-';
        else
            c = '+';

        printf("Mersenne input 2^%d %c %d determined to be faster by REDC\n", size_n, c, isMersenne);
        forceNoMersenne = 1;
        MAXBITS = NWORDS * DIGITBITS;
    }
    else
    {
        NWORDS = MAXBITS / DIGITBITS;
        NBLOCKS = NWORDS / BLOCKWORDS;
    }

    if (forceNoMersenne)
    {
        isMersenne = 0;
        size_n = mpz_sizeinbase(gmpn, 2);
    }

    printf("ECM has been configured with DIGITBITS = %u, VECLEN = %d, GMP_LIMB_BITS = %d\n",
        DIGITBITS, VECLEN, GMP_LIMB_BITS);

    printf("Choosing MAXBITS = %u, NWORDS = %d, NBLOCKS = %d based on input size %d\n",
        MAXBITS, NWORDS, NBLOCKS, size_n);

    SOE_VFLAG = 0;
	threads = SOE_THREADS = 1;
    if (argc >= 5)
    {
        threads = atoi(argv[4]);
    }
    //SOE_THREADS = 2;

    DO_STAGE2 = 1;
    if (argc >= 6)
	{
		STAGE2_MAX = strtoull(argv[5], NULL, 10);

		if (STAGE2_MAX <= STAGE1_MAX)
		{
			DO_STAGE2 = 0;
			STAGE2_MAX = STAGE1_MAX;
		}
	}

    if (argc >= 7)
    {
        sigma = strtoull(argv[6], NULL, 10);
        printf("starting with sigma = %lu\n", sigma);
    }
        

    //if (STAGE1_MAX < 1000)
    //{
    //    printf("stage 1 too small\n");
	//	exit(0);
    //}
    //
    szSOEp = 100000000;
    numSOEp = tiny_soe(65537, seed_p);

	PRIMES = soe_wrapper(seed_p, numSOEp, 0, szSOEp, 0, &limit);

    //save a batch of sieve primes too.
    spSOEprimes = (uint32_t *)xmalloc((size_t)(limit * sizeof(uint32_t)));
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
        (int)mpz_sizeinbase(gmpn, 2), threads, numcurves_per_thread);
    printf("Processing in batches of %u primes\n", PRIME_RANGE);

    tdata = (thread_data_t *)xmalloc(threads * sizeof(thread_data_t));
    // expects n to be in packed 64-bit form
    montyconst = monty_alloc();

	if (isMersenne)
    {
        montyconst->isMersenne = isMersenne;
        montyconst->nbits = size_n;
        mpz_set(montyconst->nhat, gmpn);           // remember input N
        // do all math w.r.t the Mersenne number
        mpz_set_ui(gmpn, 1);
        mpz_mul_2exp(gmpn, gmpn, size_n);
        if (isMersenne > 0)
        {
            mpz_sub_ui(gmpn, gmpn, isMersenne);
        }
        else
        {
            mpz_add_ui(gmpn, gmpn, 1);
        }
        broadcast_mpz_to_vec(montyconst->n, gmpn);
        broadcast_mpz_to_vec(montyconst->vnhat, montyconst->nhat);
        mpz_set_ui(r, 1);
        broadcast_mpz_to_vec(montyconst->one, r);
    }
    else
    {
        montyconst->isMersenne = 0;
        montyconst->nbits = mpz_sizeinbase(gmpn, 2);
        mpz_set_ui(r, 1);
        mpz_mul_2exp(r, r, DIGITBITS * NWORDS);
        mpz_invert(montyconst->nhat, gmpn, r);
        mpz_sub(montyconst->nhat, r, montyconst->nhat);
        mpz_invert(montyconst->rhat, r, gmpn);
        broadcast_mpz_to_vec(montyconst->n, gmpn);
        broadcast_mpz_to_vec(montyconst->r, r);
        broadcast_mpz_to_vec(montyconst->vrhat, montyconst->rhat);
        broadcast_mpz_to_vec(montyconst->vnhat, montyconst->nhat);
        mpz_tdiv_r(r, r, gmpn);
        broadcast_mpz_to_vec(montyconst->one, r);
    }
	
    for (i = 0; i < VECLEN; i++)
    {
        montyconst->vrho[i] = mpz_get_ui(montyconst->nhat) & MAXDIGIT;
    }
    
    if (DIGITBITS == 52)
    {
        if (montyconst->isMersenne > 1)
        {
            vecmulmod_ptr = &vecmulmod52_mersenne;
            vecsqrmod_ptr = &vecsqrmod52_mersenne;
            vecaddmod_ptr = &vecaddmod52_mersenne;
            vecsubmod_ptr = &vecsubmod52_mersenne;
            vecaddsubmod_ptr = &vec_simul_addsub52_mersenne;
            printf("Using special pseudo-Mersenne mod for factor of: 2^%d-%d\n", 
                montyconst->nbits, montyconst->isMersenne);
        }
		else if (montyconst->isMersenne > 0)
        {
            vecmulmod_ptr = &vecmulmod52_mersenne;
            vecsqrmod_ptr = &vecsqrmod52_mersenne;
            vecaddmod_ptr = &vecaddmod52_mersenne;
            vecsubmod_ptr = &vecsubmod52_mersenne;
            vecaddsubmod_ptr = &vec_simul_addsub52_mersenne;
            printf("Using special Mersenne mod for factor of: 2^%d-1\n", montyconst->nbits);
        }
        else if (montyconst->isMersenne < 0)
        {
            vecmulmod_ptr = &vecmulmod52_mersenne;
            vecsqrmod_ptr = &vecsqrmod52_mersenne;
            vecaddmod_ptr = &vecaddmod52_mersenne;
            vecsubmod_ptr = &vecsubmod52_mersenne;
            vecaddsubmod_ptr = &vec_simul_addsub52_mersenne;
            printf("Using special Mersenne mod for factor of: 2^%d+1\n", montyconst->nbits);
        }
        else
        {
            vecmulmod_ptr = &vecmulmod52;
            vecsqrmod_ptr = &vecsqrmod52;
            vecaddmod_ptr = &vecaddmod52;
            vecsubmod_ptr = &vecsubmod52;
            vecaddsubmod_ptr = &vec_simul_addsub52;
        }
    }
    else
    {
        if (montyconst->isMersenne)
        {
            vecmulmod_ptr = &vecmulmod_mersenne;
            vecsqrmod_ptr = &vecsqrmod_mersenne;
            vecaddmod_ptr = &vecaddmod_mersenne;
            vecsubmod_ptr = &vecsubmod_mersenne;
            vecaddsubmod_ptr = &vec_simul_addsub_mersenne;
            printf("Using special Mersenne mod for factor of: 2^%d-1\n", montyconst->nbits);
        }
        else
        {
            vecmulmod_ptr = &vecmulmod;
            vecsqrmod_ptr = &vecsqrmod;
            vecaddmod_ptr = &vecaddmod;
            vecsubmod_ptr = &vecsubmod;
            vecaddsubmod_ptr = &vec_simul_addsub;
        }

        
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

        if (i == 0)
        {
            uint32_t D = tdata[i].work->D;
            int k;

            tdata[i].pairmap_v = (uint32_t*)calloc(PRIME_RANGE, sizeof(uint32_t));
            tdata[i].pairmap_u = (uint32_t*)calloc(PRIME_RANGE, sizeof(uint32_t));

            tdata[i].Qmap = (uint32_t*)malloc(2 * D * sizeof(uint32_t));
            tdata[i].Qrmap = (uint32_t*)malloc(2 * D * sizeof(uint32_t));

            for (j = 0, k = 0; k < 2 * D; k++)
            {
                if (spGCD(k, 2 * D) == 1)
                {
                    tdata[i].Qmap[k] = j;
                    tdata[i].Qrmap[j++] = k;
                }
                else
                {
                    tdata[i].Qmap[k] = (uint32_t)-1;
                }
            }

            for (k = j; k < 2 * D; k++)
            {
                tdata[i].Qrmap[k] = (uint32_t)-1;
            }

            tdata[i].Q = (Queue_t * *)malloc(j * sizeof(Queue_t*));
            for (k = 0; k < j; k++)
            {
                tdata[i].Q[k] = newQueue(D);
            }
        }
        else
        {
            tdata[i].pairmap_v = tdata[0].pairmap_v;
            tdata[i].pairmap_u = tdata[0].pairmap_u;
        }

        
        if (sigma > 0)
        {
            for (j = 0; j < VECLEN; j++)
            {
                tdata[i].sigma[j] = sigma + VECLEN * i + j;
            }
        }
        else
        {
            for (j = 0; j < VECLEN; j++)
            {
                tdata[i].sigma[j] = 0;
            }
        }
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
        int j;
        uint32_t D = tdata[i].work->D;

        mpz_clear(tdata[i].factor);
        ecm_work_free(tdata[i].work);
        ecm_pt_free(tdata[i].P);
        monty_free(tdata[i].mdata);
        free(tdata[i].mdata);
        free(tdata[i].work);
        free(tdata[i].P);
        free(tdata[i].sigma);

        if (i == 0)
        {
            int k;
            free(tdata[i].pairmap_u);
            free(tdata[i].pairmap_v);

            for (j = 0, k = 0; k < 2 * D; k++)
            {
                if (spGCD(k, 2 * D) == 1)
                {
                    j++;
                }
            }

            for (k = 0; k < j; k++)
            {
                clearQueue(tdata[i].Q[k]);
                free(tdata[i].Q[k]);
            }
            free(tdata[i].Q);
            free(tdata[i].Qrmap);
            free(tdata[i].Qmap);
        }
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
    uint32_t D = tdata->work->D = 2310;

    int i, j;

    if (STAGE1_MAX <= 4096)
    {
        D = tdata->work->D = 1155;
    }

    if (STAGE1_MAX <= 2048)
    {
        D = tdata->work->D = 385;
    }

    if (STAGE1_MAX <= 512)
    {
        D = tdata->work->D = 210;
    }

    if (STAGE1_MAX <= 256)
    {
        D = tdata->work->D = 120;
    }

    if (STAGE1_MAX <= 128)
    {
        D = tdata->work->D = 60;
    }

    if (STAGE1_MAX <= 60)
    {
        D = tdata->work->D = 30;
    }

    for (j = 0, i = 0; i < 2 * D; i++)
    {
        if (spGCD(i, 2 * D) == 1)
        {
            j++;
        }
    }

    tdata->work->R = j + 3;  

	// decide on the stage 2 parameters.  Larger U means
	// more memory and more setup overhead, but more prime pairs.
	// Smaller U means the opposite.  find a good balance.
    // these pairing ratios are estimates based on Montgomery's
    // Pair algorithm Table w assuming w = 1155, for various umax.
    //static double pairing[8] = { 0.7,  0.6446, 0.6043, 0.5794, 0.5535, 0.5401, 0.5266, 0.5015};
    // these ratios are based on observations with larger (more realistic) B1/B2, 
    // specifically, B2=1e9
    static double pairing[8] = { 0.8,  0.72, 0.67, 0.63, 0.59, 0.57, 0.55, 0.54 };
    int U[8] = { 1, 2, 3, 4, 6, 8, 12, 16};
	double adds[8];
    int numadds;
    int numinv;
    double addcost, invcost;
	double best = 99999999999.;
	int bestU = 4;

	
#ifdef TARGET_KNL
    // for smaller B1, lower U can be better because initialization time
    // is more significant.  At B1=3M or above the max U is probably best.
    // If memory is an issue beyond the timings, then U will have to
    // be smaller.  (e.g., if we do a -maxmem option).
    for (i = 1; i < 6; i++)
#else
	for (i = 1; i < 8; i++)
#endif
	{
        double paircost;

        // setup cost of Pb.
        // should maybe take memory into account too, since this array can
        // get pretty big (especially with multiple threads)?
        adds[i] = (double)D * (double)U[i];;

        // runtime cost of stage 2
		numadds = (STAGE2_MAX - STAGE1_MAX) / (D);

        // total adds cost
		addcost = 6.0 * ((double)numadds + adds[i]);

        // inversion cost - we invert in batches of 2U at a time, taking 3*U muls.
        // estimating the cost of the inversion as equal to an add, but we do the 
        // 8 vector elements one at a time.
        numinv = ((double)numadds / (double)U[i] / 2.0) + 2;
        invcost = (double)numinv * (VECLEN * 6.0) + (double)numinv * 3.0;

        //// estimate number of prime pairs times 1 (1 mul per pair with batch inversion)
		//paircost = ((double)STAGE2_MAX / log((double)STAGE2_MAX) -
		//	(double)STAGE1_MAX / log((double)STAGE1_MAX)) * pairing[i] * 1.0;
        //
		//printf("estimating %u primes to pair\n", (uint32_t)((double)STAGE2_MAX / log((double)STAGE2_MAX) -
		//	(double)STAGE1_MAX / log((double)STAGE1_MAX)));
		//printf("%d stg2 adds + %d setup adds + %d inversions\n", numadds, (int)adds[i], numinv);
		//printf("addcost = %f\n", addcost);
		//printf("paircost = %f\n", paircost);
        //printf("invcost = %f\n", invcost);
		//printf("totalcost = %f\n", addcost + paircost + invcost);

		if ((addcost + paircost + invcost) < best)
		{
			best = addcost + paircost + invcost;
			bestU = U[i];
		}
	}

    tdata->work->U = bestU;
	tdata->work->L = tdata->work->U * 2;

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
	tdata->mdata->nbits = mdata->nbits;
    tdata->mdata->isMersenne = mdata->isMersenne;
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
