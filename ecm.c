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

/* Elliptic Curve Method: toplevel and stage 1 routines.

Copyright 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
2012, 2016 Paul Zimmermann, Alexander Kruppa, Cyril Bouvier, David Cleaver.

This file is part of the ECM Library.

The ECM Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The ECM Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the ECM Library; see the file COPYING.LIB.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include "avx_ecm.h"
#include "omp.h"
#include "eratosthenes/soe.h"

//#define DEBUG 1
#define DO_STAGE2_INV

// local functions
void vec_add(monty *mdata, ecm_work *work, ecm_pt *Pin, ecm_pt *Pout);
void vec_duplicate(monty *mdata, ecm_work *work, bignum *insum, bignum *indiff, ecm_pt *P);
void next_pt_vec(monty *mdata, ecm_work *work, ecm_pt *P, uint64_t c);
void euclid(monty *mdata, ecm_work *work, ecm_pt *P, uint64_t c);
void prac(monty *mdata, ecm_work *work, ecm_pt *P, uint64_t c);
int check_factor(mpz_t Z, mpz_t n, mpz_t f);
void build_one_curve(thread_data_t *tdata, mpz_t X, mpz_t Z, mpz_t A, uint64_t sigma);
void ecm_stage1(monty *mdata, ecm_work *work, ecm_pt *P, base_t b1, base_t *primes, int verbose);
void ecm_stage2_init_ref(ecm_pt* P, monty* mdata, ecm_work* work, int verbose);
void ecm_stage2_pair_ref(ecm_pt* P, monty* mdata, ecm_work* work, int verbose);
int ecm_stage2_init(ecm_pt* P, monty* mdata, ecm_work* work, int verbose);
void ecm_stage2_pair(uint32_t pairmap_steps, uint32_t* pm_v, uint32_t* pm_u,
    ecm_pt* P, monty* mdata, ecm_work* work, int verbose);
uint32_t pair(uint32_t *pm_v, uint32_t *pm_u, 
    ecm_work* work, uint64_t* primes, uint64_t B1, uint64_t B2, int verbose);


#include "threadpool.h"

void ecm_sync(void *vptr);
void ecm_dispatch(void *vptr);
void ecm_build_curve_work_fcn(void *vptr);
void ecm_stage1_work_fcn(void *vptr);
void ecm_stage2_work_fcn(void *vptr);


void ecm_sync(void *vptr)
{
    tpool_t *tpdata = (tpool_t *)vptr;
    thread_data_t *udata = (thread_data_t *)tpdata->user_data;
    uint32_t tid = tpdata->tindex;

    if (udata[tid].ecm_phase == 0)
    {
        udata[tid].phase_done = 1;
    }
    else if (udata->ecm_phase == 1)
    {
        udata[tid].phase_done = 1;
    }
    else if (udata->ecm_phase == 2)
    {
        udata[tid].phase_done = 1;
    }
    else if (udata->ecm_phase == 3)
    {
        udata[tid].phase_done = 1;
    }
    
    return;
}

void ecm_dispatch(void *vptr)
{
    tpool_t *tpdata = (tpool_t *)vptr;
    thread_data_t *udata = (thread_data_t *)tpdata->user_data;
    uint32_t tid = tpdata->tindex;
    
    if (udata[tid].ecm_phase == 0)
    {        
        if (udata[tid].phase_done == 0)
        {
            tpdata->work_fcn_id = 0;
        }
        else
        {
            tpdata->work_fcn_id = tpdata->num_work_fcn;
        }
    }
    else if (udata[tid].ecm_phase == 1)
    {
        if (udata[tid].phase_done == 0)
        {
            tpdata->work_fcn_id = 1;
        }
        else
        {
            tpdata->work_fcn_id = tpdata->num_work_fcn;
        }
    }
    else if (udata[tid].ecm_phase == 2)
    {
        if (udata[tid].phase_done == 0)
        {
            tpdata->work_fcn_id = 2;
        }
        else
        {
            tpdata->work_fcn_id = tpdata->num_work_fcn;
        }
    }
    else if (udata[tid].ecm_phase == 3)
    {
        if (udata[tid].phase_done == 0)
        {
            tpdata->work_fcn_id = 3;
        }
        else
        {
            tpdata->work_fcn_id = tpdata->num_work_fcn;
        }
    }

    return;
}

void ecm_stage1_work_fcn(void *vptr)
{
    tpool_t *tpdata = (tpool_t *)vptr;
    thread_data_t *udata = (thread_data_t *)tpdata->user_data;
    uint32_t tid = tpdata->tindex;

    ecm_stage1(udata[tid].mdata, udata[tid].work, udata[tid].P, udata[tid].b1, NULL, tid == 0);

    return;
}

void ecm_stage2_init_work_fcn(void *vptr)
{
    tpool_t *tpdata = (tpool_t *)vptr;
    thread_data_t *udata = (thread_data_t *)tpdata->user_data;
    uint32_t tid = tpdata->tindex;

#ifdef USE_STAGE2_REF
    ecm_stage2_init_ref(udata[tid].P, udata[tid].mdata, udata[tid].work, tid == 0);
#else
    ecm_stage2_init(udata[tid].P, udata[tid].mdata, udata[tid].work, tid == 0);
#endif
    

    return;
}

void ecm_stage2_work_fcn(void *vptr)
{
    tpool_t *tpdata = (tpool_t *)vptr;
    thread_data_t *udata = (thread_data_t *)tpdata->user_data;
    uint32_t tid = tpdata->tindex;

#ifdef USE_STAGE2_REF
    ecm_stage2_pair_ref(udata[tid].P, udata[tid].mdata, udata[tid].work, tid == 0);
#else
    ecm_stage2_pair(udata[tid].pairmap_steps, udata[tid].pairmap_v, udata[tid].pairmap_u,
        udata[tid].P, udata[tid].mdata, udata[tid].work, tid == 0);
#endif

    return;
}

void ecm_build_curve_work_fcn(void *vptr)
{
    tpool_t *tpdata = (tpool_t *)vptr;
    thread_data_t *tdata = (thread_data_t *)tpdata->user_data;
    uint32_t tid = tpdata->tindex;
    mpz_t X, Z, A; // , t, n, one, r;

    int i;

    mpz_init(X);
    mpz_init(Z);
    mpz_init(A);

	for (i = 0; i < VECLEN; i++)
    {
        build_one_curve(&tdata[tid], X, Z, A, tdata[tid].sigma[i]);

        insert_mpz_to_vec(tdata[tid].P->X, X, i);
        insert_mpz_to_vec(tdata[tid].P->Z, Z, i);
        insert_mpz_to_vec(tdata[tid].work->s, A, i);
        tdata[tid].sigma[i] = tdata[tid].work->sigma;
    }

    //print_vechexbignum(tdata[tid].P->X, "POINT x: ");
    //print_vechexbignum(tdata[tid].P->Z, "POINT z: ");
    //print_vechexbignum(tdata[tid].work->s, "POINT a: ");

    tdata[tid].work->diff1->size = tdata[tid].work->n->size;
    tdata[tid].work->diff2->size = tdata[tid].work->n->size;
    tdata[tid].work->sum1->size  = tdata[tid].work->n->size;
    tdata[tid].work->sum2->size  = tdata[tid].work->n->size;
    tdata[tid].work->pt1.X->size = tdata[tid].work->n->size;
    tdata[tid].work->pt1.Z->size = tdata[tid].work->n->size;
    tdata[tid].work->pt2.X->size = tdata[tid].work->n->size;
    tdata[tid].work->pt2.Z->size = tdata[tid].work->n->size;
    tdata[tid].work->tt1->size   = tdata[tid].work->n->size;
    tdata[tid].work->tt2->size   = tdata[tid].work->n->size;
    tdata[tid].work->tt3->size   = tdata[tid].work->n->size;
    tdata[tid].work->tt4->size   = tdata[tid].work->n->size;

    mpz_clear(X);
    mpz_clear(Z);
    mpz_clear(A);

    return;
}

void ecm_work_init(ecm_work *work)
{
    int i, j, m;
	uint32_t U = work->U;
	uint32_t L = work->L;
	uint32_t D = work->D;
	uint32_t R = work->R;

	work->stg1Add = 0;
	work->stg1Doub = 0;

	work->diff1 = vecInit();
	work->diff2 = vecInit();
	work->sum1 = vecInit();
	work->sum2 = vecInit();
	work->tt1 = vecInit();
	work->tt2 = vecInit();
	work->tt3 = vecInit();
	work->tt4 = vecInit();
	work->tt5 = vecInit();
	work->s = vecInit();
	work->n = vecInit();

	work->Paprod = (bignum **)malloc((2 * L) * sizeof(bignum *));
    work->Pa_inv = (bignum * *)malloc((2 * L) * sizeof(bignum*));
	work->Pa = (ecm_pt *)malloc((2 * L) * sizeof(ecm_pt));
	for (j = 0; j < (2 * L); j++)
	{
		work->Paprod[j] = vecInit();
        work->Pa_inv[j] = vecInit();
		ecm_pt_init(&work->Pa[j]);
	}

	work->Pad = (ecm_pt *)malloc(sizeof(ecm_pt));
	ecm_pt_init(work->Pad);

    work->Pdnorm = (ecm_pt*)malloc(sizeof(ecm_pt));
    ecm_pt_init(work->Pdnorm);

	// build an array to hold values of f(b).
	// will need to be of size U*R, allowed values will be
	// up to a multiple of U of the residues mod D
	work->Pb = (ecm_pt *)malloc(U * (R + 1) * sizeof(ecm_pt));
	work->Pbprod = (bignum **)malloc(U * (R + 1) * sizeof(bignum *));
	for (j = 0; j < U * (R + 1); j++)
	{
		ecm_pt_init(&work->Pb[j]);
		work->Pbprod[j] = vecInit();
	}

	//work->marks = (uint8_t *)malloc(U * D * sizeof(uint8_t));
	//work->nmarks = (uint8_t *)malloc(U * D * sizeof(uint8_t));
	work->map = (uint32_t *)calloc(U * (D + 1) + 3, sizeof(uint32_t));

	work->map[0] = 0;
	work->map[1] = 1;
	work->map[2] = 2;
	m = 3;
	for (i = 0; i < U; i++)
	{
		if (i == 0)
			j = 3;
		else
			j = 1;

		for (; j < D; j++)
		{
			if (spGCD(j, D) == 1)
			{
				work->map[i*D + j] = m++;
			}
			else
			{
				work->map[i*D + j] = 0;
			}
		}

		if (i == 0)
			work->map[i*D + j] = m++;

	}

    work->Qmap = (uint32_t*)malloc(2 * D * sizeof(uint32_t));
#ifndef USE_STAGE2_REF
	work->Qrmap = (uint32_t *)malloc(2 * D * sizeof(uint32_t));

	for (j = 0, i = 0; i < 2 * D; i++)
	{
		if (spGCD(i, 2 * D) == 1)
		{
			work->Qmap[i] = j;
			work->Qrmap[j++] = i;
		}
		else
		{
			work->Qmap[i] = (uint32_t)-1;
		}
	}

	for (i = j; i < 2 * D; i++)
	{
		work->Qrmap[i] = (uint32_t)-1;
	}

	work->Q = (Queue_t **)malloc(j * sizeof(Queue_t *));
	for (i = 0; i < j; i++)
	{
		work->Q[i] = newQueue(D);
	}
#endif

	ecm_pt_init(&work->pt1);
	ecm_pt_init(&work->pt2);
	ecm_pt_init(&work->pt3);
	ecm_pt_init(&work->pt4);
	ecm_pt_init(&work->pt5);

	work->stg2acc = vecInit();

	return;
}

void ecm_work_free(ecm_work *work)
{
	int i, j;
	uint32_t U = work->U;
	uint32_t L = work->L;
	uint32_t D = work->D;
	uint32_t R = work->R;

	vecFree(work->diff1);
	vecFree(work->diff2);
	vecFree(work->sum1);
	vecFree(work->sum2);
	vecFree(work->tt1);
	vecFree(work->tt2);
	vecFree(work->tt3);
	vecFree(work->tt4);
	vecFree(work->tt5);
	vecFree(work->s);
	vecFree(work->n);
	ecm_pt_free(&work->pt1);
	ecm_pt_free(&work->pt2);
	ecm_pt_free(&work->pt3);
	ecm_pt_free(&work->pt4);
	ecm_pt_free(&work->pt5);

	for (i = 0; i < (2 * L); i++)
	{
		ecm_pt_free(&work->Pa[i]);
		vecFree(work->Paprod[i]);
        vecFree(work->Pa_inv[i]);
	}

	ecm_pt_free(work->Pad);
    ecm_pt_free(work->Pdnorm);
	free(work->Pa);
	free(work->Pad);
    free(work->Pdnorm);
	free(work->marks);
	free(work->nmarks);
	free(work->map);
	
	for (i = 0; i < U * (R + 1); i++)
	{
		ecm_pt_free(&work->Pb[i]);
		vecFree(work->Pbprod[i]);
	}

#ifndef USE_STAGE2_REF
    for (j = 0, i = 0; i < 2 * D; i++)
    {
        if (spGCD(i, 2 * D) == 1)
        {
            j++;
        }
    }

	for (i = 0; i < j; i++) //2 * (R - 3); i++)
	{
		clearQueue(work->Q[i]);
		free(work->Q[i]);
	}
	free(work->Q);
	free(work->Qrmap);
#endif
    free(work->Qmap);

	free(work->Pbprod);
	free(work->Paprod);
	free(work->Pb);
	vecFree(work->stg2acc);

	return;
}

void ecm_pt_init(ecm_pt *pt)
{
	pt->X = vecInit();
	pt->Z = vecInit();
}

void ecm_pt_free(ecm_pt *pt)
{
	vecFree(pt->X);
	vecFree(pt->Z);
}

void vec_add(monty *mdata, ecm_work *work, ecm_pt *Pin, ecm_pt *Pout)
{
    // compute:
    //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
    //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
    // where:
    //x- = original x
    //z- = original z
    // given the sums and differences of the original points (stored in work structure).

    vecmulmod_ptr(work->diff1, work->sum2, work->tt1, work->n, work->tt4, mdata);	//U
    vecmulmod_ptr(work->sum1, work->diff2, work->tt2, work->n, work->tt4, mdata);	//V

    vecaddsubmod_ptr(work->tt1, work->tt2, work->tt3, work->tt4, mdata);
    vecsqrmod_ptr(work->tt3, work->tt1, work->n, work->tt5, mdata);					//(U + V)^2
    vecsqrmod_ptr(work->tt4, work->tt2, work->n, work->tt5, mdata);					//(U - V)^2

    // choosing the initial point Pz0 = 1 means that z_p-q = 1 and this mul isn't necessary...
    // but that involves a different way to initialize curves, so for now
    // we can't assume Z=1
	if (Pin->X == Pout->X)
	{
		base_t *swap;
		vecmulmod_ptr(work->tt1, Pin->Z, Pout->Z, work->n, work->tt4, mdata);		//Z * (U + V)^2
		vecmulmod_ptr(work->tt2, Pin->X, Pout->X, work->n, work->tt4, mdata);		//x * (U - V)^2
		swap = Pout->Z->data;
		Pout->Z->data = Pout->X->data;
		Pout->X->data = swap;
	}
	else
	{
		vecmulmod_ptr(work->tt1, Pin->Z, Pout->X, work->n, work->tt4, mdata);		//Z * (U + V)^2
		vecmulmod_ptr(work->tt2, Pin->X, Pout->Z, work->n, work->tt4, mdata);		//x * (U - V)^2
	}
	work->stg1Add++;
    return;
}

void vec_duplicate(monty *mdata, ecm_work *work, bignum *insum, bignum *indiff, ecm_pt *P)
{
    vecsqrmod_ptr(indiff, work->tt1, work->n, work->tt4, mdata);			    // V=(x1 - z1)^2
    vecsqrmod_ptr(insum, work->tt2, work->n, work->tt4, mdata);			        // U=(x1 + z1)^2
    vecmulmod_ptr(work->tt1, work->tt2, P->X, work->n, work->tt4, mdata);       // x=U*V

    vecsubmod_ptr(work->tt2, work->tt1, work->tt3, mdata);	                // w = U-V
    vecmulmod_ptr(work->tt3, work->s, work->tt2, work->n, work->tt4, mdata);    // t = (A+2)/4 * w
    vecaddmod_ptr(work->tt2, work->tt1, work->tt2, mdata);                    // t = t + V
    vecmulmod_ptr(work->tt2, work->tt3, P->Z, work->n, work->tt4, mdata);       // Z = t*w
	work->stg1Doub++;
    return;
}

#define ADD 6.0
#define DUP 5.0

static double
lucas_cost(uint64_t n, double v)
{
	uint64_t d, e, r;
	double c; /* cost */

	d = n;
	r = (uint64_t)((double)d * v + 0.5);
	if (r >= n)
		return (ADD * (double)n);
	d = n - r;
	e = 2 * r - n;
	c = DUP + ADD; /* initial duplicate and final addition */
	while (d != e)
	{
		if (d < e)
		{
			r = d;
			d = e;
			e = r;
		}
		if (d - e <= e / 4 && ((d + e) % 3) == 0)
		{ /* condition 1 */
			d = (2 * d - e) / 3;
			e = (e - d) / 2;
			c += 3.0 * ADD; /* 3 additions */
		}
		else if (d - e <= e / 4 && (d - e) % 6 == 0)
		{ /* condition 2 */
			d = (d - e) / 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
		else if ((d + 3) / 4 <= e)
		{ /* condition 3 */
			d -= e;
			c += ADD; /* one addition */
		}
		else if ((d + e) % 2 == 0)
		{ /* condition 4 */
			d = (d - e) / 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
		/* now d+e is odd */
		else if (d % 2 == 0)
		{ /* condition 5 */
			d /= 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
		/* now d is odd and e is even */
		else if (d % 3 == 0)
		{ /* condition 6 */
			d = d / 3 - e;
			c += 3.0 * ADD + DUP; /* three additions, one duplicate */
		}
		else if ((d + e) % 3 == 0)
		{ /* condition 7 */
			d = (d - 2 * e) / 3;
			c += 3.0 * ADD + DUP; /* three additions, one duplicate */
		}
		else if ((d - e) % 3 == 0)
		{ /* condition 8 */
			d = (d - e) / 3;
			c += 3.0 * ADD + DUP; /* three additions, one duplicate */
		}
		else /* necessarily e is even: catches all cases */
		{ /* condition 9 */
			e /= 2;
			c += ADD + DUP; /* one addition, one duplicate */
		}
	}

	return c;
}

void prac(monty *mdata, ecm_work *work, ecm_pt *P, uint64_t c)
{
	uint64_t d, e, r;
	double cmin, cost;
	int i;
	bignum *s1, *s2, *d1, *d2;
	base_t *sw_x, *sw_z;

#define NV 10  
	/* 1/val[0] = the golden ratio (1+sqrt(5))/2, and 1/val[i] for i>0
	   is the real number whose continued fraction expansion is all 1s
	   except for a 2 in i+1-st place */
	static double val[NV] =
	{ 0.61803398874989485, 0.72360679774997897, 0.58017872829546410,
	  0.63283980608870629, 0.61242994950949500, 0.62018198080741576,
	  0.61721461653440386, 0.61834711965622806, 0.61791440652881789,
	  0.61807966846989581 };

	/* chooses the best value of v */
	for (i = d = 0, cmin = ADD * (double)c; d < NV; d++)
	{
		cost = lucas_cost(c, val[d]);
		if (cost < cmin)
		{
			cmin = cost;
			i = d;
		}
	}

	d = c;
	r = (uint64_t)((double)d * val[i] + 0.5);

	s1 = work->sum1;
	s2 = work->sum2;
	d1 = work->diff1;
	d2 = work->diff2;

	/* first iteration always begins by Condition 3, then a swap */
	d = c - r;
	e = 2 * r - c;

	// mpres_set(xB, xA, n);
	// mpres_set(zB, zA, n); /* B=A */
	// mpres_set(xC, xA, n);
	// mpres_set(zC, zA, n); /* C=A */
	// duplicate(xA, zA, xA, zA, n, b, u, v, w); /* A = 2*A */

	// the first one is always a doubling
	// point1 is [1]P
	vecCopy(P->X, work->pt1.X);
	vecCopy(P->Z, work->pt1.Z);
	vecCopy(P->X, work->pt2.X);
	vecCopy(P->Z, work->pt2.Z);
	vecCopy(P->X, work->pt3.X);
	vecCopy(P->Z, work->pt3.Z);
	vecsubmod_ptr(work->pt1.X, work->pt1.Z, d1, mdata);
	vecaddmod_ptr(work->pt1.X, work->pt1.Z, s1, mdata);

	// point2 is [2]P
	vec_duplicate(mdata, work, s1, d1, &work->pt1);

	while (d != e)
	{
		if (d < e)
		{
			r = d;
			d = e;
			e = r;
			//mpres_swap(xA, xB, n);
			//mpres_swap(zA, zB, n);
			sw_x = work->pt1.X->data;
			sw_z = work->pt1.Z->data;
			work->pt1.X->data = work->pt2.X->data;
			work->pt1.Z->data = work->pt2.Z->data;
			work->pt2.X->data = sw_x;
			work->pt2.Z->data = sw_z;
		}
		/* do the first line of Table 4 whose condition qualifies */
		if (d - e <= e / 4 && ((d + e) % 3) == 0)
		{ /* condition 1 */
			d = (2 * d - e) / 3;
			e = (e - d) / 2;

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt3, &work->pt4); // T = A + B (C)

            vecaddsubmod_ptr(work->pt4.X, work->pt4.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt2, &work->pt5); // T2 = T + A (B)

            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt4.X, work->pt4.Z, s2, d2, mdata);

			vec_add(mdata, work, &work->pt1, &work->pt2); // B = B + T (A)

			//add3(xT, zT, xA, zA, xB, zB, xC, zC, n, u, v, w); /* T = f(A,B,C) */
			//add3(xT2, zT2, xT, zT, xA, zA, xB, zB, n, u, v, w); /* T2 = f(T,A,B) */
			//add3(xB, zB, xB, zB, xT, zT, xA, zA, n, u, v, w); /* B = f(B,T,A) */
			//mpres_swap(xA, xT2, n);
			//mpres_swap(zA, zT2, n); /* swap A and T2 */

			sw_x = work->pt1.X->data;
			sw_z = work->pt1.Z->data;
			work->pt1.X->data = work->pt5.X->data;
			work->pt1.Z->data = work->pt5.Z->data;
			work->pt5.X->data = sw_x;
			work->pt5.Z->data = sw_z;

		}
		else if (d - e <= e / 4 && (d - e) % 6 == 0)
		{ /* condition 2 */
			d = (d - e) / 2;
			
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s2, d2, mdata);

			vec_add(mdata, work, &work->pt3, &work->pt2);		// B = A + B (C)
			vec_duplicate(mdata, work, s1, d1, &work->pt1);		// A = 2A

			//add3(xB, zB, xA, zA, xB, zB, xC, zC, n, u, v, w); /* B = f(A,B,C) */
			//duplicate(xA, zA, xA, zA, n, b, u, v, w); /* A = 2*A */

		}
		else if ((d + 3) / 4 <= e)
		{ /* condition 3 */
			d -= e;
			
            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

			vec_add(mdata, work, &work->pt3, &work->pt4);		// T = B + A (C)
			//add3(xT, zT, xB, zB, xA, zA, xC, zC, n, u, v, w); /* T = f(B,A,C) */
			
			/* circular permutation (B,T,C) */
			//tmp = xB;
			//xB = xT;
			//xT = xC;
			//xC = tmp;
			//tmp = zB;
			//zB = zT;
			//zT = zC;
			//zC = tmp;

			sw_x = work->pt2.X->data;
			sw_z = work->pt2.Z->data;
			work->pt2.X->data = work->pt4.X->data;
			work->pt2.Z->data = work->pt4.Z->data;
			work->pt4.X->data = work->pt3.X->data;
			work->pt4.Z->data = work->pt3.Z->data;
			work->pt3.X->data = sw_x;
			work->pt3.Z->data = sw_z;

		}
		else if ((d + e) % 2 == 0)
		{ /* condition 4 */
			d = (d - e) / 2;

            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

			vec_add(mdata, work, &work->pt3, &work->pt2);		// B = B + A (C)
			vec_duplicate(mdata, work, s2, d2, &work->pt1);		// A = 2A
			
			//add3(xB, zB, xB, zB, xA, zA, xC, zC, n, u, v, w); /* B = f(B,A,C) */
			//duplicate(xA, zA, xA, zA, n, b, u, v, w); /* A = 2*A */
		}
		/* now d+e is odd */
		else if (d % 2 == 0)
		{ /* condition 5 */
			d /= 2;
			
            vecaddsubmod_ptr(work->pt3.X, work->pt3.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

			vec_add(mdata, work, &work->pt2, &work->pt3);		// C = C + A (B)
			vec_duplicate(mdata, work, s2, d2, &work->pt1);		// A = 2A

			//add3(xC, zC, xC, zC, xA, zA, xB, zB, n, u, v, w); /* C = f(C,A,B) */
			//duplicate(xA, zA, xA, zA, n, b, u, v, w); /* A = 2*A */
		}
		/* now d is odd, e is even */
		else if (d % 3 == 0)
		{ /* condition 6 */
			d = d / 3 - e;

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);

            vec_duplicate(mdata, work, s1, d1, &work->pt4);		// T = 2A

            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt3, &work->pt5);		// T2 = A + B (C)

            vecaddsubmod_ptr(work->pt4.X, work->pt4.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt1, &work->pt1);		// A = T + A (A)

            vecaddsubmod_ptr(work->pt5.X, work->pt5.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt3, &work->pt4);		// T = T + T2 (C)

			//duplicate(xT, zT, xA, zA, n, b, u, v, w); /* T = 2*A */
			//add3(xT2, zT2, xA, zA, xB, zB, xC, zC, n, u, v, w); /* T2 = f(A,B,C) */
			//add3(xA, zA, xT, zT, xA, zA, xA, zA, n, u, v, w); /* A = f(T,A,A) */
			//add3(xT, zT, xT, zT, xT2, zT2, xC, zC, n, u, v, w); /* T = f(T,T2,C) */

			/* circular permutation (C,B,T) */
			//tmp = xC;
			//xC = xB;
			//xB = xT;
			//xT = tmp;
			//tmp = zC;
			//zC = zB;
			//zB = zT;
			//zT = tmp;

			sw_x = work->pt3.X->data;
			sw_z = work->pt3.Z->data;
			work->pt3.X->data = work->pt2.X->data;
			work->pt3.Z->data = work->pt2.Z->data;
			work->pt2.X->data = work->pt4.X->data;
			work->pt2.Z->data = work->pt4.Z->data;
			work->pt4.X->data = sw_x;
			work->pt4.Z->data = sw_z;

		}
		else if ((d + e) % 3 == 0)
		{ /* condition 7 */
			d = (d - 2 * e) / 3;

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt3, &work->pt4);		// T = A + B (C)

            vecaddsubmod_ptr(work->pt4.X, work->pt4.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt2, &work->pt2);		// B = T + A (B)

            vec_duplicate(mdata, work, s2, d2, &work->pt4);		// T = 2A

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt4.X, work->pt4.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt1, &work->pt1);		// A = A + T (A) = 3A

			//add3(xT, zT, xA, zA, xB, zB, xC, zC, n, u, v, w); /* T = f(A,B,C) */
			//add3(xB, zB, xT, zT, xA, zA, xB, zB, n, u, v, w); /* B = f(T,A,B) */
			//duplicate(xT, zT, xA, zA, n, b, u, v, w);
			//add3(xA, zA, xA, zA, xT, zT, xA, zA, n, u, v, w); /* A = 3*A */
		}
		else if ((d - e) % 3 == 0)
		{ /* condition 8 */
			d = (d - e) / 3;

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt3, &work->pt4);		// T = A + B (C)

            vecaddsubmod_ptr(work->pt3.X, work->pt3.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt2, &work->pt3);		// C = C + A (B)

            //add3(xT, zT, xA, zA, xB, zB, xC, zC, n, u, v, w); /* T = f(A,B,C) */
            //add3(xC, zC, xC, zC, xA, zA, xB, zB, n, u, v, w); /* C = f(A,C,B) */
            //mpres_swap(xB, xT, n);
            //mpres_swap(zB, zT, n); /* swap B and T */
            sw_x = work->pt2.X->data;
            sw_z = work->pt2.Z->data;
            work->pt2.X->data = work->pt4.X->data;
            work->pt2.Z->data = work->pt4.Z->data;
            work->pt4.X->data = sw_x;
            work->pt4.Z->data = sw_z;

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s2, d2, mdata);

            vec_duplicate(mdata, work, s2, d2, &work->pt4);		// T = 2A

            vecaddsubmod_ptr(work->pt1.X, work->pt1.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt4.X, work->pt4.Z, s2, d2, mdata);

            vec_add(mdata, work, &work->pt1, &work->pt1);		// A = A + T (A) = 3A

			//duplicate(xT, zT, xA, zA, n, b, u, v, w);
			//add3(xA, zA, xA, zA, xT, zT, xA, zA, n, u, v, w); /* A = 3*A */
		}
		else /* necessarily e is even here */
		{ /* condition 9 */
			e /= 2;

            vecaddsubmod_ptr(work->pt3.X, work->pt3.Z, s1, d1, mdata);
            vecaddsubmod_ptr(work->pt2.X, work->pt2.Z, s2, d2, mdata);

			vec_add(mdata, work, &work->pt1, &work->pt3);		// C = C + B (A)
			vec_duplicate(mdata, work, s2, d2, &work->pt2);		// B = 2B

			//add3(xC, zC, xC, zC, xB, zB, xA, zA, n, u, v, w); /* C = f(C,B,A) */
			//duplicate(xB, zB, xB, zB, n, b, u, v, w); /* B = 2*B */
		}
	}

	vecsubmod_ptr(work->pt1.X, work->pt1.Z, d1, mdata);
	vecaddmod_ptr(work->pt1.X, work->pt1.Z, s1, mdata);
	vecsubmod_ptr(work->pt2.X, work->pt2.Z, d2, mdata);
	vecaddmod_ptr(work->pt2.X, work->pt2.Z, s2, mdata);

	vec_add(mdata, work, &work->pt3, P);		// A = A + B (C)

	//add3(xA, zA, xA, zA, xB, zB, xC, zC, n, u, v, w);

	if (d != 1)
	{
		printf("problem: d != 1\n");
	}

	return;

}

void next_pt_vec(monty *mdata, ecm_work *work, ecm_pt *P, uint64_t c)
{
	uint64_t mask;
	bignum *x1, *z1, *x2, *z2, *s1, *s2, *d1, *d2;
	uint64_t e, d;

	x1 = work->pt1.X;
	z1 = work->pt1.Z;
	x2 = work->pt2.X;
	z2 = work->pt2.Z;
	s1 = work->sum1;
	s2 = work->sum2;
	d1 = work->diff1;
	d2 = work->diff2;

	if (c == 1)
	{
		return;
	}

	vecCopy(P->X, x1);
	vecCopy(P->Z, z1);
	vecsubmod_ptr(P->X, P->Z, d1, mdata);
	vecaddmod_ptr(P->X, P->Z, s1, mdata);
	vec_duplicate(mdata, work, s1, d1, &work->pt2);

	//mulcnt[tid] += 3;
	//sqrcnt[tid] += 2;

	if (c == 2)
	{
		vecCopy(x2, P->X);
		vecCopy(z2, P->Z);
		return;
	}

	// find the first '1' bit then skip it
#ifdef __INTEL_COMPILER
	mask = 1ULL << (64 - _lzcnt_u64((uint64_t)c) - 2);
#else
	mask = 1ULL << (64 - __builtin_clzll((uint64_t)c) - 2);
#endif

	//goal is to compute x_c, z_c using montgomery's addition
	//and multiplication formula's for x and z
	//the procedure will be similar to p+1 lucas chain formula's
	//but this time we are simultaneously incrementing both x and z
	//rather than just Vn.  In each bit of the binary expansion of
	//c, we need just one addition and one duplication for both x and z.
	//compute loop
	//for each bit of M to the right of the most significant bit
	d = 1;
	e = 2;
	while (mask > 0)
	{
		vecaddsubmod_ptr(x2, z2, s2, d2, mdata);
		vecaddsubmod_ptr(x1, z1, s1, d1, mdata);

		//if the bit is 1
		if (c & mask)
		{
			//add x1,z1, duplicate x2,z2
			vec_add(mdata, work, P, &work->pt1);
			vec_duplicate(mdata, work, s2, d2, &work->pt2);
			d = d + e;
			e *= 2;
		}
		else
		{
			//add x2,z2, duplicate x1,z1
			vec_add(mdata, work, P, &work->pt2);
			vec_duplicate(mdata, work, s1, d1, &work->pt1);
			e = e + d;
			d *= 2;
		}

		//mulcnt[tid] += 7;
		//sqrcnt[tid] += 4;

		mask >>= 1;
	}
	if (d != c)
	{
		printf("expected %lu, ladder returned %lu\n", c, d);
		exit(1);
	}
	vecCopy(x1, P->X);
	vecCopy(z1, P->Z);

	return;
}

void array_mul(uint64_t *primes, int num_p, mpz_t piprimes)
{
	mpz_t *p;
	int i;
	int alloc;
	uint64_t stg1 = (uint64_t)STAGE1_MAX;

	if (num_p & 1)
	{
		p = (mpz_t *)malloc((num_p + 1) * sizeof(mpz_t));
		alloc = num_p + 1;
	}
	else
	{
		p = (mpz_t *)malloc((num_p + 0) * sizeof(mpz_t));
		alloc = num_p;
	}

	for (i = 0; i < num_p; i++)
	{
		uint64_t c = 1;
		uint64_t q;

		q = primes[i];
		do {
			c *= q;
		} while ((c * q) < stg1);

		mpz_init(p[i]);
		mpz_set_ui(p[i], q);
	}

	if (num_p & 1)
	{
		mpz_init(p[i]);
		mpz_set_ui(p[i], 1);
		num_p++;
	}

	while (num_p != 1)
	{
		for (i = 0; i < num_p / 2; i++)
		{
			mpz_mul(p[i], p[i], p[i + num_p / 2]);
		}

		num_p /= 2;

		if ((num_p > 1) && (num_p & 1))
		{
			mpz_set_ui(p[num_p], 1);
			num_p++;
		}
	}

	mpz_set(piprimes, p[0]);

	printf("piprimes has %lu bits\n\n", mpz_sizeinbase(piprimes, 2));

	for (i = 0; i < alloc; i++)
	{
		mpz_clear(p[i]);
	}
	free(p);

	return;
}

void vececm(thread_data_t *tdata)
{
	//attempt to factor n with the elliptic curve method
	//following brent and montgomery's papers, and CP's book
    tpool_t *tpool_data;
    uint32_t threads = tdata[0].total_threads;
	base_t i, j;
	int curve;
	FILE *save;
	char fname[80];
	int found = 0;
    int result;
    uint64_t num_found;
	bignum *one = vecInit();
    mpz_t gmpt, gmpn;
    // these track the range over which we currently have a prime list.
    uint64_t rangemin;
    uint64_t rangemax;

	// timing variables
	struct timeval stopt;	// stop time of this job
	struct timeval startt;	// start time of this job
	struct timeval fullstopt;	// stop time of this job
	struct timeval fullstartt;	// start time of this job
	double t_time;

	gettimeofday(&fullstartt, NULL);
    mpz_init(gmpt);
    mpz_init(gmpn);

    // in this function, gmpn is only used to for screen output and to 
    // check factors.  So if this is a Mersenne input, use the original
    // input number.  (all math is still done relative to the base Mersenne.)
    if (tdata[0].mdata->isMersenne == 0)
    {
        extract_bignum_from_vec_to_mpz(gmpn, tdata[0].mdata->n, 0, NWORDS);
    }
    else
    {
        extract_bignum_from_vec_to_mpz(gmpn, tdata[0].mdata->vnhat, 0, NWORDS);
    }

	for (i = 0; i < VECLEN; i++)
	{
		one->data[i] = 1;
	}
    one->size = 1;
    
    tpool_data = tpool_setup(tdata[0].total_threads, NULL, NULL, &ecm_sync,
        &ecm_dispatch, tdata);

    tpool_add_work_fcn(tpool_data, &ecm_build_curve_work_fcn);
    tpool_add_work_fcn(tpool_data, &ecm_stage1_work_fcn);
    tpool_add_work_fcn(tpool_data, &ecm_stage2_init_work_fcn);
    tpool_add_work_fcn(tpool_data, &ecm_stage2_work_fcn);

    rangemin = 0;
    rangemax = MIN(STAGE2_MAX + 1000, (uint64_t)PRIME_RANGE);

	if (PRIMES != NULL) { free(PRIMES); PRIMES = NULL; };
	PRIMES = GetPRIMESRange(spSOEprimes, szSOEp, NULL,
		rangemin, rangemax, &num_found);
	NUM_P = num_found;
	P_MIN = PRIMES[0];
	P_MAX = PRIMES[NUM_P - 1];

	printf("Found %lu primes in range [%lu : %lu]\n", NUM_P, rangemin, rangemax);

	for (curve = 0; curve < tdata[0].curves; curve += VECLEN)
	{
        uint64_t p;

        if (0)
        {
            // clean all work structures since something is impacting our
            // ability to find factors beyond the first set of curves.
            for (i = 0; i < threads; i++)
            {
                vecClear(tdata[i].work->diff1);
                vecClear(tdata[i].work->diff2);
                vecClear(tdata[i].work->sum1);
                vecClear(tdata[i].work->sum2);
                vecClear(tdata[i].work->tt1);
                vecClear(tdata[i].work->tt2);
                vecClear(tdata[i].work->tt3);
                vecClear(tdata[i].work->tt4);
                vecClear(tdata[i].work->tt5);
                vecClear(tdata[i].work->pt1.X);
                vecClear(tdata[i].work->pt2.X);
                vecClear(tdata[i].work->pt3.X);
                vecClear(tdata[i].work->pt4.X);
                vecClear(tdata[i].work->pt5.X);
                vecClear(tdata[i].work->pt1.Z);
                vecClear(tdata[i].work->pt2.Z);
                vecClear(tdata[i].work->pt3.Z);
                vecClear(tdata[i].work->pt4.Z);
                vecClear(tdata[i].work->pt5.Z);
                vecClear(tdata[i].work->s);
                vecClear(tdata[i].work->stg2acc);
                vecClear(tdata[i].mdata->mtmp1);
                vecClear(tdata[i].mdata->mtmp2);
                vecClear(tdata[i].mdata->mtmp3);
                vecClear(tdata[i].mdata->mtmp4);
                //vecClear(tdata[i].work->n);

                for (j = 0; j < (2 * tdata[0].work->L); j++)
                {
                    vecClear(tdata[i].work->Paprod[j]);
                    vecClear(tdata[i].work->Pa_inv[j]);
                    vecClear(tdata[i].work->Pa[j].X);
                    vecClear(tdata[i].work->Pa[j].Z);
                }

                vecClear(tdata[i].work->Pad->X);
                vecClear(tdata[i].work->Pad->Z);

                vecClear(tdata[i].work->Pdnorm->X);
                vecClear(tdata[i].work->Pdnorm->Z);

                for (j = 0; j < tdata[0].work->U * (tdata[0].work->R + 1); j++)
                {
                    vecClear(tdata[i].work->Pb[j].X);
                    vecClear(tdata[i].work->Pb[j].Z);
                    vecClear(tdata[i].work->Pbprod[j]);
                }
            }
        }

        // get a new batch of primes if:
        // - the current range starts after B1
        // - the current range ends after B1 AND the range isn't maximal length
        if ((rangemin > STAGE1_MAX) ||
            ((rangemax < STAGE1_MAX) && ((rangemax - rangemin) < PRIME_RANGE)))
        {
            rangemin = 0;
            rangemax = MIN(STAGE2_MAX + 1000, (uint64_t)PRIME_RANGE);

            if (PRIMES != NULL) { free(PRIMES); PRIMES = NULL; };
            PRIMES = GetPRIMESRange(spSOEprimes, szSOEp, NULL,
                rangemin, rangemax, &num_found);
            NUM_P = num_found;
            P_MIN = PRIMES[0];
            P_MAX = PRIMES[NUM_P - 1];

            printf("Found %lu primes in range [%lu : %lu]\n", NUM_P, rangemin, rangemax);
        }

		gettimeofday(&startt, NULL);
        
        // parallel curve building        
        for (i = 0; i < threads; i++)
        {
			tdata[i].work->stg1Add = 0;
			tdata[i].work->stg1Doub = 0;
			tdata[i].work->last_pid = 0;
            tdata[i].phase_done = 0;
            tdata[i].ecm_phase = 0;
        }
        tpool_go(tpool_data);

		gettimeofday(&stopt, NULL);
		t_time = my_difftime (&startt, &stopt);
		printf("\n");

		printf("Commencing curves %d-%d of %u\n", threads * curve,
			threads * (curve + VECLEN) - 1, threads * tdata[0].curves);
		
		printf("Building curves took %1.4f seconds.\n",t_time);

		// parallel stage 1
		gettimeofday(&startt, NULL);

        for (p = 0; p < STAGE1_MAX; p += PRIME_RANGE)
        {
            // get a new batch of primes if the current range ends
            // before this new one starts
			if (p >= rangemax)
			{
                rangemin = rangemax;
                rangemax = MIN(STAGE2_MAX + 1000, rangemin + (uint64_t)PRIME_RANGE);

				if (PRIMES != NULL) { free(PRIMES); PRIMES = NULL; };
				PRIMES = GetPRIMESRange(spSOEprimes, szSOEp, NULL,
                    rangemin, rangemax, &num_found);
				NUM_P = num_found;
				P_MIN = PRIMES[0];
				P_MAX = PRIMES[NUM_P - 1];

				printf("Found %lu primes in range [%lu : %lu]\n", NUM_P, rangemin, rangemax);
			}

            for (i = 0; i < threads; i++)
            {
                tdata[i].phase_done = 0;
                tdata[i].ecm_phase = 1;
            }

			printf("Commencing Stage 1 @ prime %lu\n", P_MIN);
            tpool_go(tpool_data);

#if 0
            if (PRIMES[tdata[0].work->last_pid] < STAGE1_MAX)
            {
                sprintf(fname, "checkpoint.txt");
                save = fopen(fname, "a");
                if (save != NULL)
                {
                    printf("Saving checkpoint after p=%lu\n", PRIMES[tdata[0].work->last_pid - 1]);
                    for (j = 0; j < threads; j++)
                    {
                        // GMP-ECM wants X/Z.
                        // or, equivalently, X and Z listed separately.
                        vecmulmod_ptr(tdata[j].P->X, one, tdata[j].work->tt4, tdata[j].work->n,
                            tdata[j].work->tt2, tdata[j].mdata);

                        vecmulmod_ptr(tdata[j].P->Z, one, tdata[j].work->tt3, tdata[j].work->n,
                            tdata[j].work->tt2, tdata[j].mdata);

                        for (i = 0; i < VECLEN; i++)
                        {
                            extract_bignum_from_vec_to_mpz(gmpt, tdata[j].P->Z, i, NWORDS);

                            if (mpz_cmp_ui(gmpt, 0) == 0)
                            {
                                printf("something failed: tid = %d, vec = %d has zero result\n", (int)j, (int)i);
                            }

                            result = check_factor(gmpt, gmpn, tdata[j].factor);
                            if (result == 1)
                            {
                                FILE* out = fopen("ecm_results.txt", "a");
                                int isp = mpz_probab_prime_p(tdata[j].factor, 3);
                                char ftype[8];

                                if (isp)
                                    strcpy(ftype, "PRP");
                                else
                                    strcpy(ftype, "C");

                                sprintf(ftype, "%s%d", ftype, (int)mpz_sizeinbase(tdata[j].factor, 10));

                                gmp_printf("\nfound %s factor %Zd in stage 1 (B1 = %lu): thread %d, vec %d, sigma ",
                                    ftype, tdata[j].factor, PRIMES[tdata[0].work->last_pid - 1], j, i);
                                printf("%"PRIu64"\n", tdata[j].sigma[i]);

                                if (out != NULL)
                                {
                                    gmp_fprintf(out, "\nfound %s factor %Zd in stage 1 (B1 = %lu): curve %d, "
                                        "thread %d, vec %d, sigma ",
                                        ftype, tdata[j].factor, PRIMES[tdata[0].work->last_pid - 1],
                                        threads * curve + j * VECLEN + i, j, i);
                                    fprintf(out, "%"PRIu64"\n", tdata[j].sigma[i]);
                                    fclose(out);
                                }
                                fflush(stdout);
                                found = 1;
                            }

                            fprintf(save, "METHOD=ECM; SIGMA=%"PRIu64"; B1=%"PRIu64"; ",
                                tdata[j].sigma[i], PRIMES[tdata[0].work->last_pid - 1]);
                            gmp_fprintf(save, "N=0x%Zx; ", gmpn);

                            extract_bignum_from_vec_to_mpz(gmpt, tdata[j].work->tt4, i, NWORDS);
                            gmp_fprintf(save, "X=0x%Zx; ", gmpt);

                            extract_bignum_from_vec_to_mpz(gmpt, tdata[j].work->tt3, i, NWORDS);
                            gmp_fprintf(save, "Z=0x%Zx; PROGRAM=AVX-ECM;\n", gmpt);
                        }
                    }
                    fclose(save);
                }
                else
                {
                    printf("could not open checkpoint.txt for appending, Stage 1 data will not be saved\n");
                }
            }
#endif
        }

        gettimeofday(&stopt, NULL);
        t_time = my_difftime(&startt, &stopt);
        printf("Stage 1 took %1.4f seconds\n", t_time);
	
        sprintf(fname, "save_b1.txt");
        save = fopen(fname, "a");
        if (save != NULL)
        {
            for (j = 0; j < threads; j++)
            {
                // GMP-ECM wants X/Z.
                // or, equivalently, X and Z listed separately.
                vecmulmod_ptr(tdata[j].P->X, one, tdata[j].work->tt4, tdata[j].work->n,
                    tdata[j].work->tt2, tdata[j].mdata);

                vecmulmod_ptr(tdata[j].P->Z, one, tdata[j].work->tt3, tdata[j].work->n,
                    tdata[j].work->tt2, tdata[j].mdata);

                for (i = 0; i < VECLEN; i++)
                {
                    extract_bignum_from_vec_to_mpz(gmpt, tdata[j].P->Z, i, NWORDS);

                    if (mpz_cmp_ui(gmpt, 0) == 0)
                    {
                        printf("something failed: tid = %d, vec = %d has zero result\n", (int)j, (int)i);
                    }

                    result = check_factor(gmpt, gmpn, tdata[j].factor);
                    if (result == 1)
                    {
                        FILE* out = fopen("ecm_results.txt", "a");
                        int isp = mpz_probab_prime_p(tdata[j].factor, 3);
                        char ftype[8];

                        if (isp)
                            strcpy(ftype, "PRP");
                        else
                            strcpy(ftype, "C");

                        sprintf(ftype, "%s%d", ftype, (int)mpz_sizeinbase(tdata[j].factor, 10));

                        gmp_printf("\nfound %s factor %Zd in stage 1 (B1 = %lu): thread %d, vec %d, sigma ",
                            ftype, tdata[j].factor, STAGE1_MAX, j, i);
                        printf("%"PRIu64"\n", tdata[j].sigma[i]);
                    
                        if (out != NULL)
                        {
                            gmp_fprintf(out, "\nfound %s factor %Zd in stage 1 (B1 = %lu): curve %d, "
                                "thread %d, vec %d, sigma ",
                                ftype, tdata[j].factor, STAGE1_MAX, threads * curve + j * VECLEN + i, j, i);
                            fprintf(out, "%"PRIu64"\n", tdata[j].sigma[i]);
                            fclose(out);
                        }
                        fflush(stdout);
                        found = 1;
                    }

                    fprintf(save, "METHOD=ECM; SIGMA=%"PRIu64"; B1=%"PRIu64"; ",
                        tdata[j].sigma[i], STAGE1_MAX);
                    gmp_fprintf(save, "N=0x%Zx; ", gmpn);

                    extract_bignum_from_vec_to_mpz(gmpt, tdata[j].work->tt4, i, NWORDS);
                    gmp_fprintf(save, "X=0x%Zx; ", gmpt);

                    extract_bignum_from_vec_to_mpz(gmpt, tdata[j].work->tt3, i, NWORDS);
                    gmp_fprintf(save, "Z=0x%Zx; PROGRAM=AVX-ECM;\n", gmpt);
                }
            }
            fclose(save);
        }
        else
        {
            printf("could not open save_b1.txt for appending, Stage 1 data will not be saved\n");
        }

        // always stop when a factor is found
		//if (found)
		//	break;

        if (DO_STAGE2)
        {
            uint64_t last_p = PRIMES[tdata[0].work->last_pid];

            // parallel stage 2
            gettimeofday(&startt, NULL);

			// stage 2 parallel init
            for (i = 0; i < threads; i++)
            {
                tdata[i].phase_done = 0;
                tdata[i].ecm_phase = 2;
            }
            tpool_go(tpool_data);

            for (i = 0; i < threads; i++)
            {
                if (tdata[i].work->last_pid == (uint32_t)(-1))
                {
                    // found a factor while initializing stage 2
                    //printf("received factor from stage 2 init\n");
                    //print_vechexbignum52(tdata[i].work->stg2acc, "stg2acc: ");
                    last_p = STAGE2_MAX;
                }
            }

            gettimeofday(&stopt, NULL);
            t_time = my_difftime(&startt, &stopt);
            printf("Stage 2 Init took %1.4f seconds\n", t_time);

            for (p = STAGE1_MAX; p < STAGE2_MAX; p += PRIME_RANGE)
            {
                // get a new batch of primes if the current range ends
                // before this new one starts or if the range isn't large
                // enough to cover the current step
                if ((p >= rangemax) ||
                    (rangemax < MIN(STAGE2_MAX, p + (uint64_t)PRIME_RANGE)))
				{
                    rangemin = p;
                    rangemax = MIN(STAGE2_MAX + 1000, p + (uint64_t)PRIME_RANGE);

					if (PRIMES != NULL) { free(PRIMES); PRIMES = NULL; };
                    PRIMES = GetPRIMESRange(spSOEprimes, szSOEp, NULL,
                        rangemin, rangemax, &num_found);
					NUM_P = num_found;
					P_MIN = PRIMES[0];
					P_MAX = PRIMES[NUM_P - 1];
            
					for (i = 0; i < threads; i++)
					{
						tdata[i].work->last_pid = 1;
					}
            
					printf("found %lu primes in range [%lu : %lu]\n", NUM_P, rangemin, rangemax);
				}

#ifndef USE_STAGE2_REF
                tdata[0].pairmap_steps = pair(tdata[0].pairmap_v, tdata[0].pairmap_u,
                    tdata[0].work, PRIMES, p, MIN(p + (uint64_t)PRIME_RANGE, STAGE2_MAX), 1);
#endif

                for (i = 0; i < threads; i++)
                {
#ifndef USE_STAGE2_REF
                    tdata[i].work->amin = (p + tdata[i].work->D) / (2 * tdata[i].work->D);
#endif
                    tdata[i].phase_done = 0;
                    tdata[i].ecm_phase = 3;
                }
                tpool_go(tpool_data);

				if (tdata[0].work->last_pid == NUM_P)
				{
					// we ended at the last prime we cached.  Check if we
					// need to do more.  
					if (STAGE2_MAX == PRIME_RANGE)
						break;
					else
						last_p = P_MAX;
				}
				else
				{
					last_p = PRIMES[tdata[0].work->last_pid];
				}
            }

            gettimeofday(&stopt, NULL);
            t_time = my_difftime(&startt, &stopt);
            printf("\n");
            printf("Stage 2 took %1.4f seconds\n", t_time);
            printf("performed %u point-additions and %u point-doubles in stage 2\n",
				tdata[0].work->ptadds + tdata[0].work->stg1Add, tdata[0].work->stg1Doub);

            for (j = 0; j < threads; j++)
            {
				for (i = 0; i < VECLEN; i++)
                {
                    extract_bignum_from_vec_to_mpz(gmpt, tdata[j].work->stg2acc, i, NWORDS);
                    result = check_factor(gmpt, gmpn, tdata[j].factor);

                    if (mpz_cmp_ui(gmpt, 0) == 0)
                    {
                        printf("something failed: tid = %d, vec = %d has zero result\n", (int)j, (int)i);
                    }

                    if (result == 1)
                    {
						FILE *out = fopen("ecm_results.txt", "a");
                        int isp = mpz_probab_prime_p(tdata[j].factor, 3);
                        char ftype[8];

                        if (isp)
                            strcpy(ftype, "PRP");
                        else
                            strcpy(ftype, "C");

                        sprintf(ftype, "%s%d", ftype, (int)mpz_sizeinbase(tdata[j].factor, 10));

                        gmp_printf("\nfound %s factor %Zd in stage 2 (B2 = %lu): thread %d, "
                            "vec %d, sigma ",
                            ftype, tdata[j].factor, STAGE2_MAX, j, i);
                        printf("%"PRIu64"\n", tdata[j].sigma[i]);

						if (out != NULL)
						{
							gmp_fprintf(out, "\nfound %s factor %Zd in stage 2 (B2 = %lu): curve %d, "
								"thread %d, vec %d, sigma ",
                                ftype, tdata[j].factor, STAGE2_MAX, threads * curve + j * VECLEN + i, j, i);
                            fprintf(out, "%"PRIu64"\n", tdata[j].sigma[i]);
							fclose(out);
						}

                        fflush(stdout);
                        found = 1;
                    }
                }
            }
        }

		if (found)
			break;

	}

	gettimeofday(&fullstopt, NULL);
	t_time = my_difftime(&fullstartt, &fullstopt);
	printf("Process took %1.4f seconds.\n", t_time);

	vecClear(one);
    mpz_clear(gmpn);
    mpz_clear(gmpt);
	return;
}

//#define PRINT_DEBUG

void build_one_curve(thread_data_t *tdata, mpz_t X, mpz_t Z, mpz_t A, uint64_t sigma)
{
    monty *mdata = tdata->mdata;
    ecm_work *work = tdata->work;

    mpz_t n, u, v, t1, t2, t3, t4;
    mpz_init(n);
    mpz_init(u);
    mpz_init(v);
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(t3);
    mpz_init(t4);

    extract_bignum_from_vec_to_mpz(n, mdata->n, 0, NWORDS);

    if (sigma == 0)
    {
        do
        {
            work->sigma = lcg_rand(&tdata->lcg_state);
        } while (work->sigma < 6);
    }
    else
    {
        work->sigma = sigma;
    }

    //sigma = work->sigma = 1632562926;
    //sigma = 269820583;
    //sigma = 50873471;
    //sigma = 444711979;		// both good
    //work->sigma = sigma;
    //printf("thread %d running curve on sigma = %"PRIu64"\n", tid, work->sigma);

#ifdef PRINT_DEBUG
    printf("sigma = %lu\n", work->sigma);
#endif

    // v = 4*sigma
    mpz_set_ui(v, work->sigma);
    mpz_mul_2exp(v, v, 2);

#ifdef PRINT_DEBUG
    gmp_printf("v = %Zx\n", v);
#endif

    // u = sigma^2 - 5
    mpz_set_ui(u, work->sigma);
    mpz_mul(u, u, u);
    mpz_sub_ui(u, u, 5);

#ifdef PRINT_DEBUG
    gmp_printf("u = sigma^2 - 5 = %Zx\n", u);
#endif

    // x = u^3
    mpz_mul(X, u, u);
    mpz_mul(X, X, u);
    mpz_tdiv_r(X, X, n);

#ifdef PRINT_DEBUG
    gmp_printf("u^3 = %Zx\n", X);
#endif

    // z = v^3
    mpz_mul(Z, v, v);
    mpz_mul(Z, Z, v);
    mpz_tdiv_r(Z, Z, n);

#ifdef PRINT_DEBUG
    gmp_printf("v^3 = %Zx\n", Z);
#endif


    // compute parameter A
    // (v-u)
    if (mpz_cmp(u, v) > 0)
    {
        mpz_sub(t1, v, u);
        mpz_add(t1, t1, n);
    }
    else
    {
        mpz_sub(t1, v, u);
    }

    // (v-u)^3
    mpz_mul(t2, t1, t1);
    mpz_tdiv_r(t2, t2, n);
    mpz_mul(t4, t2, t1);
    mpz_tdiv_r(t4, t4, n);

    // 3u + v
    mpz_mul_ui(t1, u, 3);
    mpz_add(t3, t1, v);
    mpz_tdiv_r(t3, t3, n);

    // a = (v-u)^3 * (3u + v)
    mpz_mul(t1, t3, t4);
    mpz_tdiv_r(t1, t1, n);

#ifdef PRINT_DEBUG
    gmp_printf("(v-u)^3 = %Zx\n", t4);
    gmp_printf("(3u + v) = %Zx\n", t3);
    gmp_printf("a = %Zx\n", t1);
#endif

    if (0)
    {
        // This is how gmp-ecm does things since sometimes they want
        // the Montgomery parameter A.  We always use Suyama's parameterization
        // so we just go ahead and build (A+2)/4.
        // 4*x*v
        mpz_mul_ui(t2, X, 4);
        mpz_mul(t4, t2, v);
        mpz_tdiv_r(t4, t4, n);

        // 4*x*v * z
        mpz_mul(t3, t4, Z);
        mpz_tdiv_r(t3, t3, n);

        /* u = 1 / (v^3 * 4*u^3*v) */
        mpz_invert(t2, t3, n);

        /* v = z^(-1) (mod n)  = 1 / v^3   */
        mpz_mul(v, t2, t4);   
        mpz_tdiv_r(v, v, n);

        /* x = x * z^(-1)      = u^3 / v^3 */
        mpz_mul(X, X, v);
        mpz_tdiv_r(X, X, n);

        /* v = b^(-1) (mod n)  = 1 / 4*u^3*v */
        mpz_mul(v, t2, Z);
        mpz_tdiv_r(v, v, n);

        /* t = ((v-u)^3 * (3*u+v)) / 4*u^3*v */
        mpz_mul(t1, t1, v);       
        mpz_tdiv_r(t1, t1, n);

        /* A = ((v-u)^3 * (3*u+v)) / 4*u^3*v - 2*/
        mpz_sub_ui(A, t1, 2);
        if (mpz_sgn(A) < 0)
        {
            mpz_add(A, A, n);
        }

        mpz_mul_2exp(X, X, DIGITBITS * NWORDS);
        mpz_tdiv_r(X, X, n);
        mpz_mul_2exp(Z, Z, DIGITBITS * NWORDS);
        mpz_tdiv_r(Z, Z, n);
        mpz_mul_2exp(A, A, DIGITBITS * NWORDS);
        mpz_tdiv_r(A, A, n);

        //gmp_printf("X/Z = %Zx\n", X);
        //gmp_printf("Z = %Zx\n", Z);
        //gmp_printf("A = %Zx\n", A);
        //
        //printf("(A+2)*B/4\n");

        mpz_set_ui(t1, 2);
        mpz_mul_2exp(t1, t1, DIGITBITS * NWORDS);
        mpz_add(A, A, t1);
        mpz_tdiv_r(A, A, n);

        if (mpz_odd_p(A))
        {
            mpz_add(A, A, n);
        }
        mpz_tdiv_q_2exp(A, A, 1);

        if (mpz_odd_p(A))
        {
            mpz_add(A, A, n);
        }
        mpz_tdiv_q_2exp(A, A, 1);

        //gmp_printf("A = %Zx\n", A);
        //exit(1);
    }
    else
    {
        // We always use Suyama's parameterization
        // so we just go ahead and build (A+2)/4.

        // 16*u^3*v
        mpz_mul_ui(t2, X, 16);
        mpz_mul(t4, t2, v);
        mpz_tdiv_r(t4, t4, n);

#ifdef PRINT_DEBUG
        gmp_printf("16*u^3*v = %Zx\n", t4);
#endif

        // accomplish the division by multiplying by the modular inverse
        // of the denom
        mpz_invert(t2, t4, n);

#ifdef PRINT_DEBUG
        gmp_printf("inv = %Zx\n", t2);
#endif

        // t1 = b = (v - u)^3 * (3*u + v) / 16u^3v)
        mpz_mul(A, t1, t2);
        mpz_tdiv_r(A, A, n);

#ifdef PRINT_DEBUG
        gmp_printf("b = %Zx\n", A);
#endif

        mpz_invert(t1, Z, n);
        mpz_mul(X, X, t1);
        mpz_set_ui(Z, 1);

        if (mdata->isMersenne == 0)
        {
            // into Monty rep
            mpz_mul_2exp(X, X, DIGITBITS * NWORDS);
            mpz_tdiv_r(X, X, n);
            mpz_mul_2exp(Z, Z, DIGITBITS * NWORDS);
            mpz_tdiv_r(Z, Z, n);
            mpz_mul_2exp(A, A, DIGITBITS * NWORDS);
            mpz_tdiv_r(A, A, n);
        }
        else
        {
            mpz_tdiv_r(X, X, n);
            mpz_tdiv_r(Z, Z, n);
            mpz_tdiv_r(A, A, n);
        }


        //mpz_mul_2exp(X, X, DIGITBITS * NWORDS);
        //mpz_tdiv_r(X, X, n);
        //mpz_mul_2exp(Z, Z, DIGITBITS * NWORDS);
        //mpz_tdiv_r(Z, Z, n);
        //mpz_mul_2exp(A, A, DIGITBITS * NWORDS);
        //mpz_tdiv_r(A, A, n);

        //gmp_printf("X = %Zx\n", X);
        //gmp_printf("Z = %Zx\n", Z);
        //gmp_printf("A = %Zx\n", A);
        //exit(1);
    }
	
    mpz_clear(n);
    mpz_clear(u);
    mpz_clear(v);
    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(t3);
    mpz_clear(t4);

	return;
}

//#define TESTMUL
void ecm_stage1(monty *mdata, ecm_work *work, ecm_pt *P, base_t b1, base_t *primes, int verbose)
{
	int i;
	uint64_t q;
	uint64_t stg1 = STAGE1_MAX;

	// handle the only even case 
	q = 2;
	while (q < STAGE1_MAX)
	{
		vecsubmod_ptr(P->X, P->Z, work->diff1, mdata);
		vecaddmod_ptr(P->X, P->Z, work->sum1, mdata);
		vec_duplicate(mdata, work, work->sum1, work->diff1, P);
		q *= 2;
	}

	for (i = 1; (i < NUM_P) && (PRIMES[i] < STAGE1_MAX); i++)
	{
		uint64_t c = 1;
	
		q = PRIMES[i];
		do {
			prac(mdata, work, P, q);
			c *= q;
		} while ((c * q) < stg1);
	
#ifdef SKYLAKEX
        if ((verbose >= 1) && ((i & 8191) == 0))
#else
		if ((verbose >= 1) && ((i & 511) == 0))
#endif
		{
			printf("accumulating prime %lu\r", q);
			fflush(stdout);
		}
	}

	work->last_pid = i;

	if (verbose >= 1)
	{
		printf("\nStage 1 completed at prime %lu with %u point-adds and %u point-doubles\n", 
			PRIMES[i-1], work->stg1Add, work->stg1Doub);
		fflush(stdout);
	}
	return;
}


#define CROSS_PRODUCT_INV \
    vecsubmod_ptr(work->Pa_inv[pa], Pb[rprime_map_U[pb]].X, work->tt1, mdata);          \
    vecmulmod_ptr(acc, work->tt1, acc, work->n, work->tt4, mdata);        

#define CROSS_PRODUCT \
    vecsubmod_ptr(Pa[pa].X, Pb[rprime_map_U[pb]].X, work->tt1, mdata);          \
    vecaddmod_ptr(Pa[pa].Z, Pb[rprime_map_U[pb]].Z, work->tt2, mdata);          \
    vecmulmod_ptr(work->tt1, work->tt2, work->tt3, work->n, work->tt4, mdata);    \
    vecaddmod_ptr(work->tt3, Pbprod[rprime_map_U[pb]], work->tt1, mdata);       \
    vecsubmod_ptr(work->tt1, Paprod[pa], work->tt2, mdata);                     \
    vecmulmod_ptr(acc, work->tt2, acc, work->n, work->tt4, mdata);      

void ecm_stage2_init_ref(ecm_pt* P, monty* mdata, ecm_work* work, int verbose)
{
    // run Montgomery's PAIR algorithm.  
    uint32_t D = work->D;
    uint32_t R = work->R;
    uint32_t w = D;
    uint32_t U = work->U;
    uint32_t L = work->L;
    uint32_t umax = U * w;
    int i, j, k, pid;

    uint32_t ainc = 2 * D;
    uint32_t ascale = 2 * D;
    uint32_t amin = work->amin = (STAGE1_MAX + w) / ascale;
    uint32_t s;
    uint32_t a;

    uint32_t numR;
    uint32_t u, ap;
    int q, mq;
    int debug = 0;
    int inverr;
    int foundDuringInv = 0;
    int doneIfFoundDuringInv = 0;

    uint32_t* rprime_map_U = work->map;
    ecm_pt* Pa = work->Pa;
    bignum** Paprod = work->Paprod;
    bignum** Pbprod = work->Pbprod;
    ecm_pt* Pb = work->Pb;
    ecm_pt* Pd = work->Pdnorm;
    bignum* acc = work->stg2acc;
    int lastMapID;
    char str[1024];

    mpz_t gmptmp, gmptmp2, gmpn;
    mpz_init(gmptmp);
    mpz_init(gmptmp2);
    mpz_init(gmpn);

    if (verbose == 1)
        printf("\n");

    work->paired = 0;
    work->numprimes = 0;
    work->ptadds = 0;
    work->stg1Add = 0;
    work->stg1Doub = 0;

    //stage 2 init
    //Q = P = result of stage 1
    //compute [d]Q for 0 < d <= D

    // [1]Q
    vecCopy(P->Z, Pb[1].Z);
    vecCopy(P->X, Pb[1].X);

    // [2]Q
    vecCopy(P->Z, Pb[2].Z);
    vecCopy(P->X, Pb[2].X);
    vecaddsubmod_ptr(P->X, P->Z, work->sum1, work->diff1, mdata);
    vec_duplicate(mdata, work, work->sum1, work->diff1, &Pb[2]);

    // [3]Q, [4]Q, ... [D]Q
    // because of the way we pick 'a' and 'D', we'll only need the 479 points that 
    // are relatively prime to D.  We need to keep a few points during the running 
    // computation i.e., ... j, j-1, j-2. The lookup table rprime_map maps the 
    // integer j into the relatively prime indices that we store. We also store
    // points 1, 2, and D, and storage point 0 is used as scratch for a total of
    // 483 stored points.

    vecCopy(Pb[1].X, work->pt2.X);
    vecCopy(Pb[1].Z, work->pt2.Z);
    vecCopy(Pb[2].X, work->pt1.X);
    vecCopy(Pb[2].Z, work->pt1.Z);

    lastMapID = 0;
    k = 0;
    for (j = 3; j <= U * D; j++)
    {
        ecm_pt* P1 = &work->pt1;			// Sd - 1
        ecm_pt* P2 = &Pb[1];				// S1
        ecm_pt* P3 = &work->pt2;			// Sd - 2
        ecm_pt* Pout = &Pb[rprime_map_U[j]];	// Sd

        if (rprime_map_U[j] > 0)
            lastMapID = rprime_map_U[j];

        // vecAdd:
        //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
        //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
        //x- = original x
        //z- = original z

        // compute Sd from Sd-1 + S1, requiring Sd-1 - S1 = Sd-2
        vecaddsubmod_ptr(P1->X, P1->Z, work->sum1, work->diff1, mdata);
        vecaddsubmod_ptr(P2->X, P2->Z, work->sum2, work->diff2, mdata);

        vecmulmod_ptr(work->diff1, work->sum2, work->tt1, work->n, work->tt4, mdata);	//U
        vecmulmod_ptr(work->sum1, work->diff2, work->tt2, work->n, work->tt4, mdata);	//V

        vecaddsubmod_ptr(work->tt1, work->tt2, Pout->X, Pout->Z, mdata);		        //U +/- V
        vecsqrmod_ptr(Pout->X, work->tt1, work->n, work->tt4, mdata);					//(U + V)^2
        vecsqrmod_ptr(Pout->Z, work->tt2, work->n, work->tt4, mdata);					//(U - V)^2

        // if gcd(j,D) != 1, Pout maps to scratch space (Pb[0])
        vecmulmod_ptr(work->tt1, P3->Z, Pout->X, work->n, work->tt4, mdata);			//Z * (U + V)^2
        vecmulmod_ptr(work->tt2, P3->X, Pout->Z, work->n, work->tt4, mdata);			//x * (U - V)^2

        //store Pb[j].X * Pb[j].Z as well
        //vecmulmod_ptr(Pout->X, Pout->Z, Pbprod[rprime_map_U[j]],
        //    work->n, work->tt4, mdata);

        work->ptadds++;

        // advance
        vecCopy(P1->X, P3->X);
        vecCopy(P1->Z, P3->Z);
        vecCopy(Pout->X, P1->X);
        vecCopy(Pout->Z, P1->Z);

        //sprintf(str, "Pb[%d].Z: ", rprime_map_U[j]);
        //print_vechex(Pout->Z->data, 0, NWORDS, str);
        //printf("rprime_map_U[%d] = %u\n", j, rprime_map_U[j]);
    }

    //printf("B table generated to umax = %d\n", U * D);

    // Pd = [2w]Q
    vecCopy(P->Z, Pd->Z);
    vecCopy(P->X, Pd->X);
    next_pt_vec(mdata, work, Pd, ainc);

    //first a value: first multiple of D greater than B1
    work->A = amin * ascale;

    //initialize info needed for giant step
    vecCopy(P->Z, Pa[0].Z);
    vecCopy(P->X, Pa[0].X);
    next_pt_vec(mdata, work, &Pa[0], work->A);

    if (verbose & (debug == 2))
        printf("Pa[0] = [%lu]Q\n", work->A);

    vecCopy(P->Z, work->Pad->Z);
    vecCopy(P->X, work->Pad->X);
    next_pt_vec(mdata, work, work->Pad, work->A - ainc);

    if (verbose & (debug == 2))
        printf("Pad = [%lu]Q\n", work->A - ainc);

    vecaddmod_ptr(Pa[0].X, Pa[0].Z, work->sum1, mdata);
    vecaddmod_ptr(Pd->X, Pd->Z, work->sum2, mdata);
    vecsubmod_ptr(Pa[0].X, Pa[0].Z, work->diff1, mdata);
    vecsubmod_ptr(Pd->X, Pd->Z, work->diff2, mdata);
    vec_add(mdata, work, work->Pad, &Pa[1]);

    work->A += ainc;
    if (verbose & (debug == 2))
        printf("Pa[1] = [%lu]Q\n", work->A + ainc);

    for (i = 2; i < 2 * L; i++)
    {
        //giant step - use the addition formula for ECM
        //Pa + Pd
        //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
        //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
        //x- = [a-d]x
        //z- = [a-d]z
        vecaddsubmod_ptr(Pa[i - 1].X, Pa[i - 1].Z, work->sum1, work->diff1, mdata);
        vecaddsubmod_ptr(Pd->X, Pd->Z, work->sum2, work->diff2, mdata);
        vec_add(mdata, work, &Pa[i - 2], &Pa[i]);

        work->A += ainc;
        work->ptadds++;
        if (verbose & (debug == 2))
            printf("Pa[%d] = [%lu]Q\n", i, work->A + i * ainc);
    }

    // initialize accumulator
    vecCopy(mdata->one, acc);

    // invert all Pb's 
    batch_invert_pt_inplace(Pb, Pbprod, mdata, work, lastMapID + 1);

    // invert all Pa's into a separate vector
    batch_invert_pt_to_bignum(Pa, work->Pa_inv, Paprod, mdata, work, 0, 2 * L);

    if (doneIfFoundDuringInv && foundDuringInv)
    {
        work->last_pid = -1;

        mpz_clear(gmptmp);
        mpz_clear(gmptmp2);
        mpz_clear(gmpn);

        return;
    }

    mpz_clear(gmptmp);
    mpz_clear(gmptmp2);
    mpz_clear(gmpn);

    if (verbose & (debug == 2))
        printf("A table generated to 2 * L = %d\n", 2 * L);

    return;
}

void ecm_stage2_pair_ref(ecm_pt* P, monty* mdata, ecm_work* work, int verbose)
{
    // run Montgomery's PAIR algorithm.  
    uint32_t D = work->D;
    uint32_t R = work->R;
    uint32_t w = D;
    uint32_t U = work->U;
    uint32_t L = work->L;
    uint32_t umax = U * w;
    int i, j, k, r, pid;
    Queue_t** Q = work->Q;
    uint32_t ainc = 2 * D;
    uint32_t ascale = 2 * D;
    uint32_t amin = work->amin;
    int debug = 0;
    int inverr;

    uint32_t* rprime_map_U = work->map;
    ecm_pt* Pa = work->Pa;      // non-inverted
    ecm_pt* Pb = work->Pb;      // inverted
    ecm_pt* Pd = work->Pdnorm;  // non-inverted Pd
    bignum* acc = work->stg2acc;
    int flags[8 * 2 * 1155];

    mpz_t gmptmp, gmptmp2, gmpn;
    mpz_init(gmptmp);
    mpz_init(gmptmp2);
    mpz_init(gmpn);

    extract_bignum_from_vec_to_mpz(gmpn, mdata->n, 0, NWORDS);

    if (verbose == 1)
        printf("\n");

    pid = work->last_pid;

    int numranges = (STAGE2_MAX - STAGE1_MAX) / (U * D * 2) + 1;
    int debugid = -1;

    while (PRIMES[pid] < amin * ascale)
    {
        pid++;
    }

    if (verbose)
    {
        printf("commencing stage 2 at p=%lu, A=%u\n"
            "w = %u, L = %u, U = %d, umax = %u, amin = %u\n",
            PRIMES[pid], amin * ascale, w, L, U, umax, amin);
    }

    for (r = 0; r < numranges; r++)
    {
        int pa, pb;

        // make the pair map for this range
        memset(flags, 0, U * 2310 * sizeof(int));

        while ((PRIMES[pid] - amin * ascale) < U * 2310)
        {
            if ((verbose == 1) && ((pid & 65535) == 0))
            {
                printf("accumulating prime %lu\r", PRIMES[pid]);
                fflush(stdout);
            }
            work->numprimes++;
            flags[PRIMES[pid] - amin * ascale] = U;
            //if (PRIMES[pid] == pdebug)
            //{
            //    printf("flag set at location %lu in range %d with amin = %u\n", 
            //        PRIMES[pid] - amin * ascale, r, amin);
            //    debugid = PRIMES[pid] - amin * ascale;
            //}
            pid++;
        }

        for (j = 0; j < U; j++)
        {
            for (i = 0; i < ainc; i++)
            {
                if (flags[j * ainc + i] == U)
                {
                    // unpaired prime.  pair with the least possible mate
                    // from the complimentary class that is up to a 
                    // max of U/2 strides away (to keep within our umax).
                    int thisclass = ainc - i;

                    for (k = j + 1; k < U; k++)
                    {
                        thisclass = ainc * k - i;
                        if ((thisclass > umax) || ((ainc * k + thisclass) > U * 2310))
                            break;

                        if (flags[ainc * k + thisclass] == U)
                        {
                            flags[j * ainc + i] = k;
                            flags[k * ainc + thisclass] = -1;
                            //printf("flag at %d paired with %d using pa=%d, pb=%d\n",
                            //    j * 2310 + i, k * 2310 + thisclass, k,
                            //    k * 2310 - (j * 2310 + i));
                        }
                    }

                    // if nothing pairs, the prime is left to be paired with
                    // the first available composite.
                }
            }
        }

        //if (debugdone)
        //    debugid = -1;

        // do the cross multiplies
        for (i = 0; i < U * 2310; i++)
        {
            //if ((i == 2473) && (amin == 1156))
            //{
            //    printf("processing flag %d=%d at amin=%u, p=%u\n", i, flags[i], amin, amin * ascale + i);
            //}

            if (flags[i] == U)
            {
                int found = 0;
                //printf("searching for pair at location %u, amin = %u... ", i, amin);
                for (pa = 1; pa <= U; pa++)
                {
                    if (((pa * 2310 - i) > 0) && (rprime_map_U[pa * 2310 - i] > 0))
                    {
                        pb = pa * 2310 - i;
                        //if (((amin * ascale + pa * 2310 - pb) == pdebug) ||
                        //    (((amin * ascale + pa * 2310 + pb) == pdebug)))
                        //{
                        //    printf("special accumulating %lu with amin=%u, pa=%d, pb=%d\n", 
                        //        (uint64_t)pdebug, amin, pa, pb);
                        //}
                        CROSS_PRODUCT_INV;
                        work->paired++;
                        found = 1;
                        break;
                    }
                }
                //if (found)
                //{
                //    printf("found at pa=%d, pb=%d\n", pa, pb);
                //}
                //else
                //{
                //    printf("special at location %u not found!\n", amin * ascale + i);
                //}
                if (!found)
                {
                    printf("unpaired location %u not found!\n", amin * ascale + i);
                }
            }
            else if (flags[i] > 0)
            {
                pa = flags[i];
                pb = pa * 2310 - i;

                //if ((i == 2473) && (amin == 1156))
                //{
                //    printf("pa=%d,pb=%d\n", pa, pb);
                //    debugdone = 1;
                //}

                //if (((amin * ascale + pa * 2310 - pb) == pdebug) ||
                //    (((amin * ascale + pa * 2310 + pb) == pdebug)))
                //{
                //    printf("accumulating %lu with amin=%u, pa=%d, pb=%d\n", 
                //        (uint64_t)pdebug, amin, pa, pb);
                //}
                CROSS_PRODUCT_INV;
                work->paired++;
            }
        }



        // more A's
        // shift out uneeded A's
        for (i = 0; i < (2 * L) - U; i++)
        {
            vecCopy(Pa[i + U].X, Pa[i].X);
            vecCopy(Pa[i + U].Z, Pa[i].Z);
            vecCopy(work->Pa_inv[i + U], work->Pa_inv[i]);
        }

        // make new A's using the last two points
        for (i = (2 * L) - U; i < (2 * L); i++)
        {
            //printf("new batch of %d A's\n", (amin - oldmin));
            //giant step - use the addition formula for ECM
            //Pa + Pd
            //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
            //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
            //x- = [a-d]x
            //z- = [a-d]z
            vecaddsubmod_ptr(Pa[i - 1].X, Pa[i - 1].Z, work->sum1, work->diff1, mdata);
            vecaddsubmod_ptr(Pd->X, Pd->Z, work->sum2, work->diff2, mdata);
            vec_add(mdata, work, &Pa[i - 2], &Pa[i]);

            work->A += ainc;
            work->ptadds++;
            amin++;
        }

        batch_invert_pt_to_bignum(Pa, work->Pa_inv, work->Paprod, mdata, work,
            (2 * L) - U, (2 * L));
    }

    pid = NUM_P;

    work->amin = amin;
    work->last_pid = pid;

    mpz_clear(gmptmp);
    mpz_clear(gmptmp2);
    mpz_clear(gmpn);

    return;
}

int batch_invert_pt_inplace(ecm_pt* pts_to_Zinvert,  
    bignum **tmp_vec, monty* mdata, ecm_work* work, int num)
{
    bignum** B;
    bignum** A = tmp_vec;
    int i;
    int j;
    int inverr;
    int foundDuringInv = 0;

    mpz_t gmptmp, gmptmp2, gmpn;
    mpz_init(gmptmp);
    mpz_init(gmptmp2);
    mpz_init(gmpn);

    // here, we have temporary space for B, A is put into the unused Pbprod, and C is Pb.Z.
    // faster batch inversion in three phases, as follows:
    // first, set A1 = z1 and Ai = zi * A(i-1) so that Ai = prod(j=1,i,zj).
    vecCopy(pts_to_Zinvert[1].Z, A[1]);
    for (i = 2; i < num; i++)
    {
        vecmulmod_ptr(pts_to_Zinvert[i].Z, A[i - 1], A[i], work->n, work->tt4, mdata);
    }

    B = (bignum * *)malloc(num * sizeof(bignum*));

    for (j = 0; j < num; j++)
    {
        B[j] = vecInit();
    }

    // now we have to take An out of monty rep so we can invert it.
    if (mdata->isMersenne == 0)
    {
        vecClear(work->tt1);
        for (j = 0; j < VECLEN; j++)
        {
            work->tt1->data[j] = 1;
        }
        work->tt1->size = 1;
        vecmulmod_ptr(A[num-1], work->tt1, B[num-1], work->n, work->tt4, mdata);
    }
    else
    {
        vecCopy(A[num-1], B[num-1]);
    }

    extract_bignum_from_vec_to_mpz(gmpn, mdata->n, 0, NWORDS);
    for (j = 0; j < VECLEN; j++)
    {
        // extract this vec position so we can use mpz_invert.
        extract_bignum_from_vec_to_mpz(gmptmp, B[num-1], j, NWORDS);

        // invert it
        inverr = mpz_invert(gmptmp2, gmptmp, gmpn);

        if (inverr == 0)
        {
            //extract_bignum_from_vec_to_mpz(gmptmp, work->tt2, j, NWORDS);
            //printf("inversion error\n");
            //gmp_printf("tried to invert %Zd mod %Zd in stage2init Pb\n", gmptmp, gmpn);
            mpz_gcd(gmptmp, gmptmp, gmpn);
            //gmp_printf("the GCD is %Zd\n", gmptmp);
            int k;
            for (k = 0; k < NWORDS; k++)
                work->stg2acc->data[k * VECLEN + j] = 0;
            insert_mpz_to_vec(work->stg2acc, gmptmp, j);
            foundDuringInv = 1;
        }

        if (mdata->isMersenne == 0)
        {
            // now put it back into Monty rep.
            mpz_mul_2exp(gmptmp2, gmptmp2, MAXBITS);
            mpz_tdiv_r(gmptmp2, gmptmp2, gmpn);
        }

        // and stuff it back in the vector.
        insert_mpz_to_vec(B[num-1], gmptmp2, j);
    }

    //if (doneIfFoundDuringInv && foundDuringInv)
    //{
    //    work->last_pid = -1;
    //
    //    mpz_clear(gmptmp);
    //    mpz_clear(gmptmp2);
    //    mpz_clear(gmpn);
    //
    //    return;
    //}


    // and continue.
    for (i = num - 2; i >= 0; i--)
    {
        vecmulmod_ptr(pts_to_Zinvert[i + 1].Z, B[i + 1], B[i], work->n, work->tt4, mdata);
    }

    // Now we have Bi = prod(j=1,i,zj^-1).
    // finally, set C1 = B1 and Ci = A(i-1) * B(i) for i > 1.
    // Then Ci = zi^-1 for i > 1.
    vecCopy(B[1], pts_to_Zinvert[1].Z);

    for (i = 2; i < num; i++)
    {
        vecmulmod_ptr(B[i], A[i - 1], pts_to_Zinvert[i].Z, work->n, work->tt4, mdata);
    }

    // each phase takes n-1 multiplications so we have 3n-3 total multiplications
    // and one inversion mod N.
    // but we still have to combine with the X coord.
    for (i = 1; i < num; i++)
    {
        vecmulmod_ptr(pts_to_Zinvert[i].X, pts_to_Zinvert[i].Z, pts_to_Zinvert[i].X,
            work->n, work->tt4, mdata);
    }

    for (j = 0; j < num; j++)
    {
        vecFree(B[j]);
    }
    free(B);

    mpz_clear(gmptmp);
    mpz_clear(gmptmp2);
    mpz_clear(gmpn);

    return foundDuringInv;

}

int batch_invert_pt_to_bignum(ecm_pt* pts_to_Zinvert, bignum **out,
    bignum** tmp_vec, monty* mdata, ecm_work* work, int startid, int stopid)
{
    bignum** B;
    bignum** A = tmp_vec;
    int i;
    int j;
    int inverr = 0;
    int foundDuringInv = 0;
    int num = stopid - startid;

    mpz_t gmptmp, gmptmp2, gmpn;
    mpz_init(gmptmp);
    mpz_init(gmptmp2);
    mpz_init(gmpn);

    // here, we have temporary space for B, A is put into the unused Pbprod, and C is Pb.Z.
    // faster batch inversion in three phases, as follows:
    // first, set A1 = z1 and Ai = zi * A(i-1) so that Ai = prod(j=1,i,zj).
    vecCopy(pts_to_Zinvert[startid].Z, A[0]);
    for (i = 1; i < num; i++)
    {
        vecmulmod_ptr(pts_to_Zinvert[startid + i].Z, A[i - 1], A[i], work->n, work->tt4, mdata);
    }

    B = (bignum * *)malloc(num * sizeof(bignum*));

    for (j = 0; j < num; j++)
    {
        B[j] = vecInit();
    }

    // now we have to take An out of monty rep so we can invert it.
    if (mdata->isMersenne == 0)
    {
        vecClear(work->tt1);
        for (j = 0; j < VECLEN; j++)
        {
            work->tt1->data[j] = 1;
        }
        work->tt1->size = 1;
        vecmulmod_ptr(A[num - 1], work->tt1, B[num - 1], work->n, work->tt4, mdata);
    }
    else
    {
        vecCopy(A[num - 1], B[num - 1]);
    }

    extract_bignum_from_vec_to_mpz(gmpn, mdata->n, 0, NWORDS);
    for (j = 0; j < VECLEN; j++)
    {
        // extract this vec position so we can use mpz_invert.
        extract_bignum_from_vec_to_mpz(gmptmp, B[num - 1], j, NWORDS);

        // invert it
        inverr = mpz_invert(gmptmp2, gmptmp, gmpn);

        if (inverr == 0)
        {
            //extract_bignum_from_vec_to_mpz(gmptmp, work->tt2, j, NWORDS);
            //printf("inversion error\n");
            //gmp_printf("tried to invert %Zd mod %Zd in stage2init Pb\n", gmptmp, gmpn);
            mpz_gcd(gmptmp, gmptmp, gmpn);
            //gmp_printf("the GCD is %Zd\n", gmptmp);
            int k;
            for (k = 0; k < NWORDS; k++)
                work->stg2acc->data[k * VECLEN + j] = 0;
            insert_mpz_to_vec(work->stg2acc, gmptmp, j);
            foundDuringInv = 1;
        }

        if (mdata->isMersenne == 0)
        {
            // now put it back into Monty rep.
            mpz_mul_2exp(gmptmp2, gmptmp2, MAXBITS);
            mpz_tdiv_r(gmptmp2, gmptmp2, gmpn);
        }

        // and stuff it back in the vector.
        insert_mpz_to_vec(B[num - 1], gmptmp2, j);
    }

    //if (doneIfFoundDuringInv && foundDuringInv)
    //{
    //    work->last_pid = -1;
    //
    //    mpz_clear(gmptmp);
    //    mpz_clear(gmptmp2);
    //    mpz_clear(gmpn);
    //
    //    return;
    //}


    // and continue.
    for (i = num - 2; i >= 0; i--)
    {
        vecmulmod_ptr(pts_to_Zinvert[startid + i + 1].Z, B[i + 1], B[i], work->n, work->tt4, mdata);
    }

    // Now we have Bi = prod(j=1,i,zj^-1).
    // finally, set C1 = B1 and Ci = A(i-1) * B(i) for i > 1.
    // Then Ci = zi^-1 for i > 1.
    vecCopy(B[0], out[startid + 0]);

    for (i = 1; i < num; i++)
    {
        vecmulmod_ptr(B[i], A[i - 1], out[startid + i], work->n, work->tt4, mdata);
    }

    // each phase takes n-1 multiplications so we have 3n-3 total multiplications
    // and one inversion mod N.
    // but we still have to combine with the X coord.
    for (i = 0; i < num; i++)
    {
        vecmulmod_ptr(pts_to_Zinvert[startid + i].X, out[startid + i], out[startid + i],
            work->n, work->tt4, mdata);
    }

    for (j = 0; j < num; j++)
    {
        vecFree(B[j]);
    }
    free(B);

    mpz_clear(gmptmp);
    mpz_clear(gmptmp2);
    mpz_clear(gmpn);

    return foundDuringInv;

}

// test cases for generic input at B1=1e6, B2=100B1:
// n = 142946323174762557214361604817789197531833590620956958433836799929503392464892596183803921
//11919771003873180376
//827341355533811391
//6409678826612327146
//13778091190526084667
//10019108749973911965
//10593445070074576128
//16327347202299112611
//13768494887674349585
//17303758977955016383
//2123812563661387803
//2330438305415445111
//12942218412106273630
//5427613898610684157
//13727269399001077418
//3087408422684406072
//8338236510647016635
//18232185847183255223
//5070879816975737551
//9793972958987869750
//1683842010542383008
//16668736769625151751
//11148653366342049109
//6736437364141805734
//8860111571919296085
//15708855786729755459
//4263089024287634346
//10705409183485702771
//5104801995378138195
//9551766994217130412
//17824508581606173922
//4444245868135963544
//14755844915853888743
//4749513976499976002
//3933740986814285076
//2498288573977543008
//18051693002182940438
//421313926042840093
//1659254194582388863
//13762123388521706810
//1318769405167840394
//14979751960240161797
//4989253092822783329
//14628970911725975539
//4759771957864370849
//17870405635651283010
//472060146
//3776270672
//3954243165
//2576580518
//416265588


void addflag(uint8_t* flags, int loc)
{
    //printf("adding flag to %u\n", loc); fflush(stdout);
    //if (flags[loc] == 1)
    //    printf("duplicate location %u\n", loc);
    flags[loc] = 1;
    return;
}

int ecm_stage2_init(ecm_pt* P, monty* mdata, ecm_work* work, int verbose)
{
    // compute points used during the stage 2 pair algorithm.
    uint32_t w = work->D;
    uint32_t U = work->U;
    uint32_t L = work->L;
    int i, j;
    uint32_t amin = work->amin = (STAGE1_MAX + w) / (2 * w);
    int wscale = 1;

    int debug = 0;
    int inverr;
    int foundDuringInv = 0;
    int doneIfFoundDuringInv = 0;

    uint32_t* rprime_map_U = work->map;
    ecm_pt* Pa = work->Pa;
    bignum** Paprod = work->Paprod;
    bignum** Pbprod = work->Pbprod;
    ecm_pt* Pb = work->Pb;
    ecm_pt* Pd = work->Pdnorm;
    bignum* acc = work->stg2acc;
    int lastMapID;

    mpz_t gmptmp, gmptmp2, gmpn;
    mpz_init(gmptmp);
    mpz_init(gmptmp2);
    mpz_init(gmpn);

    if (verbose == 1)
        printf("\n");

    work->paired = 0;
    work->numprimes = 0;
    work->ptadds = 0;
    work->stg1Add = 0;
    work->stg1Doub = 0;

    //stage 2 init
    //Q = P = result of stage 1
    //compute [d]Q for 0 < d <= D

    // [1]Q
    vecCopy(P->Z, Pb[1].Z);
    vecCopy(P->X, Pb[1].X);

    // [2]Q
    vecCopy(P->Z, Pb[2].Z);
    vecCopy(P->X, Pb[2].X);
    vecaddsubmod_ptr(P->X, P->Z, work->sum1, work->diff1, mdata);
    vec_duplicate(mdata, work, work->sum1, work->diff1, &Pb[2]);

    //printf("init: D = %d, ainc = %d, ascale = %d, U = %d, L = %d\n", D, ainc, ascale, U, L);

    // [3]Q, [4]Q, ... [D]Q
    // because of the way we pick 'a' and 'D', we'll only need the 479 points that 
    // are relatively prime to D.  We need to keep a few points during the running 
    // computation i.e., ... j, j-1, j-2. The lookup table rprime_map maps the 
    // integer j into the relatively prime indices that we store. We also store
    // points 1, 2, and D, and storage point 0 is used as scratch for a total of
    // 483 stored points.

    vecCopy(Pb[1].X, work->pt2.X);
    vecCopy(Pb[1].Z, work->pt2.Z);
    vecCopy(Pb[2].X, work->pt1.X);
    vecCopy(Pb[2].Z, work->pt1.Z);

    lastMapID = 0;
    for (j = 3; j <= U * w; j++)
    {
        ecm_pt* P1 = &work->pt1;			// Sd - 1
        ecm_pt* P2 = &Pb[1];				// S1
        ecm_pt* P3 = &work->pt2;			// Sd - 2
        ecm_pt* Pout = &Pb[rprime_map_U[j]];	// Sd

        if (rprime_map_U[j] > 0)
            lastMapID = rprime_map_U[j];

        // vecAdd:
        //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
        //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
        //x- = original x
        //z- = original z

        // compute Sd from Sd-1 + S1, requiring Sd-1 - S1 = Sd-2
        vecaddsubmod_ptr(P1->X, P1->Z, work->sum1, work->diff1, mdata);
        vecaddsubmod_ptr(P2->X, P2->Z, work->sum2, work->diff2, mdata);

        vecmulmod_ptr(work->diff1, work->sum2, work->tt1, work->n, work->tt4, mdata);	//U
        vecmulmod_ptr(work->sum1, work->diff2, work->tt2, work->n, work->tt4, mdata);	//V

        vecaddsubmod_ptr(work->tt1, work->tt2, Pout->X, Pout->Z, mdata);		        //U +/- V
        vecsqrmod_ptr(Pout->X, work->tt1, work->n, work->tt4, mdata);					//(U + V)^2
        vecsqrmod_ptr(Pout->Z, work->tt2, work->n, work->tt4, mdata);					//(U - V)^2

        // if gcd(j,D) != 1, Pout maps to scratch space (Pb[0])
        vecmulmod_ptr(work->tt1, P3->Z, Pout->X, work->n, work->tt4, mdata);			//Z * (U + V)^2
        vecmulmod_ptr(work->tt2, P3->X, Pout->Z, work->n, work->tt4, mdata);			//x * (U - V)^2

#ifndef DO_STAGE2_INV
        //store Pb[j].X * Pb[j].Z as well
        vecmulmod_ptr(Pout->X, Pout->Z, Pbprod[rprime_map_U[j]],
            work->n, work->tt4, mdata);
#endif

        work->ptadds++;

        // advance
        vecCopy(P1->X, P3->X);
        vecCopy(P1->Z, P3->Z);
        vecCopy(Pout->X, P1->X);
        vecCopy(Pout->Z, P1->Z);

        //sprintf(str, "Pb[%d].Z: ", rprime_map_U[j]);
        //print_vechex(Pout->Z->data, 0, NWORDS, str);
        if (verbose & (debug == 2))
            printf("rprime_map_U[%d] = %u\n", j, rprime_map_U[j]);
    }

    //printf("B table generated to umax = %d\n", U * D);

    // Pd = [w]Q
    vecCopy(P->Z, Pd->Z);
    vecCopy(P->X, Pd->X);
    next_pt_vec(mdata, work, Pd, wscale * w);

    if (verbose & (debug == 2))
        printf("Pd = [%u]Q\n", wscale * w);

    //first a value: first multiple of D greater than B1
    work->A = amin * w * 2;

    //initialize info needed for giant step
    vecCopy(P->Z, Pa[0].Z);
    vecCopy(P->X, Pa[0].X);
    next_pt_vec(mdata, work, &Pa[0], work->A);

    if (verbose & (debug == 2))
        printf("Pa[0] = [%lu]Q\n", work->A);

    vecCopy(P->Z, work->Pad->Z);
    vecCopy(P->X, work->Pad->X);
    next_pt_vec(mdata, work, work->Pad, work->A - wscale * w);

    if (verbose & (debug == 2))
        printf("Pad = [%lu]Q\n", work->A - wscale * w);

    vecaddmod_ptr(Pa[0].X, Pa[0].Z, work->sum1, mdata);
    vecaddmod_ptr(Pd->X, Pd->Z, work->sum2, mdata);
    vecsubmod_ptr(Pa[0].X, Pa[0].Z, work->diff1, mdata);
    vecsubmod_ptr(Pd->X, Pd->Z, work->diff2, mdata);
    vec_add(mdata, work, work->Pad, &Pa[1]);

    work->A += wscale * w;
    if (verbose & (debug == 2))
        printf("Pa[1] = [%lu]Q\n", work->A);

    for (i = 2; i < 2 * L; i++)
    {
        //giant step - use the addition formula for ECM
        //Pa + Pd
        //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
        //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
        //x- = [a-d]x
        //z- = [a-d]z
        vecaddsubmod_ptr(Pa[i - 1].X, Pa[i - 1].Z, work->sum1, work->diff1, mdata);
        vecaddsubmod_ptr(Pd->X, Pd->Z, work->sum2, work->diff2, mdata);
        vec_add(mdata, work, &Pa[i - 2], &Pa[i]);

#ifndef DO_STAGE2_INV
        vecmulmod_ptr(Pa[i].X, Pa[i].Z, work->Paprod[i], work->n, work->tt4, mdata);
#endif

        work->A += wscale * w;
        work->ptadds++;
        if (verbose & (debug == 2))
            printf("Pa[%d] = [%lu]Q\n", i, work->A);
    }

    // initialize accumulator
    vecCopy(mdata->one, acc);

#ifdef DO_STAGE2_INV
    // invert all of the Pb's
    foundDuringInv = batch_invert_pt_inplace(Pb, Pbprod, mdata, work, lastMapID + 1);

    // and invert all of the Pa's into a separate vector
    foundDuringInv |= batch_invert_pt_to_bignum(Pa, work->Pa_inv, Paprod, mdata, work, 0, 2 * L);

    if (doneIfFoundDuringInv && foundDuringInv)
    {
        work->last_pid = -1;

        mpz_clear(gmptmp);
        mpz_clear(gmptmp2);
        mpz_clear(gmpn);

        return foundDuringInv;
    }
#endif

    mpz_clear(gmptmp);
    mpz_clear(gmptmp2);
    mpz_clear(gmpn);

    if (verbose & (debug == 2))
        printf("A table generated to L = %d\n", 2 * L);

    return foundDuringInv;
}

void ecm_stage2_pair(uint32_t pairmap_steps, uint32_t *pairmap_v, uint32_t *pairmap_u,
    ecm_pt* P, monty* mdata, ecm_work* work, int verbose)
{
    // use the output of the PAIR algorithm to perform stage 2.
    uint32_t w = work->D;
    uint32_t U = work->U;
    uint32_t L = work->L;
    uint32_t umax = U * w;
    int i, pid;
    uint32_t amin = work->amin;
    int foundDuringInv;
    int mapid;
    uint8_t* flags;
    int testcoverage = 0;
    int wscale = 1;

    uint32_t* rprime_map_U = work->map;
    ecm_pt* Pa = work->Pa;      // non-inverted
    ecm_pt* Pb = work->Pb;      // inverted
    ecm_pt* Pd = work->Pdnorm;  // non-inverted Pd
#ifndef DO_STAGE2_INV
    bignum** Paprod = work->Paprod;
    bignum** Pbprod = work->Pbprod;
#endif
    bignum* acc = work->stg2acc;

    if (verbose == 1)
        printf("\n");

    if (verbose)
    {
        printf("commencing stage 2 at A=%u\n"
            "w = %u, R = %u, L = %u, U = %d, umax = %u, amin = %u\n",
            2 * amin * w, w, work->R - 3, L, U, umax, amin);
    }

    if (testcoverage)
        flags = (uint8_t*)calloc((100000 + STAGE2_MAX), sizeof(uint8_t));

    for (mapid = 0; mapid < pairmap_steps; mapid++)
    {
        int pa, pb;

        if ((verbose == 1) && ((mapid & 65535) == 0))
        {
            printf("pairmap step %u of %u\r", mapid, pairmap_steps);
            fflush(stdout);
        }

        if ((pairmap_u[mapid] == 0) && (pairmap_v[mapid] == 0))
        {
            int shiftdist = 2;

            // shift out uneeded A's.
            // update amin by U * 2 * w.
            // each point-add increments by w, so shift 2 * U times;
            for (i = 0; i < 2 * L - shiftdist * U; i++)
            {
                vecCopy(Pa[i + shiftdist * U].X, Pa[i].X);
                vecCopy(Pa[i + shiftdist * U].Z, Pa[i].Z);
                vecCopy(work->Pa_inv[i + shiftdist * U], work->Pa_inv[i]);
            }

            // make new A's: need at least two previous points;
            // therefore we can't have U = 1
            for (i = 2 * L - shiftdist * U; i < 2 * L; i++)
            {
                //giant step - use the addition formula for ECM
                //Pa + Pd
                //x+ = z- * [(x1-z1)(x2+z2) + (x1+z1)(x2-z2)]^2
                //z+ = x- * [(x1-z1)(x2+z2) - (x1+z1)(x2-z2)]^2
                //x- = [a-d]x
                //z- = [a-d]z
                vecaddsubmod_ptr(Pa[i - 1].X, Pa[i - 1].Z, work->sum1, work->diff1, mdata);
                vecaddsubmod_ptr(Pd->X, Pd->Z, work->sum2, work->diff2, mdata);
                vec_add(mdata, work, &Pa[i - 2], &Pa[i]);

#ifndef DO_STAGE2_INV
                vecmulmod_ptr(Pa[i].X, Pa[i].Z, work->Paprod[i], work->n, work->tt4, mdata);
#endif

                work->A += wscale * w;
                work->ptadds++;
            }

            // amin tracks the Pa[0] position in units of 2 * w, so
            // shifting by w, 2 * U times, is equivalent to shifting
            // by 2 * w, U times.
            amin += U;

#ifdef DO_STAGE2_INV
            foundDuringInv = batch_invert_pt_to_bignum(Pa, work->Pa_inv, 
                work->Paprod, mdata, work, 2 * L - shiftdist * U, 2 * L);
#endif
        }
        else
        {
            pa = pairmap_v[mapid] - amin;
            pb = pairmap_u[mapid];

            //printf("amin = %u, acc %d(%d)w +/- %d => %lu\n", amin, pa + amin, pa, pb,
            //    ((2 * amin + pa) * w - pb));

            if (pa >= 2 * L)
            {
                printf("error: invalid A offset: %d,%d,%u\n", pa, pb, amin);
                exit(1);
            }

            if (testcoverage)
            {
                addflag(flags, (2 * amin + pa) * w - pb);
                addflag(flags, (2 * amin + pa) * w + pb);
            }

            //if ((((2 * amin + pa) * w - pb) == 37365437) ||
            //    (((2 * amin + pa) * w + pb) == 37365437))
            //{
            //    printf("\naccumulating amin=%u,pa=%d,pb=%d, p-=%u, p+=%u\n", amin, pa, pb,
            //        ((2 * amin + pa) * w - pb), ((2 * amin + pa) * w + pb));
            //}

            if (rprime_map_U[pb] == 0)
            {
                printf("pb=%d doesn't exist\n", pb);
            }
            
#ifdef DO_STAGE2_INV
            CROSS_PRODUCT_INV;
#else
            CROSS_PRODUCT;
#endif
            work->paired++;
        }
    }

    if (testcoverage)
    {
        pid = 0;
        while (PRIMES[pid] < STAGE1_MAX) { pid++; }

        int notcovered = 0;
        while ((pid < NUM_P) && (PRIMES[pid] < STAGE2_MAX))
        {
            if (flags[PRIMES[pid]] != 1)
            {
                //printf("prime %lu not covered!\n", PRIMES[pid]);
                notcovered++;
            }
            pid++;
        }
        printf("%d primes not covered during pairing!\n", notcovered);
    }

    pid = NUM_P;

    work->amin = amin;
    work->last_pid = pid;

    return;
}

int check_factor(mpz_t Z, mpz_t n, mpz_t f)
{
    //gmp_printf("checking point Z = %Zx against input N = %Zx\n", Z, n);
    mpz_gcd(f, Z, n);

	if (mpz_cmp_ui(f, 1) > 0)
	{
		if (mpz_cmp(f, n) == 0)
		{
            mpz_set_ui(f, 0);
			return 0;
		}
		return 1;
	}
	return 0;
}

uint32_t pair(uint32_t *pairmap_v, uint32_t *pairmap_u, 
    ecm_work* work, uint64_t* primes, uint64_t B1, uint64_t B2, int verbose)
{
    int i, j, pid = 0;
    int w = work->D;
    int U = work->U;
    int L = work->L;
    int R = work->R - 3;
    int umax = w * U;
    int q, mq;
    uint64_t amin = (B1 + w) / (2 * w);
    uint64_t a, s, ap, u;
    uint32_t pairs = 0;
    uint32_t nump = 0;
    uint32_t mapid = 0;
    uint8_t* flags;
    int printpairs = 0;
    int testcoverage = 0;

    // gives an index of a queue given a residue mod w
    //printf("Qmap: \n");
    //for (i = 0; i < w; i++)
    //{
    //    printf("%u\n", work->Qmap[i]);
    //}

    // contains the value of q given an index
    //printf("Qrmap: \n");
    //for (i = 0; i < w; i++)
    //{
    //    printf("%u\n", work->Qrmap[i]);
    //}

    //printf("w = %u\n", w);
    //printf("U = %u\n", U);
    //printf("L = %u\n", L);
    //printf("R = %u\n", R);
    //printf("amin = %u\n", amin);
    //printf("umax = %u\n", umax);

    if (testcoverage)
    {
        flags = (uint8_t*)calloc((10000 + B2), sizeof(uint8_t));
    }

    if (verbose)
    {
        printf("commencing pair on range %lu:%lu\n", B1, B2);
    }
    
    while (primes[pid] < B1) { pid++; }

    while ((pid < NUM_P) && (primes[pid] < B2))
    {
        s = primes[pid];
        a = (s + w) / (2 * w);
        nump++;

        //printf("s, a: %lu, %lu\n", s, a);

        while (a >= (amin + L))
        {
            int oldmin = amin;
            amin = amin + L - U;
            //printf("amin now %u\n", amin);

            for (i = 0; i < R; i++)
            {
                int len = work->Q[i]->len;

                if (work->Qrmap[i] > w)
                {
                    q = 2 * w - work->Qrmap[i];
                    for (j = 0; j < len; j++)
                    {
                        ap = dequeue(work->Q[i]);
                        if ((uint32_t)ap < amin)
                        {
                            pairmap_v[mapid] = 2 * ap - oldmin; // 2 * ap; //2 * (ap - oldmin);
                            pairmap_u[mapid] = q;
                            mapid++;

                            if (testcoverage)
                            {
                                addflag(flags, (2 * ap) * w + q);
                                addflag(flags, (2 * ap) * w - q);
                            }
                            if (printpairs)
                            {
                                printf("pair (ap,q):(%lu,%d)  %lu:%lu\n",
                                    ap, q,
                                    2 * ap * w - q,
                                    2 * ap * w + q);
                            }
                            pairs++;
                        }
                        else
                        {
                            enqueue(work->Q[i], ap);
                        }
                    }
                }
                else
                {
                    for (j = 0; j < len; j++)
                    {
                        ap = dequeue(work->Q[i]);
                        if ((uint32_t)ap < amin)
                        {
                            pairmap_v[mapid] = 2 * ap - oldmin; //2 * ap; //2 * (ap - oldmin);
                            pairmap_u[mapid] = work->Qrmap[i];
                            mapid++;

                            if (testcoverage)
                            {
                                addflag(flags, (2 * ap) * w + work->Qrmap[i]);
                                addflag(flags, (2 * ap) * w - work->Qrmap[i]);
                            }
                            if (printpairs)
                            {
                                printf("pair (ap,q):(%lu,%d)  %lu:%lu\n",
                                    ap, work->Qrmap[i],
                                    2 * ap * w - work->Qrmap[i],
                                    2 * ap * w + work->Qrmap[i]);
                            }
                            pairs++;
                        }
                        else
                        {
                            enqueue(work->Q[i], ap);
                        }
                    }
                }
            }
            pairmap_u[mapid] = 0;
            pairmap_v[mapid] = 0;
            mapid++;
        }

        q = s - 2 * a * w;
        if (q < 0)
            mq = abs(q);
        else
            mq = 2 * w - q;

       // printf("q, mq: %d, %d\n", q, mq);

        do
        {
            if (work->Q[work->Qmap[mq]]->len > 0)
            {
                ap = dequeue(work->Q[work->Qmap[mq]]);
                if (q < 0)
                    u = w * (a - ap) - abs(q);
                else
                    u = w * (a - ap) + q;

                if (u > umax)
                {
                    if (q < 0)
                    {
                        int qq = abs(q);

                        pairmap_v[mapid] = 2 * ap - amin; //2 * ap; //2 * (ap - amin);
                        pairmap_u[mapid] = qq;
                        mapid++;

                        if (testcoverage)
                        {
                            addflag(flags, (2 * ap) * w + qq);
                            addflag(flags, (2 * ap) * w - qq);
                        }
                        if (printpairs)
                        {
                            printf("pair (ap,q):(%lu,%d)  %lu:%lu\n",
                                ap, q,
                                2 * ap * w - qq,
                                2 * ap * w + qq);
                        }
                    }
                    else
                    {
                        int qq = q;

                        if (qq >= w)
                            qq = 2 * w - qq;

                        pairmap_v[mapid] = 2 * ap - amin; //2 * ap; //2 * (ap - amin);
                        pairmap_u[mapid] = qq;
                        mapid++;

                        if (testcoverage)
                        {
                            addflag(flags, (2 * ap) * w + qq);
                            addflag(flags, (2 * ap) * w - qq);
                        }
                        if (printpairs)
                        {
                            printf("pair (ap,q):(%lu,%d)  %lu:%lu\n",
                                ap, q,
                                2 * ap * w - qq,
                                2 * ap * w + qq);
                        }
                    }
                    pairs++;
                }
                else
                {
                    pairmap_v[mapid] = a + ap - amin;
                    pairmap_u[mapid] = u;
                    mapid++;

                    if (testcoverage)
                    {
                        addflag(flags, (a + ap) * w + u);
                        addflag(flags, (a + ap) * w - u);
                    }
                    if (printpairs)
                    {
                        printf("pair (a,ap,u):(%lu,%lu,%lu)  %lu:%lu\n",
                            a, ap, u,
                            (a + ap) * w - u,
                            (a + ap) * w + u);
                    }
                    pairs++;
                }
            }
            else
            {
                //printf("queueing a=%lu in Q[%d]\n", a, abs(q));
                if (q < 0)
                {
                    //printf("queueing a=%lu in Q[%u](%u)\n", a, 2 * w + q, work->Qmap[2 * w + q]);
                    enqueue(work->Q[work->Qmap[2 * w + q]], a);
                }
                else
                {
                    //printf("queueing a=%lu in Q[%d]\n", a, q);
                    enqueue(work->Q[work->Qmap[q]], a);
                }
                u = 0;
            }
        } while (u > umax);

        pid++;
    }

    //printf("dumping leftovers in queues\n");
    // empty queues
    for (i = 0; i < R; i++)
    {
        //printf("queue %d (%u) has %d elements\n", i, work->Qrmap[i], work->Q[i]->len);
        int len = work->Q[i]->len;
        for (j = 0; j < len; j++)
        {
            ap = dequeue(work->Q[i]);
            if (work->Qrmap[i] > w)
            {
                q = 2 * w - work->Qrmap[i];

                pairmap_v[mapid] = 2 * ap - amin; //2 * ap; //2 * (ap - amin);
                pairmap_u[mapid] = q;
                mapid++;

                if (printpairs)
                {
                    printf("pair (ap,q):(%lu,%d)  %lu:%lu\n",
                        ap, q,
                        2 * ap * w - q,
                        2 * ap * w + q);
                }
                if (testcoverage)
                {
                    addflag(flags, (2 * ap) * w + q);
                    addflag(flags, (2 * ap) * w - q);
                }
            }
            else
            {
                pairmap_v[mapid] = 2 * ap - amin; //2 * ap; // 2 * (ap - amin);
                pairmap_u[mapid] = work->Qrmap[i];
                mapid++;

                if (printpairs)
                {
                    printf("pair (ap,q):(%lu,%d)  %lu:%lu\n",
                        ap, work->Qrmap[i],
                        2 * ap * w - work->Qrmap[i],
                        2 * ap * w + work->Qrmap[i]);
                }
                if (testcoverage)
                {
                    addflag(flags, (2 * ap) * w + work->Qrmap[i]);
                    addflag(flags, (2 * ap) * w - work->Qrmap[i]);
                }
            }
            pairs++;
        }
    }

    //printf("%u pairing steps generated\n", mapid);
    //amin = (B1 + w) / (2 * w);
    //printf("amin is now %lu (A = %lu)\n", amin, 2 * amin * w);
    //for (i = 0; i < mapid; i++)
    //{
    //    printf("pair: %uw+/-%u => %lu:%lu\n", pairmap_v[i], pairmap_u[i],
    //        (amin + pairmap_v[i]) * w - pairmap_u[i],
    //        (amin + pairmap_v[i]) * w + pairmap_u[i]);
    //    if (pairmap_u[i] == 0)
    //    {
    //        amin = amin + L - U;
    //        printf("amin is now %u (A = %u)\n", amin, 2 * amin * w);
    //    }
    //}

    if (testcoverage)
    {
        pid = 0;
        while (primes[pid] < B1) { pid++; }

        int notcovered = 0;
        while ((pid < NUM_P) && (primes[pid] < B2))
        {
            if (flags[primes[pid]] != 1)
            {
                printf("prime %lu not covered!\n", primes[pid]);
                notcovered++;
            }
            pid++;
        }
        printf("%d primes not covered during pairing!\n", notcovered);
        free(flags);
    }

    if (verbose)
    {
        printf("%u pairs found from %u primes (ratio = %1.2f)\n",
            pairs, nump, (double)pairs / (double)nump);
    }

    //exit(0);
    return mapid;
}

