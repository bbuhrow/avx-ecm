/*----------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Ben Buhrow. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

       				   --bbuhrow@gmail.com 7/1/10
----------------------------------------------------------------------*/

#include "soe.h"
#include "threadpool.h"

void sieve_sync(void *vptr)
{
    tpool_t *tdata = (tpool_t *)vptr;
    soe_userdata_t *udata = (soe_userdata_t *)tdata->user_data;
    soe_staticdata_t *sdata = udata->sdata;
    thread_soedata_t *t = &udata->ddata[tdata->tindex];

    if (SOE_VFLAG > 1)
    {
        //don't print status if computing primes, because lots of routines within
        //yafu do this and they don't want this side effect
        printf("sieving: %d%%\r", 
            (int)((double)sdata->sync_count / (double)(sdata->numclasses)* 100.0));
        fflush(stdout);
    }

#ifndef INPLACE_BUCKET
    if (sdata->only_count)
    {
        sdata->num_found += t->linecount;
    }

    if (t->ddata.min_sieved_val < sdata->min_sieved_val)
    {
        sdata->min_sieved_val = t->ddata.min_sieved_val;
    }
#endif

    return;
}

void sieve_dispatch(void *vptr)
{
    tpool_t *tdata = (tpool_t *)vptr;
    soe_userdata_t *udata = (soe_userdata_t *)tdata->user_data;
    soe_staticdata_t *sdata = udata->sdata;
    thread_soedata_t *t = &udata->ddata[tdata->tindex];

    // if not done, dispatch another line for sieving
    if (sdata->sync_count < sdata->numclasses)
    {
        t->current_line = (uint32_t)sdata->sync_count;
        tdata->work_fcn_id = 0;
        sdata->sync_count++;
    }
    else
    {
        tdata->work_fcn_id = tdata->num_work_fcn;
    }
    
    return;
}

void sieve_work_fcn(void *vptr)
{
    tpool_t *tdata = (tpool_t *)vptr;
    soe_userdata_t *udata = (soe_userdata_t *)tdata->user_data;
    thread_soedata_t *t = &udata->ddata[tdata->tindex];
	soe_staticdata_t *sdata = &t->sdata;

    if (sdata->only_count)
    {
		sdata->lines[t->current_line] =
            (uint8_t *)xmalloc_align(sdata->numlinebytes * sizeof(uint8_t));
        sieve_line(t);
        t->linecount = count_line(&t->sdata, t->current_line);
        align_free(t->sdata.lines[t->current_line]);
    }
    else
    {
        sieve_line(t);
    }

    return;
}



uint32_t modinv_1(uint32_t a, uint32_t p) {

    /* thanks to the folks at www.mersenneforum.org */

    uint32_t ps1, ps2, parity, dividend, divisor, rem, q, t;

    q = 1;
    rem = a;
    dividend = p;
    divisor = a;
    ps1 = 1;
    ps2 = 0;
    parity = 0;

    while (divisor > 1) {
        rem = dividend - divisor;
        t = rem - divisor;
        if (rem >= divisor) {
            q += ps1; rem = t; t -= divisor;
            if (rem >= divisor) {
                q += ps1; rem = t; t -= divisor;
                if (rem >= divisor) {
                    q += ps1; rem = t; t -= divisor;
                    if (rem >= divisor) {
                        q += ps1; rem = t; t -= divisor;
                        if (rem >= divisor) {
                            q += ps1; rem = t; t -= divisor;
                            if (rem >= divisor) {
                                q += ps1; rem = t; t -= divisor;
                                if (rem >= divisor) {
                                    q += ps1; rem = t; t -= divisor;
                                    if (rem >= divisor) {
                                        q += ps1; rem = t;
                                        if (rem >= divisor) {
                                            q = dividend / divisor;
                                            rem = dividend % divisor;
                                            q *= ps1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        q += ps2;
        parity = ~parity;
        dividend = divisor;
        divisor = rem;
        ps2 = ps1;
        ps1 = q;
    }

    if (parity == 0)
        return ps1;
    else
        return p - ps1;
}

uint32_t modinv_1c(uint32_t a, uint32_t p) {

    /* thanks to the folks at www.mersenneforum.org */
    // for use when it is known that p >> a, in which case
    // the first set of if/else blocks can be skipped
    uint32_t ps1, ps2, parity, dividend, divisor, rem, q, t;

    q = p / a;
    rem = p % a;
    dividend = a;
    divisor = rem;
    ps1 = q;
    ps2 = 1;
    parity = ~0;

    while (divisor > 1) {
        rem = dividend - divisor;
        t = rem - divisor;
        if (rem >= divisor) {
            q += ps1; rem = t; t -= divisor;
            if (rem >= divisor) {
                q += ps1; rem = t; t -= divisor;
                if (rem >= divisor) {
                    q += ps1; rem = t; t -= divisor;
                    if (rem >= divisor) {
                        q += ps1; rem = t; t -= divisor;
                        if (rem >= divisor) {
                            q += ps1; rem = t; t -= divisor;
                            if (rem >= divisor) {
                                q += ps1; rem = t; t -= divisor;
                                if (rem >= divisor) {
                                    q += ps1; rem = t; t -= divisor;
                                    if (rem >= divisor) {
                                        q += ps1; rem = t;
                                        if (rem >= divisor) {
                                            q = dividend / divisor;
                                            rem = dividend % divisor;
                                            q *= ps1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        q += ps2;
        parity = ~parity;
        dividend = divisor;
        divisor = rem;
        ps2 = ps1;
        ps1 = q;
    }

    if (parity == 0)
        return ps1;
    else
        return p - ps1;
}



uint64_t spSOE(uint32_t *sieve_p, uint32_t num_sp, 
	uint64_t lowlimit, uint64_t *highlimit, int count, uint64_t *primes)
{
	/*
	if count == 1, then the primes are simply counted, and not 
	explicitly calculated and saved in *primes.

	otherwise, store primes in the provided *primes array

	in either case, return the number of primes found
	*/

	//keep track of how much memory we've used
	uint64_t allocated_bytes = 0;

	//structure of static info
	soe_staticdata_t sdata;

	//thread data holds all data needed during sieving
	thread_soedata_t *thread_data;		//an array of thread data objects

	//*********************** BEGIN ******************************//
    //SOE_VFLAG = 3;

	//sanity check the input
	sdata.only_count = count;
	if (check_input(*highlimit, lowlimit, num_sp, sieve_p, &sdata))
		return 0;

	//determine what kind of sieve to use based on the input
	get_numclasses(*highlimit, lowlimit, &sdata);

	//allocate and initialize some stuff
	allocated_bytes += init_sieve(&sdata);
	*highlimit = sdata.highlimit;
	
	//allocate thread data structure
	thread_data = (thread_soedata_t *)malloc(SOE_THREADS * sizeof(thread_soedata_t));

	//find all roots of prime with prodN.  These are used when finding offsets.
	getRoots(&sdata, thread_data);

	//init bucket sieving
	set_bucket_depth(&sdata);

	//initialize stuff used in thread structures.
	//this is necessary even if SOE_THREADS = 1;	
	allocated_bytes += alloc_threaddata(&sdata, thread_data);

	if (SOE_VFLAG > 2)
	{	
		printf("finding requested range %lu to %lu\n",sdata.orig_llimit,sdata.orig_hlimit);
		printf("sieving range %lu to %lu\n",lowlimit,*highlimit);
		printf("using %lu primes, max prime = %lu  \n", sdata.pboundi, sieve_p[sdata.pboundi]); // sdata.pbound);
		printf("using %u residue classes\n",sdata.numclasses);
		printf("lines have %lu bytes and %lu flags\n",sdata.numlinebytes,sdata.numlinebytes * 8);
		printf("lines broken into = %lu blocks of size %u\n",sdata.blocks,BLOCKSIZE);
		printf("blocks contain %u flags and cover %lu primes\n", FLAGSIZE, sdata.blk_r);
		if (sdata.num_bucket_primes > 0)
		{
			printf("bucket sieving %u primes > %u\n",
				sdata.num_bucket_primes,sdata.sieve_p[sdata.bucket_start_id]);
			printf("allocating space for %u hits per bucket\n",sdata.bucket_alloc);
			printf("allocating space for %u hits per large bucket\n",sdata.large_bucket_alloc);
		}
		if (sdata.num_inplace_primes > 0)
		{
			printf("inplace sieving %u primes > %u\n",
				sdata.num_inplace_primes,sdata.sieve_p[sdata.inplace_start_id]);
		}
		printf("using %lu bytes for sieving storage\n",allocated_bytes);
	}

	//get 'r done.
	do_soe_sieving(&sdata, thread_data, count);

	//finish up
	finalize_sieve(&sdata, thread_data, count, primes);

	return sdata.num_found;
}

void do_soe_sieving(soe_staticdata_t *sdata, thread_soedata_t *thread_data, int count)
{
	uint64_t i;

    // threading structures
    tpool_t *tpool_data;
    soe_userdata_t udata;

	//main sieve, line by line
    sdata->num_found = 0;
    sdata->only_count = count;

    udata.sdata = sdata;
    udata.ddata = thread_data;
    tpool_data = tpool_setup(SOE_THREADS, NULL, NULL, &sieve_sync,
        &sieve_dispatch, &udata);

    if (SOE_THREADS == 1)
    {
        thread_soedata_t *t = &thread_data[0];
        sdata->sync_count = 0;
        for (i = 0; i < sdata->numclasses; i++)
        {
            t->current_line = i;
            sieve_work_fcn(tpool_data);
            sieve_sync(tpool_data);
            sdata->sync_count++;
        }
    }
    else
    {
        sdata->sync_count = 0;
        tpool_add_work_fcn(tpool_data, &sieve_work_fcn);
        tpool_go(tpool_data);
    }
    
    free(tpool_data);

    // to test: make this a stop fcn
    for (i=0; i<SOE_THREADS; i++)
    {
        align_free(thread_data[i].ddata.offsets);
    }

	return;
}

void finalize_sieve(soe_staticdata_t *sdata, 
	thread_soedata_t *thread_data, int count, uint64_t *primes)
{
	uint64_t i, j = 0, num_p = sdata->num_found;

	//printf("min sieved value = %lu\n",sdata->min_sieved_val);

	if (count)
	{
		//add in relevant sieving primes not captured in the flag arrays
		uint64_t ui_offset;

		ui_offset = 0;
		
		if (sdata->sieve_range)
			sdata->min_sieved_val += ui_offset;

		//printf("lowlimit is %lu first sieved value = %lu\n", 
		//	sdata->lowlimit, sdata->min_sieved_val);
		//printf("original limits are %lu and %lu\n", 
		//	sdata->orig_llimit, sdata->orig_hlimit);

		//PRIMES is already sized appropriately by the wrapper
		//load in the sieve primes that we need
		i = 0;
		while (((uint64_t)sdata->sieve_p[i] < sdata->min_sieved_val) && (i < sdata->bucket_start_id))
		{
			if (sdata->sieve_p[i] >= (sdata->orig_llimit + ui_offset))		
				num_p++;
			i++;
		}
		//printf("added %u primes\n", (uint32_t)(num_p - sdata->num_found));
	}
	else
	{
		//now we need to raster vertically down the lines and horizontally
		//across the lines in order to compute the primes in order.

		//first put in any sieve primes if necessary.
		//if we are in this loop, and we are sieving a range, then offset
		//is a single precision number and we need to increment the 'prime'
		//we found above by it.
		uint64_t ui_offset;
			
		ui_offset = 0;
			
		if (sdata->sieve_range)
			sdata->min_sieved_val += ui_offset;

		//printf("lowlimit is %lu first sieved value = %lu\n", 
		//	sdata->lowlimit, sdata->min_sieved_val);
		//printf("original limits are %lu and %lu\n", 
		//	sdata->orig_llimit, sdata->orig_hlimit);

		//PRIMES is already sized appropriately by the wrapper
		//load in the sieve primes that we need
		j = 0;
		i = 0;
		while (((uint64_t)sdata->sieve_p[i] < sdata->min_sieved_val) && (i < sdata->bucket_start_id))
		{
			if (sdata->sieve_p[i] >= (sdata->orig_llimit + ui_offset))					
				primes[j++] = (uint64_t)sdata->sieve_p[i];
			i++;
		}
		//printf("added %u primes\n", (uint32_t)j);

		//and then the primes in the lines
		num_p = primes_from_lineflags(sdata, thread_data, j, primes);

	}

	//update count of found primes
	sdata->num_found = num_p;

	for (i=0; i<SOE_THREADS; i++)
	{
		thread_soedata_t *thread = thread_data + i;
		free(thread->ddata.pbounds);
        align_free(thread->ddata.presieve_scratch);
	}

	if (sdata->num_bucket_primes > 0)
	{
		for (i=0; i< SOE_THREADS; i++)
		{
			thread_soedata_t *thread = thread_data + i;

			free(thread->ddata.bucket_hits);
			if (thread->ddata.large_bucket_hits != NULL)
				free(thread->ddata.large_bucket_hits);
			for (j=0; j < thread->sdata.blocks; j++)
			{
				free(thread->ddata.sieve_buckets[j]);
				if (thread->ddata.large_sieve_buckets != NULL)
					free(thread->ddata.large_sieve_buckets[j]);
			}
			free(thread->ddata.sieve_buckets);
			if (thread->ddata.large_sieve_buckets != NULL)
				free(thread->ddata.large_sieve_buckets);            
		}

	}

	if (!sdata->only_count)
	{
		for (i = 0; i<sdata->numclasses; i++)
			align_free(sdata->lines[i]);
		//align_free(sdata->lines[0]);
		//align_free(sdata->lines);
	}
    align_free(sdata->lines);
    free(sdata->root);
	free(sdata->lower_mod_prime);
	free(thread_data);
	free(sdata->rclass);

#if defined(INPLACE_BUCKET)
	if (sdata->num_inplace_primes > 0)
	{	
		free(sdata->inplace_data);
		for (i=0; i<sdata->numclasses; i++)
			free(sdata->inplace_ptrs[i]);
		free(sdata->inplace_ptrs);
	}
#endif

	return;
}

