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

uint64_t *GetPRIMESRange(uint32_t *sieve_p, uint32_t num_sp, 
	uint64_t lowlimit, uint64_t highlimit, uint64_t *num_p)
{
	uint64_t i;
	uint64_t hi_est, lo_est;
	uint64_t maxrange = 10000000000ULL;
	uint64_t *primes = NULL;
	
	// reallocate output array based on conservative estimate of the number of 
	// primes in the interval
	hi_est = (uint64_t)(highlimit/log((double)highlimit));
	if (lowlimit > 1)
		lo_est = (uint64_t)(lowlimit/log((double)lowlimit));
	else
		lo_est = 0;

	i = (uint64_t)((double)(hi_est - lo_est) * 1.25);

	primes = (uint64_t *)realloc(primes,(size_t) (i * sizeof(uint64_t)));
	if (primes == NULL)
	{
		printf("unable to allocate %lu bytes for range %lu to %lu\n",
			(uint64_t)(i * sizeof(uint64_t)),lowlimit,highlimit);
		exit(1);
	}

	// check for really big ranges ('big' is different here than when we are counting
	// primes because there are higher memory demands when computing primes)
	if ((highlimit - lowlimit) > maxrange)
	{
        printf("range too big\n");
        exit(1);
	}
	else
	{
		//find the primes in the interval
		GLOBAL_OFFSET = 0;
		*num_p = spSOE(sieve_p, num_sp, lowlimit, &highlimit, 0, primes);
	}

	return primes;
}

uint64_t *soe_wrapper(uint32_t *seed_p, uint32_t num_sp, 
	uint64_t lowlimit, uint64_t highlimit, int count, uint64_t *num_p)
{
	//public interface to the sieve.  
	uint64_t retval, tmpl, tmph, i;
	uint32_t max_p;	
	uint32_t *sieve_p;
	uint64_t *primes = NULL;

    PRIMES_TO_FILE = 0;
	PRIMES_TO_SCREEN = 0;

	if (highlimit < lowlimit)
	{
		printf("error: lowlimit must be less than highlimit\n");
		*num_p = 0;
		return primes;
	}	

	if (highlimit > (seed_p[num_sp-1] * seed_p[num_sp-1]))
	{
		//then we need to generate more sieving primes
		uint32_t range_est;
	
		//allocate array based on conservative estimate of the number of 
		//primes in the interval	
		max_p = (uint32_t)sqrt((int64_t)(highlimit)) + 65536;
		range_est = (uint32_t)estimate_primes_in_range(0, (uint64_t)max_p);
		sieve_p = (uint32_t *)xmalloc_align((size_t) (range_est * sizeof(uint32_t)));

		if (sieve_p == NULL)
		{
			printf("unable to allocate %u bytes for %u sieving primes\n",
				range_est * (uint32_t)sizeof(uint32_t), range_est);
			exit(1);
		}

		//find the sieving primes using the seed primes
		NO_STORE = 0;
		primes = GetPRIMESRange(seed_p, num_sp, 0, max_p, &retval);
        for (i = 0; i < retval; i++)
        {
            sieve_p[i] = (uint32_t)primes[i];
        }
		printf("found %u sieving primes\n",(uint32_t)retval);
		num_sp = (uint32_t)retval;
		free(primes);
		primes = NULL;
	}
	else
	{
		//seed primes are enough
        sieve_p = (uint32_t *)xmalloc_align((size_t)(num_sp * sizeof(uint32_t)));

		if (sieve_p == NULL)
		{
			printf("unable to allocate %u bytes for %u sieving primes\n",
				num_sp * (uint32_t)sizeof(uint32_t), num_sp);
			exit(1);
		}

		for (i=0; i<num_sp; i++)
			sieve_p[i] = seed_p[i];
	}

	if (count)
	{
		//this needs to be a range of at least 1e6
		if ((highlimit - lowlimit) < 1000000)
		{
			//go and get a new range.
			tmpl = lowlimit;
			tmph = tmpl + 1000000;

			//since this is a small range, we need to 
			//find a bigger range and count them.
			primes = GetPRIMESRange(sieve_p, num_sp, tmpl, tmph, &retval);

			*num_p = 0;
			//count how many are in the original range of interest
			for (i = 0; i < retval; i++)
			{
				if (primes[i] >= lowlimit && primes[i] <= highlimit)
					(*num_p)++;
			}
			free(primes);
			primes = NULL;
		}
		else
		{
			//check for really big ranges
			uint64_t maxrange = 100000000000ULL;

			if ((highlimit - lowlimit) > maxrange)
			{
				uint32_t num_ranges = (uint32_t)((highlimit - lowlimit) / maxrange);
				uint64_t remainder = (highlimit - lowlimit) % maxrange;
				uint32_t j;
				//to get time per range
				double t_time;
				struct timeval start, stop;
				
				*num_p = 0;
				tmpl = lowlimit;
				tmph = lowlimit + maxrange;
				gettimeofday (&start, NULL);

				for (j = 0; j < num_ranges; j++)
				{
					*num_p += spSOE(sieve_p, num_sp, tmpl, &tmph, 1, NULL);

					gettimeofday (&stop, NULL);
                    t_time = my_difftime(&start, &stop);

					if (SOE_VFLAG > 1)
						printf("so far, found %lu primes in %1.1f seconds\n",*num_p, t_time);
					tmpl += maxrange;
					tmph = tmpl + maxrange;
				}
				
				if (remainder > 0)
				{
					tmph = tmpl + remainder;
					*num_p += spSOE(sieve_p, num_sp, tmpl, &tmph, 1, NULL);
				}
				if (SOE_VFLAG > 1)
					printf("so far, found %lu primes\n",*num_p);
			}
			else
			{
				//we're in a sweet spot already, just get the requested range
				*num_p = spSOE(sieve_p, num_sp, lowlimit, &highlimit, 1, NULL);
			}
		}

	}
	else
	{
		tmpl = lowlimit;
		tmph = highlimit;

		//this needs to be a range of at least 1e6
		if ((tmph - tmpl) < 1000000)
		{
			//there is slack built into the sieve limit, so go ahead and increase
			//the size of the interval to make it at least 1e6.
			tmph = tmpl + 1000000;

			//since this is a small range, we need to 
			//find a bigger range and count them.
			primes = GetPRIMESRange(sieve_p, num_sp, tmpl, tmph, &retval);
			*num_p = 0;
			for (i = 0; i < retval; i++)
			{
				if (primes[i] >= lowlimit && primes[i] <= highlimit)
					(*num_p)++;
			}

		}
		else
		{
			//we don't need to mess with the requested range,
			//so GetPRIMESRange will return the requested range directly
			//and the count will be in NUM_P
			primes = GetPRIMESRange(sieve_p, num_sp, lowlimit, highlimit, num_p);
		}	
	}

	align_free(sieve_p);
	return primes;
}

