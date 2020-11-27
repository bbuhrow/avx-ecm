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


void get_offsets(thread_soedata_t *thread_data)
{
	//extract stuff from the thread data structure
	soe_dynamicdata_t *ddata = &thread_data->ddata;
	soe_staticdata_t *sdata = &thread_data->sdata;

	uint64_t i,startprime = sdata->startprime, prodN = sdata->prodN, block=0;
	uint32_t prime, root, bnum;
	uint32_t diff = sdata->rclass[thread_data->current_line] - 1;
	uint64_t tmp2;
	int s;

	// failsafe: set all blocks to sieve with all primes.  the loop below will overwrite
	// these with better limits according to the size of flags in the blocks.
	ddata->largep_offset = 0;

	for (i=0; i<sdata->blocks; i++)
	{		
		ddata->pbounds[i] = sdata->bucket_start_id;
		
		//initialize bucket
		if (ddata->bucket_depth > 0)
			ddata->bucket_hits[i] = 0;
	}

	if (sdata->sieve_range == 0)
	{
        for (i = startprime; i < sdata->bucket_start_id; i++)
        {
            prime = sdata->sieve_p[i];

            // find the first multiple of the prime which is greater than the first sieve location 
            // and also equal to the residue class mod 'prodN'.  
            // we need to solve the congruence: rclass[current_line] == kp mod prodN for k
            // xGCD gives r and s such that r*p + s*prodN = gcd(p,prodN).
            // then k = r*class/gcd(p,prodN) is a solution.
            // the gcd of p and prodN is always 1 by construction of prodN and choice of p.  
            // therefore k = r * class is a solution.  furthermore, since the gcd is 1, there
            // is only one solution.  
            // xGCD_1((int)prime,(int)prodN,&r,&s,&tmp);
               
            // To speed things up we solve and store modinv(prodN, prime) for every prime (only
            // needs to be done once, in roots.c).  Then to get the offset for the current block
            // we just need to multiply the stored root with the starting sieve location (mod p).	
               
            // if the prime is greater than the limit at which it is necessary to sieve
            // a block, start that prime in the next block.
            if (sdata->sieve_p[i] > ddata->blk_b_sqrt)
            {
                ddata->pbounds[block] = i;
                if (block < sdata->blocks)
                {
                    block++;
                }
                ddata->lblk_b = ddata->ublk_b + prodN;
                ddata->ublk_b += sdata->blk_r;
                ddata->blk_b_sqrt = (uint64_t)(sqrt((int64_t)(ddata->ublk_b + prodN))) + 1;
            }

            s = sdata->root[i];

            // the lower block bound (lblk_b) times s can exceed 64 bits for large ranges,
            // so reduce mod p here as well.
            tmp2 = (uint64_t)s * (ddata->lblk_b % (uint64_t)prime);
                
            // tmp2 = (uint64_t)s * (uint64_t)(lmp[i] + diff);
            ddata->offsets[i] = (uint32_t)(tmp2 % (uint64_t)prime);
        }
	}
	else
	{
		uint32_t modp;
		mpz_t lowz, sqrtz;
		mpz_init(lowz);
		mpz_init(sqrtz);
		mpz_set(lowz, *sdata->offset);
		mpz_add_ui(lowz, lowz, ddata->lblk_b);

		mpz_set(sqrtz, lowz);
		mpz_add_ui(sqrtz, sqrtz, sdata->blk_r);
		mpz_sqrt(sqrtz, sqrtz);
		mpz_add_ui(sqrtz, sqrtz, 1);
		//mpz_set_ui(tmpz, ddata->lblk_b);

		// if we're sieving with an offset, use all of the primes for each block
		// and just find the offset into the first block
		for (i=startprime;i<sdata->bucket_start_id;i++)
		{
			prime = sdata->sieve_p[i];
			s = sdata->root[i];

			if (mpz_cmp_ui(sqrtz, sdata->sieve_p[i]) <= 0)
			{
				ddata->pbounds[block] = i;
				block++;
				mpz_add_ui(lowz, lowz, sdata->blk_r);
				mpz_set(sqrtz, lowz);
				mpz_add_ui(sqrtz, sqrtz, sdata->blk_r);
				mpz_sqrt(sqrtz, sqrtz);
				mpz_add_ui(sqrtz, sqrtz, 1);
			}

			modp = mpz_tdiv_ui(lowz, prime);
			tmp2 =  (uint64_t)s * (uint64_t)modp;
			ddata->offsets[i] = (uint32_t)(tmp2 % (uint64_t)prime);
			//gmp_printf("p = %u, o = %u, r = %d, lblk_b = %Zd, modp = %u\n", 
			//	prime, ddata->offsets[i], s, tmpz, modp);
		}

		mpz_clear(lowz);
		mpz_clear(sqrtz);
	}

	if (ddata->bucket_depth > 0)
	{
        uint64_t **bptr;

		uint32_t *nptr;
		uint32_t linesize = FLAGSIZE * sdata->blocks;
        uint32_t *lmp = sdata->lower_mod_prime;// -sdata->bucket_start_id;
		
		nptr = ddata->bucket_hits;
		bptr = ddata->sieve_buckets;

		for (; i < sdata->inplace_start_id-1; i += 2)
		{
			uint64_t tmp3;
			uint32_t p2, r2;
			int s2;
						
			prime = sdata->sieve_p[i];
			p2 = sdata->sieve_p[i+1];

			// condition to see if the current prime only hits the sieve interval once
			if (prime > sdata->large_bucket_start_prime)
			{
				ddata->largep_offset = i;
				break;
			}

			s = sdata->root[i];
			s2 = sdata->root[i+1];
			
			// we solved for lower_mod_prime while computing the modular inverse of
			// each prime, for the residue class 1.  add the difference between this
			// residue class and 1 before multiplying by the modular inverse to find the offset.
			tmp2 = (uint64_t)s * (uint64_t)(lmp[i] + diff);
			tmp3 = (uint64_t)s2 * (uint64_t)(lmp[i + 1] + diff);
			
			root = (uint32_t)(tmp2 % (uint64_t)prime);
			r2 = (uint32_t)(tmp3 % (uint64_t)p2);

            // It is faster to update during
            // linesieve than doing it all here in a loop.
            // measured 6/2016
			if (root < linesize)			
			{	
				bnum = (root >> FLAGBITS);
				bptr[bnum][nptr[bnum]] = ((uint64_t)prime << 32) | (uint64_t)root;
				nptr[bnum]++;	
			}	

			if (r2 < linesize)			
			{	
				bnum = (r2 >> FLAGBITS);
                bptr[bnum][nptr[bnum]] = ((uint64_t)p2 << 32) | (uint64_t)r2;
				nptr[bnum]++;	
			}	
			
		}

		if ((i < sdata->inplace_start_id) && (ddata->largep_offset == 0))
		{
            uint32_t *lmp = sdata->lower_mod_prime;// -sdata->bucket_start_id;
			prime = sdata->sieve_p[i];

			s = sdata->root[i];
			
			tmp2 = (uint64_t)s * (uint64_t)(lmp[i] + diff);
			root = (uint32_t)(tmp2 % (uint64_t)prime);

			nptr = ddata->bucket_hits;
			bptr = ddata->sieve_buckets;
			
			if (root < linesize)			
			{	
				bnum = (root >> FLAGBITS);
                bptr[bnum][nptr[bnum]] = ((uint64_t)prime << 32) | (uint64_t)root;
				nptr[bnum]++;	
			}	

		}

		if (ddata->largep_offset > 0)
		{
            // primes greater than the entire sieve interval, thus they
            // at most hit one block and we don't need to save the prime
            // itself since it doesn't need to be advanced.
			uint32_t **large_bptr;
			uint32_t *large_nptr;
            uint32_t *lmp = sdata->lower_mod_prime;// -sdata->bucket_start_id;

			large_nptr = ddata->large_bucket_hits;
			large_bptr = ddata->large_sieve_buckets;

			for (i=0; i<sdata->blocks; i++)
			{		
				//initialize bucket
				large_nptr[i] = 0;
			}

			for (i = ddata->largep_offset; i<sdata->inplace_start_id-1; i+=2)
			{
				uint64_t tmp3;
				uint32_t p2, r2;
				int s2;
							
				prime = sdata->sieve_p[i];
				p2 = sdata->sieve_p[i+1];

				s = sdata->root[i];
				s2 = sdata->root[i+1];
				
				// we solved for lower_mod_prime while computing the modular inverse of
				// each prime, for the residue class 1.  add the difference between this
				// residue class and 1 before multiplying by the modular inverse.
                // could use (_mm256_mul_epu32 --> VPMULUDQ)
				tmp2 = (uint64_t)s * (uint64_t)(lmp[i] + diff);
				tmp3 = (uint64_t)s2 * (uint64_t)(lmp[i + 1] + diff);

                // would need custom solution
				root = (uint32_t)(tmp2 % (uint64_t)prime);
				r2 = (uint32_t)(tmp3 % (uint64_t)p2);

                //if (prime == 17808467)
                //if (prime == 47710219)
                //if (prime == 16777259) // 10^15 + 10^8
                //if (prime == 33554467) // 10^16 + 10^9
                if (prime == 1835027) // 10^14 + 10^7
                {
                    printf("low = %lu, prime = %u, root = %d, lmp = %u, class = %u, startloc = %u\n",
                        sdata->lowlimit, prime, s, lmp[i], sdata->rclass[thread_data->current_line], root);
                    //exit(1);
                }


                // gather may help, but writes would need to be done 1 by 1.
				if (root < linesize)			
				{	
					bnum = (root >> FLAGBITS);
					large_bptr[bnum][large_nptr[bnum]] = root;
					large_nptr[bnum]++;	
				}	

				if (r2 < linesize)			
				{		
					bnum = (r2 >> FLAGBITS);
					large_bptr[bnum][large_nptr[bnum]] = r2;
					large_nptr[bnum]++;	
				}	
				
			}

			if (i < sdata->inplace_start_id)
			{		
                uint32_t *lmp = sdata->lower_mod_prime;// -sdata->bucket_start_id;
				prime = sdata->sieve_p[i];

				s = sdata->root[i];
				
				tmp2 = (uint64_t)s * (uint64_t)(lmp[i] + diff);
				root = (uint32_t)(tmp2 % (uint64_t)prime);

				if (root < linesize)			
				{	
					bnum = (root >> FLAGBITS);
					large_bptr[bnum][large_nptr[bnum]] = root;
					large_nptr[bnum]++;	
				}	

			}
		}

	}

	return;
}
