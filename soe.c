/*----------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Ben Buhrow. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

In keeping with the public domain spirit, I will mention that some
parts of the code (and also this header), included in this distribution 
have been reused from other sources. In particular I have benefitted
greatly from the work of Jason Papadopoulos's msieve @ 
www.boo.net/~jasonp, Scott Contini's mpqs implementation, Tom St. Denis
Tom's Fast Math library, and certian numerical recipes structures.  
Many thanks to their kind donation of code to the public domain.
       				   --bbuhrow@gmail.com 12/9/08
----------------------------------------------------------------------*/


/*
implements a very fast sieve of erathostenes.  Speed enhancements
include a variable mod30 or mod210 wheel, bit packing
and cache blocking.  

cache blocking means that the sieve interval is split up into blocks,
and each block is sieved separately by all primes that are needed.
the size of the blocks and of the information about the sieving
primes and offsets into the next block are carefully chosen so
that everything fits into cache while sieving a block.

bit packing means that a number is represented as prime or not with
a single bit, rather than a byte or word.  this greatly reduces 
the storage requirements of the flags, which makes cache blocking 
more effective.

the mod30 or mod210 wheel means that depending
on the size of limit, either the primes 2,3, and 5 are presieved
or the primes 2,3,5 and 7 are presieved.  in other words, only the 
numbers not divisible by these small primes are 'written down' with flags.
this reduces the storage requirements for the prime flags, which in 
turn makes the cache blocking even more effective.  It also directly 
increases the speed because the slowest sieving steps have been removed.

*/

#include "soe.h"

static int inbits[256] = { 
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 


}; 

static int masks32[32] = {
	0x7FFFFFFF, 0xBFFFFFFF, 0xDFFFFFFF, 0xEFFFFFFF,
	0xF7FFFFFF, 0xfbffffff, 0xfdffffff, 0xfeffffff,
	0xff7fffff, 0xffbfffff, 0xffdfffff, 0xffefffff,
	0xfff7ffff, 0xfffbffff, 0xfffdffff, 0xfffeffff,
	0xffff7fff, 0xffffbfff, 0xffffdfff, 0xffffefff,
	0xfffff7ff, 0xfffffbff, 0xfffffdff, 0xfffffeff,
	0xffffff7f, 0xffffffbf, 0xffffffdf, 0xffffffef,
	0xfffffff7, 0xfffffffb, 0xfffffffd, 0xfffffffe
};


int spSOE(uint32_t *primes, uint32_t lowlimit, uint32_t *highlimit, int count)
{
	/*
	finds primes up to 'limit' using the Sieve of Erathostenes
	using the mod30 or mod210 wheel, bit packing, and block sieving

	return the number of primes found less than 'limit'
	'limit' is modified to an integer slightly larger, due to 
	cache blocking and the wheel - it is easier to finish a block
	rather than check in mid loop to see if the limit has been
	exceeded.

	if count == 1, then the primes are simply counted, and not 
	explicitly calculated and saved in *primes.
	*/

	/* TODO
	* add another layer of blocks to cut down on gathering time.  
		i.e. alternate between sieving and gathering when limit is large
	*/

	//block size = 32768 bytes
	const uint32_t flagblocksize = 32768;
	//number of flags in a block, 8 per byte
	const uint32_t flagblocklimit = flagblocksize*8;
	//masks for removing single bits in a byte
	const uint8_t masks[8] = {0x7F, 0xBF, 0xDF, 0xEF, 0xF7, 0xFB, 0xFD, 0xFE};
	//define bits per byte
	const uint32_t bitsNbyte=8;
	//masks for selecting single bits in a byte
	uint8_t nmasks[8];
	
	//variables used for the wheel
	uint32_t numclasses,prodN,startprime,lcount;

	//variables for blocking up the line structures
	uint32_t numflags, numbytes, numlinebytes; //,lineblocks,partial_limit;

	//misc
	uint32_t i,j,k,it,num_p,sp;
	uint16_t prime;
	uint8_t *flagblock;
	uint32_t *flagblock32;
	uint32_t prime32;
	int done;
	unsigned char *p;
	//uint32_t n;
	int ix,kx;

	//sieving structures
	soe_sieve16_t sieve16;
	soe_sieve32_t sieve32;
	soe_t soe;

	//timing variables
	clock_t start, stop;
	double t;
	//clock_t start2, stop2;
	double t2;

	soe.orig_hlimit = *highlimit;
	soe.orig_llimit = lowlimit;

	if (*highlimit - lowlimit < 1000000)
		*highlimit = lowlimit + 1000000;

	start = clock();

	//more efficient to sieve using mod210 when the range is big
	if ((*highlimit - lowlimit) >= 2147483648)
	{
		numclasses=48;
		prodN=210;
		startprime=4;
	}
	else
	{
		numclasses=8;
		prodN=30;
		startprime=3;
	}

	//allocate the residue classes.  
	soe.rclass = (uint8_t *)malloc(numclasses * sizeof(uint8_t));

	//create the selection masks
	for (i=0;i<bitsNbyte;i++)
		nmasks[i] = ~masks[i];
	
	//we'll need to store the offset into the next block for each prime
	sieve16.offsets = (uint16_t *)malloc(6542 * sizeof(uint16_t));
	if (sieve16.offsets == NULL)
		printf("error allocating offsets\n");
	sieve32.offsets = (uint32_t *)malloc(12251 * sizeof(uint32_t));
	if (sieve32.offsets == NULL)
		printf("error allocating offsets32\n");

	//store the primes used to sieve the rest of the flags
	//with the max sieve range set by the size of uint32_t, the number of primes
	//needed is fixed.
	sieve16.sieve_p = (uint16_t *)malloc(6542 * sizeof(uint16_t));
	if (sieve16.sieve_p == NULL)
		printf("error allocating sieve_p\n");
	sieve32.sieve_p = (uint32_t *)malloc(12251 * sizeof(uint32_t));
	if (sieve32.sieve_p == NULL)
		printf("error allocating sieve_p32\n");

	//get primes to sieve with
	sp = tiny_soe(65536,sieve16.sieve_p);

	if (*highlimit < 65536)
	{
		numlinebytes = 0;
		goto done;
	}

	//find the bound of primes we'll need to sieve with to get to limit
	soe.pbound = (uint32_t)sqrt(*highlimit);
	for (i=0;i<sp;i++)
	{
		if (sieve16.sieve_p[i] > soe.pbound)
		{
			soe.pboundi = i;
			break;
		}
	}

	//find the residue classes
	k=0;
	for (i=1;i<prodN;i++)
	{
		if (spGCD(i,prodN) == 1)
		{
			soe.rclass[k] = (uint8_t)i;
			k++;
		}
	}
	
	//temporarily set lowlimit to the first multiple of numclasses*prodN < lowlimit
	lowlimit = (lowlimit/(numclasses*prodN))*(numclasses*prodN);

	//reallocate flag structure for wheel and block sieving
	//starting at lowlimit, we need a flag for every 'numresidues' numbers out of 'prodN' up to 
	//limit.  round limit up to make this a whole number.
	numflags = (*highlimit - lowlimit)/prodN;
	numflags += ((numflags % prodN) != 0);
	numflags *= numclasses;

	//since we can pack 8 flags in a byte, we need numflags/8 bytes allocated.
	numbytes = numflags / bitsNbyte + ((numflags % bitsNbyte) != 0);

	//since there are 8 lines to sieve over, each line will contain (numflags/8)/8 bytes
	//so round numflags/8 up to the nearest multiple of 8
	numlinebytes = numbytes/numclasses + ((numbytes % numclasses) != 0);
	*highlimit = (uint32_t)((uint64_t)numlinebytes * (uint64_t)prodN * (uint64_t)bitsNbyte + lowlimit);

	start = clock();
	//we will sieve over this many bytes of flags, for each line, block by block.
	//allocate the lines
	soe.line = (uint8_t **)malloc(numclasses * sizeof(uint8_t *));
	for (i=0;i<numclasses;i++)
	{
		soe.line[i] = (uint8_t *)malloc(numlinebytes * sizeof(uint8_t));
		memset(soe.line[i],255,numlinebytes);
	}

	//a block consists of 32768 bytes of flags
	//which holds 262144 flags.
	sieve16.blocks = numlinebytes/flagblocksize;
	sieve16.partial_block_b = (numlinebytes % flagblocksize)*bitsNbyte;

	//allocate a bound for each block
	sieve16.pbounds = (uint32_t *)malloc((sieve16.blocks + (sieve16.partial_block_b > 0))*sizeof(uint32_t));

	//special case, 1 is not a prime
	if (lowlimit <= 1)
		soe.line[0][0] = 0x7F;

	sieve16.blk_r = flagblocklimit*prodN;
	sieve16.pbounds[0] = soe.pboundi;
	it=0;

	if (0)
	{	
		printf("sieve bound = %u\n",soe.pbound);
		printf("found %d residue classes\n",numclasses);
		printf("numlinebytes = %d\n",numlinebytes);
		printf("lineblocks = %d\n",sieve16.blocks);
		printf("partial limit = %d\n",sieve16.partial_block_b);
		printf("block range = %d\n",sieve16.blk_r);
	}
	
	stop = clock();
	t = (double)(stop - start)/(double)CLOCKS_PER_SEC;
	if (0)
		printf("elapsed time for init = %6.5f\n",t);
	
	t=t2=0;
	//main sieve, line by line
	start = clock();
	for (lcount=0;lcount<numclasses;lcount++)
	{
		sieve16.lblk_b = lowlimit;
		sieve16.ublk_b = sieve16.blk_r + soe.rclass[lcount] + sieve16.lblk_b - prodN;
		sieve16.blk_b_sqrt = (uint32_t)(sqrt(sieve16.ublk_b + prodN)) + 1;
		j=0;

		//for the current line, find the offsets past the low limit
		get_offsets16(sieve16,soe,startprime,prodN,lcount,j);

		/*
		printf("pbounds for line %d: ",lcount);
		for (i=0;i<sieve16.blocks;i++)
			printf("%d ",sieve16.pbounds[i]);
		printf("\n");
		*/

		//now we have primes and offsets for every prime, for this line.
		//proceed to sieve the entire flag set, block by block, for this line
		//sieve_16(line,lcount,startprime,lineblocks,sieve_p,offsets,flagblocklimit,flagblocksize,masks,partial_limit,pbounds,pboundi);
		
		//flagblock = soe.line[lcount];
		flagblock = soe.line[lcount];
		for (i=0;i<sieve16.blocks;i++)
		{
			//sieve the block with each effective prime
			for (j=startprime;j<sieve16.pbounds[i];j++)
			{
				prime = sieve16.sieve_p[j];
				for (k=sieve16.offsets[j];k<flagblocklimit;k+=prime)
					flagblock[k>>3] &= masks[k&7];

				sieve16.offsets[j]= (uint16_t)(k - flagblocklimit);
			}

			flagblock += flagblocksize;
		}

		//and the last partial block, don't worry about updating the offsets
		if (sieve16.partial_block_b > 0)
		{
			for (i=startprime;i<soe.pboundi;i++)
			{
				prime = sieve16.sieve_p[i];
				for (j=sieve16.offsets[i];j<sieve16.partial_block_b;j+=prime)
					flagblock[j>>3] &= masks[j&7];
			}
		}
	}
	stop = clock();
	t += (double)(stop - start)/(double)CLOCKS_PER_SEC;

done:
	if (0)
		printf("elapsed time for sieving = %6.5f\n",t);


	start = clock();

	if (count)
	{
		//the sieve primes are not in the line array, so they must be added
		//in if necessary
		//it=0;
		
		if (soe.pbound > lowlimit)
		{
			i=0;
			while (i<sieve16.pbounds[0])
			{ 
				if (sieve16.sieve_p[i] > lowlimit)
					it++;
				i++;
			}
		}
		
		//limit tweaking to make the sieve more efficient means we need to
		//check the first and last few primes in the line structures and discard
		//those outside the original requested boundaries.

		//just count the primes rather than compute them
		//still cache inefficient, but less so.
		
		for (j=0;j<numclasses;j++)
		{
			//quite a bit faster counting method
			flagblock32 = (uint32_t *)soe.line[j];
			for (i=0;i<numlinebytes/4;i++)
			{
				p = (unsigned char *)&flagblock32[i]; 

				it += (inbits[p[0]] + inbits[p[1]] + inbits[p[2]] + inbits[p[3]]); 
			}

			//potentially misses the last few bytes
			//use the simpler baseline method to get these few
			flagblock = soe.line[j];
			for (k=0; k<numlinebytes%4;k++)
			{
				it += (flagblock[i*4+k] & nmasks[0]) >> 7;
				it += (flagblock[i*4+k] & nmasks[1]) >> 6;
				it += (flagblock[i*4+k] & nmasks[2]) >> 5;
				it += (flagblock[i*4+k] & nmasks[3]) >> 4;
				it += (flagblock[i*4+k] & nmasks[4]) >> 3;
				it += (flagblock[i*4+k] & nmasks[5]) >> 2;
				it += (flagblock[i*4+k] & nmasks[6]) >> 1;
				it += (flagblock[i*4+k] & nmasks[7]);
			}

			//eliminate the primes flaged that are above or below the
			//actual requested limits, as both of these can change to 
			//facilitate sieving we'll need to compute them, and
			//decrement the counter if so.
			//this is a scarily nested loop, but it only should iterate
			//a few times.
			done = 0;
			for (ix=numlinebytes-1;ix>=0 && !done;ix--)
			{
				for (kx=bitsNbyte-1;kx>=0;kx--)
				{
					if (soe.line[j][ix] & nmasks[kx])
					{
						prime32 = prodN*(ix*bitsNbyte + kx) + soe.rclass[j] + lowlimit;
						if (prime32 > soe.orig_hlimit)
							it--;
						else
						{
							done = 1;
							break;
						}
					}
				}
			}
			done = 0;
			for (ix=0;ix<numlinebytes && !done;ix++)
			{
				for (kx=0;kx<8;kx++)
				{
					if (soe.line[j][ix] & nmasks[kx])
					{
						prime32 = prodN*(ix*bitsNbyte + kx) + soe.rclass[j] + lowlimit;
						if (prime32 < soe.orig_llimit)
							it--;
						else
						{
							done = 1;
							break;
						}
					}
				}
			}
		}
		
		num_p = it;
	}
	else
	{
		//the sieve primes are not in the line array, so they must be added
		//in if necessary
		it=0;
		
		if (soe.pbound > lowlimit)
		{
			i=0;
			while (i<sieve16.pbounds[0])
			{ 
				if (sieve16.sieve_p[i] > lowlimit)
				{
					primes[it] = (uint32_t)sieve16.sieve_p[it];
					it++;
				}
				i++;
			}
		}
		

		//this will find all the primes in order, but since it jumps from line to line (column search)
		//it is very cache inefficient
		for (i=0;i<numlinebytes;i++)
		{
			for (j=0;j<bitsNbyte;j++)
			{
				for (k=0;k<numclasses;k++)
				{
					if (soe.line[k][i] & nmasks[j])
					{
						primes[it] = prodN*(i*bitsNbyte + j) + soe.rclass[k] + lowlimit;
						if (primes[it] >= soe.orig_llimit)
							it++;
					}
				}
			}
		}

		//because we rounded limit up to get a whole number of bytes to sieve, 
		//the last few primes are probably bigger than limit.  ignore them.	
		for (num_p = it ; primes[num_p - 1] > soe.orig_hlimit ; num_p--) {}

		primes = (uint32_t *)realloc(primes,num_p * sizeof(uint32_t));
	}

	stop = clock();
	t = (double)(stop - start)/(double)CLOCKS_PER_SEC;
	if (0)
		printf("elapsed time for gathering = %6.5f\n",t);

	for (i=0;i<numclasses;i++)
		free(soe.line[i]);
	free(soe.line);
	free(sieve16.offsets);
	free(sieve16.sieve_p);
	free(sieve32.offsets);
	free(sieve32.sieve_p);
	free(soe.rclass);
	free(sieve16.pbounds);

	return num_p;
}

void get_offsets16(soe_sieve16_t sieve16, soe_t soe,
				   uint32_t startprime, uint32_t prodN, uint32_t lcount, uint32_t j)
{
	uint32_t i;
	uint16_t prime;
	uint32_t tmp;

	for (i=startprime;i<soe.pboundi;i++)
	{
		prime = sieve16.sieve_p[i];
		//find the first multiple of the prime which is greater than 'block1' and equal
		//to the residue class mod 'prodN'.  

		//solving the congruence: rclass[lcount] == kp mod prodN for k
		//use the eGCD?

		//if the prime is greater than the limit at which it is necessary to sieve
		//a block, start that prime in the next block.
		if (sieve16.sieve_p[i] > sieve16.blk_b_sqrt)
		{
			sieve16.pbounds[j] = i;
			j++;
			sieve16.lblk_b = sieve16.ublk_b + prodN;
			sieve16.ublk_b += sieve16.blk_r;
			sieve16.blk_b_sqrt = (uint32_t)(sqrt(sieve16.ublk_b + prodN)) + 1;
		}

		tmp = sieve16.lblk_b/prime;
		tmp *= prime;
		//tmp += prime*prime;
		do
		{
			tmp += prime;
		} while ((tmp % prodN) != soe.rclass[lcount]);

		//now find out how much bigger this is than 'block1'
		//in steps of 'prodN'.  this is exactly the offset in flags.
		sieve16.offsets[i] = (uint16_t)((tmp - sieve16.lblk_b)/prodN);
	}

	return;
}

void sieve_16(uint8_t **line, uint32_t lcount, uint32_t startprime, uint32_t lineblocks,
			  uint16_t *sieve_p, uint16_t *offsets, uint32_t flagblocklimit, uint32_t flagblocksize,
			  uint8_t *masks, uint32_t partial_limit, uint32_t *pbounds, uint32_t pboundi)
{
	uint8_t *flagblock;
	uint32_t i,j,k;
	uint16_t prime;

	flagblock = line[lcount];
	for (i=0;i<lineblocks;i++)
	{
		//sieve the block with each effective prime
		for (j=startprime;j<pbounds[i];j++)
		{
			prime = sieve_p[j];
			for (k=offsets[j];k<flagblocklimit;k+=prime)
				flagblock[k>>3] &= masks[k&7];

			offsets[j]= (uint16_t)(k - flagblocklimit);
		}

		flagblock += flagblocksize;
	}

	//and the last partial block, don't worry about updating the offsets
	if (partial_limit > 0)
	{
		for (i=startprime;i<pboundi;i++)
		{
			if (offsets[i]<partial_limit)
			{
				prime = sieve_p[i];
				for (j=offsets[i];j<partial_limit;j+=prime)
					flagblock[j>>3] &= masks[j&7];
			}
		}
	}

	return;
}

int tiny_soe(uint32_t limit, uint16_t *primes)
{
	//simple sieve of erathosthenes for small limits - not efficient
	//for large limits.
	uint8_t *flags;
	uint16_t prime;
	uint32_t i,j;
	int it;

	//allocate flags
	flags = (uint8_t *)malloc(limit/2 * sizeof(uint8_t));
	if (flags == NULL)
		printf("error allocating flags\n");
	memset(flags,1,limit/2);

	//find the sieving primes, don't bother with offsets, we'll need to find those
	//separately for each line in the main sieve.
	primes[0] = 2;
	it=1;
	
	//sieve using primes less than the sqrt of limit
	//flags are created only for odd numbers (mod2)
	for (i=1;i<(uint32_t)(sqrt(limit)/2+1);i++)
	{
		if (flags[i] > 0)
		{
			prime = (uint16_t)(2*i + 1);
			for (j=i+prime;j<limit/2;j+=prime)
				flags[j]=0;

			primes[it]=prime;
			it++;
		}
	}

	//now find all the prime flags and compute the sieving primes
	//the last few will exceed uint16_t, we can fix this later.
	for (;i<limit/2;i++)
	{
		if (flags[i] == 1)
		{
			primes[it] = (uint16_t)(2*i + 1);
			it++;
		}
	}

	free(flags);
	return it;
}

int tiny_soe32(uint32_t limit, uint32_t *primes)
{
	//simple sieve of erathosthenes for small limits - not efficient
	//for large limits.
	uint8_t *flags;
	uint32_t prime;
	uint32_t i,j;
	int it;

	//allocate flags
	flags = (uint8_t *)malloc(limit/2 * sizeof(uint8_t));
	if (flags == NULL)
		printf("error allocating flags\n");
	memset(flags,1,limit/2);

	//find the sieving primes, don't bother with offsets, we'll need to find those
	//separately for each line in the main sieve.
	primes[0] = 2;
	it=1;
	
	//sieve using primes less than the sqrt of block1
	//flags are created only for odd numbers (mod2)
	for (i=1;i<(uint32_t)(sqrt(limit)/2+1);i++)
	{
		if (flags[i] > 0)
		{
			prime = (uint32_t)(2*i + 1);
			for (j=i+prime;j<limit/2;j+=prime)
				flags[j]=0;

			primes[it]=prime;
			it++;
		}
	}

	//now find all the prime flags and compute the sieving primes
	//the last few will exceed uint16_t, we can fix this later.
	for (;i<limit/2;i++)
	{
		if (flags[i] == 1)
		{
			primes[it] = (uint32_t)(2*i + 1);
			it++;
		}
	}

	free(flags);
	return it;
}

void GetPRIMESRange(uint32_t lowlimit, uint32_t highlimit)
{
	int i;
	
	//reallocate array based on conservative estimate of the number of 
	//primes in the interval
	i = (int)(highlimit/log(highlimit)*1.2);
	if (lowlimit != 0)
		i -= (int)(lowlimit/log(lowlimit));

	PRIMES = (uint32_t *)realloc(PRIMES,(size_t) (i*sizeof(uint32_t)));
	if (PRIMES == NULL)
	{
		printf("unable to allocate %d bytes\n",i*sizeof(uint32_t));
		exit(1);
	}

	//find the primes in the interval
	NUM_P = spSOE(PRIMES,lowlimit,&highlimit,0);

	//as it exists now, the sieve may return primes less than lowlimit.  cut these out
	/*
	for (i=0;i<NUM_P;i++)
	{
		if (PRIMES[i] > lowlimit)
			break;
	}
	PRIMES = (uint32_t *)realloc(&PRIMES[i],(NUM_P - i) * sizeof(uint32_t));
	NUM_P -= i;
	*/

	//reset the global constants
	P_MIN = PRIMES[0];
	P_MAX = PRIMES[NUM_P - 1];

	return;
}

uint32_t soe_wrapper(uint32_t lowlimit, uint32_t highlimit, int count)
{
	//public interface to the sieve.  necessary because in order to keep the 
	//sieve efficient it must sieve larger blocks of numbers than a user may want,
	//and because the program keeps a cache of primes on hand which may or may 
	//not contain the range of interest.  Manage this on-hand cache and any addition
	//sieving needed.
	uint32_t retval, tmp,i,j;

	if ((lowlimit > 4000000000) || (highlimit > 4000000000))
	{
		printf("sieve limit of 4e9\n");
		return 0;
	}

	if (highlimit < lowlimit)
	{
		printf("error: lowlimit must be less than highlimit\n");
		return 0;
	}

	if (count)
	{
		tmp = highlimit;
		retval = spSOE(NULL,lowlimit,&tmp,1);
	}
	else
	{
		if (lowlimit < P_MIN || lowlimit > P_MAX || highlimit > P_MAX)
		{
			//requested range is not coverd by the current range, get a new range
			//that is at least 1e6 wide
			if (highlimit - lowlimit < 1000000)
				GetPRIMESRange(lowlimit,lowlimit+1000000);
			else
				GetPRIMESRange(lowlimit,highlimit);
		}

		//print the range
		for (i=0, j=0;j<NUM_P && PRIMES[j] < highlimit;j++)
		{
			if (PRIMES[j] >= lowlimit)
			{
				printf("%u ",PRIMES[j]);
				i++;
			}
		}
		printf("\n");
		retval = i;
	}

	return retval;
}

