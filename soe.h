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

#include "phi_ecm.h"



/*
typedef struct
{
	uint16_t *sieve_p;
	uint16_t *offsets;
	uint32_t *pbounds;
	uint32_t lblk_b;
	uint32_t ublk_b;
	uint32_t blk_b_sqrt;
	uint32_t blk_r;
	uint32_t blocks;
	uint32_t partial_block_b;
} soe_sieve16_t;

typedef struct
{
	uint32_t *sieve_p;
	uint32_t *offsets;
	uint64_t *pbounds;
	uint64_t lblk_b;
	uint64_t ublk_b;
	uint32_t blk_b_sqrt;
	uint32_t blk_r;
	uint32_t blocks;
	uint32_t partial_block_b;
} soe_sieve32_t;

typedef struct
{
	uint32_t orig_hlimit;
	uint32_t orig_llimit;
	uint32_t pbound;
	uint32_t pboundi;
	uint8_t *rclass;
	uint8_t **line;
} soe_t;


int spSOE(uint32_t *primes, uint32_t lowlimit, uint32_t *highlimit, int count);
void get_offsets16(soe_sieve16_t sieve16, soe_t soe,
				   uint32_t startprime, uint32_t prodN, uint32_t lcount, uint32_t j);
void sieve_16(uint8_t **line, uint32_t lcount, uint32_t startprime, uint32_t lineblocks,
			  uint16_t *sieve_p, uint16_t *offsets, uint32_t flagblocklimit, uint32_t flagblocksize,
			  uint8_t *masks, uint32_t partial_limit, uint32_t *pbounds, uint32_t pboundi);
int tiny_soe(uint32_t limit, uint16_t *primes);
int tiny_soe32(uint32_t limit, uint32_t *primes);
void GetPRIMESRange(uint32_t lowlimit, uint32_t highlimit);
int sieve_to_bitdepth(bignum *start, bignum *stop, int depth, uint32_t *offsets);
uint32_t soe_wrapper(uint32_t lowlimit, uint32_t highlimit, int count);

void zNextPrime(bignum *n, bignum *p, int dir);

*/
