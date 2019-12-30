/*
Copyright (c) 2014, Ben Buhrow
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

#include "phi_ecm.h"
#include "omp.h"
#include <immintrin.h>

uint32_t **gbl_bitmap;

base_t * tiny_soe2(base_t limit, base_t *nump)
{
	//simple sieve of erathosthenes for small limits - not efficient
	//for large limits.  use to generate sieve primes.
	uint8_t *flags;
	base_t *primes;
	base_t prime;
	base_t i,j;
	int it;

	//allocate flags
	flags = (uint8_t *)malloc(limit/2 * sizeof(uint8_t));
	if (flags == NULL)
		printf("error allocating flags\n");
	memset(flags,1,limit/2);
    
	//find the sieving primes, don't bother with offsets, we'll need to find those
	//separately for each line in the main sieve.
	
	//sieve using primes less than the sqrt of the desired limit
	//flags are created only for odd numbers (mod2)
	for (i = 1; i < (uint32_t)(sqrt(limit)/2+1); i++)
	{
		if (flags[i] > 0)
		{
			prime = (uint32_t)(2*i + 1);
			for (j = i + prime; j < limit/2; j += prime)
			{
				flags[j] = 0;			
			}
		}
	}

	//now find the rest of the prime flags and compute the sieving primes
	for (i = 0, it = 0; i < limit/2; i++)
	{
		if (flags[i] == 1)
		{			
			it++;
		}
	}

	primes = (base_t *)xmalloc_align(it * sizeof(base_t));
	primes[0] = 2;
	for (i = 1, it = 1; i < limit/2; i++)
	{
		if (flags[i] == 1)
		{			
			primes[it++] = 2 * i + 1;
		}
	}

	*nump = it;
	free(flags);
	return primes;
}


#define threadsPerBlock 224
#define startprime 3

uint32_t _step5[5] = { 2418280706, 604570176, 151142544, 37785636, 1083188233 };
uint32_t _step7[7] = { 1107363844, 69210240, 2151809288, 134488080,
276840961, 17302560, 537952322 };
uint32_t _step11[11] = { 33816584, 1073774848, 135266336, 132096, 541065345,
528384, 2164261380, 2113536, 67110928, 8454146, 268443712 };
uint32_t _step13[13] = { 1075838992, 16809984, 262656, 536875016, 8388672,
67239937, 1050624, 2147500064, 33554688, 268959748, 4202496,
65664, 134218754 };
uint32_t _step17[17] = { 268435488, 1073741952, 512, 2049, 8196, 32784, 131136,
524544, 2098176, 8392704, 33570816, 134283264, 537133056,
2148532224, 4194304, 16777218, 67108872 };
uint32_t _step19[19] = { 2147483712, 4096, 262176, 16779264, 1073872896, 8388608,
536870928, 1024, 65544, 4194816, 268468224, 2097152, 134217732,
256, 16386, 1048704, 67117056, 524288, 33554433 };
uint32_t _step23[23] = { 128, 2097216, 1048576, 8, 131076, 2147549184, 1073741824, 8192,
134221824, 67108864, 512, 8388864, 4194304, 32, 524304, 262144,
2, 32769, 536887296, 268435456, 2048, 33555456, 16777216 };
uint32_t _step29[29] = { 512, 65536, 8, 536871936, 0, 8388624, 1073741824,
131072, 16777216, 2048, 262144, 32, 2147487744, 0,
33554496, 0, 524289, 67108864, 8192, 1048576, 128,
16384, 2, 134217984, 0, 2097156, 268435456, 32768, 4194304 };
uint32_t _step31[31] = { 1024, 524288, 256, 131072, 64, 32768, 16, 8192, 4, 2048,
1, 1073742336, 0, 268435584, 0, 67108896, 0, 16777224, 0,
4194306, 2147483648, 1048576, 536870912, 262144, 134217728,
65536, 33554432, 16384, 8388608, 4096, 2097152 };


// here is probably the best way to do it:
// have each thread sieve a different block.
// within each thread-block, do vectorized sieving of 16 primes at a time.
// since a single thread is involved in each block, the block can
// be bit-packed and we can use mod 30 or mod 210 and sieve by lines.
// re-compute offsets for each new set of blocks.
// after each set of blocks, each threads counts primes in its block.
// based on counts, an offset into a large list of primes can be computed for each thread.
// each thread can then compute and simultaneously write to the large list.

// timings for comparison (M2050 Tesla GPU):
//3.222000 milliseconds for big sieve
//5761455 big primes(< 100000000) found
//
//36.381001 milliseconds for big sieve
//50847534 big primes(< 1000000000) found
//
//543.804993 milliseconds for big sieve
//455052511 big primes(< 10000000000) found

// currently at (3121P)
// 257 milliseconds for 10^7
// 257 milliseconds for 10^8
// 281 milliseconds for 10^9
// 648 milliseconds for 10^10

/*
#define VEC_SIZE 16
#define NLINES 8
#define BSIZE 8192  // both 4096 and 16384 were slower... 

int _64_MOD_P[9] = { 4, 1, 9, 12, 13, 7, 18, 6, 2 };
// 512 mod 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
int _512_MOD_P[14] = { 4, 1, 6, 5, 2, 18, 6, 19, 16, 31, 20, 39, 42, 35 };

// pid = 3
uint64_t _7_MASKS_512[7*8] = {
    0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL,
    0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL,
    0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL,
    0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL,
    0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL,
    0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL,
    0xbf7efdfbf7efdfbfULL, 0xdfbf7efdfbf7efdfULL, 0xefdfbf7efdfbf7efULL, 0xf7efdfbf7efdfbf7ULL, 0xfbf7efdfbf7efdfbULL, 0xfdfbf7efdfbf7efdULL, 0x7efdfbf7efdfbf7eULL, 0xbf7efdfbf7efdfbfULL};

// pid = 4
uint64_t _11_MASKS_512[11 * 8] = {
    0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL,
    0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL,
    0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL,
    0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL,
    0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL,
    0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL,
    0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL,
    0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL,
    0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL,
    0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL, 0xf7feffdffbff7fefULL, 0xdffbff7feffdffbfULL, 0x7feffdffbff7feffULL, 0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL,
    0xffbff7feffdffbffULL, 0xfeffdffbff7feffdULL, 0xfbff7feffdffbff7ULL, 0xeffdffbff7feffdfULL, 0xbff7feffdffbff7fULL, 0xffdffbff7feffdffULL, 0xff7feffdffbff7feULL, 0xfdffbff7feffdffbULL};

// pid = 5
uint64_t _13_MASKS_512[13 * 8] = {
    0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL,
    0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL,
    0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL,
    0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL,
    0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL,
    0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL,
    0xfbffdffefff7ffbfULL, 0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL,
    0xf7ffbffdffefff7fULL, 0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL,
    0xefff7ffbffdffeffULL, 0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL,
    0xdffefff7ffbffdffULL, 0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL,
    0xbffdffefff7ffbffULL, 0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL,
    0x7ffbffdffefff7ffULL, 0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL,
    0xfff7ffbffdffefffULL, 0xffefff7ffbffdffeULL, 0xffdffefff7ffbffdULL, 0xffbffdffefff7ffbULL, 0xff7ffbffdffefff7ULL, 0xfefff7ffbffdffefULL, 0xfdffefff7ffbffdfULL, 0xfbffdffefff7ffbfULL};

// pid = 6
uint64_t _17_MASKS_512[17 * 8] = {
    0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL,
    0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL,
    0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL,
    0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL,
    0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL,
    0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL,
    0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL,
    0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL,
    0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL,
    0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL,
    0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL,
    0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL,
    0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL,
    0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL, 0x7fffbfffdfffefffULL, 0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL,
    0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL, 0xffff7fffbfffdfffULL, 0xfff7fffbfffdfffeULL, 0xff7fffbfffdfffefULL, 0xf7fffbfffdfffeffULL,
    0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL, 0xfffeffff7fffbfffULL, 0xffeffff7fffbfffdULL, 0xfeffff7fffbfffdfULL, 0xeffff7fffbfffdffULL,
    0xfffbfffdfffeffffULL, 0xffbfffdfffeffff7ULL, 0xfbfffdfffeffff7fULL, 0xbfffdfffeffff7ffULL, 0xfffdfffeffff7fffULL, 0xffdfffeffff7fffbULL, 0xfdfffeffff7fffbfULL, 0xdfffeffff7fffbffULL};

// pid = 7
uint64_t _19_MASKS_512[19 * 8] = {
    0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL,
    0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL,
    0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL,
    0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL,
    0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL,
    0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL,
    0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL,
    0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL,
    0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL,
    0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL,
    0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL,
    0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL,
    0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL,
    0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL,
    0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL, 0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL,
    0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL, 0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL,
    0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL, 0xfdffffbffff7fffeULL, 0xfffbffff7fffefffULL, 0xbffff7fffeffffdfULL,
    0xff7fffeffffdffffULL, 0xfffeffffdffffbffULL, 0xeffffdffffbffff7ULL, 0xffdffffbffff7fffULL, 0xffffbffff7fffeffULL, 0xfbffff7fffeffffdULL, 0xfff7fffeffffdfffULL, 0x7fffeffffdffffbfULL,
    0xfeffffdffffbffffULL, 0xfffdffffbffff7ffULL, 0xdffffbffff7fffefULL, 0xffbffff7fffeffffULL, 0xffff7fffeffffdffULL, 0xf7fffeffffdffffbULL, 0xffeffffdffffbfffULL, 0xffffdffffbffff7fULL};

// pid = 8
uint64_t _23_MASKS_512[23 * 8] = {
    0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL,
    0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL,
    0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL,
    0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL,
    0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL,
    0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL,
    0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL,
    0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL,
    0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL,
    0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL,
    0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL,
    0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL,
    0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL,
    0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL,
    0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL,
    0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL,
    0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL,
    0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL,
    0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL, 0xdfffffbfffff7fffULL, 0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL,
    0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL, 0xbfffff7ffffeffffULL, 0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL,
    0xfffff7ffffefffffULL, 0xfffefffffdfffffbULL, 0xffdfffffbfffff7fULL, 0xfbfffff7ffffefffULL, 0x7ffffefffffdffffULL, 0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL,
    0xffffefffffdfffffULL, 0xfffdfffffbfffff7ULL, 0xffbfffff7ffffeffULL, 0xf7ffffefffffdfffULL, 0xfffffdfffffbffffULL, 0xffffbfffff7ffffeULL, 0xfff7ffffefffffdfULL, 0xfefffffdfffffbffULL,
    0xffffdfffffbfffffULL, 0xfffbfffff7ffffefULL, 0xff7ffffefffffdffULL, 0xefffffdfffffbfffULL, 0xfffffbfffff7ffffULL, 0xffff7ffffefffffdULL, 0xffefffffdfffffbfULL, 0xfdfffffbfffff7ffULL};

// pid = 9
uint64_t _29_MASKS_512[29 * 8] = {
    0xfbffffffdffffffeULL, 0xffefffffff7fffffULL, 0xffffbffffffdffffULL, 0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL,
    0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL, 0xffff7ffffffbffffULL, 0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL, 0xffefffffff7fffffULL, 0xffffbffffffdffffULL,
    0xefffffff7ffffffbULL, 0xffbffffffdffffffULL, 0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL, 0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL, 0xffff7ffffffbffffULL,
    0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL, 0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL, 0xefffffff7ffffffbULL, 0xffbffffffdffffffULL, 0xfffefffffff7ffffULL,
    0xbffffffdffffffefULL, 0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL, 0xffffefffffff7fffULL, 0xffffffbffffffdffULL, 0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL, 0xfffdffffffefffffULL,
    0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL, 0xffffff7ffffffbffULL, 0xbffffffdffffffefULL, 0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL,
    0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL, 0xffefffffff7fffffULL, 0xffffbffffffdffffULL, 0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL,
    0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL, 0xffff7ffffffbffffULL, 0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL, 0xffefffffff7fffffULL,
    0xffffffdffffffeffULL, 0xefffffff7ffffffbULL, 0xffbffffffdffffffULL, 0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL, 0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL,
    0xffffffbffffffdffULL, 0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL, 0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL, 0xefffffff7ffffffbULL, 0xffbffffffdffffffULL,
    0xffffff7ffffffbffULL, 0xbffffffdffffffefULL, 0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL, 0xffffefffffff7fffULL, 0xffffffbffffffdffULL, 0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL,
    0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL, 0xffffff7ffffffbffULL, 0xbffffffdffffffefULL, 0xfefffffff7ffffffULL,
    0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL, 0xffefffffff7fffffULL, 0xffffbffffffdffffULL, 0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL,
    0xfffffbffffffdfffULL, 0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL, 0xffff7ffffffbffffULL, 0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL,
    0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL, 0xefffffff7ffffffbULL, 0xffbffffffdffffffULL, 0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL, 0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL,
    0xffffefffffff7fffULL, 0xffffffbffffffdffULL, 0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL, 0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL, 0xefffffff7ffffffbULL,
    0xffffdffffffeffffULL, 0xffffff7ffffffbffULL, 0xbffffffdffffffefULL, 0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL, 0xffffefffffff7fffULL, 0xffffffbffffffdffULL, 0xdffffffefffffff7ULL,
    0xffffbffffffdffffULL, 0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL, 0xffffff7ffffffbffULL, 0xbffffffdffffffefULL,
    0xffff7ffffffbffffULL, 0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL, 0xffefffffff7fffffULL, 0xffffbffffffdffffULL, 0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL,
    0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL, 0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL, 0xffff7ffffffbffffULL, 0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL,
    0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL, 0xefffffff7ffffffbULL, 0xffbffffffdffffffULL, 0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL, 0xffffffefffffff7fULL,
    0xfffbffffffdfffffULL, 0xffffefffffff7fffULL, 0xffffffbffffffdffULL, 0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL, 0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL,
    0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL, 0xffffff7ffffffbffULL, 0xbffffffdffffffefULL, 0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL, 0xffffefffffff7fffULL, 0xffffffbffffffdffULL,
    0xffefffffff7fffffULL, 0xffffbffffffdffffULL, 0xfffffefffffff7ffULL, 0x7ffffffbffffffdfULL, 0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL, 0xffffff7ffffffbffULL,
    0xffdffffffeffffffULL, 0xffff7ffffffbffffULL, 0xfffffdffffffefffULL, 0xfffffff7ffffffbfULL, 0xfbffffffdffffffeULL, 0xffefffffff7fffffULL, 0xffffbffffffdffffULL, 0xfffffefffffff7ffULL,
    0xffbffffffdffffffULL, 0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL, 0xffffffefffffff7fULL, 0xf7ffffffbffffffdULL, 0xffdffffffeffffffULL, 0xffff7ffffffbffffULL, 0xfffffdffffffefffULL,
    0xff7ffffffbffffffULL, 0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL, 0xffffffdffffffeffULL, 0xefffffff7ffffffbULL, 0xffbffffffdffffffULL, 0xfffefffffff7ffffULL, 0xfffffbffffffdfffULL,
    0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL, 0xffffefffffff7fffULL, 0xffffffbffffffdffULL, 0xdffffffefffffff7ULL, 0xff7ffffffbffffffULL, 0xfffdffffffefffffULL, 0xfffff7ffffffbfffULL,
    0xfdffffffefffffffULL, 0xfff7ffffffbfffffULL, 0xffffdffffffeffffULL, 0xffffff7ffffffbffULL, 0xbffffffdffffffefULL, 0xfefffffff7ffffffULL, 0xfffbffffffdfffffULL, 0xffffefffffff7fffULL};

// pid = 10
uint64_t _31_MASKS_512[31 * 8] = {
    0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL, 0xffbfffffff7fffffULL, 0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL,
    0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL,
    0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL, 0xffbfffffff7fffffULL, 0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL,
    0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL,
    0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL, 0xffbfffffff7fffffULL, 0xffefffffffdfffffULL,
    0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL,
    0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL, 0xffbfffffff7fffffULL,
    0xffffffbfffffff7fULL, 0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL,
    0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL,
    0xfffffefffffffdffULL, 0xffffffbfffffff7fULL, 0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL,
    0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL,
    0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL, 0xffffffbfffffff7fULL, 0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL,
    0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL, 0xefffffffdfffffffULL,
    0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL, 0xffffffbfffffff7fULL, 0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL, 0xdfffffffbfffffffULL,
    0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL, 0xbfffffff7ffffffeULL,
    0xffffbfffffff7fffULL, 0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL, 0xffffffbfffffff7fULL, 0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL, 0x7ffffffefffffffdULL,
    0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL, 0xfffffffdfffffffbULL,
    0xfffefffffffdffffULL, 0xffffbfffffff7fffULL, 0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL, 0xffffffbfffffff7fULL, 0xffffffefffffffdfULL, 0xfffffffbfffffff7ULL,
    0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL, 0xfffffff7ffffffefULL,
    0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL, 0xffffbfffffff7fffULL, 0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL, 0xffffffbfffffff7fULL, 0xffffffefffffffdfULL,
    0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL, 0xffffffdfffffffbfULL,
    0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL, 0xffffbfffffff7fffULL, 0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL, 0xffffffbfffffff7fULL,
    0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL, 0xffffff7ffffffeffULL,
    0xffbfffffff7fffffULL, 0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL, 0xffffbfffffff7fffULL, 0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL, 0xfffffefffffffdffULL,
    0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL, 0xfffffdfffffffbffULL,
    0xfefffffffdffffffULL, 0xffbfffffff7fffffULL, 0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL, 0xffffbfffffff7fffULL, 0xffffefffffffdfffULL, 0xfffffbfffffff7ffULL,
    0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL, 0xfffff7ffffffefffULL,
    0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL, 0xffbfffffff7fffffULL, 0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL, 0xffffbfffffff7fffULL, 0xffffefffffffdfffULL,
    0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL, 0xffffdfffffffbfffULL,
    0xefffffffdfffffffULL, 0xfbfffffff7ffffffULL, 0xfefffffffdffffffULL, 0xffbfffffff7fffffULL, 0xffefffffffdfffffULL, 0xfffbfffffff7ffffULL, 0xfffefffffffdffffULL, 0xffffbfffffff7fffULL,
    0xdfffffffbfffffffULL, 0xf7ffffffefffffffULL, 0xfdfffffffbffffffULL, 0xff7ffffffeffffffULL, 0xffdfffffffbfffffULL, 0xfff7ffffffefffffULL, 0xfffdfffffffbffffULL, 0xffff7ffffffeffffULL};

// pid = 11 (p = 37, 2368 bytes)
uint64_t _37_MASKS_512[37 * 8] = {
    0xffffffdffffffffeULL, 0xffff7ffffffffbffULL, 0xfdffffffffefffffULL, 0xffffffffbfffffffULL, 0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL, 0xefffffffff7fffffULL, 0xfffffffdffffffffULL,
    0xffffffbffffffffdULL, 0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL, 0xffffffff7fffffffULL, 0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL, 0xfffffffbffffffffULL,
    0xffffff7ffffffffbULL, 0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL, 0xfffffbffffffffdfULL, 0xffefffffffff7fffULL, 0xbffffffffdffffffULL, 0xfffffff7ffffffffULL,
    0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL, 0xefffffffff7fffffULL, 0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL, 0x7ffffffffbffffffULL, 0xffffffefffffffffULL,
    0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL, 0xfffffffbffffffffULL, 0xffffefffffffff7fULL, 0xffbffffffffdffffULL, 0xfffffffff7ffffffULL, 0xffffffdffffffffeULL,
    0xfffffbffffffffdfULL, 0xffefffffffff7fffULL, 0xbffffffffdffffffULL, 0xfffffff7ffffffffULL, 0xffffdffffffffeffULL, 0xff7ffffffffbffffULL, 0xffffffffefffffffULL, 0xffffffbffffffffdULL,
    0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL, 0x7ffffffffbffffffULL, 0xffffffefffffffffULL, 0xffffbffffffffdffULL, 0xfefffffffff7ffffULL, 0xffffffffdfffffffULL, 0xffffff7ffffffffbULL,
    0xffffefffffffff7fULL, 0xffbffffffffdffffULL, 0xfffffffff7ffffffULL, 0xffffffdffffffffeULL, 0xffff7ffffffffbffULL, 0xfdffffffffefffffULL, 0xffffffffbfffffffULL, 0xfffffefffffffff7ULL,
    0xffffdffffffffeffULL, 0xff7ffffffffbffffULL, 0xffffffffefffffffULL, 0xffffffbffffffffdULL, 0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL, 0xffffffff7fffffffULL, 0xfffffdffffffffefULL,
    0xffffbffffffffdffULL, 0xfefffffffff7ffffULL, 0xffffffffdfffffffULL, 0xffffff7ffffffffbULL, 0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL, 0xfffffbffffffffdfULL,
    0xffff7ffffffffbffULL, 0xfdffffffffefffffULL, 0xffffffffbfffffffULL, 0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL, 0xefffffffff7fffffULL, 0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL,
    0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL, 0xffffffff7fffffffULL, 0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL, 0xfffffffbffffffffULL, 0xffffefffffffff7fULL,
    0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL, 0xfffffbffffffffdfULL, 0xffefffffffff7fffULL, 0xbffffffffdffffffULL, 0xfffffff7ffffffffULL, 0xffffdffffffffeffULL,
    0xfffbffffffffdfffULL, 0xefffffffff7fffffULL, 0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL, 0x7ffffffffbffffffULL, 0xffffffefffffffffULL, 0xffffbffffffffdffULL,
    0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL, 0xfffffffbffffffffULL, 0xffffefffffffff7fULL, 0xffbffffffffdffffULL, 0xfffffffff7ffffffULL, 0xffffffdffffffffeULL, 0xffff7ffffffffbffULL,
    0xffefffffffff7fffULL, 0xbffffffffdffffffULL, 0xfffffff7ffffffffULL, 0xffffdffffffffeffULL, 0xff7ffffffffbffffULL, 0xffffffffefffffffULL, 0xffffffbffffffffdULL, 0xfffefffffffff7ffULL,
    0xffdffffffffeffffULL, 0x7ffffffffbffffffULL, 0xffffffefffffffffULL, 0xffffbffffffffdffULL, 0xfefffffffff7ffffULL, 0xffffffffdfffffffULL, 0xffffff7ffffffffbULL, 0xfffdffffffffefffULL,
    0xffbffffffffdffffULL, 0xfffffffff7ffffffULL, 0xffffffdffffffffeULL, 0xffff7ffffffffbffULL, 0xfdffffffffefffffULL, 0xffffffffbfffffffULL, 0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL,
    0xff7ffffffffbffffULL, 0xffffffffefffffffULL, 0xffffffbffffffffdULL, 0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL, 0xffffffff7fffffffULL, 0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL,
    0xfefffffffff7ffffULL, 0xffffffffdfffffffULL, 0xffffff7ffffffffbULL, 0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL, 0xfffffbffffffffdfULL, 0xffefffffffff7fffULL,
    0xfdffffffffefffffULL, 0xffffffffbfffffffULL, 0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL, 0xefffffffff7fffffULL, 0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL,
    0xfbffffffffdfffffULL, 0xffffffff7fffffffULL, 0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL, 0xfffffffbffffffffULL, 0xffffefffffffff7fULL, 0xffbffffffffdffffULL,
    0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL, 0xfffffbffffffffdfULL, 0xffefffffffff7fffULL, 0xbffffffffdffffffULL, 0xfffffff7ffffffffULL, 0xffffdffffffffeffULL, 0xff7ffffffffbffffULL,
    0xefffffffff7fffffULL, 0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL, 0x7ffffffffbffffffULL, 0xffffffefffffffffULL, 0xffffbffffffffdffULL, 0xfefffffffff7ffffULL,
    0xdffffffffeffffffULL, 0xfffffffbffffffffULL, 0xffffefffffffff7fULL, 0xffbffffffffdffffULL, 0xfffffffff7ffffffULL, 0xffffffdffffffffeULL, 0xffff7ffffffffbffULL, 0xfdffffffffefffffULL,
    0xbffffffffdffffffULL, 0xfffffff7ffffffffULL, 0xffffdffffffffeffULL, 0xff7ffffffffbffffULL, 0xffffffffefffffffULL, 0xffffffbffffffffdULL, 0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL,
    0x7ffffffffbffffffULL, 0xffffffefffffffffULL, 0xffffbffffffffdffULL, 0xfefffffffff7ffffULL, 0xffffffffdfffffffULL, 0xffffff7ffffffffbULL, 0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL,
    0xfffffffff7ffffffULL, 0xffffffdffffffffeULL, 0xffff7ffffffffbffULL, 0xfdffffffffefffffULL, 0xffffffffbfffffffULL, 0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL, 0xefffffffff7fffffULL,
    0xffffffffefffffffULL, 0xffffffbffffffffdULL, 0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL, 0xffffffff7fffffffULL, 0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL,
    0xffffffffdfffffffULL, 0xffffff7ffffffffbULL, 0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL, 0xfffffbffffffffdfULL, 0xffefffffffff7fffULL, 0xbffffffffdffffffULL,
    0xffffffffbfffffffULL, 0xfffffefffffffff7ULL, 0xfffbffffffffdfffULL, 0xefffffffff7fffffULL, 0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL, 0x7ffffffffbffffffULL,
    0xffffffff7fffffffULL, 0xfffffdffffffffefULL, 0xfff7ffffffffbfffULL, 0xdffffffffeffffffULL, 0xfffffffbffffffffULL, 0xffffefffffffff7fULL, 0xffbffffffffdffffULL, 0xfffffffff7ffffffULL,
    0xfffffffeffffffffULL, 0xfffffbffffffffdfULL, 0xffefffffffff7fffULL, 0xbffffffffdffffffULL, 0xfffffff7ffffffffULL, 0xffffdffffffffeffULL, 0xff7ffffffffbffffULL, 0xffffffffefffffffULL,
    0xfffffffdffffffffULL, 0xfffff7ffffffffbfULL, 0xffdffffffffeffffULL, 0x7ffffffffbffffffULL, 0xffffffefffffffffULL, 0xffffbffffffffdffULL, 0xfefffffffff7ffffULL, 0xffffffffdfffffffULL,
    0xfffffffbffffffffULL, 0xffffefffffffff7fULL, 0xffbffffffffdffffULL, 0xfffffffff7ffffffULL, 0xffffffdffffffffeULL, 0xffff7ffffffffbffULL, 0xfdffffffffefffffULL, 0xffffffffbfffffffULL,
    0xfffffff7ffffffffULL, 0xffffdffffffffeffULL, 0xff7ffffffffbffffULL, 0xffffffffefffffffULL, 0xffffffbffffffffdULL, 0xfffefffffffff7ffULL, 0xfbffffffffdfffffULL, 0xffffffff7fffffffULL,
    0xffffffefffffffffULL, 0xffffbffffffffdffULL, 0xfefffffffff7ffffULL, 0xffffffffdfffffffULL, 0xffffff7ffffffffbULL, 0xfffdffffffffefffULL, 0xf7ffffffffbfffffULL, 0xfffffffeffffffffULL};


// pid = 12 (p = 41, 2624 bytes)
uint64_t _41_MASKS_512[41 * 8] = {
    0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL, 0xffffffefffffffffULL, 0xffbfffffffffdfffULL, 0xffffffff7fffffffULL, 0xfffdfffffffffeffULL, 0xfffffffffbffffffULL, 0xffffeffffffffff7ULL,
    0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL, 0xffffffdfffffffffULL, 0xff7fffffffffbfffULL, 0xfffffffeffffffffULL, 0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL, 0xffffdfffffffffefULL,
    0xfffff7fffffffffbULL, 0xdfffffffffefffffULL, 0xffffffbfffffffffULL, 0xfeffffffffff7fffULL, 0xfffffffdffffffffULL, 0xfff7fffffffffbffULL, 0xffffffffefffffffULL, 0xffffbfffffffffdfULL,
    0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL, 0xffffff7fffffffffULL, 0xfdfffffffffeffffULL, 0xfffffffbffffffffULL, 0xffeffffffffff7ffULL, 0xffffffffdfffffffULL, 0xffff7fffffffffbfULL,
    0xffffdfffffffffefULL, 0x7fffffffffbfffffULL, 0xfffffeffffffffffULL, 0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL, 0xffdfffffffffefffULL, 0xffffffffbfffffffULL, 0xfffeffffffffff7fULL,
    0xffffbfffffffffdfULL, 0xffffffffff7fffffULL, 0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL, 0xffffffefffffffffULL, 0xffbfffffffffdfffULL, 0xffffffff7fffffffULL, 0xfffdfffffffffeffULL,
    0xffff7fffffffffbfULL, 0xfffffffffeffffffULL, 0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL, 0xffffffdfffffffffULL, 0xff7fffffffffbfffULL, 0xfffffffeffffffffULL, 0xfffbfffffffffdffULL,
    0xfffeffffffffff7fULL, 0xfffffffffdffffffULL, 0xfffff7fffffffffbULL, 0xdfffffffffefffffULL, 0xffffffbfffffffffULL, 0xfeffffffffff7fffULL, 0xfffffffdffffffffULL, 0xfff7fffffffffbffULL,
    0xfffdfffffffffeffULL, 0xfffffffffbffffffULL, 0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL, 0xffffff7fffffffffULL, 0xfdfffffffffeffffULL, 0xfffffffbffffffffULL, 0xffeffffffffff7ffULL,
    0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL, 0xffffdfffffffffefULL, 0x7fffffffffbfffffULL, 0xfffffeffffffffffULL, 0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL, 0xffdfffffffffefffULL,
    0xfff7fffffffffbffULL, 0xffffffffefffffffULL, 0xffffbfffffffffdfULL, 0xffffffffff7fffffULL, 0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL, 0xffffffefffffffffULL, 0xffbfffffffffdfffULL,
    0xffeffffffffff7ffULL, 0xffffffffdfffffffULL, 0xffff7fffffffffbfULL, 0xfffffffffeffffffULL, 0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL, 0xffffffdfffffffffULL, 0xff7fffffffffbfffULL,
    0xffdfffffffffefffULL, 0xffffffffbfffffffULL, 0xfffeffffffffff7fULL, 0xfffffffffdffffffULL, 0xfffff7fffffffffbULL, 0xdfffffffffefffffULL, 0xffffffbfffffffffULL, 0xfeffffffffff7fffULL,
    0xffbfffffffffdfffULL, 0xffffffff7fffffffULL, 0xfffdfffffffffeffULL, 0xfffffffffbffffffULL, 0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL, 0xffffff7fffffffffULL, 0xfdfffffffffeffffULL,
    0xff7fffffffffbfffULL, 0xfffffffeffffffffULL, 0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL, 0xffffdfffffffffefULL, 0x7fffffffffbfffffULL, 0xfffffeffffffffffULL, 0xfbfffffffffdffffULL,
    0xfeffffffffff7fffULL, 0xfffffffdffffffffULL, 0xfff7fffffffffbffULL, 0xffffffffefffffffULL, 0xffffbfffffffffdfULL, 0xffffffffff7fffffULL, 0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL,
    0xfdfffffffffeffffULL, 0xfffffffbffffffffULL, 0xffeffffffffff7ffULL, 0xffffffffdfffffffULL, 0xffff7fffffffffbfULL, 0xfffffffffeffffffULL, 0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL,
    0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL, 0xffdfffffffffefffULL, 0xffffffffbfffffffULL, 0xfffeffffffffff7fULL, 0xfffffffffdffffffULL, 0xfffff7fffffffffbULL, 0xdfffffffffefffffULL,
    0xf7fffffffffbffffULL, 0xffffffefffffffffULL, 0xffbfffffffffdfffULL, 0xffffffff7fffffffULL, 0xfffdfffffffffeffULL, 0xfffffffffbffffffULL, 0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL,
    0xeffffffffff7ffffULL, 0xffffffdfffffffffULL, 0xff7fffffffffbfffULL, 0xfffffffeffffffffULL, 0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL, 0xffffdfffffffffefULL, 0x7fffffffffbfffffULL,
    0xdfffffffffefffffULL, 0xffffffbfffffffffULL, 0xfeffffffffff7fffULL, 0xfffffffdffffffffULL, 0xfff7fffffffffbffULL, 0xffffffffefffffffULL, 0xffffbfffffffffdfULL, 0xffffffffff7fffffULL,
    0xbfffffffffdfffffULL, 0xffffff7fffffffffULL, 0xfdfffffffffeffffULL, 0xfffffffbffffffffULL, 0xffeffffffffff7ffULL, 0xffffffffdfffffffULL, 0xffff7fffffffffbfULL, 0xfffffffffeffffffULL,
    0x7fffffffffbfffffULL, 0xfffffeffffffffffULL, 0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL, 0xffdfffffffffefffULL, 0xffffffffbfffffffULL, 0xfffeffffffffff7fULL, 0xfffffffffdffffffULL,
    0xffffffffff7fffffULL, 0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL, 0xffffffefffffffffULL, 0xffbfffffffffdfffULL, 0xffffffff7fffffffULL, 0xfffdfffffffffeffULL, 0xfffffffffbffffffULL,
    0xfffffffffeffffffULL, 0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL, 0xffffffdfffffffffULL, 0xff7fffffffffbfffULL, 0xfffffffeffffffffULL, 0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL,
    0xfffffffffdffffffULL, 0xfffff7fffffffffbULL, 0xdfffffffffefffffULL, 0xffffffbfffffffffULL, 0xfeffffffffff7fffULL, 0xfffffffdffffffffULL, 0xfff7fffffffffbffULL, 0xffffffffefffffffULL,
    0xfffffffffbffffffULL, 0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL, 0xffffff7fffffffffULL, 0xfdfffffffffeffffULL, 0xfffffffbffffffffULL, 0xffeffffffffff7ffULL, 0xffffffffdfffffffULL,
    0xfffffffff7ffffffULL, 0xffffdfffffffffefULL, 0x7fffffffffbfffffULL, 0xfffffeffffffffffULL, 0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL, 0xffdfffffffffefffULL, 0xffffffffbfffffffULL,
    0xffffffffefffffffULL, 0xffffbfffffffffdfULL, 0xffffffffff7fffffULL, 0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL, 0xffffffefffffffffULL, 0xffbfffffffffdfffULL, 0xffffffff7fffffffULL,
    0xffffffffdfffffffULL, 0xffff7fffffffffbfULL, 0xfffffffffeffffffULL, 0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL, 0xffffffdfffffffffULL, 0xff7fffffffffbfffULL, 0xfffffffeffffffffULL,
    0xffffffffbfffffffULL, 0xfffeffffffffff7fULL, 0xfffffffffdffffffULL, 0xfffff7fffffffffbULL, 0xdfffffffffefffffULL, 0xffffffbfffffffffULL, 0xfeffffffffff7fffULL, 0xfffffffdffffffffULL,
    0xffffffff7fffffffULL, 0xfffdfffffffffeffULL, 0xfffffffffbffffffULL, 0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL, 0xffffff7fffffffffULL, 0xfdfffffffffeffffULL, 0xfffffffbffffffffULL,
    0xfffffffeffffffffULL, 0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL, 0xffffdfffffffffefULL, 0x7fffffffffbfffffULL, 0xfffffeffffffffffULL, 0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL,
    0xfffffffdffffffffULL, 0xfff7fffffffffbffULL, 0xffffffffefffffffULL, 0xffffbfffffffffdfULL, 0xffffffffff7fffffULL, 0xfffffdfffffffffeULL, 0xf7fffffffffbffffULL, 0xffffffefffffffffULL,
    0xfffffffbffffffffULL, 0xffeffffffffff7ffULL, 0xffffffffdfffffffULL, 0xffff7fffffffffbfULL, 0xfffffffffeffffffULL, 0xfffffbfffffffffdULL, 0xeffffffffff7ffffULL, 0xffffffdfffffffffULL,
    0xfffffff7ffffffffULL, 0xffdfffffffffefffULL, 0xffffffffbfffffffULL, 0xfffeffffffffff7fULL, 0xfffffffffdffffffULL, 0xfffff7fffffffffbULL, 0xdfffffffffefffffULL, 0xffffffbfffffffffULL,
    0xffffffefffffffffULL, 0xffbfffffffffdfffULL, 0xffffffff7fffffffULL, 0xfffdfffffffffeffULL, 0xfffffffffbffffffULL, 0xffffeffffffffff7ULL, 0xbfffffffffdfffffULL, 0xffffff7fffffffffULL,
    0xffffffdfffffffffULL, 0xff7fffffffffbfffULL, 0xfffffffeffffffffULL, 0xfffbfffffffffdffULL, 0xfffffffff7ffffffULL, 0xffffdfffffffffefULL, 0x7fffffffffbfffffULL, 0xfffffeffffffffffULL,
    0xffffffbfffffffffULL, 0xfeffffffffff7fffULL, 0xfffffffdffffffffULL, 0xfff7fffffffffbffULL, 0xffffffffefffffffULL, 0xffffbfffffffffdfULL, 0xffffffffff7fffffULL, 0xfffffdfffffffffeULL,
    0xffffff7fffffffffULL, 0xfdfffffffffeffffULL, 0xfffffffbffffffffULL, 0xffeffffffffff7ffULL, 0xffffffffdfffffffULL, 0xffff7fffffffffbfULL, 0xfffffffffeffffffULL, 0xfffffbfffffffffdULL,
    0xfffffeffffffffffULL, 0xfbfffffffffdffffULL, 0xfffffff7ffffffffULL, 0xffdfffffffffefffULL, 0xffffffffbfffffffULL, 0xfffeffffffffff7fULL, 0xfffffffffdffffffULL, 0xfffff7fffffffffbULL};


// pid = 13 (p = 43, 2752 bytes)
uint64_t _43_MASKS_512[43 * 8] = {
    0xfffff7fffffffffeULL, 0xffffffffffbfffffULL, 0xffffeffffffffffdULL, 0xffffffffff7fffffULL, 0xffffdffffffffffbULL, 0xfffffffffeffffffULL, 0xffffbffffffffff7ULL, 0xfffffffffdffffffULL,
    0xffffeffffffffffdULL, 0xffffffffff7fffffULL, 0xffffdffffffffffbULL, 0xfffffffffeffffffULL, 0xffffbffffffffff7ULL, 0xfffffffffdffffffULL, 0xffff7fffffffffefULL, 0xfffffffffbffffffULL,
    0xffffdffffffffffbULL, 0xfffffffffeffffffULL, 0xffffbffffffffff7ULL, 0xfffffffffdffffffULL, 0xffff7fffffffffefULL, 0xfffffffffbffffffULL, 0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL,
    0xffffbffffffffff7ULL, 0xfffffffffdffffffULL, 0xffff7fffffffffefULL, 0xfffffffffbffffffULL, 0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL, 0xffffffffefffffffULL,
    0xffff7fffffffffefULL, 0xfffffffffbffffffULL, 0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL, 0xffffffffefffffffULL, 0xfffbffffffffff7fULL, 0xffffffffdfffffffULL,
    0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL, 0xffffffffefffffffULL, 0xfffbffffffffff7fULL, 0xffffffffdfffffffULL, 0xfff7fffffffffeffULL, 0xffffffffbfffffffULL,
    0xfffdffffffffffbfULL, 0xffffffffefffffffULL, 0xfffbffffffffff7fULL, 0xffffffffdfffffffULL, 0xfff7fffffffffeffULL, 0xffffffffbfffffffULL, 0xffeffffffffffdffULL, 0xffffffff7fffffffULL,
    0xfffbffffffffff7fULL, 0xffffffffdfffffffULL, 0xfff7fffffffffeffULL, 0xffffffffbfffffffULL, 0xffeffffffffffdffULL, 0xffffffff7fffffffULL, 0xffdffffffffffbffULL, 0xfffffffeffffffffULL,
    0xfff7fffffffffeffULL, 0xffffffffbfffffffULL, 0xffeffffffffffdffULL, 0xffffffff7fffffffULL, 0xffdffffffffffbffULL, 0xfffffffeffffffffULL, 0xffbffffffffff7ffULL, 0xfffffffdffffffffULL,
    0xffeffffffffffdffULL, 0xffffffff7fffffffULL, 0xffdffffffffffbffULL, 0xfffffffeffffffffULL, 0xffbffffffffff7ffULL, 0xfffffffdffffffffULL, 0xff7fffffffffefffULL, 0xfffffffbffffffffULL,
    0xffdffffffffffbffULL, 0xfffffffeffffffffULL, 0xffbffffffffff7ffULL, 0xfffffffdffffffffULL, 0xff7fffffffffefffULL, 0xfffffffbffffffffULL, 0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL,
    0xffbffffffffff7ffULL, 0xfffffffdffffffffULL, 0xff7fffffffffefffULL, 0xfffffffbffffffffULL, 0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL, 0xffffffefffffffffULL,
    0xff7fffffffffefffULL, 0xfffffffbffffffffULL, 0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL, 0xffffffefffffffffULL, 0xfbffffffffff7fffULL, 0xffffffdfffffffffULL,
    0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL, 0xffffffefffffffffULL, 0xfbffffffffff7fffULL, 0xffffffdfffffffffULL, 0xf7fffffffffeffffULL, 0xffffffbfffffffffULL,
    0xfdffffffffffbfffULL, 0xffffffefffffffffULL, 0xfbffffffffff7fffULL, 0xffffffdfffffffffULL, 0xf7fffffffffeffffULL, 0xffffffbfffffffffULL, 0xeffffffffffdffffULL, 0xffffff7fffffffffULL,
    0xfbffffffffff7fffULL, 0xffffffdfffffffffULL, 0xf7fffffffffeffffULL, 0xffffffbfffffffffULL, 0xeffffffffffdffffULL, 0xffffff7fffffffffULL, 0xdffffffffffbffffULL, 0xfffffeffffffffffULL,
    0xf7fffffffffeffffULL, 0xffffffbfffffffffULL, 0xeffffffffffdffffULL, 0xffffff7fffffffffULL, 0xdffffffffffbffffULL, 0xfffffeffffffffffULL, 0xbffffffffff7ffffULL, 0xfffffdffffffffffULL,
    0xeffffffffffdffffULL, 0xffffff7fffffffffULL, 0xdffffffffffbffffULL, 0xfffffeffffffffffULL, 0xbffffffffff7ffffULL, 0xfffffdffffffffffULL, 0x7fffffffffefffffULL, 0xfffffbffffffffffULL,
    0xdffffffffffbffffULL, 0xfffffeffffffffffULL, 0xbffffffffff7ffffULL, 0xfffffdffffffffffULL, 0x7fffffffffefffffULL, 0xfffffbffffffffffULL, 0xffffffffffdfffffULL, 0xfffff7fffffffffeULL,
    0xbffffffffff7ffffULL, 0xfffffdffffffffffULL, 0x7fffffffffefffffULL, 0xfffffbffffffffffULL, 0xffffffffffdfffffULL, 0xfffff7fffffffffeULL, 0xffffffffffbfffffULL, 0xffffeffffffffffdULL,
    0x7fffffffffefffffULL, 0xfffffbffffffffffULL, 0xffffffffffdfffffULL, 0xfffff7fffffffffeULL, 0xffffffffffbfffffULL, 0xffffeffffffffffdULL, 0xffffffffff7fffffULL, 0xffffdffffffffffbULL,
    0xffffffffffdfffffULL, 0xfffff7fffffffffeULL, 0xffffffffffbfffffULL, 0xffffeffffffffffdULL, 0xffffffffff7fffffULL, 0xffffdffffffffffbULL, 0xfffffffffeffffffULL, 0xffffbffffffffff7ULL,
    0xffffffffffbfffffULL, 0xffffeffffffffffdULL, 0xffffffffff7fffffULL, 0xffffdffffffffffbULL, 0xfffffffffeffffffULL, 0xffffbffffffffff7ULL, 0xfffffffffdffffffULL, 0xffff7fffffffffefULL,
    0xffffffffff7fffffULL, 0xffffdffffffffffbULL, 0xfffffffffeffffffULL, 0xffffbffffffffff7ULL, 0xfffffffffdffffffULL, 0xffff7fffffffffefULL, 0xfffffffffbffffffULL, 0xfffeffffffffffdfULL,
    0xfffffffffeffffffULL, 0xffffbffffffffff7ULL, 0xfffffffffdffffffULL, 0xffff7fffffffffefULL, 0xfffffffffbffffffULL, 0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL,
    0xfffffffffdffffffULL, 0xffff7fffffffffefULL, 0xfffffffffbffffffULL, 0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL, 0xffffffffefffffffULL, 0xfffbffffffffff7fULL,
    0xfffffffffbffffffULL, 0xfffeffffffffffdfULL, 0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL, 0xffffffffefffffffULL, 0xfffbffffffffff7fULL, 0xffffffffdfffffffULL, 0xfff7fffffffffeffULL,
    0xfffffffff7ffffffULL, 0xfffdffffffffffbfULL, 0xffffffffefffffffULL, 0xfffbffffffffff7fULL, 0xffffffffdfffffffULL, 0xfff7fffffffffeffULL, 0xffffffffbfffffffULL, 0xffeffffffffffdffULL,
    0xffffffffefffffffULL, 0xfffbffffffffff7fULL, 0xffffffffdfffffffULL, 0xfff7fffffffffeffULL, 0xffffffffbfffffffULL, 0xffeffffffffffdffULL, 0xffffffff7fffffffULL, 0xffdffffffffffbffULL,
    0xffffffffdfffffffULL, 0xfff7fffffffffeffULL, 0xffffffffbfffffffULL, 0xffeffffffffffdffULL, 0xffffffff7fffffffULL, 0xffdffffffffffbffULL, 0xfffffffeffffffffULL, 0xffbffffffffff7ffULL,
    0xffffffffbfffffffULL, 0xffeffffffffffdffULL, 0xffffffff7fffffffULL, 0xffdffffffffffbffULL, 0xfffffffeffffffffULL, 0xffbffffffffff7ffULL, 0xfffffffdffffffffULL, 0xff7fffffffffefffULL,
    0xffffffff7fffffffULL, 0xffdffffffffffbffULL, 0xfffffffeffffffffULL, 0xffbffffffffff7ffULL, 0xfffffffdffffffffULL, 0xff7fffffffffefffULL, 0xfffffffbffffffffULL, 0xfeffffffffffdfffULL,
    0xfffffffeffffffffULL, 0xffbffffffffff7ffULL, 0xfffffffdffffffffULL, 0xff7fffffffffefffULL, 0xfffffffbffffffffULL, 0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL,
    0xfffffffdffffffffULL, 0xff7fffffffffefffULL, 0xfffffffbffffffffULL, 0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL, 0xffffffefffffffffULL, 0xfbffffffffff7fffULL,
    0xfffffffbffffffffULL, 0xfeffffffffffdfffULL, 0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL, 0xffffffefffffffffULL, 0xfbffffffffff7fffULL, 0xffffffdfffffffffULL, 0xf7fffffffffeffffULL,
    0xfffffff7ffffffffULL, 0xfdffffffffffbfffULL, 0xffffffefffffffffULL, 0xfbffffffffff7fffULL, 0xffffffdfffffffffULL, 0xf7fffffffffeffffULL, 0xffffffbfffffffffULL, 0xeffffffffffdffffULL,
    0xffffffefffffffffULL, 0xfbffffffffff7fffULL, 0xffffffdfffffffffULL, 0xf7fffffffffeffffULL, 0xffffffbfffffffffULL, 0xeffffffffffdffffULL, 0xffffff7fffffffffULL, 0xdffffffffffbffffULL,
    0xffffffdfffffffffULL, 0xf7fffffffffeffffULL, 0xffffffbfffffffffULL, 0xeffffffffffdffffULL, 0xffffff7fffffffffULL, 0xdffffffffffbffffULL, 0xfffffeffffffffffULL, 0xbffffffffff7ffffULL,
    0xffffffbfffffffffULL, 0xeffffffffffdffffULL, 0xffffff7fffffffffULL, 0xdffffffffffbffffULL, 0xfffffeffffffffffULL, 0xbffffffffff7ffffULL, 0xfffffdffffffffffULL, 0x7fffffffffefffffULL,
    0xffffff7fffffffffULL, 0xdffffffffffbffffULL, 0xfffffeffffffffffULL, 0xbffffffffff7ffffULL, 0xfffffdffffffffffULL, 0x7fffffffffefffffULL, 0xfffffbffffffffffULL, 0xffffffffffdfffffULL,
    0xfffffeffffffffffULL, 0xbffffffffff7ffffULL, 0xfffffdffffffffffULL, 0x7fffffffffefffffULL, 0xfffffbffffffffffULL, 0xffffffffffdfffffULL, 0xfffff7fffffffffeULL, 0xffffffffffbfffffULL,
    0xfffffdffffffffffULL, 0x7fffffffffefffffULL, 0xfffffbffffffffffULL, 0xffffffffffdfffffULL, 0xfffff7fffffffffeULL, 0xffffffffffbfffffULL, 0xffffeffffffffffdULL, 0xffffffffff7fffffULL,
    0xfffffbffffffffffULL, 0xffffffffffdfffffULL, 0xfffff7fffffffffeULL, 0xffffffffffbfffffULL, 0xffffeffffffffffdULL, 0xffffffffff7fffffULL, 0xffffdffffffffffbULL, 0xfffffffffeffffffULL};


// pid = 14 (p = 47, 3008 bytes)
uint64_t _47_MASKS_512[47 * 8] = {
    0xffff7ffffffffffeULL, 0xffffffffbfffffffULL, 0xefffffffffffdfffULL, 0xfffff7ffffffffffULL, 0xfffffffffbffffffULL, 0xfefffffffffffdffULL, 0xffffff7fffffffffULL, 0xffffffffffbfffffULL,
    0xfffefffffffffffdULL, 0xffffffff7fffffffULL, 0xdfffffffffffbfffULL, 0xffffefffffffffffULL, 0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL, 0xfffffeffffffffffULL, 0xffffffffff7fffffULL,
    0xfffdfffffffffffbULL, 0xfffffffeffffffffULL, 0xbfffffffffff7fffULL, 0xffffdfffffffffffULL, 0xffffffffefffffffULL, 0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL, 0xfffffffffeffffffULL,
    0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL, 0x7ffffffffffeffffULL, 0xffffbfffffffffffULL, 0xffffffffdfffffffULL, 0xf7ffffffffffefffULL, 0xfffffbffffffffffULL, 0xfffffffffdffffffULL,
    0xfff7ffffffffffefULL, 0xfffffffbffffffffULL, 0xfffffffffffdffffULL, 0xffff7ffffffffffeULL, 0xffffffffbfffffffULL, 0xefffffffffffdfffULL, 0xfffff7ffffffffffULL, 0xfffffffffbffffffULL,
    0xffefffffffffffdfULL, 0xfffffff7ffffffffULL, 0xfffffffffffbffffULL, 0xfffefffffffffffdULL, 0xffffffff7fffffffULL, 0xdfffffffffffbfffULL, 0xffffefffffffffffULL, 0xfffffffff7ffffffULL,
    0xffdfffffffffffbfULL, 0xffffffefffffffffULL, 0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL, 0xfffffffeffffffffULL, 0xbfffffffffff7fffULL, 0xffffdfffffffffffULL, 0xffffffffefffffffULL,
    0xffbfffffffffff7fULL, 0xffffffdfffffffffULL, 0xffffffffffefffffULL, 0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL, 0x7ffffffffffeffffULL, 0xffffbfffffffffffULL, 0xffffffffdfffffffULL,
    0xff7ffffffffffeffULL, 0xffffffbfffffffffULL, 0xffffffffffdfffffULL, 0xfff7ffffffffffefULL, 0xfffffffbffffffffULL, 0xfffffffffffdffffULL, 0xffff7ffffffffffeULL, 0xffffffffbfffffffULL,
    0xfefffffffffffdffULL, 0xffffff7fffffffffULL, 0xffffffffffbfffffULL, 0xffefffffffffffdfULL, 0xfffffff7ffffffffULL, 0xfffffffffffbffffULL, 0xfffefffffffffffdULL, 0xffffffff7fffffffULL,
    0xfdfffffffffffbffULL, 0xfffffeffffffffffULL, 0xffffffffff7fffffULL, 0xffdfffffffffffbfULL, 0xffffffefffffffffULL, 0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL, 0xfffffffeffffffffULL,
    0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL, 0xfffffffffeffffffULL, 0xffbfffffffffff7fULL, 0xffffffdfffffffffULL, 0xffffffffffefffffULL, 0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL,
    0xf7ffffffffffefffULL, 0xfffffbffffffffffULL, 0xfffffffffdffffffULL, 0xff7ffffffffffeffULL, 0xffffffbfffffffffULL, 0xffffffffffdfffffULL, 0xfff7ffffffffffefULL, 0xfffffffbffffffffULL,
    0xefffffffffffdfffULL, 0xfffff7ffffffffffULL, 0xfffffffffbffffffULL, 0xfefffffffffffdffULL, 0xffffff7fffffffffULL, 0xffffffffffbfffffULL, 0xffefffffffffffdfULL, 0xfffffff7ffffffffULL,
    0xdfffffffffffbfffULL, 0xffffefffffffffffULL, 0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL, 0xfffffeffffffffffULL, 0xffffffffff7fffffULL, 0xffdfffffffffffbfULL, 0xffffffefffffffffULL,
    0xbfffffffffff7fffULL, 0xffffdfffffffffffULL, 0xffffffffefffffffULL, 0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL, 0xfffffffffeffffffULL, 0xffbfffffffffff7fULL, 0xffffffdfffffffffULL,
    0x7ffffffffffeffffULL, 0xffffbfffffffffffULL, 0xffffffffdfffffffULL, 0xf7ffffffffffefffULL, 0xfffffbffffffffffULL, 0xfffffffffdffffffULL, 0xff7ffffffffffeffULL, 0xffffffbfffffffffULL,
    0xfffffffffffdffffULL, 0xffff7ffffffffffeULL, 0xffffffffbfffffffULL, 0xefffffffffffdfffULL, 0xfffff7ffffffffffULL, 0xfffffffffbffffffULL, 0xfefffffffffffdffULL, 0xffffff7fffffffffULL,
    0xfffffffffffbffffULL, 0xfffefffffffffffdULL, 0xffffffff7fffffffULL, 0xdfffffffffffbfffULL, 0xffffefffffffffffULL, 0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL, 0xfffffeffffffffffULL,
    0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL, 0xfffffffeffffffffULL, 0xbfffffffffff7fffULL, 0xffffdfffffffffffULL, 0xffffffffefffffffULL, 0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL,
    0xffffffffffefffffULL, 0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL, 0x7ffffffffffeffffULL, 0xffffbfffffffffffULL, 0xffffffffdfffffffULL, 0xf7ffffffffffefffULL, 0xfffffbffffffffffULL,
    0xffffffffffdfffffULL, 0xfff7ffffffffffefULL, 0xfffffffbffffffffULL, 0xfffffffffffdffffULL, 0xffff7ffffffffffeULL, 0xffffffffbfffffffULL, 0xefffffffffffdfffULL, 0xfffff7ffffffffffULL,
    0xffffffffffbfffffULL, 0xffefffffffffffdfULL, 0xfffffff7ffffffffULL, 0xfffffffffffbffffULL, 0xfffefffffffffffdULL, 0xffffffff7fffffffULL, 0xdfffffffffffbfffULL, 0xffffefffffffffffULL,
    0xffffffffff7fffffULL, 0xffdfffffffffffbfULL, 0xffffffefffffffffULL, 0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL, 0xfffffffeffffffffULL, 0xbfffffffffff7fffULL, 0xffffdfffffffffffULL,
    0xfffffffffeffffffULL, 0xffbfffffffffff7fULL, 0xffffffdfffffffffULL, 0xffffffffffefffffULL, 0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL, 0x7ffffffffffeffffULL, 0xffffbfffffffffffULL,
    0xfffffffffdffffffULL, 0xff7ffffffffffeffULL, 0xffffffbfffffffffULL, 0xffffffffffdfffffULL, 0xfff7ffffffffffefULL, 0xfffffffbffffffffULL, 0xfffffffffffdffffULL, 0xffff7ffffffffffeULL,
    0xfffffffffbffffffULL, 0xfefffffffffffdffULL, 0xffffff7fffffffffULL, 0xffffffffffbfffffULL, 0xffefffffffffffdfULL, 0xfffffff7ffffffffULL, 0xfffffffffffbffffULL, 0xfffefffffffffffdULL,
    0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL, 0xfffffeffffffffffULL, 0xffffffffff7fffffULL, 0xffdfffffffffffbfULL, 0xffffffefffffffffULL, 0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL,
    0xffffffffefffffffULL, 0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL, 0xfffffffffeffffffULL, 0xffbfffffffffff7fULL, 0xffffffdfffffffffULL, 0xffffffffffefffffULL, 0xfffbfffffffffff7ULL,
    0xffffffffdfffffffULL, 0xf7ffffffffffefffULL, 0xfffffbffffffffffULL, 0xfffffffffdffffffULL, 0xff7ffffffffffeffULL, 0xffffffbfffffffffULL, 0xffffffffffdfffffULL, 0xfff7ffffffffffefULL,
    0xffffffffbfffffffULL, 0xefffffffffffdfffULL, 0xfffff7ffffffffffULL, 0xfffffffffbffffffULL, 0xfefffffffffffdffULL, 0xffffff7fffffffffULL, 0xffffffffffbfffffULL, 0xffefffffffffffdfULL,
    0xffffffff7fffffffULL, 0xdfffffffffffbfffULL, 0xffffefffffffffffULL, 0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL, 0xfffffeffffffffffULL, 0xffffffffff7fffffULL, 0xffdfffffffffffbfULL,
    0xfffffffeffffffffULL, 0xbfffffffffff7fffULL, 0xffffdfffffffffffULL, 0xffffffffefffffffULL, 0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL, 0xfffffffffeffffffULL, 0xffbfffffffffff7fULL,
    0xfffffffdffffffffULL, 0x7ffffffffffeffffULL, 0xffffbfffffffffffULL, 0xffffffffdfffffffULL, 0xf7ffffffffffefffULL, 0xfffffbffffffffffULL, 0xfffffffffdffffffULL, 0xff7ffffffffffeffULL,
    0xfffffffbffffffffULL, 0xfffffffffffdffffULL, 0xffff7ffffffffffeULL, 0xffffffffbfffffffULL, 0xefffffffffffdfffULL, 0xfffff7ffffffffffULL, 0xfffffffffbffffffULL, 0xfefffffffffffdffULL,
    0xfffffff7ffffffffULL, 0xfffffffffffbffffULL, 0xfffefffffffffffdULL, 0xffffffff7fffffffULL, 0xdfffffffffffbfffULL, 0xffffefffffffffffULL, 0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL,
    0xffffffefffffffffULL, 0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL, 0xfffffffeffffffffULL, 0xbfffffffffff7fffULL, 0xffffdfffffffffffULL, 0xffffffffefffffffULL, 0xfbfffffffffff7ffULL,
    0xffffffdfffffffffULL, 0xffffffffffefffffULL, 0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL, 0x7ffffffffffeffffULL, 0xffffbfffffffffffULL, 0xffffffffdfffffffULL, 0xf7ffffffffffefffULL,
    0xffffffbfffffffffULL, 0xffffffffffdfffffULL, 0xfff7ffffffffffefULL, 0xfffffffbffffffffULL, 0xfffffffffffdffffULL, 0xffff7ffffffffffeULL, 0xffffffffbfffffffULL, 0xefffffffffffdfffULL,
    0xffffff7fffffffffULL, 0xffffffffffbfffffULL, 0xffefffffffffffdfULL, 0xfffffff7ffffffffULL, 0xfffffffffffbffffULL, 0xfffefffffffffffdULL, 0xffffffff7fffffffULL, 0xdfffffffffffbfffULL,
    0xfffffeffffffffffULL, 0xffffffffff7fffffULL, 0xffdfffffffffffbfULL, 0xffffffefffffffffULL, 0xfffffffffff7ffffULL, 0xfffdfffffffffffbULL, 0xfffffffeffffffffULL, 0xbfffffffffff7fffULL,
    0xfffffdffffffffffULL, 0xfffffffffeffffffULL, 0xffbfffffffffff7fULL, 0xffffffdfffffffffULL, 0xffffffffffefffffULL, 0xfffbfffffffffff7ULL, 0xfffffffdffffffffULL, 0x7ffffffffffeffffULL,
    0xfffffbffffffffffULL, 0xfffffffffdffffffULL, 0xff7ffffffffffeffULL, 0xffffffbfffffffffULL, 0xffffffffffdfffffULL, 0xfff7ffffffffffefULL, 0xfffffffbffffffffULL, 0xfffffffffffdffffULL,
    0xfffff7ffffffffffULL, 0xfffffffffbffffffULL, 0xfefffffffffffdffULL, 0xffffff7fffffffffULL, 0xffffffffffbfffffULL, 0xffefffffffffffdfULL, 0xfffffff7ffffffffULL, 0xfffffffffffbffffULL,
    0xffffefffffffffffULL, 0xfffffffff7ffffffULL, 0xfdfffffffffffbffULL, 0xfffffeffffffffffULL, 0xffffffffff7fffffULL, 0xffdfffffffffffbfULL, 0xffffffefffffffffULL, 0xfffffffffff7ffffULL,
    0xffffdfffffffffffULL, 0xffffffffefffffffULL, 0xfbfffffffffff7ffULL, 0xfffffdffffffffffULL, 0xfffffffffeffffffULL, 0xffbfffffffffff7fULL, 0xffffffdfffffffffULL, 0xffffffffffefffffULL,
    0xffffbfffffffffffULL, 0xffffffffdfffffffULL, 0xf7ffffffffffefffULL, 0xfffffbffffffffffULL, 0xfffffffffdffffffULL, 0xff7ffffffffffeffULL, 0xffffffbfffffffffULL, 0xffffffffffdfffffULL};

// pid = 15 (p = 53, 3392 bytes)
uint64_t _53_MASKS_512[53 * 8] = {
    0xffdffffffffffffeULL, 0xfffffbffffffffffULL, 0xffffffff7fffffffULL, 0xffffffffffefffffULL, 0xbffffffffffffdffULL, 0xfff7ffffffffffffULL, 0xfffffeffffffffffULL, 0xffffffffdfffffffULL,
    0xffbffffffffffffdULL, 0xfffff7ffffffffffULL, 0xfffffffeffffffffULL, 0xffffffffffdfffffULL, 0x7ffffffffffffbffULL, 0xffefffffffffffffULL, 0xfffffdffffffffffULL, 0xffffffffbfffffffULL,
    0xff7ffffffffffffbULL, 0xffffefffffffffffULL, 0xfffffffdffffffffULL, 0xffffffffffbfffffULL, 0xfffffffffffff7ffULL, 0xffdffffffffffffeULL, 0xfffffbffffffffffULL, 0xffffffff7fffffffULL,
    0xfefffffffffffff7ULL, 0xffffdfffffffffffULL, 0xfffffffbffffffffULL, 0xffffffffff7fffffULL, 0xffffffffffffefffULL, 0xffbffffffffffffdULL, 0xfffff7ffffffffffULL, 0xfffffffeffffffffULL,
    0xfdffffffffffffefULL, 0xffffbfffffffffffULL, 0xfffffff7ffffffffULL, 0xfffffffffeffffffULL, 0xffffffffffffdfffULL, 0xff7ffffffffffffbULL, 0xffffefffffffffffULL, 0xfffffffdffffffffULL,
    0xfbffffffffffffdfULL, 0xffff7fffffffffffULL, 0xffffffefffffffffULL, 0xfffffffffdffffffULL, 0xffffffffffffbfffULL, 0xfefffffffffffff7ULL, 0xffffdfffffffffffULL, 0xfffffffbffffffffULL,
    0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL, 0xffffffdfffffffffULL, 0xfffffffffbffffffULL, 0xffffffffffff7fffULL, 0xfdffffffffffffefULL, 0xffffbfffffffffffULL, 0xfffffff7ffffffffULL,
    0xefffffffffffff7fULL, 0xfffdffffffffffffULL, 0xffffffbfffffffffULL, 0xfffffffff7ffffffULL, 0xfffffffffffeffffULL, 0xfbffffffffffffdfULL, 0xffff7fffffffffffULL, 0xffffffefffffffffULL,
    0xdffffffffffffeffULL, 0xfffbffffffffffffULL, 0xffffff7fffffffffULL, 0xffffffffefffffffULL, 0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL, 0xffffffdfffffffffULL,
    0xbffffffffffffdffULL, 0xfff7ffffffffffffULL, 0xfffffeffffffffffULL, 0xffffffffdfffffffULL, 0xfffffffffffbffffULL, 0xefffffffffffff7fULL, 0xfffdffffffffffffULL, 0xffffffbfffffffffULL,
    0x7ffffffffffffbffULL, 0xffefffffffffffffULL, 0xfffffdffffffffffULL, 0xffffffffbfffffffULL, 0xfffffffffff7ffffULL, 0xdffffffffffffeffULL, 0xfffbffffffffffffULL, 0xffffff7fffffffffULL,
    0xfffffffffffff7ffULL, 0xffdffffffffffffeULL, 0xfffffbffffffffffULL, 0xffffffff7fffffffULL, 0xffffffffffefffffULL, 0xbffffffffffffdffULL, 0xfff7ffffffffffffULL, 0xfffffeffffffffffULL,
    0xffffffffffffefffULL, 0xffbffffffffffffdULL, 0xfffff7ffffffffffULL, 0xfffffffeffffffffULL, 0xffffffffffdfffffULL, 0x7ffffffffffffbffULL, 0xffefffffffffffffULL, 0xfffffdffffffffffULL,
    0xffffffffffffdfffULL, 0xff7ffffffffffffbULL, 0xffffefffffffffffULL, 0xfffffffdffffffffULL, 0xffffffffffbfffffULL, 0xfffffffffffff7ffULL, 0xffdffffffffffffeULL, 0xfffffbffffffffffULL,
    0xffffffffffffbfffULL, 0xfefffffffffffff7ULL, 0xffffdfffffffffffULL, 0xfffffffbffffffffULL, 0xffffffffff7fffffULL, 0xffffffffffffefffULL, 0xffbffffffffffffdULL, 0xfffff7ffffffffffULL,
    0xffffffffffff7fffULL, 0xfdffffffffffffefULL, 0xffffbfffffffffffULL, 0xfffffff7ffffffffULL, 0xfffffffffeffffffULL, 0xffffffffffffdfffULL, 0xff7ffffffffffffbULL, 0xffffefffffffffffULL,
    0xfffffffffffeffffULL, 0xfbffffffffffffdfULL, 0xffff7fffffffffffULL, 0xffffffefffffffffULL, 0xfffffffffdffffffULL, 0xffffffffffffbfffULL, 0xfefffffffffffff7ULL, 0xffffdfffffffffffULL,
    0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL, 0xffffffdfffffffffULL, 0xfffffffffbffffffULL, 0xffffffffffff7fffULL, 0xfdffffffffffffefULL, 0xffffbfffffffffffULL,
    0xfffffffffffbffffULL, 0xefffffffffffff7fULL, 0xfffdffffffffffffULL, 0xffffffbfffffffffULL, 0xfffffffff7ffffffULL, 0xfffffffffffeffffULL, 0xfbffffffffffffdfULL, 0xffff7fffffffffffULL,
    0xfffffffffff7ffffULL, 0xdffffffffffffeffULL, 0xfffbffffffffffffULL, 0xffffff7fffffffffULL, 0xffffffffefffffffULL, 0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL,
    0xffffffffffefffffULL, 0xbffffffffffffdffULL, 0xfff7ffffffffffffULL, 0xfffffeffffffffffULL, 0xffffffffdfffffffULL, 0xfffffffffffbffffULL, 0xefffffffffffff7fULL, 0xfffdffffffffffffULL,
    0xffffffffffdfffffULL, 0x7ffffffffffffbffULL, 0xffefffffffffffffULL, 0xfffffdffffffffffULL, 0xffffffffbfffffffULL, 0xfffffffffff7ffffULL, 0xdffffffffffffeffULL, 0xfffbffffffffffffULL,
    0xffffffffffbfffffULL, 0xfffffffffffff7ffULL, 0xffdffffffffffffeULL, 0xfffffbffffffffffULL, 0xffffffff7fffffffULL, 0xffffffffffefffffULL, 0xbffffffffffffdffULL, 0xfff7ffffffffffffULL,
    0xffffffffff7fffffULL, 0xffffffffffffefffULL, 0xffbffffffffffffdULL, 0xfffff7ffffffffffULL, 0xfffffffeffffffffULL, 0xffffffffffdfffffULL, 0x7ffffffffffffbffULL, 0xffefffffffffffffULL,
    0xfffffffffeffffffULL, 0xffffffffffffdfffULL, 0xff7ffffffffffffbULL, 0xffffefffffffffffULL, 0xfffffffdffffffffULL, 0xffffffffffbfffffULL, 0xfffffffffffff7ffULL, 0xffdffffffffffffeULL,
    0xfffffffffdffffffULL, 0xffffffffffffbfffULL, 0xfefffffffffffff7ULL, 0xffffdfffffffffffULL, 0xfffffffbffffffffULL, 0xffffffffff7fffffULL, 0xffffffffffffefffULL, 0xffbffffffffffffdULL,
    0xfffffffffbffffffULL, 0xffffffffffff7fffULL, 0xfdffffffffffffefULL, 0xffffbfffffffffffULL, 0xfffffff7ffffffffULL, 0xfffffffffeffffffULL, 0xffffffffffffdfffULL, 0xff7ffffffffffffbULL,
    0xfffffffff7ffffffULL, 0xfffffffffffeffffULL, 0xfbffffffffffffdfULL, 0xffff7fffffffffffULL, 0xffffffefffffffffULL, 0xfffffffffdffffffULL, 0xffffffffffffbfffULL, 0xfefffffffffffff7ULL,
    0xffffffffefffffffULL, 0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL, 0xffffffdfffffffffULL, 0xfffffffffbffffffULL, 0xffffffffffff7fffULL, 0xfdffffffffffffefULL,
    0xffffffffdfffffffULL, 0xfffffffffffbffffULL, 0xefffffffffffff7fULL, 0xfffdffffffffffffULL, 0xffffffbfffffffffULL, 0xfffffffff7ffffffULL, 0xfffffffffffeffffULL, 0xfbffffffffffffdfULL,
    0xffffffffbfffffffULL, 0xfffffffffff7ffffULL, 0xdffffffffffffeffULL, 0xfffbffffffffffffULL, 0xffffff7fffffffffULL, 0xffffffffefffffffULL, 0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL,
    0xffffffff7fffffffULL, 0xffffffffffefffffULL, 0xbffffffffffffdffULL, 0xfff7ffffffffffffULL, 0xfffffeffffffffffULL, 0xffffffffdfffffffULL, 0xfffffffffffbffffULL, 0xefffffffffffff7fULL,
    0xfffffffeffffffffULL, 0xffffffffffdfffffULL, 0x7ffffffffffffbffULL, 0xffefffffffffffffULL, 0xfffffdffffffffffULL, 0xffffffffbfffffffULL, 0xfffffffffff7ffffULL, 0xdffffffffffffeffULL,
    0xfffffffdffffffffULL, 0xffffffffffbfffffULL, 0xfffffffffffff7ffULL, 0xffdffffffffffffeULL, 0xfffffbffffffffffULL, 0xffffffff7fffffffULL, 0xffffffffffefffffULL, 0xbffffffffffffdffULL,
    0xfffffffbffffffffULL, 0xffffffffff7fffffULL, 0xffffffffffffefffULL, 0xffbffffffffffffdULL, 0xfffff7ffffffffffULL, 0xfffffffeffffffffULL, 0xffffffffffdfffffULL, 0x7ffffffffffffbffULL,
    0xfffffff7ffffffffULL, 0xfffffffffeffffffULL, 0xffffffffffffdfffULL, 0xff7ffffffffffffbULL, 0xffffefffffffffffULL, 0xfffffffdffffffffULL, 0xffffffffffbfffffULL, 0xfffffffffffff7ffULL,
    0xffffffefffffffffULL, 0xfffffffffdffffffULL, 0xffffffffffffbfffULL, 0xfefffffffffffff7ULL, 0xffffdfffffffffffULL, 0xfffffffbffffffffULL, 0xffffffffff7fffffULL, 0xffffffffffffefffULL,
    0xffffffdfffffffffULL, 0xfffffffffbffffffULL, 0xffffffffffff7fffULL, 0xfdffffffffffffefULL, 0xffffbfffffffffffULL, 0xfffffff7ffffffffULL, 0xfffffffffeffffffULL, 0xffffffffffffdfffULL,
    0xffffffbfffffffffULL, 0xfffffffff7ffffffULL, 0xfffffffffffeffffULL, 0xfbffffffffffffdfULL, 0xffff7fffffffffffULL, 0xffffffefffffffffULL, 0xfffffffffdffffffULL, 0xffffffffffffbfffULL,
    0xffffff7fffffffffULL, 0xffffffffefffffffULL, 0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL, 0xffffffdfffffffffULL, 0xfffffffffbffffffULL, 0xffffffffffff7fffULL,
    0xfffffeffffffffffULL, 0xffffffffdfffffffULL, 0xfffffffffffbffffULL, 0xefffffffffffff7fULL, 0xfffdffffffffffffULL, 0xffffffbfffffffffULL, 0xfffffffff7ffffffULL, 0xfffffffffffeffffULL,
    0xfffffdffffffffffULL, 0xffffffffbfffffffULL, 0xfffffffffff7ffffULL, 0xdffffffffffffeffULL, 0xfffbffffffffffffULL, 0xffffff7fffffffffULL, 0xffffffffefffffffULL, 0xfffffffffffdffffULL,
    0xfffffbffffffffffULL, 0xffffffff7fffffffULL, 0xffffffffffefffffULL, 0xbffffffffffffdffULL, 0xfff7ffffffffffffULL, 0xfffffeffffffffffULL, 0xffffffffdfffffffULL, 0xfffffffffffbffffULL,
    0xfffff7ffffffffffULL, 0xfffffffeffffffffULL, 0xffffffffffdfffffULL, 0x7ffffffffffffbffULL, 0xffefffffffffffffULL, 0xfffffdffffffffffULL, 0xffffffffbfffffffULL, 0xfffffffffff7ffffULL,
    0xffffefffffffffffULL, 0xfffffffdffffffffULL, 0xffffffffffbfffffULL, 0xfffffffffffff7ffULL, 0xffdffffffffffffeULL, 0xfffffbffffffffffULL, 0xffffffff7fffffffULL, 0xffffffffffefffffULL,
    0xffffdfffffffffffULL, 0xfffffffbffffffffULL, 0xffffffffff7fffffULL, 0xffffffffffffefffULL, 0xffbffffffffffffdULL, 0xfffff7ffffffffffULL, 0xfffffffeffffffffULL, 0xffffffffffdfffffULL,
    0xffffbfffffffffffULL, 0xfffffff7ffffffffULL, 0xfffffffffeffffffULL, 0xffffffffffffdfffULL, 0xff7ffffffffffffbULL, 0xffffefffffffffffULL, 0xfffffffdffffffffULL, 0xffffffffffbfffffULL,
    0xffff7fffffffffffULL, 0xffffffefffffffffULL, 0xfffffffffdffffffULL, 0xffffffffffffbfffULL, 0xfefffffffffffff7ULL, 0xffffdfffffffffffULL, 0xfffffffbffffffffULL, 0xffffffffff7fffffULL,
    0xfffeffffffffffffULL, 0xffffffdfffffffffULL, 0xfffffffffbffffffULL, 0xffffffffffff7fffULL, 0xfdffffffffffffefULL, 0xffffbfffffffffffULL, 0xfffffff7ffffffffULL, 0xfffffffffeffffffULL,
    0xfffdffffffffffffULL, 0xffffffbfffffffffULL, 0xfffffffff7ffffffULL, 0xfffffffffffeffffULL, 0xfbffffffffffffdfULL, 0xffff7fffffffffffULL, 0xffffffefffffffffULL, 0xfffffffffdffffffULL,
    0xfffbffffffffffffULL, 0xffffff7fffffffffULL, 0xffffffffefffffffULL, 0xfffffffffffdffffULL, 0xf7ffffffffffffbfULL, 0xfffeffffffffffffULL, 0xffffffdfffffffffULL, 0xfffffffffbffffffULL,
    0xfff7ffffffffffffULL, 0xfffffeffffffffffULL, 0xffffffffdfffffffULL, 0xfffffffffffbffffULL, 0xefffffffffffff7fULL, 0xfffdffffffffffffULL, 0xffffffbfffffffffULL, 0xfffffffff7ffffffULL,
    0xffefffffffffffffULL, 0xfffffdffffffffffULL, 0xffffffffbfffffffULL, 0xfffffffffff7ffffULL, 0xdffffffffffffeffULL, 0xfffbffffffffffffULL, 0xffffff7fffffffffULL, 0xffffffffefffffffULL};

    */

__inline uint32_t modinv_1(uint32_t a, uint32_t p) {

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

void MultiSegSieve2(uint32_t *count, uint32_t *primes_dev, uint32_t *pinv_dev,
    int start_block, int num_blocks, int maxp, uint64_t maxID)
{
    uint32_t bid;

#pragma omp parallel for
    for (bid = startprime; bid < maxp; bid++)
    {
        uint32_t p = primes_dev[bid];
        pinv_dev[bid] = p - modinv_1(30, p);
    }

#pragma omp parallel for
    for (bid = start_block; bid < start_block + num_blocks; bid++)
    {
        uint32_t sieve[BSIZE];
        //uint32_t *sieve;
        uint32_t pid;
        int i, j, k, lnum;
        uint32_t p = 0;
        uint32_t residues[NLINES] = { 1, 7, 11, 13, 17, 19, 23, 29};
        uint64_t block_start = (uint64_t)bid * BSIZE * 32 * 30;

        count[bid] = 0;
        //sieve = (uint32_t *)xmalloc_align(BSIZE * sizeof(uint32_t));

        // for each line in the sieve there are BSIZE * 32 bits.  
        // In total, the lines of the block represent BSIZE * 32 * 30 integers
        for (lnum = 0; lnum < NLINES; lnum++)
        {
            uint64_t line_start = block_start + (uint64_t)residues[lnum];
            uint64_t tmp;
            int mask_step, mask_num;
            uint32_t offset;
            uint64_t *sieve64 = (uint64_t *)sieve;
            __m512i zero = _mm512_set1_epi32(0);
            __m512i one = _mm512_set1_epi32(1);
            __m512i bsz = _mm512_set1_epi32(BSIZE * 32);
            __m512i v31 = _mm512_set1_epi32(31);
            __m512i vls = _mm512_set1_epi64(line_start);
            __m512i vfull = _mm512_set1_epi32(0xffffffff);
            __m512i vmaskfull[17];
            __m512i vmask;


            ///////////////////////////////////////////////////////////////////////////
            // presieve
            pid = 3;
            p = primes_dev[pid];

            
            // compute the offset into this block using the precomputed inverse modulo 30
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0; k < p; k++)
            {
                vmaskfull[k] = _mm512_load_epi32(&_7_MASKS_512[k*8]);
            }

            for (k = 0, mask_step = _512_MOD_P[1], mask_num = offset; 
                k < (BSIZE * 32 >> 9); k++)
            {
                _mm512_store_epi64(&sieve64[k * 8], vmaskfull[mask_num]);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 7 + mask_num;
            }

            pid = 4;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0; k < p; k++)
            {
                vmaskfull[k] = _mm512_load_epi32(&_11_MASKS_512[k * 8]);
            }

            for (k = 0, mask_step = _512_MOD_P[2], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_and_epi32(vmaskfull[mask_num], vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 11 + mask_num;
            }

            pid = 5;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0; k < p; k++)
            {
                vmaskfull[k] = _mm512_load_epi32(&_13_MASKS_512[k * 8]);
            }

            for (k = 0, mask_step = _512_MOD_P[3], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_and_epi32(vmaskfull[mask_num], vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 13 + mask_num;
            }

            pid = 6;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0; k < p; k++)
            {
                vmaskfull[k] = _mm512_load_epi32(&_17_MASKS_512[k * 8]);
            }

            for (k = 0, mask_step = _512_MOD_P[4], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_and_epi32(vmaskfull[mask_num], vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 17 + mask_num;
            }                      


            pid = 7;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[5], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_19_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 19 + mask_num;
            }

            pid = 8;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[6], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_23_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 23 + mask_num;
            }

            pid = 9;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[7], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_29_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 29 + mask_num;
            }

            pid = 10;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[8], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_31_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 31 + mask_num;
            }

            pid = 11;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[9], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_37_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 37 + mask_num;
            }

            pid = 12;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[10], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_41_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 41 + mask_num;
            }

            pid = 13;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[11], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_43_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 43 + mask_num;
            }

            pid = 14;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[12], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_47_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 47 + mask_num;
            }

            pid = 15;
            p = primes_dev[pid];
            tmp = (uint64_t)pinv_dev[pid] * (line_start % (uint64_t)p);
            offset = (uint32_t)(tmp % (uint64_t)p);

            for (k = 0, mask_step = _512_MOD_P[13], mask_num = offset;
                k < (BSIZE * 32 >> 9); k++)
            {
                __m512i vsieve = _mm512_load_epi32(&sieve64[k * 8]);
                vmask = _mm512_load_epi32(&_53_MASKS_512[mask_num * 8]);
                vmask = _mm512_and_epi32(vmask, vsieve);
                _mm512_store_epi64(&sieve64[k * 8], vmask);
                mask_num -= mask_step;
                if (mask_num < 0) mask_num = 53 + mask_num;
            }


            ///////////////////////////////////////////////////////////////////////////
            // regular sieve

            for (pid = 16; pid < (maxp - 16); pid += 16)
            {                
                __m512i vprime;
                __m512i vpinv;
                __m512i a, b, c, d, s, ep, op;
                __mmask16 wmask;
                __declspec(align(64)) uint32_t t[16];
                __declspec(align(64)) uint32_t t2[16];

                // load primes and inverses
                vprime = _mm512_load_epi32(primes_dev + pid);
                vpinv = _mm512_load_epi32(pinv_dev + pid);                

                // take line_start mod p so it is 32-bit
                // (there does not appear to be a 64-bit vector integer multiply operation.)
                ep = _mm512_mask_swizzle_epi32(zero, 0x5555, vprime, _MM_SWIZ_REG_NONE);
                op = _mm512_mask_swizzle_epi32(zero, 0x5555, vprime, _MM_SWIZ_REG_CDAB);                

                b = _mm512_rem_epu64(vls, ep);
                c = _mm512_rem_epu64(vls, op);

                // re-combine two 64-bit remainders into one 32-bit vector again
                c = _mm512_swizzle_epi32(c, _MM_SWIZ_REG_CDAB);
                a = _mm512_or_epi32(c, b);

                // multiply with pinv
                b = _mm512_mulhi_epu32(a, vpinv);
                a = _mm512_mullo_epi32(a, vpinv);
                
                // re-arrange the two 32-bit vectors into 2 64-bit vector products.
                // swizzle-swap the hi products to get the even 64-bit products
                c = _mm512_mask_swizzle_epi32(a, 0xaaaa, b, _MM_SWIZ_REG_CDAB);

                // swizzle-swap the lo products to get the odd 64-bit products
                d = _mm512_mask_swizzle_epi32(b, 0x5555, a, _MM_SWIZ_REG_CDAB);

                // offset = product mod p
                b = _mm512_rem_epu64(d, op);
                a = _mm512_rem_epu64(c, ep);

                // re-combine two 64-bit modular products into one 32-bit vector again
                b = _mm512_swizzle_epi32(b, _MM_SWIZ_REG_CDAB);
                a = _mm512_or_epi32(b, a);
                               
                // hybrid collision-free sieve that is correct 
                // (unlike gather/scatter) yet faster than a fully non-vector sieve.
                wmask = _mm512_cmp_epi32_mask(a, bsz, _MM_CMPINT_LT);
                while (wmask > 0)
                {
                    c = _mm512_srli_epi32(a, 5);        // word location
                    b = _mm512_and_epi32(a, v31);
                    b = _mm512_sllv_epi32(one, b);        // bit location
                    b = _mm512_andnot_epi32(b, vfull);      // sieve &= not(bit location)

                    _mm512_store_epi32(t, c);
                    _mm512_store_epi32(t2, b);

                    // this is where gather/scatter fails... because more
                    // than one offset may be at the same t[k] and only
                    // the last t2 written will survive when using scatter.
                    // we'd have to change to a write only sieve (byte-wide
                    // writes of 0) to use gather/scatter.  unsure if the
                    // resulting expansion of the sieve area will be a win
                    // with that approach.
                    if (wmask == 0xffff)
                    {
                        sieve[t[0]] &= t2[0];
                        sieve[t[1]] &= t2[1];
                        sieve[t[2]] &= t2[2];
                        sieve[t[3]] &= t2[3];
                        sieve[t[4]] &= t2[4];
                        sieve[t[5]] &= t2[5];
                        sieve[t[6]] &= t2[6];
                        sieve[t[7]] &= t2[7];
                        sieve[t[8]] &= t2[8];
                        sieve[t[9]] &= t2[9];
                        sieve[t[10]] &= t2[10];
                        sieve[t[11]] &= t2[11];
                        sieve[t[12]] &= t2[12];
                        sieve[t[13]] &= t2[13];
                        sieve[t[14]] &= t2[14];
                        sieve[t[15]] &= t2[15];
                    }
                    else
                    {
#pragma unroll(16)
                        for (k = 0; k < 16; k++)
                        {
                            if (wmask & (1 << k))
                            {
                                sieve[t[k]] &= t2[k];
                            }
                        }
                    }

                    a = _mm512_add_epi32(a, vprime);
                    wmask = _mm512_cmp_epi32_mask(a, bsz, _MM_CMPINT_LT);
                }

            }

            // ending peel loop
            for (; pid < maxp; pid++)
            {
                p = primes_dev[pid];

                // compute the offset into this block using the precomputed inverse modulo 30
                tmp = (uint64_t)pinv_dev[pid] * line_start;
                offset = (uint32_t)(tmp % (uint64_t)p);

                // then sieve the block
                for (i = offset; i < BSIZE * 32; i += p)
                {
                    sieve[i >> 5] &= (~(1 << (i & 31)));
                }
            }
            

            ///////////////////////////////////////////////////////////////////////////
            // clear last bits

            if (bid == (num_blocks + start_block - 1))
            {
                // zero locations outside the requested range.
                // first zero whole 32-bit words.
                for (j = BSIZE - 1; j >= 0; j--)
                {
                    if ((block_start + j * 32 * 30 + residues[lnum]) > maxID)
                        sieve[j] = 0;
                    else
                        break;
                }

                // then any last bits
                for (k = 31; k >= 0; k--)
                {
                    if ((block_start + (j * 32 + k) * 30 + residues[lnum]) > maxID)
                        sieve[j] &= (~(1 << (k & 31)));
                    else
                        break;
                }
            }

            ///////////////////////////////////////////////////////////////////////////
            // count primes found in this block
            k = 0;
#pragma ivdep
            for (j = 0; j < BSIZE; j++)
            {
                k += _mm_countbits_32(sieve[j]);
            }

            count[bid] += k;
        }        

        //align_free(sieve);
    }

    return;
}

///////////////////////////////////////////////////////////////////////////
// public interface functions
///////////////////////////////////////////////////////////////////////////

// call once to create the bitmap
int gen_primes(uint64_t N, int threads)
{
    uint32_t Nsmall;
    int numblocks;
    int primes_per_thread;
    uint64_t array_size;
    uint32_t* primes;
    uint32_t* pinv;
    uint32_t* device_primes;
    uint32_t* device_pinv;
    uint32_t* block_counts;
    uint32_t np;
    unsigned int thandle;
    uint32_t* block_counts_on_host;
    clock_t start, stop;
    int i;

    // timing variables
    struct timeval stopt;	// stop time of this job
    struct timeval startt;	// start time of this job
    double t_time;
    
    Nsmall = (uint32_t)sqrt((double)N);

    printf("generating primes up to %lu using primes up to %u\n", N, Nsmall);

    if (N > 1000000000000)
    {
        printf("input range too large, limit is 10e11");
        exit(0);
    }

    // find seed primes
    start = clock();
    primes = tiny_soe(Nsmall, &np);
    printf("%d small primes (< %d) found\n", np, Nsmall);

    // allocate space for the inverse of p mod the wheel size, 
    // which is used to speed up a time critical operation.
    pinv = (uint32_t *)xmalloc_align(np*sizeof(uint32_t));

    stop = clock();

    printf("init time was %1.2f sec\n", (double)(stop - start) / (double)CLOCKS_PER_SEC);

    gettimeofday(&startt, NULL);

    omp_set_num_threads(threads);

    //if (N < 10000000)
    //{
    //    BSIZE = 256;
    //}
    //else if (N < 100000000)
    //{
    //    BSIZE = 512;
    //}
    //else if (N < 1000000000)
    //{
    //    BSIZE = 2048;
    //}
    //else if (N < 10000000000)
    //{
    //    BSIZE = 4096;
    //}
    //else
    //{
    //    BSIZE = 8192;
    //}

    numblocks = (N / 30 / BSIZE / 32 + 1);

    // init result array of block counts
    block_counts = (uint32_t *)xmalloc_align(numblocks*sizeof(uint32_t));

    printf("using %d blocks with %d bits per block and 8 residues\n",
        numblocks, BSIZE * 32);
    printf("sieved blocks have %d extra flags\n",
        numblocks * BSIZE * 32 * 30 - N);

    MultiSegSieve2(block_counts, primes, pinv, 0, numblocks, np, N);

    gettimeofday(&stopt, NULL);
    t_time = my_difftime(&startt, &stopt);

    printf("%f seconds for big sieve\n", t_time);

    uint32_t nbig = startprime - 1;		// start here because we aren't sieving 3, and the sieve for
    // primes less than 32 is special and crosses off those primes.
    for (i = 0; i<numblocks; i++)
    {
        nbig += block_counts[i];
    }

    printf("\n%u big primes (< %lu) found\n", nbig + np - startprime, N);

    align_free(block_counts);
    align_free(primes);
    align_free(pinv);

    return 0;
}

// call as needed to extract a chunk of primes from the bitmap
uint32_t extract_prime_range(uint64_t *primes, uint64_t start)
{
    // the bitmap is laid out like this:
    //
    // residue 0       :  block 0    block 1    block 2...  block numblks-1
    // residue 1       :  block 0    block 1    block 2...  block numblks-1
    // residue 2       :  block 0    block 1    block 2...  block numblks-1
    // ...
    // residue Nres - 1:  block 0    block 1    block 2...  block numblks-1
    //
    // where each block consists of BSIZE 32-bit words (currently 262144 bits).
    // The primes are contiguous along columns e.g., residue 0, bit 0 then
    // residue 1, bit 0 ... residue N-1, bit 0, then residue 0, bit 1, etc.

    // we first compute which column of blocks contains the requested start.
    // we then compute all primes in all bits of all residues in that column.
    // currently this is 2097152 bits spanning 7864320 integers, of which a
    // fraction will be primes (depending on 'start').  worst case this is
    // about 500k primes per column of blocks ('start' == 0).
    // finally we order these primes into the output list starting at the 
    // the requested 'start' prime (or next prime greater than 'start' if
    // 'start' is not prime).

    // the two difficult tasks are computing primes and ordering them.  To
    // compute them, we transform each '1' bit in each 32-bit word into
    // a 64-bit output.  This can be done sequentially but we'd like to do
    // it in SIMD if possible.
    return 0;
}


