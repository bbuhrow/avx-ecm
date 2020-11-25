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

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef _MSC_VER
#include <sys/time.h>	//for gettimeofday using gcc
#include <unistd.h>
#endif

#if defined (__INTEL_COMPILER)
#define ALIGNED_MEM __declspec(align(64))
#else
#define ALIGNED_MEM __attribute__((aligned(64)))
#endif

#define MIN(a,b) ((a) < (b)? (a) : (b))
#define MAX(a,b) ((a) > (b)? (a) : (b))
#define SIGN(a) ((a) < 0 ? -1 : 1)
#define LOWER(x) ((x) & HALFMASK)
#define UPPER(x) ((x) >> HALFBITS)

#if defined (_MSC_VER) || defined(__MINGW32__)
#define align_free _aligned_free
#elif defined (__GNUC__)
#define align_free free
#endif

uint64_t spGCD(uint64_t x, uint64_t y);

// random
uint64_t hash64(uint64_t in);
uint64_t lcg_rand(uint64_t *lcg_state);
uint64_t spRand64(uint64_t *state);

void * xmalloc_align(size_t len);
void * xmalloc(size_t len);
void * xcalloc(size_t num, size_t len);
void * xrealloc(void *iptr, size_t len);

double my_difftime(struct timeval * start, struct timeval * end);

