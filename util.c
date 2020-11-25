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

#include "util.h"

// memory and other utilities
void * xmalloc_align(size_t len)
{
#if defined (_MSC_VER) || defined(__MINGW32__)
    void *ptr = _aligned_malloc(len, 64);

#elif defined (__APPLE__)
    void *ptr = malloc(len);

#elif defined (__GNUC__)
    //void *ptr = memalign(64, len);
	void *ptr;
	posix_memalign(&ptr, 64, len);

#else
    void *ptr = malloc(len);

#endif

    return ptr;
}

void * xmalloc(size_t len) {
    void *ptr = malloc(len);
    if (ptr == NULL) {
        printf("failed to allocate %u bytes\n", (uint32_t)len);
        exit(-1);
    }
    return ptr;
}

void * xcalloc(size_t num, size_t len) {
    void *ptr = calloc(num, len);
    if (ptr == NULL) {
        printf("failed to calloc %u bytes\n", (uint32_t)(num * len));
        exit(-1);
    }
    return ptr;
}

void * xrealloc(void *iptr, size_t len) {
    void *ptr = realloc(iptr, len);
    if (ptr == NULL) {
        printf("failed to reallocate %u bytes\n", (uint32_t)len);
        exit(-1);
    }
    return ptr;
}

uint64_t spRand64(uint64_t *state)
{
    // advance the state of the LCG and return the appropriate result.
    // assume lower = 0 and upper = maxint
    *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
    return *state;
}

uint64_t spGCD(uint64_t x, uint64_t y)
{
    uint64_t a, b, c;
    a = x; b = y;
    while (b != 0)
    {
        c = a % b;
        a = b;
        b = c;
    }
    return a;
}

