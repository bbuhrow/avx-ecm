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

#include "queue.h"
#include <stdio.h>

Queue_t * newQueue(uint32_t sz)
{
	Queue_t *Q = (Queue_t *)malloc(sizeof(Queue_t));
	Q->Q = (uint32_t *)malloc(sz * sizeof(uint32_t));
	Q->sz = sz;
	Q->head = 0;
	Q->tail = 0;
	Q->len = 0;
	return Q;
}

void enqueue(Queue_t *Q, uint32_t e)
{
	Q->Q[Q->tail++] = e;
	Q->len++;

	if (Q->tail == Q->sz)
	{
		Q->tail = 0;
	}

	if (Q->len >= Q->sz)
	{
		printf("warning: Q overflowed\n");
		exit(1);
	}
	return;
}

uint32_t dequeue(Queue_t *Q)
{
	uint32_t e = -1;

	if (Q->len > 0)
	{
		e = Q->Q[Q->head];
		Q->head++;
		if (Q->head == Q->sz)
		{
			Q->head = 0;
		}
		Q->len--;
	}
	else
	{
		printf("warning: attempted to dequeue from an empty queue\n");
        exit(1);
	}

	return e;
}

uint32_t peekqueue(Queue_t *Q, int offset)
{
	uint32_t e = -1;
	if (Q->len > 0)
	{
		e = Q->Q[(Q->head + offset) % Q->sz];
	}
	return e;
}

void clearQueue(Queue_t *Q)
{
	free(Q->Q);
	Q->len = 0;
	Q->sz = 0;
	Q->head = 0;
	Q->tail = 0;
}
