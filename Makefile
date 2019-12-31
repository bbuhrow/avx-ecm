# 
# Copyright (c) 2019, Ben Buhrow
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies, 
# either expressed or implied, of the FreeBSD Project.
# 
# 

#--------------------------- flags -------------------------
CC = gcc-7.3.0
WARN_FLAGS = -Wall #-W -Wconversion
OPT_FLAGS = -O3
INC = -I. 
LIBS =
BINNAME = avx-ecm

#--------------------------- make options -------------------------

ifeq ($(COMPILER),mingw)
# NOTE: Using -fcall-used instead of -ffixed is much better and still works.
# -fcall-used simply prevents the named registers from being saved/restored while
# -ffixed prevents them from being used at all.  The code benefits a lot from being
# able to use all 32 zmm registers.
	CC = gcc
	CFLAGS += -fopenmp
    CFLAGS += -fcall-used-xmm16 -fcall-used-xmm17 -fcall-used-xmm18 -fcall-used-xmm19
    CFLAGS += -fcall-used-xmm20 -fcall-used-xmm21 -fcall-used-xmm22 -fcall-used-xmm23
    CFLAGS += -fcall-used-xmm24 -fcall-used-xmm25 -fcall-used-xmm26 -fcall-used-xmm27
    CFLAGS += -fcall-used-xmm28 -fcall-used-xmm29 -fcall-used-xmm30 -fcall-used-xmm31
    INC = -I. -I/y/projects/factoring/gmp/include/mingw
    LIBS = -L/y/projects/factoring/gmp/lib/mingw/x86_64
else ifeq ($(COMPILER),gcc)
    CC = gcc
	CFLAGS += -fopenmp 
    INC = -I. -I/projects/gmp-6.0.0a/install/include
    LIBS = -L/projects/gmp-6.0.0a/install/lib
else ifeq ($(COMPILER),gcc730)
    CC = gcc-7.3.0
	CFLAGS += -fopenmp 
    INC = -I. -I/projects/gmp-6.0.0a/install/include
    LIBS = -L/projects/gmp-6.0.0a/install/lib
else
    CC = icc
	CFLAGS += -qopenmp
    INC = -I. -I/projects/gmp-6.0.0a/install/include
    LIBS = -L/projects/gmp-6.0.0a/install/lib
endif

ifdef MAXBITS
	CFLAGS += -DMAXBITS=$(MAXBITS)
endif

ifdef DIGITBITS
	CFLAGS += -DDIGITBITS=$(DIGITBITS)
endif

ifeq ($(KNL),1)
    ifeq ($(COMPILER),icc)
        CFLAGS += -xMIC-AVX512 -DTARGET_KNL
    else
        CFLAGS += -march=knl -DTARGET_KNL
    endif
	BINNAME := ${BINNAME:%=%_knl}
    OBJ_EXT = .ko
else
    ifeq ($(SKYLAKEX),1)
        OPT_FLAGS += -march=skylake-avx512 -DSKYLAKEX
        OBJ_EXT = .o
    else
        OBJ_EXT = .o
        OPT_FLAGS += -mavx
    endif
endif


ifeq ($(CC),icc)
	ifeq ($(KNL),1)
		CFLAGS += -mkl 
	else
		CFLAGS += -L/usr/lib/gcc/x86_64-redhat-linux/4.4.4 -L/lib -mkl
	endif
endif

ifeq ($(PROFILE),1)
	CFLAGS += -pg
	BINNAME := ${BINNAME:%=%_prof}
endif


CFLAGS += -g $(OPT_FLAGS) $(WARN_FLAGS) $(INC)

ifeq ($(STATIC),1)
	CFLAGS += -static-intel -static
	LIBS += -L/usr/lib/x86_64-redhat-linux6E/lib64/ -lm
else
	LIBS += -lm -lgmp
endif
	
#--------------------------- file lists -------------------------
SRCS = \
	eratosthenes/presieve.c \
	eratosthenes/count.c \
	eratosthenes/offsets.c \
	eratosthenes/primes.c \
	eratosthenes/roots.c \
	eratosthenes/linesieve.c \
	eratosthenes/soe.c \
	eratosthenes/tiny.c \
	eratosthenes/worker.c \
	eratosthenes/soe_util.c \
	eratosthenes/wrapper.c \
	threadpool.c \
	main.c \
	ecm.c \
	util.c \
	vecarith.c \
	vecarith52.c \
	vec_common.c \
    queue.c


OBJS = $(SRCS:.c=$(OBJ_EXT))



#---------------------------Header file lists -------------------------
HEAD = \
	avx_ecm.h \
	eratosthenes/soe.h \
	threadpool.h \
	util.h \
	queue.h

#---------------------------Make Targets -------------------------

all: $(OBJS)
	rm -f libavxecm.a
	ar r libavxecm.a $(OBJS)
	ranlib libavxecm.a
	$(CC) $(CFLAGS) $(OBJS) -o $(BINNAME) libavxecm.a $(LIBS)


clean:
	rm -f $(OBJS)
	
#---------------------------Build Rules -------------------------

	
%$(OBJ_EXT): %.c $(HEAD)
	$(CC) $(CFLAGS) -c -o $@ $<

