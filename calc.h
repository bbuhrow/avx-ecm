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

#include "gmp.h"
#include "stdint.h"

typedef struct
{
    char *s;		//pointer to beginning of s
    int nchars;		//number of valid characters in s (including \0)
    int alloc;		//bytes allocated to s
} str_t;

typedef struct
{
    char name[40];
    mpz_t data;
} uvar_t;

typedef struct
{
    char name[40];
    char *data;
} strvar_t;

typedef struct
{
    uvar_t *vars;
    int num;
    int alloc;
} uvars_t;

typedef struct
{
    strvar_t *vars;
    int num;
    int alloc;
} strvars_t;

typedef struct
{
    str_t **elements;	//an array of pointers to elements
    int num;			//number of elements
    int size;			//allocated number of elements in stack
    int top;			//the top element
    int type;			//is this a stack (0) or a queue (1)?
} bstack_t;

#define GSTR_MAXSIZE 1024

// stack types
#define QUEUE 1
#define STACK 0

int stack_init(int num, bstack_t *stack, int type);
int stack_free(bstack_t *stack);
void push(str_t *str, bstack_t *stack);
int pop(str_t *str, bstack_t *stack);

// string stuff
void sInit(str_t *s);
void sFree(str_t *s);
void sClear(str_t *s);
void sCopy(str_t *dest, str_t *src);
void sGrow(str_t *s, int size);
void toStr(char *src, str_t *dest);
void sAppend(const char *src, str_t *dest);

//symbols in calc
#define EOE 1
#define IMM 2
#define NUM 3
#define OP 4
#define RP 5
#define LP 6
#define CH 7
#define AMBIG 8
#define COMMA 9
#define SPACE 10

//operator associativity
#define RIGHT 1
#define LEFT 0

#define NUM_FUNC 100

//arbitrary precision calculator
void testcalc(void);
void handle_singleop(char *arg1, int op);
int single_op(char s);
int dual_op(char s);
str_t * preprocess(str_t *str, int *num);
int get_el_type(char s);
int processIMM(int opcode, str_t *str);
int calc(str_t *str);
int isInt(char s);
int getIMM(char s);
int op_precedence(char *s1, char *s2, int assoc);
int getAssoc(char *s);
int processOP(char *s, str_t *n1, str_t *n2);
int getOP(char s);
int isEOE(char s);
int getFunc(char *s, int *nargs);
int feval(int func, int nargs);
int get_uvar(const char *name, mpz_t data);
void free_uvars();
int new_strvar(const char *name, char *data);
int set_strvar(const char *name, char *data);
int get_strvar(const char *name, char *data);
char * get_strvarname(const char *data);
int is_strvar(const char *name);
void free_strvars();
int invalid_dest(char *dest);
int invalid_num(char *num);
int calc2(str_t *in);
char** tokenize(char *in, int *token_types, int *num_tokens);
int get_el_type2(char s);
int is_new_token(int el_type, int el_type2);
void calc_finalize();
int calc_init();

#ifndef _MSC_VER
#define strtok_s strtok_r
#endif

