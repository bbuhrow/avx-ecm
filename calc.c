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

/*
Implements an arbitrary precision calculator.
Supports function calls with optional arguments.
Simplifed version of that found in yafu (https://sourceforge.net/projects/yafu/).
*/

#include "calc.h"
#include <stdio.h>
#include <stdlib.h>
#ifndef WIN64
#include <string.h>
#endif
#include <ctype.h>
#include <math.h>

#define CALC_VERBOSE 0

char opchar[9] = { '=', '<', '>', '+', '-', '*', '/', '%', '^' }; // , '='};
char imms[3] = {'!','#','-'};
const int numopchars = 9;
char choperands[5][GSTR_MAXSIZE];
mpz_t operands[5];
int for_cnt = 0;
int forp_cnt = 0;
int forf_cnt = 0;
int if_cnt = 0;

//user variables
uvars_t uvars;
strvars_t strvars;

//a few global strings
str_t gstr1, gstr2, gstr3;

//rng
gmp_randstate_t gmp_randstate;

// ================================================================================
/*
    implements a custom stack/queue data structure
    the type parameter specifies which it is - a stack or a queue
    the only difference is that the queue pops from node 0 and
    then adjusts all the pointers down 1, while the stack
    pops from the top and no adjustment is necessary.

    the type of element in either structure is a str_t, which
    implements a variable length string
*/

int stack_init(int num, bstack_t *stack, int type)
{
    int i;
    stack->elements = (str_t **)malloc(num * sizeof(str_t*));		//array of elements
    //space for each element (a str_t)
    for (i = 0; i < num; i++)
    {
        stack->elements[i] = (str_t *)malloc(sizeof(str_t));
        //init each element (char array);
        sInit(stack->elements[i]);
    }
    stack->size = num;				//number of allocated stack elements
    stack->num = 0;					//number of currently occupied stack elements
    stack->top = 0;
    stack->type = type;

    return 0;
}

int stack_free(bstack_t *stack)
{
    int i;

    for (i = 0; i < stack->size; i++)
    {
        sFree(stack->elements[i]);	//first free any occupied stack elements
        free(stack->elements[i]);
    }
    free(stack->elements);			//then free the stack

    return 0;
}

void push(str_t *str, bstack_t *stack)
{
    //str_t *newstr;

    //add an element to the stack, growing the stack if necessary
    if (stack->num >= stack->size)
    {
        stack->size *= 2;
        stack->elements = (str_t **)realloc(stack->elements,
            stack->size * sizeof(str_t*));
        if (stack->elements == NULL)
        {
            printf("error allocating stack space\n");
            return;
        }
    }

    //create a new string
    //newstr = (str_t *)malloc(sizeof(str_t));
    //sInit(newstr);
    //sCopy(newstr,str);
    sCopy(stack->elements[stack->num], str);
    stack->num++;

    //both stacks and queues push to the same side of the array
    //the top element and the number of elements are the same
    stack->top = stack->num - 1;
    //store the pointer to it in the stack
    //stack->elements[stack->top] = newstr;

    return;
}

int pop(str_t *str, bstack_t *stack)
{
    //take an element off the stack.  return 0 if there are no elements
    //pass in a pointer to a string.  if necessary, this routine will 
    //reallocate space for the string to accomodate its size.  If this happens
    //the pointer to the string's (likely) new location is automatically
    //updated and returned.
    int i;

    //copy out the string at the top of the stack
    //then free the stack's copy.
    if (stack->num != 0)
    {
        stack->num--;
        if (stack->type == QUEUE)
        {
            //for queues, the top element is always node 0
            sCopy(str, stack->elements[0]);
            sFree(stack->elements[0]);
            free(stack->elements[0]);
            stack->top--;
            //now we need to adjust all the pointers down 1
            for (i = 1; i < stack->num; i++)
                stack->elements[i - 1] = stack->elements[i];
        }
        else
        {
            sCopy(str, stack->elements[stack->top]);
            //sFree(stack->elements[stack->top]);
            //free(stack->elements[stack->top]);
            stack->top--;
        }
        return 1;
    }
    else
        return 0;
}
// ================================================================================


// ================================================================================
// stuff for strings
// ================================================================================
void sInit(str_t *s)
{
    s->s = (char *)malloc(GSTR_MAXSIZE * sizeof(char));
    if (s->s == NULL)
    {
        printf("couldn't allocate str_t in sInit\n");
        exit(-1);
    }
    s->s[0] = '\0';
    s->nchars = 1;
    s->alloc = GSTR_MAXSIZE;
    return;
}

void sFree(str_t *s)
{
    free(s->s);
    return;
}

void sClear(str_t *s)
{
    s->s[0] = '\0';
    s->nchars = 1;
    return;
}

void toStr(char *src, str_t *dest)
{
    if ((int)strlen(src) > dest->alloc)
    {
        sGrow(dest, strlen(src) + 10);
        dest->alloc = strlen(src) + 10;
    }
    memcpy(dest->s, src, strlen(src) * sizeof(char));
    dest->s[strlen(src)] = '\0';
    dest->nchars = strlen(src) + 1;

    return;
}

void sGrow(str_t *str, int size)
{
    //printf("growing str_t size...\n");
    str->s = (char *)realloc(str->s, size * sizeof(char));
    if (str->s == NULL)
    {
        printf("unable to reallocate string in sGrow\n");
        exit(-1);
    }
    str->alloc = size;

    return;
}

void sAppend(const char *src, str_t *dest)
{
    if (((int)strlen(src) + dest->nchars) >= dest->alloc)
    {
        sGrow(dest, strlen(src) + dest->nchars + 10);
        dest->alloc = strlen(src) + dest->nchars + 10;
    }

    memcpy(dest->s + dest->nchars - 1, src, strlen(src) * sizeof(char));
    dest->nchars += strlen(src);	//already has a null char accounted for
    dest->s[dest->nchars - 1] = '\0';

    return;
}

void sCopy(str_t *dest, str_t *src)
{
    if (dest->alloc < src->nchars + 2)
    {
        dest->s = (char *)realloc(dest->s, (src->nchars + 2) * sizeof(char));
        dest->alloc = src->nchars + 2;
    }
    memcpy(dest->s, src->s, src->nchars * sizeof(char));
    dest->nchars = src->nchars;
    return;
}
// ================================================================================



int calc_init()
{
	int i;
	// user variables space
	uvars.vars = (uvar_t *)malloc(10 * sizeof(uvar_t));
	uvars.alloc = 10;
	for (i=0;i<uvars.alloc;i++)
		mpz_init(uvars.vars[i].data);
	strcpy(uvars.vars[0].name,"ans");
	uvars.num = 1;

    // string variable space
    strvars.vars = (strvar_t *)malloc(10 * sizeof(strvar_t));
    strvars.alloc = 10;
    for (i = 0; i<strvars.alloc; i++)
        strvars.vars[i].data = (char *)malloc(GSTR_MAXSIZE * sizeof(char));
    strvars.num = 0;

    // mpz operands to functions
    for (i = 0; i<5; i++)
        mpz_init(operands[i]);

    sInit(&gstr1);
    sInit(&gstr2);
    sInit(&gstr3);

    gmp_randinit_default(gmp_randstate);

	return 1;
}

void calc_finalize()
{
    int i;
	free_uvars();
    free_strvars();
    for (i = 0; i < 5; i++)
        mpz_clear(operands[i]);
    sFree(&gstr1);
    sFree(&gstr2);
    sFree(&gstr3);

    gmp_randclear(gmp_randstate);
    return;
}

int get_el_type2(char s)
{
	//there are several types of characters in an expression.  
	//decide which type this is
	if (isdigit(s) || (s <= 90 && s >= 65))
		return NUM;
	else if (s == '(')
		return LP;
	else if (s == ')')
		return RP;
	else if (s == '-')
		return AMBIG;
	else if (getIMM(s) >= 0)
		return IMM;
	else if (getOP(s) >= 0)
		return OP;
	else if (isEOE(s))
		return EOE;
	else if (s == ',')
		return COMMA;
	else if ((s <= 122 && s >= 95) || s == 39)
		return CH;
	else if (isspace(s))
		return SPACE;
	else
		return -1;
}

int isEOE(char s)
{
	if (s == 0)
		return 1;
	else
		return 0;
}

int getIMM(char s)
{
	int i;
	for (i=0;i<3;i++)
	{
		if (s == imms[i])
			return i;
	}

	return -1;
}

int getOP(char s)
{
	//return >=0 if this char is a opchar
	int i;

	for (i=0;i<numopchars;i++)
	{
		if (opchar[i] == s)
			return i;
	}

	return -1;
}

int getAssoc(char *s)
{
	if (strcmp(s,"^") == 0)
		return RIGHT;
	else
		return LEFT;
}

int getPrecedence(char *s)
{
    if (strcmp(s, "=") == 0)  return -1;
    if (strcmp(s, "<<") == 0) return 0;
    if (strcmp(s, ">>") == 0) return 0;
    if (strcmp(s, "+") == 0)  return 1;
    if (strcmp(s, "-") == 0)  return 1;
    if (strcmp(s, "*") == 0)  return 2;
    if (strcmp(s, "/") == 0)  return 2;
    if (strcmp(s, "%") == 0)  return 2;
    if (strcmp(s, "^") == 0)  return 3;
    if (strcmp(s, "\\") == 0) return 4;
    return 0;
}

int op_precedence(char *s1, char *s2, int assoc)
{
	//if associativity is RIGHT, then use strictly greater than
	//else use greater than or equal to
	int p1=0,p2=0;

    p1 = getPrecedence(s1);
    p2 = getPrecedence(s2);

	if (assoc == LEFT)
		return p1 >= p2;
	else 
		return p1 > p2;
}

int is_new_token(int el_type, int el_type2)
{

	if (el_type == EOE || el_type == LP || el_type == RP)
		return 1;

	if (el_type != el_type2)
	{
		//types are different
		if (el_type == CH && el_type2 == NUM)
		{
			//but this could be a function or variable name
			//so not different
			return 0;
		}
		else if (el_type == NUM && el_type2 == CH)
		{
			//but this could be a function or variable name
			//so not different
			return 0;
		}
		else
			return 1;
	}
	return 0;
}

char** tokenize(char *in, int *token_types, int *num_tokens)
{
	// take a string as input
	// break it into tokens
	// create an array of strings for each token
	// return the pointer to the array and the number of elements
	// in the array.  this will all have to be freed later
	// by the caller

	//  a token in this context is one of the following things:
	//    a number, possibly including a base prefix (0x, 0d, 0b, etc)
	//    a variable name
	//    a function name
	//    an operator string (includes parens, commas)

	// read the string one character at a time
	// for each character read, decide if we've found the start of a new token

	int inpos, i, el_type, el_type2, token_alloc, tmpsize = GSTR_MAXSIZE;
	int len = strlen(in);
	char ch;
	char *tmp;
	char **tokens;

	token_alloc = 100;		//100 tokens
	tokens = (char **)malloc(token_alloc * sizeof(char *));
	*num_tokens = 0;

	tmp = (char *)malloc(GSTR_MAXSIZE * sizeof(char));

	// get the first character and check the type
	inpos = 0;
	i=1;
	ch = in[inpos];
	tmp[i-1] = ch;
	el_type = get_el_type2(ch);

	// when an expression gets cast into postfix, it aquires a leading
	// space which we can skip here
	if (el_type == SPACE)
	{
		inpos = 1;
		i=1;
		ch = in[inpos];
		tmp[i-1] = ch;
		el_type = get_el_type2(ch);
	}

	// ambiguous types:
	// a "-" can be either a num (if a negative sign) or an operator
	// a number can be a number or a string (num or func/var name)
	// a letter can be a string or a number (hex or func/var name)
	// we can tell them apart from the surrounding context
	// 
	// negative signs never have a num type before (or anything that
	// can be evaluated as a num , i.e. ")"
	// 
	// if we are reading CH's don't stop interpreting them as CH's until
	// we find a non-CH or non-NUM (use a flag)
	// 
	// watch for magic combinations '0x' '0d', etc.  set a flag to 
	// interpret what follows as the appropriate kind of num, and
	// discard the '0x', etc.
	// once this is fixed, change the final stack evaluation to print hex
	// strings to save some conversion time.
	if (el_type == AMBIG)
	{
		if (get_el_type2(in[inpos+1]) == NUM)
			el_type = NUM;
		else
			el_type = OP;
	}
	while (inpos < len)
	{
		// get another character and check the type
		inpos++;
		// if el_type == EOE, then no reason to keep reading.  This bug didn't seem to cause
		// any crashes, but couldn't have been healthy...
		if (el_type == EOE)
			break;
		ch = in[inpos];
		el_type2 = get_el_type2(ch);
		if (el_type2 == AMBIG)
		{
			switch (get_el_type2(in[inpos-1]))
			{
			case OP:
				el_type2 = NUM;
				break;
			case LP:
				el_type2 = NUM;
				break;
			case RP:
				el_type2 = OP;
				break;
			case CH:
				el_type2 = OP;
				break;
			case IMM:
				el_type2 = OP;
				break;
			case NUM:
				el_type2 = OP;
				break;
			case COMMA:
				el_type2 = NUM;
				break;
			case SPACE:
				// when processing postfix strings, we need this
				el_type2 = OP;
				break;
			default:
				printf("misplaced - sign\n");
				for (i=0;i< *num_tokens; i++)
					free(tokens[i]);
				free(tokens);
				free(tmp);
				return NULL;
			}
		}

		if (is_new_token(el_type,el_type2) || el_type == EOE)
		{
			if (el_type == EOE)
				break;

			if (el_type == -1)
			{
				// unrecognized character.  clear all tokens and return;
				printf("unrecognized character in input: %d\n", el_type);
				for (i=0;i< *num_tokens; i++)
					free(tokens[i]);
				free(tokens);
				free(tmp);
				return NULL;
			}

			if (el_type != SPACE)
			{
				// create a new token
				tmp[i] = '\0';
				tokens[*num_tokens] = (char *)malloc((strlen(tmp) + 2) * sizeof(char));
				strcpy(tokens[*num_tokens],tmp);
				token_types[*num_tokens] = el_type;
				*num_tokens = *num_tokens + 1;

				if (*num_tokens >= token_alloc)
				{
					tokens = (char **)realloc(tokens, token_alloc * 2 * sizeof(char *));
					token_types = (int *)realloc(token_types, token_alloc * 2 * sizeof(int));
					token_alloc *= 2;
				}
			}
		
			// then cycle the types
			el_type = el_type2;
			i=1;
			strcpy(tmp,&ch);
		}
		else
		{
			if (i == (tmpsize - 1))
			{
				//printf("growing tmpsize in tokenize...\n");
				tmpsize += GSTR_MAXSIZE;
				tmp = (char *)realloc(tmp,tmpsize * sizeof(char));
			}
			tmp[i] = ch;
			i++;
		}
	}

	free(tmp);
	return tokens;
}

int isNumber(char *str)
{
	int i = 0;
	int base = 10;
	int first_char_is_zero = 0;

	for (i = 0; i < strlen(str); i++)
	{
		if ((i == 0) && (str[i] == '0'))
		{
			first_char_is_zero = 1;
			continue;
		}
		if ((i == 1) && first_char_is_zero)
		{
			if (str[i] == 'b') base = 2;
			else if (str[i] == 'o') base = 8;
			else if (str[i] == 'd' || (str[i] >= '0' && str[i] <= '9')) base = 10;
			else if (str[i] == 'x') base = 16;
			else return 0;
			continue;
		}
		if ((base == 2) && !(str[i] >= '0' && str[i] <= '1')) return 0;
		if ((base == 8) && !(str[i] >= '0' && str[i] <= '7')) return 0;
		if ((base == 10) && !(str[i] >= '0' && str[i] <= '9')) return 0;
		if ((base == 16) && !(isdigit(str[i]) || (str[i] >= 'a' && str[i] <= 'f') || (str[i] >= 'A' && str[i] <= 'F'))) return 0;
	}
	return 1;
}

int isOperator(char *str)
{
	if (str[0] == '+' || str[0] == '-' || str[0] == '*' || str[0] == '/' || str[0] == '%' || str[0] == '^') return 1;
    if (str[0] == '!' || str[0] == '#' || str[0] == '=') return 1;
	return 0;
}

/* check to see if a string contains an operator */
int hasOperator(char *str)
{
	int i = 0;

	for (i = 0; i < strlen(str); i++)
		if (isOperator(&str[i]))
			return 1;

	return 0;
}

/* check to see if str contains + or - (Addition or Subtraction)*/
int hasOperatorAS(char *str)
{
	int i = 0;

	for (i = 0; i < strlen(str); i++)
		if (str[i] == '+' || str[i] == '-')
			return 1;

	return 0;
}

int calc(str_t *in)
{

	/*
	Dijkstra's shunting algorithm 

	While there are tokens to be read: 
		* Read a token. 
		* If the token is a number, then add it to the output queue. 
		* If the token is a function token, then push it onto the stack. 
		* If the token is a function argument separator (e.g., a comma): 
			* Until the topmost element of the stack is a left parenthesis, pop the element onto the 
				output queue. If no left parentheses are encountered, either the separator was misplaced 
				or parentheses were mismatched. 
		* If the token is an operator, o1, then: 
			* while there is an operator, o2, at the top of the stack, and either 
				o1 is associative or left-associative and its precedence is less than 
				(lower precedence) or equal to that of o2, or o1 is right-associative and its precedence 
				is less than (lower precedence) that of o2,
				* pop o2 off the stack, onto the output queue; 
			* push o1 onto the operator stack. 
		* If the token is a left parenthesis, then push it onto the stack. 
		* If the token is a right parenthesis: 
			* Until the token at the top of the stack is a left parenthesis, pop operators off the stack 
				onto the output queue. 
			* Pop the left parenthesis from the stack, but not onto the output queue. 
			* If the token at the top of the stack is a function token, pop it and onto the output queue. 
			* If the stack runs out without finding a left parenthesis, then there are mismatched parentheses. 
	* When there are no more tokens to read: 
		* While there are still operator tokens in the stack: 
			* If the operator token on the top of the stack is a parenthesis, then there are mismatched 
				parenthesis. 
			* Pop the operator onto the output queue. 
	Exit. 

    In addition, we have immediate operators to deal with.  for post immediates (i.e. 4!, 9#) add
	to the output queue, just like a number

	a couple test cases
	trial(4+5^78*(81345-4),21345)

	*/

	int i,retval,na,func,j,k;
	bstack_t stk;		//general purpose stack
	str_t *tmp;			//temporary str_t
	str_t *post;		//post fix expression
	char **tokens;		//pointer to an array of strings holding tokens
	char *tok;
	char delim[2];
	int *token_types;	//type of each token
	int num_tokens;		//number of tokens in the array.
	int varstate;
	mpz_t tmpz;
    char *tok_context;
	    
	retval = 0;

	//initialize and find tokens
	token_types = (int *)malloc(100 * sizeof(int));
	tokens = tokenize(in->s, token_types, &num_tokens);
	if (tokens == NULL)
	{		
		free(token_types);
		return 1;
	}

	stack_init(20,&stk,STACK);
	tmp = (str_t *)malloc(sizeof(str_t));
	sInit(tmp);	
	mpz_init(tmpz);
	post = (str_t *)malloc(sizeof(str_t));
	sInit(post);

	//run the shunting algorithm
	i=0;
	post->s[0] = '\0';
	while (i<num_tokens)
	{
		switch (token_types[i])
		{
		case 3:
			//NUM
			//to num output queue
			sAppend(" ",post);
			sAppend(tokens[i],post);
			break;
		case 6:
			//LP
			toStr(tokens[i],tmp);
			push(tmp,&stk);
			break;
		case 7:
			//string (function or variable name)
			varstate = get_uvar(tokens[i],tmpz);
			if (varstate == 0)
			{
				//found a variable with that name, copy it's value
				//to num output queue
				sAppend(" ",post);
				sAppend(tokens[i],post);
			}
			else if (varstate == 2)
			{
				//do nothing, special case
			}
			else if (getFunc(tokens[i],&na) >= 0) 
			{
				//valid function, push it onto stack
				toStr(tokens[i],tmp);
				push(tmp,&stk);
			}
			else
			{
				// a non-numeric string that is not a variable or a function.
                // the only thing that makes sense is for it to be a string
                // representing a new assignment (new variable).  Assume that's the
                // case.  if not, errors will be raised further down the line.
				//printf("unrecognized token: %s\n",tokens[i]);
				//retval=1;
				//goto free;
                sAppend(" ", post);
                sAppend(tokens[i], post);
			}
			break;
		case 9:
			//comma (function argument separator)
			while (1)
			{
				if (pop(tmp,&stk) == 0)
				{
					//stack empty and we are still looking for a LP
					printf("bad function separator position or mismatched parens\n");
					retval = 1;
					goto free;
				}

				if (strcmp(tmp->s,"(") == 0)
				{
					//found a left paren.  put it back and continue
					push(tmp,&stk);
					break;
				}
				else
				{
					//copy to output operator queue
					sAppend(" ",post);
					sAppend(tmp->s,post);
				}
			}
			break;
		case 5:
			//right paren
			while (1)
			{
				if (pop(tmp,&stk) == 0)
				{
					//stack empty and we are still looking for a LP
					printf("mismatched parens\n");
					retval = 1;
					goto free;
				}

				if (strcmp(tmp->s,"(") == 0)
				{
					//found a left paren.  ignore it.
					if (pop(tmp,&stk) != 0)
					{
						//is the top of stack a function?
						if ((getFunc(tmp->s,&na) >= 0) && (strlen(tmp->s) > 1))
						{
							//the extra check for strlen > 1 fixes
							//the case where the string is an operator, not
							//a function.  for multichar operators this won't work
							//instead should probably separate out the operators
							//from the functions in getFunc


							//yes, put it on the output queue as well
							sAppend(" ",post);
							sAppend(tmp->s,post);
						}
						else
						{
							//no, put it back
							push(tmp,&stk);
						}
					}
					break;
				}
				else
				{
					//copy to output queue
					sAppend(" ",post);
					sAppend(tmp->s,post);
				}
			}
			break;
		case 4:
			//operator
			while (pop(tmp,&stk))
			{
				if (strlen(tmp->s) == 1 && getOP(tmp->s[0]) > 0)
				{
					//its an operator
					//check the precedence
					if (op_precedence(tmp->s,tokens[i],getAssoc(tmp->s)))
					{
						//push to output op queue
						sAppend(" ",post);
						sAppend(tmp->s,post);
					}
					else
					{
						//put the tmp one back and bail
						push(tmp,&stk);
						break;
					}
				}
				else
				{
					//its not an operator, put it back and bail
					push(tmp,&stk);
					break;
				}
			}
			//push the current op onto the stack.
			toStr(tokens[i],tmp);
			push(tmp,&stk);

			break;
		case 2:
			//post unary operator
			//I think we can jush push these into the output operator queue
			toStr(tokens[i],tmp);
			sAppend(" ",post);
			sAppend(tmp->s,post);
			break;
		}
		i++;
	}

	//now pop all operations left on the stack to the output queue
	while (pop(tmp,&stk))
	{
		if (strcmp(tmp->s,"(") == 0 || strcmp(tmp->s,")") == 0)
		{
			printf("mismatched parens\n");
			retval = 1;
			goto free;
		}
		sAppend(" ",post);
		sAppend(tmp->s,post);
	}

	//free the input tokens
	for (i=0;i<num_tokens;i++)
		free(tokens[i]);
	free(tokens);

	// process the output postfix expression:
	// this can be done with a simple stack
	// all tokens are separated by spaces
	// all tokens consist of numbers or functions

	// now evaluate the RPN expression
    if (CALC_VERBOSE)
	    printf("processing postfix expression: %s\n",post->s);
	delim[0] = ' ';
	delim[1] = '\0';

    // need to use strtok_s: some functions need to
    // do their own tokenizing so we need to remember
    // this one.
    tok_context = NULL;
	tok = strtok_s(post->s,delim,&tok_context);
	if (tok == NULL)
	{
		// printf("nothing to process\n");
        sClear(in);
		goto free;
	}

	do
	{
        if (CALC_VERBOSE)
        {
            printf("stack contents: ");
            for (i = 0; i < stk.num; i++)
                printf("%s ", stk.elements[i]->s);
            printf("\n");
            printf("current token: %s\n\n", &tok[0]);
        }

		switch (get_el_type2(tok[0]))
		{
		case NUM:
			//printf("pushing %s onto stack\n",tok);
			toStr(tok,tmp);
			push(tmp,&stk);
			break;
		case AMBIG:
			// could be a number or a function
			// if the next character is a number, this is too
			// else it's an operator
			//printf("peeking at tok + 1: %d\n",(int)tok[1]);
			if (get_el_type2(tok[1]) == NUM)
			{
				//printf("pushing %s onto stack\n",tok);
				toStr(tok,tmp);
				push(tmp,&stk);
				break;
			}

			// if not a num, proceed into the next switch (function handle)
		default:
			func = getFunc(tok,&na);
            if (CALC_VERBOSE)
			    printf("processing function %d\n",func);

			if (func >= 0)
			{
				//pop those args and put them in a global array
				for (j=0;j<na;j++)
				{
					// right now we must get all of the operands.
					// somewhere in there we should make allowances
					// for getting a reduced number (i.e. for unary "-"
					// and for variable numbers of arguments
					int r;
					k = pop(tmp,&stk);

                    if (k == 0)
                    {
                        // didn't get the expected number of arguments
                        // for this function.  This may be ok, if the
                        // function accepts varable argument lists.
                        // feval will handle it.
                        break;
                    }

					// try to make a number out of it
					//printf("looking at argument %s\n",tmp->s);
					r = mpz_set_str(tmpz, tmp->s, 0);
                    if (r < 0)
                    {
                        //printf("input is not a valid number in a discoverable base\n");
                        //printf("placing %s into character operand array\n", tmp->s);
                        strcpy(choperands[na - j - 1], tmp->s);
                    }
                    else
                    {
                        // it is a number, put it in the operand pile
                        //gmp_printf("found numerical argument %Zd\n", tmpz);
                        mpz_set(operands[na - j - 1], tmpz);
                    }
				}

				na = j;
				// call the function evaluator with the 
				// operator string and the number of args available
				na = feval(func,na);

				// put result back on stack
				for (j=0;j<na;j++)
				{
					int sz = mpz_sizeinbase(operands[j], 10) + 10;
					if (tmp->alloc < sz)
					{
						tmp->s = (char *)realloc(tmp->s, sz * sizeof(char));
						tmp->alloc = sz;
					}
					mpz_get_str(tmp->s, 10, operands[j]);
					tmp->nchars = strlen(tmp->s)+1;
					push(tmp,&stk);
				}
			}
            else if (get_uvar(tok, tmpz) == 0)
            {
                int sz;

                // the string token is not a function, check if it's a defined variable

                sz = mpz_sizeinbase(tmpz, 10) + 10;
                if (gstr1.alloc < sz)
                {
                    gstr1.s = (char *)realloc(gstr1.s, sz * sizeof(char));
                    gstr1.alloc = sz;
                }
                mpz_get_str(gstr1.s, 10, tmpz);
                gstr1.nchars = strlen(gstr1.s) + 1;
                sCopy(tmp, &gstr1);
                push(tmp, &stk);
            }
            else if (is_strvar(tok))
            {
                // is a string variable... push it onto the stack
                toStr(tok, tmp);
                push(tmp, &stk);
            }
            else
			{
                printf("unrecognized variable or function '%s'\n", tok);
                sClear(in);
                goto free;
			}			
		}

        tok = strtok_s((char *)0, delim, &tok_context);
	} while (tok != NULL);
	pop(in,&stk);

free:	
	free(token_types);
	stack_free(&stk);
	mpz_clear(tmpz);
	sFree(tmp);
	free(tmp);
	sFree(post);
	free(post);
	return retval;
}

static char func[NUM_FUNC][11] = { 
    "fib", "luc", "dummy", "dummy", "dummy",
    "gcd", "jacobi", "dummy", "rand", "lg2",
    "log", "ln", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "sqrt", "modinv", "modexp",
    "nroot", "shift", "dummy", "dummy", "dummy",
    "dummy", "randb", "dummy", "+", "-",
    "*", "/", "!", "#", "dummy",
    "<<", ">>", "%", "^", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "xor", "and", "or", "not", "dummy",
    "dummy", "dummy", "lte", "gte", "<",
    ">", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "abs", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy",
    "dummy", "dummy", "dummy", "dummy", "dummy" };

static int args[NUM_FUNC] = { 
    1, 1, 2, 1, 1,
    2, 2, 1, 1, 1,
    1, 1, 1, 2, 1,
    2, 1, 2, 1, 1,
    1, 1, 1, 2, 3,
    2, 2, 1, 3, 1,
    2, 1, 2, 2, 2,
    2, 2, 1, 1, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 1, 0,
    2, 1, 1, 4, 1,
    0, 4, 3, 1, 0,
    2, 2, 2, 1, 2,
    1, 1, 2, 2, 2,
    2, 2, 3, 1, 4,
    3, 0, 2, 1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1 };

int getFunc(char *s, int *nargs)
{
	// return the opcode associated with the function, and
	// the number of arguments it takes
	int i,j;

	for (i = 0; i < NUM_FUNC; i++)
	{
		j = strcmp(func[i],s);
		if (j == 0)
		{
			*nargs = args[i];
			return i;
		}
	}

	return -1;
}

int check_args(int funcnum, int nargs)
{
    if (nargs != args[funcnum])
    {
        printf("wrong number of arguments in %s, expected %d\n", 
            func[funcnum], args[funcnum]);
        return 1;
    }
    else
        return 0;
}

int feval(int funcnum, int nargs)
{
	// evaluate the function 'func', with 'nargs' argument(s) located
	// in the mpz_t array 'operands'.
	// place return values in operands[0]
	mpz_t mp1, mp2, mp3, tmp1, tmp2;

	str_t str;

	mpz_init(mp1);
	mpz_init(mp2);
	mpz_init(mp3);
	mpz_init(tmp1);
	mpz_init(tmp2);
	sInit(&str);

	switch (funcnum)
	{
	case 0:
		//fib - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_fib_ui(operands[0], mpz_get_ui(operands[0]));
		break;
	case 1:
		//luc - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_lucnum_ui(operands[0], mpz_get_ui(operands[0]));
		break;
		
	case 2:
		// snfs - two arguments
        
		break;
	case 3:
		// expr - one argument
		
		break;
	case 4:
		//rsa - one argument
        
		break;
	case 5:
		//gcd - two arguments
        if (check_args(funcnum, nargs)) break;			
		mpz_gcd(operands[0], operands[0], operands[1]);

		break;
	case 6:
		//jacobi - two arguments
        if (check_args(funcnum, nargs)) break;

		if (mpz_odd_p(operands[1]))
			mpz_set_si(operands[0], mpz_jacobi(operands[0], operands[1]));
		else
			printf("jacobi defined only for odd denominators!\n");

		break;
	case 7:
		//factor - one argument
        
		break;
	case 8:
		//rand - one argument
        if (check_args(funcnum, nargs)) break;

		mpz_set_ui(operands[1], 10);
		mpz_pow_ui(operands[1], operands[1], mpz_get_ui(operands[0]));
		mpz_urandomm(operands[0], gmp_randstate, operands[1]);
		break;
	case 9:
		//lg2 - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_sizeinbase(operands[0], 2));
		break;
	case 10:
		//log - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_sizeinbase(operands[0], 10));
		break;
	case 11:
		//ln - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], (uint32_t)((mpz_sizeinbase(operands[0], 2)-1) * log(2.0)));
		break;
	case 12:
		//pm1 - one argument
        
		break;
	case 13:
		//pp1 - two arguments, one optional
		
		break;
	case 14:
		//rho - one argument
        
		break;
	case 15:
		//trial - two arguments
		
		break;
	case 16:
		//mpqs - one argument
		
		break;

	case 17:
		//next prime - two arguments
		
		break;
	case 18:
		//size - one argument
        
		break;
	case 19:
		//issquare
        
		break;
	case 20:
		//isprime - one argument
        
		break;
	case 21:
		//shanks - one argument
        
		break;
	case 22:
		//sqrt - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_root(operands[0], operands[0], 2);
		break;
	case 23:
		//modinv - two arguments
        if (check_args(funcnum, nargs)) break;
        //printf("modinv_1 = %u\n", modinv_1(operands[0]->_mp_d[0], operands[1]->_mp_d[0]));
		mpz_invert(operands[0], operands[0], operands[1]);        
		break;
	case 24:
		//modexp - three arguments
        if (check_args(funcnum, nargs)) break;
		mpz_powm(operands[0], operands[0], operands[1], operands[2]);
		break;
	case 25:
		//nroot - two arguments
        if (check_args(funcnum, nargs)) break;
		mpz_root(operands[0], operands[0], mpz_get_ui(operands[1]));
		break;
	case 26:
		//shift - two arguments
        if (check_args(funcnum, nargs)) break;

		if (mpz_sgn(operands[1]) >= 0)
			mpz_mul_2exp(operands[0], operands[0], mpz_get_ui(operands[1]));
		else
			mpz_tdiv_q_2exp(operands[0], operands[0], -1*mpz_get_si(operands[1]));

		break;
	case 27:
		//siqs - one argument
        
		break;

	case 28:
		//primes
		
		break;
	case 29:
		//ispow - one argument
       
		break;

	case 30:
		//torture - two arguments
        
		break;
	case 31:
		//randb - one argument
        if (check_args(funcnum, nargs)) break;
		mpz_urandomb(operands[0], gmp_randstate, mpz_get_ui(operands[0]));
		break;
	case 32:
		//ecm - two arguments
		
		break;

	case 33:
		//add
        if (check_args(funcnum, nargs)) break;
		mpz_add(operands[0], operands[0], operands[1]);
		break;

	case 34:
		//subtract or negate
		if (nargs == 1)
		{
			mpz_neg(operands[0], operands[0]);
		}
		else if (nargs == 2)
		{
			mpz_sub(operands[0], operands[0], operands[1]);
		}
		else
		{
			printf("wrong number of arguments in sub/neg\n");
			break;
		}

		break;

	case 35:
		//mul
        if (check_args(funcnum, nargs)) break;
		mpz_mul(operands[0], operands[0], operands[1]);
		break;

	case 36:
		//div
        if (check_args(funcnum, nargs)) break;
		mpz_tdiv_q(operands[0], operands[0], operands[1]);
		break;

	case 37:
		//!
        if (check_args(funcnum, nargs)) break;
		mpz_fac_ui(operands[0], mpz_get_ui(operands[0]));
		break;

	case 38:
		//primorial
        if (check_args(funcnum, nargs)) break;
		mpz_primorial_ui(operands[0], mpz_get_ui(operands[0]));
		break;

	case 39:
		// eq
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_cmp(operands[0], operands[1]) == 0);
		break;

	case 40:
		//<<
        if (check_args(funcnum, nargs)) break;
		mpz_mul_2exp(operands[0], operands[0], mpz_get_ui(operands[1]));
		break;

	case 41:
		//>>
        if (check_args(funcnum, nargs)) break;
		mpz_tdiv_q_2exp(operands[0], operands[0], mpz_get_ui(operands[1]));
		break;

	case 42:
		//mod
        if (check_args(funcnum, nargs)) break;
		mpz_mod(operands[0], operands[0], operands[1]);
		break;

	case 43:
		//exp
        if (check_args(funcnum, nargs)) break;
		mpz_pow_ui(operands[0], operands[0], mpz_get_ui(operands[1]));
		break;

	case 44:
        // REDC (redc)
        
		break;

	case 45:
        // modmul

		break;

	case 46:
		//sieve
		break;

	case 47:
		//algebraic
		break;

	case 48:
		//lucas lehmer test
        
		break;

	case 49:
		//siqsbench
        
		break;

	case 50:
		// sigma - sum of divisors function
        
		break;
	case 51: 
        // Euler's totient function
        
		break;

	case 52:
		//smallmpqs - 1 argument
        
		break;
	case 53:
		//testrange - 4 arguments (low, high, depth, witnesses)
        
		break;

	case 54:
		break;

	case 55:
        // move to soe library?

		break;

	case 56: 

		//sieverange - 4 arguments
        
		break;

	case 57:
		//fermat - three arguments
		
		break;

	case 58:
		//nfs - one argument
       
		break;

	case 59:
		//tune, no arguments
        
		break;

	case 60:
		// xor
        if (check_args(funcnum, nargs)) break;
        mpz_xor(operands[0], operands[0], operands[1]);
		break;

	case 61:
		// and
        if (check_args(funcnum, nargs)) break;
		mpz_and(operands[0], operands[0], operands[1]);
		break;

	case 62:
		// or
        if (check_args(funcnum, nargs)) break;
		mpz_ior(operands[0], operands[0], operands[1]);
		break;

	case 63:
		// not
        if (check_args(funcnum, nargs)) break;
		mpz_com(operands[0], operands[0]);
		break;

	case 64:
		// factor a range of single precision numbers
        
		break;

	case 65:
		/* bpsw */
        
		break;

	case 66:
		/* aprcl */
        
		break;

	case 67:
		// lte
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_cmp(operands[0], operands[1]) <= 0);
		break;

	case 68:
		// gte
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_cmp(operands[0], operands[1]) >= 0);
		break;

	case 69:
		// lt
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_cmp(operands[0], operands[1]) < 0);
		break;

	case 70:
		// gt
        if (check_args(funcnum, nargs)) break;
		mpz_set_ui(operands[0], mpz_cmp(operands[0], operands[1]) > 0);

		break;
    case 71:
        // = (assignment)
        
        break;
    case 72:
        // if
        
        break;
    case 73:
        // print
        
        break;

    case 74:
        // for
        
        break;

    case 75:
        // forprime
        
        break;

    case 76:
        // exit
        break;

    case 77:
        // forfactors
        
        break;

	case 78:
		// abs
		if (check_args(funcnum, nargs)) break;
		mpz_abs(operands[0], operands[0]);

		break;

	default:
		printf("unrecognized function code\n");
		mpz_set_ui(operands[0], 0);
		break;
	}

	sFree(&str);
	mpz_clear(mp1);
	mpz_clear(mp2);
	mpz_clear(mp3);
	mpz_clear(tmp1);
	mpz_clear(tmp2);
	return 1;
}

int get_uvar(const char *name, mpz_t data)
{
	//look for 'name' in the global uvars structure
	//if found, copy out data and return 0
	//else return 1 if not found
	int i;

    /*
	//first look if it is a global constant
	if (strcmp(name,"IBASE") == 0) {
		mpz_set_ui(data, IBASE); return 0;}
	else if (strcmp(name,"OBASE") == 0) {
		mpz_set_ui(data, OBASE); return 0;}
	else if (strcmp(name,"NUM_WITNESSES") == 0) {
		mpz_set_ui(data, NUM_WITNESSES); return 0;}
	else if (strcmp(name,"LOGFLAG") == 0) {
		mpz_set_ui(data, LOGFLAG); return 0;}
	else if (strcmp(name,"VFLAG") == 0) {
		mpz_set_ui(data, VFLAG); return 0;}
	else if (strcmp(name,"PRIMES_TO_FILE") == 0) {
		mpz_set_ui(data, PRIMES_TO_FILE); return 0;}
	else if (strcmp(name,"PRIMES_TO_SCREEN") == 0) {
		mpz_set_ui(data, PRIMES_TO_SCREEN); return 0;}

	for (i=0;i<uvars.num;i++)
	{
		if (strcmp(uvars.vars[i].name,name) == 0)
		{
			mpz_set(data, uvars.vars[i].data);
			return 0;
		}
	}

	if (strcmp(name,"vars") == 0) {
		printf("dumping variable name data:\n");
		printf("IBASE              %u\n",IBASE);
		printf("OBASE              %u\n",OBASE);		
		printf("NUM_WITNESSES      %u\n",NUM_WITNESSES);
		printf("LOGFLAG            %u\n",LOGFLAG);
		printf("VFLAG              %u\n",VFLAG);
		printf("PRIMES_TO_FILE     %u\n",PRIMES_TO_FILE);
		printf("PRIMES_TO_SCREEN   %u\n",PRIMES_TO_SCREEN);

		for (i=0;i<uvars.num;i++)
			printf("%s      %s\n",uvars.vars[i].name,mpz_get_str(gstr1.s, 10, uvars.vars[i].data));

		return 2;
	}
    */

    if (strcmp(name, "strvars") == 0) {
        printf("dumping string variable name data:\n");

        for (i = 0; i<strvars.num; i++)
            printf("%s      %s\n", strvars.vars[i].name, strvars.vars[i].data);

        return 2;
    }

	return 1;
}

void free_uvars()
{
	int i;
	for (i=0;i<uvars.alloc;i++)
		mpz_clear(uvars.vars[i].data);
	free(uvars.vars);
}

int new_strvar(const char *name, char *data)
{
    int i;
    //create a new user variable with name 'name', and return
    //its location in the global uvars structure
    if (strvars.num == strvars.alloc)
    {
        //need more room for variables
        strvars.vars = (strvar_t *)realloc(strvars.vars, strvars.num * 2 * sizeof(strvar_t));
        strvars.alloc *= 2;
        for (i = strvars.num; i<strvars.alloc; i++)
            strvars.vars[i].data = (char *)malloc(GSTR_MAXSIZE * sizeof(char));
    }

    strcpy(strvars.vars[strvars.num].name, name);
    strcpy(strvars.vars[strvars.num].data, data);
    strvars.num++;
    return strvars.num - 1;
}

int set_strvar(const char *name, char *data)
{
    // look for 'name' in the global uvars structure
    // if found, copy in data and return 0
    // else return 1
    int i;

    for (i = 0; i<strvars.num; i++)
    {
        if (strcmp(strvars.vars[i].name, name) == 0)
        {
            strcpy(strvars.vars[i].data, data);
            return 0;
        }
    }
    return 1;
}

int is_strvar(const char *name)
{
    // look for 'name' in the global uvars structure
    // if found, return 1
    // else return 0 if not found
    int i;

    for (i = 0; i<strvars.num; i++)
    {
        if (strcmp(strvars.vars[i].name, name) == 0)
        {
            return 1;
        }
    }
    return 0;
}

char * get_strvarname(const char *data)
{
    // look for 'data' in the global uvars structure
    // if found, return the name of the variable
    int i;
    char *name = NULL;

    for (i = 0; i<strvars.num; i++)
    {
        if (strcmp(strvars.vars[i].data, data) == 0)
        {
            name = strvars.vars[i].name;
            break;
        }
    }
    return name;
}

int get_strvar(const char *name, char *data)
{
    // look for 'name' in the global uvars structure
    // if found, copy out data and return 0
    // else return 1 if not found
    int i;

    for (i = 0; i<strvars.num; i++)
    {
        if (strcmp(strvars.vars[i].name, name) == 0)
        {
            strcpy(data, strvars.vars[i].data);
            return 0;
        }
    }

    if (strcmp(name, "strvars") == 0) {
        printf("dumping string variable name data:\n");

        for (i = 0; i<strvars.num; i++)
            printf("%s      %s\n", strvars.vars[i].name, strvars.vars[i].data);

        return 2;
    }

    return 1;
}

void free_strvars()
{
    int i;
    for (i = 0; i<strvars.alloc; i++)
        free(strvars.vars[i].data);
    free(strvars.vars);
}


