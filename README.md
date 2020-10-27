# avx-ecm
Computes parallel curves of the ECM (Elliptic Curve Method) factoring algorithm using AVX-512 vector arithmetic.  On CPUs where AVX512 is supported, AVX-ECM has been observed to have about 1.5 to 2.5 times the throughput (curves/sec) of GMP-ECM, for numbers up to a thousand bits or so.  Large numbers will be more efficient with GMP-ECM because AVX-ECM so far does not use sub-quadratic multiplier algorithms.

Both stage 1 and stage 2 are computed in parallel, but stage 2 is just the standard continuation with pairing (by default, B2=100xB1).

The program is also multi-threaded using the pthreads library.

It will output a savefile after stage 1 that will work with GMP-ECM stage 2, if desired.
e.g.:
ecm -resume save_b1.txt 1000000 < input.txt

Compile on linux using either icc or gcc-7.3.0 or above (I've only tested icc and gcc-7.3.0).

e.g.:
make COMPILER=gcc730 SKYLAKEX=1

COMPILER options currently are gcc730 (which invokes gcc-7.3.0), gcc (which invokes gcc), mingw (using MSYS2 and mingw64 on windows) and icc (icc).  Edit the makefile if you have some other compiler name (ymmv).

Use SKYLAKEX=1 to build for skylakex CPUs.  Alternatively use KNL=1 to build for Knight's Landing Xeon Phi systems.

The program will run constant-time curves in steps of 208 bits (meaning, e.g., that 417 bit curves take the same amount of time as 623-bit curves) and 8 curves are performed in parallel per thread.

You can optionally specify DIGITBITS=32 during make.  Doing so results in constant-time curves in steps of 128 bits and 16 curves performed in parallel per thread.

The 52-bit version is generally faster (and the default) but for some sizes a 32-bit version can have higher throughput (because of the 208-bit jumps between sizes).

Command line:
avx-ecm input curves B1 threads B2

The input number can be specified using these operators if desired: +,-,*,/,^,%,# (primorial), ! (factorial), fib(), and luc()

Example:
./avx-ecm "fib(791)/13/677/216416017" 8 1000000 1

See https://www.mersenneforum.org/showthread.php?t=25056 for more info.

Happy factoring!
