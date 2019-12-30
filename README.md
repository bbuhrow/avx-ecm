# avx-ecm
Computes parallel curves of the ECM (Elliptic Curve Method) factoring algorithm using AVX-512 vector arithmetic.

Both stage 1 and stage 2 are computed in parallel, but stage 2 is just the standard continuation with pairing (by default, B2=100xB1).

The program is also multi-threaded using the pthreads library.

It will output a savefile after stage 1 that will work with GMP-ECM stage 2, if desired.
e.g.:
ecm -resume save_b1.txt 1000000 < input.txt

Compile on linux using either icc or gcc-7.3.0 or above (I've only tested icc and gcc-7.3.0).

e.g.:
make MAXBITS=416 COMPILER=gcc730 SKYLAKEX=1

the MAXBITS parameter must be a multiple of 208. When this is the case, 8 curves are performed in parallel per thread.

Alternatively, if DIGITBITS=32 is specified during make, then MAXBITS must be a mutiple of 128 and 16 curves are performed in parallel per thread.

The 52-bit version is generally faster (and the default) but for some sizes a 32-bit version can have higher throughput (because of the 208-bit jumps between sizes).

Command line:
avx-ecm input curves B1 threads B2

Happy factoring!
