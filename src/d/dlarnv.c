/**
 * @file dlarnv.c
 * @brief DLARNV returns a vector of random numbers from a uniform or normal distribution.
 */

/**
 * DLARNV returns a vector of n random real numbers from a uniform or
 * normal distribution.
 *
 * This routine calls the auxiliary routine DLARUV to generate random
 * real numbers from a uniform (0,1) distribution, in batches of up to
 * 128 using vectorisable code. The Box-Muller method is used to
 * transform numbers from a uniform to a normal distribution.
 *
 * @param[in]     idist  Specifies the distribution of the random numbers:
 *                       = 1: uniform (0,1)
 *                       = 2: uniform (-1,1)
 *                       = 3: normal (0,1)
 * @param[in,out] iseed  Integer array, dimension (4).
 *                       On entry, the seed of the random number generator;
 *                       the array elements must be between 0 and 4095, and
 *                       iseed[3] must be odd.
 *                       On exit, the seed is updated.
 * @param[in]     n      The number of random numbers to be generated.
 * @param[out]    X      Double precision array, dimension (n).
 *                       The generated random numbers.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/* LV = max number of uniform random values generated per dlaruv call */
#define LV 128

#define TWO_PI 6.2831853071795864769252867665590057683943

void dlarnv(const int idist, int* restrict iseed, const int n,
            f64* restrict X)
{
    f64 U[LV];
    int i, iv, il, il2;

    for (iv = 0; iv < n; iv += LV / 2) {
        il = LV / 2;
        if (n - iv < il) {
            il = n - iv;
        }

        if (idist == 3) {
            il2 = 2 * il;
        } else {
            il2 = il;
        }

        /* Call dlaruv to generate il2 numbers from a uniform (0,1)
         * distribution (il2 <= LV) */
        dlaruv(iseed, il2, U);

        if (idist == 1) {
            /* Copy generated numbers */
            for (i = 0; i < il; i++) {
                X[iv + i] = U[i];
            }
        } else if (idist == 2) {
            /* Convert generated numbers to uniform (-1,1) distribution */
            for (i = 0; i < il; i++) {
                X[iv + i] = 2.0 * U[i] - 1.0;
            }
        } else if (idist == 3) {
            /* Convert generated numbers to normal (0,1) distribution
             * using the Box-Muller transform */
            for (i = 0; i < il; i++) {
                X[iv + i] = sqrt(-2.0 * log(U[2 * i])) *
                            cos(TWO_PI * U[2 * i + 1]);
            }
        }
    }
}
