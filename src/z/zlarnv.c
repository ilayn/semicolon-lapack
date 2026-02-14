/**
 * @file zlarnv.c
 * @brief ZLARNV returns a vector of random numbers from a uniform or normal distribution.
 */

/**
 * ZLARNV returns a vector of n random complex numbers from a uniform or
 * normal distribution.
 *
 * This routine calls the auxiliary routine DLARUV to generate random
 * real numbers from a uniform (0,1) distribution, in batches of up to
 * 128 using vectorisable code. The Box-Muller method is used to
 * transform numbers from a uniform to a normal distribution.
 *
 * @param[in]     idist  Specifies the distribution of the random numbers:
 *                       = 1: real and imaginary parts each uniform (0,1)
 *                       = 2: real and imaginary parts each uniform (-1,1)
 *                       = 3: real and imaginary parts each normal (0,1)
 *                       = 4: uniformly distributed on the disc abs(z) < 1
 *                       = 5: uniformly distributed on the circle abs(z) = 1
 * @param[in,out] iseed  Integer array, dimension (4).
 *                       On entry, the seed of the random number generator;
 *                       the array elements must be between 0 and 4095, and
 *                       iseed[3] must be odd.
 *                       On exit, the seed is updated.
 * @param[in]     n      The number of random numbers to be generated.
 * @param[out]    X      Complex*16 array, dimension (n).
 *                       The generated random numbers.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"
#include "semicolon_lapack_double.h"

#define LV 128
#define TWOPI 6.2831853071795864769252867665590057683943

void zlarnv(const int idist, int* restrict iseed, const int n,
            c128* restrict X)
{
    f64 U[LV];
    int i, iv, il;

    for (iv = 0; iv < n; iv += LV / 2) {
        il = LV / 2;
        if (n - iv < il) {
            il = n - iv;
        }

        /*  Call DLARUV to generate 2*IL real numbers from a uniform (0,1)
         *  distribution (2*IL <= LV) */
        dlaruv(iseed, 2 * il, U);

        if (idist == 1) {

            /* Copy generated numbers */

            for (i = 0; i < il; i++) {
                X[iv + i] = CMPLX(U[2 * i], U[2 * i + 1]);
            }
        } else if (idist == 2) {

            /* Convert generated numbers to uniform (-1,1) distribution */

            for (i = 0; i < il; i++) {
                X[iv + i] = CMPLX(2.0 * U[2 * i] - 1.0,
                                   2.0 * U[2 * i + 1] - 1.0);
            }
        } else if (idist == 3) {

            /* Convert generated numbers to normal (0,1) distribution */

            for (i = 0; i < il; i++) {
                X[iv + i] = sqrt(-2.0 * log(U[2 * i])) *
                            cexp(CMPLX(0.0, TWOPI * U[2 * i + 1]));
            }
        } else if (idist == 4) {

            /* Convert generated numbers to complex numbers uniformly
             * distributed on the unit disk */

            for (i = 0; i < il; i++) {
                X[iv + i] = sqrt(U[2 * i]) *
                            cexp(CMPLX(0.0, TWOPI * U[2 * i + 1]));
            }
        } else if (idist == 5) {

            /* Convert generated numbers to complex numbers uniformly
             * distributed on the unit circle */

            for (i = 0; i < il; i++) {
                X[iv + i] = cexp(CMPLX(0.0, TWOPI * U[2 * i + 1]));
            }
        }
    }
}
