/**
 * @file dlaord.c
 * @brief DLAORD sorts the elements of a vector in increasing or decreasing order.
 *
 * Port of LAPACK TESTING/LIN/dlaord.f to C.
 */

#include <math.h>
#include "verify.h"

/**
 * DLAORD sorts the elements of a vector x in increasing or decreasing order.
 *
 * @param[in]     job   = 'I': Sort in increasing order
 *                        = 'D': Sort in decreasing order
 * @param[in]     n     The length of the vector X.
 * @param[in,out] X     Array of length (1+(n-1)*incx).
 *                      On entry, the vector to be sorted.
 *                      On exit, sorted in the prescribed order.
 * @param[in]     incx  The spacing between successive elements of X. incx >= 0.
 */
void dlaord(const char* job, const int n, f64* X, const int incx)
{
    int inc = incx < 0 ? -incx : incx;
    f64 temp;
    int i, ix, ixnext;

    if (job[0] == 'I' || job[0] == 'i') {
        /* Sort in increasing order using insertion sort */
        for (i = 1; i < n; i++) {
            ix = i * inc;
            while (ix > 0) {
                ixnext = ix - inc;
                if (X[ix] >= X[ixnext]) {
                    break;
                }
                temp = X[ix];
                X[ix] = X[ixnext];
                X[ixnext] = temp;
                ix = ixnext;
            }
        }
    } else if (job[0] == 'D' || job[0] == 'd') {
        /* Sort in decreasing order using insertion sort */
        for (i = 1; i < n; i++) {
            ix = i * inc;
            while (ix > 0) {
                ixnext = ix - inc;
                if (X[ix] <= X[ixnext]) {
                    break;
                }
                temp = X[ix];
                X[ix] = X[ixnext];
                X[ixnext] = temp;
                ix = ixnext;
            }
        }
    }
}
