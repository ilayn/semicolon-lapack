/**
 * @file dlasq1.c
 * @brief DLASQ1 computes the singular values of a real square bidiagonal matrix.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>
#include <math.h>

/**
 * DLASQ1 computes the singular values of a real N-by-N bidiagonal
 * matrix with diagonal D and off-diagonal E. The singular values
 * are computed to high relative accuracy, in the absence of
 * denormalization, underflow and overflow.
 *
 * The algorithm was first presented in
 * "Accurate singular values and differential qd algorithms" by K. V.
 * Fernando and B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230,
 * 1994.
 *
 * @param[in]     n     The number of rows and columns in the matrix. n >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, D contains the diagonal elements of the
 *                      bidiagonal matrix whose SVD is desired. On normal exit,
 *                      D contains the singular values in decreasing order.
 * @param[in,out] E     Double precision array, dimension (n).
 *                      On entry, elements E[0:n-2] contain the off-diagonal elements
 *                      of the bidiagonal matrix whose SVD is desired.
 *                      On exit, E is overwritten.
 * @param[out]    work  Double precision array, dimension (4*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: the algorithm failed
 *                         - = 1, a split was marked by a positive value in E
 *                         - = 2, current block of Z not diagonalized after 100*N
 *                           iterations (in inner while loop). On exit D and E
 *                           represent a matrix with the same singular values
 *                           which the calling subroutine could use to finish the
 *                           computation, or even feed back into DLASQ1
 *                         - = 3, termination criterion of outer while loop not met
 *                           (program created more than N unreduced blocks)
 */
void dlasq1(const int n, f64* restrict D, f64* restrict E,
            f64* restrict work, int* info)
{
    int i, iinfo;
    f64 eps, scale, safmin, sigmn, sigmx;

    *info = 0;
    if (n < 0) {
        *info = -1;
        xerbla("DLASQ1", 1);
        return;
    } else if (n == 0) {
        return;
    } else if (n == 1) {
        D[0] = fabs(D[0]);
        return;
    } else if (n == 2) {
        dlas2(D[0], E[0], D[1], &sigmn, &sigmx);
        D[0] = sigmx;
        D[1] = sigmn;
        return;
    }

    /* Estimate the largest singular value */
    sigmx = 0.0;
    for (i = 0; i < n - 1; i++) {
        D[i] = fabs(D[i]);
        if (fabs(E[i]) > sigmx) {
            sigmx = fabs(E[i]);
        }
    }
    D[n - 1] = fabs(D[n - 1]);

    /* Early return if SIGMX is zero (matrix is already diagonal) */
    if (sigmx == 0.0) {
        dlasrt("D", n, D, &iinfo);
        return;
    }

    for (i = 0; i < n; i++) {
        if (D[i] > sigmx) {
            sigmx = D[i];
        }
    }

    /*
     * Copy D and E into WORK (in the Z format) and scale (squaring the
     * input data makes scaling by a power of the radix pointless).
     */
    eps = dlamch("P");
    safmin = dlamch("S");
    scale = sqrt(eps / safmin);

    cblas_dcopy(n, D, 1, &work[0], 2);
    cblas_dcopy(n - 1, E, 1, &work[1], 2);

    dlascl("G", 0, 0, sigmx, scale, 2 * n - 1, 1, work, 2 * n - 1, &iinfo);

    /* Compute the q's and e's */
    for (i = 0; i < 2 * n - 1; i++) {
        work[i] = work[i] * work[i];
    }
    work[2 * n - 1] = 0.0;

    dlasq2(n, work, info);

    if (*info == 0) {
        for (i = 0; i < n; i++) {
            D[i] = sqrt(work[i]);
        }
        dlascl("G", 0, 0, scale, sigmx, n, 1, D, n, &iinfo);
    } else if (*info == 2) {
        /*
         * Maximum number of iterations exceeded. Move data from WORK
         * into D and E so the calling subroutine can try to finish
         */
        for (i = 0; i < n; i++) {
            D[i] = sqrt(work[2 * i]);
            E[i] = sqrt(work[2 * i + 1]);
        }
        dlascl("G", 0, 0, scale, sigmx, n, 1, D, n, &iinfo);
        dlascl("G", 0, 0, scale, sigmx, n, 1, E, n, &iinfo);
    }
}
