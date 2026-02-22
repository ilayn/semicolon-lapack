/**
 * @file dlaexc.c
 * @brief DLAEXC swaps adjacent diagonal blocks of a real upper
 *        quasi-triangular matrix in Schur canonical form.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLAEXC swaps adjacent diagonal blocks T11 and T22 of order 1 or 2 in
 * an upper quasi-triangular matrix T by an orthogonal similarity
 * transformation.
 *
 * T must be in Schur canonical form, that is, block upper triangular
 * with 1-by-1 and 2-by-2 diagonal blocks; each 2-by-2 diagonal block
 * has its diagonal elements equal and its off-diagonal elements of
 * opposite sign.
 *
 * @param[in]     wantq  If nonzero, accumulate the transformation in Q.
 * @param[in]     n      The order of the matrix T. n >= 0.
 * @param[in,out] T      On entry, the upper quasi-triangular matrix T, in
 *                       Schur canonical form. On exit, the updated matrix T.
 *                       Dimension (ldt, n).
 * @param[in]     ldt    The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] Q      On entry, if wantq is nonzero, the orthogonal matrix Q.
 *                       On exit, if wantq is nonzero, the updated Q.
 *                       Not referenced if wantq is zero. Dimension (ldq, n).
 * @param[in]     ldq    The leading dimension of Q. ldq >= 1, and if wantq
 *                       is nonzero, ldq >= n.
 * @param[in]     j1     The index of the first row of the first block T11.
 *                       (0-based indexing)
 * @param[in]     n1     The order of the first block T11. n1 = 0, 1 or 2.
 * @param[in]     n2     The order of the second block T22. n2 = 0, 1 or 2.
 * @param[out]    work   Workspace array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - = 1: the transformed matrix T would be too far from
 *                           Schur form; the blocks are not swapped and T and
 *                           Q are unchanged.
 */
void dlaexc(const INT wantq, const INT n, f64* T, const INT ldt,
            f64* Q, const INT ldq, const INT j1, const INT n1,
            const INT n2, f64* work, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const INT LDD = 4;
    const INT LDX = 2;

    INT ierr, j2, j3, j4, k, nd;
    f64 cs, dnorm, eps, scale, smlnum, sn, t11, t22;
    f64 t33, tau, tau1, tau2, temp, thresh, wi1, wi2;
    f64 wr1, wr2, xnorm;
    f64 D[16];  /* 4x4 stored column-major */
    f64 U[3], U1[3], U2[3], X[4];  /* X is 2x2 stored column-major */

    *info = 0;

    /* Quick return if possible */
    if (n == 0 || n1 == 0 || n2 == 0)
        return;
    if (j1 + n1 > n)
        return;

    j2 = j1 + 1;
    j3 = j1 + 2;
    j4 = j1 + 3;

    if (n1 == 1 && n2 == 1) {
        /* Swap two 1-by-1 blocks. */
        t11 = T[j1 + j1 * ldt];
        t22 = T[j2 + j2 * ldt];

        /* Determine the transformation to perform the interchange. */
        dlartg(T[j1 + j2 * ldt], t22 - t11, &cs, &sn, &temp);

        /* Apply transformation to the matrix T. */
        if (j3 <= n - 1) {
            cblas_drot(n - j1 - 2, &T[j1 + j3 * ldt], ldt, &T[j2 + j3 * ldt], ldt, cs, sn);
        }
        cblas_drot(j1, &T[j1 * ldt], 1, &T[j2 * ldt], 1, cs, sn);

        T[j1 + j1 * ldt] = t22;
        T[j2 + j2 * ldt] = t11;

        if (wantq) {
            /* Accumulate transformation in the matrix Q. */
            cblas_drot(n, &Q[j1 * ldq], 1, &Q[j2 * ldq], 1, cs, sn);
        }

    } else {
        /* Swapping involves at least one 2-by-2 block.
           Copy the diagonal block of order N1+N2 to the local array D
           and compute its norm. */
        nd = n1 + n2;
        dlacpy("Full", nd, nd, &T[j1 + j1 * ldt], ldt, D, LDD);
        dnorm = dlange("Max", nd, nd, D, LDD, work);

        /* Compute machine-dependent threshold for test for accepting swap. */
        eps = dlamch("P");
        smlnum = dlamch("S") / eps;
        thresh = TEN * eps * dnorm;
        if (thresh < smlnum) thresh = smlnum;

        /* Solve T11*X - X*T22 = scale*T12 for X. */
        dlasy2(0, 0, -1, n1, n2, D, LDD, &D[n1 + n1 * LDD], LDD,
               &D[n1 * LDD], LDD, &scale, X, LDX, &xnorm, &ierr);

        /* Swap the adjacent diagonal blocks. */
        k = n1 + n1 + n2 - 3;

        if (k == 1) {
            /* N1 = 1, N2 = 2: generate elementary reflector H so that:
               ( scale, X11, X12 ) H = ( 0, 0, * ) */
            U[0] = scale;
            U[1] = X[0];
            U[2] = X[LDX];
            dlarfg(3, &U[2], U, 1, &tau);
            U[2] = ONE;
            t11 = T[j1 + j1 * ldt];

            /* Perform swap provisionally on diagonal block in D. */
            dlarfx("L", 3, 3, U, tau, D, LDD, work);
            dlarfx("R", 3, 3, U, tau, D, LDD, work);

            /* Test whether to reject swap. */
            if (fabs(D[2 + 0 * LDD]) > thresh ||
                fabs(D[2 + 1 * LDD]) > thresh ||
                fabs(D[2 + 2 * LDD] - t11) > thresh) {
                *info = 1;
                return;
            }

            /* Accept swap: apply transformation to the entire matrix T. */
            dlarfx("L", 3, n - j1, U, tau, &T[j1 + j1 * ldt], ldt, work);
            dlarfx("R", j2 + 1, 3, U, tau, &T[j1 * ldt], ldt, work);

            T[j3 + j1 * ldt] = ZERO;
            T[j3 + j2 * ldt] = ZERO;
            T[j3 + j3 * ldt] = t11;

            if (wantq) {
                /* Accumulate transformation in the matrix Q. */
                dlarfx("R", n, 3, U, tau, &Q[j1 * ldq], ldq, work);
            }

        } else if (k == 2) {
            /* N1 = 2, N2 = 1: generate elementary reflector H so that:
               H ( -X11 ) = ( * )
                 ( -X21 )   ( 0 )
                 ( scale)   ( 0 ) */
            U[0] = -X[0];
            U[1] = -X[1];
            U[2] = scale;
            dlarfg(3, &U[0], &U[1], 1, &tau);
            U[0] = ONE;
            t33 = T[j3 + j3 * ldt];

            /* Perform swap provisionally on diagonal block in D. */
            dlarfx("L", 3, 3, U, tau, D, LDD, work);
            dlarfx("R", 3, 3, U, tau, D, LDD, work);

            /* Test whether to reject swap. */
            if (fabs(D[1 + 0 * LDD]) > thresh ||
                fabs(D[2 + 0 * LDD]) > thresh ||
                fabs(D[0 + 0 * LDD] - t33) > thresh) {
                *info = 1;
                return;
            }

            /* Accept swap: apply transformation to the entire matrix T. */
            dlarfx("R", j3 + 1, 3, U, tau, &T[j1 * ldt], ldt, work);
            dlarfx("L", 3, n - j1 - 1, U, tau, &T[j1 + j2 * ldt], ldt, work);

            T[j1 + j1 * ldt] = t33;
            T[j2 + j1 * ldt] = ZERO;
            T[j3 + j1 * ldt] = ZERO;

            if (wantq) {
                /* Accumulate transformation in the matrix Q. */
                dlarfx("R", n, 3, U, tau, &Q[j1 * ldq], ldq, work);
            }

        } else {
            /* N1 = 2, N2 = 2: generate elementary reflectors H(1) and H(2) so
               that:
               H(2) H(1) ( -X11  -X12 ) = ( *  * )
                         ( -X21  -X22 )   ( 0  * )
                         ( scale   0  )   ( 0  0 )
                         (   0  scale )   ( 0  0 ) */
            U1[0] = -X[0];
            U1[1] = -X[1];
            U1[2] = scale;
            dlarfg(3, &U1[0], &U1[1], 1, &tau1);
            U1[0] = ONE;

            temp = -tau1 * (X[LDX] + U1[1] * X[1 + LDX]);
            U2[0] = -temp * U1[1] - X[1 + LDX];
            U2[1] = -temp * U1[2];
            U2[2] = scale;
            dlarfg(3, &U2[0], &U2[1], 1, &tau2);
            U2[0] = ONE;

            /* Perform swap provisionally on diagonal block in D. */
            dlarfx("L", 3, 4, U1, tau1, D, LDD, work);
            dlarfx("R", 4, 3, U1, tau1, D, LDD, work);
            dlarfx("L", 3, 4, U2, tau2, &D[1], LDD, work);
            dlarfx("R", 4, 3, U2, tau2, &D[LDD], LDD, work);

            /* Test whether to reject swap. */
            if (fabs(D[2 + 0 * LDD]) > thresh ||
                fabs(D[2 + 1 * LDD]) > thresh ||
                fabs(D[3 + 0 * LDD]) > thresh ||
                fabs(D[3 + 1 * LDD]) > thresh) {
                *info = 1;
                return;
            }

            /* Accept swap: apply transformation to the entire matrix T. */
            dlarfx("L", 3, n - j1, U1, tau1, &T[j1 + j1 * ldt], ldt, work);
            dlarfx("R", j4 + 1, 3, U1, tau1, &T[j1 * ldt], ldt, work);
            dlarfx("L", 3, n - j1, U2, tau2, &T[j2 + j1 * ldt], ldt, work);
            dlarfx("R", j4 + 1, 3, U2, tau2, &T[j2 * ldt], ldt, work);

            T[j3 + j1 * ldt] = ZERO;
            T[j3 + j2 * ldt] = ZERO;
            T[j4 + j1 * ldt] = ZERO;
            T[j4 + j2 * ldt] = ZERO;

            if (wantq) {
                /* Accumulate transformation in the matrix Q. */
                dlarfx("R", n, 3, U1, tau1, &Q[j1 * ldq], ldq, work);
                dlarfx("R", n, 3, U2, tau2, &Q[j2 * ldq], ldq, work);
            }
        }

        if (n2 == 2) {
            /* Standardize new 2-by-2 block T11 */
            dlanv2(&T[j1 + j1 * ldt], &T[j1 + j2 * ldt], &T[j2 + j1 * ldt],
                   &T[j2 + j2 * ldt], &wr1, &wi1, &wr2, &wi2, &cs, &sn);
            if (j1 + 2 <= n - 1) {
                cblas_drot(n - j1 - 2, &T[j1 + (j1 + 2) * ldt], ldt,
                          &T[j2 + (j1 + 2) * ldt], ldt, cs, sn);
            }
            cblas_drot(j1, &T[j1 * ldt], 1, &T[j2 * ldt], 1, cs, sn);
            if (wantq) {
                cblas_drot(n, &Q[j1 * ldq], 1, &Q[j2 * ldq], 1, cs, sn);
            }
        }

        if (n1 == 2) {
            /* Standardize new 2-by-2 block T22 */
            j3 = j1 + n2;
            j4 = j3 + 1;
            dlanv2(&T[j3 + j3 * ldt], &T[j3 + j4 * ldt], &T[j4 + j3 * ldt],
                   &T[j4 + j4 * ldt], &wr1, &wi1, &wr2, &wi2, &cs, &sn);
            if (j3 + 2 <= n - 1) {
                cblas_drot(n - j3 - 2, &T[j3 + (j3 + 2) * ldt], ldt,
                          &T[j4 + (j3 + 2) * ldt], ldt, cs, sn);
            }
            cblas_drot(j3, &T[j3 * ldt], 1, &T[j4 * ldt], 1, cs, sn);
            if (wantq) {
                cblas_drot(n, &Q[j3 * ldq], 1, &Q[j4 * ldq], 1, cs, sn);
            }
        }
    }
}
