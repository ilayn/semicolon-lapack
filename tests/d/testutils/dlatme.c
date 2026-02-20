/**
 * @file dlatme.c
 * @brief DLATME generates random non-symmetric square matrices with
 *        specified eigenvalues for testing LAPACK programs.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlatme.f
 * Uses xoshiro256+ RNG via test_rng.h instead of LAPACK's 48-bit LCG.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations for library functions */
extern void xerbla(const char* srname, const int info);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern void dlarfg(const int n, f64* alpha, f64* x, const int incx,
                   f64* tau);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);

/**
 * DLATME generates random non-symmetric square matrices with
 * specified eigenvalues for testing LAPACK programs.
 *
 * DLATME operates by applying the following sequence of operations:
 *
 * 1. Set the diagonal to D, where D may be input or computed according
 *    to MODE, COND, DMAX, and RSIGN as described below.
 *
 * 2. If complex conjugate pairs are desired (MODE=0 and EI(0)='R',
 *    or MODE=5), certain pairs of adjacent elements of D are
 *    interpreted as the real and complex parts of a complex
 *    conjugate pair; A thus becomes block diagonal, with 1x1
 *    and 2x2 blocks.
 *
 * 3. If UPPER='T', the upper triangle of A is set to random values
 *    out of distribution DIST.
 *
 * 4. If SIM='T', A is multiplied on the left by a random matrix
 *    X, whose singular values are specified by DS, MODES, and
 *    CONDS, and on the right by X inverse.
 *
 * 5. If KL < N-1, the lower bandwidth is reduced to KL using
 *    Householder transformations.  If KU < N-1, the upper
 *    bandwidth is reduced to KU.
 *
 * 6. If ANORM is not negative, the matrix is scaled to have
 *    maximum-element-norm ANORM.
 *
 * @param[in] n
 *     The number of columns (or rows) of A.  n >= 0.
 *
 * @param[in] dist
 *     Specifies the distribution for random numbers:
 *     'U' => UNIFORM( 0, 1 )
 *     'S' => UNIFORM( -1, 1 )
 *     'N' => NORMAL( 0, 1 )
 *
 * @param[in,out] D
 *     Array, dimension (n).
 *     On entry, if MODE=0, contains the eigenvalues.
 *     On exit, D is modified according to MODE, COND, DMAX, RSIGN.
 *
 * @param[in] mode
 *     Specifies how eigenvalues are computed (see dlatm1).
 *
 * @param[in] cond
 *     Condition number, used if MODE != 0, 6, -6.  cond >= 1.
 *
 * @param[in] dmax
 *     Scale factor for D if MODE != 0, 6, -6.
 *
 * @param[in] ei
 *     Character array, dimension (n).
 *     If MODE=0 and ei[0] != ' ', specifies which elements of D
 *     are real eigenvalues ('R') and which are parts of complex
 *     conjugate pairs ('I').
 *
 * @param[in] rsign
 *     If rsign='T' and MODE != 0, 6, -6, elements of D are multiplied
 *     by random +1 or -1.
 *
 * @param[in] upper
 *     If upper='T', upper triangle of A is set to random values.
 *
 * @param[in] sim
 *     If sim='T', A is transformed by similarity: X * A * X^{-1}.
 *
 * @param[in,out] DS
 *     Array, dimension (n).
 *     Singular values of transformation matrix X.
 *
 * @param[in] modes
 *     Mode for computing DS (same as MODE for D).
 *
 * @param[in] conds
 *     Condition number for DS.
 *
 * @param[in] kl
 *     Lower bandwidth.  kl >= 1.
 *
 * @param[in] ku
 *     Upper bandwidth.  ku >= 1.  At most one of kl, ku can be < n-1.
 *
 * @param[in] anorm
 *     If anorm >= 0, A is scaled to have max-element-norm = anorm.
 *
 * @param[out] A
 *     Array, dimension (lda, n).  The generated matrix.
 *
 * @param[in] lda
 *     Leading dimension of A.  lda >= max(1, n).
 *
 * @param[out] work
 *     Workspace array, dimension (3*n).
 *
 * @param[out] info
 *     = 0: success
 *     < 0: illegal argument
 *     > 0: error in called routine
 */
void dlatme(const int n, const char* dist, f64* D,
            const int mode, const f64 cond, const f64 dmax,
            const char* ei, const char* rsign, const char* upper,
            const char* sim, f64* DS, const int modes, const f64 conds,
            const int kl, const int ku, const f64 anorm,
            f64* A, const int lda, f64* work, int* info,
            uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    int badei, bads, useei;
    int i, ic, icols, idist, iinfo, ir, irows, irsign, isim, iupper, j, jc, jcr, jr;
    f64 alpha, tau, temp, xnorms;

    *info = 0;

    if (n == 0) {
        return;
    }

    /* Decode DIST */
    if (dist[0] == 'U' || dist[0] == 'u') {
        idist = 1;
    } else if (dist[0] == 'S' || dist[0] == 's') {
        idist = 2;
    } else if (dist[0] == 'N' || dist[0] == 'n') {
        idist = 3;
    } else {
        idist = -1;
    }

    /* Check EI */
    useei = 1;
    badei = 0;
    if (ei[0] == ' ' || mode != 0) {
        useei = 0;
    } else {
        if (ei[0] == 'R' || ei[0] == 'r') {
            for (j = 1; j < n; j++) {
                if (ei[j] == 'I' || ei[j] == 'i') {
                    if (ei[j-1] == 'I' || ei[j-1] == 'i') {
                        badei = 1;
                    }
                } else {
                    if (!(ei[j] == 'R' || ei[j] == 'r')) {
                        badei = 1;
                    }
                }
            }
        } else {
            badei = 1;
        }
    }

    /* Decode RSIGN */
    if (rsign[0] == 'T' || rsign[0] == 't') {
        irsign = 1;
    } else if (rsign[0] == 'F' || rsign[0] == 'f') {
        irsign = 0;
    } else {
        irsign = -1;
    }

    /* Decode UPPER */
    if (upper[0] == 'T' || upper[0] == 't') {
        iupper = 1;
    } else if (upper[0] == 'F' || upper[0] == 'f') {
        iupper = 0;
    } else {
        iupper = -1;
    }

    /* Decode SIM */
    if (sim[0] == 'T' || sim[0] == 't') {
        isim = 1;
    } else if (sim[0] == 'F' || sim[0] == 'f') {
        isim = 0;
    } else {
        isim = -1;
    }

    /* Check DS, if MODES=0 and ISIM=1 */
    bads = 0;
    if (modes == 0 && isim == 1) {
        for (j = 0; j < n; j++) {
            if (DS[j] == ZERO) {
                bads = 1;
            }
        }
    }

    /* Set INFO if an error */
    if (n < 0) {
        *info = -1;
    } else if (idist == -1) {
        *info = -2;
    } else if (mode < -6 || mode > 6) {
        *info = -5;
    } else if ((mode != 0 && mode != 6 && mode != -6) && cond < ONE) {
        *info = -6;
    } else if (badei) {
        *info = -8;
    } else if (irsign == -1) {
        *info = -9;
    } else if (iupper == -1) {
        *info = -10;
    } else if (isim == -1) {
        *info = -11;
    } else if (bads) {
        *info = -12;
    } else if (isim == 1 && (modes < -5 || modes > 5)) {
        *info = -13;
    } else if (isim == 1 && modes != 0 && conds < ONE) {
        *info = -14;
    } else if (kl < 1) {
        *info = -15;
    } else if (ku < 1 || (ku < n - 1 && kl < n - 1)) {
        *info = -16;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -19;
    }

    if (*info != 0) {
        xerbla("DLATME", -(*info));
        return;
    }

    /* 2) Set up diagonal of A - Compute D according to COND and MODE */
    dlatm1(mode, cond, irsign, idist, D, n, &iinfo, state);
    if (iinfo != 0) {
        *info = 1;
        return;
    }

    if (mode != 0 && mode != 6 && mode != -6) {
        /* Scale by DMAX */
        temp = fabs(D[0]);
        for (i = 1; i < n; i++) {
            if (fabs(D[i]) > temp) {
                temp = fabs(D[i]);
            }
        }

        if (temp > ZERO) {
            alpha = dmax / temp;
        } else if (dmax != ZERO) {
            *info = 2;
            return;
        } else {
            alpha = ZERO;
        }

        cblas_dscal(n, alpha, D, 1);
    }

    dlaset("F", n, n, ZERO, ZERO, A, lda);
    cblas_dcopy(n, D, 1, A, lda + 1);

    /* Set up complex conjugate pairs */
    if (mode == 0) {
        if (useei) {
            for (j = 1; j < n; j++) {
                if (ei[j] == 'I' || ei[j] == 'i') {
                    A[(j-1) + j * lda] = A[j + j * lda];
                    A[j + (j-1) * lda] = -A[j + j * lda];
                    A[j + j * lda] = A[(j-1) + (j-1) * lda];
                }
            }
        }
    } else if (mode == 5 || mode == -5) {
        for (j = 1; j < n; j += 2) {
            if (dlaran_rng(state) > HALF) {
                A[(j-1) + j * lda] = A[j + j * lda];
                A[j + (j-1) * lda] = -A[j + j * lda];
                A[j + j * lda] = A[(j-1) + (j-1) * lda];
            }
        }
    }

    /* 3) If UPPER='T', set upper triangle of A to random numbers. */
    if (iupper != 0) {
        for (jc = 1; jc < n; jc++) {
            if (A[(jc-1) + jc * lda] != ZERO) {
                jr = jc - 2;
            } else {
                jr = jc - 1;
            }
            if (jr >= 0) {
                dlarnv_rng(idist, jr + 1, &A[0 + jc * lda], state);
            }
        }
    }

    /* 4) If SIM='T', apply similarity transformation.
     *    Transform is  X A X^{-1}, where X = U S V, thus
     *    it is  U S V A V' (1/S) U' */
    if (isim != 0) {
        /* Compute S (singular values of the eigenvector matrix)
         * according to CONDS and MODES */
        dlatm1(modes, conds, 0, 0, DS, n, &iinfo, state);
        if (iinfo != 0) {
            *info = 3;
            return;
        }

        /* Multiply by V and V' */
        dlarge(n, A, lda, work, &iinfo, state);
        if (iinfo != 0) {
            *info = 4;
            return;
        }

        /* Multiply by S and (1/S) */
        for (j = 0; j < n; j++) {
            cblas_dscal(n, DS[j], &A[j], lda);
            if (DS[j] != ZERO) {
                cblas_dscal(n, ONE / DS[j], &A[j * lda], 1);
            } else {
                *info = 5;
                return;
            }
        }

        /* Multiply by U and U' */
        dlarge(n, A, lda, work, &iinfo, state);
        if (iinfo != 0) {
            *info = 4;
            return;
        }
    }

    /* 5) Reduce the bandwidth. */
    if (kl < n - 1) {
        /* Reduce bandwidth -- kill column */
        for (jcr = kl; jcr < n - 1; jcr++) {
            ic = jcr - kl;
            irows = n - jcr;
            icols = n - 1 + kl - jcr;

            cblas_dcopy(irows, &A[jcr + ic * lda], 1, work, 1);
            xnorms = work[0];
            dlarfg(irows, &xnorms, &work[1], 1, &tau);
            work[0] = ONE;

            cblas_dgemv(CblasColMajor, CblasTrans, irows, icols, ONE,
                        &A[jcr + (ic + 1) * lda], lda, work, 1, ZERO,
                        &work[irows], 1);
            cblas_dger(CblasColMajor, irows, icols, -tau, work, 1,
                       &work[irows], 1, &A[jcr + (ic + 1) * lda], lda);

            cblas_dgemv(CblasColMajor, CblasNoTrans, n, irows, ONE,
                        &A[0 + jcr * lda], lda, work, 1, ZERO,
                        &work[irows], 1);
            cblas_dger(CblasColMajor, n, irows, -tau, &work[irows], 1,
                       work, 1, &A[0 + jcr * lda], lda);

            A[jcr + ic * lda] = xnorms;
            dlaset("F", irows - 1, 1, ZERO, ZERO, &A[jcr + 1 + ic * lda], lda);
        }
    } else if (ku < n - 1) {
        /* Reduce upper bandwidth -- kill a row at a time. */
        for (jcr = ku; jcr < n - 1; jcr++) {
            ir = jcr - ku;
            irows = n - 1 + ku - jcr;
            icols = n - jcr;

            cblas_dcopy(icols, &A[ir + jcr * lda], lda, work, 1);
            xnorms = work[0];
            dlarfg(icols, &xnorms, &work[1], 1, &tau);
            work[0] = ONE;

            cblas_dgemv(CblasColMajor, CblasNoTrans, irows, icols, ONE,
                        &A[ir + 1 + jcr * lda], lda, work, 1, ZERO,
                        &work[icols], 1);
            cblas_dger(CblasColMajor, irows, icols, -tau, &work[icols], 1,
                       work, 1, &A[ir + 1 + jcr * lda], lda);

            cblas_dgemv(CblasColMajor, CblasTrans, icols, n, ONE,
                        &A[jcr], lda, work, 1, ZERO,
                        &work[icols], 1);
            cblas_dger(CblasColMajor, icols, n, -tau, work, 1,
                       &work[icols], 1, &A[jcr], lda);

            A[ir + jcr * lda] = xnorms;
            dlaset("F", 1, icols - 1, ZERO, ZERO, &A[ir + (jcr + 1) * lda], lda);
        }
    }

    /* Scale the matrix to have norm ANORM */
    if (anorm >= ZERO) {
        temp = dlange("M", n, n, A, lda, work);
        if (temp > ZERO) {
            alpha = anorm / temp;
            for (j = 0; j < n; j++) {
                cblas_dscal(n, alpha, &A[j * lda], 1);
            }
        }
    }

}
