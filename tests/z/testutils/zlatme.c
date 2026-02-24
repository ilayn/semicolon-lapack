/**
 * @file zlatme.c
 * @brief ZLATME generates random non-symmetric square matrices with
 *        specified eigenvalues for testing LAPACK programs.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlatme.f
 * Uses xoshiro256+ RNG via test_rng.h instead of LAPACK's 48-bit LCG.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

extern void xerbla(const char* srname, const INT info);

/**
 * ZLATME generates random non-symmetric square matrices with
 * specified eigenvalues for testing LAPACK programs.
 *
 * ZLATME operates by applying the following sequence of operations:
 *
 * 1. Set the diagonal to D, where D may be input or computed according
 *    to MODE, COND, DMAX, and RSIGN as described below.
 *
 * 2. If UPPER='T', the upper triangle of A is set to random values
 *    out of distribution DIST.
 *
 * 3. If SIM='T', A is multiplied on the left by a random matrix
 *    X, whose singular values are specified by DS, MODES, and
 *    CONDS, and on the right by X inverse.
 *
 * 4. If KL < N-1, the lower bandwidth is reduced to KL using
 *    Householder transformations.  If KU < N-1, the upper
 *    bandwidth is reduced to KU.
 *
 * 5. If ANORM is not negative, the matrix is scaled to have
 *    maximum-element-norm ANORM.
 *
 * (Note: since the matrix cannot be reduced beyond Hessenberg form,
 *  no packing options are available.)
 *
 * @param[in] n
 *     The number of columns (or rows) of A.  n >= 0.
 *
 * @param[in] dist
 *     Specifies the distribution for random numbers:
 *     'U' => UNIFORM( 0, 1 )
 *     'S' => UNIFORM( -1, 1 )
 *     'N' => NORMAL( 0, 1 )
 *     'D' => uniform on the complex disc |z| < 1
 *
 * @param[in,out] D
 *     Complex array, dimension (n).
 *     On entry, if MODE=0, contains the eigenvalues.
 *     On exit, D is modified according to MODE, COND, DMAX, RSIGN.
 *
 * @param[in] mode
 *     Specifies how eigenvalues are computed (see zlatm1).
 *
 * @param[in] cond
 *     Condition number, used if MODE != 0, 6, -6.  cond >= 1.
 *
 * @param[in] dmax
 *     Complex scale factor for D if MODE != 0, 6, -6.
 *     If RSIGN='F' then the largest (absolute) eigenvalue will be
 *     equal to DMAX.
 *
 * @param[in] rsign
 *     If rsign='T' and MODE != 0, 6, -6, elements of D are multiplied
 *     by a random complex number from the unit circle |z| = 1.
 *
 * @param[in] upper
 *     If upper='T', upper triangle of A is set to random values.
 *
 * @param[in] sim
 *     If sim='T', A is transformed by similarity: X * A * X^{-1}.
 *
 * @param[in,out] DS
 *     Double precision array, dimension (n).
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
 *     Complex array, dimension (lda, n).  The generated matrix.
 *
 * @param[in] lda
 *     Leading dimension of A.  lda >= max(1, n).
 *
 * @param[out] work
 *     Complex workspace array, dimension (3*n).
 *
 * @param[out] info
 *     = 0: success
 *     < 0: illegal argument
 *     > 0: error in called routine
 */
void zlatme(const INT n, const char* dist, c128* D,
            const INT mode, const f64 cond, const c128 dmax,
            const char* rsign, const char* upper,
            const char* sim, f64* DS, const INT modes, const f64 conds,
            const INT kl, const INT ku, const f64 anorm,
            c128* A, const INT lda, c128* work, INT* info,
            uint64_t state[static 4])
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT bads;
    INT i, ic, icols, idist, iinfo, ir, irows, irsign, isim, iupper, j, jc, jcr;
    f64 ralpha, temp;
    c128 alpha, tau, xnorms;
    f64 tempa[1];

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
    } else if (dist[0] == 'D' || dist[0] == 'd') {
        idist = 4;
    } else {
        idist = -1;
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
        xerbla("ZLATME", -(*info));
        return;
    }

    /* 2) Set up diagonal of A - Compute D according to COND and MODE */
    zlatm1(mode, cond, irsign, idist, D, n, &iinfo, state);
    if (iinfo != 0) {
        *info = 1;
        return;
    }

    if (mode != 0 && mode != 6 && mode != -6) {
        /* Scale by DMAX */
        temp = cabs(D[0]);
        for (i = 1; i < n; i++) {
            f64 t = cabs(D[i]);
            if (t > temp) {
                temp = t;
            }
        }

        if (temp > ZERO) {
            alpha = dmax / temp;
        } else {
            *info = 2;
            return;
        }

        cblas_zscal(n, &alpha, D, 1);
    }

    zlaset("F", n, n, CZERO, CZERO, A, lda);
    cblas_zcopy(n, D, 1, A, lda + 1);

    /* 3) If UPPER='T', set upper triangle of A to random numbers. */
    if (iupper != 0) {
        for (jc = 1; jc < n; jc++) {
            zlarnv_rng(idist, jc, &A[0 + jc * lda], state);
        }
    }

    /* 4) If SIM='T', apply similarity transformation.
     *                                -1
     *    Transform is  X A X  , where X = U S V, thus
     *
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
        zlarge(n, A, lda, work, &iinfo, state);
        if (iinfo != 0) {
            *info = 4;
            return;
        }

        /* Multiply by S and (1/S) */
        for (j = 0; j < n; j++) {
            cblas_zdscal(n, DS[j], &A[j], lda);
            if (DS[j] != ZERO) {
                cblas_zdscal(n, ONE / DS[j], &A[j * lda], 1);
            } else {
                *info = 5;
                return;
            }
        }

        /* Multiply by U and U' */
        zlarge(n, A, lda, work, &iinfo, state);
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

            cblas_zcopy(irows, &A[jcr + ic * lda], 1, work, 1);
            xnorms = work[0];
            zlarfg(irows, &xnorms, &work[1], 1, &tau);
            tau = conj(tau);
            work[0] = CONE;
            alpha = zlarnd_rng(5, state);

            cblas_zgemv(CblasColMajor, CblasConjTrans, irows, icols, &CONE,
                        &A[jcr + (ic + 1) * lda], lda, work, 1, &CZERO,
                        &work[irows], 1);
            c128 neg_tau = -tau;
            cblas_zgerc(CblasColMajor, irows, icols, &neg_tau, work, 1,
                        &work[irows], 1, &A[jcr + (ic + 1) * lda], lda);

            cblas_zgemv(CblasColMajor, CblasNoTrans, n, irows, &CONE,
                        &A[0 + jcr * lda], lda, work, 1, &CZERO,
                        &work[irows], 1);
            c128 neg_conj_tau = -conj(tau);
            cblas_zgerc(CblasColMajor, n, irows, &neg_conj_tau, &work[irows], 1,
                        work, 1, &A[0 + jcr * lda], lda);

            A[jcr + ic * lda] = xnorms;
            zlaset("F", irows - 1, 1, CZERO, CZERO, &A[jcr + 1 + ic * lda], lda);

            cblas_zscal(icols + 1, &alpha, &A[jcr + ic * lda], lda);
            c128 conj_alpha = conj(alpha);
            cblas_zscal(n, &conj_alpha, &A[0 + jcr * lda], 1);
        }
    } else if (ku < n - 1) {
        /* Reduce upper bandwidth -- kill a row at a time. */
        for (jcr = ku; jcr < n - 1; jcr++) {
            ir = jcr - ku;
            irows = n - 1 + ku - jcr;
            icols = n - jcr;

            cblas_zcopy(icols, &A[ir + jcr * lda], lda, work, 1);
            xnorms = work[0];
            zlarfg(icols, &xnorms, &work[1], 1, &tau);
            tau = conj(tau);
            work[0] = CONE;
            zlacgv(icols - 1, &work[1], 1);
            alpha = zlarnd_rng(5, state);

            cblas_zgemv(CblasColMajor, CblasNoTrans, irows, icols, &CONE,
                        &A[ir + 1 + jcr * lda], lda, work, 1, &CZERO,
                        &work[icols], 1);
            c128 neg_tau = -tau;
            cblas_zgerc(CblasColMajor, irows, icols, &neg_tau, &work[icols], 1,
                        work, 1, &A[ir + 1 + jcr * lda], lda);

            cblas_zgemv(CblasColMajor, CblasConjTrans, icols, n, &CONE,
                        &A[jcr], lda, work, 1, &CZERO,
                        &work[icols], 1);
            c128 neg_conj_tau = -conj(tau);
            cblas_zgerc(CblasColMajor, icols, n, &neg_conj_tau, work, 1,
                        &work[icols], 1, &A[jcr], lda);

            A[ir + jcr * lda] = xnorms;
            zlaset("F", 1, icols - 1, CZERO, CZERO, &A[ir + (jcr + 1) * lda], lda);

            cblas_zscal(irows + 1, &alpha, &A[ir + jcr * lda], 1);
            c128 conj_alpha = conj(alpha);
            cblas_zscal(n, &conj_alpha, &A[jcr], lda);
        }
    }

    /* Scale the matrix to have norm ANORM */
    if (anorm >= ZERO) {
        temp = zlange("M", n, n, A, lda, tempa);
        if (temp > ZERO) {
            ralpha = anorm / temp;
            for (j = 0; j < n; j++) {
                cblas_zdscal(n, ralpha, &A[j * lda], 1);
            }
        }
    }
}
