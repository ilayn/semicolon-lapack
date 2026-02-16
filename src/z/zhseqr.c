/**
 * @file zhseqr.c
 * @brief ZHSEQR computes eigenvalues of a Hessenberg matrix and optionally
 *        the Schur decomposition.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/** @cond */
static int iparmq_nmin(void)
{
    return 75;
}
/** @endcond */

/**
 * ZHSEQR computes the eigenvalues of a Hessenberg matrix H
 * and, optionally, the matrices T and Z from the Schur decomposition
 * H = Z T Z^H, where T is an upper triangular matrix (the
 * Schur form), and Z is the unitary matrix of Schur vectors.
 *
 * Optionally Z may be postmultiplied into an input unitary
 * matrix Q so that this routine can give the Schur factorization
 * of a matrix A which has been reduced to the Hessenberg form H
 * by the unitary matrix Q:  A = Q*H*Q^H = (QZ)*T*(QZ)^H.
 *
 * @param[in] job     'E' for eigenvalues only; 'S' for Schur form T.
 * @param[in] compz   'N' no Schur vectors; 'I' initialize Z to identity;
 *                    'V' Z contains input unitary matrix Q.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ilo     First index of isolated block (0-based).
 * @param[in] ihi     Last index of isolated block (0-based).
 * @param[in,out] H   Complex array, dimension (ldh, n).
 *                    On entry, the upper Hessenberg matrix H.
 *                    On exit, if info = 0 and job = 'S', then H contains
 *                    the upper triangular matrix T (the Schur form).
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[out] W      Complex array, dimension (n).
 *                    The computed eigenvalues.
 * @param[in,out] Z   Complex array, dimension (ldz, n).
 * @param[in] ldz     Leading dimension of Z. ldz >= 1; if compz = 'I' or 'V',
 *                    ldz >= max(1, n).
 * @param[out] work   Complex array, dimension (lwork).
 * @param[in] lwork   Dimension of work array. lwork >= max(1, n).
 *                    If lwork = -1, workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, ZHSEQR failed to compute all eigenvalues.
 */
void zhseqr(const char* job, const char* compz, const int n,
            const int ilo, const int ihi,
            c128* H, const int ldh,
            c128* W,
            c128* Z, const int ldz,
            c128* work, const int lwork, int* info)
{
    const int ntiny = 15;
    const int nl = 49;
    const c128 czero = 0.0;
    const c128 cone = 1.0;

    c128 hl[49 * 49];
    c128 workl[49];

    int kbot, nmin;
    int initz, lquery, wantt, wantz;

    wantt = (job[0] == 'S' || job[0] == 's');
    initz = (compz[0] == 'I' || compz[0] == 'i');
    wantz = initz || (compz[0] == 'V' || compz[0] == 'v');
    work[0] = (f64)(1 > n ? 1 : n);
    lquery = (lwork == -1);

    *info = 0;
    if (!(job[0] == 'E' || job[0] == 'e') && !wantt) {
        *info = -1;
    } else if (!(compz[0] == 'N' || compz[0] == 'n') && !wantz) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 0 || ilo > (n > 0 ? n - 1 : 0)) {
        *info = -4;
    } else if (n > 0 && (ihi < ilo || ihi > n - 1)) {
        *info = -5;
    } else if (n == 0 && ihi != -1) {
        *info = -5;
    } else if (ldh < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldz < 1 || (wantz && ldz < (1 > n ? 1 : n))) {
        *info = -10;
    } else if (lwork < (1 > n ? 1 : n) && !lquery) {
        *info = -12;
    }

    if (*info != 0) {
        xerbla("ZHSEQR", -(*info));
        return;
    } else if (n == 0) {
        return;
    } else if (lquery) {
        zlaqr0(wantt, wantz, n, ilo, ihi, H, ldh, W, ilo,
               ihi, Z, ldz, work, lwork, info);
        if (creal(work[0]) < (f64)(1 > n ? 1 : n))
            work[0] = (f64)(1 > n ? 1 : n);
        return;
    } else {
        /* Copy eigenvalues isolated by ZGEBAL */
        if (ilo > 0)
            cblas_zcopy(ilo, H, ldh + 1, W, 1);
        if (ihi < n - 1)
            cblas_zcopy(n - 1 - ihi, &H[(ihi + 1) + (ihi + 1) * ldh],
                        ldh + 1, &W[ihi + 1], 1);

        /* Initialize Z, if requested */
        if (initz)
            zlaset("A", n, n, czero, cone, Z, ldz);

        /* Quick return if possible */
        if (ilo == ihi) {
            W[ilo] = H[ilo + ilo * ldh];
            return;
        }

        nmin = iparmq_nmin();
        if (nmin < ntiny) nmin = ntiny;

        /* ZLAQR0 for big matrices; ZLAHQR for small ones */
        if (n > nmin) {
            zlaqr0(wantt, wantz, n, ilo, ihi, H, ldh, W,
                   ilo, ihi, Z, ldz, work, lwork, info);
        } else {
            zlahqr(wantt, wantz, n, ilo, ihi, H, ldh, W,
                   ilo, ihi, Z, ldz, info);

            if (*info > 0) {
                /* A rare ZLAHQR failure! ZLAQR0 sometimes succeeds
                 * when ZLAHQR fails. */
                kbot = *info - 1;

                if (n >= nl) {
                    zlaqr0(wantt, wantz, n, ilo, kbot, H, ldh, W,
                           ilo, ihi, Z, ldz, work, lwork, info);
                } else {
                    /* Tiny matrices must be copied into a larger
                     * array before calling ZLAQR0. */
                    zlacpy("A", n, n, H, ldh, hl, nl);
                    hl[n + (n - 1) * nl] = czero;
                    zlaset("A", nl, nl - n, czero, czero, &hl[n * nl], nl);
                    zlaqr0(wantt, wantz, nl, ilo, kbot, hl, nl, W,
                           ilo, ihi, Z, ldz, workl, nl, info);
                    if (wantt || *info != 0)
                        zlacpy("A", n, n, hl, nl, H, ldh);
                }
            }
        }

        /* Clear out the trash, if necessary */
        if ((wantt || *info != 0) && n > 2)
            zlaset("L", n - 2, n - 2, czero, czero, &H[2], ldh);

        /* Ensure reported workspace size is backward-compatible */
        if (creal(work[0]) < (f64)(1 > n ? 1 : n))
            work[0] = (f64)(1 > n ? 1 : n);
    }
}
