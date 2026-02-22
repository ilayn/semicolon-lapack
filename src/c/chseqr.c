/**
 * @file chseqr.c
 * @brief CHSEQR computes eigenvalues of a Hessenberg matrix and optionally
 *        the Schur decomposition.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/** @cond */
static INT iparmq_nmin(void)
{
    return 75;
}
/** @endcond */

/**
 * CHSEQR computes the eigenvalues of a Hessenberg matrix H
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
 *                         - > 0: if info = i, CHSEQR failed to compute all eigenvalues.
 */
void chseqr(const char* job, const char* compz, const INT n,
            const INT ilo, const INT ihi,
            c64* H, const INT ldh,
            c64* W,
            c64* Z, const INT ldz,
            c64* work, const INT lwork, INT* info)
{
    const INT ntiny = 15;
    const INT nl = 49;
    const c64 czero = 0.0f;
    const c64 cone = 1.0f;

    c64 hl[49 * 49];
    c64 workl[49];

    INT kbot, nmin;
    INT initz, lquery, wantt, wantz;

    wantt = (job[0] == 'S' || job[0] == 's');
    initz = (compz[0] == 'I' || compz[0] == 'i');
    wantz = initz || (compz[0] == 'V' || compz[0] == 'v');
    work[0] = (f32)(1 > n ? 1 : n);
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
        xerbla("CHSEQR", -(*info));
        return;
    } else if (n == 0) {
        return;
    } else if (lquery) {
        claqr0(wantt, wantz, n, ilo, ihi, H, ldh, W, ilo,
               ihi, Z, ldz, work, lwork, info);
        if (crealf(work[0]) < (f32)(1 > n ? 1 : n))
            work[0] = (f32)(1 > n ? 1 : n);
        return;
    } else {
        /* Copy eigenvalues isolated by CGEBAL */
        if (ilo > 0)
            cblas_ccopy(ilo, H, ldh + 1, W, 1);
        if (ihi < n - 1)
            cblas_ccopy(n - 1 - ihi, &H[(ihi + 1) + (ihi + 1) * ldh],
                        ldh + 1, &W[ihi + 1], 1);

        /* Initialize Z, if requested */
        if (initz)
            claset("A", n, n, czero, cone, Z, ldz);

        /* Quick return if possible */
        if (ilo == ihi) {
            W[ilo] = H[ilo + ilo * ldh];
            return;
        }

        nmin = iparmq_nmin();
        if (nmin < ntiny) nmin = ntiny;

        /* CLAQR0 for big matrices; CLAHQR for small ones */
        if (n > nmin) {
            claqr0(wantt, wantz, n, ilo, ihi, H, ldh, W,
                   ilo, ihi, Z, ldz, work, lwork, info);
        } else {
            clahqr(wantt, wantz, n, ilo, ihi, H, ldh, W,
                   ilo, ihi, Z, ldz, info);

            if (*info > 0) {
                /* A rare CLAHQR failure! CLAQR0 sometimes succeeds
                 * when CLAHQR fails. */
                kbot = *info - 1;

                if (n >= nl) {
                    claqr0(wantt, wantz, n, ilo, kbot, H, ldh, W,
                           ilo, ihi, Z, ldz, work, lwork, info);
                } else {
                    /* Tiny matrices must be copied into a larger
                     * array before calling CLAQR0. */
                    clacpy("A", n, n, H, ldh, hl, nl);
                    hl[n + (n - 1) * nl] = czero;
                    claset("A", nl, nl - n, czero, czero, &hl[n * nl], nl);
                    claqr0(wantt, wantz, nl, ilo, kbot, hl, nl, W,
                           ilo, ihi, Z, ldz, workl, nl, info);
                    if (wantt || *info != 0)
                        clacpy("A", n, n, hl, nl, H, ldh);
                }
            }
        }

        /* Clear out the trash, if necessary */
        if ((wantt || *info != 0) && n > 2)
            claset("L", n - 2, n - 2, czero, czero, &H[2], ldh);

        /* Ensure reported workspace size is backward-compatible */
        if (crealf(work[0]) < (f32)(1 > n ? 1 : n))
            work[0] = (f32)(1 > n ? 1 : n);
    }
}
