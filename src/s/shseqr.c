/**
 * @file shseqr.c
 * @brief SHSEQR computes eigenvalues of a Hessenberg matrix and optionally
 *        the Schur decomposition.
 */

#include "semicolon_lapack_single.h"

/* ISPEC=12: NMIN - crossover to SLAHQR (from iparmq.f) */
static int iparmq_nmin(void)
{
    return 75;
}

/**
 * SHSEQR computes the eigenvalues of a Hessenberg matrix H
 * and, optionally, the matrices T and Z from the Schur decomposition
 * H = Z T Z^T, where T is an upper quasi-triangular matrix (the
 * Schur form), and Z is the orthogonal matrix of Schur vectors.
 *
 * Optionally Z may be postmultiplied into an input orthogonal
 * matrix Q so that this routine can give the Schur factorization
 * of a matrix A which has been reduced to the Hessenberg form H
 * by the orthogonal matrix Q:  A = Q*H*Q^T = (QZ)*T*(QZ)^T.
 *
 * @param[in] job     'E' for eigenvalues only; 'S' for Schur form T.
 * @param[in] compz   'N' no Schur vectors; 'I' initialize Z to identity;
 *                    'V' Z contains input orthogonal matrix Q.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ilo     First index of isolated block (0-based).
 *                    0 <= ilo <= ihi <= n-1, if n > 0; ilo=0 and ihi=-1, if n=0.
 * @param[in] ihi     Last index of isolated block (0-based).
 *                    It is assumed that H is already upper triangular in rows
 *                    and columns 0:ilo-1 and ihi+1:n-1.
 * @param[in,out] H   Double precision array, dimension (ldh, n).
 *                    On entry, the upper Hessenberg matrix H.
 *                    On exit, if info = 0 and job = 'S', then H contains
 *                    the upper quasi-triangular matrix T (the Schur form).
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[out] wr     Double precision array, dimension (n).
 *                    Real parts of the computed eigenvalues.
 * @param[out] wi     Double precision array, dimension (n).
 *                    Imaginary parts of the computed eigenvalues.
 * @param[in,out] Z   Double precision array, dimension (ldz, n).
 *                    If compz = 'N', Z is not referenced.
 *                    If compz = 'I', Z is initialized to identity and returns
 *                    the orthogonal matrix of Schur vectors.
 *                    If compz = 'V', Z contains an orthogonal matrix Q on entry,
 *                    and returns Q*Z.
 * @param[in] ldz     Leading dimension of Z. ldz >= 1; if compz = 'I' or 'V',
 *                    ldz >= max(1, n).
 * @param[out] work   Double precision array, dimension (lwork).
 * @param[in] lwork   Dimension of work array. lwork >= max(1, n).
 *                    If lwork = -1, workspace query is assumed.
 * @param[out] info   = 0: successful exit
 *                    < 0: if info = -i, the i-th argument had an illegal value
 *                    > 0: if info = i, SHSEQR failed to compute all eigenvalues.
 *                         Elements 0:ilo-1 and i+1:n-1 of WR and WI contain those
 *                         eigenvalues which have been successfully computed.
 */
void shseqr(const char* job, const char* compz, const int n,
                          const int ilo, const int ihi,
                          float* H, const int ldh,
                          float* wr, float* wi,
                          float* Z, const int ldz,
                          float* work, const int lwork, int* info)
{
    /* Parameters */
    const int ntiny = 15;  /* Matrices of order NTINY or smaller use SLAHQR */
    /* NL = 49: allocates local workspace to help small matrices through
     * a rare SLAHQR failure. Allows up to six simultaneous shifts and
     * a 16-by-16 deflation window. */
    const int nl = 49;
    const float zero = 0.0f;
    const float one = 1.0f;

    /* Local arrays - use explicit size to avoid VLA */
    float hl[49 * 49];
    float workl[49];

    /* Local scalars */
    int i, kbot, nmin;
    int initz, lquery, wantt, wantz;

    /* Decode and check the input parameters */
    wantt = (job[0] == 'S' || job[0] == 's');
    initz = (compz[0] == 'I' || compz[0] == 'i');
    wantz = initz || (compz[0] == 'V' || compz[0] == 'v');
    work[0] = (float)(1 > n ? 1 : n);
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
        *info = -11;
    } else if (lwork < (1 > n ? 1 : n) && !lquery) {
        *info = -13;
    }

    if (*info != 0) {
        /* Quick return in case of invalid argument */
        xerbla("SHSEQR", -(*info));
        return;
    } else if (n == 0) {
        /* Quick return in case N = 0; nothing to do */
        return;
    } else if (lquery) {
        /* Quick return in case of a workspace query */
        slaqr0(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, ilo,
               ihi, Z, ldz, work, lwork, info);
        /* Ensure reported workspace size is backward-compatible with
         * previous LAPACK versions */
        if (work[0] < (float)(1 > n ? 1 : n))
            work[0] = (float)(1 > n ? 1 : n);
        return;
    } else {
        /* Copy eigenvalues isolated by SGEBAL */
        /* These are diagonal elements in rows 0:ilo-1 and ihi+1:n-1 */
        for (i = 0; i < ilo; i++) {
            wr[i] = H[i + i * ldh];
            wi[i] = zero;
        }
        for (i = ihi + 1; i < n; i++) {
            wr[i] = H[i + i * ldh];
            wi[i] = zero;
        }

        /* Initialize Z, if requested */
        if (initz)
            slaset("A", n, n, zero, one, Z, ldz);

        /* Quick return if possible */
        if (ilo == ihi) {
            wr[ilo] = H[ilo + ilo * ldh];
            wi[ilo] = zero;
            return;
        }

        /* SLAHQR/SLAQR0 crossover point */
        nmin = iparmq_nmin();
        if (nmin < ntiny) nmin = ntiny;

        /* SLAQR0 for big matrices; SLAHQR for small ones */
        if (n > nmin) {
            slaqr0(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi,
                   ilo, ihi, Z, ldz, work, lwork, info);
        } else {
            /* Small matrix */
            slahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi,
                   ilo, ihi, Z, ldz, info);

            if (*info > 0) {
                /* A rare SLAHQR failure! SLAQR0 sometimes succeeds
                 * when SLAHQR fails. */
                /* info is 1-based from slahqr, convert to 0-based kbot */
                kbot = *info - 1;

                if (n >= nl) {
                    /* Larger matrices have enough subdiagonal scratch
                     * space to call SLAQR0 directly. */
                    slaqr0(wantt, wantz, n, ilo, kbot, H, ldh, wr,
                           wi, ilo, ihi, Z, ldz, work, lwork, info);
                } else {
                    /* Tiny matrices don't have enough subdiagonal
                     * scratch space to benefit from SLAQR0. Hence,
                     * tiny matrices must be copied into a larger
                     * array before calling SLAQR0. */
                    slacpy("A", n, n, H, ldh, hl, nl);
                    hl[n + n * nl] = zero;  /* HL(N+1, N) = ZERO (in 1-based) */
                    slaset("A", nl, nl - n, zero, zero, &hl[n * nl], nl);
                    slaqr0(wantt, wantz, nl, ilo, kbot, hl, nl, wr,
                           wi, ilo, ihi, Z, ldz, workl, nl, info);
                    if (wantt || *info != 0)
                        slacpy("A", n, n, hl, nl, H, ldh);
                }
            }
        }

        /* Clear out the trash, if necessary */
        if ((wantt || *info != 0) && n > 2)
            slaset("L", n - 2, n - 2, zero, zero, &H[2], ldh);

        /* Ensure reported workspace size is backward-compatible with
         * previous LAPACK versions */
        if (work[0] < (float)(1 > n ? 1 : n))
            work[0] = (float)(1 > n ? 1 : n);
    }
}
