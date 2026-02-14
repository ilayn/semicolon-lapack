/**
 * @file sgesvd.c
 * @brief SGESVD computes the singular value decomposition (SVD) of a general matrix.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

/*
 * SGESVD computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix. The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order. The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * Note that the routine returns V**T, not V.
 *
 * @param[in]     jobu   Specifies options for computing all or part of U:
 *                       = 'A': all M columns of U are returned in array U;
 *                       = 'S': the first min(m,n) columns of U are returned in U;
 *                       = 'O': the first min(m,n) columns of U are overwritten on A;
 *                       = 'N': no columns of U are computed.
 * @param[in]     jobvt  Specifies options for computing all or part of V**T:
 *                       = 'A': all N rows of V**T are returned in array VT;
 *                       = 'S': the first min(m,n) rows of V**T are returned in VT;
 *                       = 'O': the first min(m,n) rows of V**T are overwritten on A;
 *                       = 'N': no rows of V**T are computed.
 *                       JOBVT and JOBU cannot both be 'O'.
 * @param[in]     m      The number of rows of the input matrix A. m >= 0.
 * @param[in]     n      The number of columns of the input matrix A. n >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, contents depend on jobu and jobvt.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    S      Double precision array, dimension (min(m,n)).
 *                       The singular values of A, sorted so that S[i] >= S[i+1].
 * @param[out]    U      Double precision array, dimension (ldu, ucol).
 *                       If jobu = 'A', U contains the M-by-M orthogonal matrix U;
 *                       if jobu = 'S', U contains the first min(m,n) columns of U;
 *                       if jobu = 'N' or 'O', U is not referenced.
 * @param[in]     ldu    The leading dimension of the array U. ldu >= 1; if
 *                       jobu = 'S' or 'A', ldu >= m.
 * @param[out]    VT     Double precision array, dimension (ldvt, n).
 *                       If jobvt = 'A', VT contains the N-by-N orthogonal matrix V**T;
 *                       if jobvt = 'S', VT contains the first min(m,n) rows of V**T;
 *                       if jobvt = 'N' or 'O', VT is not referenced.
 * @param[in]     ldvt   The leading dimension of the array VT. ldvt >= 1; if
 *                       jobvt = 'A', ldvt >= n; if jobvt = 'S', ldvt >= min(m,n).
 * @param[out]    work   Double precision array, dimension (max(1,lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If lwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if SBDSQR did not converge.
 */
void sgesvd(const char* jobu, const char* jobvt,
            const int m, const int n,
            f32* restrict A, const int lda,
            f32* restrict S,
            f32* restrict U, const int ldu,
            f32* restrict VT, const int ldvt,
            f32* restrict work, const int lwork,
            int* info)
{
    /* Constants */
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    /* Helper macro for 3-way max (matches Fortran MAX(a,b,c)) */
    #define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

    /* Local variables */
    int wntua, wntus, wntuas, wntuo, wntun;
    int wntva, wntvs, wntvas, wntvo, wntvn;
    int lquery, minmn, mnthr;
    int bdspac = 0, minwrk, maxwrk, wrkbl = 0;
    int lwork_dgeqrf, lwork_dorgqr_n, lwork_dorgqr_m;
    int lwork_dgebrd, lwork_dorgbr_p, lwork_dorgbr_q;
    int lwork_dgelqf, lwork_dorglq_n, lwork_dorglq_m;
    int ie = 0, itau, itauq, itaup, iwork, ir, iu, chunk;
    int i, ierr, iscl, ncu, ncvt, nru, nrvt;
    int ldwrkr, ldwrku;
    f32 anrm, bignum, eps, smlnum;
    f32 dum[1];

    /* Parse job options */
    wntua = (jobu[0] == 'A' || jobu[0] == 'a');
    wntus = (jobu[0] == 'S' || jobu[0] == 's');
    wntuas = wntua || wntus;
    wntuo = (jobu[0] == 'O' || jobu[0] == 'o');
    wntun = (jobu[0] == 'N' || jobu[0] == 'n');

    wntva = (jobvt[0] == 'A' || jobvt[0] == 'a');
    wntvs = (jobvt[0] == 'S' || jobvt[0] == 's');
    wntvas = wntva || wntvs;
    wntvo = (jobvt[0] == 'O' || jobvt[0] == 'o');
    wntvn = (jobvt[0] == 'N' || jobvt[0] == 'n');

    lquery = (lwork == -1);
    minmn = (m < n) ? m : n;

    /* Test the input arguments */
    *info = 0;
    if (!(wntua || wntus || wntuo || wntun)) {
        *info = -1;
    } else if (!(wntva || wntvs || wntvo || wntvn) || (wntvo && wntuo)) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < ((1 > m) ? 1 : m)) {
        *info = -6;
    } else if (ldu < 1 || (wntuas && ldu < m)) {
        *info = -9;
    } else if (ldvt < 1 || (wntva && ldvt < n) || (wntvs && ldvt < minmn)) {
        *info = -11;
    }

    /* Compute workspace requirements */
    if (*info == 0) {
        minwrk = 1;
        maxwrk = 1;

        if (m >= n && minmn > 0) {
            /* Space needed for SBDSQR */
            mnthr = lapack_get_mnthr("GESVD", m, n);
            bdspac = 5 * n;

            /* Query workspace for subroutines - pass NULL for unused arrays */
            sgeqrf(m, n, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dgeqrf = (int)dum[0];
            sorgqr(m, n, n, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dorgqr_n = (int)dum[0];
            sorgqr(m, m, n, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dorgqr_m = (int)dum[0];
            sgebrd(n, n, NULL, lda, NULL, NULL, NULL, NULL, dum, -1, &ierr);
            lwork_dgebrd = (int)dum[0];
            sorgbr("P", n, n, n, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dorgbr_p = (int)dum[0];
            sorgbr("Q", n, n, n, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dorgbr_q = (int)dum[0];

            if (m >= mnthr) {
                if (wntun) {
                    /* Path 1 (M much larger than N, JOBU='N') */
                    maxwrk = n + lwork_dgeqrf;
                    maxwrk = (maxwrk > 3*n + lwork_dgebrd) ? maxwrk : 3*n + lwork_dgebrd;
                    if (wntvo || wntvas) {
                        maxwrk = (maxwrk > 3*n + lwork_dorgbr_p) ? maxwrk : 3*n + lwork_dorgbr_p;
                    }
                    maxwrk = (maxwrk > bdspac) ? maxwrk : bdspac;
                    minwrk = (4*n > bdspac) ? 4*n : bdspac;
                } else if (wntuo && wntvn) {
                    /* Path 2 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_n) ? wrkbl : n + lwork_dorgqr_n;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = (n*n + wrkbl > n*n + m*n + n) ? n*n + wrkbl : n*n + m*n + n;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntuo && wntvas) {
                    /* Path 3 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_n) ? wrkbl : n + lwork_dorgqr_n;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_p) ? wrkbl : 3*n + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = (n*n + wrkbl > n*n + m*n + n) ? n*n + wrkbl : n*n + m*n + n;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntus && wntvn) {
                    /* Path 4 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_n) ? wrkbl : n + lwork_dorgqr_n;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = n*n + wrkbl;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntus && wntvo) {
                    /* Path 5 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_n) ? wrkbl : n + lwork_dorgqr_n;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_p) ? wrkbl : 3*n + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = 2*n*n + wrkbl;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntus && wntvas) {
                    /* Path 6 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_n) ? wrkbl : n + lwork_dorgqr_n;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_p) ? wrkbl : 3*n + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = n*n + wrkbl;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntua && wntvn) {
                    /* Path 7 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_m) ? wrkbl : n + lwork_dorgqr_m;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = n*n + wrkbl;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntua && wntvo) {
                    /* Path 8 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_m) ? wrkbl : n + lwork_dorgqr_m;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_p) ? wrkbl : 3*n + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = 2*n*n + wrkbl;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                } else if (wntua && wntvas) {
                    /* Path 9 */
                    wrkbl = n + lwork_dgeqrf;
                    wrkbl = (wrkbl > n + lwork_dorgqr_m) ? wrkbl : n + lwork_dorgqr_m;
                    wrkbl = (wrkbl > 3*n + lwork_dgebrd) ? wrkbl : 3*n + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_q) ? wrkbl : 3*n + lwork_dorgbr_q;
                    wrkbl = (wrkbl > 3*n + lwork_dorgbr_p) ? wrkbl : 3*n + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = n*n + wrkbl;
                    minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
                }
            } else {
                /* Path 10 (M at least N, but not much larger) */
                sgebrd(m, n, NULL, lda, NULL, NULL, NULL, NULL, dum, -1, &ierr);
                lwork_dgebrd = (int)dum[0];
                maxwrk = 3*n + lwork_dgebrd;
                if (wntus || wntuo) {
                    sorgbr("Q", m, n, n, NULL, lda, NULL, dum, -1, &ierr);
                    lwork_dorgbr_q = (int)dum[0];
                    maxwrk = (maxwrk > 3*n + lwork_dorgbr_q) ? maxwrk : 3*n + lwork_dorgbr_q;
                }
                if (wntua) {
                    sorgbr("Q", m, m, n, NULL, lda, NULL, dum, -1, &ierr);
                    lwork_dorgbr_q = (int)dum[0];
                    maxwrk = (maxwrk > 3*n + lwork_dorgbr_q) ? maxwrk : 3*n + lwork_dorgbr_q;
                }
                if (!wntvn) {
                    maxwrk = (maxwrk > 3*n + lwork_dorgbr_p) ? maxwrk : 3*n + lwork_dorgbr_p;
                }
                maxwrk = (maxwrk > bdspac) ? maxwrk : bdspac;
                minwrk = (3*n + m > bdspac) ? 3*n + m : bdspac;
            }
        } else if (minmn > 0) {
            /* M < N case */
            mnthr = lapack_get_mnthr("GESVD", m, n);
            bdspac = 5 * m;

            /* Query workspace for subroutines - pass NULL for unused arrays */
            sgelqf(m, n, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dgelqf = (int)dum[0];
            sorglq(n, n, m, NULL, n, NULL, dum, -1, &ierr);
            lwork_dorglq_n = (int)dum[0];
            sorglq(m, n, m, NULL, lda, NULL, dum, -1, &ierr);
            lwork_dorglq_m = (int)dum[0];
            sgebrd(m, m, NULL, lda, NULL, NULL, NULL, NULL, dum, -1, &ierr);
            lwork_dgebrd = (int)dum[0];
            sorgbr("P", m, m, m, NULL, n, NULL, dum, -1, &ierr);
            lwork_dorgbr_p = (int)dum[0];
            sorgbr("Q", m, m, m, NULL, n, NULL, dum, -1, &ierr);
            lwork_dorgbr_q = (int)dum[0];

            if (n >= mnthr) {
                if (wntvn) {
                    /* Path 1t */
                    maxwrk = m + lwork_dgelqf;
                    maxwrk = (maxwrk > 3*m + lwork_dgebrd) ? maxwrk : 3*m + lwork_dgebrd;
                    if (wntuo || wntuas) {
                        maxwrk = (maxwrk > 3*m + lwork_dorgbr_q) ? maxwrk : 3*m + lwork_dorgbr_q;
                    }
                    maxwrk = (maxwrk > bdspac) ? maxwrk : bdspac;
                    minwrk = (4*m > bdspac) ? 4*m : bdspac;
                } else if (wntvo && wntun) {
                    /* Path 2t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_m) ? wrkbl : m + lwork_dorglq_m;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = (m*m + wrkbl > m*m + m*n + m) ? m*m + wrkbl : m*m + m*n + m;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntvo && wntuas) {
                    /* Path 3t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_m) ? wrkbl : m + lwork_dorglq_m;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_q) ? wrkbl : 3*m + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = (m*m + wrkbl > m*m + m*n + m) ? m*m + wrkbl : m*m + m*n + m;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntvs && wntun) {
                    /* Path 4t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_m) ? wrkbl : m + lwork_dorglq_m;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = m*m + wrkbl;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntvs && wntuo) {
                    /* Path 5t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_m) ? wrkbl : m + lwork_dorglq_m;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_q) ? wrkbl : 3*m + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = 2*m*m + wrkbl;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntvs && wntuas) {
                    /* Path 6t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_m) ? wrkbl : m + lwork_dorglq_m;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_q) ? wrkbl : 3*m + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = m*m + wrkbl;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntva && wntun) {
                    /* Path 7t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_n) ? wrkbl : m + lwork_dorglq_n;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = m*m + wrkbl;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntva && wntuo) {
                    /* Path 8t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_n) ? wrkbl : m + lwork_dorglq_n;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_q) ? wrkbl : 3*m + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = 2*m*m + wrkbl;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                } else if (wntva && wntuas) {
                    /* Path 9t */
                    wrkbl = m + lwork_dgelqf;
                    wrkbl = (wrkbl > m + lwork_dorglq_n) ? wrkbl : m + lwork_dorglq_n;
                    wrkbl = (wrkbl > 3*m + lwork_dgebrd) ? wrkbl : 3*m + lwork_dgebrd;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_p) ? wrkbl : 3*m + lwork_dorgbr_p;
                    wrkbl = (wrkbl > 3*m + lwork_dorgbr_q) ? wrkbl : 3*m + lwork_dorgbr_q;
                    wrkbl = (wrkbl > bdspac) ? wrkbl : bdspac;
                    maxwrk = m*m + wrkbl;
                    minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
                }
            } else {
                /* Path 10t (N greater than M, but not much larger) */
                sgebrd(m, n, NULL, lda, NULL, NULL, NULL, NULL, dum, -1, &ierr);
                lwork_dgebrd = (int)dum[0];
                maxwrk = 3*m + lwork_dgebrd;
                if (wntvs || wntvo) {
                    sorgbr("P", m, n, m, NULL, n, NULL, dum, -1, &ierr);
                    lwork_dorgbr_p = (int)dum[0];
                    maxwrk = (maxwrk > 3*m + lwork_dorgbr_p) ? maxwrk : 3*m + lwork_dorgbr_p;
                }
                if (wntva) {
                    sorgbr("P", n, n, m, NULL, n, NULL, dum, -1, &ierr);
                    lwork_dorgbr_p = (int)dum[0];
                    maxwrk = (maxwrk > 3*m + lwork_dorgbr_p) ? maxwrk : 3*m + lwork_dorgbr_p;
                }
                if (!wntun) {
                    maxwrk = (maxwrk > 3*m + lwork_dorgbr_q) ? maxwrk : 3*m + lwork_dorgbr_q;
                }
                maxwrk = (maxwrk > bdspac) ? maxwrk : bdspac;
                minwrk = (3*m + n > bdspac) ? 3*m + n : bdspac;
            }
        }
        maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
        work[0] = (f32)maxwrk;

        if (lwork < minwrk && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("SGESVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Get machine constants */
    eps = slamch("P");
    smlnum = sqrtf(slamch("S")) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = slange("M", m, n, A, lda, dum);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        iscl = 1;
        slascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        slascl("G", 0, 0, anrm, bignum, m, n, A, lda, &ierr);
    }

    mnthr = lapack_get_mnthr("GESVD", m, n);

    if (m >= n) {
        /*
         * A has at least as many rows as columns. If A has sufficiently
         * more rows than columns, first reduce using the QR decomposition
         */
        if (m >= mnthr) {
            /* Paths 1-9: M much larger than N */
            if (wntun) {
                /* Path 1: No left singular vectors */
                itau = 0;
                iwork = itau + n;

                /* Compute A=Q*R */
                sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                /* Zero out below R */
                if (n > 1) {
                    slaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
                }
                ie = 0;
                itauq = ie + n;
                itaup = itauq + n;
                iwork = itaup + n;

                /* Bidiagonalize R in A */
                sgebrd(n, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                       &work[iwork], lwork - iwork, &ierr);
                ncvt = 0;
                if (wntvo || wntvas) {
                    /* Generate P' */
                    sorgbr("P", n, n, n, A, lda, &work[itaup], &work[iwork],
                           lwork - iwork, &ierr);
                    ncvt = n;
                }
                iwork = ie + n;

                /* Perform bidiagonal QR iteration */
                sbdsqr("U", n, ncvt, 0, 0, S, &work[ie], A, lda,
                       NULL, 1, NULL, 1, &work[iwork], info);

                /* Copy to VT if desired */
                if (wntvas) {
                    slacpy("F", n, n, A, lda, VT, ldvt);
                }
            } else if (wntuo && wntvn) {
                /* Path 2: U overwritten on A, no V^T */
                if (lwork >= n*n + ((4*n > bdspac) ? 4*n : bdspac)) {
                    /* Fast algorithm with workspace */
                    ir = 0;
                    ldwrkr = (lwork >= wrkbl + lda*n) ? lda : n;
                    itau = ir + ldwrkr * n;
                    iwork = itau + n;

                    sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    slacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                    slaset("L", n - 1, n - 1, ZERO, ZERO, &work[ir + 1], ldwrkr);
                    sorgqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = itau;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    sgebrd(n, n, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    sorgbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + n;

                    sbdsqr("U", n, 0, n, 0, S, &work[ie], NULL, 1,
                           &work[ir], ldwrkr, NULL, 1, &work[iwork], info);

                    iu = ie + n;
                    ldwrku = n;

                    for (i = 0; i < m; i += ldwrku) {
                        chunk = ((m - i) < ldwrku) ? (m - i) : ldwrku;
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    chunk, n, n, ONE, &A[i], lda,
                                    &work[ir], ldwrkr, ZERO, &work[iu], ldwrku);
                        slacpy("F", chunk, n, &work[iu], ldwrku, &A[i], lda);
                    }
                } else {
                    /* Slow algorithm without extra workspace */
                    ie = 0;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    sgebrd(m, n, A, lda, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    sorgbr("Q", m, n, n, A, lda, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + n;

                    sbdsqr("U", n, 0, m, 0, S, &work[ie], NULL, 1,
                           A, lda, NULL, 1, &work[iwork], info);
                }
            } else if (wntuo && wntvas) {
                /* Path 3: U overwritten on A, V^T in VT */
                if (lwork >= n*n + ((4*n > bdspac) ? 4*n : bdspac)) {
                    ir = 0;
                    ldwrkr = (lwork >= wrkbl + lda*n) ? lda : n;
                    itau = ir + ldwrkr * n;
                    iwork = itau + n;

                    sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    slacpy("U", n, n, A, lda, VT, ldvt);
                    if (n > 1) {
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &VT[1], ldvt);
                    }
                    sorgqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = itau;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    sgebrd(n, n, VT, ldvt, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    slacpy("L", n, n, VT, ldvt, &work[ir], ldwrkr);
                    sorgbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    sorgbr("P", n, n, n, VT, ldvt, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + n;

                    sbdsqr("U", n, n, n, 0, S, &work[ie], VT, ldvt,
                           &work[ir], ldwrkr, dum, 1, &work[iwork], info);

                    iu = ie + n;
                    ldwrku = n;

                    for (i = 0; i < m; i += ldwrku) {
                        chunk = ((m - i) < ldwrku) ? (m - i) : ldwrku;
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    chunk, n, n, ONE, &A[i], lda,
                                    &work[ir], ldwrkr, ZERO, &work[iu], ldwrku);
                        slacpy("F", chunk, n, &work[iu], ldwrku, &A[i], lda);
                    }
                } else {
                    itau = 0;
                    iwork = itau + n;

                    sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    slacpy("U", n, n, A, lda, VT, ldvt);
                    if (n > 1) {
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &VT[1], ldvt);
                    }
                    sorgqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = itau;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    sgebrd(n, n, VT, ldvt, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    sormbr("Q", "R", "N", m, n, n, VT, ldvt, &work[itauq],
                           A, lda, &work[iwork], lwork - iwork, &ierr);
                    sorgbr("P", n, n, n, VT, ldvt, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + n;

                    sbdsqr("U", n, n, m, 0, S, &work[ie], VT, ldvt,
                           A, lda, dum, 1, &work[iwork], info);
                }
            } else if (wntus) {
                /* Paths 4, 5, 6: U in U array */
                if (wntvn) {
                    /* Path 4: U in U, no V^T */
                    if (lwork >= n*n + ((4*n > bdspac) ? 4*n : bdspac)) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &work[ir + 1], ldwrkr);
                        sorgqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        sgebrd(n, n, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, 0, n, 0, S, &work[ie], NULL, 1,
                               &work[ir], ldwrkr, NULL, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, ONE, A, lda, &work[ir], ldwrkr, ZERO, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);
                        sorgqr(m, n, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            slaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
                        }
                        sgebrd(n, n, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, 0, m, 0, S, &work[ie], NULL, 1,
                               U, ldu, NULL, 1, &work[iwork], info);
                    }
                } else if (wntvo) {
                    /* Path 5: U in U, V^T overwritten on A */
                    if (lwork >= 2*n*n + ((4*n > bdspac) ? 4*n : bdspac)) {
                        iu = 0;
                        if (lwork >= wrkbl + 2*lda*n) {
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = lda;
                        } else if (lwork >= wrkbl + (lda + n)*n) {
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        } else {
                            ldwrku = n;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        }
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &work[iu + 1], ldwrku);
                        sorgqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        sgebrd(n, n, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, &work[iu], ldwrku, &work[ir], ldwrkr);
                        sorgbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", n, n, n, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, n, n, 0, S, &work[ie], &work[ir], ldwrkr,
                               &work[iu], ldwrku, dum, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, ONE, A, lda, &work[iu], ldwrku, ZERO, U, ldu);
                        slacpy("F", n, n, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);
                        sorgqr(m, n, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            slaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
                        }
                        sgebrd(n, n, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", n, n, n, A, lda, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, n, m, 0, S, &work[ie], A, lda,
                               U, ldu, dum, 1, &work[iwork], info);
                    }
                } else if (wntvas) {
                    /* Path 6: U in U, V^T in VT */
                    if (lwork >= n*n + ((4*n > bdspac) ? 4*n : bdspac)) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = iu + ldwrku * n;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &work[iu + 1], ldwrku);
                        sorgqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        sgebrd(n, n, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, &work[iu], ldwrku, VT, ldvt);
                        sorgbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, n, n, 0, S, &work[ie], VT, ldvt,
                               &work[iu], ldwrku, dum, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, ONE, A, lda, &work[iu], ldwrku, ZERO, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);
                        sorgqr(m, n, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        slacpy("U", n, n, A, lda, VT, ldvt);
                        if (n > 1) {
                            slaset("L", n - 1, n - 1, ZERO, ZERO, &VT[1], ldvt);
                        }

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        sgebrd(n, n, VT, ldvt, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("Q", "R", "N", m, n, n, VT, ldvt, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, n, m, 0, S, &work[ie], VT, ldvt,
                               U, ldu, dum, 1, &work[iwork], info);
                    }
                }
            } else if (wntua) {
                /* Paths 7, 8, 9: Full U */
                if (wntvn) {
                    /* Path 7: Full U, no V^T */
                    if (lwork >= n*n + MAX3(n + m, 4*n, bdspac)) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;

                        /* Compute A=Q*R, copying result to U */
                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);

                        /* Copy R to WORK(IR), zeroing out below it */
                        slacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &work[ir + 1], ldwrkr);

                        /* Generate Q in U */
                        sorgqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        /* Bidiagonalize R in WORK(IR) */
                        sgebrd(n, n, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);

                        /* Generate left bidiagonalizing vectors in WORK(IR) */
                        sorgbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        /* Perform bidiagonal QR iteration */
                        sbdsqr("U", n, 0, n, 0, S, &work[ie], NULL, 1,
                               &work[ir], ldwrkr, NULL, 1, &work[iwork], info);

                        /* Multiply Q in U by left singular vectors of R in WORK(IR), storing result in A */
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, ONE, U, ldu, &work[ir], ldwrkr, ZERO, A, lda);

                        /* Copy left singular vectors of A from A to U */
                        slacpy("F", m, n, A, lda, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);
                        sorgqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            slaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
                        }
                        sgebrd(n, n, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, 0, m, 0, S, &work[ie], NULL, 1,
                               U, ldu, NULL, 1, &work[iwork], info);
                    }
                } else if (wntvo) {
                    /* Path 8: Full U, V^T overwritten on A */
                    if (lwork >= 2*n*n + MAX3(n + m, 4*n, bdspac)) {
                        iu = 0;
                        if (lwork >= wrkbl + 2*lda*n) {
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = lda;
                        } else if (lwork >= wrkbl + (lda + n)*n) {
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        } else {
                            ldwrku = n;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        }
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;

                        /* Compute A=Q*R, copying result to U */
                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);

                        /* Generate Q in U */
                        sorgqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        /* Copy R to WORK(IU), zeroing out below it */
                        slacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &work[iu + 1], ldwrku);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        /* Bidiagonalize R in WORK(IU), copying result to WORK(IR) */
                        sgebrd(n, n, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, &work[iu], ldwrku, &work[ir], ldwrkr);

                        /* Generate left bidiagonalizing vectors in WORK(IU) */
                        sorgbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);

                        /* Generate right bidiagonalizing vectors in WORK(IR) */
                        sorgbr("P", n, n, n, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        /* Perform bidiagonal QR iteration */
                        sbdsqr("U", n, n, n, 0, S, &work[ie], &work[ir], ldwrkr,
                               &work[iu], ldwrku, dum, 1, &work[iwork], info);

                        /* Multiply Q in U by left singular vectors of R in WORK(IU), storing result in A */
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, ONE, U, ldu, &work[iu], ldwrku, ZERO, A, lda);

                        /* Copy left singular vectors of A from A to U */
                        slacpy("F", m, n, A, lda, U, ldu);

                        /* Copy right singular vectors of R from WORK(IR) to A */
                        slacpy("F", n, n, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);
                        sorgqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            slaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
                        }
                        sgebrd(n, n, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", n, n, n, A, lda, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, n, m, 0, S, &work[ie], A, lda,
                               U, ldu, dum, 1, &work[iwork], info);
                    }
                } else if (wntvas) {
                    /* Path 9: Full U, V^T in VT */
                    if (lwork >= n*n + MAX3(n + m, 4*n, bdspac)) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = iu + ldwrku * n;
                        iwork = itau + n;

                        /* Compute A=Q*R, copying result to U */
                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);

                        /* Generate Q in U */
                        sorgqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        /* Copy R to WORK(IU), zeroing out below it */
                        slacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        slaset("L", n - 1, n - 1, ZERO, ZERO, &work[iu + 1], ldwrku);

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        /* Bidiagonalize R in WORK(IU), copying result to VT */
                        sgebrd(n, n, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", n, n, &work[iu], ldwrku, VT, ldvt);

                        /* Generate left bidiagonalizing vectors in WORK(IU) */
                        sorgbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);

                        /* Generate right bidiagonalizing vectors in VT */
                        sorgbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        /* Perform bidiagonal QR iteration */
                        sbdsqr("U", n, n, n, 0, S, &work[ie], VT, ldvt,
                               &work[iu], ldwrku, dum, 1, &work[iwork], info);

                        /* Multiply Q in U by left singular vectors of R in WORK(IU), storing result in A */
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, ONE, U, ldu, &work[iu], ldwrku, ZERO, A, lda);

                        /* Copy left singular vectors of A from A to U */
                        slacpy("F", m, n, A, lda, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        sgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, n, A, lda, U, ldu);
                        sorgqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        slacpy("U", n, n, A, lda, VT, ldvt);
                        if (n > 1) {
                            slaset("L", n - 1, n - 1, ZERO, ZERO, &VT[1], ldvt);
                        }

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        sgebrd(n, n, VT, ldvt, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("Q", "R", "N", m, n, n, VT, ldvt, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + n;

                        sbdsqr("U", n, n, m, 0, S, &work[ie], VT, ldvt,
                               U, ldu, dum, 1, &work[iwork], info);
                    }
                }
            }
        } else {
            /* Path 10: M >= N but M < MNTHR - direct bidiagonalization */
            ie = 0;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;

            /* Bidiagonalize A */
            sgebrd(m, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                   &work[iwork], lwork - iwork, &ierr);

            if (wntuas) {
                /* Copy result to U and generate left bidiagonalizing vectors */
                slacpy("L", m, n, A, lda, U, ldu);
                ncu = wntus ? n : m;
                sorgbr("Q", m, ncu, n, U, ldu, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvas) {
                /* Copy result to VT and generate right bidiagonalizing vectors */
                slacpy("U", n, n, A, lda, VT, ldvt);
                sorgbr("P", n, n, n, VT, ldvt, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntuo) {
                /* Generate left bidiagonalizing vectors in A */
                sorgbr("Q", m, n, n, A, lda, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvo) {
                /* Generate right bidiagonalizing vectors in A */
                sorgbr("P", n, n, n, A, lda, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }

            iwork = ie + n;
            nru = (wntuas || wntuo) ? m : 0;
            ncvt = (wntvas || wntvo) ? n : 0;

            if (!wntuo && !wntvo) {
                /* Compute SVD of bidiagonal matrix */
                sbdsqr("U", n, ncvt, nru, 0, S, &work[ie], VT, ldvt,
                       U, ldu, dum, 1, &work[iwork], info);
            } else if (!wntuo && wntvo) {
                sbdsqr("U", n, ncvt, nru, 0, S, &work[ie], A, lda,
                       U, ldu, dum, 1, &work[iwork], info);
            } else {
                sbdsqr("U", n, ncvt, nru, 0, S, &work[ie], VT, ldvt,
                       A, lda, dum, 1, &work[iwork], info);
            }
        }
    } else {
        /*
         * A has more columns than rows. If A has sufficiently more
         * columns than rows, first reduce using the LQ decomposition
         */
        if (n >= mnthr) {
            /* Paths 1t-9t: N much larger than M */
            if (wntvn) {
                /* Path 1t: No right singular vectors */
                itau = 0;
                iwork = itau + m;

                /* Compute A=L*Q */
                sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                /* Zero out above L */
                if (m > 1) {
                    slaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);
                }
                ie = 0;
                itauq = ie + m;
                itaup = itauq + m;
                iwork = itaup + m;

                /* Bidiagonalize L in A */
                sgebrd(m, m, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                       &work[iwork], lwork - iwork, &ierr);

                if (wntuo || wntuas) {
                    /* Generate Q */
                    sorgbr("Q", m, m, m, A, lda, &work[itauq], &work[iwork],
                           lwork - iwork, &ierr);
                }
                iwork = ie + m;
                nru = (wntuo || wntuas) ? m : 0;

                /* Perform bidiagonal QR iteration */
                sbdsqr("U", m, 0, nru, 0, S, &work[ie], NULL, 1,
                       A, lda, NULL, 1, &work[iwork], info);

                /* Copy to U if desired */
                if (wntuas) {
                    slacpy("F", m, m, A, lda, U, ldu);
                }
            } else if (wntvo && wntun) {
                /* Path 2t: V^T overwritten on A, no U */
                if (lwork >= m*m + ((4*m > bdspac) ? 4*m : bdspac)) {
                    ir = 0;
                    ldwrkr = (lwork >= wrkbl + lda*m) ? lda : m;
                    itau = ir + ldwrkr * m;
                    iwork = itau + m;

                    sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    slacpy("L", m, m, A, lda, &work[ir], ldwrkr);
                    slaset("U", m - 1, m - 1, ZERO, ZERO, &work[ir + ldwrkr], ldwrkr);
                    sorglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    sgebrd(m, m, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    sorgbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + m;

                    sbdsqr("U", m, m, 0, 0, S, &work[ie], &work[ir], ldwrkr,
                           NULL, 1, NULL, 1, &work[iwork], info);

                    iu = ie + m;
                    ldwrku = m;

                    for (i = 0; i < n; i += ldwrku) {
                        chunk = ((n - i) < ldwrku) ? (n - i) : ldwrku;
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, chunk, m, ONE, &work[ir], ldwrkr,
                                    &A[i * lda], lda, ZERO, &work[iu], m);
                        slacpy("F", m, chunk, &work[iu], m, &A[i * lda], lda);
                    }
                } else {
                    ie = 0;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    sgebrd(m, n, A, lda, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    sorgbr("P", m, n, m, A, lda, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + m;

                    sbdsqr("L", m, n, 0, 0, S, &work[ie], A, lda,
                           NULL, 1, NULL, 1, &work[iwork], info);
                }
            } else if (wntvo && wntuas) {
                /* Path 3t: V^T overwritten on A, U in U */
                if (lwork >= m*m + ((4*m > bdspac) ? 4*m : bdspac)) {
                    ir = 0;
                    ldwrkr = (lwork >= wrkbl + lda*m) ? lda : m;
                    itau = ir + ldwrkr * m;
                    iwork = itau + m;

                    sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    slacpy("L", m, m, A, lda, U, ldu);
                    if (m > 1) {
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &U[ldu], ldu);
                    }
                    sorglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    sgebrd(m, m, U, ldu, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    slacpy("U", m, m, U, ldu, &work[ir], ldwrkr);
                    sorgbr("Q", m, m, m, U, ldu, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    sorgbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + m;

                    sbdsqr("U", m, m, m, 0, S, &work[ie], &work[ir], ldwrkr,
                           U, ldu, dum, 1, &work[iwork], info);

                    iu = ie + m;
                    ldwrku = m;

                    for (i = 0; i < n; i += ldwrku) {
                        chunk = ((n - i) < ldwrku) ? (n - i) : ldwrku;
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, chunk, m, ONE, &work[ir], ldwrkr,
                                    &A[i * lda], lda, ZERO, &work[iu], m);
                        slacpy("F", m, chunk, &work[iu], m, &A[i * lda], lda);
                    }
                } else {
                    itau = 0;
                    iwork = itau + m;

                    sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    slacpy("L", m, m, A, lda, U, ldu);
                    if (m > 1) {
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &U[ldu], ldu);
                    }
                    sorglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    sgebrd(m, m, U, ldu, S, &work[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    sormbr("P", "L", "T", m, n, m, U, ldu, &work[itaup],
                           A, lda, &work[iwork], lwork - iwork, &ierr);
                    sorgbr("Q", m, m, m, U, ldu, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    iwork = ie + m;

                    sbdsqr("U", m, n, m, 0, S, &work[ie], A, lda,
                           U, ldu, dum, 1, &work[iwork], info);
                }
            } else if (wntvs) {
                /* Paths 4t, 5t, 6t: V^T in VT */
                if (wntun) {
                    /* Path 4t: V^T in VT, no U */
                    if (lwork >= m*m + ((4*m > bdspac) ? 4*m : bdspac)) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, A, lda, &work[ir], ldwrkr);
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &work[ir + ldwrkr], ldwrkr);
                        sorglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, m, 0, 0, S, &work[ie], &work[ir], ldwrkr,
                               NULL, 1, NULL, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, ONE, &work[ir], ldwrkr, A, lda, ZERO, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        sorglq(m, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        if (m > 1) {
                            slaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);
                        }
                        sgebrd(m, m, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("P", "L", "T", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, n, 0, 0, S, &work[ie], VT, ldvt,
                               NULL, 1, NULL, 1, &work[iwork], info);
                    }
                } else if (wntuo) {
                    /* Path 5t: V^T in VT, U overwritten on A */
                    if (lwork >= 2*m*m + ((4*m > bdspac) ? 4*m : bdspac)) {
                        iu = 0;
                        if (lwork >= wrkbl + 2*lda*m) {
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = lda;
                        } else if (lwork >= wrkbl + (lda + m)*m) {
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        } else {
                            ldwrku = m;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        }
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &work[iu + ldwrku], ldwrku);
                        sorglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, &work[iu], ldwrku, &work[ir], ldwrkr);
                        sorgbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, m, m, 0, S, &work[ie], &work[iu], ldwrku,
                               &work[ir], ldwrkr, dum, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, ONE, &work[iu], ldwrku, A, lda, ZERO, VT, ldvt);
                        slacpy("F", m, m, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        sorglq(m, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        if (m > 1) {
                            slaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);
                        }
                        sgebrd(m, m, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("P", "L", "T", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, A, lda, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, n, m, 0, S, &work[ie], VT, ldvt,
                               A, lda, dum, 1, &work[iwork], info);
                    }
                } else if (wntuas) {
                    /* Path 6t: V^T in VT, U in U */
                    if (lwork >= m*m + ((4*m > bdspac) ? 4*m : bdspac)) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = iu + ldwrku * m;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &work[iu + ldwrku], ldwrku);
                        sorglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, &work[iu], ldwrku, U, ldu);
                        sorgbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, m, m, 0, S, &work[ie], &work[iu], ldwrku,
                               U, ldu, dum, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, ONE, &work[iu], ldwrku, A, lda, ZERO, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        sorglq(m, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        slacpy("L", m, m, A, lda, U, ldu);
                        if (m > 1) {
                            slaset("U", m - 1, m - 1, ZERO, ZERO, &U[ldu], ldu);
                        }

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, U, ldu, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("P", "L", "T", m, n, m, U, ldu, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, n, m, 0, S, &work[ie], VT, ldvt,
                               U, ldu, dum, 1, &work[iwork], info);
                    }
                }
            } else if (wntva) {
                /* Paths 7t, 8t, 9t: Full V^T */
                if (wntun) {
                    /* Path 7t: Full V^T, no U */
                    if (lwork >= m*m + MAX3(n + m, 4*m, bdspac)) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        slacpy("L", m, m, A, lda, &work[ir], ldwrkr);
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &work[ir + ldwrkr], ldwrkr);
                        sorglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sorgbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, m, 0, 0, S, &work[ie], &work[ir], ldwrkr,
                               NULL, 1, NULL, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, ONE, &work[ir], ldwrkr, VT, ldvt, ZERO, A, lda);
                        slacpy("F", m, n, A, lda, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        sorglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        if (m > 1) {
                            slaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);
                        }
                        sgebrd(m, m, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("P", "L", "T", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, n, 0, 0, S, &work[ie], VT, ldvt,
                               NULL, 1, NULL, 1, &work[iwork], info);
                    }
                } else if (wntuo) {
                    /* Path 8t: Full V^T, U overwritten on A */
                    if (lwork >= 2*m*m + MAX3(n + m, 4*m, bdspac)) {
                        iu = 0;
                        if (lwork >= wrkbl + 2*lda*m) {
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = lda;
                        } else if (lwork >= wrkbl + (lda + m)*m) {
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        } else {
                            ldwrku = m;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        }
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        slacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &work[iu + ldwrku], ldwrku);
                        sorglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, &work[iu], ldwrku, &work[ir], ldwrkr);
                        sorgbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, m, m, 0, S, &work[ie], &work[iu], ldwrku,
                               &work[ir], ldwrkr, dum, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, ONE, &work[iu], ldwrku, VT, ldvt, ZERO, A, lda);
                        slacpy("F", m, n, A, lda, VT, ldvt);
                        slacpy("F", m, m, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        sorglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        if (m > 1) {
                            slaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);
                        }
                        sgebrd(m, m, A, lda, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("P", "L", "T", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, A, lda, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, n, m, 0, S, &work[ie], VT, ldvt,
                               A, lda, dum, 1, &work[iwork], info);
                    }
                } else if (wntuas) {
                    /* Path 9t: Full V^T, U in U */
                    if (lwork >= m*m + MAX3(n + m, 4*m, bdspac)) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = iu + ldwrku * m;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        slacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        slaset("U", m - 1, m - 1, ZERO, ZERO, &work[iu + ldwrku], ldwrku);
                        sorglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, &work[iu], ldwrku, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        slacpy("L", m, m, &work[iu], ldwrku, U, ldu);
                        sorgbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, m, m, 0, S, &work[ie], &work[iu], ldwrku,
                               U, ldu, dum, 1, &work[iwork], info);

                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, ONE, &work[iu], ldwrku, VT, ldvt, ZERO, A, lda);
                        slacpy("F", m, n, A, lda, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        sgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        slacpy("U", m, n, A, lda, VT, ldvt);
                        sorglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        slacpy("L", m, m, A, lda, U, ldu);
                        if (m > 1) {
                            slaset("U", m - 1, m - 1, ZERO, ZERO, &U[ldu], ldu);
                        }

                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        sgebrd(m, m, U, ldu, S, &work[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        sormbr("P", "L", "T", m, n, m, U, ldu, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        sorgbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        iwork = ie + m;

                        sbdsqr("U", m, n, m, 0, S, &work[ie], VT, ldvt,
                               U, ldu, dum, 1, &work[iwork], info);
                    }
                }
            }
        } else {
            /* Path 10t: N > M but N < MNTHR - direct bidiagonalization */
            ie = 0;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;

            /* Bidiagonalize A */
            sgebrd(m, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                   &work[iwork], lwork - iwork, &ierr);

            if (wntuas) {
                /* Copy result to U and generate left bidiagonalizing vectors */
                slacpy("L", m, m, A, lda, U, ldu);
                sorgbr("Q", m, m, n, U, ldu, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvas) {
                /* Copy result to VT and generate right bidiagonalizing vectors */
                slacpy("U", m, n, A, lda, VT, ldvt);
                nrvt = wntva ? n : m;
                sorgbr("P", nrvt, n, m, VT, ldvt, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntuo) {
                /* Generate left bidiagonalizing vectors in A */
                sorgbr("Q", m, m, n, A, lda, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvo) {
                /* Generate right bidiagonalizing vectors in A */
                sorgbr("P", m, n, m, A, lda, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }

            iwork = ie + m;
            nru = (wntuas || wntuo) ? m : 0;
            ncvt = (wntvas || wntvo) ? n : 0;

            if (!wntuo && !wntvo) {
                /* Compute SVD of bidiagonal matrix */
                sbdsqr("L", m, ncvt, nru, 0, S, &work[ie], VT, ldvt,
                       U, ldu, dum, 1, &work[iwork], info);
            } else if (!wntuo && wntvo) {
                sbdsqr("L", m, ncvt, nru, 0, S, &work[ie], A, lda,
                       U, ldu, dum, 1, &work[iwork], info);
            } else {
                sbdsqr("L", m, ncvt, nru, 0, S, &work[ie], VT, ldvt,
                       A, lda, dum, 1, &work[iwork], info);
            }
        }
    }

    /* If SBDSQR failed to converge, copy unconverged superdiagonals to WORK(2:MINMN) */
    if (*info != 0) {
        if (ie > 1) {
            for (i = 0; i < minmn - 1; i++) {
                work[i + 1] = work[i + ie];
            }
        }
        if (ie < 1) {
            for (i = minmn - 2; i >= 0; i--) {
                work[i + 1] = work[i + ie];
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            slascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (*info != 0 && anrm > bignum) {
            slascl("G", 0, 0, bignum, anrm, minmn - 1, 1, &work[1], minmn, &ierr);
        }
        if (anrm < smlnum) {
            slascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (*info != 0 && anrm < smlnum) {
            slascl("G", 0, 0, smlnum, anrm, minmn - 1, 1, &work[1], minmn, &ierr);
        }
    }

    /* Return optimal workspace */
    work[0] = (f32)maxwrk;

    #undef MAX3
}
