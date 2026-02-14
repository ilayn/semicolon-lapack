/**
 * @file zgesvd.c
 * @brief ZGESVD computes the singular value decomposition (SVD) of a complex matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/*
 * ZGESVD computes the singular value decomposition (SVD) of a complex
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * conjugate-transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
 * V is an N-by-N unitary matrix. The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order. The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * Note that the routine returns V**H, not V.
 *
 * @param[in]     jobu   Specifies options for computing all or part of U:
 *                       = 'A': all M columns of U are returned in array U;
 *                       = 'S': the first min(m,n) columns of U are returned in U;
 *                       = 'O': the first min(m,n) columns of U are overwritten on A;
 *                       = 'N': no columns of U are computed.
 * @param[in]     jobvt  Specifies options for computing all or part of V**H:
 *                       = 'A': all N rows of V**H are returned in array VT;
 *                       = 'S': the first min(m,n) rows of V**H are returned in VT;
 *                       = 'O': the first min(m,n) rows of V**H are overwritten on A;
 *                       = 'N': no rows of V**H are computed.
 *                       JOBVT and JOBU cannot both be 'O'.
 * @param[in]     m      The number of rows of the input matrix A. m >= 0.
 * @param[in]     n      The number of columns of the input matrix A. n >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, contents depend on jobu and jobvt.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    S      Double precision array, dimension (min(m,n)).
 *                       The singular values of A, sorted so that S[i] >= S[i+1].
 * @param[out]    U      Complex*16 array, dimension (ldu, ucol).
 *                       If jobu = 'A', U contains the M-by-M unitary matrix U;
 *                       if jobu = 'S', U contains the first min(m,n) columns of U;
 *                       if jobu = 'N' or 'O', U is not referenced.
 * @param[in]     ldu    The leading dimension of the array U. ldu >= 1; if
 *                       jobu = 'S' or 'A', ldu >= m.
 * @param[out]    VT     Complex*16 array, dimension (ldvt, n).
 *                       If jobvt = 'A', VT contains the N-by-N unitary matrix V**H;
 *                       if jobvt = 'S', VT contains the first min(m,n) rows of V**H;
 *                       if jobvt = 'N' or 'O', VT is not referenced.
 * @param[in]     ldvt   The leading dimension of the array VT. ldvt >= 1; if
 *                       jobvt = 'A', ldvt >= n; if jobvt = 'S', ldvt >= min(m,n).
 * @param[out]    work   Complex*16 array, dimension (max(1,lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If lwork = -1, a workspace query is assumed.
 * @param[out]    rwork  Double precision array, dimension (5*min(m,n)).
 *                       On exit, if info > 0, rwork[0:min(m,n)-2] contains the
 *                       unconverged superdiagonal elements of an upper bidiagonal
 *                       matrix B whose diagonal is in S.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if ZBDSQR did not converge.
 */
void zgesvd(const char* jobu, const char* jobvt,
            const int m, const int n,
            c128* restrict A, const int lda,
            f64* restrict S,
            c128* restrict U, const int ldu,
            c128* restrict VT, const int ldvt,
            c128* restrict work, const int lwork,
            f64* restrict rwork, int* info)
{
    /* Constants */
    static const f64 ZERO = 0.0;
    static const f64 ONE = 1.0;
    static const c128 CZERO = CMPLX(0.0, 0.0);
    static const c128 CONE = CMPLX(1.0, 0.0);

    /* Helper macro for 3-way max (matches Fortran MAX(a,b,c)) */
    #define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

    /* Local variables */
    int wntua, wntus, wntuas, wntuo, wntun;
    int wntva, wntvs, wntvas, wntvo, wntvn;
    int lquery, minmn, mnthr;
    int minwrk, maxwrk, wrkbl = 0;
    int lwork_zgeqrf, lwork_zungqr_n, lwork_zungqr_m;
    int lwork_zgebrd, lwork_zungbr_p, lwork_zungbr_q;
    int lwork_zgelqf, lwork_zunglq_n, lwork_zunglq_m;
    int ie = 0, irwork, itau, itauq, itaup, iwork, ir, iu, chunk, blk;
    int i, ierr, iscl, ncu, ncvt, nru, nrvt;
    int ldwrkr, ldwrku;
    f64 anrm, bignum, eps, smlnum;
    f64 dum[1];
    c128 cdum[1];

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
            mnthr = lapack_get_mnthr("GESVD", m, n);

            /* Query workspace for subroutines */
            zgeqrf(m, n, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zgeqrf = (int)creal(cdum[0]);
            zungqr(m, n, n, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zungqr_n = (int)creal(cdum[0]);
            zungqr(m, m, n, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zungqr_m = (int)creal(cdum[0]);
            zgebrd(n, n, NULL, lda, NULL, NULL, NULL, NULL, cdum, -1, &ierr);
            lwork_zgebrd = (int)creal(cdum[0]);
            zungbr("P", n, n, n, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zungbr_p = (int)creal(cdum[0]);
            zungbr("Q", n, n, n, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zungbr_q = (int)creal(cdum[0]);

            if (m >= mnthr) {
                if (wntun) {
                    /* Path 1 (M much larger than N, JOBU='N') */
                    maxwrk = n + lwork_zgeqrf;
                    maxwrk = (maxwrk > 2*n + lwork_zgebrd) ? maxwrk : 2*n + lwork_zgebrd;
                    if (wntvo || wntvas) {
                        maxwrk = (maxwrk > 2*n + lwork_zungbr_p) ? maxwrk : 2*n + lwork_zungbr_p;
                    }
                    minwrk = 3*n;
                } else if (wntuo && wntvn) {
                    /* Path 2 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_n) ? wrkbl : n + lwork_zungqr_n;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    maxwrk = (n*n + wrkbl > n*n + m*n) ? n*n + wrkbl : n*n + m*n;
                    minwrk = 2*n + m;
                } else if (wntuo && wntvas) {
                    /* Path 3 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_n) ? wrkbl : n + lwork_zungqr_n;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_p) ? wrkbl : 2*n + lwork_zungbr_p;
                    maxwrk = (n*n + wrkbl > n*n + m*n) ? n*n + wrkbl : n*n + m*n;
                    minwrk = 2*n + m;
                } else if (wntus && wntvn) {
                    /* Path 4 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_n) ? wrkbl : n + lwork_zungqr_n;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    maxwrk = n*n + wrkbl;
                    minwrk = 2*n + m;
                } else if (wntus && wntvo) {
                    /* Path 5 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_n) ? wrkbl : n + lwork_zungqr_n;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_p) ? wrkbl : 2*n + lwork_zungbr_p;
                    maxwrk = 2*n*n + wrkbl;
                    minwrk = 2*n + m;
                } else if (wntus && wntvas) {
                    /* Path 6 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_n) ? wrkbl : n + lwork_zungqr_n;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_p) ? wrkbl : 2*n + lwork_zungbr_p;
                    maxwrk = n*n + wrkbl;
                    minwrk = 2*n + m;
                } else if (wntua && wntvn) {
                    /* Path 7 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_m) ? wrkbl : n + lwork_zungqr_m;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    maxwrk = n*n + wrkbl;
                    minwrk = 2*n + m;
                } else if (wntua && wntvo) {
                    /* Path 8 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_m) ? wrkbl : n + lwork_zungqr_m;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_p) ? wrkbl : 2*n + lwork_zungbr_p;
                    maxwrk = 2*n*n + wrkbl;
                    minwrk = 2*n + m;
                } else if (wntua && wntvas) {
                    /* Path 9 */
                    wrkbl = n + lwork_zgeqrf;
                    wrkbl = (wrkbl > n + lwork_zungqr_m) ? wrkbl : n + lwork_zungqr_m;
                    wrkbl = (wrkbl > 2*n + lwork_zgebrd) ? wrkbl : 2*n + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_q) ? wrkbl : 2*n + lwork_zungbr_q;
                    wrkbl = (wrkbl > 2*n + lwork_zungbr_p) ? wrkbl : 2*n + lwork_zungbr_p;
                    maxwrk = n*n + wrkbl;
                    minwrk = 2*n + m;
                }
            } else {
                /* Path 10 (M at least N, but not much larger) */
                zgebrd(m, n, NULL, lda, NULL, NULL, NULL, NULL, cdum, -1, &ierr);
                lwork_zgebrd = (int)creal(cdum[0]);
                maxwrk = 2*n + lwork_zgebrd;
                if (wntus || wntuo) {
                    zungbr("Q", m, n, n, NULL, lda, NULL, cdum, -1, &ierr);
                    lwork_zungbr_q = (int)creal(cdum[0]);
                    maxwrk = (maxwrk > 2*n + lwork_zungbr_q) ? maxwrk : 2*n + lwork_zungbr_q;
                }
                if (wntua) {
                    zungbr("Q", m, m, n, NULL, lda, NULL, cdum, -1, &ierr);
                    lwork_zungbr_q = (int)creal(cdum[0]);
                    maxwrk = (maxwrk > 2*n + lwork_zungbr_q) ? maxwrk : 2*n + lwork_zungbr_q;
                }
                if (!wntvn) {
                    maxwrk = (maxwrk > 2*n + lwork_zungbr_p) ? maxwrk : 2*n + lwork_zungbr_p;
                }
                minwrk = 2*n + m;
            }
        } else if (minmn > 0) {
            /* M < N case */
            mnthr = lapack_get_mnthr("GESVD", m, n);

            /* Query workspace for subroutines */
            zgelqf(m, n, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zgelqf = (int)creal(cdum[0]);
            zunglq(n, n, m, NULL, n, NULL, cdum, -1, &ierr);
            lwork_zunglq_n = (int)creal(cdum[0]);
            zunglq(m, n, m, NULL, lda, NULL, cdum, -1, &ierr);
            lwork_zunglq_m = (int)creal(cdum[0]);
            zgebrd(m, m, NULL, lda, NULL, NULL, NULL, NULL, cdum, -1, &ierr);
            lwork_zgebrd = (int)creal(cdum[0]);
            zungbr("P", m, m, m, NULL, n, NULL, cdum, -1, &ierr);
            lwork_zungbr_p = (int)creal(cdum[0]);
            zungbr("Q", m, m, m, NULL, n, NULL, cdum, -1, &ierr);
            lwork_zungbr_q = (int)creal(cdum[0]);

            if (n >= mnthr) {
                if (wntvn) {
                    /* Path 1t */
                    maxwrk = m + lwork_zgelqf;
                    maxwrk = (maxwrk > 2*m + lwork_zgebrd) ? maxwrk : 2*m + lwork_zgebrd;
                    if (wntuo || wntuas) {
                        maxwrk = (maxwrk > 2*m + lwork_zungbr_q) ? maxwrk : 2*m + lwork_zungbr_q;
                    }
                    minwrk = 3*m;
                } else if (wntvo && wntun) {
                    /* Path 2t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_m) ? wrkbl : m + lwork_zunglq_m;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    maxwrk = (m*m + wrkbl > m*m + m*n) ? m*m + wrkbl : m*m + m*n;
                    minwrk = 2*m + n;
                } else if (wntvo && wntuas) {
                    /* Path 3t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_m) ? wrkbl : m + lwork_zunglq_m;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_q) ? wrkbl : 2*m + lwork_zungbr_q;
                    maxwrk = (m*m + wrkbl > m*m + m*n) ? m*m + wrkbl : m*m + m*n;
                    minwrk = 2*m + n;
                } else if (wntvs && wntun) {
                    /* Path 4t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_m) ? wrkbl : m + lwork_zunglq_m;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    maxwrk = m*m + wrkbl;
                    minwrk = 2*m + n;
                } else if (wntvs && wntuo) {
                    /* Path 5t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_m) ? wrkbl : m + lwork_zunglq_m;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_q) ? wrkbl : 2*m + lwork_zungbr_q;
                    maxwrk = 2*m*m + wrkbl;
                    minwrk = 2*m + n;
                } else if (wntvs && wntuas) {
                    /* Path 6t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_m) ? wrkbl : m + lwork_zunglq_m;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_q) ? wrkbl : 2*m + lwork_zungbr_q;
                    maxwrk = m*m + wrkbl;
                    minwrk = 2*m + n;
                } else if (wntva && wntun) {
                    /* Path 7t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_n) ? wrkbl : m + lwork_zunglq_n;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    maxwrk = m*m + wrkbl;
                    minwrk = 2*m + n;
                } else if (wntva && wntuo) {
                    /* Path 8t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_n) ? wrkbl : m + lwork_zunglq_n;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_q) ? wrkbl : 2*m + lwork_zungbr_q;
                    maxwrk = 2*m*m + wrkbl;
                    minwrk = 2*m + n;
                } else if (wntva && wntuas) {
                    /* Path 9t */
                    wrkbl = m + lwork_zgelqf;
                    wrkbl = (wrkbl > m + lwork_zunglq_n) ? wrkbl : m + lwork_zunglq_n;
                    wrkbl = (wrkbl > 2*m + lwork_zgebrd) ? wrkbl : 2*m + lwork_zgebrd;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_p) ? wrkbl : 2*m + lwork_zungbr_p;
                    wrkbl = (wrkbl > 2*m + lwork_zungbr_q) ? wrkbl : 2*m + lwork_zungbr_q;
                    maxwrk = m*m + wrkbl;
                    minwrk = 2*m + n;
                }
            } else {
                /* Path 10t (N greater than M, but not much larger) */
                zgebrd(m, n, NULL, lda, NULL, NULL, NULL, NULL, cdum, -1, &ierr);
                lwork_zgebrd = (int)creal(cdum[0]);
                maxwrk = 2*m + lwork_zgebrd;
                if (wntvs || wntvo) {
                    zungbr("P", m, n, m, NULL, n, NULL, cdum, -1, &ierr);
                    lwork_zungbr_p = (int)creal(cdum[0]);
                    maxwrk = (maxwrk > 2*m + lwork_zungbr_p) ? maxwrk : 2*m + lwork_zungbr_p;
                }
                if (wntva) {
                    zungbr("P", n, n, m, NULL, n, NULL, cdum, -1, &ierr);
                    lwork_zungbr_p = (int)creal(cdum[0]);
                    maxwrk = (maxwrk > 2*m + lwork_zungbr_p) ? maxwrk : 2*m + lwork_zungbr_p;
                }
                if (!wntun) {
                    maxwrk = (maxwrk > 2*m + lwork_zungbr_q) ? maxwrk : 2*m + lwork_zungbr_q;
                }
                minwrk = 2*m + n;
            }
        }
        maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
        work[0] = CMPLX((f64)maxwrk, 0.0);

        if (lwork < minwrk && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("ZGESVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");
    smlnum = sqrt(dlamch("S")) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = zlange("M", m, n, A, lda, dum);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        iscl = 1;
        zlascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        zlascl("G", 0, 0, anrm, bignum, m, n, A, lda, &ierr);
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

                zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                if (n > 1) {
                    zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                }
                ie = 0;
                itauq = 0;
                itaup = itauq + n;
                iwork = itaup + n;

                zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
                       &work[iwork], lwork - iwork, &ierr);
                ncvt = 0;
                if (wntvo || wntvas) {
                    zungbr("P", n, n, n, A, lda, &work[itaup], &work[iwork],
                           lwork - iwork, &ierr);
                    ncvt = n;
                }
                irwork = ie + n;

                zbdsqr("U", n, ncvt, 0, 0, S, &rwork[ie], A, lda,
                       NULL, 1, NULL, 1, &rwork[irwork], info);

                if (wntvas) {
                    zlacpy("F", n, n, A, lda, VT, ldvt);
                }
            } else if (wntuo && wntvn) {
                /* Path 2: U overwritten on A, no V^H */
                if (lwork >= n*n + 3*n) {
                    ir = 0;
                    if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + lda*n) {
                        ldwrku = lda;
                        ldwrkr = lda;
                    } else if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + n*n) {
                        ldwrku = lda;
                        ldwrkr = n;
                    } else {
                        ldwrku = (lwork - n*n) / n;
                        ldwrkr = n;
                    }
                    itau = ir + ldwrkr * n;
                    iwork = itau + n;

                    zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                    zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[ir + 1], ldwrkr);
                    zungqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = 0;
                    itauq = itau;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    zgebrd(n, n, &work[ir], ldwrkr, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zungbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + n;

                    zbdsqr("U", n, 0, n, 0, S, &rwork[ie], NULL, 1,
                           &work[ir], ldwrkr, NULL, 1, &rwork[irwork], info);
                    iu = itauq;

                    for (i = 0; i < m; i += ldwrku) {
                        chunk = ((m - i) < ldwrku) ? (m - i) : ldwrku;
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    chunk, n, n, &CONE, &A[i], lda,
                                    &work[ir], ldwrkr, &CZERO, &work[iu], ldwrku);
                        zlacpy("F", chunk, n, &work[iu], ldwrku, &A[i], lda);
                    }
                } else {
                    ie = 0;
                    itauq = 0;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zungbr("Q", m, n, n, A, lda, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + n;

                    zbdsqr("U", n, 0, m, 0, S, &rwork[ie], NULL, 1,
                           A, lda, NULL, 1, &rwork[irwork], info);
                }
            } else if (wntuo && wntvas) {
                /* Path 3: U overwritten on A, V^H in VT */
                if (lwork >= n*n + 3*n) {
                    ir = 0;
                    if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + lda*n) {
                        ldwrku = lda;
                        ldwrkr = lda;
                    } else if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + n*n) {
                        ldwrku = lda;
                        ldwrkr = n;
                    } else {
                        ldwrku = (lwork - n*n) / n;
                        ldwrkr = n;
                    }
                    itau = ir + ldwrkr * n;
                    iwork = itau + n;

                    zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("U", n, n, A, lda, VT, ldvt);
                    if (n > 1) {
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &VT[1], ldvt);
                    }
                    zungqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = 0;
                    itauq = itau;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    zgebrd(n, n, VT, ldvt, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("L", n, n, VT, ldvt, &work[ir], ldwrkr);
                    zungbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + n;

                    zbdsqr("U", n, n, n, 0, S, &rwork[ie], VT, ldvt,
                           &work[ir], ldwrkr, NULL, 1, &rwork[irwork], info);
                    iu = itauq;

                    for (i = 0; i < m; i += ldwrku) {
                        chunk = ((m - i) < ldwrku) ? (m - i) : ldwrku;
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    chunk, n, n, &CONE, &A[i], lda,
                                    &work[ir], ldwrkr, &CZERO, &work[iu], ldwrku);
                        zlacpy("F", chunk, n, &work[iu], ldwrku, &A[i], lda);
                    }
                } else {
                    itau = 0;
                    iwork = itau + n;

                    zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("U", n, n, A, lda, VT, ldvt);
                    if (n > 1) {
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &VT[1], ldvt);
                    }
                    zungqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = 0;
                    itauq = itau;
                    itaup = itauq + n;
                    iwork = itaup + n;

                    zgebrd(n, n, VT, ldvt, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zunmbr("Q", "R", "N", m, n, n, VT, ldvt, &work[itauq],
                           A, lda, &work[iwork], lwork - iwork, &ierr);
                    zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + n;

                    zbdsqr("U", n, n, m, 0, S, &rwork[ie], VT, ldvt,
                           A, lda, NULL, 1, &rwork[irwork], info);
                }
            } else if (wntus) {
                /* Paths 4, 5, 6: U in U array */
                if (wntvn) {
                    /* Path 4: U in U, no V^H */
                    if (lwork >= n*n + 3*n) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[ir + 1], ldwrkr);
                        zungqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, &work[ir], ldwrkr, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, 0, n, 0, S, &rwork[ie], NULL, 1,
                               &work[ir], ldwrkr, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, &CONE, A, lda, &work[ir], ldwrkr, &CZERO, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, n, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                        }
                        zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, 0, m, 0, S, &rwork[ie], NULL, 1,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntvo) {
                    /* Path 5: U in U, V^H overwritten on A */
                    if (lwork >= 2*n*n + 3*n) {
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

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[iu + 1], ldwrku);
                        zungqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, &work[iu], ldwrku, &work[ir], ldwrkr);
                        zungbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, n, 0, S, &rwork[ie], &work[ir], ldwrkr,
                               &work[iu], ldwrku, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, &CONE, A, lda, &work[iu], ldwrku, &CZERO, U, ldu);
                        zlacpy("F", n, n, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, n, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                        }
                        zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, A, lda, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, m, 0, S, &rwork[ie], A, lda,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntvas) {
                    /* Path 6: U in U, V^H in VT */
                    if (lwork >= n*n + 3*n) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = iu + ldwrku * n;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[iu + 1], ldwrku);
                        zungqr(m, n, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, &work[iu], ldwrku, VT, ldvt);
                        zungbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, n, 0, S, &rwork[ie], VT, ldvt,
                               &work[iu], ldwrku, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, &CONE, A, lda, &work[iu], ldwrku, &CZERO, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, n, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        zlacpy("U", n, n, A, lda, VT, ldvt);
                        if (n > 1) {
                            zlaset("L", n - 1, n - 1, CZERO, CZERO, &VT[1], ldvt);
                        }

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, VT, ldvt, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("Q", "R", "N", m, n, n, VT, ldvt, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, m, 0, S, &rwork[ie], VT, ldvt,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                }
            } else if (wntua) {
                /* Paths 7, 8, 9: Full U */
                if (wntvn) {
                    /* Path 7: Full U, no V^H */
                    if (lwork >= n*n + ((n + m > 3*n) ? n + m : 3*n)) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[ir + 1], ldwrkr);
                        zungqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, &work[ir], ldwrkr, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", n, n, n, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, 0, n, 0, S, &rwork[ie], NULL, 1,
                               &work[ir], ldwrkr, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, &CONE, U, ldu, &work[ir], ldwrkr, &CZERO, A, lda);
                        zlacpy("F", m, n, A, lda, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                        }
                        zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, 0, m, 0, S, &rwork[ie], NULL, 1,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntvo) {
                    /* Path 8: Full U, V^H overwritten on A */
                    if (lwork >= 2*n*n + ((n + m > 3*n) ? n + m : 3*n)) {
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

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[iu + 1], ldwrku);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, &work[iu], ldwrku, &work[ir], ldwrkr);
                        zungbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, n, 0, S, &rwork[ie], &work[ir], ldwrkr,
                               &work[iu], ldwrku, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, &CONE, U, ldu, &work[iu], ldwrku, &CZERO, A, lda);
                        zlacpy("F", m, n, A, lda, U, ldu);
                        zlacpy("F", n, n, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        if (n > 1) {
                            zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                        }
                        zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("Q", "R", "N", m, n, n, A, lda, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, A, lda, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, m, 0, S, &rwork[ie], A, lda,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntvas) {
                    /* Path 9: Full U, V^H in VT */
                    if (lwork >= n*n + ((n + m > 3*n) ? n + m : 3*n)) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*n) ? lda : n;
                        itau = iu + ldwrku * n;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, A, lda, &work[iu], ldwrku);
                        zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[iu + 1], ldwrku);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", n, n, &work[iu], ldwrku, VT, ldvt);
                        zungbr("Q", n, n, n, &work[iu], ldwrku, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, n, 0, S, &rwork[ie], VT, ldvt,
                               &work[iu], ldwrku, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, n, &CONE, U, ldu, &work[iu], ldwrku, &CZERO, A, lda);
                        zlacpy("F", m, n, A, lda, U, ldu);
                    } else {
                        itau = 0;
                        iwork = itau + n;

                        zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, n, A, lda, U, ldu);
                        zungqr(m, m, n, U, ldu, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        zlacpy("U", n, n, A, lda, VT, ldvt);
                        if (n > 1) {
                            zlaset("L", n - 1, n - 1, CZERO, CZERO, &VT[1], ldvt);
                        }

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + n;
                        iwork = itaup + n;

                        zgebrd(n, n, VT, ldvt, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("Q", "R", "N", m, n, n, VT, ldvt, &work[itauq],
                               U, ldu, &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + n;

                        zbdsqr("U", n, n, m, 0, S, &rwork[ie], VT, ldvt,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                }
            }
        } else {
            /* Path 10: M >= N but M < MNTHR - direct bidiagonalization */
            ie = 0;
            itauq = 0;
            itaup = itauq + n;
            iwork = itaup + n;

            zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
                   &work[iwork], lwork - iwork, &ierr);

            if (wntuas) {
                zlacpy("L", m, n, A, lda, U, ldu);
                ncu = wntus ? n : m;
                zungbr("Q", m, ncu, n, U, ldu, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvas) {
                zlacpy("U", n, n, A, lda, VT, ldvt);
                zungbr("P", n, n, n, VT, ldvt, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntuo) {
                zungbr("Q", m, n, n, A, lda, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvo) {
                zungbr("P", n, n, n, A, lda, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }

            irwork = ie + n;
            nru = (wntuas || wntuo) ? m : 0;
            ncvt = (wntvas || wntvo) ? n : 0;

            if (!wntuo && !wntvo) {
                zbdsqr("U", n, ncvt, nru, 0, S, &rwork[ie], VT, ldvt,
                       U, ldu, NULL, 1, &rwork[irwork], info);
            } else if (!wntuo && wntvo) {
                zbdsqr("U", n, ncvt, nru, 0, S, &rwork[ie], A, lda,
                       U, ldu, NULL, 1, &rwork[irwork], info);
            } else {
                zbdsqr("U", n, ncvt, nru, 0, S, &rwork[ie], VT, ldvt,
                       A, lda, NULL, 1, &rwork[irwork], info);
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

                zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                ie = 0;
                itauq = 0;
                itaup = itauq + m;
                iwork = itaup + m;

                zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
                       &work[iwork], lwork - iwork, &ierr);

                if (wntuo || wntuas) {
                    zungbr("Q", m, m, m, A, lda, &work[itauq], &work[iwork],
                           lwork - iwork, &ierr);
                }
                irwork = ie + m;
                nru = (wntuo || wntuas) ? m : 0;

                zbdsqr("U", m, 0, nru, 0, S, &rwork[ie], NULL, 1,
                       A, lda, NULL, 1, &rwork[irwork], info);

                if (wntuas) {
                    zlacpy("F", m, m, A, lda, U, ldu);
                }
            } else if (wntvo && wntun) {
                /* Path 2t: V^H overwritten on A, no U */
                if (lwork >= m*m + 3*m) {
                    ir = 0;
                    if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + lda*m) {
                        ldwrku = lda;
                        chunk = n;
                        ldwrkr = lda;
                    } else if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + m*m) {
                        ldwrku = lda;
                        chunk = n;
                        ldwrkr = m;
                    } else {
                        ldwrku = m;
                        chunk = (lwork - m*m) / m;
                        ldwrkr = m;
                    }
                    itau = ir + ldwrkr * m;
                    iwork = itau + m;

                    zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("L", m, m, A, lda, &work[ir], ldwrkr);
                    zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[ir + ldwrkr], ldwrkr);
                    zunglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = 0;
                    itauq = itau;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    zgebrd(m, m, &work[ir], ldwrkr, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zungbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + m;

                    zbdsqr("U", m, m, 0, 0, S, &rwork[ie], &work[ir], ldwrkr,
                           NULL, 1, NULL, 1, &rwork[irwork], info);
                    iu = itauq;

                    for (i = 0; i < n; i += chunk) {
                        blk = ((n - i) < chunk) ? (n - i) : chunk;
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, blk, m, &CONE, &work[ir], ldwrkr,
                                    &A[i * (long)lda], lda, &CZERO, &work[iu], ldwrku);
                        zlacpy("F", m, blk, &work[iu], ldwrku, &A[i * (long)lda], lda);
                    }
                } else {
                    ie = 0;
                    itauq = 0;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zungbr("P", m, n, m, A, lda, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + m;

                    zbdsqr("L", m, n, 0, 0, S, &rwork[ie], A, lda,
                           NULL, 1, NULL, 1, &rwork[irwork], info);
                }
            } else if (wntvo && wntuas) {
                /* Path 3t: V^H overwritten on A, U in U */
                if (lwork >= m*m + 3*m) {
                    ir = 0;
                    if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + lda*m) {
                        ldwrku = lda;
                        chunk = n;
                        ldwrkr = lda;
                    } else if (lwork >= ((wrkbl > lda*n) ? wrkbl : lda*n) + m*m) {
                        ldwrku = lda;
                        chunk = n;
                        ldwrkr = m;
                    } else {
                        ldwrku = m;
                        chunk = (lwork - m*m) / m;
                        ldwrkr = m;
                    }
                    itau = ir + ldwrkr * m;
                    iwork = itau + m;

                    zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("L", m, m, A, lda, U, ldu);
                    if (m > 1) {
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &U[ldu], ldu);
                    }
                    zunglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = 0;
                    itauq = itau;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    zgebrd(m, m, U, ldu, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("U", m, m, U, ldu, &work[ir], ldwrkr);
                    zungbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                           &work[iwork], lwork - iwork, &ierr);
                    zungbr("Q", m, m, m, U, ldu, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + m;

                    zbdsqr("U", m, m, m, 0, S, &rwork[ie], &work[ir], ldwrkr,
                           U, ldu, NULL, 1, &rwork[irwork], info);
                    iu = itauq;

                    for (i = 0; i < n; i += chunk) {
                        blk = ((n - i) < chunk) ? (n - i) : chunk;
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, blk, m, &CONE, &work[ir], ldwrkr,
                                    &A[i * (long)lda], lda, &CZERO, &work[iu], ldwrku);
                        zlacpy("F", m, blk, &work[iu], ldwrku, &A[i * (long)lda], lda);
                    }
                } else {
                    itau = 0;
                    iwork = itau + m;

                    zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                    zlacpy("L", m, m, A, lda, U, ldu);
                    if (m > 1) {
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &U[ldu], ldu);
                    }
                    zunglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                    ie = 0;
                    itauq = itau;
                    itaup = itauq + m;
                    iwork = itaup + m;

                    zgebrd(m, m, U, ldu, S, &rwork[ie], &work[itauq],
                           &work[itaup], &work[iwork], lwork - iwork, &ierr);
                    zunmbr("P", "L", "C", m, n, m, U, ldu, &work[itaup],
                           A, lda, &work[iwork], lwork - iwork, &ierr);
                    zungbr("Q", m, m, m, U, ldu, &work[itauq],
                           &work[iwork], lwork - iwork, &ierr);
                    irwork = ie + m;

                    zbdsqr("U", m, n, m, 0, S, &rwork[ie], A, lda,
                           U, ldu, NULL, 1, &rwork[irwork], info);
                }
            } else if (wntvs) {
                /* Paths 4t, 5t, 6t: V^H in VT */
                if (wntun) {
                    /* Path 4t: V^H in VT, no U */
                    if (lwork >= m*m + 3*m) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, A, lda, &work[ir], ldwrkr);
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[ir + ldwrkr], ldwrkr);
                        zunglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, &work[ir], ldwrkr, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, m, 0, 0, S, &rwork[ie], &work[ir], ldwrkr,
                               NULL, 1, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, &CONE, &work[ir], ldwrkr, A, lda, &CZERO, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(m, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                        zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("P", "L", "C", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, n, 0, 0, S, &rwork[ie], VT, ldvt,
                               NULL, 1, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntuo) {
                    /* Path 5t: V^H in VT, U overwritten on A */
                    if (lwork >= 2*m*m + 3*m) {
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

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[iu + ldwrku], ldwrku);
                        zunglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, &work[iu], ldwrku, &work[ir], ldwrkr);
                        zungbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, m, m, 0, S, &rwork[ie], &work[iu], ldwrku,
                               &work[ir], ldwrkr, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, &CONE, &work[iu], ldwrku, A, lda, &CZERO, VT, ldvt);
                        zlacpy("F", m, m, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(m, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                        zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("P", "L", "C", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, A, lda, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, n, m, 0, S, &rwork[ie], VT, ldvt,
                               A, lda, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntuas) {
                    /* Path 6t: V^H in VT, U in U */
                    if (lwork >= m*m + 3*m) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = iu + ldwrku * m;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[iu + ldwrku], ldwrku);
                        zunglq(m, n, m, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, &work[iu], ldwrku, U, ldu);
                        zungbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, m, m, 0, S, &rwork[ie], &work[iu], ldwrku,
                               U, ldu, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, &CONE, &work[iu], ldwrku, A, lda, &CZERO, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(m, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        zlacpy("L", m, m, A, lda, U, ldu);
                        if (m > 1) {
                            zlaset("U", m - 1, m - 1, CZERO, CZERO, &U[ldu], ldu);
                        }

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, U, ldu, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("P", "L", "C", m, n, m, U, ldu, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, n, m, 0, S, &rwork[ie], VT, ldvt,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                }
            } else if (wntva) {
                /* Paths 7t, 8t, 9t: Full V^H */
                if (wntun) {
                    /* Path 7t: Full V^H, no U */
                    if (lwork >= m*m + ((n + m > 3*m) ? n + m : 3*m)) {
                        ir = 0;
                        ldwrkr = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zlacpy("L", m, m, A, lda, &work[ir], ldwrkr);
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[ir + ldwrkr], ldwrkr);
                        zunglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, &work[ir], ldwrkr, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zungbr("P", m, m, m, &work[ir], ldwrkr, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, m, 0, 0, S, &rwork[ie], &work[ir], ldwrkr,
                               NULL, 1, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, &CONE, &work[ir], ldwrkr, VT, ldvt, &CZERO, A, lda);
                        zlacpy("F", m, n, A, lda, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                        zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("P", "L", "C", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, n, 0, 0, S, &rwork[ie], VT, ldvt,
                               NULL, 1, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntuo) {
                    /* Path 8t: Full V^H, U overwritten on A */
                    if (lwork >= 2*m*m + ((n + m > 3*m) ? n + m : 3*m)) {
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

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[iu + ldwrku], ldwrku);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, &work[iu], ldwrku, &work[ir], ldwrkr);
                        zungbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, &work[ir], ldwrkr, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, m, m, 0, S, &rwork[ie], &work[iu], ldwrku,
                               &work[ir], ldwrkr, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, &CONE, &work[iu], ldwrku, VT, ldvt, &CZERO, A, lda);
                        zlacpy("F", m, n, A, lda, VT, ldvt);
                        zlacpy("F", m, m, &work[ir], ldwrkr, A, lda);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                        zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("P", "L", "C", m, n, m, A, lda, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, A, lda, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, n, m, 0, S, &rwork[ie], VT, ldvt,
                               A, lda, NULL, 1, &rwork[irwork], info);
                    }
                } else if (wntuas) {
                    /* Path 9t: Full V^H, U in U */
                    if (lwork >= m*m + ((n + m > 3*m) ? n + m : 3*m)) {
                        iu = 0;
                        ldwrku = (lwork >= wrkbl + lda*m) ? lda : m;
                        itau = iu + ldwrku * m;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zlacpy("L", m, m, A, lda, &work[iu], ldwrku);
                        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[iu + ldwrku], ldwrku);
                        zunglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, &work[iu], ldwrku, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("L", m, m, &work[iu], ldwrku, U, ldu);
                        zungbr("P", m, m, m, &work[iu], ldwrku, &work[itaup],
                               &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, m, m, 0, S, &rwork[ie], &work[iu], ldwrku,
                               U, ldu, NULL, 1, &rwork[irwork], info);

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n, m, &CONE, &work[iu], ldwrku, VT, ldvt, &CZERO, A, lda);
                        zlacpy("F", m, n, A, lda, VT, ldvt);
                    } else {
                        itau = 0;
                        iwork = itau + m;

                        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &ierr);
                        zlacpy("U", m, n, A, lda, VT, ldvt);
                        zunglq(n, n, m, VT, ldvt, &work[itau], &work[iwork], lwork - iwork, &ierr);

                        zlacpy("L", m, m, A, lda, U, ldu);
                        if (m > 1) {
                            zlaset("U", m - 1, m - 1, CZERO, CZERO, &U[ldu], ldu);
                        }

                        ie = 0;
                        itauq = itau;
                        itaup = itauq + m;
                        iwork = itaup + m;

                        zgebrd(m, m, U, ldu, S, &rwork[ie], &work[itauq],
                               &work[itaup], &work[iwork], lwork - iwork, &ierr);
                        zunmbr("P", "L", "C", m, n, m, U, ldu, &work[itaup],
                               VT, ldvt, &work[iwork], lwork - iwork, &ierr);
                        zungbr("Q", m, m, m, U, ldu, &work[itauq],
                               &work[iwork], lwork - iwork, &ierr);
                        irwork = ie + m;

                        zbdsqr("U", m, n, m, 0, S, &rwork[ie], VT, ldvt,
                               U, ldu, NULL, 1, &rwork[irwork], info);
                    }
                }
            }
        } else {
            /* Path 10t: N > M but N < MNTHR - direct bidiagonalization */
            ie = 0;
            itauq = 0;
            itaup = itauq + m;
            iwork = itaup + m;

            zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
                   &work[iwork], lwork - iwork, &ierr);

            if (wntuas) {
                zlacpy("L", m, m, A, lda, U, ldu);
                zungbr("Q", m, m, n, U, ldu, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvas) {
                zlacpy("U", m, n, A, lda, VT, ldvt);
                nrvt = wntva ? n : m;
                zungbr("P", nrvt, n, m, VT, ldvt, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntuo) {
                zungbr("Q", m, m, n, A, lda, &work[itauq], &work[iwork],
                       lwork - iwork, &ierr);
            }
            if (wntvo) {
                zungbr("P", m, n, m, A, lda, &work[itaup], &work[iwork],
                       lwork - iwork, &ierr);
            }

            irwork = ie + m;
            nru = (wntuas || wntuo) ? m : 0;
            ncvt = (wntvas || wntvo) ? n : 0;

            if (!wntuo && !wntvo) {
                zbdsqr("L", m, ncvt, nru, 0, S, &rwork[ie], VT, ldvt,
                       U, ldu, NULL, 1, &rwork[irwork], info);
            } else if (!wntuo && wntvo) {
                zbdsqr("L", m, ncvt, nru, 0, S, &rwork[ie], A, lda,
                       U, ldu, NULL, 1, &rwork[irwork], info);
            } else {
                zbdsqr("L", m, ncvt, nru, 0, S, &rwork[ie], VT, ldvt,
                       A, lda, NULL, 1, &rwork[irwork], info);
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            dlascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (*info != 0 && anrm > bignum) {
            dlascl("G", 0, 0, bignum, anrm, minmn - 1, 1, &rwork[ie], minmn, &ierr);
        }
        if (anrm < smlnum) {
            dlascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (*info != 0 && anrm < smlnum) {
            dlascl("G", 0, 0, smlnum, anrm, minmn - 1, 1, &rwork[ie], minmn, &ierr);
        }
    }

    /* Return optimal workspace */
    work[0] = CMPLX((f64)maxwrk, 0.0);

    #undef MAX3
}
