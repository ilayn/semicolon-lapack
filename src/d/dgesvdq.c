/**
 * @file dgesvdq.c
 * @brief DGESVDQ computes SVD with a QR-Preconditioned QR SVD Method.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <math.h>
#include "semicolon_cblas.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;

/**
 * DGESVDQ computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V**T
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N orthogonal matrix.
 */
/**
 * DGESVDQ computes the singular value decomposition (SVD) of a real
 * `m`-by-`n` matrix `A`, where `m>=n`. The SVD of `A` is written as
 * @rst
 * .. code-block:: text
 *
 *                                    [++]   [xx]   [x0]   [xx]
 *              A = U * SIGMA * V^*,  [++] = [xx] * [ox] * [xx]
 *                                    [++]   [xx]
 * @endrst
 * where SIGMA is an `n`-by-`n` diagonal matrix, U is an `m`-by-`n` orthonormal
 * matrix, and V is an `n`-by-`n` orthogonal matrix. The diagonal elements
 * of SIGMA are the singular values of `A`. The columns of U and V are the
 * left and the right singular vectors of `A`, respectively.
 *
 * @param[in]     joba  Specifies the level of accuracy in the computed SVD.
 *                      `joba='A'`: The requested accuracy corresponds to having the
 *                      backward error bounded by `|| delta A ||_F <= f(m,n)*EPS*|| A ||_F`,
 *                      where `EPS = dlamch("Epsilon")`. This authorises DGESVDQ to
 *                      truncate the computed triangular factor in a rank revealing
 *                      QR factorization whenever the truncated part is below the
 *                      threshold of the order of `EPS * ||A||_F`. This is aggressive
 *                      truncation level.
 *                      `joba='M'`: Similarly as with `'A'`, but the truncation is more
 *                      gentle: it is allowed only when there is a drop on the diagonal
 *                      of the triangular factor in the QR factorization. This is medium
 *                      truncation level.
 *                      `joba='H'`: High accuracy requested. No numerical rank
 *                      determination based on the rank revealing QR factorization is
 *                      attempted.
 *                      `joba='E'`: Same as `'H'`, and in addition the condition number
 *                      of column scaled `A` is estimated and returned in `rwork[0]`.
 *                      `N^(-1/4)*rwork[0] <= ||pinv(A_scaled)||_2 <= N^(1/4)*rwork[0]`.
 * @param[in]     jobp  `jobp='P'`: The rows of `A` are ordered in decreasing order with
 *                      respect to `||A(i,:)||_\infty`. This enhances numerical accuracy
 *                      at the cost of extra data movement. Recommended for numerical
 *                      robustness.
 *                      `jobp='N'`: No row pivoting.
 * @param[in]     jobr  `jobr='T'`: After the initial pivoted QR factorization, DGESVD is
 *                      applied to the transposed `R**T` of the computed triangular factor
 *                      R. This involves some extra data movement (matrix transpositions).
 *                      Useful for experiments, research and development.
 *                      `jobr='N'`: The triangular factor R is given as input to DGESVD.
 *                      This may be preferred as it involves less data movement.
 * @param[in]     jobu  `jobu='A'`: All `m` left singular vectors are computed and returned
 *                      in the matrix `U`. See the description of `U`.
 *                      `jobu='S'` or `jobu='U'`: `n = min(m,n)` left singular vectors are
 *                      computed and returned in the matrix `U`. See the description of `U`.
 *                      `jobu='R'`: Numerical rank `numrank` is determined and only
 *                      `numrank` left singular vectors are computed and returned in the
 *                      matrix `U`.
 *                      `jobu='F'`: The `n` left singular vectors are returned in factored
 *                      form as the product of the Q factor from the initial QR
 *                      factorization and the `n` left singular vectors of `(R**T, 0)**T`.
 *                      If row pivoting is used, then the necessary information on the row
 *                      pivoting is stored in `iwork[n:n+m-2]`.
 *                      `jobu='N'`: The left singular vectors are not computed.
 * @param[in]     jobv  `jobv='A'` or `jobv='V'`: All `n` right singular vectors are
 *                      computed and returned in the matrix `V`.
 *                      `jobv='R'`: Numerical rank `numrank` is determined and only
 *                      `numrank` right singular vectors are computed and returned in the
 *                      matrix `V`. This option is allowed only if `jobu='R'` or `jobu='N'`;
 *                      otherwise it is illegal.
 *                      `jobv='N'`: The right singular vectors are not computed.
 * @param[in]     m     The number of rows of the input matrix `A`. `m>=0`.
 * @param[in]     n     The number of columns of the input matrix `A`. `m>=n>=0`.
 * @param[in,out] A     Array of dimensions `lda` x `n`. On entry, the input matrix `A`.
 *                      On exit, if `jobu!='N'` or `jobv!='N'`, the lower triangle of `A`
 *                      contains the Householder vectors as stored by DGEQP3. If
 *                      `jobu='F'`, these Householder vectors together with `work[0:n-1]`
 *                      can be used to restore the Q factors from the initial pivoted QR
 *                      factorization of `A`. See the description of `U`.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,m)`.
 * @param[out]    S     Array of dimension `n`. The singular values of `A`, ordered so
 *                      that `S[i]>=S[i+1]`.
 * @param[out]    U     Array of dimension `ldu` x `m` if `jobu='A'`; see the description
 *                      of `ldu`. In this case, on exit, `U` contains the `m` left singular
 *                      vectors.
 *                      `ldu` x `n` if `jobu='S'`, `'U'`, `'R'`; see the description of
 *                      `ldu`. In this case, `U` contains the leading `n` or the leading
 *                      `numrank` left singular vectors.
 *                      `ldu` x `n` if `jobu='F'`; see the description of `ldu`. In this
 *                      case `U` contains `n` x `n` orthogonal matrix that can be used to
 *                      form the left singular vectors.
 *                      If `jobu='N'`, `U` is not referenced.
 * @param[in]     ldu   The leading dimension of the array `U`.
 *                      If `jobu='A'`, `'S'`, `'U'`, `'R'`, `ldu>=max(1,m)`.
 *                      If `jobu='F'`, `ldu>=max(1,n)`.
 *                      Otherwise, `ldu>=1`.
 * @param[out]    V     Array of dimension `ldv` x `n` if `jobv='A'`, `'V'`, `'R'` or if
 *                      `joba='E'`.
 *                      If `jobv='A'` or `'V'`, `V` contains the `n`-by-`n` orthogonal
 *                      matrix `V**T`;
 *                      If `jobv='R'`, `V` contains the first `numrank` rows of `V**T`
 *                      (the right singular vectors, stored rowwise, of the `numrank`
 *                      largest singular values).
 *                      If `jobv='N'` and `joba='E'`, `V` is used as a workspace.
 *                      If `jobv='N'` and `joba!='E'`, `V` is not referenced.
 * @param[in]     ldv   The leading dimension of the array `V`.
 *                      If `jobv='A'`, `'V'`, `'R'`, or `joba='E'`, `ldv>=max(1,n)`.
 *                      Otherwise, `ldv>=1`.
 * @param[out]    numrank `numrank` is the numerical rank first determined after the rank
 *                      revealing QR factorization, following the strategy specified by the
 *                      value of `joba`. If `jobv='R'` and `jobu='R'`, only `numrank`
 *                      leading singular values and vectors are then requested in the call
 *                      of DGESVD. The final value of `numrank` might be further reduced if
 *                      some singular values are computed as zeros.
 * @param[out]    iwork Integer array of dimension (`max(1,liwork)`).
 *                      On exit, `iwork[0:n-1]` contains column pivoting permutation of the
 *                      rank revealing QR factorization.
 *                      If `jobp='P'`, `iwork[n:n+m-2]` contains the indices of the sequence
 *                      of row swaps used in row pivoting. These can be used to restore the
 *                      left singular vectors in the case `jobu='F'`.
 *                      If `liwork`, `lwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `iwork[0]` returns the minimal `liwork`.
 * @param[in]     liwork The dimension of the array `iwork`.
 *                      `liwork >= n+m-1`, if `jobp='P'` and `joba!='E'`;
 *                      `liwork >= n`, if `jobp='N'` and `joba!='E'`;
 *                      `liwork >= n+m-1+n`, if `jobp='P'` and `joba='E'`;
 *                      `liwork >= n+n`, if `jobp='N'` and `joba='E'`.
 *                      If `liwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates and returns the optimal and minimal sizes
 *                      for the `work`, `iwork`, and `rwork` arrays, and no error
 *                      message related to `lwork` is issued by XERBLA.
 * @param[out]    work  Array of dimension (`max(2,lwork)`), used as a workspace.
 *                      On exit, if, on entry, `lwork!=-1`, `work[0:n-1]` contains
 *                      parameters needed to recover the Q factor from the QR factorization
 *                      computed by DGEQP3.
 *                      If `liwork`, `lwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `work[0]` returns the optimal `lwork`, and
 *                      `work[1]` returns the minimal `lwork`.
 * @param[in,out] lwork The dimension of the array `work`. It is determined as follows:
 *                       @rst
 *                       .. code-block:: text
 *
 *                           Let  LWQP3 = 3*N+1,  LWCON = 3*N, and let
 *                           LWORQ = { MAX( N, 1 ),  if JOBU = 'R', 'S', or 'U'
 *                                   { MAX( M, 1 ),  if JOBU = 'A'
 *                           LWSVD = MAX( 5*N, 1 )
 *                           LWLQF = MAX( N/2, 1 ), LWSVD2 = MAX( 5*(N/2), 1 ), LWORLQ = MAX( N, 1 ),
 *                           LWQRF = MAX( N/2, 1 ), LWORQ2 = MAX( N, 1 )
 *                           Then the minimal value of LWORK is:
 *                           = MAX( N + LWQP3, LWSVD )        if only the singular values are needed;
 *                           = MAX( N + LWQP3, LWCON, LWSVD ) if only the singular values are needed,
 *                                                    and a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD, LWORQ ) if the singular values and the left
 *                                                    singular vectors are requested;
 *                           = N + MAX( LWQP3, LWCON, LWSVD, LWORQ ) if the singular values and the left
 *                                                    singular vectors are requested, and also
 *                                                    a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD )        if the singular values and the right
 *                                                    singular vectors are requested;
 *                           = N + MAX( LWQP3, LWCON, LWSVD ) if the singular values and the right
 *                                                    singular vectors are requested, and also
 *                                                    a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD, LWORQ ) if the full SVD is requested with JOBV = 'R';
 *                                                    independent of JOBR;
 *                           = N + MAX( LWQP3, LWCON, LWSVD, LWORQ ) if the full SVD is requested,
 *                                                    JOBV = 'R' and, also a scaled condition
 *                                                    estimate requested; independent of JOBR;
 *                           = MAX( N + MAX( LWQP3, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, N/2+LWLQF, N/2+LWSVD2, N/2+LWORLQ, LWORQ) ) if the
 *                                          full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='N'
 *                           = MAX( N + MAX( LWQP3, LWCON, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, LWCON, N/2+LWLQF, N/2+LWSVD2, N/2+LWORLQ, LWORQ ) )
 *                                          if the full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='N', and also a scaled condition number estimate
 *                                          requested.
 *                           = MAX( N + MAX( LWQP3, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, N/2+LWQRF, N/2+LWSVD2, N/2+LWORQ2, LWORQ ) ) if the
 *                                          full SVD is requested with JOBV = 'A', 'V', and JOBR ='T'
 *                           = MAX( N + MAX( LWQP3, LWCON, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, LWCON, N/2+LWQRF, N/2+LWSVD2, N/2+LWORQ2, LWORQ ) )
 *                                          if the full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='T', and also a scaled condition number estimate
 *                                          requested.
 *                       @endrst
 *                       Finally, `lwork` must be at least two: `lwork = max(2,lwork)`.
 *                       If `lwork=-1`, then a workspace query is assumed; the routine
 *                       only calculates and returns the optimal and minimal sizes
 *                       for the `work`, `iwork`, and `rwork` arrays, and no error
 *                       message related to `lwork` is issued by XERBLA.
 * @param[out]    rwork Array of dimension (`max(1,lrwork)`). On exit,
 *                      1. If `joba='E'`, `rwork[0]` contains an estimate of the condition
 *                      number of column scaled `A`. If `A = C * D` where D is diagonal and C
 *                      has unit columns in the Euclidean norm, then, assuming full column
 *                      rank, `N^(-1/4) * rwork[0] <= ||pinv(C)||_2 <= N^(1/4) * rwork[0]`.
 *                      Otherwise, `rwork[0] = -1`.
 *                      2. `rwork[1]` contains the number of singular values computed as
 *                      exact zeros in DGESVD applied to the upper triangular or trapezoidal
 *                      R (from the initial QR factorization). In case of early exit (no call
 *                      to DGESVD, such as in the case of zero matrix) `rwork[1] = -1`.
 *                      If `liwork`, `lwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `rwork[0]` returns the minimal `lrwork`.
 * @param[in]     lrwork The dimension of the array `rwork`.
 *                      If `jobp='P'`, then `lrwork >= max(2,m)`.
 *                      Otherwise, `lrwork >= 2`.
 *                      If `lrwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates and returns the optimal and minimal sizes
 *                      for the `work`, `iwork`, and `rwork` arrays, and no error
 *                      message related to `lwork` is issued by XERBLA.
 * @param[out]    info  `info=0`: successful exit.
 *                      `info<0`: if `info=-i`, the i-th argument had an illegal value.
 *                      `info>0`: if DBDSQR did not converge, `info` specifies how many
 *                      superdiagonals of an intermediate bidiagonal form B (computed in
 *                      DGESVD) did not converge to zero.
 *
 * @par Further Details:
 * @rst
 * 1. The data movement (matrix transpose) is coded using simple nested
 * DO-loops because BLAS and LAPACK do not provide corresponding subroutines.
 * Those DO-loops are easily identified in this source code - by the CONTINUE
 * statements labeled with 11**. In an optimized version of this code, the
 * nested DO loops should be replaced with calls to an optimized subroutine.
 *
 * 2. This code scales A by 1/SQRT(M) if the largest ABS(A(i,j)) could cause
 * column norm overflow. This is the minimal precaution and it is left to the
 * SVD routine (DGESVD) to do its own preemptive scaling if potential over-
 * or underflows are detected. To avoid repeated scanning of the array A,
 * an optimal implementation would do all necessary scaling before calling
 * DGESVD and the scaling in DGESVD can be switched off.
 *
 * 3. Other comments related to code optimization are given in comments in the
 * code, enclosed in [[double brackets]].
 * @endrst
 *
 * @par Bugs, examples and comments:
 * @rst
 * Please report all bugs and send interesting examples and/or comments to
 * drmac@math.hr. Thank you.
 * @endrst
 *
 * @par References:
 * @rst
 * [1] Zlatko Drmac, Algorithm 977: A QR-Preconditioned QR SVD Method for
 * Computing the SVD with High Accuracy. ACM Trans. Math. Softw.
 * 44(1): 11:1-11:30 (2017)
 *
 * SIGMA library, xGESVDQ section updated February 2016.
 * Developed and coded by Zlatko Drmac, Department of Mathematics
 * University of Zagreb, Croatia, drmac@math.hr
 * @endrst
 *
 * @par Contributors:
 * @rst
 * Developed and coded by Zlatko Drmac, Department of Mathematics
 * University of Zagreb, Croatia, drmac@math.hr
 * @endrst
 */
void dgesvdq(const char* joba, const char* jobp, const char* jobr,
             const char* jobu, const char* jobv,
             const INT m, const INT n, f64* restrict A, const INT lda,
             f64* restrict S, f64* restrict U, const INT ldu,
             f64* restrict V, const INT ldv, INT* numrank,
             INT* restrict iwork, const INT liwork,
             f64* restrict work, const INT lwork,
             f64* restrict rwork, const INT lrwork, INT* info)
{
    INT wntus, wntur, wntua, wntuf, lsvc0, lsvec, dntwu;
    INT wntvr, wntva, rsvec, dntwv;
    INT accla, acclm, acclh, conda;
    INT rowprm, rtrans, lquery;
    INT lwcon, lwqp3, lwrk_dgelqf, lwrk_dgesvd, lwrk_dgesvd2;
    INT lwrk_dgeqp3 = 0, lwrk_dgeqrf, lwrk_dormlq, lwrk_dormqr = 0;
    INT lwrk_dormqr2, lwlqf, lwqrf, lwsvd, lwsvd2, lworq;
    INT lworq2, lworlq, minwrk, minwrk2, optwrk, optwrk2;
    INT iminwrk, rminwrk;
    INT ierr, iwoff, nr, n1, optratio, p, q;
    INT ascaled;
    f64 big, epsln, sconda, sfmin, rtmp;
    f64 rdummy[1];

    /* Test the input arguments */
    wntus = (jobu[0] == 'S' || jobu[0] == 's' || jobu[0] == 'U' || jobu[0] == 'u');
    wntur = (jobu[0] == 'R' || jobu[0] == 'r');
    wntua = (jobu[0] == 'A' || jobu[0] == 'a');
    wntuf = (jobu[0] == 'F' || jobu[0] == 'f');
    lsvc0 = wntus || wntur || wntua;
    lsvec = lsvc0 || wntuf;
    dntwu = (jobu[0] == 'N' || jobu[0] == 'n');

    wntvr = (jobv[0] == 'R' || jobv[0] == 'r');
    wntva = (jobv[0] == 'A' || jobv[0] == 'a' || jobv[0] == 'V' || jobv[0] == 'v');
    rsvec = wntvr || wntva;
    dntwv = (jobv[0] == 'N' || jobv[0] == 'n');

    accla = (joba[0] == 'A' || joba[0] == 'a');
    acclm = (joba[0] == 'M' || joba[0] == 'm');
    conda = (joba[0] == 'E' || joba[0] == 'e');
    acclh = (joba[0] == 'H' || joba[0] == 'h') || conda;

    rowprm = (jobp[0] == 'P' || jobp[0] == 'p');
    rtrans = (jobr[0] == 'T' || jobr[0] == 't');

    if (rowprm) {
        if (conda) {
            iminwrk = (1 > n + m - 1 + n) ? 1 : n + m - 1 + n;
        } else {
            iminwrk = (1 > n + m - 1) ? 1 : n + m - 1;
        }
        rminwrk = (2 > m) ? 2 : m;
    } else {
        if (conda) {
            iminwrk = (1 > n + n) ? 1 : n + n;
        } else {
            iminwrk = (1 > n) ? 1 : n;
        }
        rminwrk = 2;
    }
    lquery = (liwork == -1 || lwork == -1 || lrwork == -1);
    *info = 0;
    if (!accla && !acclm && !acclh) {
        *info = -1;
    } else if (!rowprm && !(jobp[0] == 'N' || jobp[0] == 'n')) {
        *info = -2;
    } else if (!rtrans && !(jobr[0] == 'N' || jobr[0] == 'n')) {
        *info = -3;
    } else if (!lsvec && !dntwu) {
        *info = -4;
    } else if (wntur && wntva) {
        *info = -5;
    } else if (!rsvec && !dntwv) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (n < 0 || n > m) {
        *info = -7;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -9;
    } else if (ldu < 1 || (lsvc0 && ldu < m) || (wntuf && ldu < n)) {
        *info = -12;
    } else if (ldv < 1 || (rsvec && ldv < n) || (conda && ldv < n)) {
        *info = -14;
    } else if (liwork < iminwrk && !lquery) {
        *info = -17;
    }

    if (*info == 0) {
        /*
         * .. compute the minimal and the optimal workspace lengths
         * [[The expressions for computing the minimal and the optimal
         * values of LWORK are written with a lot of redundancy and
         * can be simplified. However, this detailed form is easier for
         * maintenance and modifications of the code.]]
         */

        /* .. minimal workspace length for DGEQP3 of an M x N matrix */
        lwqp3 = 3 * n + 1;
        /* .. minimal workspace length for DORMQR to build left singular vectors */
        if (wntus || wntur) {
            lworq = (n > 1) ? n : 1;
        } else if (wntua) {
            lworq = (m > 1) ? m : 1;
        } else {
            lworq = 0;
        }
        /* .. minimal workspace length for DPOCON of an N x N matrix */
        lwcon = 3 * n;
        /* .. DGESVD of an N x N matrix */
        lwsvd = (5 * n > 1) ? 5 * n : 1;
        if (lquery) {
            dgeqp3(m, n, A, lda, iwork, NULL, rdummy, -1, &ierr);
            lwrk_dgeqp3 = (INT)rdummy[0];
            if (wntus || wntur) {
                dormqr("L", "N", m, n, n, A, lda, NULL, U,
                       ldu, rdummy, -1, &ierr);
                lwrk_dormqr = (INT)rdummy[0];
            } else if (wntua) {
                dormqr("L", "N", m, m, n, A, lda, NULL, U,
                       ldu, rdummy, -1, &ierr);
                lwrk_dormqr = (INT)rdummy[0];
            } else {
                lwrk_dormqr = 0;
            }
        }
        minwrk = 2;
        optwrk = 2;
        if (!lsvec && !rsvec) {
            /*
             * .. minimal and optimal sizes of the workspace if
             * only the singular values are requested
             */
            if (conda) {
                minwrk = n + lwqp3;
                if (minwrk < lwcon) minwrk = lwcon;
                if (minwrk < lwsvd) minwrk = lwsvd;
            } else {
                minwrk = n + lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
            }
            if (lquery) {
                dgesvd("N", "N", n, n, A, lda, S, U, ldu,
                       V, ldv, rdummy, -1, &ierr);
                lwrk_dgesvd = (INT)rdummy[0];
                if (conda) {
                    optwrk = n + lwrk_dgeqp3;
                    if (optwrk < n + lwcon) optwrk = n + lwcon;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                } else {
                    optwrk = n + lwrk_dgeqp3;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                }
            }
        } else if (lsvec && !rsvec) {
            /*
             * .. minimal and optimal sizes of the workspace if the
             * singular values and the left singular vectors are requested
             */
            if (conda) {
                minwrk = lwqp3;
                if (minwrk < lwcon) minwrk = lwcon;
                if (minwrk < lwsvd) minwrk = lwsvd;
                if (minwrk < lworq) minwrk = lworq;
                minwrk = n + minwrk;
            } else {
                minwrk = lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
                if (minwrk < lworq) minwrk = lworq;
                minwrk = n + minwrk;
            }
            if (lquery) {
                if (rtrans) {
                    dgesvd("N", "O", n, n, A, lda, S, U, ldu,
                           V, ldv, rdummy, -1, &ierr);
                } else {
                    dgesvd("O", "N", n, n, A, lda, S, U, ldu,
                           V, ldv, rdummy, -1, &ierr);
                }
                lwrk_dgesvd = (INT)rdummy[0];
                if (conda) {
                    optwrk = lwrk_dgeqp3;
                    if (optwrk < lwcon) optwrk = lwcon;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                    if (optwrk < lwrk_dormqr) optwrk = lwrk_dormqr;
                    optwrk = n + optwrk;
                } else {
                    optwrk = lwrk_dgeqp3;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                    if (optwrk < lwrk_dormqr) optwrk = lwrk_dormqr;
                    optwrk = n + optwrk;
                }
            }
        } else if (rsvec && !lsvec) {
            /*
             * .. minimal and optimal sizes of the workspace if the
             * singular values and the right singular vectors are requested
             */
            if (conda) {
                minwrk = lwqp3;
                if (minwrk < lwcon) minwrk = lwcon;
                if (minwrk < lwsvd) minwrk = lwsvd;
                minwrk = n + minwrk;
            } else {
                minwrk = lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
                minwrk = n + minwrk;
            }
            if (lquery) {
                if (rtrans) {
                    dgesvd("O", "N", n, n, A, lda, S, U, ldu,
                           V, ldv, rdummy, -1, &ierr);
                } else {
                    dgesvd("N", "O", n, n, A, lda, S, U, ldu,
                           V, ldv, rdummy, -1, &ierr);
                }
                lwrk_dgesvd = (INT)rdummy[0];
                if (conda) {
                    optwrk = lwrk_dgeqp3;
                    if (optwrk < lwcon) optwrk = lwcon;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                    optwrk = n + optwrk;
                } else {
                    optwrk = lwrk_dgeqp3;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                    optwrk = n + optwrk;
                }
            }
        } else {
            /*
             * .. minimal and optimal sizes of the workspace if the
             * full SVD is requested
             */
            if (rtrans) {
                minwrk = lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
                if (minwrk < lworq) minwrk = lworq;
                if (conda && minwrk < lwcon) minwrk = lwcon;
                minwrk = minwrk + n;
                if (wntva) {
                    /* .. minimal workspace length for N x N/2 DGEQRF */
                    lwqrf = (n / 2 > 1) ? n / 2 : 1;
                    /* .. minimal workspace length for N/2 x N/2 DGESVD */
                    lwsvd2 = (5 * (n / 2) > 1) ? 5 * (n / 2) : 1;
                    lworq2 = (n > 1) ? n : 1;
                    minwrk2 = lwqp3;
                    if (minwrk2 < n / 2 + lwqrf) minwrk2 = n / 2 + lwqrf;
                    if (minwrk2 < n / 2 + lwsvd2) minwrk2 = n / 2 + lwsvd2;
                    if (minwrk2 < n / 2 + lworq2) minwrk2 = n / 2 + lworq2;
                    if (minwrk2 < lworq) minwrk2 = lworq;
                    if (conda && minwrk2 < lwcon) minwrk2 = lwcon;
                    minwrk2 = n + minwrk2;
                    if (minwrk < minwrk2) minwrk = minwrk2;
                }
            } else {
                minwrk = lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
                if (minwrk < lworq) minwrk = lworq;
                if (conda && minwrk < lwcon) minwrk = lwcon;
                minwrk = minwrk + n;
                if (wntva) {
                    /* .. minimal workspace length for N/2 x N DGELQF */
                    lwlqf = (n / 2 > 1) ? n / 2 : 1;
                    lwsvd2 = (5 * (n / 2) > 1) ? 5 * (n / 2) : 1;
                    lworlq = (n > 1) ? n : 1;
                    minwrk2 = lwqp3;
                    if (minwrk2 < n / 2 + lwlqf) minwrk2 = n / 2 + lwlqf;
                    if (minwrk2 < n / 2 + lwsvd2) minwrk2 = n / 2 + lwsvd2;
                    if (minwrk2 < n / 2 + lworlq) minwrk2 = n / 2 + lworlq;
                    if (minwrk2 < lworq) minwrk2 = lworq;
                    if (conda && minwrk2 < lwcon) minwrk2 = lwcon;
                    minwrk2 = n + minwrk2;
                    if (minwrk < minwrk2) minwrk = minwrk2;
                }
            }
            if (lquery) {
                if (rtrans) {
                    dgesvd("O", "A", n, n, A, lda, S, U, ldu,
                           V, ldv, rdummy, -1, &ierr);
                    lwrk_dgesvd = (INT)rdummy[0];
                    optwrk = lwrk_dgeqp3;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                    if (optwrk < lwrk_dormqr) optwrk = lwrk_dormqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        dgeqrf(n, n / 2, U, ldu, NULL, rdummy, -1, &ierr);
                        lwrk_dgeqrf = (INT)rdummy[0];
                        dgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                               ldu, NULL, 1, rdummy, -1, &ierr);
                        lwrk_dgesvd2 = (INT)rdummy[0];
                        dormqr("R", "C", n, n, n / 2, U, ldu, NULL,
                               V, ldv, rdummy, -1, &ierr);
                        lwrk_dormqr2 = (INT)rdummy[0];
                        optwrk2 = lwrk_dgeqp3;
                        if (optwrk2 < n / 2 + lwrk_dgeqrf) optwrk2 = n / 2 + lwrk_dgeqrf;
                        if (optwrk2 < n / 2 + lwrk_dgesvd2) optwrk2 = n / 2 + lwrk_dgesvd2;
                        if (optwrk2 < n / 2 + lwrk_dormqr2) optwrk2 = n / 2 + lwrk_dormqr2;
                        if (conda && optwrk2 < lwcon) optwrk2 = lwcon;
                        optwrk2 = n + optwrk2;
                        if (optwrk < optwrk2) optwrk = optwrk2;
                    }
                } else {
                    dgesvd("S", "O", n, n, A, lda, S, U, ldu,
                           V, ldv, rdummy, -1, &ierr);
                    lwrk_dgesvd = (INT)rdummy[0];
                    optwrk = lwrk_dgeqp3;
                    if (optwrk < lwrk_dgesvd) optwrk = lwrk_dgesvd;
                    if (optwrk < lwrk_dormqr) optwrk = lwrk_dormqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        dgelqf(n / 2, n, U, ldu, NULL, rdummy, -1, &ierr);
                        lwrk_dgelqf = (INT)rdummy[0];
                        dgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                               ldu, NULL, 1, rdummy, -1, &ierr);
                        lwrk_dgesvd2 = (INT)rdummy[0];
                        dormlq("R", "N", n, n, n / 2, U, ldu, NULL,
                               V, ldv, rdummy, -1, &ierr);
                        lwrk_dormlq = (INT)rdummy[0];
                        optwrk2 = lwrk_dgeqp3;
                        if (optwrk2 < n / 2 + lwrk_dgelqf) optwrk2 = n / 2 + lwrk_dgelqf;
                        if (optwrk2 < n / 2 + lwrk_dgesvd2) optwrk2 = n / 2 + lwrk_dgesvd2;
                        if (optwrk2 < n / 2 + lwrk_dormlq) optwrk2 = n / 2 + lwrk_dormlq;
                        if (conda && optwrk2 < lwcon) optwrk2 = lwcon;
                        optwrk2 = n + optwrk2;
                        if (optwrk < optwrk2) optwrk = optwrk2;
                    }
                }
            }
        }

        if (minwrk < 2) minwrk = 2;
        if (optwrk < 2) optwrk = 2;
        if (lwork < minwrk && !lquery) *info = -19;
    }

    if (*info == 0 && lrwork < rminwrk && !lquery) {
        *info = -21;
    }
    if (*info != 0) {
        xerbla("DGESVDQ", -(*info));
        return;
    } else if (lquery) {
        /* Return optimal workspace */
        iwork[0] = iminwrk;
        work[0] = (f64)optwrk;
        work[1] = (f64)minwrk;
        rwork[0] = (f64)rminwrk;
        return;
    }

    /* Quick return if the matrix is void. */
    if (m == 0 || n == 0) {
        /* .. all output is void. */
        return;
    }

    big = dlamch("O");
    epsln = dlamch("E");
    sfmin = dlamch("S");

    ascaled = 0;
    iwoff = 0;
    sconda = -ONE;
    n1 = n;

    if (rowprm) {
        iwoff = m - 1;
        /*
         * .. reordering the rows in decreasing sequence in the
         * ell-infinity norm - this enhances numerical robustness in
         * the case of differently scaled rows.
         */
        for (p = 0; p < m; p++) {
            rwork[p] = dlange("M", 1, n, &A[p], lda, NULL);
            /* .. check for NaN's and Inf's */
            if (rwork[p] != rwork[p] || (rwork[p] * ZERO) != ZERO) {
                *info = -8;
                xerbla("DGESVDQ", -(*info));
                return;
            }
        }
        for (p = 0; p < m - 1; p++) {
            q = cblas_idamax(m - p, &rwork[p], 1) + p;
            iwork[n + p] = q;
            if (p != q) {
                rtmp = rwork[p];
                rwork[p] = rwork[q];
                rwork[q] = rtmp;
            }
        }

        /* Quick return: A is the M x N zero matrix. */
        if (rwork[0] == ZERO) {
            *numrank = 0;
            dlaset("G", n, 1, ZERO, ZERO, S, n);
            if (wntus) dlaset("G", m, n, ZERO, ONE, U, ldu);
            if (wntua) dlaset("G", m, m, ZERO, ONE, U, ldu);
            if (wntva) dlaset("G", n, n, ZERO, ONE, V, ldv);
            if (wntuf) {
                dlaset("G", n, 1, ZERO, ZERO, work, n);
                dlaset("G", m, n, ZERO, ONE, U, ldu);
            }
            for (p = 0; p < n; p++) {
                iwork[p] = p;
            }
            if (rowprm) {
                for (p = n; p < n + m - 1; p++) {
                    iwork[p] = p - n;
                }
            }
            if (conda) rwork[0] = -ONE;
            rwork[1] = -ONE;
            return;
        }

        /*
         * .. to prevent overflow in the QR factorization, scale the
         * matrix by 1/sqrt(M) if too large entry detected
         */
        if (rwork[0] > big / sqrt((f64)m)) {
            dlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
        dlaswp(n, A, lda, 0, m - 2, &iwork[n], 1);
    } else {
        /*
         * .. At this stage, preemptive scaling is done only to avoid column
         * norms overflows during the QR factorization. The SVD procedure should
         * have its own scaling to save the singular values from overflows and
         * underflows. That depends on the SVD procedure.
         */
        rtmp = dlange("M", m, n, A, lda, NULL);
        if (rtmp != rtmp || (rtmp * ZERO) != ZERO) {
            *info = -8;
            xerbla("DGESVDQ", -(*info));
            return;
        }
        /*
         * .. to prevent overflow in the QR factorization, scale the
         * matrix by 1/sqrt(M) if too large entry detected
         */
        if (rtmp > big / sqrt((f64)m)) {
            dlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
    }

    /*
     * .. QR factorization with column pivoting
     *
     * A * P = Q * [ R ]
     *             [ 0 ]
     */
    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }
    dgeqp3(m, n, A, lda, iwork, work, &work[n], lwork - n, &ierr);

    /*
     * If the user requested accuracy level allows truncation in the
     * computed upper triangular factor, the matrix R is examined and,
     * if possible, replaced with its leading upper trapezoidal part.
     */
    nr = 1;
    if (accla) {
        /*
         * Standard absolute error bound suffices. All sigma_i with
         * sigma_i < N*EPS*||A||_F are flushed to zero. This is an
         * aggressive enforcement of lower numerical rank by introducing a
         * backward error of the order of N*EPS*||A||_F.
         */
        rtmp = sqrt((f64)n) * epsln;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < rtmp * fabs(A[0])) break;
            nr++;
        }
    } else if (acclm) {
        /*
         * .. similarly as above, only slightly more gentle (less aggressive).
         * Sudden drop on the diagonal of R is used as the criterion for being
         * close-to-rank-deficient. The threshold is set to EPSLN=DLAMCH('E').
         * [[This can be made more flexible by replacing this hard-coded value
         * with a user specified threshold.]] Also, the values that underflow
         * will be truncated.
         */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < epsln * fabs(A[(p-1) + (p-1) * lda]) ||
                fabs(A[p + p * lda]) < sfmin) break;
            nr++;
        }
    } else {
        /*
         * .. RRQR not authorized to determine numerical rank except in the
         * obvious case of zero pivots.
         * .. inspect R for exact zeros on the diagonal;
         * R(i,i)=0 => R(i:n-1,i:n-1)=0.
         */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) == ZERO) break;
            nr++;
        }

        /*
         * Estimate the scaled condition number of A. Use the fact that it is
         * the same as the scaled condition number of R.
         */
        if (conda) {
            dlacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < nr; p++) {
                rtmp = cblas_dnrm2(p + 1, &V[p * ldv], 1);
                cblas_dscal(p + 1, ONE / rtmp, &V[p * ldv], 1);
            }
            if (!lsvec && !rsvec) {
                dpocon("U", nr, V, ldv, ONE, &rtmp, work, &iwork[n + iwoff], &ierr);
            } else {
                dpocon("U", nr, V, ldv, ONE, &rtmp, &work[n], &iwork[n + iwoff], &ierr);
            }
            sconda = ONE / sqrt(rtmp);
        }
    }

    if (wntur) {
        n1 = nr;
    } else if (wntus || wntuf) {
        n1 = n;
    } else if (wntua) {
        n1 = m;
    }

    if (!rsvec && !lsvec) {
        /*
         * .. only the singular values are requested
         */
        if (rtrans) {
            /*
             * .. compute the singular values of R**T = [A](0:nr-1,0:n-1)**T
             * .. set the lower triangle of [A] to [A](0:nr-1,0:n-1)**T and
             * the upper triangle of [A] to zero.
             */
            INT minmn = (n < nr) ? n : nr;
            for (p = 0; p < minmn; p++) {
                for (q = p + 1; q < n; q++) {
                    A[q + p * lda] = A[p + q * lda];
                    if (q < nr) A[p + q * lda] = ZERO;
                }
            }
            dgesvd("N", "N", n, nr, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        } else {
            /* .. compute the singular values of R = [A](0:nr-1,0:n-1) */
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &A[1], lda);
            }
            dgesvd("N", "N", nr, n, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        }

    } else if (lsvec && !rsvec) {
        /*
         * .. the singular values and the left singular vectors requested
         */
        if (rtrans) {
            /*
             * .. apply DGESVD to R**T
             * .. copy R**T into [U] and overwrite [U] with the right singular
             * vectors of R
             */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    U[q + p * ldu] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[ldu], ldu);
            }
            /*
             * .. the left singular vectors not computed, the NR right singular
             * vectors overwrite [U](0:nr-1,0:nr-1) as transposed. These
             * will be pre-multiplied by Q to build the left singular vectors of A.
             */
            dgesvd("N", "O", n, nr, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
            for (p = 0; p < nr; p++) {
                for (q = p + 1; q < nr; q++) {
                    rtmp = U[q + p * ldu];
                    U[q + p * ldu] = U[p + q * ldu];
                    U[p + q * ldu] = rtmp;
                }
            }
        } else {
            /*
             * .. apply DGESVD to R
             * .. copy R into [U] and overwrite [U] with the left singular vectors
             */
            dlacpy("U", nr, n, A, lda, U, ldu);
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[1], ldu);
            }
            /*
             * .. the right singular vectors not computed, the NR left singular
             * vectors overwrite [U](0:nr-1,0:nr-1)
             */
            dgesvd("O", "N", nr, n, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
        }

        /*
         * .. assemble the left singular vector matrix U of dimensions
         * (M x NR) or (M x N) or (M x M).
         */
        if (nr < m && !wntuf) {
            dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
            if (nr < n1) {
                dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
            }
        }

        /*
         * The Q matrix from the first QRF is built into the left singular
         * vectors matrix U.
         */
        if (!wntuf) {
            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        if (rowprm && !wntuf) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }

    } else if (rsvec && !lsvec) {
        /*
         * .. the singular values and the right singular vectors requested
         */
        if (rtrans) {
            /*
             * .. apply DGESVD to R**T
             * .. copy R**T into V and overwrite V with the left singular vectors
             */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    V[q + p * ldv] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
            }

            if (wntvr || nr == n) {
                /*
                 * .. the left singular vectors of R**T overwrite V, the right singular
                 * vectors not computed
                 */
                dgesvd("O", "N", n, nr, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = V[q + p * ldv];
                        }
                    }
                }
                dlapmt(0, nr, n, V, ldv, iwork);
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * [!] This is simple implementation that augments [V](0:n-1,0:nr-1)
                 * by padding a zero block. In the case NR << N, a more efficient
                 * way is to first use the QR factorization. For more details
                 * how to implement this, see the " FULL SVD " branch.
                 */
                dlaset("G", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                dgesvd("O", "N", n, n, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                for (p = 0; p < n; p++) {
                    for (q = p + 1; q < n; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                dlapmt(0, n, n, V, ldv, iwork);
            }
        } else {
            /*
             * .. aply DGESVD to R
             * .. copy R into V and overwrite V with the right singular vectors
             */
            dlacpy("U", nr, n, A, lda, V, ldv);
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
            }

            if (wntvr || nr == n) {
                /*
                 * .. the right singular vectors overwrite V, the NR left singular
                 * vectors stored in U(0:nr-1,0:nr-1)
                 */
                dgesvd("N", "O", nr, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, nr, n, V, ldv, iwork);
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * [!] This is simple implementation that augments [V](0:nr-1,0:n-1)
                 * by padding a zero block. In the case NR << N, a more efficient
                 * way is to first use the LQ factorization. For more details
                 * how to implement this, see the " FULL SVD " branch.
                 */
                dlaset("G", n - nr, n, ZERO, ZERO, &V[nr], ldv);
                dgesvd("N", "O", n, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, n, n, V, ldv, iwork);
            }
        }

    } else {
        /*
         * .. FULL SVD requested
         */
        if (rtrans) {
            /*
             * .. apply DGESVD to R**T [[this option is left for R&D&T]]
             */
            if (wntvr || nr == n) {
                /*
                 * .. copy R**T into [V] and overwrite [V] with the left singular
                 * vectors of R**T
                 */
                for (p = 0; p < nr; p++) {
                    for (q = p; q < n; q++) {
                        V[q + p * ldv] = A[p + q * lda];
                    }
                }
                if (nr > 1) {
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                }

                dgesvd("O", "A", n, nr, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                /* .. assemble V */
                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = V[q + p * ldv];
                        }
                    }
                }
                dlapmt(0, nr, n, V, ldv, iwork);

                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = U[q + p * ldu];
                        U[q + p * ldu] = U[p + q * ldu];
                        U[p + q * ldu] = rtmp;
                    }
                }

                if (nr < m && !wntuf) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * .. copy R**T into [V] and overwrite [V] with the left singular
                 * vectors of R**T
                 * [[The optimal ratio N/NR for using QRF instead of padding
                 *   with zeros. Here hard coded to 2; it must be at least
                 *   two due to work space constraints.]]
                 */
                optratio = 2;
                if (optratio * nr > n) {
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            V[q + p * ldv] = A[p + q * lda];
                        }
                    }
                    if (nr > 1) {
                        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    }
                    dlaset("A", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);

                    dgesvd("O", "A", n, n, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = V[q + p * ldv];
                            V[q + p * ldv] = V[p + q * ldv];
                            V[p + q * ldv] = rtmp;
                        }
                    }
                    dlapmt(0, n, n, V, ldv, iwork);

                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = U[q + p * ldu];
                            U[q + p * ldu] = U[p + q * ldu];
                            U[p + q * ldu] = rtmp;
                        }
                    }

                    /*
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x N1), i.e. (M x N) or (M x M).
                     */
                    if (n < m && !wntuf) {
                        dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    /*
                     * .. copy R**T into [U] and overwrite [U] with the right
                     * singular vectors of R
                     */
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            U[q + (nr + p) * ldu] = A[p + q * lda];
                        }
                    }
                    if (nr > 1) {
                        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[(nr + 1) * ldu], ldu);
                    }
                    dgeqrf(n, nr, &U[nr * ldu], ldu, &work[n], &work[n + nr], lwork - n - nr, &ierr);
                    for (p = 0; p < nr; p++) {
                        for (q = 0; q < n; q++) {
                            V[q + p * ldv] = U[p + (nr + q) * ldu];
                        }
                    }
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    dgesvd("S", "O", nr, nr, V, ldv, S, U, ldu, NULL, 1, &work[n + nr], lwork - n - nr, info);
                    dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                    dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                    dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    dormqr("R", "C", n, n, nr, &U[nr * ldu], ldu, &work[n], V, ldv, &work[n + nr], lwork - n - nr, &ierr);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /*
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x NR) or (M x N) or (M x M).
                     */
                    if (nr < m && !wntuf) {
                        dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                        if (nr < n1) {
                            dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                            dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        } else {
            /*
             * .. apply DGESVD to R [[this is the recommended option]]
             */
            if (wntvr || nr == n) {
                /* .. copy R into [V] and overwrite V with the right singular vectors */
                dlacpy("U", nr, n, A, lda, V, ldv);
                if (nr > 1) {
                    dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                }

                dgesvd("S", "O", nr, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, nr, n, V, ldv, iwork);

                /*
                 * .. now [V](0:nr-1,0:n-1) contains V(0:n-1,0:nr-1)**T
                 * .. assemble the left singular vector matrix U of dimensions
                 * (M x NR) or (M x N) or (M x M).
                 */
                if (nr < m && !wntuf) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * .. the requested number of the left singular vectors
                 *  is then N1 (N or M)
                 * [[The optimal ratio N/NR for using LQ instead of padding
                 *   with zeros. Here hard coded to 2; it must be at least
                 *   two due to work space constraints.]]
                 */
                optratio = 2;
                if (optratio * nr > n) {
                    dlacpy("U", nr, n, A, lda, V, ldv);
                    if (nr > 1) {
                        dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                    }
                    dlaset("A", n - nr, n, ZERO, ZERO, &V[nr], ldv);

                    dgesvd("S", "O", n, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /*
                     * .. now [V] contains the transposed matrix of the right
                     * singular vectors of A. The leading N left singular vectors
                     * are in [U](0:n-1,0:n-1)
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x N1), i.e. (M x N) or (M x M).
                     */
                    if (n < m && !wntuf) {
                        dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    dlacpy("U", nr, n, A, lda, &U[nr], ldu);
                    if (nr > 1) {
                        dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[nr + 1], ldu);
                    }
                    dgelqf(nr, n, &U[nr], ldu, &work[n], &work[n + nr], lwork - n - nr, &ierr);
                    dlacpy("L", nr, nr, &U[nr], ldu, V, ldv);
                    if (nr > 1) {
                        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    }
                    dgesvd("S", "O", nr, nr, V, ldv, S, U, ldu, NULL, 1, &work[n + nr], lwork - n - nr, info);
                    dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                    dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                    dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    dormlq("R", "N", n, n, nr, &U[nr], ldu, &work[n], V, ldv, &work[n + nr], lwork - n - nr, &ierr);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /*
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x NR) or (M x N) or (M x M).
                     */
                    if (nr < m && !wntuf) {
                        dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                        if (nr < n1) {
                            dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                            dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        }

        /*
         * The Q matrix from the first QRF is built into the left singular
         * vectors matrix U.
         */
        if (!wntuf) {
            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        if (rowprm && !wntuf) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }
    }

    /*
     * Check whether some singular values are returned as zeros, e.g.
     * due to underflow, and update the numerical rank.
     */
    p = nr;
    for (q = nr - 1; q >= 0; q--) {
        if (S[q] > ZERO) break;
        nr--;
    }

    /*
     * .. if numerical rank deficiency is detected, the truncated
     * singular values are set to zero.
     */
    if (nr < n) {
        dlaset("G", n - nr, 1, ZERO, ZERO, &S[nr], n);
    }

    /*
     * .. undo scaling; this may cause overflow in the largest singular
     * values.
     */
    if (ascaled) {
        dlascl("G", 0, 0, ONE, sqrt((f64)m), nr, 1, S, n, &ierr);
    }

    if (conda) rwork[0] = sconda;
    rwork[1] = (f64)(p - nr);

    *numrank = nr;
}
