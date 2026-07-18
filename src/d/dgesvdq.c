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
    INT lwqp3, lwcon, lwsvd, lworq, minwrk, optwrk;
    INT iminwrk, rminwrk;
    INT ierr, iwoff, nr, n1, optratio, p, q;
    INT ascaled;
    f64 big, epsln, sconda, sfmin, rtmp;

    /* Decode job parameters */
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

    /* Workspace requirements */
    if (rowprm) {
        iminwrk = conda ? (n + m - 1 + n) : (n + m - 1);
        iminwrk = (iminwrk > 1) ? iminwrk : 1;
        rminwrk = (m > 2) ? m : 2;
    } else {
        iminwrk = conda ? (n + n) : n;
        iminwrk = (iminwrk > 1) ? iminwrk : 1;
        rminwrk = 2;
    }

    lquery = (liwork == -1 || lwork == -1 || lrwork == -1);

    /* Workspace sizes */
    lwqp3 = 3 * n + 1;
    if (wntus || wntur) {
        lworq = (n > 1) ? n : 1;
    } else if (wntua) {
        lworq = (m > 1) ? m : 1;
    } else {
        lworq = 1;
    }
    lwcon = 3 * n;
    lwsvd = (5 * n > 1) ? 5 * n : 1;

    if (!lsvec && !rsvec) {
        /* only the singular values are requested */
        minwrk = n + lwqp3;
        if (conda && minwrk < lwcon) minwrk = lwcon;
        if (minwrk < lwsvd) minwrk = lwsvd;
    } else if (lsvec && !rsvec) {
        /* singular values and the left singular vectors are requested */
        INT t = lwqp3;
        if (t < lwsvd) t = lwsvd;
        if (t < lworq) t = lworq;
        if (conda && t < lwcon) t = lwcon;
        minwrk = n + t;
    } else if (rsvec && !lsvec) {
        /* singular values and the right singular vectors are requested */
        INT t = lwqp3;
        if (t < lwsvd) t = lwsvd;
        if (conda && t < lwcon) t = lwcon;
        minwrk = n + t;
    } else {
        /* full SVD is requested */
        INT t = lwqp3;
        if (t < lwsvd) t = lwsvd;
        if (t < lworq) t = lworq;
        if (conda && t < lwcon) t = lwcon;
        minwrk = n + t;
        if (wntva) {
            const INT half = n / 2;
            INT lwsvd2 = 5 * half;
            if (lwsvd2 < 1) lwsvd2 = 1;
            INT minwrk2;
            if (rtrans) {
                /* minimal workspace for N x N/2 DGEQRF */
                INT lwqrf = (half > 1) ? half : 1;
                INT lworq2 = (n > 1) ? n : 1;
                minwrk2 = lwqp3;
                if (minwrk2 < half + lwqrf) minwrk2 = half + lwqrf;
                if (minwrk2 < half + lwsvd2) minwrk2 = half + lwsvd2;
                if (minwrk2 < half + lworq2) minwrk2 = half + lworq2;
                if (minwrk2 < lworq) minwrk2 = lworq;
            } else {
                /* minimal workspace for N/2 x N DGELQF */
                INT lwlqf = (half > 1) ? half : 1;
                INT lworlq = (n > 1) ? n : 1;
                minwrk2 = lwqp3;
                if (minwrk2 < half + lwlqf) minwrk2 = half + lwlqf;
                if (minwrk2 < half + lwsvd2) minwrk2 = half + lwsvd2;
                if (minwrk2 < half + lworlq) minwrk2 = half + lworlq;
                if (minwrk2 < lworq) minwrk2 = lworq;
            }
            if (conda && minwrk2 < lwcon) minwrk2 = lwcon;
            minwrk2 = n + minwrk2;
            if (minwrk < minwrk2) minwrk = minwrk2;
        }
    }
    if (minwrk < 2) minwrk = 2;
    optwrk = minwrk;

    /* Check input arguments */
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
    } else if (ldv < 1 || ((rsvec || conda) && ldv < n)) {
        *info = -14;
    } else if (liwork < iminwrk && !lquery) {
        *info = -17;
    }

    if (*info == 0 && !lquery) {
        if (lwork < minwrk) {
            *info = -19;
        }
        if (lrwork < rminwrk) {
            *info = -21;
        }
    }

    if (*info != 0) {
        xerbla("DGESVDQ", -(*info));
        return;
    }

    if (lquery) {
        iwork[0] = iminwrk;
        work[0] = (f64)optwrk;
        work[1] = (f64)minwrk;
        rwork[0] = (f64)rminwrk;
        return;
    }

    /* Quick return */
    if (m == 0 || n == 0) {
        *numrank = 0;
        return;
    }

    /* Get machine parameters */
    big = dlamch("O");
    epsln = dlamch("E");
    sfmin = dlamch("S");

    /* Initialize */
    ascaled = 0;
    iwoff = 1;
    sconda = -ONE;
    n1 = n;

    /*
     * Row pivoting
     */
    if (rowprm) {
        iwoff = m;
        /* Compute row infinity norms */
        for (p = 0; p < m; p++) {
            rwork[p] = dlange("M", 1, n, &A[p], lda, NULL);
            /* Check for NaN/Inf */
            if (rwork[p] != rwork[p] || (rwork[p] * ZERO) != ZERO) {
                *info = -8;
                xerbla("DGESVDQ", -(*info));
                return;
            }
        }
        /* Sort rows by decreasing infinity norm */
        for (p = 0; p < m - 1; p++) {
            q = cblas_idamax(m - p, &rwork[p], 1) + p;
            iwork[n + p] = q;
            if (p != q) {
                rtmp = rwork[p];
                rwork[p] = rwork[q];
                rwork[q] = rtmp;
            }
        }

        /* Quick return for zero matrix */
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
                iwork[p] = p + 1;
            }
            if (rowprm) {
                for (p = n; p < n + m - 1; p++) {
                    iwork[p] = p - n + 1;
                }
            }
            if (conda) rwork[0] = -ONE;
            rwork[1] = -ONE;
            return;
        }

        /* Scale if needed to prevent overflow */
        if (rwork[0] > big / sqrt((f64)m)) {
            dlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
        /* Apply row permutation */
        dlaswp(n, A, lda, 0, m - 2, &iwork[n], 1);
    } else {
        /* No row pivoting */
        rtmp = dlange("M", m, n, A, lda, NULL);
        /* Check for NaN/Inf */
        if (rtmp != rtmp || (rtmp * ZERO) != ZERO) {
            *info = -8;
            xerbla("DGESVDQ", -(*info));
            return;
        }
        /* Scale if needed to prevent overflow */
        if (rtmp > big / sqrt((f64)m)) {
            dlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
    }

    /*
     * QR factorization with column pivoting: A * P = Q * R
     */
    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }
    dgeqp3(m, n, A, lda, iwork, work, &work[n], lwork - n, &ierr);

    /*
     * Numerical rank determination
     */
    nr = 1;
    if (accla) {
        /* Aggressive truncation: |R(i,i)| < sqrt(n)*eps*|R(1,1)| */
        rtmp = sqrt((f64)n) * epsln;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < rtmp * fabs(A[0])) break;
            nr++;
        }
    } else if (acclm) {
        /* Medium truncation: sudden drop or underflow */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < epsln * fabs(A[(p-1) + (p-1) * lda]) ||
                fabs(A[p + p * lda]) < sfmin) break;
            nr++;
        }
    } else {
        /* High accuracy: only exact zeros */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) == ZERO) break;
            nr++;
        }

        /* Condition number estimation for ACCLH with CONDA */
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

    /*
     * Compute N1: dimension for left singular vectors output
     */
    if (wntur) {
        n1 = nr;
    } else if (wntus || wntuf) {
        n1 = n;
    } else if (wntua) {
        n1 = m;
    }

    /*
     * Main computation branches
     */
    if (!rsvec && !lsvec) {
        /*
         * Singular values only
         */
        if (rtrans) {
            /* Compute SVD of R^T */
            INT minmn = (n < nr) ? n : nr;
            for (p = 0; p < minmn; p++) {
                for (q = p + 1; q < n; q++) {
                    A[q + p * lda] = A[p + q * lda];
                    if (q < nr) A[p + q * lda] = ZERO;
                }
            }
            dgesvd("N", "N", n, nr, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        } else {
            /* Compute SVD of R */
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &A[1], lda);
            }
            dgesvd("N", "N", nr, n, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        }

    } else if (lsvec && !rsvec) {
        /*
         * Left singular vectors requested
         */
        if (rtrans) {
            /* Apply DGESVD to R^T */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    U[q + p * ldu] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[ldu], ldu);
            }
            /* JOBU='N': U not referenced; JOBVT='O': overwrites A */
            dgesvd("N", "O", n, nr, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
            /* Transpose result */
            for (p = 0; p < nr; p++) {
                for (q = p + 1; q < nr; q++) {
                    rtmp = U[q + p * ldu];
                    U[q + p * ldu] = U[p + q * ldu];
                    U[p + q * ldu] = rtmp;
                }
            }
        } else {
            /* Apply DGESVD to R */
            dlacpy("U", nr, n, A, lda, U, ldu);
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[1], ldu);
            }
            /* JOBU='O': overwrites A; JOBVT='N': VT not referenced */
            dgesvd("O", "N", nr, n, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
        }

        /* Assemble left singular vectors */
        if (nr < m && !wntuf) {
            dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
            if (nr < n1) {
                dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
            }
        }

        /* Apply Q factor */
        if (!wntuf) {
            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        /* Undo row pivoting */
        if (rowprm && !wntuf) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }

    } else if (rsvec && !lsvec) {
        /*
         * Right singular vectors requested
         */
        if (rtrans) {
            /* Apply DGESVD to R^T */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    V[q + p * ldv] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
            }

            if (wntvr || nr == n) {
                /* JOBU='O': overwrites A; JOBVT='N': VT not referenced */
                dgesvd("O", "N", n, nr, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                /* Transpose */
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
                /* Need all N right vectors, NR < N */
                dlaset("G", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                /* JOBU='O': overwrites A; JOBVT='N': VT not referenced */
                dgesvd("O", "N", n, n, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                /* Transpose */
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
            /* Apply DGESVD to R */
            dlacpy("U", nr, n, A, lda, V, ldv);
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
            }

            if (wntvr || nr == n) {
                /* JOBU='N': U not referenced; JOBVT='O': overwrites A */
                dgesvd("N", "O", nr, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, nr, n, V, ldv, iwork);
            } else {
                /* Need all N right vectors, NR < N */
                dlaset("G", n - nr, n, ZERO, ZERO, &V[nr], ldv);
                /* JOBU='N': U not referenced; JOBVT='O': overwrites A */
                dgesvd("N", "O", n, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, n, n, V, ldv, iwork);
            }
        }

    } else {
        /*
         * Full SVD: both left and right singular vectors
         */
        if (rtrans) {
            /*
             * Apply DGESVD to R^T
             */
            if (wntvr || nr == n) {
                /* Copy R^T to V */
                for (p = 0; p < nr; p++) {
                    for (q = p; q < n; q++) {
                        V[q + p * ldv] = A[p + q * lda];
                    }
                }
                if (nr > 1) {
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                }

                dgesvd("O", "A", n, nr, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                /* Transpose V */
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

                /* Transpose U */
                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = U[q + p * ldu];
                        U[q + p * ldu] = U[p + q * ldu];
                        U[p + q * ldu] = rtmp;
                    }
                }

                /* Assemble U */
                if (nr < m && !wntuf) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /* Need all N right vectors, NR < N */
                optratio = 2;
                if (optratio * nr > n) {
                    /* Padding approach */
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

                    /* Transpose V */
                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = V[q + p * ldv];
                            V[q + p * ldv] = V[p + q * ldv];
                            V[p + q * ldv] = rtmp;
                        }
                    }
                    dlapmt(0, n, n, V, ldv, iwork);

                    /* Transpose U */
                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = U[q + p * ldu];
                            U[q + p * ldu] = U[p + q * ldu];
                            U[p + q * ldu] = rtmp;
                        }
                    }

                    /* Assemble U */
                    if (n < m && !wntuf) {
                        dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    /* QR optimization */
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

                    /* Assemble U */
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
             * Apply DGESVD to R (recommended option)
             */
            if (wntvr || nr == n) {
                /* Copy R to V */
                dlacpy("U", nr, n, A, lda, V, ldv);
                if (nr > 1) {
                    dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                }

                dgesvd("S", "O", nr, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, nr, n, V, ldv, iwork);

                /* Assemble U */
                if (nr < m && !wntuf) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /* Need all N right vectors, NR < N */
                optratio = 2;
                if (optratio * nr > n) {
                    /* Padding approach */
                    dlacpy("U", nr, n, A, lda, V, ldv);
                    if (nr > 1) {
                        dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                    }
                    dlaset("A", n - nr, n, ZERO, ZERO, &V[nr], ldv);

                    dgesvd("S", "O", n, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /* Assemble U */
                    if (n < m && !wntuf) {
                        dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    /* LQ optimization */
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

                    /* Assemble U */
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

        /* Apply Q factor */
        if (!wntuf) {
            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        /* Undo row pivoting */
        if (rowprm && !wntuf) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }
    }

    /*
     * Check for zeros in S and update rank
     */
    p = nr;
    for (q = nr - 1; q >= 0; q--) {
        if (S[q] > ZERO) break;
        nr--;
    }

    /* Zero truncated singular values */
    if (nr < n) {
        dlaset("G", n - nr, 1, ZERO, ZERO, &S[nr], n);
    }

    /* Undo scaling */
    if (ascaled) {
        dlascl("G", 0, 0, ONE, sqrt((f64)m), nr, 1, S, n, &ierr);
    }

    if (conda) rwork[0] = sconda;
    rwork[1] = (f64)(p - nr);

    *numrank = nr;
    *info = ierr;
}
