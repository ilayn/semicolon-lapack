/**
 * @file dgejsv.c
 * @brief DGEJSV computes the SVD of a real M-by-N matrix using preconditioned
 *        Jacobi rotations with sophisticated preprocessing for high accuracy.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include "semicolon_cblas.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;

/** @cond */
static inline INT max3i(INT a, INT b, INT c) {
    INT m = (a > b) ? a : b;
    return (m > c) ? m : c;
}
/** @endcond */

/**
 * DGEJSV computes the singular value decomposition (SVD) of a real M-by-N
 * matrix [A], where M >= N. The SVD of [A] is written as
 *
 *              [A] = [U] * [SIGMA] * [V]^t,
 *
 * where [SIGMA] is an N-by-N (M-by-N) matrix which is zero except for its N
 * diagonal elements, [U] is an M-by-N (or M-by-M) orthonormal matrix, and
 * [V] is an N-by-N orthogonal matrix. The diagonal elements of [SIGMA] are
 * the singular values of [A]. The columns of [U] and [V] are the left and
 * the right singular vectors of [A], respectively. The matrices [U] and [V]
 * are computed and stored in the arrays U and V, respectively. The diagonal
 * of [SIGMA] is computed and stored in the array SVA.
 *
 * DGEJSV can sometimes compute tiny singular values and their singular vectors
 * much more accurately than other SVD routines, see below under Further Details.
 *
 * @param[in] joba
 *        Specifies the level of accuracy:
 *        - 'C': This option works well (high relative accuracy) if A = B * D,
 *              with well-conditioned B and arbitrary diagonal matrix D.
 *              The accuracy cannot be spoiled by COLUMN scaling. The
 *              accuracy of the computed output depends on the condition of
 *              B, and the procedure aims at the best theoretical accuracy.
 *              The relative error max_{i=1:N}|d sigma_i| / sigma_i is
 *              bounded by f(M,N)*epsilon* cond(B), independent of D.
 *              The input matrix is preprocessed with the QRF with column
 *              pivoting. This initial preprocessing and preconditioning by
 *              a rank revealing QR factorization is common for all values of
 *              JOBA. Additional actions are specified as follows:
 *        - 'E': Computation as with 'C' with an additional estimate of the
 *              condition number of B. It provides a realistic error bound.
 *        - 'F': If A = D1 * C * D2 with ill-conditioned diagonal scalings
 *              D1, D2, and well-conditioned matrix C, this option gives
 *              higher accuracy than the 'C' option. If the structure of the
 *              input matrix is not known, and relative accuracy is
 *              desirable, then this option is advisable. The input matrix A
 *              is preprocessed with QR factorization with FULL (row and
 *              column) pivoting.
 *        - 'G': Computation as with 'F' with an additional estimate of the
 *              condition number of B, where A=D*B. If A has heavily weighted
 *              rows, then using this condition number gives too pessimistic
 *              error bound.
 *        - 'A': Small singular values are the noise and the matrix is treated
 *              as numerically rank deficient. The error in the computed
 *              singular values is bounded by f(m,n)*epsilon*||A||.
 *              The computed SVD A = U * S * V^t restores A up to
 *              f(m,n)*epsilon*||A||.
 *              This gives the procedure the licence to discard (set to zero)
 *              all singular values below N*epsilon*||A||.
 *        - 'R': Similar as in 'A'. Rank revealing property of the initial
 *              QR factorization is used do reveal (using triangular factor)
 *              a gap sigma_{r+1} < epsilon * sigma_r in which case the
 *              numerical RANK is declared to be r. The SVD is computed with
 *              absolute error bounds, but more accurately than with 'A'.
 *
 * @param[in] jobu
 *        Specifies whether to compute the columns of U:
 *        - 'U': N columns of U are returned in the array U.
 *        - 'F': full set of M left sing. vectors is returned in the array U.
 *        - 'W': U may be used as workspace of length M*N. See the description
 *              of U.
 *        - 'N': U is not computed.
 *
 * @param[in] jobv
 *        Specifies whether to compute the matrix V:
 *        - 'V': N columns of V are returned in the array V; Jacobi rotations
 *              are not explicitly accumulated.
 *        - 'J': N columns of V are returned in the array V, but they are
 *              computed as the product of Jacobi rotations. This option is
 *              allowed only if JOBU .NE. 'N', i.e. in computing the full SVD.
 *        - 'W': V may be used as workspace of length N*N. See the description
 *              of V.
 *        - 'N': V is not computed.
 *
 * @param[in] jobr
 *        Specifies the RANGE for the singular values. Issues the licence to
 *        set to zero small positive singular values if they are outside
 *        specified range. If A .NE. 0 is scaled so that the largest singular
 *        value of c*A is around sqrt(BIG), BIG=DLAMCH('O'), then JOBR issues
 *        the licence to kill columns of A whose norm in c*A is less than
 *        sqrt(SFMIN) (for JOBR = 'R'), or less than SMALL=SFMIN/EPSLN,
 *        where SFMIN=DLAMCH('S'), EPSLN=DLAMCH('E').
 *        - 'N': Do not kill small columns of c*A. This option assumes that
 *              BLAS and QR factorizations and triangular solvers are
 *              implemented to work in that range. If the condition of A
 *              is greater than BIG, use DGESVJ.
 *        - 'R': RESTRICTED range for sigma(c*A) is [sqrt(SFMIN), sqrt(BIG)]
 *              (roughly, as described above). This option is recommended.
 *        For computing the singular values in the FULL range [SFMIN,BIG]
 *        use DGESVJ.
 *
 * @param[in] jobt
 *        If the matrix is square then the procedure may determine to use
 *        transposed A if A^t seems to be better with respect to convergence.
 *        If the matrix is not square, JOBT is ignored. This is subject to
 *        changes in the future.
 *        The decision is based on two values of entropy over the adjoint
 *        orbit of A^t * A. See the descriptions of work[5] and work[6].
 *        - 'T': transpose if entropy test indicates possibly faster
 *              convergence of Jacobi process if A^t is taken as input. If A is
 *              replaced with A^t, then the row pivoting is included automatically.
 *        - 'N': do not speculate.
 *        This option can be used to compute only the singular values, or the
 *        full SVD (U, SIGMA and V). For only one set of singular vectors
 *        (U or V), the caller should provide both U and V, as one of the
 *        matrices is used as workspace if the matrix A is transposed.
 *        The implementer can easily remove this constraint and make the
 *        code more complicated. See the descriptions of U and V.
 *
 * @param[in] jobp
 *        Issues the licence to introduce structured perturbations to drown
 *        denormalized numbers. This licence should be active if the
 *        denormals are poorly implemented, causing slow computation,
 *        especially in cases of fast convergence (!). For details see [1,2].
 *        For the sake of simplicity, this perturbations are included only
 *        when the full SVD or only the singular values are requested. The
 *        implementer/user can easily add the perturbation for the cases of
 *        computing one set of singular vectors.
 *        - 'P': introduce perturbation
 *        - 'N': do not perturb
 *
 * @param[in] m
 *        The number of rows of the input matrix A. M >= 0.
 *
 * @param[in] n
 *        The number of columns of the input matrix A. M >= N >= 0.
 *
 * @param[in,out] A
 *        Double precision array, dimension (lda, n).
 *        On entry, the M-by-N matrix A.
 *
 * @param[in] lda
 *        The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] SVA
 *        Double precision array, dimension (n).
 *        On exit:
 *        - For work[0]/work[1] = 1: The singular values of A. During the
 *          computation SVA contains Euclidean column norms of the
 *          iterated matrices in the array A.
 *        - For work[0] != work[1]: The singular values of A are
 *          (work[0]/work[1]) * SVA[0:n-1]. This factored form is used if
 *          sigma_max(A) overflows or if small singular values have been
 *          saved from underflow by scaling the input matrix A.
 *        - If JOBR='R' then some of the singular values may be returned
 *          as exact zeros obtained by "set to zero" because they are
 *          below the numerical rank threshold or are denormalized numbers.
 *
 * @param[out] U
 *        Double precision array, dimension (ldu, n) or (ldu, m).
 *        - If JOBU = 'U', then U contains on exit the M-by-N matrix of
 *          the left singular vectors.
 *        - If JOBU = 'F', then U contains on exit the M-by-M matrix of
 *          the left singular vectors, including an ONB of the orthogonal
 *          complement of the Range(A).
 *        - If JOBU = 'W' and (JOBV = 'V' and JOBT = 'T' and M = N),
 *          then U is used as workspace if the procedure replaces A with A^t.
 *          In that case, [V] is computed in U as left singular vectors of A^t
 *          and then copied back to the V array. This 'W' option is just
 *          a reminder to the caller that in this case U is reserved as
 *          workspace of length N*N.
 *        - If JOBU = 'N', U is not referenced, unless JOBT='T'.
 *
 * @param[in] ldu
 *        The leading dimension of the array U. ldu >= 1.
 *        If JOBU = 'U' or 'F' or 'W', then ldu >= m.
 *
 * @param[out] V
 *        Double precision array, dimension (ldv, n).
 *        - If JOBV = 'V' or 'J', then V contains on exit the N-by-N matrix of
 *          the right singular vectors.
 *        - If JOBV = 'W' and (JOBU = 'U' and JOBT = 'T' and M = N),
 *          then V is used as workspace if the procedure replaces A with A^t.
 *          In that case, [U] is computed in V as right singular vectors of A^t
 *          and then copied back to the U array. This 'W' option is just
 *          a reminder to the caller that in this case V is reserved as
 *          workspace of length N*N.
 *        - If JOBV = 'N', V is not referenced, unless JOBT='T'.
 *
 * @param[in] ldv
 *        The leading dimension of the array V. ldv >= 1.
 *        If JOBV = 'V' or 'J' or 'W', then ldv >= n.
 *
 * @param[out] work
 *        Double precision array, dimension (max(7, lwork)).
 *        On exit, if n > 0 and m > 0 (else not referenced):
 *        - work[0] = SCALE = work[1] / work[0] is the scaling factor such
 *                   that SCALE*SVA[0:n-1] are the computed singular values
 *                   of A. (See the description of SVA.)
 *        - work[1] = See the description of work[0].
 *        - work[2] = SCONDA is an estimate for the condition number of
 *                   column equilibrated A. (If JOBA = 'E' or 'G')
 *                   SCONDA is an estimate of sqrt(||(R^t * R)^(-1)||_1).
 *                   It is computed using DPOCON. It holds
 *                   N^(-1/4) * SCONDA <= ||R^(-1)||_2 <= N^(1/4) * SCONDA
 *                   where R is the triangular factor from the QRF of A.
 *                   However, if R is truncated and the numerical rank is
 *                   determined to be strictly smaller than N, SCONDA is
 *                   returned as -1, thus indicating that the smallest
 *                   singular values might be lost.
 *
 *        If full SVD is needed, the following two condition numbers are
 *        useful for the analysis of the algorithm. They are provided for
 *        a developer/implementer who is familiar with the details of
 *        the method.
 *
 *        - work[3] = an estimate of the scaled condition number of the
 *                   triangular factor in the first QR factorization.
 *        - work[4] = an estimate of the scaled condition number of the
 *                   triangular factor in the second QR factorization.
 *
 *        The following two parameters are computed if JOBT = 'T'.
 *        They are provided for a developer/implementer who is familiar
 *        with the details of the method.
 *
 *        - work[5] = the entropy of A^t*A :: this is the Shannon entropy
 *                   of diag(A^t*A) / Trace(A^t*A) taken as point in the
 *                   probability simplex.
 *        - work[6] = the entropy of A*A^t.
 *
 * @param[in] lwork
 *        Length of work to confirm proper allocation of work space.
 *        LWORK depends on the job:
 *
 *        If only SIGMA is needed (JOBU = 'N', JOBV = 'N') and
 *          - no scaled condition estimate required (JOBA != 'E','G'):
 *            lwork >= max(2*m+n, 4*n+1, 7). This is the minimal requirement.
 *            For optimal performance (blocked code) the optimal value
 *            is lwork >= max(2*m+n, 3*n+(n+1)*NB, 7). Here NB is the optimal
 *            block size for DGEQP3 and DGEQRF.
 *          - an estimate of the scaled condition number of A is
 *            required (JOBA='E', 'G'). In this case, lwork is the maximum
 *            of the above and n*n+4*n, i.e. lwork >= max(2*m+n, n*n+4*n, 7).
 *
 *        If SIGMA and the right singular vectors are needed (JOBV = 'V'):
 *            lwork >= max(2*m+n, 4*n+1, 7).
 *
 *        If SIGMA and the left singular vectors are needed:
 *            lwork >= max(2*m+n, 4*n+1, 7).
 *            if JOBU = 'F': lwork >= max(2*m+n, 3*n+(n+1)*NB, n+m*NB, 7).
 *
 *        If the full SVD is needed: (JOBU = 'U' or JOBU = 'F') and
 *          - if JOBV = 'V': lwork >= max(2*m+n, 6*n+2*n*n).
 *          - if JOBV = 'J': lwork >= max(2*m+n, 4*n+n*n, 2*n+n*n+6).
 *
 * @param[out] iwork
 *        Integer array, dimension (max(3, m+3*n)).
 *        On exit:
 *        - iwork[0] = the numerical rank determined after the initial
 *                    QR factorization with pivoting. See the descriptions
 *                    of JOBA and JOBR.
 *        - iwork[1] = the number of the computed nonzero singular values
 *        - iwork[2] = if nonzero, a warning message:
 *                    If iwork[2] = 1 then some of the column norms of A
 *                    were denormalized floats. The requested high accuracy
 *                    is not warranted by the data.
 *
 * @param[out] info
 *                           - < 0: if info = -i, then the i-th argument had an illegal value.
 *                           - = 0: successful exit.
 *                           - > 0: DGEJSV did not converge in the maximal allowed number
 *                           of sweeps. The computed values may be inaccurate.
 * @par Further Details:
 *
 * DGEJSV implements a preconditioned Jacobi SVD algorithm. It uses DGEQP3,
 * DGEQRF, and DGELQF as preprocessors and preconditioners. Optionally, an
 * additional row pivoting can be used as a preprocessor, which in some
 * cases results in much higher accuracy. An example is matrix A with the
 * structure A = D1 * C * D2, where D1, D2 are arbitrarily ill-conditioned
 * diagonal matrices and C is well-conditioned matrix. In that case, complete
 * pivoting in the first QR factorizations provides accuracy dependent on the
 * condition number of C, and independent of D1, D2. Such higher accuracy is
 * not completely understood theoretically, but it works well in practice.
 * Further, if A can be written as A = B*D, with well-conditioned B and some
 * diagonal D, then the high accuracy is guaranteed, both theoretically and
 * in software, independent of D. For more details see [1], [2].
 *
 * The computational range for the singular values can be the full range
 * (UNDERFLOW, OVERFLOW), provided that the machine arithmetic and the BLAS
 * & LAPACK routines called by DGEJSV are implemented to work in that range.
 * If that is not the case, then the restriction for safe computation with
 * the singular values in the range of normalized IEEE numbers is that the
 * spectral condition number kappa(A)=sigma_max(A)/sigma_min(A) does not
 * overflow. This code (DGEJSV) is best used in this restricted range,
 * meaning that singular values of magnitude below ||A||_2 / DLAMCH('O') are
 * returned as zeros. See JOBR for details on this.
 *
 * Further, this implementation is somewhat slower than the one described
 * in [1,2] due to replacement of some non-LAPACK components, and because
 * the choice of some tuning parameters in the iterative part (DGESVJ) is
 * left to the implementer on a particular machine.
 *
 * The rank revealing QR factorization (in this code: DGEQP3) should be
 * implemented as in [3]. We have a new version of DGEQP3 under development
 * that is more robust than the current one in LAPACK, with a cleaner cut in
 * rank deficient cases. It will be available in the SIGMA library [4].
 * If M is much larger than N, it is obvious that the initial QRF with
 * column pivoting can be preprocessed by the QRF without pivoting. That
 * well known trick is not used in DGEJSV because in some cases heavy row
 * weighting can be treated with complete pivoting. The overhead in cases
 * M much larger than N is then only due to pivoting, but the benefits in
 * terms of accuracy have prevailed. The implementer/user can incorporate
 * this extra QRF step easily. The implementer can also improve data movement
 * (matrix transpose, matrix copy, matrix transposed copy) - this
 * implementation of DGEJSV uses only the simplest, naive data movement.
 *
 * @par Contributors:
 * Zlatko Drmac (Zagreb, Croatia) and Kresimir Veselic (Hagen, Germany)
 *
 * @par References:
 * [1] Z. Drmac and K. Veselic: New fast and accurate Jacobi SVD algorithm I.
 *     SIAM J. Matrix Anal. Appl. Vol. 35, No. 2 (2008), pp. 1322-1342.
 *     LAPACK Working note 169.
 *
 * [2] Z. Drmac and K. Veselic: New fast and accurate Jacobi SVD algorithm II.
 *     SIAM J. Matrix Anal. Appl. Vol. 35, No. 2 (2008), pp. 1343-1362.
 *     LAPACK Working note 170.
 *
 * [3] Z. Drmac and Z. Bujanovic: On the failure of rank-revealing QR
 *     factorization software - a case study.
 *     ACM Trans. Math. Softw. Vol. 35, No 2 (2008), pp. 1-28.
 *     LAPACK Working note 176.
 *
 * [4] Z. Drmac: SIGMA - mathematical software library for accurate SVD, PSV,
 *     QSVD, (H,K)-SVD computations.
 *     Department of Mathematics, University of Zagreb, 2008.
 *
 * @par Bugs, examples and comments:
 * Please report all bugs and send interesting examples and/or comments to
 * drmac@math.hr. Thank you.
 */
void dgejsv(const char* joba, const char* jobu, const char* jobv,
            const char* jobr, const char* jobt, const char* jobp,
            const INT m, const INT n,
            f64* restrict A, const INT lda,
            f64* restrict SVA,
            f64* restrict U, const INT ldu,
            f64* restrict V, const INT ldv,
            f64* restrict work, const INT lwork,
            INT* restrict iwork, INT* info)
{
    f64 aapp, aaqq, aatmax, aatmin, big, big1, cond_ok;
    f64 condr1, condr2, entra, entrat, epsln, maxprj, scalem;
    f64 sconda, sfmin, small, temp1, uscal1, uscal2, xsc, tbig;
    INT ierr, n1, nr, numrank, p, q, warning;
    INT almort, defr, errest, goscal, jracc, kill, lsvec;
    INT l2aber, l2kill, l2pert, l2rank, l2tran;
    INT noscal, rowpiv, rsvec, transp;

    /* Test the input arguments */

    lsvec  = (jobu[0] == 'U' || jobu[0] == 'u' || jobu[0] == 'F' || jobu[0] == 'f');
    jracc  = (jobv[0] == 'J' || jobv[0] == 'j');
    rsvec  = (jobv[0] == 'V' || jobv[0] == 'v') || jracc;
    rowpiv = (joba[0] == 'F' || joba[0] == 'f' || joba[0] == 'G' || joba[0] == 'g');
    l2rank = (joba[0] == 'R' || joba[0] == 'r');
    l2aber = (joba[0] == 'A' || joba[0] == 'a');
    errest = (joba[0] == 'E' || joba[0] == 'e' || joba[0] == 'G' || joba[0] == 'g');
    l2tran = (jobt[0] == 'T' || jobt[0] == 't');
    l2kill = (jobr[0] == 'R' || jobr[0] == 'r');
    defr   = (jobr[0] == 'N' || jobr[0] == 'n');
    l2pert = (jobp[0] == 'P' || jobp[0] == 'p');

    *info = 0;

    if (!(rowpiv || l2rank || l2aber || errest ||
          joba[0] == 'C' || joba[0] == 'c')) {
        *info = -1;
    }
    else if (!(lsvec || jobu[0] == 'N' || jobu[0] == 'n' ||
                        jobu[0] == 'W' || jobu[0] == 'w')) {
        *info = -2;
    }
    else if (!(rsvec || jobv[0] == 'N' || jobv[0] == 'n' ||
                        jobv[0] == 'W' || jobv[0] == 'w') ||
             (jracc && !lsvec)) {
        *info = -3;
    }
    else if (!(l2kill || defr)) {
        *info = -4;
    }
    else if (!(l2tran || jobt[0] == 'N' || jobt[0] == 'n')) {
        *info = -5;
    }
    else if (!(l2pert || jobp[0] == 'N' || jobp[0] == 'n')) {
        *info = -6;
    }
    else if (m < 0) {
        *info = -7;
    }
    else if (n < 0 || n > m) {
        *info = -8;
    }
    else if (lda < m) {
        *info = -10;
    }
    else if (lsvec && ldu < m) {
        *info = -13;
    }
    else if (rsvec && ldv < n) {
        *info = -15;
    }
    else if (
        (!lsvec && !rsvec && !errest && lwork < max3i(7, 4*n+1, 2*m+n)) ||
        (!lsvec && !rsvec && errest && lwork < max3i(7, 4*n+n*n, 2*m+n)) ||
        (lsvec && !rsvec && lwork < max3i(7, 2*m+n, 4*n+1)) ||
        (rsvec && !lsvec && lwork < max3i(7, 2*m+n, 4*n+1)) ||
        (lsvec && rsvec && !jracc && lwork < (2*m+n > 6*n+2*n*n ? 2*m+n : 6*n+2*n*n)) ||
        (lsvec && rsvec && jracc && lwork < max3i(2*m+n, 4*n+n*n, 2*n+n*n+6))
    ) {
        *info = -17;
    }

    if (*info != 0) {
        xerbla("DGEJSV", -(*info));
        return;
    }

    /* Quick return for void matrix */
    if (m == 0 || n == 0) {
        iwork[0] = 0;
        iwork[1] = 0;
        iwork[2] = 0;
        for (p = 0; p < 7; p++) work[p] = ZERO;
        return;
    }

    /* Determine whether the matrix U should be M x N or M x M */
    n1 = n;
    if (lsvec && (jobu[0] == 'F' || jobu[0] == 'f')) {
        n1 = m;
    }

    /* Set numerical parameters */
    epsln = dlamch("E");
    sfmin = dlamch("S");
    small = sfmin / epsln;
    big   = dlamch("O");

    /* Initialize SVA(1:N) = diag( ||A e_i||_2 )_1^N */

    scalem = ONE / sqrt((f64)m * (f64)n);
    noscal = 1;
    goscal = 1;

    for (p = 0; p < n; p++) {
        aapp = ZERO;
        aaqq = ONE;
        dlassq(m, &A[p * lda], 1, &aapp, &aaqq);
        if (aapp > big) {
            *info = -9;
            xerbla("DGEJSV", -(*info));
            return;
        }
        aaqq = sqrt(aaqq);
        tbig = big;
        if (aaqq > ONE) tbig = big / aaqq;
        if (aapp < tbig && noscal) {
            SVA[p] = aapp * aaqq;
        } else {
            noscal = 0;
            SVA[p] = aapp * (aaqq * scalem);
            if (goscal) {
                goscal = 0;
                cblas_dscal(p, scalem, SVA, 1);
            }
        }
    }

    if (noscal) scalem = ONE;

    aapp = ZERO;
    aaqq = big;
    for (p = 0; p < n; p++) {
        aapp = (aapp > SVA[p]) ? aapp : SVA[p];
        if (SVA[p] != ZERO) aaqq = (aaqq < SVA[p]) ? aaqq : SVA[p];
    }

    /* Quick return for zero M x N matrix */
    if (aapp == ZERO) {
        if (lsvec) dlaset("G", m, n1, ZERO, ONE, U, ldu);
        if (rsvec) dlaset("G", n, n, ZERO, ONE, V, ldv);
        work[0] = ONE;
        work[1] = ONE;
        if (errest) work[2] = ONE;
        if (lsvec && rsvec) {
            work[3] = ONE;
            work[4] = ONE;
        }
        if (l2tran) {
            work[5] = ZERO;
            work[6] = ZERO;
        }
        iwork[0] = 0;
        iwork[1] = 0;
        iwork[2] = 0;
        return;
    }

    /* Issue warning if denormalized column norms detected. Override the
     * high relative accuracy request. Issue licence to kill columns
     * (set them to zero) whose norm is less than sigma_max / BIG (roughly). */
    warning = 0;
    if (aaqq <= sfmin) {
        l2rank = 1;
        l2kill = 1;
        warning = 1;
    }

    /* Quick return for one-column matrix */
    if (n == 1) {
        if (lsvec) {
            dlascl("G", 0, 0, SVA[0], scalem, m, 1, A, lda, &ierr);
            dlacpy("A", m, 1, A, lda, U, ldu);
            /* computing all M left singular vectors of the M x 1 matrix */
            if (n1 != n) {
                dgeqrf(m, n, U, ldu, work, &work[n], lwork - n, &ierr);
                dorgqr(m, n1, 1, U, ldu, work, &work[n], lwork - n, &ierr);
                cblas_dcopy(m, A, 1, U, 1);
            }
        }
        if (rsvec) {
            V[0] = ONE;
        }
        tbig = big;
        if (scalem == ZERO) scalem = ONE;
        if (scalem < ONE) tbig = big * scalem;
        if (SVA[0] < tbig) {
            SVA[0] = SVA[0] / scalem;
            scalem = ONE;
        }
        work[0] = ONE / scalem;
        work[1] = ONE;
        if (SVA[0] != ZERO) {
            iwork[0] = 1;
            if ((SVA[0] / scalem) >= sfmin) {
                iwork[1] = 1;
            } else {
                iwork[1] = 0;
            }
        } else {
            iwork[0] = 0;
            iwork[1] = 0;
        }
        iwork[2] = 0;
        if (errest) work[2] = ONE;
        if (lsvec && rsvec) {
            work[3] = ONE;
            work[4] = ONE;
        }
        if (l2tran) {
            work[5] = ZERO;
            work[6] = ZERO;
        }
        return;
    }

    transp = 0;
    l2tran = l2tran && (m == n);

    aatmax = -ONE;
    aatmin = big;

    if (rowpiv || l2tran) {
        /* Compute the row norms, needed to determine row pivoting sequence
         * (in the case of heavily row weighted A, row pivoting is strongly
         * advised) and to collect information needed to compare the
         * structures of A * A^t and A^t * A (in the case L2TRAN.EQ..TRUE.). */
        if (l2tran) {
            for (p = 0; p < m; p++) {
                xsc = ZERO;
                temp1 = ONE;
                dlassq(n, &A[p], lda, &xsc, &temp1);
                /* DLASSQ gets both the ell_2 and the ell_infinity norm
                 * in one pass through the vector */
                work[m + n + p] = xsc * scalem;
                work[n + p] = xsc * (scalem * sqrt(temp1));
                aatmax = (aatmax > work[n + p]) ? aatmax : work[n + p];
                if (work[n + p] != ZERO) aatmin = (aatmin < work[n + p]) ? aatmin : work[n + p];
            }
        } else {
            for (p = 0; p < m; p++) {
                work[m + n + p] = scalem * fabs(A[p + cblas_idamax(n, &A[p], lda) * lda]);
                aatmax = (aatmax > work[m + n + p]) ? aatmax : work[m + n + p];
                aatmin = (aatmin < work[m + n + p]) ? aatmin : work[m + n + p];
            }
        }
    }

    /* For square matrix A try to determine whether A^t  would be  better
     * input for the preconditioned Jacobi SVD, with faster convergence.
     * The decision is based on an O(N) function of the vector of column
     * and row norms of A, based on the Shannon entropy. This should give
     * the right choice in most cases when the difference actually matters.
     * It may fail and pick the slower converging side. */
    entra = ZERO;
    entrat = ZERO;

    if (l2tran) {
        xsc = ZERO;
        temp1 = ONE;
        dlassq(n, SVA, 1, &xsc, &temp1);
        temp1 = ONE / temp1;

        entra = ZERO;
        for (p = 0; p < n; p++) {
            big1 = ((SVA[p] / xsc) * (SVA[p] / xsc)) * temp1;
            if (big1 != ZERO) entra = entra + big1 * log(big1);
        }
        entra = -entra / log((f64)n);

        /* Now, SVA().^2/Trace(A^t * A) is a point in the probability simplex.
         * It is derived from the diagonal of  A^t * A.  Do the same with the
         * diagonal of A * A^t, compute the entropy of the corresponding
         * probability distribution. Note that A * A^t and A^t * A have the
         * same trace. */
        entrat = ZERO;
        for (p = 0; p < m; p++) {
            big1 = ((work[n + p] / xsc) * (work[n + p] / xsc)) * temp1;
            if (big1 != ZERO) entrat = entrat + big1 * log(big1);
        }
        entrat = -entrat / log((f64)m);

        /* Analyze the entropies and decide A or A^t. Smaller entropy
         * usually means better input for the algorithm. */
        transp = (entrat < entra);

        if (transp) {
            /* In an optimal implementation, this trivial transpose
             * should be replaced with faster transpose. */
            for (p = 0; p < n - 1; p++) {
                for (q = p + 1; q < n; q++) {
                    temp1 = A[q + p * lda];
                    A[q + p * lda] = A[p + q * lda];
                    A[p + q * lda] = temp1;
                }
            }
            for (p = 0; p < n; p++) {
                work[m + n + p] = SVA[p];
                SVA[p] = work[n + p];
            }
            temp1 = aapp;
            aapp = aatmax;
            aatmax = temp1;
            temp1 = aaqq;
            aaqq = aatmin;
            aatmin = temp1;
            kill = lsvec;
            lsvec = rsvec;
            rsvec = kill;
            if (lsvec) n1 = n;
            rowpiv = 1;
        }
    }

    /* Scale the matrix so that its maximal singular value remains less
     * than DSQRT(BIG) -- the matrix is scaled so that its maximal column
     * has Euclidean norm equal to DSQRT(BIG/N). The only reason to keep
     * DSQRT(BIG) instead of BIG is the fact that DGEJSV uses LAPACK and
     * BLAS routines that, in some implementations, are not capable of
     * working in the full interval [SFMIN,BIG] and that they may provoke
     * overflows in the intermediate results. If the singular values spread
     * from SFMIN to BIG, then DGESVJ will compute them. So, in that case,
     * one should use DGESVJ instead of DGEJSV. */

    big1 = sqrt(big);
    temp1 = sqrt(big / (f64)n);

    dlascl("G", 0, 0, aapp, temp1, n, 1, SVA, n, &ierr);
    if (aaqq > aapp * sfmin) {
        aaqq = (aaqq / aapp) * temp1;
    } else {
        aaqq = (aaqq * temp1) / aapp;
    }

    temp1 = temp1 * scalem;
    dlascl("G", 0, 0, aapp, temp1, m, n, A, lda, &ierr);

    /* To undo scaling at the end of this procedure, multiply the
     * computed singular values with USCAL2 / USCAL1. */
    uscal1 = temp1;
    uscal2 = aapp;

    if (l2kill) {
        xsc = sqrt(sfmin);
    } else {
        xsc = small;
        /* Now, if the condition number of A is too big,
         * sigma_max(A) / sigma_min(A) .GT. DSQRT(BIG/N) * EPSLN / SFMIN,
         * as a precaution measure, the full SVD is computed using DGESVJ
         * with accumulated Jacobi rotations. This provides numerically
         * more robust computation, at the cost of slightly increased run
         * time. Depending on the concrete implementation of BLAS and LAPACK
         * (i.e. how they behave in presence of extreme ill-conditioning) the
         * implementor may decide to remove this switch. */
        if (aaqq < sqrt(sfmin) && lsvec && rsvec) {
            jracc = 1;
        }
    }

    if (aaqq < xsc) {
        for (p = 0; p < n; p++) {
            if (SVA[p] < xsc) {
                dlaset("A", m, 1, ZERO, ZERO, &A[p * lda], lda);
                SVA[p] = ZERO;
            }
        }
    }

    /* Preconditioning using QR factorization with pivoting */

    if (rowpiv) {
        /* Optional row permutation (Bjoerck row pivoting):
         * A result by Cox and Higham shows that the Bjoerck's
         * row pivoting combined with standard column pivoting
         * has similar effect as Powell-Reid complete pivoting.
         * The ell-infinity norms of A are made nonincreasing. */
        for (p = 0; p < m - 1; p++) {
            q = cblas_idamax(m - p, &work[m + n + p], 1) + p;
            iwork[2 * n + p] = q;
            if (p != q) {
                temp1 = work[m + n + p];
                work[m + n + p] = work[m + n + q];
                work[m + n + q] = temp1;
            }
        }
        dlaswp(n, A, lda, 0, m - 2, &iwork[2 * n], 1);
    }

    /* End of the preparation phase (scaling, optional sorting and
     * transposing, optional flushing of small columns).
     *
     * Preconditioning
     *
     * If the full SVD is needed, the right singular vectors are computed
     * from a matrix equation, and for that we need theoretical analysis
     * of the Businger-Golub pivoting. So we use DGEQP3 as the first RR QRF.
     * In all other cases the first RR QRF can be chosen by other criteria
     * (eg speed by replacing global with restricted window pivoting, such
     * as in SGEQPX from TOMS # 782). Good results will be obtained using
     * SGEQPX with properly (!) chosen numerical parameters.
     * Any improvement of DGEQP3 improves overall performance of DGEJSV.
     *
     * A * P1 = Q1 * [ R1^t 0]^t: */
    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }

    dgeqp3(m, n, A, lda, iwork, work, &work[n], lwork - n, &ierr);

    /* The upper triangular matrix R1 from the first QRF is inspected for
     * rank deficiency and possibilities for deflation, or possible
     * ill-conditioning. Depending on the user specified flag L2RANK,
     * the procedure explores possibilities to reduce the numerical
     * rank by inspecting the computed upper triangular factor. If
     * L2RANK or L2ABER are up, then DGEJSV will compute the SVD of
     * A + dA, where ||dA|| <= f(M,N)*EPSLN. */
    nr = 1;

    if (l2aber) {
        /* Standard absolute error bound suffices. All sigma_i with
         * sigma_i < N*EPSLN*||A|| are flushed to zero. This is an
         * aggressive enforcement of lower numerical rank by introducing a
         * backward error of the order of N*EPSLN*||A||. */
        temp1 = sqrt((f64)n) * epsln;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) >= temp1 * fabs(A[0])) {
                nr++;
            } else {
                break;
            }
        }
    } else if (l2rank) {
        /* .. similarly as above, only slightly more gentle (less aggressive).
         * Sudden drop on the diagonal of R1 is used as the criterion for
         * close-to-rank-deficient. */
        temp1 = sqrt(sfmin);
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < epsln * fabs(A[(p-1) + (p-1) * lda]) ||
                fabs(A[p + p * lda]) < small ||
                (l2kill && fabs(A[p + p * lda]) < temp1)) {
                break;
            }
            nr++;
        }
    } else {
        /* The goal is high relative accuracy. However, if the matrix
         * has high scaled condition number the relative accuracy is in
         * general not feasible. Later on, a condition number estimator
         * will be deployed to estimate the scaled condition number.
         * Here we just remove the underflowed part of the triangular
         * factor. This prevents the situation in which the code is
         * working hard to get the accuracy not warranted by the data. */
        temp1 = sqrt(sfmin);
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < small ||
                (l2kill && fabs(A[p + p * lda]) < temp1)) {
                break;
            }
            nr++;
        }
    }

    almort = 0;
    if (nr == n) {
        maxprj = ONE;
        for (p = 1; p < n; p++) {
            temp1 = fabs(A[p + p * lda]) / SVA[iwork[p]];
            maxprj = (maxprj < temp1) ? maxprj : temp1;
        }
        if (maxprj * maxprj >= ONE - (f64)n * epsln) {
            almort = 1;
        }
    }

    sconda = -ONE;
    condr1 = -ONE;
    condr2 = -ONE;

    if (errest && nr == n) {
        if (rsvec) {
            /* .. V is available as workspace */
            dlacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < n; p++) {
                temp1 = SVA[iwork[p]];
                cblas_dscal(p + 1, ONE / temp1, &V[p * ldv], 1);
            }
            dpocon("U", n, V, ldv, ONE, &temp1, &work[n], &iwork[2*n + m], &ierr);
        } else if (lsvec) {
            /* .. U is available as workspace */
            dlacpy("U", n, n, A, lda, U, ldu);
            for (p = 0; p < n; p++) {
                temp1 = SVA[iwork[p]];
                cblas_dscal(p + 1, ONE / temp1, &U[p * ldu], 1);
            }
            dpocon("U", n, U, ldu, ONE, &temp1, &work[n], &iwork[2*n + m], &ierr);
        } else {
            dlacpy("U", n, n, A, lda, &work[n], n);
            for (p = 0; p < n; p++) {
                temp1 = SVA[iwork[p]];
                cblas_dscal(p + 1, ONE / temp1, &work[n + p * n], 1);
            }
            dpocon("U", n, &work[n], n, ONE, &temp1, &work[n + n*n], &iwork[2*n + m], &ierr);
        }
        sconda = ONE / sqrt(temp1);
    }

    /* If there is no violent scaling, artificial perturbation is not needed */
    l2pert = l2pert && (fabs(A[0] / A[(nr-1) + (nr-1) * lda]) > sqrt(big1));

    /* Phase 3: */


    if (!rsvec && !lsvec) {
        /* Singular Values only */

        /* .. transpose A(1:NR,1:N) */
        for (p = 0; p < (n - 1 < nr ? n - 1 : nr); p++) {
            cblas_dcopy(n - p - 1, &A[p + (p + 1) * lda], lda, &A[(p + 1) + p * lda], 1);
        }

        /* The following two DO-loops introduce small relative perturbation
         * into the strict upper triangle of the lower triangular matrix.
         * Small entries below the main diagonal are also changed.
         * This modification is useful if the computing environment does not
         * provide/allow FLUSH TO ZERO underflow, for it prevents many
         * annoying denormalized numbers in case of strongly scaled matrices.
         * The perturbation is structured so that it does not introduce any
         * new perturbation of the singular values, and it does not destroy
         * the job done by the preconditioner.
         * The licence for this perturbation is in the variable L2PERT, which
         * should be .FALSE. if FLUSH TO ZERO underflow is active. */
        if (!almort) {
            if (l2pert) {
                xsc = epsln / (f64)n;
                for (q = 0; q < nr; q++) {
                    temp1 = xsc * fabs(A[q + q * lda]);
                    for (p = 0; p < n; p++) {
                        if ((p > q && fabs(A[p + q * lda]) <= temp1) || p < q) {
                            A[p + q * lda] = copysign(temp1, A[p + q * lda]);
                        }
                    }
                }
            } else {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &A[1 * lda], lda);
            }

            /* .. second preconditioning using the QR factorization */
            dgeqrf(n, nr, A, lda, work, &work[n], lwork - n, &ierr);

            /* .. and transpose upper to lower triangular */
            for (p = 0; p < nr - 1; p++) {
                cblas_dcopy(nr - p - 1, &A[p + (p + 1) * lda], lda, &A[(p + 1) + p * lda], 1);
            }
        }

        /* Row-cyclic Jacobi SVD algorithm with column pivoting
         *
         * .. again some perturbation (a "background noise") is added
         * to drown denormals */
        if (l2pert) {
            xsc = epsln / (f64)n;
            for (q = 0; q < nr; q++) {
                temp1 = xsc * fabs(A[q + q * lda]);
                for (p = 0; p < nr; p++) {
                    if ((p > q && fabs(A[p + q * lda]) <= temp1) || p < q) {
                        A[p + q * lda] = copysign(temp1, A[p + q * lda]);
                    }
                }
            }
        } else {
            dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &A[1 * lda], lda);
        }

        /* .. and one-sided Jacobi rotations are started on a lower
         * triangular matrix (plus perturbation which is ignored in
         * the part which destroys triangular form (confusing?!)) */
        dgesvj("L", "N", "N", nr, nr, A, lda, SVA, n, V, ldv,
               work, lwork, info);
        scalem = work[0];
        numrank = (INT)(work[1] + 0.5);

    } else if (rsvec && !lsvec) {
        /* -> Singular Values and Right Singular Vectors <- */

        if (almort) {
            /* .. in this case NR equals N */
            for (p = 0; p < nr; p++) {
                cblas_dcopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
            }
            dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);

            dgesvj("L", "U", "N", n, nr, V, ldv, SVA, nr, A, lda,
                   work, lwork, info);
            scalem = work[0];
            numrank = (INT)(work[1] + 0.5);

        } else {
            /* .. two more QR factorizations ( one QRF is not enough, two require
             * accumulated product of Jacobi rotations, three are perfect ) */
            dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &A[1], lda);
            dgelqf(nr, n, A, lda, work, &work[n], lwork - n, &ierr);
            dlacpy("L", nr, nr, A, lda, V, ldv);
            dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
            dgeqrf(nr, nr, V, ldv, &work[n], &work[2 * n], lwork - 2 * n, &ierr);
            for (p = 0; p < nr; p++) {
                cblas_dcopy(nr - p, &V[p + p * ldv], ldv, &V[p + p * ldv], 1);
            }
            dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);

            dgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                   &work[n], lwork - n, info);
            scalem = work[n];
            numrank = (INT)(work[n + 1] + 0.5);

            if (nr < n) {
                dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
            }

            dormlq("L", "T", n, n, nr, A, lda, work, V, ldv, &work[n], lwork - n, &ierr);
        }

        for (p = 0; p < n; p++) {
            cblas_dcopy(n, &V[p], ldv, &A[iwork[p]], lda);
        }
        dlacpy("A", n, n, A, lda, V, ldv);

        if (transp) {
            dlacpy("A", n, n, V, ldv, U, ldu);
        }

    } else if (lsvec && !rsvec) {
        /* .. Singular Values and Left Singular Vectors .. */

        /* .. second preconditioning step to avoid need to accumulate
         * Jacobi rotations in the Jacobi iterations. */
        for (p = 0; p < nr; p++) {
            cblas_dcopy(n - p, &A[p + p * lda], lda, &U[p + p * ldu], 1);
        }
        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[1 * ldu], ldu);

        dgeqrf(n, nr, U, ldu, &work[n], &work[2 * n], lwork - 2 * n, &ierr);

        for (p = 0; p < nr - 1; p++) {
            cblas_dcopy(nr - p - 1, &U[p + (p + 1) * ldu], ldu, &U[(p + 1) + p * ldu], 1);
        }
        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[1 * ldu], ldu);

        dgesvj("L", "U", "N", nr, nr, U, ldu, SVA, nr, A, lda,
               &work[n], lwork - n, info);
        scalem = work[n];
        numrank = (INT)(work[n + 1] + 0.5);

        if (nr < m) {
            dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
            if (nr < n1) {
                dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
            }
        }

        dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

        if (rowpiv) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
        }

        for (p = 0; p < n1; p++) {
            xsc = ONE / cblas_dnrm2(m, &U[p * ldu], 1);
            cblas_dscal(m, xsc, &U[p * ldu], 1);
        }

        if (transp) {
            dlacpy("A", n, n, U, ldu, V, ldv);
        }

    } else {
        /* .. Full SVD .. */

        if (!jracc) {
            if (!almort) {
                /* Second Preconditioning Step (QRF [with pivoting]) */
                for (p = 0; p < nr; p++) {
                    cblas_dcopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
                }

                /* .. the following two loops perturb small entries to avoid
                 * denormals in the second QR factorization, where they are
                 * as good as zeros. This is done to avoid painfully slow
                 * computation with denormals. The relative size of the perturbation
                 * is a parameter that can be changed by the implementer.
                 * This perturbation device will be obsolete on machines with
                 * properly implemented arithmetic.
                 * To switch it off, set L2PERT=.FALSE. To remove it from  the
                 * code, remove the action under L2PERT=.TRUE., leave the ELSE part.
                 * The following two loops should be blocked and fused with the
                 * transposed copy above. */
                if (l2pert) {
                    xsc = sqrt(small);
                    for (q = 0; q < nr; q++) {
                        temp1 = xsc * fabs(V[q + q * ldv]);
                        for (p = 0; p < n; p++) {
                            if ((p > q && fabs(V[p + q * ldv]) <= temp1) || p < q) {
                                V[p + q * ldv] = copysign(temp1, V[p + q * ldv]);
                            }
                            if (p < q) V[p + q * ldv] = -V[p + q * ldv];
                        }
                    }
                } else {
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
                }

                /* Estimate the row scaled condition number of R1 */
                dlacpy("L", nr, nr, V, ldv, &work[2 * n], nr);
                for (p = 0; p < nr; p++) {
                    temp1 = cblas_dnrm2(nr - p, &work[2 * n + p * nr + p], 1);
                    cblas_dscal(nr - p, ONE / temp1, &work[2 * n + p * nr + p], 1);
                }
                dpocon("L", nr, &work[2 * n], nr, ONE, &temp1,
                       &work[2 * n + nr * nr], &iwork[m + 2 * n], &ierr);
                condr1 = ONE / sqrt(temp1);

                cond_ok = sqrt((f64)nr);

                if (condr1 < cond_ok) {
                    /* .. the second QRF without pivoting. */
                    dgeqrf(n, nr, V, ldv, &work[n], &work[2 * n], lwork - 2 * n, &ierr);

                    if (l2pert) {
                        xsc = sqrt(small) / epsln;
                        for (p = 1; p < nr; p++) {
                            for (q = 0; q < p; q++) {
                                temp1 = xsc * ((fabs(V[p + p * ldv]) < fabs(V[q + q * ldv]))
                                    ? fabs(V[p + p * ldv]) : fabs(V[q + q * ldv]));
                                if (fabs(V[q + p * ldv]) <= temp1) {
                                    V[q + p * ldv] = copysign(temp1, V[q + p * ldv]);
                                }
                            }
                        }
                    }

                    if (nr != n) {
                        dlacpy("A", n, nr, V, ldv, &work[2 * n], n);
                    }

                    /* .. this transposed copy should be better than naive */
                    for (p = 0; p < nr - 1; p++) {
                        cblas_dcopy(nr - p - 1, &V[p + (p + 1) * ldv], ldv, &V[(p + 1) + p * ldv], 1);
                    }

                    condr2 = condr1;

                } else {
                    /* .. ill-conditioned case: second QRF with pivoting */
                    for (p = 0; p < nr; p++) {
                        iwork[n + p] = 0;
                    }
                    dgeqp3(n, nr, V, ldv, &iwork[n], &work[n], &work[2 * n], lwork - 2 * n, &ierr);

                    if (l2pert) {
                        xsc = sqrt(small);
                        for (p = 1; p < nr; p++) {
                            for (q = 0; q < p; q++) {
                                temp1 = xsc * ((fabs(V[p + p * ldv]) < fabs(V[q + q * ldv]))
                                    ? fabs(V[p + p * ldv]) : fabs(V[q + q * ldv]));
                                if (fabs(V[q + p * ldv]) <= temp1) {
                                    V[q + p * ldv] = copysign(temp1, V[q + p * ldv]);
                                }
                            }
                        }
                    }

                    dlacpy("A", n, nr, V, ldv, &work[2 * n], n);

                    if (l2pert) {
                        xsc = sqrt(small);
                        for (p = 1; p < nr; p++) {
                            for (q = 0; q < p; q++) {
                                temp1 = xsc * ((fabs(V[p + p * ldv]) < fabs(V[q + q * ldv]))
                                    ? fabs(V[p + p * ldv]) : fabs(V[q + q * ldv]));
                                V[p + q * ldv] = -copysign(temp1, V[q + p * ldv]);
                            }
                        }
                    } else {
                        dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                    }

                    /* Now, compute R2 = L3 * Q3, the LQ factorization. */
                    dgelqf(nr, nr, V, ldv, &work[2 * n + n * nr],
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                    /* .. and estimate the condition number */
                    dlacpy("L", nr, nr, V, ldv, &work[2 * n + n * nr + nr], nr);
                    for (p = 0; p < nr; p++) {
                        temp1 = cblas_dnrm2(p + 1, &work[2 * n + n * nr + nr + p], nr);
                        cblas_dscal(p + 1, ONE / temp1, &work[2 * n + n * nr + nr + p], nr);
                    }
                    dpocon("L", nr, &work[2 * n + n * nr + nr], nr, ONE, &temp1,
                           &work[2 * n + n * nr + nr + nr * nr], &iwork[m + 2 * n], &ierr);
                    condr2 = ONE / sqrt(temp1);

                    if (condr2 >= cond_ok) {
                        /* .. save the Householder vectors used for Q3 */
                        dlacpy("U", nr, nr, V, ldv, &work[2 * n], n);
                    }
                }

                if (l2pert) {
                    xsc = sqrt(small);
                    for (q = 1; q < nr; q++) {
                        temp1 = xsc * V[q + q * ldv];
                        for (p = 0; p < q; p++) {
                            V[p + q * ldv] = -copysign(temp1, V[p + q * ldv]);
                        }
                    }
                } else {
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
                }

                /* Second preconditioning finished; continue with Jacobi SVD */
                if (condr1 < cond_ok) {
                    dgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, info);
                    scalem = work[2 * n + n * nr + nr];
                    numrank = (INT)(work[2 * n + n * nr + nr + 1] + 0.5);

                    for (p = 0; p < nr; p++) {
                        cblas_dcopy(nr, &V[p * ldv], 1, &U[p * ldu], 1);
                        cblas_dscal(nr, SVA[p], &V[p * ldv], 1);
                    }

                    if (nr == n) {
                        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                    nr, nr, ONE, A, lda, V, ldv);
                    } else {
                        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                                    nr, nr, ONE, &work[2 * n], n, V, ldv);
                        if (nr < n) {
                            dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                            dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                            dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                        }
                        dormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                               V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);
                    }

                } else if (condr2 < cond_ok) {
                    /* :)           .. the input matrix A is very likely a relative of
                     *              the Kahan matrix :)
                     *              The matrix R2 is inverted. The solution of the matrix equation
                     *              is Q3^T*V3 = the product of the Jacobi rotations (applied to
                     *              the lower triangular L3 from the LQ factorization of
                     *              R2=L3*Q3), pre-multiplied with the transposed Q3. */
                    dgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, info);
                    scalem = work[2 * n + n * nr + nr];
                    numrank = (INT)(work[2 * n + n * nr + nr + 1] + 0.5);

                    for (p = 0; p < nr; p++) {
                        cblas_dcopy(nr, &V[p * ldv], 1, &U[p * ldu], 1);
                        cblas_dscal(nr, SVA[p], &U[p * ldu], 1);
                    }

                    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                nr, nr, ONE, &work[2 * n], n, U, ldu);

                    /* .. apply the permutation from the second QR factorization */
                    for (q = 0; q < nr; q++) {
                        for (p = 0; p < nr; p++) {
                            work[2 * n + n * nr + nr + iwork[n + p]] = U[p + q * ldu];
                        }
                        for (p = 0; p < nr; p++) {
                            U[p + q * ldu] = work[2 * n + n * nr + nr + p];
                        }
                    }

                    if (nr < n) {
                        dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                        dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                        dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    }
                    dormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                           V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                } else {
                    /* Last line of defense. */
                    dgesvj("L", "U", "V", nr, nr, V, ldv, SVA, nr, U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, info);
                    scalem = work[2 * n + n * nr + nr];
                    numrank = (INT)(work[2 * n + n * nr + nr + 1] + 0.5);

                    if (nr < n) {
                        dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                        dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                        dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    }
                    dormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                           V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                    dormlq("L", "T", nr, nr, nr, &work[2 * n], n,
                           &work[2 * n + n * nr], U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                    for (q = 0; q < nr; q++) {
                        for (p = 0; p < nr; p++) {
                            work[2 * n + n * nr + nr + iwork[n + p]] = U[p + q * ldu];
                        }
                        for (p = 0; p < nr; p++) {
                            U[p + q * ldu] = work[2 * n + n * nr + nr + p];
                        }
                    }
                }

                /* Permute the rows of V using the (column) permutation from the
                 * first QRF. Also, scale the columns to make them unit in
                 * Euclidean norm. This applies to all cases. */
                temp1 = sqrt((f64)n) * epsln;
                for (q = 0; q < n; q++) {
                    for (p = 0; p < n; p++) {
                        work[2 * n + n * nr + nr + iwork[p]] = V[p + q * ldv];
                    }
                    for (p = 0; p < n; p++) {
                        V[p + q * ldv] = work[2 * n + n * nr + nr + p];
                    }
                    xsc = ONE / cblas_dnrm2(n, &V[q * ldv], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_dscal(n, xsc, &V[q * ldv], 1);
                    }
                }

                /* At this moment, V contains the right singular vectors of A.
                 * Next, assemble the left singular vector matrix U (M x N). */
                if (nr < m) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }

                dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

                /* The columns of U are normalized. */
                temp1 = sqrt((f64)m) * epsln;
                for (p = 0; p < nr; p++) {
                    xsc = ONE / cblas_dnrm2(m, &U[p * ldu], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_dscal(m, xsc, &U[p * ldu], 1);
                    }
                }

                /* If the initial QRF is computed with row pivoting, the left
                 * singular vectors must be adjusted. */
                if (rowpiv) {
                    dlaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
                }

            } else {
                /* .. the initial matrix A has almost orthogonal columns and
                 * the second QRF is not needed */

                dlacpy("U", n, n, A, lda, &work[n], n);

                if (l2pert) {
                    xsc = sqrt(small);
                    for (p = 1; p < n; p++) {
                        temp1 = xsc * work[n + p * n + p];
                        for (q = 0; q < p; q++) {
                            work[n + q * n + p] = -copysign(temp1, work[n + p * n + q]);
                        }
                    }
                } else {
                    dlaset("L", n - 1, n - 1, ZERO, ZERO, &work[n + 1], n);
                }

                dgesvj("U", "U", "N", n, n, &work[n], n, SVA, n, U, ldu,
                       &work[n + n * n], lwork - n - n * n, info);
                scalem = work[n + n * n];
                numrank = (INT)(work[n + n * n + 1] + 0.5);

                for (p = 0; p < n; p++) {
                    cblas_dcopy(n, &work[n + p * n], 1, &U[p * ldu], 1);
                    cblas_dscal(n, SVA[p], &work[n + p * n], 1);
                }

                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            n, n, ONE, A, lda, &work[n], n);

                for (p = 0; p < n; p++) {
                    cblas_dcopy(n, &work[n + p], n, &V[iwork[p]], ldv);
                }

                temp1 = sqrt((f64)n) * epsln;
                for (p = 0; p < n; p++) {
                    xsc = ONE / cblas_dnrm2(n, &V[p * ldv], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_dscal(n, xsc, &V[p * ldv], 1);
                    }
                }

                /* Assemble the left singular vector matrix U (M x N). */
                if (n < m) {
                    dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                    if (n < n1) {
                        dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                        dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                    }
                }

                dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

                temp1 = sqrt((f64)m) * epsln;
                for (p = 0; p < n1; p++) {
                    xsc = ONE / cblas_dnrm2(m, &U[p * ldu], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_dscal(m, xsc, &U[p * ldu], 1);
                    }
                }

                if (rowpiv) {
                    dlaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
                }
            }

        } else {
            /* This branch deploys a preconditioned Jacobi SVD with explicitly
             * accumulated rotations. It is included as optional, mainly for
             * experimental purposes. It does perform well, and can also be used.
             * In this implementation, this branch will be automatically activated
             * if the  condition number sigma_max(A) / sigma_min(A) is predicted
             * to be greater than the overflow threshold. This is because the
             * a posteriori computation of the singular vectors assumes robust
             * implementation of BLAS and some LAPACK procedures, capable of working
             * in presence of extreme values. Since that is not always the case, ... */

            for (p = 0; p < nr; p++) {
                cblas_dcopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
            }

            if (l2pert) {
                xsc = sqrt(small / epsln);
                for (q = 0; q < nr; q++) {
                    temp1 = xsc * fabs(V[q + q * ldv]);
                    for (p = 0; p < n; p++) {
                        if ((p > q && fabs(V[p + q * ldv]) <= temp1) || p < q) {
                            V[p + q * ldv] = copysign(temp1, V[p + q * ldv]);
                        }
                        if (p < q) V[p + q * ldv] = -V[p + q * ldv];
                    }
                }
            } else {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
            }

            dgeqrf(n, nr, V, ldv, &work[n], &work[2 * n], lwork - 2 * n, &ierr);
            dlacpy("L", n, nr, V, ldv, &work[2 * n], n);

            for (p = 0; p < nr; p++) {
                cblas_dcopy(nr - p, &V[p + p * ldv], ldv, &U[p + p * ldu], 1);
            }

            if (l2pert) {
                xsc = sqrt(small / epsln);
                for (q = 1; q < nr; q++) {
                    for (p = 0; p < q; p++) {
                        temp1 = xsc * ((fabs(U[p + p * ldu]) < fabs(U[q + q * ldu]))
                            ? fabs(U[p + p * ldu]) : fabs(U[q + q * ldu]));
                        U[p + q * ldu] = -copysign(temp1, U[q + p * ldu]);
                    }
                }
            } else {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[1 * ldu], ldu);
            }

            dgesvj("G", "U", "V", nr, nr, U, ldu, SVA, n, V, ldv,
                   &work[2 * n + n * nr], lwork - 2 * n - n * nr, info);
            scalem = work[2 * n + n * nr];
            numrank = (INT)(work[2 * n + n * nr + 1] + 0.5);

            if (nr < n) {
                dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
            }

            dormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                   V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

            /* Permute the rows of V using the (column) permutation from the
             * first QRF. Also, scale the columns to make them unit in
             * Euclidean norm. This applies to all cases. */
            temp1 = sqrt((f64)n) * epsln;
            for (q = 0; q < n; q++) {
                for (p = 0; p < n; p++) {
                    work[2 * n + n * nr + nr + iwork[p]] = V[p + q * ldv];
                }
                for (p = 0; p < n; p++) {
                    V[p + q * ldv] = work[2 * n + n * nr + nr + p];
                }
                xsc = ONE / cblas_dnrm2(n, &V[q * ldv], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                    cblas_dscal(n, xsc, &V[q * ldv], 1);
                }
            }

            /* At this moment, V contains the right singular vectors of A.
             * Next, assemble the left singular vector matrix U (M x N). */
            if (nr < m) {
                dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                if (nr < n1) {
                    dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                    dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                }
            }

            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

            if (rowpiv) {
                dlaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
            }
        }

        /* .. swap U and V because the procedure worked on A^t */
        if (transp) {
            for (p = 0; p < n; p++) {
                cblas_dswap(n, &U[p * ldu], 1, &V[p * ldv], 1);
            }
        }
    }

    /* end of the full SVD */

    /* Undo scaling, if necessary (and possible) */
    if (uscal2 <= (big / SVA[0]) * uscal1) {
        dlascl("G", 0, 0, uscal1, uscal2, nr, 1, SVA, n, &ierr);
        uscal1 = ONE;
        uscal2 = ONE;
    }

    for (p = nr; p < n; p++) {
        SVA[p] = ZERO;
    }

    work[0] = uscal2 * scalem;
    work[1] = uscal1;
    if (errest) work[2] = sconda;
    if (lsvec && rsvec) {
        work[3] = condr1;
        work[4] = condr2;
    }
    if (l2tran) {
        work[5] = entra;
        work[6] = entrat;
    }

    iwork[0] = nr;
    iwork[1] = numrank;
    iwork[2] = warning;
}
