/**
 * @file zgesvj.c
 * @brief ZGESVJ computes the SVD of a complex M-by-N matrix using Jacobi rotations.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const double ZERO = 0.0;
static const double HALF = 0.5;
static const double ONE = 1.0;
static const double complex CZERO = CMPLX(0.0, 0.0);
static const double complex CONE = CMPLX(1.0, 0.0);
static const int NSWEEP = 30;

/**
 * ZGESVJ computes the singular value decomposition (SVD) of a complex
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V^*,
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N unitary matrix. The diagonal elements
 * of SIGMA are the singular values of A. The columns of U and V are the
 * left and the right singular vectors of A, respectively.
 * ZGESVJ can sometimes compute tiny singular values and their singular
 * vectors much more accurately than other SVD routines, see below under
 * Further Details.
 *
 * @param[in] joba
 *          Specifies the structure of A.
 *          = 'L': The input matrix A is lower triangular;
 *          = 'U': The input matrix A is upper triangular;
 *          = 'G': The input matrix A is general M-by-N matrix, M >= N.
 *
 * @param[in] jobu
 *          Specifies whether to compute the left singular vectors
 *          (columns of U):
 *          = 'U' or 'F': The left singular vectors corresponding to the nonzero
 *                 singular values are computed and returned in the leading
 *                 columns of A. See more details in the description of A.
 *                 The default numerical orthogonality threshold is set to
 *                 approximately TOL=CTOL*EPS, CTOL=SQRT(M), EPS=DLAMCH('E').
 *          = 'C': Analogous to JOBU='U', except that user can control the
 *                 level of numerical orthogonality of the computed left
 *                 singular vectors. TOL can be set to TOL = CTOL*EPS, where
 *                 CTOL is given on input in the array RWORK.
 *                 No CTOL smaller than ONE is allowed. CTOL greater
 *                 than 1 / EPS is meaningless. The option 'C'
 *                 can be used if M*EPS is satisfactory orthogonality
 *                 of the computed left singular vectors, so CTOL=M could
 *                 save few sweeps of Jacobi rotations.
 *                 See the descriptions of A and RWORK(1).
 *          = 'N': The matrix U is not computed. However, see the
 *                 description of A.
 *
 * @param[in] jobv
 *          Specifies whether to compute the right singular vectors, that
 *          is, the matrix V:
 *          = 'V' or 'J': the matrix V is computed and returned in the array V
 *          = 'A': the Jacobi rotations are applied to the MV-by-N
 *                 array V. In other words, the right singular vector
 *                 matrix V is not computed explicitly; instead it is
 *                 applied to an MV-by-N matrix initially stored in the
 *                 first MV rows of V.
 *          = 'N': the matrix V is not computed and the array V is not
 *                 referenced
 *
 * @param[in] m
 *          The number of rows of the input matrix A. 1/DLAMCH('E') > M >= 0.
 *
 * @param[in] n
 *          The number of columns of the input matrix A.
 *          M >= N >= 0.
 *
 * @param[in,out] A
 *          Complex*16 array, dimension (lda, n).
 *          On entry, the M-by-N matrix A.
 *          On exit:
 *          If JOBU = 'U' or JOBU = 'F' or JOBU = 'C':
 *                 If INFO = 0:
 *                 RANKA orthonormal columns of U are returned in the
 *                 leading RANKA columns of the array A. Here RANKA <= N
 *                 is the number of computed singular values of A that are
 *                 above the underflow threshold DLAMCH('S'). The singular
 *                 vectors corresponding to underflowed or zero singular
 *                 values are not computed. The value of RANKA is returned
 *                 in the array RWORK as RANKA=NINT(RWORK(2)). Also see the
 *                 descriptions of SVA and RWORK. The computed columns of U
 *                 are mutually numerically orthogonal up to approximately
 *                 TOL=SQRT(M)*EPS (default); or TOL=CTOL*EPS (JOBU = 'C'),
 *                 see the description of JOBU.
 *                 If INFO > 0:
 *                 the procedure ZGESVJ did not converge in the given number
 *                 of iterations (sweeps). In that case, the computed
 *                 columns of U may not be orthogonal up to TOL. The output
 *                 U (stored in A), SIGMA (given by the computed singular
 *                 values in SVA(1:N)) and V is still a decomposition of the
 *                 input matrix A in the sense that the residual
 *                 ||A-SCALE*U*SIGMA*V^*||_2 / ||A||_2 is small.
 *          If JOBU = 'N':
 *                 If INFO = 0:
 *                 Note that the left singular vectors are 'for free' in the
 *                 one-sided Jacobi SVD algorithm. However, if only the
 *                 singular values are needed, the level of numerical
 *                 orthogonality of U is not an issue and iterations are
 *                 stopped when the columns of the iterated matrix are
 *                 numerically orthogonal up to approximately M*EPS. Thus,
 *                 on exit, A contains the columns of U scaled with the
 *                 corresponding singular values.
 *                 If INFO > 0:
 *                 the procedure ZGESVJ did not converge in the given number
 *                 of iterations (sweeps).
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] SVA
 *          Double precision array, dimension (n).
 *          On exit:
 *          If INFO = 0:
 *          depending on the value SCALE = RWORK(1), we have:
 *                 If SCALE = ONE:
 *                 SVA(1:N) contains the computed singular values of A.
 *                 During the computation SVA contains the Euclidean column
 *                 norms of the iterated matrices in the array A.
 *                 If SCALE /= ONE:
 *                 The singular values of A are SCALE*SVA(1:N), and this
 *                 factored representation is due to the fact that some of the
 *                 singular values of A might underflow or overflow.
 *          If INFO > 0:
 *          the procedure ZGESVJ did not converge in the given number of
 *          iterations (sweeps) and SCALE*SVA(1:N) may not be accurate.
 *
 * @param[in] mv
 *          If JOBV = 'A', then the product of Jacobi rotations in ZGESVJ
 *          is applied to the first MV rows of V. See the description of JOBV.
 *
 * @param[in,out] V
 *          Complex*16 array, dimension (ldv, n).
 *          If JOBV = 'V' or 'J', then V contains on exit the N-by-N matrix of
 *                         the right singular vectors;
 *          If JOBV = 'A', then V contains the product of the computed right
 *                         singular vector matrix and the initial matrix in
 *                         the array V.
 *          If JOBV = 'N', then V is not referenced.
 *
 * @param[in] ldv
 *          The leading dimension of the array V, ldv >= 1.
 *          If JOBV = 'V' or 'J', then ldv >= max(1, n).
 *          If JOBV = 'A', then ldv >= max(1, mv).
 *
 * @param[in,out] cwork
 *          Complex*16 array, dimension (max(1, lwork)).
 *          Used as workspace.
 *
 * @param[in] lwork
 *          Length of CWORK.
 *          LWORK >= 1, if MIN(M,N) = 0, and LWORK >= M+N, otherwise.
 *          If on entry LWORK = -1, then a workspace query is assumed and
 *          no computation is done; CWORK(1) is set to the minimal (and optimal)
 *          length of CWORK.
 *
 * @param[in,out] rwork
 *          Double precision array, dimension (max(6, lrwork)).
 *          On entry:
 *          If JOBU = 'C':
 *          rwork[0] = CTOL, where CTOL defines the threshold for convergence.
 *                    The process stops if all columns of A are mutually
 *                    orthogonal up to CTOL*EPS, EPS=DLAMCH('E').
 *                    It is required that CTOL >= ONE, i.e. it is not
 *                    allowed to force the routine to obtain orthogonality
 *                    below EPSILON.
 *          On exit:
 *          rwork[0] = SCALE is the scaling factor such that SCALE*SVA(1:N)
 *                    are the computed singular values of A.
 *                    (See description of SVA.)
 *          rwork[1] = NINT(rwork[1]) is the number of the computed nonzero
 *                    singular values.
 *          rwork[2] = NINT(rwork[2]) is the number of the computed singular
 *                    values that are larger than the underflow threshold.
 *          rwork[3] = NINT(rwork[3]) is the number of sweeps of Jacobi
 *                    rotations needed for numerical convergence.
 *          rwork[4] = max_{i/=j} |COS(A(:,i),A(:,j))| in the last sweep.
 *                    This is useful information in cases when ZGESVJ did
 *                    not converge, as it can be used to estimate whether
 *                    the output is still useful and for post festum analysis.
 *          rwork[5] = the largest absolute value over all sines of the
 *                    Jacobi rotation angles in the last sweep. It can be
 *                    useful for a post festum analysis.
 *
 * @param[in] lrwork
 *          Length of RWORK.
 *          LRWORK >= 1, if MIN(M,N) = 0, and LRWORK >= MAX(6,N), otherwise.
 *          If on entry LRWORK = -1, then a workspace query is assumed and
 *          no computation is done; RWORK(1) is set to the minimal (and optimal)
 *          length of RWORK.
 *
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, then the i-th argument had an illegal value
 *                         - > 0: ZGESVJ did not converge in the maximal allowed number (30)
 *                           of sweeps. The output may still be useful. See the
 *                           description of rwork.
 */
void zgesvj(const char* joba, const char* jobu, const char* jobv,
            const int m, const int n, double complex* const restrict A, const int lda,
            double* const restrict SVA, const int mv,
            double complex* const restrict V, const int ldv,
            double complex* const restrict cwork, const int lwork,
            double* const restrict rwork, const int lrwork, int* info)
{
    int lsvec, uctol, rsvec, applv, upper, lower, lquery;
    int minmn, lwmin, lrwmin, mvl = 0;
    int i, ibr, igl, ir1, p, q, kbl, nbl;
    int rowskip, lkahead, swband, blskip;
    int notrot, pskipped, emptsw, iswrot;
    int ijblsk, jbc, jgl;
    int n2, n4, n34;
    int ierr;
    int rotok, noscale, goscale;
    double complex aapq, ompq;
    double aapp, aapp0, aapq1, aaqq, apoaq, aqoap;
    double big, bigtheta, cs, sn, t, temp1, theta, thsign;
    double ctol, epsln, mxaapq, mxsinj, rootbig, rooteps;
    double rootsfmin, sfmin, skl, small, tol, roottol;

    lsvec = (jobu[0] == 'U' || jobu[0] == 'u' || jobu[0] == 'F' || jobu[0] == 'f');
    uctol = (jobu[0] == 'C' || jobu[0] == 'c');
    rsvec = (jobv[0] == 'V' || jobv[0] == 'v' || jobv[0] == 'J' || jobv[0] == 'j');
    applv = (jobv[0] == 'A' || jobv[0] == 'a');
    upper = (joba[0] == 'U' || joba[0] == 'u');
    lower = (joba[0] == 'L' || joba[0] == 'l');

    minmn = (m < n) ? m : n;
    if (minmn == 0) {
        lwmin = 1;
        lrwmin = 1;
    } else {
        lwmin = m + n;
        lrwmin = (6 > n) ? 6 : n;
    }

    lquery = (lwork == -1) || (lrwork == -1);
    if (!(upper || lower || joba[0] == 'G' || joba[0] == 'g')) {
        *info = -1;
    } else if (!(lsvec || uctol || jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -2;
    } else if (!(rsvec || applv || jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0 || n > m) {
        *info = -5;
    } else if (lda < m) {
        *info = -7;
    } else if (mv < 0) {
        *info = -9;
    } else if ((rsvec && ldv < n) || (applv && ldv < mv)) {
        *info = -11;
    } else if (uctol && rwork[0] <= ONE) {
        *info = -12;
    } else if (lwork < lwmin && !lquery) {
        *info = -13;
    } else if (lrwork < lrwmin && !lquery) {
        *info = -15;
    } else {
        *info = 0;
    }

    if (*info != 0) {
        xerbla("ZGESVJ", -(*info));
        return;
    } else if (lquery) {
        cwork[0] = (double complex)lwmin;
        rwork[0] = (double)lrwmin;
        return;
    }

    if (minmn == 0) return;

    if (uctol) {
        ctol = rwork[0];
    } else {
        if (lsvec || rsvec || applv) {
            ctol = sqrt((double)m);
        } else {
            ctol = (double)m;
        }
    }

    epsln = dlamch("E");
    rooteps = sqrt(epsln);
    sfmin = dlamch("S");
    rootsfmin = sqrt(sfmin);
    small = sfmin / epsln;
    big = dlamch("O");
    rootbig = ONE / rootsfmin;
    bigtheta = ONE / rooteps;

    tol = ctol * epsln;
    roottol = sqrt(tol);

    if ((double)m * epsln >= ONE) {
        *info = -4;
        xerbla("ZGESVJ", -(*info));
        return;
    }

    if (rsvec) {
        mvl = n;
        zlaset("A", mvl, n, CZERO, CONE, V, ldv);
    } else if (applv) {
        mvl = mv;
    }
    rsvec = rsvec || applv;

    skl = ONE / sqrt((double)m * (double)n);
    noscale = 1;
    goscale = 1;

    if (lower) {
        for (p = 0; p < n; p++) {
            aapp = ZERO;
            aaqq = ONE;
            zlassq(m - p, &A[p + p * lda], 1, &aapp, &aaqq);
            if (aapp > big) {
                *info = -6;
                xerbla("ZGESVJ", -(*info));
                return;
            }
            aaqq = sqrt(aaqq);
            if (aapp < (big / aaqq) && noscale) {
                SVA[p] = aapp * aaqq;
            } else {
                noscale = 0;
                SVA[p] = aapp * (aaqq * skl);
                if (goscale) {
                    goscale = 0;
                    for (q = 0; q < p; q++) {
                        SVA[q] = SVA[q] * skl;
                    }
                }
            }
        }
    } else if (upper) {
        for (p = 0; p < n; p++) {
            aapp = ZERO;
            aaqq = ONE;
            zlassq(p + 1, &A[p * lda], 1, &aapp, &aaqq);
            if (aapp > big) {
                *info = -6;
                xerbla("ZGESVJ", -(*info));
                return;
            }
            aaqq = sqrt(aaqq);
            if (aapp < (big / aaqq) && noscale) {
                SVA[p] = aapp * aaqq;
            } else {
                noscale = 0;
                SVA[p] = aapp * (aaqq * skl);
                if (goscale) {
                    goscale = 0;
                    for (q = 0; q < p; q++) {
                        SVA[q] = SVA[q] * skl;
                    }
                }
            }
        }
    } else {
        for (p = 0; p < n; p++) {
            aapp = ZERO;
            aaqq = ONE;
            zlassq(m, &A[p * lda], 1, &aapp, &aaqq);
            if (aapp > big) {
                *info = -6;
                xerbla("ZGESVJ", -(*info));
                return;
            }
            aaqq = sqrt(aaqq);
            if (aapp < (big / aaqq) && noscale) {
                SVA[p] = aapp * aaqq;
            } else {
                noscale = 0;
                SVA[p] = aapp * (aaqq * skl);
                if (goscale) {
                    goscale = 0;
                    for (q = 0; q < p; q++) {
                        SVA[q] = SVA[q] * skl;
                    }
                }
            }
        }
    }

    if (noscale) skl = ONE;

    aapp = ZERO;
    aaqq = big;
    for (p = 0; p < n; p++) {
        if (SVA[p] != ZERO) aaqq = fmin(aaqq, SVA[p]);
        aapp = fmax(aapp, SVA[p]);
    }

    if (aapp == ZERO) {
        if (lsvec) zlaset("G", m, n, CZERO, CONE, A, lda);
        rwork[0] = ONE;
        rwork[1] = ZERO;
        rwork[2] = ZERO;
        rwork[3] = ZERO;
        rwork[4] = ZERO;
        rwork[5] = ZERO;
        return;
    }

    if (n == 1) {
        if (lsvec) zlascl("G", 0, 0, SVA[0], skl, m, 1, A, lda, &ierr);
        rwork[0] = ONE / skl;
        rwork[1] = (SVA[0] >= sfmin) ? ONE : ZERO;
        rwork[2] = ZERO;
        rwork[3] = ZERO;
        rwork[4] = ZERO;
        rwork[5] = ZERO;
        return;
    }

    sn = sqrt(sfmin / epsln);
    temp1 = sqrt(big / (double)n);
    if ((aapp <= sn) || (aaqq >= temp1) ||
        ((sn <= aaqq) && (aapp <= temp1))) {
        temp1 = fmin(big, temp1 / aapp);
    } else if ((aaqq <= sn) && (aapp <= temp1)) {
        temp1 = fmin(sn / aaqq, big / (aapp * sqrt((double)n)));
    } else if ((aaqq >= sn) && (aapp >= temp1)) {
        temp1 = fmax(sn / aaqq, temp1 / aapp);
    } else if ((aaqq <= sn) && (aapp >= temp1)) {
        temp1 = fmin(sn / aaqq, big / (sqrt((double)n) * aapp));
    } else {
        temp1 = ONE;
    }

    if (temp1 != ONE) {
        dlascl("G", 0, 0, ONE, temp1, n, 1, SVA, n, &ierr);
    }
    skl = temp1 * skl;
    if (skl != ONE) {
        zlascl(joba, 0, 0, ONE, skl, m, n, A, lda, &ierr);
        skl = ONE / skl;
    }

    emptsw = (n * (n - 1)) / 2;
    notrot = 0;

    for (q = 0; q < n; q++) {
        cwork[q] = CONE;
    }

    swband = 3;
    kbl = (8 < n) ? 8 : n;
    nbl = n / kbl;
    if (nbl * kbl != n) nbl = nbl + 1;
    blskip = kbl * kbl;
    rowskip = (5 < kbl) ? 5 : kbl;
    lkahead = 1;

    if ((lower || upper) && (n > ((64 > 4 * kbl) ? 64 : 4 * kbl))) {
        n4 = n / 4;
        n2 = n / 2;
        n34 = 3 * n4;
        if (applv) {
            q = 0;
        } else {
            q = 1;
        }

        if (lower) {

            zgsvj0(jobv, m - n34, n - n34, &A[n34 + n34 * lda], lda,
                   &cwork[n34], &SVA[n34], mvl,
                   &V[n34 * q + n34 * ldv], ldv, epsln, sfmin, tol,
                   2, &cwork[n], lwork - n, &ierr);

            zgsvj0(jobv, m - n2, n34 - n2, &A[n2 + n2 * lda], lda,
                   &cwork[n2], &SVA[n2], mvl,
                   &V[n2 * q + n2 * ldv], ldv, epsln, sfmin, tol, 2,
                   &cwork[n], lwork - n, &ierr);

            zgsvj1(jobv, m - n2, n - n2, n4, &A[n2 + n2 * lda], lda,
                   &cwork[n2], &SVA[n2], mvl,
                   &V[n2 * q + n2 * ldv], ldv, epsln, sfmin, tol, 1,
                   &cwork[n], lwork - n, &ierr);

            zgsvj0(jobv, m - n4, n2 - n4, &A[n4 + n4 * lda], lda,
                   &cwork[n4], &SVA[n4], mvl,
                   &V[n4 * q + n4 * ldv], ldv, epsln, sfmin, tol, 1,
                   &cwork[n], lwork - n, &ierr);

            zgsvj0(jobv, m, n4, A, lda, cwork, SVA, mvl, V, ldv,
                   epsln, sfmin, tol, 1, &cwork[n], lwork - n,
                   &ierr);

            zgsvj1(jobv, m, n2, n4, A, lda, cwork, SVA, mvl, V,
                   ldv, epsln, sfmin, tol, 1, &cwork[n],
                   lwork - n, &ierr);

        } else if (upper) {

            zgsvj0(jobv, n4, n4, A, lda, cwork, SVA, mvl, V,
                   ldv,
                   epsln, sfmin, tol, 2, &cwork[n], lwork - n,
                   &ierr);

            zgsvj0(jobv, n2, n4, &A[n4 * lda], lda,
                   &cwork[n4],
                   &SVA[n4], mvl, &V[n4 * q + n4 * ldv], ldv,
                   epsln, sfmin, tol, 1, &cwork[n], lwork - n,
                   &ierr);

            zgsvj1(jobv, n2, n2, n4, A, lda, cwork, SVA, mvl, V,
                   ldv, epsln, sfmin, tol, 1, &cwork[n],
                   lwork - n, &ierr);

            zgsvj0(jobv, n2 + n4, n4, &A[n2 * lda], lda,
                   &cwork[n2], &SVA[n2], mvl,
                   &V[n2 * q + n2 * ldv], ldv, epsln, sfmin, tol, 1,
                   &cwork[n], lwork - n, &ierr);

        }
    }

    for (i = 0; i < NSWEEP; i++) {

        mxaapq = ZERO;
        mxsinj = ZERO;
        iswrot = 0;

        notrot = 0;
        pskipped = 0;

        for (ibr = 0; ibr < nbl; ibr++) {
            igl = ibr * kbl;

            for (ir1 = 0; ir1 <= ((lkahead < nbl - ibr - 1) ? lkahead : nbl - ibr - 1); ir1++) {

                igl = (ibr + ir1) * kbl;

                for (p = igl; p < (((igl + kbl - 1) < (n - 2)) ? (igl + kbl - 1) : (n - 2)) + 1; p++) {

                    q = cblas_idamax(n - p, &SVA[p], 1) + p;
                    if (p != q) {
                        cblas_zswap(m, &A[p * lda], 1, &A[q * lda], 1);
                        if (rsvec) cblas_zswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
                        temp1 = SVA[p];
                        SVA[p] = SVA[q];
                        SVA[q] = temp1;
                        aapq = cwork[p];
                        cwork[p] = cwork[q];
                        cwork[q] = aapq;
                    }

                    if (ir1 == 0) {
                        if (SVA[p] < rootbig && SVA[p] > rootsfmin) {
                            SVA[p] = cblas_dznrm2(m, &A[p * lda], 1);
                        } else {
                            temp1 = ZERO;
                            aapp = ONE;
                            zlassq(m, &A[p * lda], 1, &temp1, &aapp);
                            SVA[p] = temp1 * sqrt(aapp);
                        }
                        aapp = SVA[p];
                    } else {
                        aapp = SVA[p];
                    }

                    if (aapp > ZERO) {

                        pskipped = 0;

                        for (q = p + 1; q < (((igl + kbl - 1) < (n - 1)) ? (igl + kbl - 1) : (n - 1)) + 1; q++) {

                            aaqq = SVA[q];

                            if (aaqq > ZERO) {

                                aapp0 = aapp;
                                if (aaqq >= ONE) {
                                    rotok = (small * aapp) <= aaqq;
                                    if (aapp < (big / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aaqq) / aapp;
                                    } else {
                                        cblas_zcopy(m, &A[p * lda], 1, &cwork[n], 1);
                                        zlascl("G", 0, 0, aapp, ONE, m, 1, &cwork[n], lda, &ierr);
                                        cblas_zdotc_sub(m, &cwork[n], 1, &A[q * lda], 1, &aapq);
                                        aapq = aapq / aaqq;
                                    }
                                } else {
                                    rotok = aapp <= (aaqq / small);
                                    if (aapp > (small / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aapp) / aaqq;
                                    } else {
                                        cblas_zcopy(m, &A[q * lda], 1, &cwork[n], 1);
                                        zlascl("G", 0, 0, aaqq, ONE, m, 1, &cwork[n], lda, &ierr);
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &cwork[n], 1, &aapq);
                                        aapq = aapq / aapp;
                                    }
                                }

                                aapq1 = -cabs(aapq);
                                mxaapq = fmax(mxaapq, -aapq1);

                                if (fabs(aapq1) > tol) {
                                    ompq = aapq / cabs(aapq);

                                    if (ir1 == 0) {
                                        notrot = 0;
                                        pskipped = 0;
                                        iswrot = iswrot + 1;
                                    }

                                    if (rotok) {

                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabs(aqoap - apoaq) / aapq1;

                                        if (fabs(theta) > bigtheta) {

                                            t = HALF / theta;
                                            cs = ONE;

                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * t);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * t);
                                            }

                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));
                                            mxsinj = fmax(mxsinj, fabs(t));

                                        } else {

                                            thsign = -copysign(ONE, aapq1);
                                            t = ONE / (theta + thsign * sqrt(ONE + theta * theta));
                                            cs = sqrt(ONE / (ONE + t * t));
                                            sn = t * cs;

                                            mxsinj = fmax(mxsinj, fabs(sn));
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));

                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * sn);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * sn);
                                            }
                                        }
                                        cwork[p] = -cwork[q] * ompq;

                                    } else {
                                        cblas_zcopy(m, &A[p * lda], 1, &cwork[n], 1);
                                        zlascl("G", 0, 0, aapp, ONE, m, 1, &cwork[n], lda, &ierr);
                                        zlascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                        {
                                            double complex neg_aapq = -aapq;
                                            cblas_zaxpy(m, &neg_aapq, &cwork[n], 1, &A[q * lda], 1);
                                        }
                                        zlascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                        SVA[q] = aaqq * sqrt(fmax(ZERO, ONE - aapq1 * aapq1));
                                        mxsinj = fmax(mxsinj, sfmin);
                                    }

                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_dznrm2(m, &A[q * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            zlassq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrt(aaqq);
                                        }
                                    }
                                    if ((aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_dznrm2(m, &A[p * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            zlassq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrt(aapp);
                                        }
                                        SVA[p] = aapp;
                                    }

                                } else {
                                    if (ir1 == 0) notrot++;
                                    pskipped++;
                                }
                            } else {
                                if (ir1 == 0) notrot++;
                                pskipped++;
                            }

                            if (i < swband && pskipped > rowskip) {
                                if (ir1 == 0) aapp = -aapp;
                                notrot = 0;
                                break;
                            }

                        } /* end q-loop */

                        SVA[p] = aapp;

                    } else {
                        SVA[p] = aapp;
                        if (ir1 == 0 && aapp == ZERO)
                            notrot += (((igl + kbl - 1) < (n - 1)) ? (igl + kbl - 1) : (n - 1)) - p;
                    }

                } /* end p-loop */

            } /* end ir1-loop */

            igl = ibr * kbl;

            for (jbc = ibr + 1; jbc < nbl; jbc++) {

                jgl = jbc * kbl;

                ijblsk = 0;
                for (p = igl; p < (((igl + kbl - 1) < (n - 1)) ? (igl + kbl - 1) : (n - 1)) + 1; p++) {

                    aapp = SVA[p];
                    if (aapp > ZERO) {

                        pskipped = 0;

                        for (q = jgl; q < (((jgl + kbl - 1) < (n - 1)) ? (jgl + kbl - 1) : (n - 1)) + 1; q++) {

                            aaqq = SVA[q];
                            if (aaqq > ZERO) {
                                aapp0 = aapp;

                                if (aaqq >= ONE) {
                                    if (aapp >= aaqq) {
                                        rotok = (small * aapp) <= aaqq;
                                    } else {
                                        rotok = (small * aaqq) <= aapp;
                                    }
                                    if (aapp < (big / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aaqq) / aapp;
                                    } else {
                                        cblas_zcopy(m, &A[p * lda], 1, &cwork[n], 1);
                                        zlascl("G", 0, 0, aapp, ONE, m, 1, &cwork[n], lda, &ierr);
                                        cblas_zdotc_sub(m, &cwork[n], 1, &A[q * lda], 1, &aapq);
                                        aapq = aapq / aaqq;
                                    }
                                } else {
                                    if (aapp >= aaqq) {
                                        rotok = aapp <= (aaqq / small);
                                    } else {
                                        rotok = aaqq <= (aapp / small);
                                    }
                                    if (aapp > (small / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        {
                                            double mx = fmax(aaqq, aapp);
                                            double mn = fmin(aaqq, aapp);
                                            aapq = (aapq / mx) / mn;
                                        }
                                    } else {
                                        cblas_zcopy(m, &A[q * lda], 1, &cwork[n], 1);
                                        zlascl("G", 0, 0, aaqq, ONE, m, 1, &cwork[n], lda, &ierr);
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &cwork[n], 1, &aapq);
                                        aapq = aapq / aapp;
                                    }
                                }

                                aapq1 = -cabs(aapq);
                                mxaapq = fmax(mxaapq, -aapq1);

                                if (fabs(aapq1) > tol) {
                                    ompq = aapq / cabs(aapq);
                                    notrot = 0;
                                    pskipped = 0;
                                    iswrot = iswrot + 1;

                                    if (rotok) {

                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabs(aqoap - apoaq) / aapq1;
                                        if (aaqq > aapp0) theta = -theta;

                                        if (fabs(theta) > bigtheta) {

                                            t = HALF / theta;
                                            cs = ONE;
                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * t);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * t);
                                            }
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));
                                            mxsinj = fmax(mxsinj, fabs(t));

                                        } else {

                                            thsign = -copysign(ONE, aapq1);
                                            if (aaqq > aapp0) thsign = -thsign;
                                            t = ONE / (theta + thsign * sqrt(ONE + theta * theta));
                                            cs = sqrt(ONE / (ONE + t * t));
                                            sn = t * cs;

                                            mxsinj = fmax(mxsinj, fabs(sn));
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));

                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * sn);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * sn);
                                            }
                                        }
                                        cwork[p] = -cwork[q] * ompq;

                                    } else {
                                        if (aapp > aaqq) {
                                            cblas_zcopy(m, &A[p * lda], 1, &cwork[n], 1);
                                            zlascl("G", 0, 0, aapp, ONE, m, 1, &cwork[n], lda, &ierr);
                                            zlascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                            {
                                                double complex neg_aapq = -aapq;
                                                cblas_zaxpy(m, &neg_aapq, &cwork[n], 1, &A[q * lda], 1);
                                            }
                                            zlascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE - aapq1 * aapq1));
                                            mxsinj = fmax(mxsinj, sfmin);
                                        } else {
                                            cblas_zcopy(m, &A[q * lda], 1, &cwork[n], 1);
                                            zlascl("G", 0, 0, aaqq, ONE, m, 1, &cwork[n], lda, &ierr);
                                            zlascl("G", 0, 0, aapp, ONE, m, 1, &A[p * lda], lda, &ierr);
                                            {
                                                double complex neg_conj_aapq = -conj(aapq);
                                                cblas_zaxpy(m, &neg_conj_aapq, &cwork[n], 1, &A[p * lda], 1);
                                            }
                                            zlascl("G", 0, 0, ONE, aapp, m, 1, &A[p * lda], lda, &ierr);
                                            SVA[p] = aapp * sqrt(fmax(ZERO, ONE - aapq1 * aapq1));
                                            mxsinj = fmax(mxsinj, sfmin);
                                        }
                                    }

                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_dznrm2(m, &A[q * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            zlassq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrt(aaqq);
                                        }
                                    }
                                    if ((aapp / aapp0) * (aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_dznrm2(m, &A[p * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            zlassq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrt(aapp);
                                        }
                                        SVA[p] = aapp;
                                    }

                                } else {
                                    notrot = notrot + 1;
                                    pskipped = pskipped + 1;
                                    ijblsk = ijblsk + 1;
                                }
                            } else {
                                notrot = notrot + 1;
                                pskipped = pskipped + 1;
                                ijblsk = ijblsk + 1;
                            }

                            if (i < swband && ijblsk >= blskip) {
                                SVA[p] = aapp;
                                notrot = 0;
                                goto offdiag_cleanup;
                            }
                            if (i < swband && pskipped > rowskip) {
                                aapp = -aapp;
                                notrot = 0;
                                break;
                            }

                        } /* end q-loop */

                        SVA[p] = aapp;

                    } else {

                        if (aapp == ZERO) notrot = notrot +
                            (((jgl + kbl - 1) < (n - 1)) ? (jgl + kbl - 1) : (n - 1)) - jgl + 1;
                        if (aapp < ZERO) notrot = 0;

                    }

                } /* end p-loop */

            } /* end jbc-loop */

            offdiag_cleanup:
            {
                int pp;
                for (pp = igl; pp < ((igl + kbl < n) ? igl + kbl : n); pp++) {
                    SVA[pp] = fabs(SVA[pp]);
                }
            }

        } /* end ibr-loop */

        if (SVA[n - 1] < rootbig && SVA[n - 1] > rootsfmin) {
            SVA[n - 1] = cblas_dznrm2(m, &A[(n - 1) * lda], 1);
        } else {
            t = ZERO;
            aapp = ONE;
            zlassq(m, &A[(n - 1) * lda], 1, &t, &aapp);
            SVA[n - 1] = t * sqrt(aapp);
        }

        if ((i + 1 < swband) && ((mxaapq <= roottol) || (iswrot <= n)))
            swband = i + 1;

        if ((i > swband) && (mxaapq < sqrt((double)n) * tol)
            && ((double)n * mxaapq * mxsinj < tol)) {
            break;
        }

        if (notrot >= emptsw) break;

    } /* end sweep loop */

    if (i >= NSWEEP) {
        *info = NSWEEP - 1;
    } else {
        *info = 0;
    }

    n2 = 0;
    n4 = 0;
    for (p = 0; p < n - 1; p++) {
        q = cblas_idamax(n - p, &SVA[p], 1) + p;
        if (p != q) {
            temp1 = SVA[p];
            SVA[p] = SVA[q];
            SVA[q] = temp1;
            cblas_zswap(m, &A[p * lda], 1, &A[q * lda], 1);
            if (rsvec) cblas_zswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
        }
        if (SVA[p] != ZERO) {
            n4 = n4 + 1;
            if (SVA[p] * skl > sfmin) n2 = n2 + 1;
        }
    }
    if (SVA[n - 1] != ZERO) {
        n4 = n4 + 1;
        if (SVA[n - 1] * skl > sfmin) n2 = n2 + 1;
    }

    if (lsvec || uctol) {
        for (p = 0; p < n4; p++) {
            zlascl("G", 0, 0, SVA[p], ONE, m, 1, &A[p * lda], m, &ierr);
        }
    }

    if (rsvec) {
        for (p = 0; p < n; p++) {
            temp1 = ONE / cblas_dznrm2(mvl, &V[p * ldv], 1);
            cblas_zdscal(mvl, temp1, &V[p * ldv], 1);
        }
    }

    if (((skl > ONE) && (SVA[0] < (big / skl)))
        || ((skl < ONE) && (SVA[((n2 > 1) ? n2 : 1) - 1] > (sfmin / skl)))) {
        for (p = 0; p < n; p++) {
            SVA[p] = skl * SVA[p];
        }
        skl = ONE;
    }

    rwork[0] = skl;
    rwork[1] = (double)n4;
    rwork[2] = (double)n2;
    rwork[3] = (double)(i + 1);
    rwork[4] = mxaapq;
    rwork[5] = mxsinj;

    return;
}
