/**
 * @file dsytrd_sb2st.c
 * @brief DSYTRD_SB2ST reduces a real symmetric band matrix to tridiagonal form.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DSYTRD_SB2ST reduces a real symmetric band matrix `A` to real symmetric
 * tridiagonal form T by an orthogonal similarity transformation:
 * `Q**T * A * Q = T`.
 *
 * @param[in]     stage1 `'N'`: "No": to mention that the stage 1 of the reduction
 *                              from dense to band using the dsytrd_sy2sb routine was
 *                              not called before this routine to reproduce AB. In
 *                              other term this routine is called as standalone.
 *                       `'Y'`: "Yes": to mention that the stage 1 of the reduction
 *                              from dense to band using the dsytrd_sy2sb routine has
 *                              been called to produce AB (e.g., AB is the output of
 *                              dsytrd_sy2sb).
 * @param[in]     vect   `'N'`: No need for the Householder representation, and thus
 *                              `lhous` is of size `max(1,4*n)`;
 *                       `'V'`: the Householder representation is needed to either
 *                              generate or to apply Q later on, then `lhous` is to be
 *                              queried and computed. (NOT AVAILABLE IN THIS RELEASE).
 * @param[in]     uplo   `'U'`: Upper triangle of `A` is stored;
 *                       `'L'`: Lower triangle of `A` is stored.
 * @param[in]     n      The order of the matrix `A`. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix `A` if `uplo='U'`,
 *                       or the number of subdiagonals if `uplo='L'`. `kd>=0`.
 * @param[in,out] AB     On entry, the upper or lower triangle of the symmetric band
 *                       matrix A, stored in the first `kd+1` rows of the array. The
 *                       j-th column of A is stored in the j-th column of the array `AB`
 *                       as follows: if `uplo='U'`, `AB[kd+i-j,j]=A[i,j]` for
 *                       `max(0,j-kd)<=i<=j`; if `uplo='L'`, `AB[i-j,j]=A[i,j]` for
 *                       `j<=i<=min(n-1,j+kd)`.
 *                       On exit, the diagonal elements of `AB` are overwritten by the
 *                       diagonal elements of the tridiagonal matrix T; if `kd>0`, the
 *                       elements on the first superdiagonal (if `uplo='U'`) or the
 *                       first subdiagonal (if `uplo='L'`) are overwritten by the
 *                       off-diagonal elements of T; the rest of `AB` is overwritten by
 *                       values generated during the reduction.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[out]    D      Array of dimension (`n`). The diagonal elements of the
 *                       tridiagonal matrix T.
 * @param[out]    E      Array of dimension (`n-1`). The off-diagonal elements of the
 *                       tridiagonal matrix T: `E[i]=T[i,i+1]` if `uplo='U'`;
 *                       `E[i]=T[i+1,i]` if `uplo='L'`.
 * @param[out]    hous   Array of dimension (`max(1,lhous)`). Stores the Householder
 *                       representation.
 * @param[in]     lhous  The dimension of the array `hous`.
 *                       If `n=0` or `kd<=1`, `lhous>=1`, else `lhous = max(1, dimension)`.
 *                       If `lwork=-1`, or `lhous=-1`, then a query is assumed; the
 *                       routine only calculates the optimal size of the `hous` array,
 *                       returns this value as the first entry of the `hous` array.
 *                       `lhous = max(1, dimension)` where `dimension = 4*n` if `vect='N'`,
 *                       not available now if `vect='H'`.
 * @param[out]    work   Array of dimension (`max(1,lwork)`). On exit, if `info=0`,
 *                       `work[0]` returns the optimal `lwork`.
 * @param[in]     lwork  The dimension of the array `work`.
 *                       If `n=0` or `kd<=1`, `lwork>=1`, else `lwork = max(1, dimension)`.
 *                       If `lwork=-1`, or `lhous=-1`, then a workspace query is assumed;
 *                       the routine only calculates the optimal size of the `work` array,
 *                       returns this value as the first entry of the `work` array.
 *                       `lwork = max(1, dimension)` where `dimension = (2*kd+1)*n + kd*NTHREADS`,
 *                       where `kd` is the blocking size of the reduction, FACTOPTNB is the
 *                       blocking used by the QR or LQ algorithm, usually FACTOPTNB=128 is a
 *                       good choice, NTHREADS is the number of threads used when openMP
 *                       compilation is enabled, otherwise =1.
 * @param[out]    info   `info=0`: successful exit
 *                       `info<0`: if `info=-i`, the i-th argument had an illegal value
 *
 * @par Further Details:
 * @rst
 * Implemented by Azzam Haidar.
 *
 * All details are available on technical report, SC11, SC13 papers.
 *
 * Azzam Haidar, Hatem Ltaief, and Jack Dongarra.
 * Parallel reduction to condensed forms for symmetric eigenvalue problems
 * using aggregated fine-grained and memory-aware kernels. In Proceedings
 * of 2011 International Conference for High Performance Computing,
 * Networking, Storage and Analysis (SC '11), New York, NY, USA,
 * Article 8 , 11 pages.
 * https://doi.org/10.1145/2063384.2063394
 *
 * A. Haidar, J. Kurzak, P. Luszczek, 2013.
 * An improved parallel singular value algorithm and its implementation
 * for multicore hardware, In Proceedings of 2013 International Conference
 * for High Performance Computing, Networking, Storage and Analysis (SC '13).
 * Denver, Colorado, USA, 2013.
 * Article 90, 12 pages.
 * https://doi.org/10.1145/2503210.2503292
 *
 * A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.
 * A novel hybrid CPU-GPU generalized eigensolver for electronic structure
 * calculations based on fine-grained memory aware tasks.
 * International Journal of High Performance Computing Applications.
 * Volume 28 Issue 2, Pages 196-209, May 2014.
 * https://doi.org/10.1177/1094342013502097
 * @endrst
 */
void dsytrd_sb2st(const char* stage1, const char* vect, const char* uplo,
                  const INT n, const INT kd,
                  f64* AB, const INT ldab,
                  f64* D, f64* E,
                  f64* hous, const INT lhous,
                  f64* work, const INT lwork, INT* info)
{
    const f64 rzero = 0.0;
    const f64 zero = 0.0;

    INT lquery, wantq, upper, afters1;
    INT i, m, k, ib, sweepid, myid, shift, stt, st;
    INT ed, stind, edind, blklastind, colpt, thed;
    INT stepercol, grsiz, thgrsiz, thgrnb, thgrid;
    INT ttype;
    /* nbtiles, nthreads: set but unused (OpenMP threading placeholders) */
    INT abdpos, abofdpos, dpos, ofdpos, awpos;
    INT inda, indw, apos, sizea, lda, indv, indtau;
    INT sizetau, ldv, lhmin, lwmin;

    *info = 0;
    afters1 = (stage1[0] == 'Y' || stage1[0] == 'y');
    wantq = (vect[0] == 'V' || vect[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1) || (lhous == -1);

    ib = ilaenv2stage(2, "DSYTRD_SB2ST", vect, n, kd, -1, -1);
    if (n == 0 || kd <= 1) {
        lhmin = 1;
        lwmin = 1;
    } else {
        lhmin = ilaenv2stage(3, "DSYTRD_SB2ST", vect, n, kd, ib, -1);
        lwmin = ilaenv2stage(4, "DSYTRD_SB2ST", vect, n, kd, ib, -1);
    }

    if (!afters1 && !(stage1[0] == 'N' || stage1[0] == 'n')) {
        *info = -1;
    } else if (!(vect[0] == 'N' || vect[0] == 'n')) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (kd < 0) {
        *info = -5;
    } else if (ldab < (kd + 1)) {
        *info = -7;
    } else if (lhous < lhmin && !lquery) {
        *info = -11;
    } else if (lwork < lwmin && !lquery) {
        *info = -13;
    }

    if (*info == 0) {
        hous[0] = (f64)lhmin;
        work[0] = (f64)lwmin;
    }

    if (*info != 0) {
        xerbla("DSYTRD_SB2ST", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        hous[0] = 1.0;
        work[0] = 1.0;
        return;
    }

    ldv = kd + ib;
    sizetau = 2 * n;
    indtau = 0;
    indv = indtau + sizetau;
    lda = 2 * kd + 1;
    sizea = lda * n;
    inda = 0;
    indw = inda + sizea;
    (void)1;  /* nthreads = 1; unused OpenMP placeholder */

    if (upper) {
        apos = inda + kd;
        awpos = inda;
        dpos = apos + kd;
        ofdpos = dpos - 1;
        abdpos = kd;
        abofdpos = kd - 1;
    } else {
        apos = inda;
        awpos = inda + kd + 1;
        dpos = apos;
        ofdpos = dpos + 1;
        abdpos = 0;
        abofdpos = 1;
    }

    if (kd == 0) {
        for (i = 0; i < n; i++) {
            D[i] = AB[abdpos + i * ldab];
        }
        for (i = 0; i < n - 1; i++) {
            E[i] = rzero;
        }

        hous[0] = 1.0;
        work[0] = 1.0;
        return;
    }

    if (kd == 1) {
        for (i = 0; i < n; i++) {
            D[i] = AB[abdpos + i * ldab];
        }

        if (upper) {
            for (i = 0; i < n - 1; i++) {
                E[i] = AB[abofdpos + (i + 1) * ldab];
            }
        } else {
            for (i = 0; i < n - 1; i++) {
                E[i] = AB[abofdpos + i * ldab];
            }
        }

        hous[0] = 1.0;
        work[0] = 1.0;
        return;
    }

    thgrsiz = n;
    grsiz = 1;
    shift = 3;
    (void)ceil((f64)n / (f64)kd);  /* nbtiles: unused OpenMP placeholder */
    stepercol = (INT)ceil((f64)shift / (f64)grsiz);
    thgrnb = (INT)ceil((f64)(n - 1) / (f64)thgrsiz);

    dlacpy("A", kd + 1, n, AB, ldab, &work[apos], lda);
    dlaset("A", kd, n, zero, zero, &work[awpos], lda);

    for (thgrid = 1; thgrid <= thgrnb; thgrid++) {
        stt = (thgrid - 1) * thgrsiz + 1;
        thed = ((stt + thgrsiz - 1) < (n - 1)) ? (stt + thgrsiz - 1) : (n - 1);
        for (i = stt; i <= n - 1; i++) {
            ed = (i < thed) ? i : thed;
            if (stt > ed) break;
            for (m = 1; m <= stepercol; m++) {
                st = stt;
                for (sweepid = st; sweepid <= ed; sweepid++) {
                    for (k = 1; k <= grsiz; k++) {
                        myid = (i - sweepid) * (stepercol * grsiz)
                               + (m - 1) * grsiz + k;
                        if (myid == 1) {
                            ttype = 1;
                        } else {
                            ttype = (myid % 2) + 2;
                        }

                        if (ttype == 2) {
                            colpt = (myid / 2) * kd + sweepid;
                            stind = colpt - kd + 1;
                            edind = (colpt < n) ? colpt : n;
                            blklastind = colpt;
                        } else {
                            colpt = ((myid + 1) / 2) * kd + sweepid;
                            stind = colpt - kd + 1;
                            edind = (colpt < n) ? colpt : n;
                            if ((stind >= edind - 1) && (edind == n)) {
                                blklastind = n;
                            } else {
                                blklastind = 0;
                            }
                        }

                        dsb2st_kernels(uplo, wantq, ttype,
                                       stind, edind, sweepid, n, kd, ib,
                                       &work[inda], lda,
                                       &hous[indv], &hous[indtau], ldv,
                                       &work[indw]);

                        if (blklastind >= (n - 1)) {
                            stt = stt + 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (i = 0; i < n; i++) {
        D[i] = work[dpos + i * lda];
    }

    if (upper) {
        for (i = 0; i < n - 1; i++) {
            E[i] = work[ofdpos + (i + 1) * lda];
        }
    } else {
        for (i = 0; i < n - 1; i++) {
            E[i] = work[ofdpos + i * lda];
        }
    }

    work[0] = (f64)lwmin;
}
