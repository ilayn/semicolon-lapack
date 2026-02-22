/**
 * @file chetrd_hb2st.c
 * @brief CHETRD_HB2ST reduces a complex Hermitian band matrix to real symmetric tridiagonal form.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRD_HB2ST reduces a complex Hermitian band matrix A to real symmetric
 * tridiagonal form T by a unitary similarity transformation:
 * Q**H * A * Q = T.
 *
 * @param[in]     stage1  = 'N':  the stage 1 of the reduction from dense to
 *                          band using the chetrd_he2hb routine was not called
 *                          before this routine to reproduce AB. In other term
 *                          this routine is called as standalone.
 *                          = 'Y':  the stage 1 of the reduction from dense to
 *                          band using the chetrd_he2hb routine has been called
 *                          to produce AB (e.g., AB is the output of
 *                          chetrd_he2hb).
 * @param[in]     vect    = 'N':  No need for the Householder representation,
 *                          and thus LHOUS is of size max(1, 4*N);
 *                          = 'V':  the Householder representation is needed to
 *                          either generate or to apply Q later on,
 *                          then LHOUS is to be queried and computed.
 *                          (NOT AVAILABLE IN THIS RELEASE).
 * @param[in]     uplo    = 'U':  Upper triangle of A is stored;
 *                          = 'L':  Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A.  N >= 0.
 * @param[in]     kd      The number of superdiagonals of the matrix A if
 *                         UPLO = 'U', or the number of subdiagonals if
 *                         UPLO = 'L'.  KD >= 0.
 * @param[in,out] AB      Single complex array, dimension (LDAB,N).
 *                         On entry, the upper or lower triangle of the
 *                         Hermitian band matrix A, stored in the first KD+1
 *                         rows of the array.  The j-th column of A is stored
 *                         in the j-th column of the array AB as follows:
 *                         if UPLO = 'U', AB(kd+i-j,j) = A(i,j) for
 *                         max(0,j-kd)<=i<=j;
 *                         if UPLO = 'L', AB(i-j,j) = A(i,j) for
 *                         j<=i<=min(n-1,j+kd).
 *                         On exit, the diagonal elements of AB are overwritten
 *                         by the diagonal elements of the tridiagonal matrix T;
 *                         if KD > 0, the elements on the first superdiagonal
 *                         (if UPLO = 'U') or the first subdiagonal
 *                         (if UPLO = 'L') are overwritten by the off-diagonal
 *                         elements of T; the rest of AB is overwritten by
 *                         values generated during the reduction.
 * @param[in]     ldab    The leading dimension of the array AB.  LDAB >= KD+1.
 * @param[out]    D       Single precision array, dimension (N).
 *                         The diagonal elements of the tridiagonal matrix T.
 * @param[out]    E       Single precision array, dimension (N-1).
 *                         The off-diagonal elements of the tridiagonal matrix T:
 *                         E(i) = T(i,i+1) if UPLO = 'U';
 *                         E(i) = T(i+1,i) if UPLO = 'L'.
 * @param[out]    hous    Single complex array, dimension (MAX(1,LHOUS)).
 *                         Stores the Householder representation.
 * @param[in]     lhous   The dimension of the array HOUS.
 *                         If N = 0 or KD <= 1, LHOUS >= 1, else
 *                         LHOUS = MAX(1, dimension) where
 *                         dimension = 4*N if VECT='N'.
 * @param[out]    work    Single complex array, dimension (MAX(1,LWORK)).
 *                         On exit, if INFO = 0, WORK(1) returns the optimal
 *                         LWORK.
 * @param[in]     lwork   The dimension of the array WORK.
 *                         If N = 0 or KD <= 1, LWORK >= 1, else
 *                         LWORK = MAX(1, dimension) where
 *                         dimension = (2KD+1)*N + KD*NTHREADS
 *                         where NTHREADS is the number of threads used when
 *                         openMP compilation is enabled, otherwise =1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void chetrd_hb2st(const char* stage1, const char* vect, const char* uplo,
                  const INT n, const INT kd,
                  c64* AB, const INT ldab,
                  f32* D, f32* E,
                  c64* hous, const INT lhous,
                  c64* work, const INT lwork, INT* info)
{
    const f32 rzero = 0.0f;
    const c64 zero = CMPLXF(0.0f, 0.0f);
    const c64 one = CMPLXF(1.0f, 0.0f);

    INT lquery, wantq, upper, afters1;
    INT i, m, k, ib, sweepid, myid, shift, stt, st;
    INT ed, stind, edind, blklastind, colpt, thed;
    INT stepercol, grsiz, thgrsiz, thgrnb, thgrid;
    INT ttype;
    INT abdpos, abofdpos, dpos, ofdpos, awpos;
    INT inda, indw, apos, sizea, lda, indv, indtau;
    INT sizetau, ldv, lhmin, lwmin;
    f32 abstmp;
    c64 tmp;

    *info = 0;
    afters1 = (stage1[0] == 'Y' || stage1[0] == 'y');
    wantq = (vect[0] == 'V' || vect[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1) || (lhous == -1);

    ib = ilaenv2stage(2, "CHETRD_HB2ST", vect, n, kd, -1, -1);
    if (n == 0 || kd <= 1) {
        lhmin = 1;
        lwmin = 1;
    } else {
        lhmin = ilaenv2stage(3, "CHETRD_HB2ST", vect, n, kd, ib, -1);
        lwmin = ilaenv2stage(4, "CHETRD_HB2ST", vect, n, kd, ib, -1);
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
        hous[0] = CMPLXF((f32)lhmin, 0.0f);
        work[0] = CMPLXF((f32)lwmin, 0.0f);
    }

    if (*info != 0) {
        xerbla("CHETRD_HB2ST", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        hous[0] = CMPLXF(1.0f, 0.0f);
        work[0] = CMPLXF(1.0f, 0.0f);
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

    /*
     * Case KD=0:
     * The matrix is diagonal. We just copy it (convert to real for
     * complex because D is f32 and the imaginary part should be 0)
     * and store it in D. A sequential code here is better or
     * in a parallel environment it might need two cores for D and E
     */
    if (kd == 0) {
        for (i = 0; i < n; i++) {
            D[i] = crealf(AB[abdpos + i * ldab]);
        }
        for (i = 0; i < n - 1; i++) {
            E[i] = rzero;
        }

        hous[0] = CMPLXF(1.0f, 0.0f);
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    /*
     * Case KD=1:
     * The matrix is already Tridiagonal. We have to make diagonal
     * and offdiagonal elements real, and store them in D and E.
     * For that, for real precision just copy the diag and offdiag
     * to D and E while for the COMPLEX case the bulge chasing is
     * performed to convert the hermetian tridiagonal to symmetric
     * tridiagonal. A simpler conversion formula might be used, but then
     * updating the Q matrix will be required and based if Q is generated
     * or not this might complicate the story.
     */
    if (kd == 1) {
        for (i = 0; i < n; i++) {
            D[i] = crealf(AB[abdpos + i * ldab]);
        }

        if (upper) {
            for (i = 0; i < n - 1; i++) {
                tmp = AB[abofdpos + (i + 1) * ldab];
                abstmp = cabsf(tmp);
                AB[abofdpos + (i + 1) * ldab] = CMPLXF(abstmp, 0.0f);
                E[i] = abstmp;
                if (abstmp != rzero) {
                    tmp = tmp / CMPLXF(abstmp, 0.0f);
                } else {
                    tmp = one;
                }
                if (i < n - 2)
                    AB[abofdpos + (i + 2) * ldab] = AB[abofdpos + (i + 2) * ldab] * tmp;
            }
        } else {
            for (i = 0; i < n - 1; i++) {
                tmp = AB[abofdpos + i * ldab];
                abstmp = cabsf(tmp);
                AB[abofdpos + i * ldab] = CMPLXF(abstmp, 0.0f);
                E[i] = abstmp;
                if (abstmp != rzero) {
                    tmp = tmp / CMPLXF(abstmp, 0.0f);
                } else {
                    tmp = one;
                }
                if (i < n - 2)
                    AB[abofdpos + (i + 1) * ldab] = AB[abofdpos + (i + 1) * ldab] * tmp;
            }
        }

        hous[0] = CMPLXF(1.0f, 0.0f);
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    /*
     * Main code start here.
     * Reduce the hermitian band of A to a tridiagonal matrix.
     */
    thgrsiz = n;
    grsiz = 1;
    shift = 3;
    (void)ceilf((f32)n / (f32)kd);  /* nbtiles: unused OpenMP placeholder */
    stepercol = (INT)ceilf((f32)shift / (f32)grsiz);
    thgrnb = (INT)ceilf((f32)(n - 1) / (f32)thgrsiz);

    clacpy("A", kd + 1, n, AB, ldab, &work[apos], lda);
    claset("A", kd, n, zero, zero, &work[awpos], lda);

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

                        chb2st_kernels(uplo, wantq, ttype,
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
        D[i] = crealf(work[dpos + i * lda]);
    }

    if (upper) {
        for (i = 0; i < n - 1; i++) {
            E[i] = crealf(work[ofdpos + (i + 1) * lda]);
        }
    } else {
        for (i = 0; i < n - 1; i++) {
            E[i] = crealf(work[ofdpos + i * lda]);
        }
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
