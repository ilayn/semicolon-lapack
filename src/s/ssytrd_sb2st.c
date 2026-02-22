/**
 * @file ssytrd_sb2st.c
 * @brief SSYTRD_SB2ST reduces a real symmetric band matrix to tridiagonal form.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
#include <math.h>

void ssytrd_sb2st(const char* stage1, const char* vect, const char* uplo,
                  const INT n, const INT kd,
                  f32* AB, const INT ldab,
                  f32* D, f32* E,
                  f32* hous, const INT lhous,
                  f32* work, const INT lwork, INT* info)
{
    const f32 rzero = 0.0f;
    const f32 zero = 0.0f;

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

    ib = ilaenv2stage(2, "SSYTRD_SB2ST", vect, n, kd, -1, -1);
    if (n == 0 || kd <= 1) {
        lhmin = 1;
        lwmin = 1;
    } else {
        lhmin = ilaenv2stage(3, "SSYTRD_SB2ST", vect, n, kd, ib, -1);
        lwmin = ilaenv2stage(4, "SSYTRD_SB2ST", vect, n, kd, ib, -1);
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
        hous[0] = (f32)lhmin;
        work[0] = (f32)lwmin;
    }

    if (*info != 0) {
        xerbla("SSYTRD_SB2ST", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        hous[0] = 1.0f;
        work[0] = 1.0f;
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

        hous[0] = 1.0f;
        work[0] = 1.0f;
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

        hous[0] = 1.0f;
        work[0] = 1.0f;
        return;
    }

    thgrsiz = n;
    grsiz = 1;
    shift = 3;
    (void)ceilf((f32)n / (f32)kd);  /* nbtiles: unused OpenMP placeholder */
    stepercol = (INT)ceilf((f32)shift / (f32)grsiz);
    thgrnb = (INT)ceilf((f32)(n - 1) / (f32)thgrsiz);

    slacpy("A", kd + 1, n, AB, ldab, &work[apos], lda);
    slaset("A", kd, n, zero, zero, &work[awpos], lda);

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

                        ssb2st_kernels(uplo, wantq, ttype,
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

    work[0] = (f32)lwmin;
}
