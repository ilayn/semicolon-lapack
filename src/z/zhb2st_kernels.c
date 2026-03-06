/**
 * @file zhb2st_kernels.c
 * @brief ZHB2ST_KERNELS is an internal routine used by the ZHETRD_HB2ST subroutine.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHB2ST_KERNELS is an internal routine used by the ZHETRD_HB2ST
 * subroutine.
 *
 * @param[in]     uplo   'U' or 'L'
 * @param[in]     wantz  Indicates if eigenvectors are requested.
 * @param[in]     ttype  Internal type parameter.
 * @param[in]     st     Internal index parameter (1-based from caller).
 * @param[in]     ed     Internal index parameter (1-based from caller).
 * @param[in]     sweep  Internal index parameter (1-based from caller).
 * @param[in]     n      The order of the matrix A.
 * @param[in]     nb     The size of the band.
 * @param[in]     ib     Internal block size parameter.
 * @param[in,out] A      Double complex array. A pointer to the matrix A.
 * @param[in]     lda    The leading dimension of the matrix A.
 * @param[out]    V      Double complex array, dimension 2*n.
 * @param[out]    TAU    Double complex array, dimension 2*n.
 * @param[in]     ldvt   Leading dimension parameter for V/TAU storage.
 * @param[out]    WORK   Double complex array. Workspace of size nb.
 */
void zhb2st_kernels(
    const char* uplo,
    const INT wantz,
    const INT ttype,
    const INT st,
    const INT ed,
    const INT sweep,
    const INT n,
    const INT nb,
    const INT ib,
    c128* restrict A,
    const INT lda,
    c128* restrict V,
    c128* restrict TAU,
    const INT ldvt,
    c128* restrict WORK)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 ONE = CMPLX(1.0, 0.0);

    (void)wantz; (void)ib; (void)ldvt;

    INT upper;
    INT i, j1, j2, lm, ln, vpos, taupos, dpos, ofdpos;
    c128 ctmp;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        dpos   = 2 * nb;
        ofdpos = 2 * nb - 1;
    } else {
        dpos   = 0;
        ofdpos = 1;
    }

    /*
     * Upper case
     */
    if (upper) {

        vpos   = ((sweep - 1) % 2) * n + st - 1;
        taupos = ((sweep - 1) % 2) * n + st - 1;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = ONE;
            for (i = 1; i <= lm - 1; i++) {
                V[vpos + i]                              = conj(A[(ofdpos - i) + (st - 1 + i) * lda]);
                A[(ofdpos - i) + (st - 1 + i) * lda]     = ZERO;
            }
            ctmp = conj(A[ofdpos + (st - 1) * lda]);
            zlarfg(lm, &ctmp, &V[vpos + 1], 1, &TAU[taupos]);
            A[ofdpos + (st - 1) * lda] = ctmp;

            lm = ed - st + 1;
            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + (st - 1) * lda], lda - 1, WORK);
        }

        if (ttype == 3) {

            lm = ed - st + 1;
            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + (st - 1) * lda], lda - 1, WORK);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;
            if (lm > 0) {
                zlarfx("Left", ln, lm, &V[vpos],
                       conj(TAU[taupos]),
                       &A[(dpos - nb) + (j1 - 1) * lda], lda - 1, WORK);

                vpos   = ((sweep - 1) % 2) * n + j1 - 1;
                taupos = ((sweep - 1) % 2) * n + j1 - 1;

                V[vpos] = ONE;
                for (i = 1; i <= lm - 1; i++) {
                    V[vpos + i]                               = conj(A[(dpos - nb - i) + (j1 - 1 + i) * lda]);
                    A[(dpos - nb - i) + (j1 - 1 + i) * lda]   = ZERO;
                }
                ctmp = conj(A[(dpos - nb) + (j1 - 1) * lda]);
                zlarfg(lm, &ctmp, &V[vpos + 1], 1, &TAU[taupos]);
                A[(dpos - nb) + (j1 - 1) * lda] = ctmp;

                zlarfx("Right", ln - 1, lm, &V[vpos],
                       TAU[taupos],
                       &A[(dpos - nb + 1) + (j1 - 1) * lda], lda - 1, WORK);
            }
        }

    /*
     * Lower case
     */
    } else {

        vpos   = ((sweep - 1) % 2) * n + st - 1;
        taupos = ((sweep - 1) % 2) * n + st - 1;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = ONE;
            for (i = 1; i <= lm - 1; i++) {
                V[vpos + i]                          = A[(ofdpos + i) + (st - 2) * lda];
                A[(ofdpos + i) + (st - 2) * lda]     = ZERO;
            }
            zlarfg(lm, &A[ofdpos + (st - 2) * lda], &V[vpos + 1], 1,
                   &TAU[taupos]);

            lm = ed - st + 1;

            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + (st - 1) * lda], lda - 1, WORK);

        }

        if (ttype == 3) {
            lm = ed - st + 1;

            zlarfy(uplo, lm, &V[vpos], 1,
                   conj(TAU[taupos]),
                   &A[dpos + (st - 1) * lda], lda - 1, WORK);

        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;

            if (lm > 0) {
                zlarfx("Right", lm, ln, &V[vpos],
                       TAU[taupos], &A[(dpos + nb) + (st - 1) * lda],
                       lda - 1, WORK);

                vpos   = ((sweep - 1) % 2) * n + j1 - 1;
                taupos = ((sweep - 1) % 2) * n + j1 - 1;

                V[vpos] = ONE;
                for (i = 1; i <= lm - 1; i++) {
                    V[vpos + i]                      = A[(dpos + nb + i) + (st - 1) * lda];
                    A[(dpos + nb + i) + (st - 1) * lda]     = ZERO;
                }
                zlarfg(lm, &A[(dpos + nb) + (st - 1) * lda], &V[vpos + 1], 1,
                       &TAU[taupos]);

                zlarfx("Left", lm, ln - 1, &V[vpos],
                       conj(TAU[taupos]),
                       &A[(dpos + nb - 1) + st * lda], lda - 1, WORK);

            }
        }
    }

    return;
}
