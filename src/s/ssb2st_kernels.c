/**
 * @file ssb2st_kernels.c
 * @brief SSB2ST_KERNELS is an internal routine used by SSYTRD_SB2ST.
 */

#include "semicolon_lapack_single.h"

void ssb2st_kernels(const char* uplo, const int wantz, const int ttype,
                    const int st, const int ed, const int sweep,
                    const int n, const int nb, const int ib,
                    float* A, const int lda,
                    float* V, float* tau, const int ldvt,
                    float* work)
{
    const float zero = 0.0f;
    const float one = 1.0f;

    int upper;
    int i, j1, j2, lm, ln, vpos, taupos, dpos, ofdpos;
    float ctmp;

    (void)ib;
    (void)ldvt;
    (void)wantz;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        dpos = 2 * nb;
        ofdpos = 2 * nb - 1;
    } else {
        dpos = 0;
        ofdpos = 1;
    }

    if (upper) {
        vpos = ((sweep - 1) % 2) * n + st - 1;
        taupos = ((sweep - 1) % 2) * n + st - 1;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = one;
            for (i = 1; i <= lm - 1; i++) {
                V[vpos + i] = A[(ofdpos - i) + (st - 1 + i) * lda];
                A[(ofdpos - i) + (st - 1 + i) * lda] = zero;
            }
            ctmp = A[ofdpos + (st - 1) * lda];
            slarfg(lm, &ctmp, &V[vpos + 1], 1, &tau[taupos]);
            A[ofdpos + (st - 1) * lda] = ctmp;

            lm = ed - st + 1;
            slarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 3) {
            lm = ed - st + 1;
            slarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;
            if (lm > 0) {
                slarfx("L", ln, lm, &V[vpos], tau[taupos],
                       &A[(dpos - nb) + (j1 - 1) * lda], lda - 1, work);

                vpos = ((sweep - 1) % 2) * n + j1 - 1;
                taupos = ((sweep - 1) % 2) * n + j1 - 1;

                V[vpos] = one;
                for (i = 1; i <= lm - 1; i++) {
                    V[vpos + i] = A[(dpos - nb - i) + (j1 - 1 + i) * lda];
                    A[(dpos - nb - i) + (j1 - 1 + i) * lda] = zero;
                }
                ctmp = A[(dpos - nb) + (j1 - 1) * lda];
                slarfg(lm, &ctmp, &V[vpos + 1], 1, &tau[taupos]);
                A[(dpos - nb) + (j1 - 1) * lda] = ctmp;

                slarfx("R", ln - 1, lm, &V[vpos], tau[taupos],
                       &A[(dpos - nb + 1) + (j1 - 1) * lda], lda - 1, work);
            }
        }

    } else {
        vpos = ((sweep - 1) % 2) * n + st - 1;
        taupos = ((sweep - 1) % 2) * n + st - 1;

        if (ttype == 1) {
            lm = ed - st + 1;

            V[vpos] = one;
            for (i = 1; i <= lm - 1; i++) {
                V[vpos + i] = A[(ofdpos + i) + (st - 2) * lda];
                A[(ofdpos + i) + (st - 2) * lda] = zero;
            }
            slarfg(lm, &A[ofdpos + (st - 2) * lda], &V[vpos + 1], 1, &tau[taupos]);

            lm = ed - st + 1;

            slarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 3) {
            lm = ed - st + 1;

            slarfy(uplo, lm, &V[vpos], 1, tau[taupos],
                   &A[dpos + (st - 1) * lda], lda - 1, work);
        }

        if (ttype == 2) {
            j1 = ed + 1;
            j2 = (ed + nb < n) ? ed + nb : n;
            ln = ed - st + 1;
            lm = j2 - j1 + 1;

            if (lm > 0) {
                slarfx("R", lm, ln, &V[vpos], tau[taupos],
                       &A[(dpos + nb) + (st - 1) * lda], lda - 1, work);

                vpos = ((sweep - 1) % 2) * n + j1 - 1;
                taupos = ((sweep - 1) % 2) * n + j1 - 1;

                V[vpos] = one;
                for (i = 1; i <= lm - 1; i++) {
                    V[vpos + i] = A[(dpos + nb + i) + (st - 1) * lda];
                    A[(dpos + nb + i) + (st - 1) * lda] = zero;
                }
                slarfg(lm, &A[(dpos + nb) + (st - 1) * lda], &V[vpos + 1], 1, &tau[taupos]);

                slarfx("L", lm, ln - 1, &V[vpos], tau[taupos],
                       &A[(dpos + nb - 1) + st * lda], lda - 1, work);
            }
        }
    }
}
