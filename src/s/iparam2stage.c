/**
 * @file iparam2stage.c
 * @brief IPARAM2STAGE sets problem and machine dependent parameters for 2-stage algorithms.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <string.h>

int iparam2stage(const int ispec, const char* name, const char* opts,
                 const int ni, const int nbi, const int ibi, const int nxi)
{
    int kd, ib, lhous, lwork, nthreads;
    int factoptnb, qroptnb, lqoptnb;
    int rprec, cprec;
    char prec, algo[4], stag[6], vect;
    char subnam[13];
    int i, ic;

    if (ispec < 17 || ispec > 21) {
        return -1;
    }

    nthreads = 1;

    if (ispec != 19) {
        strncpy(subnam, name, 12);
        subnam[12] = '\0';

        for (i = 0; i < 12 && subnam[i] != '\0'; i++) {
            ic = (unsigned char)subnam[i];
            if (ic >= 97 && ic <= 122) {
                subnam[i] = (char)(ic - 32);
            }
        }

        prec = subnam[0];
        algo[0] = subnam[3];
        algo[1] = subnam[4];
        algo[2] = subnam[5];
        algo[3] = '\0';
        stag[0] = subnam[7];
        stag[1] = subnam[8];
        stag[2] = subnam[9];
        stag[3] = subnam[10];
        stag[4] = subnam[11];
        stag[5] = '\0';

        rprec = (prec == 'S' || prec == 'D');
        cprec = (prec == 'C' || prec == 'Z');

        if (!(rprec || cprec)) {
            return -1;
        }
    }

    if (ispec == 17 || ispec == 18) {
        if (nthreads > 4) {
            if (cprec) {
                kd = 128;
                ib = 32;
            } else {
                kd = 160;
                ib = 40;
            }
        } else if (nthreads > 1) {
            /* LAPACK has identical branches */
            kd = 64;
            ib = 32;
        } else {
            if (cprec) {
                kd = 16;
                ib = 16;
            } else {
                kd = 32;
                ib = 16;
            }
        }
        if (ispec == 17) return kd;
        if (ispec == 18) return ib;

    } else if (ispec == 19) {
        vect = opts[0];
        if (vect == 'N' || vect == 'n') {
            lhous = (4 * ni > 1) ? 4 * ni : 1;
        } else {
            lhous = ((4 * ni > 1) ? 4 * ni : 1) + ibi;
        }
        if (lhous >= 0) {
            return lhous;
        } else {
            return -1;
        }

    } else if (ispec == 20) {
        lwork = -1;
        qroptnb = lapack_get_nb("GEQRF");
        lqoptnb = lapack_get_nb("GELQF");
        factoptnb = (qroptnb > lqoptnb) ? qroptnb : lqoptnb;

        if (strcmp(algo, "TRD") == 0) {
            if (strcmp(stag, "2STAG") == 0) {
                int tmp1 = 2 * nbi * nbi;
                int tmp2 = nbi * nthreads;
                int maxval = (tmp1 > tmp2) ? tmp1 : tmp2;
                int factmax = (nbi + 1 > factoptnb) ? nbi + 1 : factoptnb;
                lwork = ni * nbi + ni * factmax + maxval + (nbi + 1) * ni;
            } else if (strcmp(stag, "HE2HB") == 0 || strcmp(stag, "SY2SB") == 0) {
                int factmax = (nbi > factoptnb) ? nbi : factoptnb;
                lwork = ni * nbi + ni * factmax + 2 * nbi * nbi;
            } else if (strcmp(stag, "HB2ST") == 0 || strcmp(stag, "SB2ST") == 0) {
                lwork = (2 * nbi + 1) * ni + nbi * nthreads;
            }
        } else if (strcmp(algo, "BRD") == 0) {
            if (strcmp(stag, "2STAG") == 0) {
                int tmp1 = 2 * nbi * nbi;
                int tmp2 = nbi * nthreads;
                int maxval = (tmp1 > tmp2) ? tmp1 : tmp2;
                int factmax = (nbi + 1 > factoptnb) ? nbi + 1 : factoptnb;
                lwork = 2 * ni * nbi + ni * factmax + maxval + (nbi + 1) * ni;
            } else if (strcmp(stag, "GE2GB") == 0) {
                int factmax = (nbi > factoptnb) ? nbi : factoptnb;
                lwork = ni * nbi + ni * factmax + 2 * nbi * nbi;
            } else if (strcmp(stag, "GB2BD") == 0) {
                lwork = (3 * nbi + 1) * ni + nbi * nthreads;
            }
        }
        lwork = (lwork > 1) ? lwork : 1;

        if (lwork > 0) {
            return lwork;
        } else {
            return -1;
        }

    } else if (ispec == 21) {
        return nxi;
    }

    return -1;
}
