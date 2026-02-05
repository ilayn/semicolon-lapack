/**
 * @file dlasda.c
 * @brief DLASDA computes the singular value decomposition (SVD) of a real
 *        upper bidiagonal matrix with diagonal d and off-diagonal e using
 *        divide and conquer. Used by dbdsdc.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>

void dlasda(const int icompq, const int smlsiz, const int n, const int sqre,
            double* const restrict D, double* const restrict E,
            double* const restrict U, const int ldu,
            double* const restrict VT, int* const restrict K,
            double* const restrict DIFL, double* const restrict DIFR,
            double* const restrict Z, double* const restrict POLES,
            int* const restrict GIVPTR, int* const restrict GIVCOL,
            const int ldgcol, int* const restrict PERM,
            double* const restrict GIVNUM,
            double* const restrict C, double* const restrict S,
            double* const restrict work, int* const restrict IWORK, int* info)
{
    int i, i1, ic, idxq, idxqi, im1, inode, itemp, iwk;
    int j, lf, ll, lvl, lvl2, m, ncc, nd, ndb1, ndiml, ndimr;
    int nl, nlf, nlp1, nlvl, nr, nrf, nrp1, nru;
    int nwork1, nwork2, smlszp, sqrei, vf, vfi, vl, vli;
    double alpha, beta;

    *info = 0;

    if (icompq < 0 || icompq > 1) {
        *info = -1;
    } else if (smlsiz < 3) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (sqre < 0 || sqre > 1) {
        *info = -4;
    } else if (ldu < (n + sqre)) {
        *info = -8;
    } else if (ldgcol < n) {
        *info = -17;
    }
    if (*info != 0) {
        xerbla("DLASDA", -(*info));
        return;
    }

    m = n + sqre;

    if (n <= smlsiz) {
        if (icompq == 0) {
            dlasdq("U", sqre, n, 0, 0, 0, D, E, VT, ldu, NULL, 1, NULL, 1, work, info);
        } else {
            dlasdq("U", sqre, n, m, n, 0, D, E, VT, ldu, U, ldu, NULL, 1, work, info);
        }
        return;
    }

    inode = 0;
    ndiml = inode + n;
    ndimr = ndiml + n;
    idxq = ndimr + n;
    iwk = idxq + n;

    ncc = 0;
    nru = 0;

    smlszp = smlsiz + 1;
    vf = 0;
    vl = vf + m;
    nwork1 = vl + m;
    nwork2 = nwork1 + smlszp * smlszp;

    dlasdt(n, &nlvl, &nd, &IWORK[inode], &IWORK[ndiml], &IWORK[ndimr], smlsiz);

    ndb1 = (nd + 1) / 2;

    for (i = ndb1; i <= nd; i++) {
        i1 = i - 1;
        ic = IWORK[inode + i1];
        nl = IWORK[ndiml + i1];
        nlp1 = nl + 1;
        nr = IWORK[ndimr + i1];
        nlf = ic - nl;
        nrf = ic + 1;
        idxqi = idxq + nlf;
        vfi = vf + nlf;
        vli = vl + nlf;
        sqrei = 1;

        if (icompq == 0) {
            dlaset("A", nlp1, nlp1, 0.0, 1.0, &work[nwork1], smlszp);
            dlasdq("U", sqrei, nl, nlp1, nru, ncc,
                   &D[nlf], &E[nlf],
                   &work[nwork1], smlszp,
                   NULL, 1,
                   NULL, 1,
                   &work[nwork2], info);
            itemp = nwork1 + nl * smlszp;
            cblas_dcopy(nlp1, &work[nwork1], 1, &work[vfi], 1);
            cblas_dcopy(nlp1, &work[itemp], 1, &work[vli], 1);
        } else {
            dlaset("A", nl, nl, 0.0, 1.0, &U[nlf + 0 * ldu], ldu);
            dlaset("A", nlp1, nlp1, 0.0, 1.0, &VT[nlf + 0 * ldu], ldu);
            dlasdq("U", sqrei, nl, nlp1, nl, ncc,
                   &D[nlf], &E[nlf],
                   &VT[nlf + 0 * ldu], ldu,
                   &U[nlf + 0 * ldu], ldu,
                   NULL, 1,
                   &work[nwork1], info);
            cblas_dcopy(nlp1, &VT[nlf + 0 * ldu], 1, &work[vfi], 1);
            cblas_dcopy(nlp1, &VT[nlf + (nlp1 - 1) * ldu], 1, &work[vli], 1);
        }
        if (*info != 0) {
            return;
        }

        for (j = 0; j < nl; j++) {
            IWORK[idxqi + j] = j + 1;
        }

        if (i == nd && sqre == 0) {
            sqrei = 0;
        } else {
            sqrei = 1;
        }
        idxqi = idxqi + nlp1;
        vfi = vfi + nlp1;
        vli = vli + nlp1;
        nrp1 = nr + sqrei;

        if (icompq == 0) {
            dlaset("A", nrp1, nrp1, 0.0, 1.0, &work[nwork1], smlszp);
            dlasdq("U", sqrei, nr, nrp1, nru, ncc,
                   &D[nrf], &E[nrf],
                   &work[nwork1], smlszp,
                   NULL, 1,
                   NULL, 1,
                   &work[nwork2], info);
            itemp = nwork1 + (nrp1 - 1) * smlszp;
            cblas_dcopy(nrp1, &work[nwork1], 1, &work[vfi], 1);
            cblas_dcopy(nrp1, &work[itemp], 1, &work[vli], 1);
        } else {
            dlaset("A", nr, nr, 0.0, 1.0, &U[nrf + 0 * ldu], ldu);
            dlaset("A", nrp1, nrp1, 0.0, 1.0, &VT[nrf + 0 * ldu], ldu);
            dlasdq("U", sqrei, nr, nrp1, nr, ncc,
                   &D[nrf], &E[nrf],
                   &VT[nrf + 0 * ldu], ldu,
                   &U[nrf + 0 * ldu], ldu,
                   NULL, 1,
                   &work[nwork1], info);
            cblas_dcopy(nrp1, &VT[nrf + 0 * ldu], 1, &work[vfi], 1);
            cblas_dcopy(nrp1, &VT[nrf + (nrp1 - 1) * ldu], 1, &work[vli], 1);
        }
        if (*info != 0) {
            return;
        }

        for (j = 0; j < nr; j++) {
            IWORK[idxqi + j] = j + 1;
        }
    }

    j = 1 << nlvl;

    for (lvl = nlvl; lvl >= 1; lvl--) {
        lvl2 = lvl * 2 - 1;

        if (lvl == 1) {
            lf = 1;
            ll = 1;
        } else {
            lf = 1 << (lvl - 1);
            ll = 2 * lf - 1;
        }

        for (i = lf; i <= ll; i++) {
            im1 = i - 1;
            ic = IWORK[inode + im1];
            nl = IWORK[ndiml + im1];
            nr = IWORK[ndimr + im1];
            nlf = ic - nl;
            nrf = ic + 1;
            if (i == ll) {
                sqrei = sqre;
            } else {
                sqrei = 1;
            }
            vfi = vf + nlf;
            vli = vl + nlf;
            idxqi = idxq + nlf;
            alpha = D[ic];
            beta = E[ic];

            if (icompq == 0) {
                dlasd6(icompq, nl, nr, sqrei, &D[nlf],
                       &work[vfi], &work[vli], &alpha, &beta,
                       &IWORK[idxqi], PERM, &GIVPTR[0], GIVCOL,
                       ldgcol, GIVNUM, ldu, POLES, DIFL, DIFR, Z,
                       &K[0], &C[0], &S[0], &work[nwork1],
                       &IWORK[iwk], info);
            } else {
                j = j - 1;
                dlasd6(icompq, nl, nr, sqrei, &D[nlf],
                       &work[vfi], &work[vli], &alpha, &beta,
                       &IWORK[idxqi],
                       &PERM[nlf + (lvl - 1) * ldgcol],
                       &GIVPTR[j - 1],
                       &GIVCOL[nlf + (lvl2 - 1) * ldgcol], ldgcol,
                       &GIVNUM[nlf + (lvl2 - 1) * ldu], ldu,
                       &POLES[nlf + (lvl2 - 1) * ldu],
                       &DIFL[nlf + (lvl - 1) * ldu],
                       &DIFR[nlf + (lvl2 - 1) * ldu],
                       &Z[nlf + (lvl - 1) * ldu],
                       &K[j - 1], &C[j - 1], &S[j - 1],
                       &work[nwork1], &IWORK[iwk], info);
            }
            if (*info != 0) {
                return;
            }
        }
    }
}
