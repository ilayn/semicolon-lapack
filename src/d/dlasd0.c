/**
 * @file dlasd0.c
 * @brief DLASD0 computes the singular values of a real upper bidiagonal
 *        n-by-m matrix B with diagonal d and off-diagonal e using
 *        divide and conquer.
 */

#include "semicolon_lapack_double.h"
#include <stddef.h>

/**
 * Using a divide and conquer approach, DLASD0 computes the singular
 * value decomposition (SVD) of a real upper bidiagonal N-by-M
 * matrix B with diagonal D and offdiagonal E, where M = N + SQRE.
 * The algorithm computes orthogonal matrices U and VT such that
 * B = U * S * VT. The singular values S are overwritten on D.
 *
 * A related subroutine, DLASDA, computes only the singular values,
 * and optionally, the singular vectors in compact form.
 *
 * @param[in]     n       Row dimension of upper bidiagonal matrix.
 * @param[in]     sqre    = 0: bidiagonal matrix has column dimension M = N.
 *                         = 1: bidiagonal matrix has column dimension M = N+1.
 * @param[in,out] D       Array of dimension n. Main diagonal on entry,
 *                        singular values on exit.
 * @param[in,out] E       Array of dimension m-1. Offdiagonal entries.
 * @param[in,out] U       Array (ldu, n). Left singular vectors if U passed
 *                        in as N-by-N identity.
 * @param[in]     ldu     Leading dimension of U.
 * @param[in,out] VT      Array (ldvt, m). Right singular vectors transposed
 *                        if VT passed in as M-by-M identity.
 * @param[in]     ldvt    Leading dimension of VT.
 * @param[in]     smlsiz  Maximum size of subproblems at bottom of tree.
 * @param[out]    IWORK   Integer array of dimension 8*n.
 * @param[out]    work    Double array of dimension 3*m^2 + 2*m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: not converged.
 */
void dlasd0(const int n, const int sqre, f64* restrict D,
            f64* restrict E, f64* restrict U, const int ldu,
            f64* restrict VT, const int ldvt, const int smlsiz,
            int* restrict IWORK, f64* restrict work, int* info)
{
    int i, ic, idxq, idxqc, inode, itemp, iwk;
    int j, lf, ll, lvl, m, ncc, nd, ndb1, ndiml, ndimr;
    int nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqrei;
    f64 alpha, beta;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (sqre < 0 || sqre > 1) {
        *info = -2;
    }

    m = n + sqre;

    if (ldu < n) {
        *info = -6;
    } else if (ldvt < m) {
        *info = -8;
    } else if (smlsiz < 3) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DLASD0", -(*info));
        return;
    }

    if (n <= smlsiz) {
        dlasdq("U", sqre, n, m, n, 0, D, E, VT, ldvt, U, ldu, NULL, 1, work, info);
        return;
    }

    inode = 0;
    ndiml = inode + n;
    ndimr = ndiml + n;
    idxq = ndimr + n;
    iwk = idxq + n;

    dlasdt(n, &nlvl, &nd, &IWORK[inode], &IWORK[ndiml], &IWORK[ndimr], smlsiz);

    ndb1 = (nd + 1) / 2;
    ncc = 0;

    for (i = ndb1; i <= nd; i++) {
        ic = IWORK[inode + i - 1];
        nl = IWORK[ndiml + i - 1];
        nlp1 = nl + 1;
        nr = IWORK[ndimr + i - 1];
        nlf = ic - nl;
        nrf = ic + 1;
        sqrei = 1;

        dlasdq("U", sqrei, nl, nlp1, nl, ncc,
               &D[nlf], &E[nlf],
               &VT[nlf + nlf * ldvt], ldvt,
               &U[nlf + nlf * ldu], ldu,
               NULL, 1,
               work, info);
        if (*info != 0) {
            return;
        }

        itemp = idxq + nlf;
        for (j = 0; j < nl; j++) {
            IWORK[itemp + j] = j;
        }

        if (i == nd) {
            sqrei = sqre;
        } else {
            sqrei = 1;
        }
        nrp1 = nr + sqrei;

        dlasdq("U", sqrei, nr, nrp1, nr, ncc,
               &D[nrf], &E[nrf],
               &VT[nrf + nrf * ldvt], ldvt,
               &U[nrf + nrf * ldu], ldu,
               NULL, 1,
               work, info);
        if (*info != 0) {
            return;
        }

        itemp = idxq + nrf;
        for (j = 0; j < nr; j++) {
            IWORK[itemp + j] = j;
        }
    }

    for (lvl = nlvl; lvl >= 1; lvl--) {
        if (lvl == 1) {
            lf = 1;
            ll = 1;
        } else {
            lf = 1 << (lvl - 1);
            ll = 2 * lf - 1;
        }

        for (i = lf; i <= ll; i++) {
            ic = IWORK[inode + i - 1];
            nl = IWORK[ndiml + i - 1];
            nr = IWORK[ndimr + i - 1];
            nlf = ic - nl;
            if (sqre == 0 && i == ll) {
                sqrei = sqre;
            } else {
                sqrei = 1;
            }
            idxqc = idxq + nlf;
            alpha = D[ic];
            beta = E[ic];

            dlasd1(nl, nr, sqrei, &D[nlf], &alpha, &beta,
                   &U[nlf + nlf * ldu], ldu,
                   &VT[nlf + nlf * ldvt], ldvt,
                   &IWORK[idxqc], &IWORK[iwk], work, info);

            if (*info != 0) {
                return;
            }
        }
    }
}
