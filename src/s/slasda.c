/**
 * @file slasda.c
 * @brief SLASDA computes the singular value decomposition (SVD) of a real
 *        upper bidiagonal matrix with diagonal d and off-diagonal e using
 *        divide and conquer. Used by sbdsdc.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"

/**
 * Using a divide and conquer approach, SLASDA computes the singular
 * value decomposition (SVD) of a real upper bidiagonal `n`-by-`m` matrix
 * B with diagonal `D` and offdiagonal `E`, where `m = n + sqre`. The
 * algorithm computes the singular values in the SVD `B = U * S * VT`.
 * The orthogonal matrices U and VT are optionally computed in
 * compact form.
 *
 * A related subroutine, SLASD0, computes the singular values and
 * the singular vectors in explicit form.
 *
 * @param[in]     icompq Specifies whether singular vectors are to be computed
 *                       in compact form, as follows
 *                       `icompq=0`: Compute singular values only.
 *                       `icompq=1`: Compute singular vectors of upper bidiagonal
 *                       matrix in compact form.
 * @param[in]     smlsiz The maximum size of the subproblems at the bottom of the
 *                       computation tree.
 * @param[in]     n      The row dimension of the upper bidiagonal matrix. This is
 *                       also the dimension of the main diagonal array `D`.
 * @param[in]     sqre   Specifies the column dimension of the bidiagonal matrix.
 *                       `sqre=0`: The bidiagonal matrix has column dimension `m=n`;
 *                       `sqre=1`: The bidiagonal matrix has column dimension `m=n+1`.
 * @param[in,out] D      Array of dimension (`n`). On entry `D` contains the main
 *                       diagonal of the bidiagonal matrix. On exit `D`, if `info=0`,
 *                       contains its singular values.
 * @param[in]     E      Array of dimension (`m-1`). Contains the subdiagonal entries
 *                       of the bidiagonal matrix. On exit, `E` has been destroyed.
 * @param[out]    U      Array of dimension (`ldu`, `smlsiz`) if `icompq=1`, and not
 *                       referenced if `icompq=0`. If `icompq=1`, on exit, `U` contains
 *                       the left singular vector matrices of all subproblems at the
 *                       bottom level.
 * @param[in]     ldu    `ldu>=n`. The leading dimension of arrays `U`, `VT`, `DIFL`,
 *                       `DIFR`, `POLES`, `GIVNUM`, and `Z`.
 * @param[out]    VT     Array of dimension (`ldu`, `smlsiz+1`) if `icompq=1`, and not
 *                       referenced if `icompq=0`. If `icompq=1`, on exit, `VT**T`
 *                       contains the right singular vector matrices of all subproblems
 *                       at the bottom level.
 * @param[out]    K      Integer array of dimension (`n`) if `icompq=1` and dimension 1
 *                       if `icompq=0`. If `icompq=1`, on exit, `K[i]` is the dimension
 *                       of the i-th secular equation on the computation tree.
 * @param[out]    DIFL   Array of dimension (`ldu`, `nlvl`), where
 *                       `nlvl = floor(log_2(n/smlsiz))`.
 * @param[out]    DIFR   Array of dimension (`ldu`, `2*nlvl`) if `icompq=1` and
 *                       dimension (`n`) if `icompq=0`.
 *                       If `icompq=1`, on exit, `DIFL[0:n-1, i]` and
 *                       `DIFR[0:n-1, 2*i]` record distances between singular values
 *                       on the i-th level and singular values on the (i-1)-th level, and
 *                       `DIFR[0:n-1, 2*i+1]` contains the normalizing factors for
 *                       the right singular vector matrix. See SLASD8 for details.
 * @param[out]    Z      Array of dimension (`ldu`, `nlvl`) if `icompq=1` and
 *                       dimension (`n`) if `icompq=0`.
 *                       The first `K` elements of `Z[0, i]` contain the components of
 *                       the deflation-adjusted updating row vector for subproblems
 *                       on the i-th level.
 * @param[out]    POLES  Array of dimension (`ldu`, `2*nlvl`) if `icompq=1`, and not
 *                       referenced if `icompq=0`. If `icompq=1`, on exit,
 *                       `POLES[0, 2*i]` and `POLES[0, 2*i+1]` contain the new and old
 *                       singular values involved in the secular equations on the i-th
 *                       level.
 * @param[out]    GIVPTR Integer array of dimension (`n`) if `icompq=1`, and not
 *                       referenced if `icompq=0`. If `icompq=1`, on exit, `GIVPTR[i]`
 *                       records the number of Givens rotations performed on the i-th
 *                       problem on the computation tree.
 * @param[out]    GIVCOL Integer array of dimension (`ldgcol`, `2*nlvl`) if `icompq=1`,
 *                       and not referenced if `icompq=0`. If `icompq=1`, on exit, for
 *                       each i, `GIVCOL[0, 2*i]` and `GIVCOL[0, 2*i+1]` record the
 *                       locations of Givens rotations performed on the i-th level on the
 *                       computation tree.
 * @param[in]     ldgcol `ldgcol>=n`. The leading dimension of arrays `GIVCOL` and `PERM`.
 * @param[out]    PERM   Integer array of dimension (`ldgcol`, `nlvl`) if `icompq=1`, and
 *                       not referenced if `icompq=0`. If `icompq=1`, on exit,
 *                       `PERM[0, i]` records permutations done on the i-th level of the
 *                       computation tree.
 * @param[out]    GIVNUM Array of dimension (`ldu`, `2*nlvl`) if `icompq=1`, and not
 *                       referenced if `icompq=0`. If `icompq=1`, on exit, for each i,
 *                       `GIVNUM[0, 2*i]` and `GIVNUM[0, 2*i+1]` record the C- and S-
 *                       values of Givens rotations performed on the i-th level on
 *                       the computation tree.
 * @param[out]    C      Array of dimension (`n`) if `icompq=1`, and dimension 1 if
 *                       `icompq=0`. If `icompq=1` and the i-th subproblem is not square,
 *                       on exit, `C[i]` contains the C-value of a Givens rotation
 *                       related to the right null space of the i-th subproblem.
 * @param[out]    S      Array of dimension (`n`) if `icompq=1`, and dimension 1 if
 *                       `icompq=0`. If `icompq=1` and the i-th subproblem is not square,
 *                       on exit, `S[i]` contains the S-value of a Givens rotation
 *                       related to the right null space of the i-th subproblem.
 * @param[out]    work   Array of dimension (`6*n + (smlsiz+1)*(smlsiz+1)`).
 * @param[out]    IWORK  Integer array of dimension (`7*n`).
 * @param[out]    info   `info=0`: successful exit.
 *                       `info<0`: if `info=-i`, the i-th argument had an illegal value.
 *                       `info>0`: if `info=1`, a singular value did not converge.
 *
 * @par Contributors:
 * @rst
 * Ming Gu and Huan Ren, Computer Science Division, University of
 * California at Berkeley, USA
 * @endrst
 */
void slasda(const INT icompq, const INT smlsiz, const INT n, const INT sqre,
            f32* restrict D, f32* restrict E,
            f32* restrict U, const INT ldu,
            f32* restrict VT, INT* restrict K,
            f32* restrict DIFL, f32* restrict DIFR,
            f32* restrict Z, f32* restrict POLES,
            INT* restrict GIVPTR, INT* restrict GIVCOL,
            const INT ldgcol, INT* restrict PERM,
            f32* restrict GIVNUM,
            f32* restrict C, f32* restrict S,
            f32* restrict work, INT* restrict IWORK, INT* info)
{
    INT i, ic, idxq, idxqi, inode, itemp, iwk;
    INT j, lf, ll, lvl, lvl2, m, ncc, nd, ndb1, ndiml, ndimr;
    INT nl, nlf, nlp1, nlvl, nr, nrf, nrp1, nru;
    INT nwork1, nwork2, smlszp, sqrei, vf, vfi, vl, vli;
    f32 alpha, beta;

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
        xerbla("SLASDA", -(*info));
        return;
    }

    m = n + sqre;

    if (n <= smlsiz) {
        if (icompq == 0) {
            slasdq("U", sqre, n, 0, 0, 0, D, E, VT, ldu, NULL, 1, NULL, 1, work, info);
        } else {
            slasdq("U", sqre, n, m, n, 0, D, E, VT, ldu, U, ldu, NULL, 1, work, info);
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

    slasdt(n, &nlvl, &nd, &IWORK[inode], &IWORK[ndiml], &IWORK[ndimr], smlsiz);

    ndb1 = (nd + 1) / 2;

    for (i = ndb1 - 1; i < nd; i++) {
        ic = IWORK[inode + i];
        nl = IWORK[ndiml + i];
        nlp1 = nl + 1;
        nr = IWORK[ndimr + i];
        nlf = ic - nl;
        nrf = ic + 1;
        idxqi = idxq + nlf;
        vfi = vf + nlf;
        vli = vl + nlf;
        sqrei = 1;

        if (icompq == 0) {
            slaset("A", nlp1, nlp1, 0.0f, 1.0f, &work[nwork1], smlszp);
            slasdq("U", sqrei, nl, nlp1, nru, ncc,
                   &D[nlf], &E[nlf],
                   &work[nwork1], smlszp,
                   NULL, 1,
                   NULL, 1,
                   &work[nwork2], info);
            itemp = nwork1 + nl * smlszp;
            cblas_scopy(nlp1, &work[nwork1], 1, &work[vfi], 1);
            cblas_scopy(nlp1, &work[itemp], 1, &work[vli], 1);
        } else {
            slaset("A", nl, nl, 0.0f, 1.0f, &U[nlf + 0 * ldu], ldu);
            slaset("A", nlp1, nlp1, 0.0f, 1.0f, &VT[nlf + 0 * ldu], ldu);
            slasdq("U", sqrei, nl, nlp1, nl, ncc,
                   &D[nlf], &E[nlf],
                   &VT[nlf + 0 * ldu], ldu,
                   &U[nlf + 0 * ldu], ldu,
                   NULL, 1,
                   &work[nwork1], info);
            cblas_scopy(nlp1, &VT[nlf + 0 * ldu], 1, &work[vfi], 1);
            cblas_scopy(nlp1, &VT[nlf + (nlp1 - 1) * ldu], 1, &work[vli], 1);
        }
        if (*info != 0) {
            return;
        }

        for (j = 0; j < nl; j++) {
            IWORK[idxqi + j] = j;
        }

        if (i == nd - 1 && sqre == 0) {
            sqrei = 0;
        } else {
            sqrei = 1;
        }
        idxqi = idxqi + nlp1;
        vfi = vfi + nlp1;
        vli = vli + nlp1;
        nrp1 = nr + sqrei;

        if (icompq == 0) {
            slaset("A", nrp1, nrp1, 0.0f, 1.0f, &work[nwork1], smlszp);
            slasdq("U", sqrei, nr, nrp1, nru, ncc,
                   &D[nrf], &E[nrf],
                   &work[nwork1], smlszp,
                   NULL, 1,
                   NULL, 1,
                   &work[nwork2], info);
            itemp = nwork1 + (nrp1 - 1) * smlszp;
            cblas_scopy(nrp1, &work[nwork1], 1, &work[vfi], 1);
            cblas_scopy(nrp1, &work[itemp], 1, &work[vli], 1);
        } else {
            slaset("A", nr, nr, 0.0f, 1.0f, &U[nrf + 0 * ldu], ldu);
            slaset("A", nrp1, nrp1, 0.0f, 1.0f, &VT[nrf + 0 * ldu], ldu);
            slasdq("U", sqrei, nr, nrp1, nr, ncc,
                   &D[nrf], &E[nrf],
                   &VT[nrf + 0 * ldu], ldu,
                   &U[nrf + 0 * ldu], ldu,
                   NULL, 1,
                   &work[nwork1], info);
            cblas_scopy(nrp1, &VT[nrf + 0 * ldu], 1, &work[vfi], 1);
            cblas_scopy(nrp1, &VT[nrf + (nrp1 - 1) * ldu], 1, &work[vli], 1);
        }
        if (*info != 0) {
            return;
        }

        for (j = 0; j < nr; j++) {
            IWORK[idxqi + j] = j;
        }
    }

    j = (1 << nlvl) - 1;

    for (lvl = nlvl - 1; lvl >= 0; lvl--) {
        lvl2 = lvl * 2;

        if (lvl == 0) {
            lf = 0;
            ll = 0;
        } else {
            lf = (1 << lvl) - 1;
            ll = 2 * lf;
        }

        for (i = lf; i <= ll; i++) {
            ic = IWORK[inode + i];
            nl = IWORK[ndiml + i];
            nr = IWORK[ndimr + i];
            nlf = ic - nl;
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
                slasd6(icompq, nl, nr, sqrei, &D[nlf],
                       &work[vfi], &work[vli], &alpha, &beta,
                       &IWORK[idxqi], PERM, &GIVPTR[0], GIVCOL,
                       ldgcol, GIVNUM, ldu, POLES, DIFL, DIFR, Z,
                       &K[0], &C[0], &S[0], &work[nwork1],
                       &IWORK[iwk], info);
            } else {
                j = j - 1;
                slasd6(icompq, nl, nr, sqrei, &D[nlf],
                       &work[vfi], &work[vli], &alpha, &beta,
                       &IWORK[idxqi],
                       &PERM[nlf + lvl * ldgcol],
                       &GIVPTR[j],
                       &GIVCOL[nlf + lvl2 * ldgcol], ldgcol,
                       &GIVNUM[nlf + lvl2 * ldu], ldu,
                       &POLES[nlf + lvl2 * ldu],
                       &DIFL[nlf + lvl * ldu],
                       &DIFR[nlf + lvl2 * ldu],
                       &Z[nlf + lvl * ldu],
                       &K[j], &C[j], &S[j],
                       &work[nwork1], &IWORK[iwk], info);
            }
            if (*info != 0) {
                return;
            }
        }
    }
}
