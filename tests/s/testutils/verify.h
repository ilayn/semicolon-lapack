/**
 * @file verify.h
 * @brief Prototypes for LAPACK test verification routines.
 *
 * These are ports of the verification routines from LAPACK/TESTING/LIN/
 * and LAPACK/TESTING/EIG/.
 */

#ifndef VERIFY_H
#define VERIFY_H

#include <stdint.h>
#include "semicolon_lapack/semicolon_lapack.h"

/* General (SGE) verification routines */
void sget01(const INT m, const INT n, const f32 * const restrict A, const INT lda,
            f32 * const restrict AFAC, const INT ldafac, const INT * const restrict ipiv,
            f32 * const restrict rwork, f32 *resid);

void sget02(const char* trans, const INT m, const INT n, const INT nrhs,
            const f32 * const restrict A, const INT lda, const f32 * const restrict X,
            const INT ldx, f32 * const restrict B, const INT ldb,
            f32 * const restrict rwork, f32 *resid);

void sget03(const INT n, const f32 * const restrict A, const INT lda,
            const f32 * const restrict AINV, const INT ldainv, f32 * const restrict work,
            const INT ldwork, f32 * const restrict rwork, f32 *rcond, f32 *resid);

void sget04(const INT n, const INT nrhs, const f32 * const restrict X, const INT ldx,
            const f32 * const restrict XACT, const INT ldxact, const f32 rcond,
            f32 *resid);

f32 sget06(const f32 rcond, const f32 rcondc);

void sget07(const char* trans, const INT n, const INT nrhs,
            const f32 * const restrict A, const INT lda,
            const f32 * const restrict B, const INT ldb,
            const f32 * const restrict X, const INT ldx,
            const f32 * const restrict XACT, const INT ldxact,
            const f32 * const restrict ferr, const INT chkferr,
            const f32 * const restrict berr, f32 * const restrict reslts);

void sget08(const char* trans, const INT m, const INT n, const INT nrhs,
            const f32* A, const INT lda, const f32* X, const INT ldx,
            f32* B, const INT ldb, f32* rwork, f32* resid);

/* Banded (GB) verification routines */
void sgbt01(INT m, INT n, INT kl, INT ku,
            const f32* A, INT lda,
            const f32* AFAC, INT ldafac,
            const INT* ipiv,
            f32* work,
            f32* resid);

void sgbt02(const char* trans, INT m, INT n, INT kl, INT ku, INT nrhs,
            const f32* A, INT lda,
            const f32* X, INT ldx,
            f32* B, INT ldb,
            f32* rwork,
            f32* resid);

void sgbt05(const char* trans, INT n, INT kl, INT ku, INT nrhs,
            const f32* AB, INT ldab,
            const f32* B, INT ldb,
            const f32* X, INT ldx,
            const f32* XACT, INT ldxact,
            const f32* FERR,
            const f32* BERR,
            f32* reslts);

/* Positive definite banded (PB) verification routines */
void spbt01(const char* uplo, const INT n, const INT kd,
            const f32* A, const INT lda,
            f32* AFAC, const INT ldafac,
            f32* rwork, f32* resid);

void spbt02(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const f32* A, const INT lda,
            const f32* X, const INT ldx,
            f32* B, const INT ldb,
            f32* rwork, f32* resid);

void spbt05(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const f32* AB, const INT ldab,
            const f32* B, const INT ldb,
            const f32* X, const INT ldx,
            const f32* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

/* Tridiagonal (SGT) verification routines */
void sgtt01(const INT n, const f32 * const restrict DL, const f32 * const restrict D,
            const f32 * const restrict DU, const f32 * const restrict DLF,
            const f32 * const restrict DF, const f32 * const restrict DUF,
            const f32 * const restrict DU2, const INT * const restrict ipiv,
            f32 * const restrict work, const INT ldwork, f32 *resid);

void sgtt02(const char* trans, const INT n, const INT nrhs,
            const f32 * const restrict DL, const f32 * const restrict D,
            const f32 * const restrict DU, const f32 * const restrict X, const INT ldx,
            f32 * const restrict B, const INT ldb, f32 *resid);

void sgtt05(const char* trans, const INT n, const INT nrhs,
            const f32 * const restrict DL, const f32 * const restrict D,
            const f32 * const restrict DU, const f32 * const restrict B, const INT ldb,
            const f32 * const restrict X, const INT ldx,
            const f32 * const restrict XACT, const INT ldxact,
            const f32 * const restrict ferr, const f32 * const restrict berr,
            f32 * const restrict reslts);

/* Matrix generation routines */
void slatb4(const char* path, const INT imat, const INT m, const INT n,
            char* type, INT* kl, INT* ku, f32* anorm, INT* mode,
            f32* cndnum, char* dist);

void slatms(const INT m, const INT n, const char* dist,
            const char* sym, f32 *d, const INT mode, const f32 cond,
            const f32 dmax, const INT kl, const INT ku, const char* pack,
            f32 *A, const INT lda, f32 *work, INT* info,
            uint64_t state[static 4]);

void slatmt(const INT m, const INT n, const char* dist,
            const char* sym, f32* d, const INT mode,
            const f32 cond, const f32 dmax, const INT rank,
            const INT kl, const INT ku, const char* pack,
            f32* A, const INT lda, f32* work, INT* info,
            uint64_t state[static 4]);

void slatm1(const INT mode, const f32 cond, const INT irsign,
            const INT idist, f32* d, const INT n, INT* info,
            uint64_t state[static 4]);

void slagge(const INT m, const INT n, const INT kl, const INT ku,
            const f32* d, f32* A, const INT lda,
            f32* work, INT* info, uint64_t state[static 4]);

void slagsy(const INT n, const INT k, const f32* d, f32* A,
            const INT lda, f32* work, INT* info,
            uint64_t state[static 4]);

void slarot(const INT lrows, const INT lleft, const INT lright,
            const INT nl, const f32 c, const f32 s,
            f32* A, const INT lda, f32* xleft, f32* xright);

void slarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const INT m, const INT n, const INT kl,
            const INT ku, const INT nrhs, const f32* A, const INT lda,
            f32* X, const INT ldx, f32* B, const INT ldb,
            INT* info, uint64_t state[static 4]);

void slaord(const char* job, const INT n, f32* X, const INT incx);

/* Symmetric (SY) verification routines */
void slavsy(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f32* const restrict A, const INT lda,
            const INT* const restrict ipiv, f32* const restrict B, const INT ldb, INT* info);

void ssyt01(const char* uplo, const INT n, const f32* const restrict A, const INT lda,
            const f32* const restrict AFAC, const INT ldafac, const INT* const restrict ipiv,
            f32* const restrict C, const INT ldc, f32* const restrict rwork, f32* resid);

/* Cholesky (PO) verification routines */
void spot01(const char* uplo, const INT n, const f32* const restrict A, const INT lda,
            f32* const restrict AFAC, const INT ldafac, f32* const restrict rwork,
            f32* resid);

void spot02(const char* uplo, const INT n, const INT nrhs,
            const f32* const restrict A, const INT lda,
            const f32* const restrict X, const INT ldx,
            f32* const restrict B, const INT ldb,
            f32* const restrict rwork, f32* resid);

void spot03(const char* uplo, const INT n, const f32* const restrict A, const INT lda,
            f32* const restrict AINV, const INT ldainv, f32* const restrict work,
            const INT ldwork, f32* const restrict rwork, f32* rcond, f32* resid);

void spot05(const char* uplo, const INT n, const INT nrhs,
            const f32* const restrict A, const INT lda,
            const f32* const restrict B, const INT ldb,
            const f32* const restrict X, const INT ldx,
            const f32* const restrict XACT, const INT ldxact,
            const f32* const restrict ferr, const f32* const restrict berr,
            f32* const restrict reslts);

void spot06(const char* uplo, const INT n, const INT nrhs,
            const f32* A, const INT lda, const f32* X, const INT ldx,
            f32* B, const INT ldb, f32* rwork, f32* resid);

/* Positive semidefinite pivoted Cholesky (PS) verification routines */
INT sgennd(const INT m, const INT n, const f32* const restrict A, const INT lda);

void spst01(const char* uplo, const INT n,
            const f32* const restrict A, const INT lda,
            f32* const restrict AFAC, const INT ldafac,
            f32* const restrict PERM, const INT ldperm,
            const INT* const restrict piv,
            f32* const restrict rwork, f32* resid, const INT rank);

/* Packed Cholesky (PP) verification routines */
void sppt01(const char* uplo, const INT n, const f32* const restrict A,
            f32* const restrict AFAC, f32* const restrict rwork,
            f32* resid);

void sppt02(const char* uplo, const INT n, const INT nrhs,
            const f32* const restrict A,
            const f32* const restrict X, const INT ldx,
            f32* const restrict B, const INT ldb,
            f32* const restrict rwork, f32* resid);

void sppt03(const char* uplo, const INT n, const f32* const restrict A,
            const f32* const restrict AINV, f32* const restrict work,
            const INT ldwork, f32* const restrict rwork,
            f32* rcond, f32* resid);

void sppt05(const char* uplo, const INT n, const INT nrhs,
            const f32* const restrict AP,
            const f32* const restrict B, const INT ldb,
            const f32* const restrict X, const INT ldx,
            const f32* const restrict XACT, const INT ldxact,
            const f32* const restrict FERR, const f32* const restrict BERR,
            f32* const restrict reslts);

/* QR verification routines */
void sqrt01(const INT m, const INT n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict R,
            const INT lda, f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void sqrt02(const INT m, const INT n, const INT k, const f32 * const restrict A,
            const f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict R,
            const INT lda, const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void sqrt03(const INT m, const INT n, const INT k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const INT lda, const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void sqrt04(const INT m, const INT n, const INT nb, f32 * restrict result);

void sqrt05(const INT m, const INT n, const INT l, const INT nb, f32 * restrict result);

void sqrt01p(const INT m, const INT n,
             const f32 * const restrict A,
             f32 * const restrict AF,
             f32 * const restrict Q,
             f32 * const restrict R,
             const INT lda,
             f32 * const restrict tau,
             f32 * const restrict work, const INT lwork,
             f32 * const restrict rwork,
             f32 * restrict result);

/* LQ verification routines */
void slqt01(const INT m, const INT n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict L,
            const INT lda, f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void slqt02(const INT m, const INT n, const INT k, const f32 * const restrict A,
            const f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict L,
            const INT lda, const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void slqt03(const INT m, const INT n, const INT k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const INT lda, const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void slqt04(const INT m, const INT n, const INT nb, f32 * restrict result);

void slqt05(const INT m, const INT n, const INT l, const INT nb, f32 * restrict result);

/* Householder reconstruction verification routines */
void sorhr_col01(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f32 * restrict result);

void sorhr_col02(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f32 * restrict result);

/* QL verification routines */
void sqlt01(const INT m, const INT n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict L,
            const INT lda, f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void sqlt02(const INT m, const INT n, const INT k,
            const f32 * const restrict A, const f32 * const restrict AF,
            f32 * const restrict Q, f32 * const restrict L, const INT lda,
            const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void sqlt03(const INT m, const INT n, const INT k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const INT lda, const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

/* RQ verification routines */
void srqt01(const INT m, const INT n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict R,
            const INT lda, f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void srqt02(const INT m, const INT n, const INT k,
            const f32 * const restrict A, const f32 * const restrict AF,
            f32 * const restrict Q, f32 * const restrict R, const INT lda,
            const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

void srqt03(const INT m, const INT n, const INT k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const INT lda, const f32 * const restrict tau, f32 * const restrict work,
            const INT lwork, f32 * const restrict rwork, f32 * restrict result);

/* QR with pivoting verification routines */
f32 sqpt01(const INT m, const INT n, const INT k, const f32* A, const f32* AF,
              const INT lda, const f32* tau, const INT* jpvt,
              f32* work, const INT lwork);

f32 sqrt11(const INT m, const INT k, const f32* A, const INT lda,
              const f32* tau, f32* work, const INT lwork);

f32 sqrt12(const INT m, const INT n, const f32* A, const INT lda,
              const f32* S, f32* work, const INT lwork);

/* RZ factorization verification routines */
f32 srzt01(const INT m, const INT n, const f32* A, const f32* AF,
              const INT lda, const f32* tau, f32* work, const INT lwork);

f32 srzt02(const INT m, const INT n, const f32* AF, const INT lda,
              const f32* tau, f32* work, const INT lwork);

/* Least squares verification routines */
void sqrt13(const INT scale, const INT m, const INT n,
            f32* A, const INT lda, f32* norma,
            uint64_t state[static 4]);

f32 sqrt14(const char* trans, const INT m, const INT n, const INT nrhs,
              const f32* A, const INT lda, const f32* X, const INT ldx,
              f32* work, const INT lwork);

void sqrt15(const INT scale, const INT rksel,
            const INT m, const INT n, const INT nrhs,
            f32* A, const INT lda, f32* B, const INT ldb,
            f32* S, INT* rank, f32* norma, f32* normb,
            f32* work, const INT lwork,
            uint64_t state[static 4]);

void sqrt16(const char* trans, const INT m, const INT n, const INT nrhs,
            const f32* A, const INT lda,
            const f32* X, const INT ldx,
            f32* B, const INT ldb,
            f32* rwork, f32* resid);

f32 sqrt17(const char* trans, const INT iresid,
              const INT m, const INT n, const INT nrhs,
              const f32* A, const INT lda,
              const f32* X, const INT ldx,
              const f32* B, const INT ldb,
              f32* C,
              f32* work, const INT lwork);

/* Orthogonal random matrix generator */
void slaror(const char* side, const char* init,
            const INT m, const INT n,
            f32* A, const INT lda,
            f32* X, INT* info,
            uint64_t state[static 4]);

void slarge(const INT n, f32* A, const INT lda,
            f32* work, INT* info, uint64_t state[static 4]);

void slarnv_rng(const INT idist, const INT n, f32* x,
                uint64_t state[static 4]);

f32 slaran_rng(uint64_t state[static 4]);

/* Triangular verification routines */
void strt01(const char* uplo, const char* diag, const INT n,
            const f32* A, const INT lda, f32* AINV, const INT ldainv,
            f32* rcond, f32* work, f32* resid);

void strt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f32* A, const INT lda,
            const f32* X, const INT ldx, const f32* B, const INT ldb,
            f32* work, f32* resid);

void strt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f32* A, const INT lda,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const f32* X, const INT ldx, const f32* B, const INT ldb,
            f32* work, f32* resid);

void strt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f32* A, const INT lda,
            const f32* B, const INT ldb, const f32* X, const INT ldx,
            const f32* XACT, const INT ldxact,
            const f32* ferr, const f32* berr, f32* reslts);

void strt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const INT n,
            const f32* A, const INT lda, f32* work, f32* rat);

/* Triangular matrix generation */
void slattr(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, f32* A, const INT lda,
            f32* B, f32* work, INT* info,
            uint64_t state[static 4]);

/* Triangular packed (TP) verification routines */
void stpt01(const char* uplo, const char* diag, const INT n,
            const f32* AP, f32* AINVP,
            f32* rcond, f32* work, f32* resid);

void stpt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f32* AP, const f32* X, const INT ldx,
            const f32* B, const INT ldb,
            f32* work, f32* resid);

void stpt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f32* AP, const f32 scale, const f32* cnorm,
            const f32 tscal, const f32* X, const INT ldx,
            const f32* B, const INT ldb,
            f32* work, f32* resid);

void stpt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f32* AP, const f32* B, const INT ldb,
            const f32* X, const INT ldx,
            const f32* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

void stpt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const INT n,
            const f32* AP, f32* work, f32* rat);

/* Triangular packed matrix generation */
void slattp(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, f32* AP, f32* B, f32* work,
            INT* info, uint64_t state[static 4]);

/* Eigenvalue verification routines */
void sstech(const INT n, const f32* const restrict A, const f32* const restrict B,
            const f32* const restrict eig, const f32 tol,
            f32* const restrict work, INT* info);

void sstt21(const INT n, const INT kband, const f32* const restrict AD,
            const f32* const restrict AE, const f32* const restrict SD,
            const f32* const restrict SE, const f32* const restrict U, const INT ldu,
            f32* const restrict work, f32* restrict result);

void sstt22(const INT n, const INT m, const INT kband,
            const f32* const restrict AD, const f32* const restrict AE,
            const f32* const restrict SD, const f32* const restrict SE,
            const f32* const restrict U, const INT ldu,
            f32* const restrict work, const INT ldwork,
            f32* restrict result);

void ssyt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const f32* const restrict A, const INT lda,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const INT ldu,
            f32* restrict V, const INT ldv,
            const f32* const restrict tau,
            f32* const restrict work, f32* restrict result);

void ssyt22(const INT itype, const char* uplo, const INT n, const INT m,
            const INT kband, const f32* const restrict A, const INT lda,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const INT ldu,
            const f32* const restrict V, const INT ldv,
            const f32* const restrict tau,
            f32* const restrict work, f32* restrict result);

f32 ssxt1(const INT ijob, const f32* const restrict D1, const INT n1,
             const f32* const restrict D2, const INT n2,
             const f32 abstol, const f32 ulp, const f32 unfl);

/* Symmetric band eigenvector verification */
void ssbt21(const char* uplo, const INT n, const INT ka, const INT ks,
            const f32* A, const INT lda,
            const f32* D, const f32* E,
            const f32* U, const INT ldu,
            f32* work, f32* result);

/* Symmetric generalized eigenvector verification */
void ssgt01(const INT itype, const char* uplo, const INT n, const INT m,
            const f32* A, const INT lda,
            const f32* B, const INT ldb,
            f32* Z, const INT ldz,
            const f32* D, f32* work, f32* result);

/* Symmetric packed eigenvector verification */
void sspt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const f32* AP, const f32* D, const f32* E,
            const f32* U, const INT ldu,
            f32* VP, const f32* tau,
            f32* work, f32* result);

/* Tridiagonal eigenvalue count (Sturm sequence) */
void sstect(const INT n, const f32* a, const f32* b,
            const f32 shift, INT* num);

/* Positive definite tridiagonal (PT) verification routines */
void sptt01(const INT n, const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict DF, const f32* const restrict EF,
            f32* const restrict work, f32* resid);

void sptt02(const INT n, const INT nrhs, const f32* const restrict D,
            const f32* const restrict E, const f32* const restrict X, const INT ldx,
            f32* const restrict B, const INT ldb, f32* resid);

void sptt05(const INT n, const INT nrhs, const f32* const restrict D,
            const f32* const restrict E, const f32* const restrict B, const INT ldb,
            const f32* const restrict X, const INT ldx,
            const f32* const restrict XACT, const INT ldxact,
            const f32* const restrict FERR, const f32* const restrict BERR,
            f32* const restrict reslts);

void slaptm(const INT n, const INT nrhs, const f32 alpha,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict X, const INT ldx, const f32 beta,
            f32* const restrict B, const INT ldb);

/* SVD verification routines */
void sbdt01(const INT m, const INT n, const INT kd,
            const f32* const restrict A, const INT lda,
            const f32* const restrict Q, const INT ldq,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict PT, const INT ldpt,
            f32* const restrict work, f32* resid);

void sbdt02(const INT m, const INT n,
            const f32* const restrict B, const INT ldb,
            const f32* const restrict C, const INT ldc,
            const f32* const restrict U, const INT ldu,
            f32* const restrict work, f32* resid);

void sbdt03(const char* uplo, const INT n, const INT kd,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const INT ldu,
            const f32* const restrict S,
            const f32* const restrict VT, const INT ldvt,
            f32* const restrict work, f32* resid);

void sbdt04(const char* uplo, const INT n,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict S, const INT ns,
            const f32* const restrict U, const INT ldu,
            const f32* const restrict VT, const INT ldvt,
            f32* const restrict work, f32* resid);

void sort03(const char* rc, const INT mu, const INT mv, const INT n,
            const INT k, const f32* const restrict U, const INT ldu,
            const f32* const restrict V, const INT ldv,
            f32* const restrict work, const INT lwork,
            f32* result, INT* info);

void sbdt05(const INT m, const INT n, const f32* const restrict A, const INT lda,
            const f32* const restrict S, const INT ns,
            const f32* const restrict U, const INT ldu,
            const f32* const restrict VT, const INT ldvt,
            f32* const restrict work, f32* resid);

/* Generalized eigenvalue verification routines */
void sget51(const INT itype, const INT n,
            const f32* A, const INT lda,
            const f32* B, const INT ldb,
            const f32* U, const INT ldu,
            const f32* V, const INT ldv,
            f32* work, f32* result);

void sget52(const INT left, const INT n,
            const f32* A, const INT lda,
            const f32* B, const INT ldb,
            const f32* E, const INT lde,
            const f32* alphar, const f32* alphai, const f32* beta,
            f32* work, f32* result);

void sget53(const f32* A, const INT lda,
            const f32* B, const INT ldb,
            const f32 scale, const f32 wr, const f32 wi,
            f32* result, INT* info);

/* Non-symmetric eigenvalue verification routines */
void sget10(const INT m, const INT n,
            const f32* const restrict A, const INT lda,
            const f32* const restrict B, const INT ldb,
            f32* const restrict work, f32* result);

void sort01(const char* rowcol, const INT m, const INT n,
            const f32* U, const INT ldu,
            f32* work, const INT lwork, f32* resid);

void shst01(const INT n, const INT ilo, const INT ihi,
            const f32* A, const INT lda,
            const f32* H, const INT ldh,
            const f32* Q, const INT ldq,
            f32* work, const INT lwork, f32* result);

void sget22(const char* transa, const char* transe, const char* transw,
            const INT n, const f32* A, const INT lda,
            const f32* E, const INT lde,
            const f32* wr, const f32* wi,
            f32* work, f32* result);

void slatm4(const INT itype, const INT n, const INT nz1, const INT nz2,
            const INT isign, const f32 amagn, const f32 rcond,
            const f32 triang, const INT idist,
            f32* A, const INT lda, uint64_t state[static 4]);

void slatme(const INT n, const char* dist, f32* D,
            const INT mode, const f32 cond, const f32 dmax,
            const char* ei, const char* rsign, const char* upper,
            const char* sim, f32* DS, const INT modes, const f32 conds,
            const INT kl, const INT ku, const f32 anorm,
            f32* A, const INT lda, f32* work, INT* info,
            uint64_t state[static 4]);

f32 slatm2(const INT m, const INT n, const INT i, const INT j,
              const INT kl, const INT ku, const INT idist,
              const f32* d, const INT igrade,
              const f32* dl, const f32* dr,
              const INT ipvtng, const INT* iwork, const f32 sparse,
              uint64_t state[static 4]);

f32 slatm3(const INT m, const INT n, const INT i, const INT j,
              INT* isub, INT* jsub, const INT kl, const INT ku,
              const INT idist, const f32* d, const INT igrade,
              const f32* dl, const f32* dr,
              const INT ipvtng, const INT* iwork, const f32 sparse,
              uint64_t state[static 4]);

void slatmr(const INT m, const INT n, const char* dist, const char* sym,
            f32* d, const INT mode, const f32 cond, const f32 dmax,
            const char* rsign, const char* grade, f32* dl,
            const INT model, const f32 condl, f32* dr,
            const INT moder, const f32 condr, const char* pivtng,
            const INT* ipivot, const INT kl, const INT ku,
            const f32 sparse, const f32 anorm, const char* pack,
            f32* A, const INT lda, INT* iwork, INT* info,
            uint64_t state[static 4]);

/* GLM verification routines */
void sglmts(const INT n, const INT m, const INT p,
            const f32* A, f32* AF, const INT lda,
            const f32* B, f32* BF, const INT ldb,
            const f32* D, f32* DF, f32* X, f32* U,
            f32* work, const INT lwork, f32* rwork,
            f32* result);

/* LSE verification routines */
void slsets(const INT m, const INT p, const INT n,
            const f32* A, f32* AF, const INT lda,
            const f32* B, f32* BF, const INT ldb,
            const f32* C, f32* CF, const f32* D, f32* DF,
            f32* X, f32* work, const INT lwork,
            f32* rwork, f32* result);

/* Matrix parameter setup for GLM/GQR/GRQ/GSV/LSE tests */
void slatb9(const char* path, const INT imat, const INT m, const INT p, const INT n,
            char* type, INT* kla, INT* kua, INT* klb, INT* kub,
            f32* anorm, f32* bnorm, INT* modea, INT* modeb,
            f32* cndnma, f32* cndnmb, char* dista, char* distb);

/* GQR/GRQ verification routines */
void sgqrts(const INT n, const INT m, const INT p,
            const f32* A, f32* AF, f32* Q, f32* R,
            const INT lda, f32* taua,
            const f32* B, f32* BF, f32* Z, f32* T,
            f32* BWK, const INT ldb, f32* taub,
            f32* work, const INT lwork, f32* rwork,
            f32* result);

void sgrqts(const INT m, const INT p, const INT n,
            const f32* A, f32* AF, f32* Q, f32* R,
            const INT lda, f32* taua,
            const f32* B, f32* BF, f32* Z, f32* T,
            f32* BWK, const INT ldb, f32* taub,
            f32* work, const INT lwork, f32* rwork,
            f32* result);

/* GSVD verification routines */
void sgsvts3(const INT m, const INT p, const INT n,
             const f32* A, f32* AF, const INT lda,
             const f32* B, f32* BF, const INT ldb,
             f32* U, const INT ldu,
             f32* V, const INT ldv,
             f32* Q, const INT ldq,
             f32* alpha, f32* beta,
             f32* R, const INT ldr,
             INT* iwork,
             f32* work, const INT lwork,
             f32* rwork,
             f32* result);

/* Additional matrix generators (MATGEN) */

/* Hilbert matrix generator */
void slahilb(const INT n, const INT nrhs,
             f32* A, const INT lda,
             f32* X, const INT ldx,
             f32* B, const INT ldb,
             f32* work, INT* info);

/* Kronecker product block matrix */
void slakf2(const INT m, const INT n,
            const f32* A, const INT lda,
            const f32* B, const f32* D, const f32* E,
            f32* Z, const INT ldz);

/* Singular value distribution */
void slatm7(const INT mode, const f32 cond, const INT irsign,
            const INT idist, f32* d, const INT n, const INT rank,
            INT* info, uint64_t state[static 4]);

/* Generalized Sylvester test matrices */
void slatm5(const INT prtype, const INT m, const INT n,
            f32* A, const INT lda,
            f32* B, const INT ldb,
            f32* C, const INT ldc,
            f32* D, const INT ldd,
            f32* E, const INT lde,
            f32* F, const INT ldf,
            f32* R, const INT ldr,
            f32* L, const INT ldl,
            const f32 alpha, INT qblcka, INT qblckb);

/* Generalized eigenvalue test matrices */
void slatm6(const INT type, const INT n,
            f32* A, const INT lda, f32* B,
            f32* X, const INT ldx, f32* Y, const INT ldy,
            const f32 alpha, const f32 beta,
            const f32 wx, const f32 wy,
            f32* S, f32* DIF);

/* Matrix parameter setup for tridiagonal/banded tests */
void slatb5(const char* path, const INT imat, const INT n,
            char* type, INT* kl, INT* ku, f32* anorm,
            INT* mode, f32* cndnum, char* dist);

/* QL/RQ solve helpers */
void sgeqls(const INT m, const INT n, const INT nrhs,
            f32* A, const INT lda, const f32* tau,
            f32* B, const INT ldb,
            f32* work, const INT lwork, INT* info);

void sgerqs(const INT m, const INT n, const INT nrhs,
            f32* A, const INT lda, const f32* tau,
            f32* B, const INT ldb,
            f32* work, const INT lwork, INT* info);

/* Packed symmetric (SP) verification routines */
void sspt01(const char* uplo, const INT n, const f32* A,
            const f32* AFAC, const INT* ipiv, f32* C, const INT ldc,
            f32* rwork, f32* resid);

/* Packed symmetric multiply (from SSPTRF factorization) */
void slavsp(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f32* const restrict A,
            const INT* const restrict ipiv,
            f32* const restrict B, const INT ldb, INT* info);

/* Symmetric Rook multiply (from SSYTRF_ROOK factorization) */
void slavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const INT n, const INT nrhs,
                 const f32* const restrict A, const INT lda,
                 const INT* const restrict ipiv,
                 f32* const restrict B, const INT ldb, INT* info);

/* SGEEVX verification */
void sget23(const INT comp, const char* balanc, const INT jtype,
            const f32 thresh, const INT n,
            f32* A, const INT lda, f32* H,
            f32* wr, f32* wi, f32* wr1, f32* wi1,
            f32* VL, const INT ldvl, f32* VR, const INT ldvr,
            f32* LRE, const INT ldlre,
            f32* rcondv, f32* rcndv1, const f32* rcdvin,
            f32* rconde, f32* rcnde1, const f32* rcdein,
            f32* scale, f32* scale1, f32* result,
            f32* work, const INT lwork, INT* iwork, INT* info);

/* SGEESX verification */
void sget24(const INT comp, const INT jtype, const f32 thresh,
            const INT n, f32* A, const INT lda,
            f32* H, f32* HT,
            f32* wr, f32* wi, f32* wrt, f32* wit,
            f32* wrtmp, f32* witmp,
            f32* VS, const INT ldvs, f32* VS1,
            const f32 rcdein, const f32 rcdvin,
            const INT nslct, const INT* islct,
            f32* result, f32* work, const INT lwork,
            INT* iwork, INT* bwork, INT* info);

/* Generalized Schur decomposition verify */
void sget54(const INT n, const f32* A, const INT lda,
            const f32* B, const INT ldb,
            const f32* S, const INT lds,
            const f32* T, const INT ldt,
            const f32* U, const INT ldu,
            const f32* V, const INT ldv,
            f32* work, f32* result);

/* SLALN2 test (small linear system solver) */
void sget31(f32* rmax, INT* lmax, INT ninfo[2], INT* knt);

/* SLASY2 test (Sylvester-like equation solver) */
void sget32(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

/* SLANV2 test (2x2 standardization) */
void sget33(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

/* SLAEXC test (block swap in Schur form) */
void sget34(f32* rmax, INT* lmax, INT ninfo[2], INT* knt);

/* STRSYL test (Sylvester equation solver) */
void sget35(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

/* STREXC test (Schur block reordering) */
void sget36(f32* rmax, INT* lmax, INT ninfo[3], INT* knt);

/* STRSNA test (eigenvalue/eigenvector condition numbers) */
void sget37(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);

/* STRSEN test (cluster condition numbers) */
void sget38(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);

/* SLAQTR test (quasi-triangular solve) */
void sget39(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

/* STGEXC test (generalized Schur block swap) */
void sget40(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

/* Sylvester equation test (STRSYL + STRSYL3) */
void ssyl01(const f32 thresh, INT* nfail, f32* rmax, INT* ninfo, INT* knt);

/* Error exit testing infrastructure */
extern INT    xerbla_infot;
extern INT    xerbla_nout;
extern INT    xerbla_ok;
extern INT    xerbla_lerr;
extern char   xerbla_srnamt[33];
void chkxer(const char* srnamt, INT infot, INT* lerr, INT* ok);
void serrec(INT* ok, INT* nt);

/* SVD singular value verification */
void ssvdct(const INT n, const f32* s, const f32* e, const f32 shift, INT* num);
void ssvdch(const INT n, const f32* s, const f32* e,
            const f32* svd, const f32 tol, INT* info);

/* Triangular banded (TB) verification routines */
void stbt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f32* AB, const INT ldab,
            const f32* X, const INT ldx,
            const f32* B, const INT ldb,
            f32* work, f32* resid);

void stbt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f32* AB, const INT ldab,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const f32* X, const INT ldx,
            const f32* B, const INT ldb,
            f32* work, f32* resid);

void stbt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f32* AB, const INT ldab,
            const f32* B, const INT ldb,
            const f32* X, const INT ldx,
            const f32* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

void stbt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const INT n, const INT kd,
            const f32* AB, const INT ldab, f32* work, f32* rat);

/* Triangular banded matrix generation */
void slattb(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, const INT kd,
            f32* AB, const INT ldab, f32* B,
            f32* work, INT* info, uint64_t state[static 4]);

/* Symmetric verification routines (BK variants) */
void ssyt01_3(const char* uplo, const INT n,
              const f32* const restrict A, const INT lda,
              f32* const restrict AFAC, const INT ldafac,
              f32* const restrict E,
              INT* const restrict ipiv,
              f32* const restrict C, const INT ldc,
              f32* const restrict rwork, f32* resid);

void ssyt01_aa(const char* uplo, const INT n,
               const f32* const restrict A, const INT lda,
               const f32* const restrict AFAC, const INT ldafac,
               const INT* const restrict ipiv,
               f32* const restrict C, const INT ldc,
               f32* const restrict rwork, f32* resid);

void ssyt01_rook(const char* uplo, const INT n,
                 const f32* const restrict A, const INT lda,
                 const f32* const restrict AFAC, const INT ldafac,
                 const INT* const restrict ipiv,
                 f32* const restrict C, const INT ldc,
                 f32* const restrict rwork, f32* resid);

/* TSQR verification */
void stsqr01(const char* tssw, const INT m, const INT n, const INT mb,
             const INT nb, f32* result);

/* CS decomposition verification */
void scsdts(const INT m, const INT p, const INT q,
            const f32* X, f32* XF, const INT ldx,
            f32* U1, const INT ldu1,
            f32* U2, const INT ldu2,
            f32* V1T, const INT ldv1t,
            f32* V2T, const INT ldv2t,
            f32* theta, INT* iwork,
            f32* work, const INT lwork,
            f32* rwork, f32* result);

#endif /* VERIFY_H */
