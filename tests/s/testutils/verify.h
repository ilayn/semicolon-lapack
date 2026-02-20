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
#include "semicolon_lapack/types.h"

/* General (SGE) verification routines */
void sget01(const int m, const int n, const f32 * const restrict A, const int lda,
            f32 * const restrict AFAC, const int ldafac, const int * const restrict ipiv,
            f32 * const restrict rwork, f32 *resid);

void sget02(const char* trans, const int m, const int n, const int nrhs,
            const f32 * const restrict A, const int lda, const f32 * const restrict X,
            const int ldx, f32 * const restrict B, const int ldb,
            f32 * const restrict rwork, f32 *resid);

void sget03(const int n, const f32 * const restrict A, const int lda,
            const f32 * const restrict AINV, const int ldainv, f32 * const restrict work,
            const int ldwork, f32 * const restrict rwork, f32 *rcond, f32 *resid);

void sget04(const int n, const int nrhs, const f32 * const restrict X, const int ldx,
            const f32 * const restrict XACT, const int ldxact, const f32 rcond,
            f32 *resid);

f32 sget06(const f32 rcond, const f32 rcondc);

void sget07(const char* trans, const int n, const int nrhs,
            const f32 * const restrict A, const int lda,
            const f32 * const restrict B, const int ldb,
            const f32 * const restrict X, const int ldx,
            const f32 * const restrict XACT, const int ldxact,
            const f32 * const restrict ferr, const int chkferr,
            const f32 * const restrict berr, f32 * const restrict reslts);

void sget08(const char* trans, const int m, const int n, const int nrhs,
            const f32* A, const int lda, const f32* X, const int ldx,
            f32* B, const int ldb, f32* rwork, f32* resid);

/* Banded (GB) verification routines */
void sgbt01(int m, int n, int kl, int ku,
            const f32* A, int lda,
            const f32* AFAC, int ldafac,
            const int* ipiv,
            f32* work,
            f32* resid);

void sgbt02(const char* trans, int m, int n, int kl, int ku, int nrhs,
            const f32* A, int lda,
            const f32* X, int ldx,
            f32* B, int ldb,
            f32* rwork,
            f32* resid);

void sgbt05(const char* trans, int n, int kl, int ku, int nrhs,
            const f32* AB, int ldab,
            const f32* B, int ldb,
            const f32* X, int ldx,
            const f32* XACT, int ldxact,
            const f32* FERR,
            const f32* BERR,
            f32* reslts);

/* Positive definite banded (PB) verification routines */
void spbt01(const char* uplo, const int n, const int kd,
            const f32* A, const int lda,
            f32* AFAC, const int ldafac,
            f32* rwork, f32* resid);

void spbt02(const char* uplo, const int n, const int kd, const int nrhs,
            const f32* A, const int lda,
            const f32* X, const int ldx,
            f32* B, const int ldb,
            f32* rwork, f32* resid);

void spbt05(const char* uplo, const int n, const int kd, const int nrhs,
            const f32* AB, const int ldab,
            const f32* B, const int ldb,
            const f32* X, const int ldx,
            const f32* XACT, const int ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

/* Tridiagonal (SGT) verification routines */
void sgtt01(const int n, const f32 * const restrict DL, const f32 * const restrict D,
            const f32 * const restrict DU, const f32 * const restrict DLF,
            const f32 * const restrict DF, const f32 * const restrict DUF,
            const f32 * const restrict DU2, const int * const restrict ipiv,
            f32 * const restrict work, const int ldwork, f32 *resid);

void sgtt02(const char* trans, const int n, const int nrhs,
            const f32 * const restrict DL, const f32 * const restrict D,
            const f32 * const restrict DU, const f32 * const restrict X, const int ldx,
            f32 * const restrict B, const int ldb, f32 *resid);

void sgtt05(const char* trans, const int n, const int nrhs,
            const f32 * const restrict DL, const f32 * const restrict D,
            const f32 * const restrict DU, const f32 * const restrict B, const int ldb,
            const f32 * const restrict X, const int ldx,
            const f32 * const restrict XACT, const int ldxact,
            const f32 * const restrict ferr, const f32 * const restrict berr,
            f32 * const restrict reslts);

/* Matrix generation routines */
void slatb4(const char *path, const int imat, const int m, const int n,
            char *type, int *kl, int *ku, f32 *anorm, int *mode,
            f32 *cndnum, char *dist);

void slatms(const int m, const int n, const char* dist,
            const char* sym, f32 *d, const int mode, const f32 cond,
            const f32 dmax, const int kl, const int ku, const char* pack,
            f32 *A, const int lda, f32 *work, int *info,
            uint64_t state[static 4]);

void slatmt(const int m, const int n, const char* dist,
            const char* sym, f32* d, const int mode,
            const f32 cond, const f32 dmax, const int rank,
            const int kl, const int ku, const char* pack,
            f32* A, const int lda, f32* work, int* info,
            uint64_t state[static 4]);

void slatm1(const int mode, const f32 cond, const int irsign,
            const int idist, f32* d, const int n, int* info,
            uint64_t state[static 4]);

void slagge(const int m, const int n, const int kl, const int ku,
            const f32* d, f32* A, const int lda,
            f32* work, int* info, uint64_t state[static 4]);

void slagsy(const int n, const int k, const f32* d, f32* A,
            const int lda, f32* work, int* info,
            uint64_t state[static 4]);

void slarot(const int lrows, const int lleft, const int lright,
            const int nl, const f32 c, const f32 s,
            f32* A, const int lda, f32* xleft, f32* xright);

void slarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const int m, const int n, const int kl,
            const int ku, const int nrhs, const f32* A, const int lda,
            f32* X, const int ldx, f32* B, const int ldb,
            int* info, uint64_t state[static 4]);

void slaord(const char* job, const int n, f32* X, const int incx);

/* Symmetric (SY) verification routines */
void slavsy(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f32* const restrict A, const int lda,
            const int* const restrict ipiv, f32* const restrict B, const int ldb, int* info);

void ssyt01(const char* uplo, const int n, const f32* const restrict A, const int lda,
            const f32* const restrict AFAC, const int ldafac, const int* const restrict ipiv,
            f32* const restrict C, const int ldc, f32* const restrict rwork, f32* resid);

/* Cholesky (PO) verification routines */
void spot01(const char* uplo, const int n, const f32* const restrict A, const int lda,
            f32* const restrict AFAC, const int ldafac, f32* const restrict rwork,
            f32* resid);

void spot02(const char* uplo, const int n, const int nrhs,
            const f32* const restrict A, const int lda,
            const f32* const restrict X, const int ldx,
            f32* const restrict B, const int ldb,
            f32* const restrict rwork, f32* resid);

void spot03(const char* uplo, const int n, const f32* const restrict A, const int lda,
            f32* const restrict AINV, const int ldainv, f32* const restrict work,
            const int ldwork, f32* const restrict rwork, f32* rcond, f32* resid);

void spot05(const char* uplo, const int n, const int nrhs,
            const f32* const restrict A, const int lda,
            const f32* const restrict B, const int ldb,
            const f32* const restrict X, const int ldx,
            const f32* const restrict XACT, const int ldxact,
            const f32* const restrict ferr, const f32* const restrict berr,
            f32* const restrict reslts);

void spot06(const char* uplo, const int n, const int nrhs,
            const f32* A, const int lda, const f32* X, const int ldx,
            f32* B, const int ldb, f32* rwork, f32* resid);

/* Positive semidefinite pivoted Cholesky (PS) verification routines */
int sgennd(const int m, const int n, const f32* const restrict A, const int lda);

void spst01(const char* uplo, const int n,
            const f32* const restrict A, const int lda,
            f32* const restrict AFAC, const int ldafac,
            f32* const restrict PERM, const int ldperm,
            const int* const restrict piv,
            f32* const restrict rwork, f32* resid, const int rank);

/* Packed Cholesky (PP) verification routines */
void sppt01(const char* uplo, const int n, const f32* const restrict A,
            f32* const restrict AFAC, f32* const restrict rwork,
            f32* resid);

void sppt02(const char* uplo, const int n, const int nrhs,
            const f32* const restrict A,
            const f32* const restrict X, const int ldx,
            f32* const restrict B, const int ldb,
            f32* const restrict rwork, f32* resid);

void sppt03(const char* uplo, const int n, const f32* const restrict A,
            const f32* const restrict AINV, f32* const restrict work,
            const int ldwork, f32* const restrict rwork,
            f32* rcond, f32* resid);

void sppt05(const char* uplo, const int n, const int nrhs,
            const f32* const restrict AP,
            const f32* const restrict B, const int ldb,
            const f32* const restrict X, const int ldx,
            const f32* const restrict XACT, const int ldxact,
            const f32* const restrict FERR, const f32* const restrict BERR,
            f32* const restrict reslts);

/* QR verification routines */
void sqrt01(const int m, const int n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict R,
            const int lda, f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void sqrt02(const int m, const int n, const int k, const f32 * const restrict A,
            const f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict R,
            const int lda, const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void sqrt03(const int m, const int n, const int k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const int lda, const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void sqrt04(const int m, const int n, const int nb, f32 * restrict result);

void sqrt05(const int m, const int n, const int l, const int nb, f32 * restrict result);

void sqrt01p(const int m, const int n,
             const f32 * const restrict A,
             f32 * const restrict AF,
             f32 * const restrict Q,
             f32 * const restrict R,
             const int lda,
             f32 * const restrict tau,
             f32 * const restrict work, const int lwork,
             f32 * const restrict rwork,
             f32 * restrict result);

/* LQ verification routines */
void slqt01(const int m, const int n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict L,
            const int lda, f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void slqt02(const int m, const int n, const int k, const f32 * const restrict A,
            const f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict L,
            const int lda, const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void slqt03(const int m, const int n, const int k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const int lda, const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void slqt04(const int m, const int n, const int nb, f32 * restrict result);

void slqt05(const int m, const int n, const int l, const int nb, f32 * restrict result);

/* Householder reconstruction verification routines */
void sorhr_col01(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, f32 * restrict result);

void sorhr_col02(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, f32 * restrict result);

/* QL verification routines */
void sqlt01(const int m, const int n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict L,
            const int lda, f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void sqlt02(const int m, const int n, const int k,
            const f32 * const restrict A, const f32 * const restrict AF,
            f32 * const restrict Q, f32 * const restrict L, const int lda,
            const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void sqlt03(const int m, const int n, const int k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const int lda, const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

/* RQ verification routines */
void srqt01(const int m, const int n, const f32 * const restrict A,
            f32 * const restrict AF, f32 * const restrict Q, f32 * const restrict R,
            const int lda, f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void srqt02(const int m, const int n, const int k,
            const f32 * const restrict A, const f32 * const restrict AF,
            f32 * const restrict Q, f32 * const restrict R, const int lda,
            const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

void srqt03(const int m, const int n, const int k, const f32 * const restrict AF,
            f32 * const restrict C, f32 * const restrict CC, f32 * const restrict Q,
            const int lda, const f32 * const restrict tau, f32 * const restrict work,
            const int lwork, f32 * const restrict rwork, f32 * restrict result);

/* QR with pivoting verification routines */
f32 sqpt01(const int m, const int n, const int k, const f32* A, const f32* AF,
              const int lda, const f32* tau, const int* jpvt,
              f32* work, const int lwork);

f32 sqrt11(const int m, const int k, const f32* A, const int lda,
              const f32* tau, f32* work, const int lwork);

f32 sqrt12(const int m, const int n, const f32* A, const int lda,
              const f32* S, f32* work, const int lwork);

/* RZ factorization verification routines */
f32 srzt01(const int m, const int n, const f32* A, const f32* AF,
              const int lda, const f32* tau, f32* work, const int lwork);

f32 srzt02(const int m, const int n, const f32* AF, const int lda,
              const f32* tau, f32* work, const int lwork);

/* Least squares verification routines */
void sqrt13(const int scale, const int m, const int n,
            f32* A, const int lda, f32* norma,
            uint64_t state[static 4]);

f32 sqrt14(const char* trans, const int m, const int n, const int nrhs,
              const f32* A, const int lda, const f32* X, const int ldx,
              f32* work, const int lwork);

void sqrt15(const int scale, const int rksel,
            const int m, const int n, const int nrhs,
            f32* A, const int lda, f32* B, const int ldb,
            f32* S, int* rank, f32* norma, f32* normb,
            f32* work, const int lwork,
            uint64_t state[static 4]);

void sqrt16(const char* trans, const int m, const int n, const int nrhs,
            const f32* A, const int lda,
            const f32* X, const int ldx,
            f32* B, const int ldb,
            f32* rwork, f32* resid);

f32 sqrt17(const char* trans, const int iresid,
              const int m, const int n, const int nrhs,
              const f32* A, const int lda,
              const f32* X, const int ldx,
              const f32* B, const int ldb,
              f32* C,
              f32* work, const int lwork);

/* Orthogonal random matrix generator */
void slaror(const char* side, const char* init,
            const int m, const int n,
            f32* A, const int lda,
            f32* X, int* info,
            uint64_t state[static 4]);

void slarge(const int n, f32* A, const int lda,
            f32* work, int* info, uint64_t state[static 4]);

void slarnv_rng(const int idist, const int n, f32* x,
                uint64_t state[static 4]);

f32 slaran_rng(uint64_t state[static 4]);

/* Triangular verification routines */
void strt01(const char* uplo, const char* diag, const int n,
            const f32* A, const int lda, f32* AINV, const int ldainv,
            f32* rcond, f32* work, f32* resid);

void strt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f32* A, const int lda,
            const f32* X, const int ldx, const f32* B, const int ldb,
            f32* work, f32* resid);

void strt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f32* A, const int lda,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const f32* X, const int ldx, const f32* B, const int ldb,
            f32* work, f32* resid);

void strt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f32* A, const int lda,
            const f32* B, const int ldb, const f32* X, const int ldx,
            const f32* XACT, const int ldxact,
            const f32* ferr, const f32* berr, f32* reslts);

void strt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const int n,
            const f32* A, const int lda, f32* work, f32* rat);

/* Triangular matrix generation */
void slattr(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, f32* A, const int lda,
            f32* B, f32* work, int* info,
            uint64_t state[static 4]);

/* Triangular packed (TP) verification routines */
void stpt01(const char* uplo, const char* diag, const int n,
            const f32* AP, f32* AINVP,
            f32* rcond, f32* work, f32* resid);

void stpt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f32* AP, const f32* X, const int ldx,
            const f32* B, const int ldb,
            f32* work, f32* resid);

void stpt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f32* AP, const f32 scale, const f32* cnorm,
            const f32 tscal, const f32* X, const int ldx,
            const f32* B, const int ldb,
            f32* work, f32* resid);

void stpt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f32* AP, const f32* B, const int ldb,
            const f32* X, const int ldx,
            const f32* XACT, const int ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

void stpt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const int n,
            const f32* AP, f32* work, f32* rat);

/* Triangular packed matrix generation */
void slattp(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, f32* AP, f32* B, f32* work,
            int* info, uint64_t state[static 4]);

/* Eigenvalue verification routines */
void sstech(const int n, const f32* const restrict A, const f32* const restrict B,
            const f32* const restrict eig, const f32 tol,
            f32* const restrict work, int* info);

void sstt21(const int n, const int kband, const f32* const restrict AD,
            const f32* const restrict AE, const f32* const restrict SD,
            const f32* const restrict SE, const f32* const restrict U, const int ldu,
            f32* const restrict work, f32* restrict result);

void sstt22(const int n, const int m, const int kband,
            const f32* const restrict AD, const f32* const restrict AE,
            const f32* const restrict SD, const f32* const restrict SE,
            const f32* const restrict U, const int ldu,
            f32* const restrict work, const int ldwork,
            f32* restrict result);

void ssyt21(const int itype, const char* uplo, const int n, const int kband,
            const f32* const restrict A, const int lda,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const int ldu,
            f32* restrict V, const int ldv,
            const f32* const restrict tau,
            f32* const restrict work, f32* restrict result);

void ssyt22(const int itype, const char* uplo, const int n, const int m,
            const int kband, const f32* const restrict A, const int lda,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const int ldu,
            const f32* const restrict V, const int ldv,
            const f32* const restrict tau,
            f32* const restrict work, f32* restrict result);

f32 ssxt1(const int ijob, const f32* const restrict D1, const int n1,
             const f32* const restrict D2, const int n2,
             const f32 abstol, const f32 ulp, const f32 unfl);

/* Symmetric band eigenvector verification */
void ssbt21(const char* uplo, const int n, const int ka, const int ks,
            const f32* A, const int lda,
            const f32* D, const f32* E,
            const f32* U, const int ldu,
            f32* work, f32* result);

/* Symmetric generalized eigenvector verification */
void ssgt01(const int itype, const char* uplo, const int n, const int m,
            const f32* A, const int lda,
            const f32* B, const int ldb,
            f32* Z, const int ldz,
            const f32* D, f32* work, f32* result);

/* Symmetric packed eigenvector verification */
void sspt21(const int itype, const char* uplo, const int n, const int kband,
            const f32* AP, const f32* D, const f32* E,
            const f32* U, const int ldu,
            f32* VP, const f32* tau,
            f32* work, f32* result);

/* Tridiagonal eigenvalue count (Sturm sequence) */
void sstect(const int n, const f32* a, const f32* b,
            const f32 shift, int* num);

/* Positive definite tridiagonal (PT) verification routines */
void sptt01(const int n, const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict DF, const f32* const restrict EF,
            f32* const restrict work, f32* resid);

void sptt02(const int n, const int nrhs, const f32* const restrict D,
            const f32* const restrict E, const f32* const restrict X, const int ldx,
            f32* const restrict B, const int ldb, f32* resid);

void sptt05(const int n, const int nrhs, const f32* const restrict D,
            const f32* const restrict E, const f32* const restrict B, const int ldb,
            const f32* const restrict X, const int ldx,
            const f32* const restrict XACT, const int ldxact,
            const f32* const restrict FERR, const f32* const restrict BERR,
            f32* const restrict reslts);

void slaptm(const int n, const int nrhs, const f32 alpha,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict X, const int ldx, const f32 beta,
            f32* const restrict B, const int ldb);

/* SVD verification routines */
void sbdt01(const int m, const int n, const int kd,
            const f32* const restrict A, const int lda,
            const f32* const restrict Q, const int ldq,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict PT, const int ldpt,
            f32* const restrict work, f32* resid);

void sbdt02(const int m, const int n,
            const f32* const restrict B, const int ldb,
            const f32* const restrict C, const int ldc,
            const f32* const restrict U, const int ldu,
            f32* const restrict work, f32* resid);

void sbdt03(const char* uplo, const int n, const int kd,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const int ldu,
            const f32* const restrict S,
            const f32* const restrict VT, const int ldvt,
            f32* const restrict work, f32* resid);

void sbdt04(const char* uplo, const int n,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict S, const int ns,
            const f32* const restrict U, const int ldu,
            const f32* const restrict VT, const int ldvt,
            f32* const restrict work, f32* resid);

void sort03(const char* rc, const int mu, const int mv, const int n,
            const int k, const f32* const restrict U, const int ldu,
            const f32* const restrict V, const int ldv,
            f32* const restrict work, const int lwork,
            f32* result, int* info);

void sbdt05(const int m, const int n, const f32* const restrict A, const int lda,
            const f32* const restrict S, const int ns,
            const f32* const restrict U, const int ldu,
            const f32* const restrict VT, const int ldvt,
            f32* const restrict work, f32* resid);

/* Generalized eigenvalue verification routines */
void sget51(const int itype, const int n,
            const f32* A, const int lda,
            const f32* B, const int ldb,
            const f32* U, const int ldu,
            const f32* V, const int ldv,
            f32* work, f32* result);

void sget52(const int left, const int n,
            const f32* A, const int lda,
            const f32* B, const int ldb,
            const f32* E, const int lde,
            const f32* alphar, const f32* alphai, const f32* beta,
            f32* work, f32* result);

void sget53(const f32* A, const int lda,
            const f32* B, const int ldb,
            const f32 scale, const f32 wr, const f32 wi,
            f32* result, int* info);

/* Non-symmetric eigenvalue verification routines */
void sget10(const int m, const int n,
            const f32* const restrict A, const int lda,
            const f32* const restrict B, const int ldb,
            f32* const restrict work, f32* result);

void sort01(const char* rowcol, const int m, const int n,
            const f32* U, const int ldu,
            f32* work, const int lwork, f32* resid);

void shst01(const int n, const int ilo, const int ihi,
            const f32* A, const int lda,
            const f32* H, const int ldh,
            const f32* Q, const int ldq,
            f32* work, const int lwork, f32* result);

void sget22(const char* transa, const char* transe, const char* transw,
            const int n, const f32* A, const int lda,
            const f32* E, const int lde,
            const f32* wr, const f32* wi,
            f32* work, f32* result);

void slatm4(const int itype, const int n, const int nz1, const int nz2,
            const int isign, const f32 amagn, const f32 rcond,
            const f32 triang, const int idist,
            f32* A, const int lda, uint64_t state[static 4]);

void slatme(const int n, const char* dist, f32* D,
            const int mode, const f32 cond, const f32 dmax,
            const char* ei, const char* rsign, const char* upper,
            const char* sim, f32* DS, const int modes, const f32 conds,
            const int kl, const int ku, const f32 anorm,
            f32* A, const int lda, f32* work, int* info,
            uint64_t state[static 4]);

f32 slatm2(const int m, const int n, const int i, const int j,
              const int kl, const int ku, const int idist,
              const f32* d, const int igrade,
              const f32* dl, const f32* dr,
              const int ipvtng, const int* iwork, const f32 sparse,
              uint64_t state[static 4]);

f32 slatm3(const int m, const int n, const int i, const int j,
              int* isub, int* jsub, const int kl, const int ku,
              const int idist, const f32* d, const int igrade,
              const f32* dl, const f32* dr,
              const int ipvtng, const int* iwork, const f32 sparse,
              uint64_t state[static 4]);

void slatmr(const int m, const int n, const char* dist, const char* sym,
            f32* d, const int mode, const f32 cond, const f32 dmax,
            const char* rsign, const char* grade, f32* dl,
            const int model, const f32 condl, f32* dr,
            const int moder, const f32 condr, const char* pivtng,
            const int* ipivot, const int kl, const int ku,
            const f32 sparse, const f32 anorm, const char* pack,
            f32* A, const int lda, int* iwork, int* info,
            uint64_t state[static 4]);

/* GLM verification routines */
void sglmts(const int n, const int m, const int p,
            const f32* A, f32* AF, const int lda,
            const f32* B, f32* BF, const int ldb,
            const f32* D, f32* DF, f32* X, f32* U,
            f32* work, const int lwork, f32* rwork,
            f32* result);

/* LSE verification routines */
void slsets(const int m, const int p, const int n,
            const f32* A, f32* AF, const int lda,
            const f32* B, f32* BF, const int ldb,
            const f32* C, f32* CF, const f32* D, f32* DF,
            f32* X, f32* work, const int lwork,
            f32* rwork, f32* result);

/* Matrix parameter setup for GLM/GQR/GRQ/GSV/LSE tests */
void slatb9(const char* path, const int imat, const int m, const int p, const int n,
            char* type, int* kla, int* kua, int* klb, int* kub,
            f32* anorm, f32* bnorm, int* modea, int* modeb,
            f32* cndnma, f32* cndnmb, char* dista, char* distb);

/* GQR/GRQ verification routines */
void sgqrts(const int n, const int m, const int p,
            const f32* A, f32* AF, f32* Q, f32* R,
            const int lda, f32* taua,
            const f32* B, f32* BF, f32* Z, f32* T,
            f32* BWK, const int ldb, f32* taub,
            f32* work, const int lwork, f32* rwork,
            f32* result);

void sgrqts(const int m, const int p, const int n,
            const f32* A, f32* AF, f32* Q, f32* R,
            const int lda, f32* taua,
            const f32* B, f32* BF, f32* Z, f32* T,
            f32* BWK, const int ldb, f32* taub,
            f32* work, const int lwork, f32* rwork,
            f32* result);

/* GSVD verification routines */
void sgsvts3(const int m, const int p, const int n,
             const f32* A, f32* AF, const int lda,
             const f32* B, f32* BF, const int ldb,
             f32* U, const int ldu,
             f32* V, const int ldv,
             f32* Q, const int ldq,
             f32* alpha, f32* beta,
             f32* R, const int ldr,
             int* iwork,
             f32* work, const int lwork,
             f32* rwork,
             f32* result);

/* Additional matrix generators (MATGEN) */

/* Hilbert matrix generator */
void slahilb(const int n, const int nrhs,
             f32* A, const int lda,
             f32* X, const int ldx,
             f32* B, const int ldb,
             f32* work, int* info);

/* Kronecker product block matrix */
void slakf2(const int m, const int n,
            const f32* A, const int lda,
            const f32* B, const f32* D, const f32* E,
            f32* Z, const int ldz);

/* Singular value distribution */
void slatm7(const int mode, const f32 cond, const int irsign,
            const int idist, f32* d, const int n, const int rank,
            int* info, uint64_t state[static 4]);

/* Generalized Sylvester test matrices */
void slatm5(const int prtype, const int m, const int n,
            f32* A, const int lda,
            f32* B, const int ldb,
            f32* C, const int ldc,
            f32* D, const int ldd,
            f32* E, const int lde,
            f32* F, const int ldf,
            f32* R, const int ldr,
            f32* L, const int ldl,
            const f32 alpha, int qblcka, int qblckb);

/* Generalized eigenvalue test matrices */
void slatm6(const int type, const int n,
            f32* A, const int lda, f32* B,
            f32* X, const int ldx, f32* Y, const int ldy,
            const f32 alpha, const f32 beta,
            const f32 wx, const f32 wy,
            f32* S, f32* DIF);

/* Matrix parameter setup for tridiagonal/banded tests */
void slatb5(const char* path, const int imat, const int n,
            char* type, int* kl, int* ku, f32* anorm,
            int* mode, f32* cndnum, char* dist);

/* QL/RQ solve helpers */
void sgeqls(const int m, const int n, const int nrhs,
            f32* A, const int lda, const f32* tau,
            f32* B, const int ldb,
            f32* work, const int lwork, int* info);

void sgerqs(const int m, const int n, const int nrhs,
            f32* A, const int lda, const f32* tau,
            f32* B, const int ldb,
            f32* work, const int lwork, int* info);

/* Packed symmetric (SP) verification routines */
void sspt01(const char* uplo, const int n, const f32* A,
            const f32* AFAC, const int* ipiv, f32* C, const int ldc,
            f32* rwork, f32* resid);

/* Packed symmetric multiply (from SSPTRF factorization) */
void slavsp(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f32* const restrict A,
            const int* const restrict ipiv,
            f32* const restrict B, const int ldb, int* info);

/* Symmetric Rook multiply (from SSYTRF_ROOK factorization) */
void slavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const int n, const int nrhs,
                 const f32* const restrict A, const int lda,
                 const int* const restrict ipiv,
                 f32* const restrict B, const int ldb, int* info);

/* SGEEVX verification */
void sget23(const int comp, const char* balanc, const int jtype,
            const f32 thresh, const int n,
            f32* A, const int lda, f32* H,
            f32* wr, f32* wi, f32* wr1, f32* wi1,
            f32* VL, const int ldvl, f32* VR, const int ldvr,
            f32* LRE, const int ldlre,
            f32* rcondv, f32* rcndv1, const f32* rcdvin,
            f32* rconde, f32* rcnde1, const f32* rcdein,
            f32* scale, f32* scale1, f32* result,
            f32* work, const int lwork, int* iwork, int* info);

/* SGEESX verification */
void sget24(const int comp, const int jtype, const f32 thresh,
            const int n, f32* A, const int lda,
            f32* H, f32* HT,
            f32* wr, f32* wi, f32* wrt, f32* wit,
            f32* wrtmp, f32* witmp,
            f32* VS, const int ldvs, f32* VS1,
            const f32 rcdein, const f32 rcdvin,
            const int nslct, const int* islct,
            f32* result, f32* work, const int lwork,
            int* iwork, int* bwork, int* info);

/* Generalized Schur decomposition verify */
void sget54(const int n, const f32* A, const int lda,
            const f32* B, const int ldb,
            const f32* S, const int lds,
            const f32* T, const int ldt,
            const f32* U, const int ldu,
            const f32* V, const int ldv,
            f32* work, f32* result);

/* SLALN2 test (small linear system solver) */
void sget31(f32* rmax, int* lmax, int ninfo[2], int* knt);

/* SLASY2 test (Sylvester-like equation solver) */
void sget32(f32* rmax, int* lmax, int* ninfo, int* knt);

/* SLANV2 test (2x2 standardization) */
void sget33(f32* rmax, int* lmax, int* ninfo, int* knt);

/* SLAEXC test (block swap in Schur form) */
void sget34(f32* rmax, int* lmax, int ninfo[2], int* knt);

/* STRSYL test (Sylvester equation solver) */
void sget35(f32* rmax, int* lmax, int* ninfo, int* knt);

/* STREXC test (Schur block reordering) */
void sget36(f32* rmax, int* lmax, int ninfo[3], int* knt);

/* STRSNA test (eigenvalue/eigenvector condition numbers) */
void sget37(f32 rmax[3], int lmax[3], int ninfo[3], int* knt);

/* STRSEN test (cluster condition numbers) */
void sget38(f32 rmax[3], int lmax[3], int ninfo[3], int* knt);

/* SLAQTR test (quasi-triangular solve) */
void sget39(f32* rmax, int* lmax, int* ninfo, int* knt);

/* STGEXC test (generalized Schur block swap) */
void sget40(f32* rmax, int* lmax, int* ninfo, int* knt);

/* Sylvester equation test (STRSYL + STRSYL3) */
void ssyl01(const f32 thresh, int* nfail, f32* rmax, int* ninfo, int* knt);

/* Error exit testing infrastructure */
extern int    xerbla_infot;
extern int    xerbla_nout;
extern int    xerbla_ok;
extern int    xerbla_lerr;
extern char   xerbla_srnamt[33];
void chkxer(const char* srnamt, int infot, int* lerr, int* ok);
void serrec(int* ok, int* nt);

/* SVD singular value verification */
void ssvdct(const int n, const f32* s, const f32* e, const f32 shift, int* num);
void ssvdch(const int n, const f32* s, const f32* e,
            const f32* svd, const f32 tol, int* info);

/* Triangular banded (TB) verification routines */
void stbt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f32* AB, const int ldab,
            const f32* X, const int ldx,
            const f32* B, const int ldb,
            f32* work, f32* resid);

void stbt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f32* AB, const int ldab,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const f32* X, const int ldx,
            const f32* B, const int ldb,
            f32* work, f32* resid);

void stbt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f32* AB, const int ldab,
            const f32* B, const int ldb,
            const f32* X, const int ldx,
            const f32* XACT, const int ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

void stbt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const int n, const int kd,
            const f32* AB, const int ldab, f32* work, f32* rat);

/* Triangular banded matrix generation */
void slattb(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, const int kd,
            f32* AB, const int ldab, f32* B,
            f32* work, int* info, uint64_t state[static 4]);

/* Symmetric verification routines (BK variants) */
void ssyt01_3(const char* uplo, const int n,
              const f32* const restrict A, const int lda,
              f32* const restrict AFAC, const int ldafac,
              f32* const restrict E,
              int* const restrict ipiv,
              f32* const restrict C, const int ldc,
              f32* const restrict rwork, f32* resid);

void ssyt01_aa(const char* uplo, const int n,
               const f32* const restrict A, const int lda,
               const f32* const restrict AFAC, const int ldafac,
               const int* const restrict ipiv,
               f32* const restrict C, const int ldc,
               f32* const restrict rwork, f32* resid);

void ssyt01_rook(const char* uplo, const int n,
                 const f32* const restrict A, const int lda,
                 const f32* const restrict AFAC, const int ldafac,
                 const int* const restrict ipiv,
                 f32* const restrict C, const int ldc,
                 f32* const restrict rwork, f32* resid);

/* TSQR verification */
void stsqr01(const char* tssw, const int m, const int n, const int mb,
             const int nb, f32* result);

/* CS decomposition verification */
void scsdts(const int m, const int p, const int q,
            const f32* X, f32* XF, const int ldx,
            f32* U1, const int ldu1,
            f32* U2, const int ldu2,
            f32* V1T, const int ldv1t,
            f32* V2T, const int ldv2t,
            f32* theta, int* iwork,
            f32* work, const int lwork,
            f32* rwork, f32* result);

#endif /* VERIFY_H */
