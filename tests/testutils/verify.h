/**
 * @file verify.h
 * @brief Prototypes for LAPACK test verification routines.
 *
 * These are ports of the verification routines from LAPACK/TESTING/LIN/
 * and LAPACK/TESTING/EIG/.
 */

#ifndef VERIFY_H
#define VERIFY_H

#include <stdbool.h>
#include <stdint.h>
#include "semicolon_lapack/types.h"

/* General (DGE) verification routines */
void dget01(const int m, const int n, const f64 * const restrict A, const int lda,
            f64 * const restrict AFAC, const int ldafac, const int * const restrict ipiv,
            f64 * const restrict rwork, f64 *resid);

void dget02(const char* trans, const int m, const int n, const int nrhs,
            const f64 * const restrict A, const int lda, const f64 * const restrict X,
            const int ldx, f64 * const restrict B, const int ldb,
            f64 * const restrict rwork, f64 *resid);

void dget03(const int n, const f64 * const restrict A, const int lda,
            const f64 * const restrict AINV, const int ldainv, f64 * const restrict work,
            const int ldwork, f64 * const restrict rwork, f64 *rcond, f64 *resid);

void dget04(const int n, const int nrhs, const f64 * const restrict X, const int ldx,
            const f64 * const restrict XACT, const int ldxact, const f64 rcond,
            f64 *resid);

f64 dget06(const f64 rcond, const f64 rcondc);

void dget07(const char* trans, const int n, const int nrhs,
            const f64 * const restrict A, const int lda,
            const f64 * const restrict B, const int ldb,
            const f64 * const restrict X, const int ldx,
            const f64 * const restrict XACT, const int ldxact,
            const f64 * const restrict ferr, const bool chkferr,
            const f64 * const restrict berr, f64 * const restrict reslts);

void dget08(const char* trans, const int m, const int n, const int nrhs,
            const f64* A, const int lda, const f64* X, const int ldx,
            f64* B, const int ldb, f64* rwork, f64* resid);

/* Banded (GB) verification routines */
void dgbt01(int m, int n, int kl, int ku,
            const f64* A, int lda,
            const f64* AFAC, int ldafac,
            const int* ipiv,
            f64* work,
            f64* resid);

void dgbt02(const char* trans, int m, int n, int kl, int ku, int nrhs,
            const f64* A, int lda,
            const f64* X, int ldx,
            f64* B, int ldb,
            f64* rwork,
            f64* resid);

void dgbt05(const char* trans, int n, int kl, int ku, int nrhs,
            const f64* AB, int ldab,
            const f64* B, int ldb,
            const f64* X, int ldx,
            const f64* XACT, int ldxact,
            const f64* FERR,
            const f64* BERR,
            f64* reslts);

/* Positive definite banded (PB) verification routines */
void dpbt01(const char* uplo, const int n, const int kd,
            const f64* A, const int lda,
            f64* AFAC, const int ldafac,
            f64* rwork, f64* resid);

void dpbt02(const char* uplo, const int n, const int kd, const int nrhs,
            const f64* A, const int lda,
            const f64* X, const int ldx,
            f64* B, const int ldb,
            f64* rwork, f64* resid);

void dpbt05(const char* uplo, const int n, const int kd, const int nrhs,
            const f64* AB, const int ldab,
            const f64* B, const int ldb,
            const f64* X, const int ldx,
            const f64* XACT, const int ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts);

/* Tridiagonal (DGT) verification routines */
void dgtt01(const int n, const f64 * const restrict DL, const f64 * const restrict D,
            const f64 * const restrict DU, const f64 * const restrict DLF,
            const f64 * const restrict DF, const f64 * const restrict DUF,
            const f64 * const restrict DU2, const int * const restrict ipiv,
            f64 * const restrict work, const int ldwork, f64 *resid);

void dgtt02(const char* trans, const int n, const int nrhs,
            const f64 * const restrict DL, const f64 * const restrict D,
            const f64 * const restrict DU, const f64 * const restrict X, const int ldx,
            f64 * const restrict B, const int ldb, f64 *resid);

void dgtt05(const char* trans, const int n, const int nrhs,
            const f64 * const restrict DL, const f64 * const restrict D,
            const f64 * const restrict DU, const f64 * const restrict B, const int ldb,
            const f64 * const restrict X, const int ldx,
            const f64 * const restrict XACT, const int ldxact,
            const f64 * const restrict ferr, const f64 * const restrict berr,
            f64 * const restrict reslts);

/* Matrix generation routines */
void dlatb4(const char *path, const int imat, const int m, const int n,
            char *type, int *kl, int *ku, f64 *anorm, int *mode,
            f64 *cndnum, char *dist);

void dlatms(const int m, const int n, const char* dist,
            const char* sym, f64 *d, const int mode, const f64 cond,
            const f64 dmax, const int kl, const int ku, const char* pack,
            f64 *A, const int lda, f64 *work, int *info,
            uint64_t state[static 4]);

void dlatmt(const int m, const int n, const char* dist,
            const char* sym, f64* d, const int mode,
            const f64 cond, const f64 dmax, const int rank,
            const int kl, const int ku, const char* pack,
            f64* A, const int lda, f64* work, int* info,
            uint64_t state[static 4]);

void dlatm1(const int mode, const f64 cond, const int irsign,
            const int idist, f64* d, const int n, int* info,
            uint64_t state[static 4]);

void dlagge(const int m, const int n, const int kl, const int ku,
            const f64* d, f64* A, const int lda,
            f64* work, int* info, uint64_t state[static 4]);

void dlagsy(const int n, const int k, const f64* d, f64* A,
            const int lda, f64* work, int* info,
            uint64_t state[static 4]);

void dlarot(const int lrows, const int lleft, const int lright,
            const int nl, const f64 c, const f64 s,
            f64* A, const int lda, f64* xleft, f64* xright);

void dlarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const int m, const int n, const int kl,
            const int ku, const int nrhs, const f64* A, const int lda,
            f64* X, const int ldx, f64* B, const int ldb,
            int* info, uint64_t state[static 4]);

void dlaord(const char* job, const int n, f64* X, const int incx);

/* Symmetric (SY) verification routines */
void dlavsy(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f64* const restrict A, const int lda,
            const int* const restrict ipiv, f64* const restrict B, const int ldb, int* info);

void dsyt01(const char* uplo, const int n, const f64* const restrict A, const int lda,
            const f64* const restrict AFAC, const int ldafac, const int* const restrict ipiv,
            f64* const restrict C, const int ldc, f64* const restrict rwork, f64* resid);

/* Cholesky (PO) verification routines */
void dpot01(const char* uplo, const int n, const f64* const restrict A, const int lda,
            f64* const restrict AFAC, const int ldafac, f64* const restrict rwork,
            f64* resid);

void dpot02(const char* uplo, const int n, const int nrhs,
            const f64* const restrict A, const int lda,
            const f64* const restrict X, const int ldx,
            f64* const restrict B, const int ldb,
            f64* const restrict rwork, f64* resid);

void dpot03(const char* uplo, const int n, const f64* const restrict A, const int lda,
            f64* const restrict AINV, const int ldainv, f64* const restrict work,
            const int ldwork, f64* const restrict rwork, f64* rcond, f64* resid);

void dpot05(const char* uplo, const int n, const int nrhs,
            const f64* const restrict A, const int lda,
            const f64* const restrict B, const int ldb,
            const f64* const restrict X, const int ldx,
            const f64* const restrict XACT, const int ldxact,
            const f64* const restrict ferr, const f64* const restrict berr,
            f64* const restrict reslts);

void dpot06(const char* uplo, const int n, const int nrhs,
            const f64* A, const int lda, const f64* X, const int ldx,
            f64* B, const int ldb, f64* rwork, f64* resid);

/* Positive semidefinite pivoted Cholesky (PS) verification routines */
int dgennd(const int m, const int n, const f64* const restrict A, const int lda);

void dpst01(const char* uplo, const int n,
            const f64* const restrict A, const int lda,
            f64* const restrict AFAC, const int ldafac,
            f64* const restrict PERM, const int ldperm,
            const int* const restrict piv,
            f64* const restrict rwork, f64* resid, const int rank);

/* Packed Cholesky (PP) verification routines */
void dppt01(const char* uplo, const int n, const f64* const restrict A,
            f64* const restrict AFAC, f64* const restrict rwork,
            f64* resid);

void dppt02(const char* uplo, const int n, const int nrhs,
            const f64* const restrict A,
            const f64* const restrict X, const int ldx,
            f64* const restrict B, const int ldb,
            f64* const restrict rwork, f64* resid);

void dppt03(const char* uplo, const int n, const f64* const restrict A,
            const f64* const restrict AINV, f64* const restrict work,
            const int ldwork, f64* const restrict rwork,
            f64* rcond, f64* resid);

void dppt05(const char* uplo, const int n, const int nrhs,
            const f64* const restrict AP,
            const f64* const restrict B, const int ldb,
            const f64* const restrict X, const int ldx,
            const f64* const restrict XACT, const int ldxact,
            const f64* const restrict FERR, const f64* const restrict BERR,
            f64* const restrict reslts);

/* QR verification routines */
void dqrt01(const int m, const int n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict R,
            const int lda, f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dqrt02(const int m, const int n, const int k, const f64 * const restrict A,
            const f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict R,
            const int lda, const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dqrt03(const int m, const int n, const int k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const int lda, const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dqrt04(const int m, const int n, const int nb, f64 * restrict result);

void dqrt05(const int m, const int n, const int l, const int nb, f64 * restrict result);

void dqrt01p(const int m, const int n,
             const f64 * const restrict A,
             f64 * const restrict AF,
             f64 * const restrict Q,
             f64 * const restrict R,
             const int lda,
             f64 * const restrict tau,
             f64 * const restrict work, const int lwork,
             f64 * const restrict rwork,
             f64 * restrict result);

/* LQ verification routines */
void dlqt01(const int m, const int n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict L,
            const int lda, f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dlqt02(const int m, const int n, const int k, const f64 * const restrict A,
            const f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict L,
            const int lda, const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dlqt03(const int m, const int n, const int k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const int lda, const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dlqt04(const int m, const int n, const int nb, f64 * restrict result);

void dlqt05(const int m, const int n, const int l, const int nb, f64 * restrict result);

/* Householder reconstruction verification routines */
void dorhr_col01(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, f64 * restrict result);

void dorhr_col02(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, f64 * restrict result);

/* QL verification routines */
void dqlt01(const int m, const int n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict L,
            const int lda, f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dqlt02(const int m, const int n, const int k,
            const f64 * const restrict A, const f64 * const restrict AF,
            f64 * const restrict Q, f64 * const restrict L, const int lda,
            const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void dqlt03(const int m, const int n, const int k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const int lda, const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

/* RQ verification routines */
void drqt01(const int m, const int n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict R,
            const int lda, f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void drqt02(const int m, const int n, const int k,
            const f64 * const restrict A, const f64 * const restrict AF,
            f64 * const restrict Q, f64 * const restrict R, const int lda,
            const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

void drqt03(const int m, const int n, const int k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const int lda, const f64 * const restrict tau, f64 * const restrict work,
            const int lwork, f64 * const restrict rwork, f64 * restrict result);

/* QR with pivoting verification routines */
f64 dqpt01(const int m, const int n, const int k, const f64* A, const f64* AF,
              const int lda, const f64* tau, const int* jpvt,
              f64* work, const int lwork);

f64 dqrt11(const int m, const int k, const f64* A, const int lda,
              const f64* tau, f64* work, const int lwork);

f64 dqrt12(const int m, const int n, const f64* A, const int lda,
              const f64* S, f64* work, const int lwork);

/* RZ factorization verification routines */
f64 drzt01(const int m, const int n, const f64* A, const f64* AF,
              const int lda, const f64* tau, f64* work, const int lwork);

f64 drzt02(const int m, const int n, const f64* AF, const int lda,
              const f64* tau, f64* work, const int lwork);

/* Least squares verification routines */
void dqrt13(const int scale, const int m, const int n,
            f64* A, const int lda, f64* norma,
            uint64_t state[static 4]);

f64 dqrt14(const char* trans, const int m, const int n, const int nrhs,
              const f64* A, const int lda, const f64* X, const int ldx,
              f64* work, const int lwork);

void dqrt15(const int scale, const int rksel,
            const int m, const int n, const int nrhs,
            f64* A, const int lda, f64* B, const int ldb,
            f64* S, int* rank, f64* norma, f64* normb,
            f64* work, const int lwork,
            uint64_t state[static 4]);

void dqrt16(const char* trans, const int m, const int n, const int nrhs,
            const f64* A, const int lda,
            const f64* X, const int ldx,
            f64* B, const int ldb,
            f64* rwork, f64* resid);

f64 dqrt17(const char* trans, const int iresid,
              const int m, const int n, const int nrhs,
              const f64* A, const int lda,
              const f64* X, const int ldx,
              const f64* B, const int ldb,
              f64* C,
              f64* work, const int lwork);

/* Orthogonal random matrix generator */
void dlaror(const char* side, const char* init,
            const int m, const int n,
            f64* A, const int lda,
            f64* X, int* info,
            uint64_t state[static 4]);

void dlarge(const int n, f64* A, const int lda,
            f64* work, int* info, uint64_t state[static 4]);

void dlarnv_rng(const int idist, const int n, f64* x,
                uint64_t state[static 4]);

f64 dlaran_rng(uint64_t state[static 4]);

/* Triangular verification routines */
void dtrt01(const char* uplo, const char* diag, const int n,
            const f64* A, const int lda, f64* AINV, const int ldainv,
            f64* rcond, f64* work, f64* resid);

void dtrt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f64* A, const int lda,
            const f64* X, const int ldx, const f64* B, const int ldb,
            f64* work, f64* resid);

void dtrt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f64* A, const int lda,
            const f64 scale, const f64* cnorm, const f64 tscal,
            const f64* X, const int ldx, const f64* B, const int ldb,
            f64* work, f64* resid);

void dtrt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f64* A, const int lda,
            const f64* B, const int ldb, const f64* X, const int ldx,
            const f64* XACT, const int ldxact,
            const f64* ferr, const f64* berr, f64* reslts);

void dtrt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const int n,
            const f64* A, const int lda, f64* work, f64* rat);

/* Triangular matrix generation */
void dlattr(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, f64* A, const int lda,
            f64* B, f64* work, int* info,
            uint64_t state[static 4]);

/* Triangular packed (TP) verification routines */
void dtpt01(const char* uplo, const char* diag, const int n,
            const f64* AP, f64* AINVP,
            f64* rcond, f64* work, f64* resid);

void dtpt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f64* AP, const f64* X, const int ldx,
            const f64* B, const int ldb,
            f64* work, f64* resid);

void dtpt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f64* AP, const f64 scale, const f64* cnorm,
            const f64 tscal, const f64* X, const int ldx,
            const f64* B, const int ldb,
            f64* work, f64* resid);

void dtpt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f64* AP, const f64* B, const int ldb,
            const f64* X, const int ldx,
            const f64* XACT, const int ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts);

void dtpt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const int n,
            const f64* AP, f64* work, f64* rat);

/* Triangular packed matrix generation */
void dlattp(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, f64* AP, f64* B, f64* work,
            int* info, uint64_t state[static 4]);

/* Eigenvalue verification routines */
void dstech(const int n, const f64* const restrict A, const f64* const restrict B,
            const f64* const restrict eig, const f64 tol,
            f64* const restrict work, int* info);

void dstt21(const int n, const int kband, const f64* const restrict AD,
            const f64* const restrict AE, const f64* const restrict SD,
            const f64* const restrict SE, const f64* const restrict U, const int ldu,
            f64* const restrict work, f64* restrict result);

void dstt22(const int n, const int m, const int kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const f64* const restrict U, const int ldu,
            f64* const restrict work, const int ldwork,
            f64* restrict result);

void dsyt21(const int itype, const char* uplo, const int n, const int kband,
            const f64* const restrict A, const int lda,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict U, const int ldu,
            f64* restrict V, const int ldv,
            const f64* const restrict tau,
            f64* const restrict work, f64* restrict result);

void dsyt22(const int itype, const char* uplo, const int n, const int m,
            const int kband, const f64* const restrict A, const int lda,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict U, const int ldu,
            const f64* const restrict V, const int ldv,
            const f64* const restrict tau,
            f64* const restrict work, f64* restrict result);

f64 dsxt1(const int ijob, const f64* const restrict D1, const int n1,
             const f64* const restrict D2, const int n2,
             const f64 abstol, const f64 ulp, const f64 unfl);

/* Symmetric band eigenvector verification */
void dsbt21(const char* uplo, const int n, const int ka, const int ks,
            const f64* A, const int lda,
            const f64* D, const f64* E,
            const f64* U, const int ldu,
            f64* work, f64* result);

/* Symmetric generalized eigenvector verification */
void dsgt01(const int itype, const char* uplo, const int n, const int m,
            const f64* A, const int lda,
            const f64* B, const int ldb,
            f64* Z, const int ldz,
            const f64* D, f64* work, f64* result);

/* Symmetric packed eigenvector verification */
void dspt21(const int itype, const char* uplo, const int n, const int kband,
            const f64* AP, const f64* D, const f64* E,
            const f64* U, const int ldu,
            f64* VP, const f64* tau,
            f64* work, f64* result);

/* Tridiagonal eigenvalue count (Sturm sequence) */
void dstect(const int n, const f64* a, const f64* b,
            const f64 shift, int* num);

/* Positive definite tridiagonal (PT) verification routines */
void dptt01(const int n, const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict DF, const f64* const restrict EF,
            f64* const restrict work, f64* resid);

void dptt02(const int n, const int nrhs, const f64* const restrict D,
            const f64* const restrict E, const f64* const restrict X, const int ldx,
            f64* const restrict B, const int ldb, f64* resid);

void dptt05(const int n, const int nrhs, const f64* const restrict D,
            const f64* const restrict E, const f64* const restrict B, const int ldb,
            const f64* const restrict X, const int ldx,
            const f64* const restrict XACT, const int ldxact,
            const f64* const restrict FERR, const f64* const restrict BERR,
            f64* const restrict reslts);

void dlaptm(const int n, const int nrhs, const f64 alpha,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict X, const int ldx, const f64 beta,
            f64* const restrict B, const int ldb);

/* SVD verification routines */
void dbdt01(const int m, const int n, const int kd,
            const f64* const restrict A, const int lda,
            const f64* const restrict Q, const int ldq,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict PT, const int ldpt,
            f64* const restrict work, f64* resid);

void dbdt02(const int m, const int n,
            const f64* const restrict B, const int ldb,
            const f64* const restrict C, const int ldc,
            const f64* const restrict U, const int ldu,
            f64* const restrict work, f64* resid);

void dbdt03(const char* uplo, const int n, const int kd,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict U, const int ldu,
            const f64* const restrict S,
            const f64* const restrict VT, const int ldvt,
            f64* const restrict work, f64* resid);

void dbdt04(const char* uplo, const int n,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict S, const int ns,
            const f64* const restrict U, const int ldu,
            const f64* const restrict VT, const int ldvt,
            f64* const restrict work, f64* resid);

void dort03(const char* rc, const int mu, const int mv, const int n,
            const int k, const f64* const restrict U, const int ldu,
            const f64* const restrict V, const int ldv,
            f64* const restrict work, const int lwork,
            f64* result, int* info);

void dbdt05(const int m, const int n, const f64* const restrict A, const int lda,
            const f64* const restrict S, const int ns,
            const f64* const restrict U, const int ldu,
            const f64* const restrict VT, const int ldvt,
            f64* const restrict work, f64* resid);

/* Generalized eigenvalue verification routines */
void dget51(const int itype, const int n,
            const f64* A, const int lda,
            const f64* B, const int ldb,
            const f64* U, const int ldu,
            const f64* V, const int ldv,
            f64* work, f64* result);

void dget52(const int left, const int n,
            const f64* A, const int lda,
            const f64* B, const int ldb,
            const f64* E, const int lde,
            const f64* alphar, const f64* alphai, const f64* beta,
            f64* work, f64* result);

void dget53(const f64* A, const int lda,
            const f64* B, const int ldb,
            const f64 scale, const f64 wr, const f64 wi,
            f64* result, int* info);

/* Non-symmetric eigenvalue verification routines */
void dget10(const int m, const int n,
            const f64* const restrict A, const int lda,
            const f64* const restrict B, const int ldb,
            f64* const restrict work, f64* result);

void dort01(const char* rowcol, const int m, const int n,
            const f64* U, const int ldu,
            f64* work, const int lwork, f64* resid);

void dhst01(const int n, const int ilo, const int ihi,
            const f64* A, const int lda,
            const f64* H, const int ldh,
            const f64* Q, const int ldq,
            f64* work, const int lwork, f64* result);

void dget22(const char* transa, const char* transe, const char* transw,
            const int n, const f64* A, const int lda,
            const f64* E, const int lde,
            const f64* wr, const f64* wi,
            f64* work, f64* result);

void dlatm4(const int itype, const int n, const int nz1, const int nz2,
            const int isign, const f64 amagn, const f64 rcond,
            const f64 triang, const int idist,
            f64* A, const int lda, uint64_t state[static 4]);

void dlatme(const int n, const char* dist, f64* D,
            const int mode, const f64 cond, const f64 dmax,
            const char* ei, const char* rsign, const char* upper,
            const char* sim, f64* DS, const int modes, const f64 conds,
            const int kl, const int ku, const f64 anorm,
            f64* A, const int lda, f64* work, int* info,
            uint64_t state[static 4]);

f64 dlatm2(const int m, const int n, const int i, const int j,
              const int kl, const int ku, const int idist,
              const f64* d, const int igrade,
              const f64* dl, const f64* dr,
              const int ipvtng, const int* iwork, const f64 sparse,
              uint64_t state[static 4]);

f64 dlatm3(const int m, const int n, const int i, const int j,
              int* isub, int* jsub, const int kl, const int ku,
              const int idist, const f64* d, const int igrade,
              const f64* dl, const f64* dr,
              const int ipvtng, const int* iwork, const f64 sparse,
              uint64_t state[static 4]);

void dlatmr(const int m, const int n, const char* dist, const char* sym,
            f64* d, const int mode, const f64 cond, const f64 dmax,
            const char* rsign, const char* grade, f64* dl,
            const int model, const f64 condl, f64* dr,
            const int moder, const f64 condr, const char* pivtng,
            const int* ipivot, const int kl, const int ku,
            const f64 sparse, const f64 anorm, const char* pack,
            f64* A, const int lda, int* iwork, int* info,
            uint64_t state[static 4]);

/* GLM verification routines */
void dglmts(const int n, const int m, const int p,
            const f64* A, f64* AF, const int lda,
            const f64* B, f64* BF, const int ldb,
            const f64* D, f64* DF, f64* X, f64* U,
            f64* work, const int lwork, f64* rwork,
            f64* result);

/* LSE verification routines */
void dlsets(const int m, const int p, const int n,
            const f64* A, f64* AF, const int lda,
            const f64* B, f64* BF, const int ldb,
            const f64* C, f64* CF, const f64* D, f64* DF,
            f64* X, f64* work, const int lwork,
            f64* rwork, f64* result);

/* Matrix parameter setup for GLM/GQR/GRQ/GSV/LSE tests */
void dlatb9(const char* path, const int imat, const int m, const int p, const int n,
            char* type, int* kla, int* kua, int* klb, int* kub,
            f64* anorm, f64* bnorm, int* modea, int* modeb,
            f64* cndnma, f64* cndnmb, char* dista, char* distb);

/* GQR/GRQ verification routines */
void dgqrts(const int n, const int m, const int p,
            const f64* A, f64* AF, f64* Q, f64* R,
            const int lda, f64* taua,
            const f64* B, f64* BF, f64* Z, f64* T,
            f64* BWK, const int ldb, f64* taub,
            f64* work, const int lwork, f64* rwork,
            f64* result);

void dgrqts(const int m, const int p, const int n,
            const f64* A, f64* AF, f64* Q, f64* R,
            const int lda, f64* taua,
            const f64* B, f64* BF, f64* Z, f64* T,
            f64* BWK, const int ldb, f64* taub,
            f64* work, const int lwork, f64* rwork,
            f64* result);

/* GSVD verification routines */
void dgsvts3(const int m, const int p, const int n,
             const f64* A, f64* AF, const int lda,
             const f64* B, f64* BF, const int ldb,
             f64* U, const int ldu,
             f64* V, const int ldv,
             f64* Q, const int ldq,
             f64* alpha, f64* beta,
             f64* R, const int ldr,
             int* iwork,
             f64* work, const int lwork,
             f64* rwork,
             f64* result);

/* Additional matrix generators (MATGEN) */

/* Hilbert matrix generator */
void dlahilb(const int n, const int nrhs,
             f64* A, const int lda,
             f64* X, const int ldx,
             f64* B, const int ldb,
             f64* work, int* info);

/* Kronecker product block matrix */
void dlakf2(const int m, const int n,
            const f64* A, const int lda,
            const f64* B, const f64* D, const f64* E,
            f64* Z, const int ldz);

/* Singular value distribution */
void dlatm7(const int mode, const f64 cond, const int irsign,
            const int idist, f64* d, const int n, const int rank,
            int* info, uint64_t state[static 4]);

/* Generalized Sylvester test matrices */
void dlatm5(const int prtype, const int m, const int n,
            f64* A, const int lda,
            f64* B, const int ldb,
            f64* C, const int ldc,
            f64* D, const int ldd,
            f64* E, const int lde,
            f64* F, const int ldf,
            f64* R, const int ldr,
            f64* L, const int ldl,
            const f64 alpha, int qblcka, int qblckb);

/* Generalized eigenvalue test matrices */
void dlatm6(const int type, const int n,
            f64* A, const int lda, f64* B,
            f64* X, const int ldx, f64* Y, const int ldy,
            const f64 alpha, const f64 beta,
            const f64 wx, const f64 wy,
            f64* S, f64* DIF);

/* Matrix parameter setup for tridiagonal/banded tests */
void dlatb5(const char* path, const int imat, const int n,
            char* type, int* kl, int* ku, f64* anorm,
            int* mode, f64* cndnum, char* dist);

/* QL/RQ solve helpers */
void dgeqls(const int m, const int n, const int nrhs,
            f64* A, const int lda, const f64* tau,
            f64* B, const int ldb,
            f64* work, const int lwork, int* info);

void dgerqs(const int m, const int n, const int nrhs,
            f64* A, const int lda, const f64* tau,
            f64* B, const int ldb,
            f64* work, const int lwork, int* info);

/* Packed symmetric (SP) verification routines */
void dspt01(const char* uplo, const int n, const f64* A,
            const f64* AFAC, const int* ipiv, f64* C, const int ldc,
            f64* rwork, f64* resid);

/* Packed symmetric multiply (from DSPTRF factorization) */
void dlavsp(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f64* const restrict A,
            const int* const restrict ipiv,
            f64* const restrict B, const int ldb, int* info);

/* Symmetric Rook multiply (from DSYTRF_ROOK factorization) */
void dlavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const int n, const int nrhs,
                 const f64* const restrict A, const int lda,
                 const int* const restrict ipiv,
                 f64* const restrict B, const int ldb, int* info);

/* DGEEVX verification */
void dget23(const int comp, const char* balanc, const int jtype,
            const f64 thresh, const int n,
            f64* A, const int lda, f64* H,
            f64* wr, f64* wi, f64* wr1, f64* wi1,
            f64* VL, const int ldvl, f64* VR, const int ldvr,
            f64* LRE, const int ldlre,
            f64* rcondv, f64* rcndv1, const f64* rcdvin,
            f64* rconde, f64* rcnde1, const f64* rcdein,
            f64* scale, f64* scale1, f64* result,
            f64* work, const int lwork, int* iwork, int* info);

/* DGEESX verification */
void dget24(const int comp, const int jtype, const f64 thresh,
            const int n, f64* A, const int lda,
            f64* H, f64* HT,
            f64* wr, f64* wi, f64* wrt, f64* wit,
            f64* wrtmp, f64* witmp,
            f64* VS, const int ldvs, f64* VS1,
            const f64 rcdein, const f64 rcdvin,
            const int nslct, const int* islct,
            f64* result, f64* work, const int lwork,
            int* iwork, int* bwork, int* info);

/* Triangular banded (TB) verification routines */
void dtbt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f64* AB, const int ldab,
            const f64* X, const int ldx,
            const f64* B, const int ldb,
            f64* work, f64* resid);

void dtbt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f64* AB, const int ldab,
            const f64 scale, const f64* cnorm, const f64 tscal,
            const f64* X, const int ldx,
            const f64* B, const int ldb,
            f64* work, f64* resid);

void dtbt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f64* AB, const int ldab,
            const f64* B, const int ldb,
            const f64* X, const int ldx,
            const f64* XACT, const int ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts);

void dtbt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const int n, const int kd,
            const f64* AB, const int ldab, f64* work, f64* rat);

/* Triangular banded matrix generation */
void dlattb(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, const int kd,
            f64* AB, const int ldab, f64* B,
            f64* work, int* info, uint64_t state[static 4]);

/* Symmetric verification routines (BK variants) */
void dsyt01_3(const char* uplo, const int n,
              const f64* const restrict A, const int lda,
              f64* const restrict AFAC, const int ldafac,
              f64* const restrict E,
              int* const restrict ipiv,
              f64* const restrict C, const int ldc,
              f64* const restrict rwork, f64* resid);

void dsyt01_aa(const char* uplo, const int n,
               const f64* const restrict A, const int lda,
               const f64* const restrict AFAC, const int ldafac,
               const int* const restrict ipiv,
               f64* const restrict C, const int ldc,
               f64* const restrict rwork, f64* resid);

void dsyt01_rook(const char* uplo, const int n,
                 const f64* const restrict A, const int lda,
                 const f64* const restrict AFAC, const int ldafac,
                 const int* const restrict ipiv,
                 f64* const restrict C, const int ldc,
                 f64* const restrict rwork, f64* resid);

/* TSQR verification */
void dtsqr01(const char* tssw, const int m, const int n, const int mb,
             const int nb, f64* result);

#endif /* VERIFY_H */
