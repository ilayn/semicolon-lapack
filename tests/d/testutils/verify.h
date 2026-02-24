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

/* General (DGE) verification routines */
void dget01(const INT m, const INT n, const f64 * const restrict A, const INT lda,
            f64 * const restrict AFAC, const INT ldafac, const INT * const restrict ipiv,
            f64 * const restrict rwork, f64 *resid);

void dget02(const char* trans, const INT m, const INT n, const INT nrhs,
            const f64 * const restrict A, const INT lda, const f64 * const restrict X,
            const INT ldx, f64 * const restrict B, const INT ldb,
            f64 * const restrict rwork, f64 *resid);

void dget03(const INT n, const f64 * const restrict A, const INT lda,
            const f64 * const restrict AINV, const INT ldainv, f64 * const restrict work,
            const INT ldwork, f64 * const restrict rwork, f64 *rcond, f64 *resid);

void dget04(const INT n, const INT nrhs, const f64 * const restrict X, const INT ldx,
            const f64 * const restrict XACT, const INT ldxact, const f64 rcond,
            f64 *resid);

f64 dget06(const f64 rcond, const f64 rcondc);

void dget07(const char* trans, const INT n, const INT nrhs,
            const f64 * const restrict A, const INT lda,
            const f64 * const restrict B, const INT ldb,
            const f64 * const restrict X, const INT ldx,
            const f64 * const restrict XACT, const INT ldxact,
            const f64 * const restrict ferr, const INT chkferr,
            const f64 * const restrict berr, f64 * const restrict reslts);

void dget08(const char* trans, const INT m, const INT n, const INT nrhs,
            const f64* A, const INT lda, const f64* X, const INT ldx,
            f64* B, const INT ldb, f64* rwork, f64* resid);

/* Banded (GB) verification routines */
void dgbt01(INT m, INT n, INT kl, INT ku,
            const f64* A, INT lda,
            const f64* AFAC, INT ldafac,
            const INT* ipiv,
            f64* work,
            f64* resid);

void dgbt02(const char* trans, INT m, INT n, INT kl, INT ku, INT nrhs,
            const f64* A, INT lda,
            const f64* X, INT ldx,
            f64* B, INT ldb,
            f64* rwork,
            f64* resid);

void dgbt05(const char* trans, INT n, INT kl, INT ku, INT nrhs,
            const f64* AB, INT ldab,
            const f64* B, INT ldb,
            const f64* X, INT ldx,
            const f64* XACT, INT ldxact,
            const f64* FERR,
            const f64* BERR,
            f64* reslts);

/* Positive definite banded (PB) verification routines */
void dpbt01(const char* uplo, const INT n, const INT kd,
            const f64* A, const INT lda,
            f64* AFAC, const INT ldafac,
            f64* rwork, f64* resid);

void dpbt02(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const f64* A, const INT lda,
            const f64* X, const INT ldx,
            f64* B, const INT ldb,
            f64* rwork, f64* resid);

void dpbt05(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const f64* AB, const INT ldab,
            const f64* B, const INT ldb,
            const f64* X, const INT ldx,
            const f64* XACT, const INT ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts);

/* Tridiagonal (DGT) verification routines */
void dgtt01(const INT n, const f64 * const restrict DL, const f64 * const restrict D,
            const f64 * const restrict DU, const f64 * const restrict DLF,
            const f64 * const restrict DF, const f64 * const restrict DUF,
            const f64 * const restrict DU2, const INT * const restrict ipiv,
            f64 * const restrict work, const INT ldwork, f64 *resid);

void dgtt02(const char* trans, const INT n, const INT nrhs,
            const f64 * const restrict DL, const f64 * const restrict D,
            const f64 * const restrict DU, const f64 * const restrict X, const INT ldx,
            f64 * const restrict B, const INT ldb, f64 *resid);

void dgtt05(const char* trans, const INT n, const INT nrhs,
            const f64 * const restrict DL, const f64 * const restrict D,
            const f64 * const restrict DU, const f64 * const restrict B, const INT ldb,
            const f64 * const restrict X, const INT ldx,
            const f64 * const restrict XACT, const INT ldxact,
            const f64 * const restrict ferr, const f64 * const restrict berr,
            f64 * const restrict reslts);

/* Matrix generation routines */
void dlatb4(const char* path, const INT imat, const INT m, const INT n,
            char* type, INT* kl, INT* ku, f64* anorm, INT* mode,
            f64* cndnum, char* dist);

void dlatms(const INT m, const INT n, const char* dist,
            const char* sym, f64 *d, const INT mode, const f64 cond,
            const f64 dmax, const INT kl, const INT ku, const char* pack,
            f64 *A, const INT lda, f64 *work, INT* info,
            uint64_t state[static 4]);

void dlatmt(const INT m, const INT n, const char* dist,
            const char* sym, f64* d, const INT mode,
            const f64 cond, const f64 dmax, const INT rank,
            const INT kl, const INT ku, const char* pack,
            f64* A, const INT lda, f64* work, INT* info,
            uint64_t state[static 4]);

void dlatm1(const INT mode, const f64 cond, const INT irsign,
            const INT idist, f64* d, const INT n, INT* info,
            uint64_t state[static 4]);

void dlagge(const INT m, const INT n, const INT kl, const INT ku,
            const f64* d, f64* A, const INT lda,
            f64* work, INT* info, uint64_t state[static 4]);

void dlagsy(const INT n, const INT k, const f64* d, f64* A,
            const INT lda, f64* work, INT* info,
            uint64_t state[static 4]);

void dlarot(const INT lrows, const INT lleft, const INT lright,
            const INT nl, const f64 c, const f64 s,
            f64* A, const INT lda, f64* xleft, f64* xright);

void dlarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const INT m, const INT n, const INT kl,
            const INT ku, const INT nrhs, const f64* A, const INT lda,
            f64* X, const INT ldx, f64* B, const INT ldb,
            INT* info, uint64_t state[static 4]);

void dlaord(const char* job, const INT n, f64* X, const INT incx);

/* Symmetric (SY) verification routines */
void dlavsy(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f64* const restrict A, const INT lda,
            const INT* const restrict ipiv, f64* const restrict B, const INT ldb, INT* info);

void dsyt01(const char* uplo, const INT n, const f64* const restrict A, const INT lda,
            const f64* const restrict AFAC, const INT ldafac, const INT* const restrict ipiv,
            f64* const restrict C, const INT ldc, f64* const restrict rwork, f64* resid);

/* Cholesky (PO) verification routines */
void dpot01(const char* uplo, const INT n, const f64* const restrict A, const INT lda,
            f64* const restrict AFAC, const INT ldafac, f64* const restrict rwork,
            f64* resid);

void dpot02(const char* uplo, const INT n, const INT nrhs,
            const f64* const restrict A, const INT lda,
            const f64* const restrict X, const INT ldx,
            f64* const restrict B, const INT ldb,
            f64* const restrict rwork, f64* resid);

void dpot03(const char* uplo, const INT n, const f64* const restrict A, const INT lda,
            f64* const restrict AINV, const INT ldainv, f64* const restrict work,
            const INT ldwork, f64* const restrict rwork, f64* rcond, f64* resid);

void dpot05(const char* uplo, const INT n, const INT nrhs,
            const f64* const restrict A, const INT lda,
            const f64* const restrict B, const INT ldb,
            const f64* const restrict X, const INT ldx,
            const f64* const restrict XACT, const INT ldxact,
            const f64* const restrict ferr, const f64* const restrict berr,
            f64* const restrict reslts);

void dpot06(const char* uplo, const INT n, const INT nrhs,
            const f64* A, const INT lda, const f64* X, const INT ldx,
            f64* B, const INT ldb, f64* rwork, f64* resid);

/* Positive semidefinite pivoted Cholesky (PS) verification routines */
INT dgennd(const INT m, const INT n, const f64* const restrict A, const INT lda);

void dpst01(const char* uplo, const INT n,
            const f64* const restrict A, const INT lda,
            f64* const restrict AFAC, const INT ldafac,
            f64* const restrict PERM, const INT ldperm,
            const INT* const restrict piv,
            f64* const restrict rwork, f64* resid, const INT rank);

/* Packed Cholesky (PP) verification routines */
void dppt01(const char* uplo, const INT n, const f64* const restrict A,
            f64* const restrict AFAC, f64* const restrict rwork,
            f64* resid);

void dppt02(const char* uplo, const INT n, const INT nrhs,
            const f64* const restrict A,
            const f64* const restrict X, const INT ldx,
            f64* const restrict B, const INT ldb,
            f64* const restrict rwork, f64* resid);

void dppt03(const char* uplo, const INT n, const f64* const restrict A,
            const f64* const restrict AINV, f64* const restrict work,
            const INT ldwork, f64* const restrict rwork,
            f64* rcond, f64* resid);

void dppt05(const char* uplo, const INT n, const INT nrhs,
            const f64* const restrict AP,
            const f64* const restrict B, const INT ldb,
            const f64* const restrict X, const INT ldx,
            const f64* const restrict XACT, const INT ldxact,
            const f64* const restrict FERR, const f64* const restrict BERR,
            f64* const restrict reslts);

/* QR verification routines */
void dqrt01(const INT m, const INT n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict R,
            const INT lda, f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dqrt02(const INT m, const INT n, const INT k, const f64 * const restrict A,
            const f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict R,
            const INT lda, const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dqrt03(const INT m, const INT n, const INT k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const INT lda, const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dqrt04(const INT m, const INT n, const INT nb, f64 * restrict result);

void dqrt05(const INT m, const INT n, const INT l, const INT nb, f64 * restrict result);

void dqrt01p(const INT m, const INT n,
             const f64 * const restrict A,
             f64 * const restrict AF,
             f64 * const restrict Q,
             f64 * const restrict R,
             const INT lda,
             f64 * const restrict tau,
             f64 * const restrict work, const INT lwork,
             f64 * const restrict rwork,
             f64 * restrict result);

/* LQ verification routines */
void dlqt01(const INT m, const INT n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict L,
            const INT lda, f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dlqt02(const INT m, const INT n, const INT k, const f64 * const restrict A,
            const f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict L,
            const INT lda, const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dlqt03(const INT m, const INT n, const INT k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const INT lda, const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dlqt04(const INT m, const INT n, const INT nb, f64 * restrict result);

void dlqt05(const INT m, const INT n, const INT l, const INT nb, f64 * restrict result);

/* Householder reconstruction verification routines */
void dorhr_col01(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f64 * restrict result);

void dorhr_col02(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f64 * restrict result);

/* QL verification routines */
void dqlt01(const INT m, const INT n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict L,
            const INT lda, f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dqlt02(const INT m, const INT n, const INT k,
            const f64 * const restrict A, const f64 * const restrict AF,
            f64 * const restrict Q, f64 * const restrict L, const INT lda,
            const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void dqlt03(const INT m, const INT n, const INT k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const INT lda, const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

/* RQ verification routines */
void drqt01(const INT m, const INT n, const f64 * const restrict A,
            f64 * const restrict AF, f64 * const restrict Q, f64 * const restrict R,
            const INT lda, f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void drqt02(const INT m, const INT n, const INT k,
            const f64 * const restrict A, const f64 * const restrict AF,
            f64 * const restrict Q, f64 * const restrict R, const INT lda,
            const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

void drqt03(const INT m, const INT n, const INT k, const f64 * const restrict AF,
            f64 * const restrict C, f64 * const restrict CC, f64 * const restrict Q,
            const INT lda, const f64 * const restrict tau, f64 * const restrict work,
            const INT lwork, f64 * const restrict rwork, f64 * restrict result);

/* QR with pivoting verification routines */
f64 dqpt01(const INT m, const INT n, const INT k, const f64* A, const f64* AF,
              const INT lda, const f64* tau, const INT* jpvt,
              f64* work, const INT lwork);

f64 dqrt11(const INT m, const INT k, const f64* A, const INT lda,
              const f64* tau, f64* work, const INT lwork);

f64 dqrt12(const INT m, const INT n, const f64* A, const INT lda,
              const f64* S, f64* work, const INT lwork);

/* RZ factorization verification routines */
f64 drzt01(const INT m, const INT n, const f64* A, const f64* AF,
              const INT lda, const f64* tau, f64* work, const INT lwork);

f64 drzt02(const INT m, const INT n, const f64* AF, const INT lda,
              const f64* tau, f64* work, const INT lwork);

/* Least squares verification routines */
void dqrt13(const INT scale, const INT m, const INT n,
            f64* A, const INT lda, f64* norma,
            uint64_t state[static 4]);

f64 dqrt14(const char* trans, const INT m, const INT n, const INT nrhs,
              const f64* A, const INT lda, const f64* X, const INT ldx,
              f64* work, const INT lwork);

void dqrt15(const INT scale, const INT rksel,
            const INT m, const INT n, const INT nrhs,
            f64* A, const INT lda, f64* B, const INT ldb,
            f64* S, INT* rank, f64* norma, f64* normb,
            f64* work, const INT lwork,
            uint64_t state[static 4]);

void dqrt16(const char* trans, const INT m, const INT n, const INT nrhs,
            const f64* A, const INT lda,
            const f64* X, const INT ldx,
            f64* B, const INT ldb,
            f64* rwork, f64* resid);

f64 dqrt17(const char* trans, const INT iresid,
              const INT m, const INT n, const INT nrhs,
              const f64* A, const INT lda,
              const f64* X, const INT ldx,
              const f64* B, const INT ldb,
              f64* C,
              f64* work, const INT lwork);

/* Orthogonal random matrix generator */
void dlaror(const char* side, const char* init,
            const INT m, const INT n,
            f64* A, const INT lda,
            f64* X, INT* info,
            uint64_t state[static 4]);

void dlarge(const INT n, f64* A, const INT lda,
            f64* work, INT* info, uint64_t state[static 4]);

void dlarnv_rng(const INT idist, const INT n, f64* x,
                uint64_t state[static 4]);

f64 dlaran_rng(uint64_t state[static 4]);

/* Triangular verification routines */
void dtrt01(const char* uplo, const char* diag, const INT n,
            const f64* A, const INT lda, f64* AINV, const INT ldainv,
            f64* rcond, f64* work, f64* resid);

void dtrt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f64* A, const INT lda,
            const f64* X, const INT ldx, const f64* B, const INT ldb,
            f64* work, f64* resid);

void dtrt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f64* A, const INT lda,
            const f64 scale, const f64* cnorm, const f64 tscal,
            const f64* X, const INT ldx, const f64* B, const INT ldb,
            f64* work, f64* resid);

void dtrt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f64* A, const INT lda,
            const f64* B, const INT ldb, const f64* X, const INT ldx,
            const f64* XACT, const INT ldxact,
            const f64* ferr, const f64* berr, f64* reslts);

void dtrt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const INT n,
            const f64* A, const INT lda, f64* work, f64* rat);

/* Triangular matrix generation */
void dlattr(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, f64* A, const INT lda,
            f64* B, f64* work, INT* info,
            uint64_t state[static 4]);

/* Triangular packed (TP) verification routines */
void dtpt01(const char* uplo, const char* diag, const INT n,
            const f64* AP, f64* AINVP,
            f64* rcond, f64* work, f64* resid);

void dtpt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f64* AP, const f64* X, const INT ldx,
            const f64* B, const INT ldb,
            f64* work, f64* resid);

void dtpt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f64* AP, const f64 scale, const f64* cnorm,
            const f64 tscal, const f64* X, const INT ldx,
            const f64* B, const INT ldb,
            f64* work, f64* resid);

void dtpt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f64* AP, const f64* B, const INT ldb,
            const f64* X, const INT ldx,
            const f64* XACT, const INT ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts);

void dtpt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const INT n,
            const f64* AP, f64* work, f64* rat);

/* Triangular packed matrix generation */
void dlattp(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, f64* AP, f64* B, f64* work,
            INT* info, uint64_t state[static 4]);

/* Eigenvalue verification routines */
void dstech(const INT n, const f64* const restrict A, const f64* const restrict B,
            const f64* const restrict eig, const f64 tol,
            f64* const restrict work, INT* info);

void dstt21(const INT n, const INT kband, const f64* const restrict AD,
            const f64* const restrict AE, const f64* const restrict SD,
            const f64* const restrict SE, const f64* const restrict U, const INT ldu,
            f64* const restrict work, f64* restrict result);

void dstt22(const INT n, const INT m, const INT kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const f64* const restrict U, const INT ldu,
            f64* const restrict work, const INT ldwork,
            f64* restrict result);

void dsyt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const f64* const restrict A, const INT lda,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict U, const INT ldu,
            f64* restrict V, const INT ldv,
            const f64* const restrict tau,
            f64* const restrict work, f64* restrict result);

void dsyt22(const INT itype, const char* uplo, const INT n, const INT m,
            const INT kband, const f64* const restrict A, const INT lda,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict U, const INT ldu,
            const f64* const restrict V, const INT ldv,
            const f64* const restrict tau,
            f64* const restrict work, f64* restrict result);

f64 dsxt1(const INT ijob, const f64* const restrict D1, const INT n1,
             const f64* const restrict D2, const INT n2,
             const f64 abstol, const f64 ulp, const f64 unfl);

/* Symmetric band eigenvector verification */
void dsbt21(const char* uplo, const INT n, const INT ka, const INT ks,
            const f64* A, const INT lda,
            const f64* D, const f64* E,
            const f64* U, const INT ldu,
            f64* work, f64* result);

/* Symmetric generalized eigenvector verification */
void dsgt01(const INT itype, const char* uplo, const INT n, const INT m,
            const f64* A, const INT lda,
            const f64* B, const INT ldb,
            f64* Z, const INT ldz,
            const f64* D, f64* work, f64* result);

/* Symmetric packed eigenvector verification */
void dspt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const f64* AP, const f64* D, const f64* E,
            const f64* U, const INT ldu,
            f64* VP, const f64* tau,
            f64* work, f64* result);

/* Tridiagonal eigenvalue count (Sturm sequence) */
void dstect(const INT n, const f64* a, const f64* b,
            const f64 shift, INT* num);

/* Positive definite tridiagonal (PT) verification routines */
void dptt01(const INT n, const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict DF, const f64* const restrict EF,
            f64* const restrict work, f64* resid);

void dptt02(const INT n, const INT nrhs, const f64* const restrict D,
            const f64* const restrict E, const f64* const restrict X, const INT ldx,
            f64* const restrict B, const INT ldb, f64* resid);

void dptt05(const INT n, const INT nrhs, const f64* const restrict D,
            const f64* const restrict E, const f64* const restrict B, const INT ldb,
            const f64* const restrict X, const INT ldx,
            const f64* const restrict XACT, const INT ldxact,
            const f64* const restrict FERR, const f64* const restrict BERR,
            f64* const restrict reslts);

void dlaptm(const INT n, const INT nrhs, const f64 alpha,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict X, const INT ldx, const f64 beta,
            f64* const restrict B, const INT ldb);

/* SVD verification routines */
void dbdt01(const INT m, const INT n, const INT kd,
            const f64* const restrict A, const INT lda,
            const f64* const restrict Q, const INT ldq,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict PT, const INT ldpt,
            f64* const restrict work, f64* resid);

void dbdt02(const INT m, const INT n,
            const f64* const restrict B, const INT ldb,
            const f64* const restrict C, const INT ldc,
            const f64* const restrict U, const INT ldu,
            f64* const restrict work, f64* resid);

void dbdt03(const char* uplo, const INT n, const INT kd,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict U, const INT ldu,
            const f64* const restrict S,
            const f64* const restrict VT, const INT ldvt,
            f64* const restrict work, f64* resid);

void dbdt04(const char* uplo, const INT n,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict S, const INT ns,
            const f64* const restrict U, const INT ldu,
            const f64* const restrict VT, const INT ldvt,
            f64* const restrict work, f64* resid);

void dort03(const char* rc, const INT mu, const INT mv, const INT n,
            const INT k, const f64* const restrict U, const INT ldu,
            const f64* const restrict V, const INT ldv,
            f64* const restrict work, const INT lwork,
            f64* result, INT* info);

void dbdt05(const INT m, const INT n, const f64* const restrict A, const INT lda,
            const f64* const restrict S, const INT ns,
            const f64* const restrict U, const INT ldu,
            const f64* const restrict VT, const INT ldvt,
            f64* const restrict work, f64* resid);

/* Generalized eigenvalue verification routines */
void dget51(const INT itype, const INT n,
            const f64* A, const INT lda,
            const f64* B, const INT ldb,
            const f64* U, const INT ldu,
            const f64* V, const INT ldv,
            f64* work, f64* result);

void dget52(const INT left, const INT n,
            const f64* A, const INT lda,
            const f64* B, const INT ldb,
            const f64* E, const INT lde,
            const f64* alphar, const f64* alphai, const f64* beta,
            f64* work, f64* result);

void dget53(const f64* A, const INT lda,
            const f64* B, const INT ldb,
            const f64 scale, const f64 wr, const f64 wi,
            f64* result, INT* info);

/* Non-symmetric eigenvalue verification routines */
void dget10(const INT m, const INT n,
            const f64* const restrict A, const INT lda,
            const f64* const restrict B, const INT ldb,
            f64* const restrict work, f64* result);

void dort01(const char* rowcol, const INT m, const INT n,
            const f64* U, const INT ldu,
            f64* work, const INT lwork, f64* resid);

void dhst01(const INT n, const INT ilo, const INT ihi,
            const f64* A, const INT lda,
            const f64* H, const INT ldh,
            const f64* Q, const INT ldq,
            f64* work, const INT lwork, f64* result);

void dget22(const char* transa, const char* transe, const char* transw,
            const INT n, const f64* A, const INT lda,
            const f64* E, const INT lde,
            const f64* wr, const f64* wi,
            f64* work, f64* result);

void dlatm4(const INT itype, const INT n, const INT nz1, const INT nz2,
            const INT isign, const f64 amagn, const f64 rcond,
            const f64 triang, const INT idist,
            f64* A, const INT lda, uint64_t state[static 4]);

void dlatme(const INT n, const char* dist, f64* D,
            const INT mode, const f64 cond, const f64 dmax,
            const char* ei, const char* rsign, const char* upper,
            const char* sim, f64* DS, const INT modes, const f64 conds,
            const INT kl, const INT ku, const f64 anorm,
            f64* A, const INT lda, f64* work, INT* info,
            uint64_t state[static 4]);

f64 dlatm2(const INT m, const INT n, const INT i, const INT j,
              const INT kl, const INT ku, const INT idist,
              const f64* d, const INT igrade,
              const f64* dl, const f64* dr,
              const INT ipvtng, const INT* iwork, const f64 sparse,
              uint64_t state[static 4]);

f64 dlatm3(const INT m, const INT n, const INT i, const INT j,
              INT* isub, INT* jsub, const INT kl, const INT ku,
              const INT idist, const f64* d, const INT igrade,
              const f64* dl, const f64* dr,
              const INT ipvtng, const INT* iwork, const f64 sparse,
              uint64_t state[static 4]);

void dlatmr(const INT m, const INT n, const char* dist, const char* sym,
            f64* d, const INT mode, const f64 cond, const f64 dmax,
            const char* rsign, const char* grade, f64* dl,
            const INT model, const f64 condl, f64* dr,
            const INT moder, const f64 condr, const char* pivtng,
            const INT* ipivot, const INT kl, const INT ku,
            const f64 sparse, const f64 anorm, const char* pack,
            f64* A, const INT lda, INT* iwork, INT* info,
            uint64_t state[static 4]);

/* GLM verification routines */
void dglmts(const INT n, const INT m, const INT p,
            const f64* A, f64* AF, const INT lda,
            const f64* B, f64* BF, const INT ldb,
            const f64* D, f64* DF, f64* X, f64* U,
            f64* work, const INT lwork, f64* rwork,
            f64* result);

/* LSE verification routines */
void dlsets(const INT m, const INT p, const INT n,
            const f64* A, f64* AF, const INT lda,
            const f64* B, f64* BF, const INT ldb,
            const f64* C, f64* CF, const f64* D, f64* DF,
            f64* X, f64* work, const INT lwork,
            f64* rwork, f64* result);

/* Matrix parameter setup for GLM/GQR/GRQ/GSV/LSE tests */
void dlatb9(const char* path, const INT imat, const INT m, const INT p, const INT n,
            char* type, INT* kla, INT* kua, INT* klb, INT* kub,
            f64* anorm, f64* bnorm, INT* modea, INT* modeb,
            f64* cndnma, f64* cndnmb, char* dista, char* distb);

/* GQR/GRQ verification routines */
void dgqrts(const INT n, const INT m, const INT p,
            const f64* A, f64* AF, f64* Q, f64* R,
            const INT lda, f64* taua,
            const f64* B, f64* BF, f64* Z, f64* T,
            f64* BWK, const INT ldb, f64* taub,
            f64* work, const INT lwork, f64* rwork,
            f64* result);

void dgrqts(const INT m, const INT p, const INT n,
            const f64* A, f64* AF, f64* Q, f64* R,
            const INT lda, f64* taua,
            const f64* B, f64* BF, f64* Z, f64* T,
            f64* BWK, const INT ldb, f64* taub,
            f64* work, const INT lwork, f64* rwork,
            f64* result);

/* GSVD verification routines */
void dgsvts3(const INT m, const INT p, const INT n,
             const f64* A, f64* AF, const INT lda,
             const f64* B, f64* BF, const INT ldb,
             f64* U, const INT ldu,
             f64* V, const INT ldv,
             f64* Q, const INT ldq,
             f64* alpha, f64* beta,
             f64* R, const INT ldr,
             INT* iwork,
             f64* work, const INT lwork,
             f64* rwork,
             f64* result);

/* Additional matrix generators (MATGEN) */

/* Hilbert matrix generator */
void dlahilb(const INT n, const INT nrhs,
             f64* A, const INT lda,
             f64* X, const INT ldx,
             f64* B, const INT ldb,
             f64* work, INT* info);

/* Kronecker product block matrix */
void dlakf2(const INT m, const INT n,
            const f64* A, const INT lda,
            const f64* B, const f64* D, const f64* E,
            f64* Z, const INT ldz);

/* Singular value distribution */
void dlatm7(const INT mode, const f64 cond, const INT irsign,
            const INT idist, f64* d, const INT n, const INT rank,
            INT* info, uint64_t state[static 4]);

/* Generalized Sylvester test matrices */
void dlatm5(const INT prtype, const INT m, const INT n,
            f64* A, const INT lda,
            f64* B, const INT ldb,
            f64* C, const INT ldc,
            f64* D, const INT ldd,
            f64* E, const INT lde,
            f64* F, const INT ldf,
            f64* R, const INT ldr,
            f64* L, const INT ldl,
            const f64 alpha, INT qblcka, INT qblckb);

/* Generalized eigenvalue test matrices */
void dlatm6(const INT type, const INT n,
            f64* A, const INT lda, f64* B,
            f64* X, const INT ldx, f64* Y, const INT ldy,
            const f64 alpha, const f64 beta,
            const f64 wx, const f64 wy,
            f64* S, f64* DIF);

/* Matrix parameter setup for tridiagonal/banded tests */
void dlatb5(const char* path, const INT imat, const INT n,
            char* type, INT* kl, INT* ku, f64* anorm,
            INT* mode, f64* cndnum, char* dist);

/* QL/RQ solve helpers */
void dgeqls(const INT m, const INT n, const INT nrhs,
            f64* A, const INT lda, const f64* tau,
            f64* B, const INT ldb,
            f64* work, const INT lwork, INT* info);

void dgerqs(const INT m, const INT n, const INT nrhs,
            f64* A, const INT lda, const f64* tau,
            f64* B, const INT ldb,
            f64* work, const INT lwork, INT* info);

/* Packed symmetric (SP) verification routines */
void dspt01(const char* uplo, const INT n, const f64* A,
            const f64* AFAC, const INT* ipiv, f64* C, const INT ldc,
            f64* rwork, f64* resid);

/* Packed symmetric multiply (from DSPTRF factorization) */
void dlavsp(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f64* const restrict A,
            const INT* const restrict ipiv,
            f64* const restrict B, const INT ldb, INT* info);

/* Symmetric Rook multiply (from DSYTRF_ROOK factorization) */
void dlavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const INT n, const INT nrhs,
                 const f64* const restrict A, const INT lda,
                 const INT* const restrict ipiv,
                 f64* const restrict B, const INT ldb, INT* info);

/* DGEEVX verification */
void dget23(const INT comp, const char* balanc, const INT jtype,
            const f64 thresh, const INT n,
            f64* A, const INT lda, f64* H,
            f64* wr, f64* wi, f64* wr1, f64* wi1,
            f64* VL, const INT ldvl, f64* VR, const INT ldvr,
            f64* LRE, const INT ldlre,
            f64* rcondv, f64* rcndv1, const f64* rcdvin,
            f64* rconde, f64* rcnde1, const f64* rcdein,
            f64* scale, f64* scale1, f64* result,
            f64* work, const INT lwork, INT* iwork, INT* info);

/* DGEESX verification */
void dget24(const INT comp, const INT jtype, const f64 thresh,
            const INT n, f64* A, const INT lda,
            f64* H, f64* HT,
            f64* wr, f64* wi, f64* wrt, f64* wit,
            f64* wrtmp, f64* witmp,
            f64* VS, const INT ldvs, f64* VS1,
            const f64 rcdein, const f64 rcdvin,
            const INT nslct, const INT* islct,
            f64* result, f64* work, const INT lwork,
            INT* iwork, INT* bwork, INT* info);

/* Generalized Schur decomposition verify */
void dget54(const INT n, const f64* A, const INT lda,
            const f64* B, const INT ldb,
            const f64* S, const INT lds,
            const f64* T, const INT ldt,
            const f64* U, const INT ldu,
            const f64* V, const INT ldv,
            f64* work, f64* result);

/* DLALN2 test (small linear system solver) */
void dget31(f64* rmax, INT* lmax, INT ninfo[2], INT* knt);

/* DLASY2 test (Sylvester-like equation solver) */
void dget32(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

/* DLANV2 test (2x2 standardization) */
void dget33(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

/* DLAEXC test (block swap in Schur form) */
void dget34(f64* rmax, INT* lmax, INT ninfo[2], INT* knt);

/* DTRSYL test (Sylvester equation solver) */
void dget35(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

/* DTREXC test (Schur block reordering) */
void dget36(f64* rmax, INT* lmax, INT ninfo[3], INT* knt);

/* DTRSNA test (eigenvalue/eigenvector condition numbers) */
void dget37(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);

/* DTRSEN test (cluster condition numbers) */
void dget38(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);

/* DLAQTR test (quasi-triangular solve) */
void dget39(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

/* DTGEXC test (generalized Schur block swap) */
void dget40(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

/* Sylvester equation test (DTRSYL + DTRSYL3) */
void dsyl01(const f64 thresh, INT* nfail, f64* rmax, INT* ninfo, INT* knt);

/* Error exit testing infrastructure */
extern INT    xerbla_infot;
extern INT    xerbla_nout;
extern INT    xerbla_ok;
extern INT    xerbla_lerr;
extern char   xerbla_srnamt[33];
void chkxer(const char* srnamt, INT infot, INT* lerr, INT* ok);
void derrec(INT* ok, INT* nt);

/* SVD singular value verification */
void dsvdct(const INT n, const f64* s, const f64* e, const f64 shift, INT* num);
void dsvdch(const INT n, const f64* s, const f64* e,
            const f64* svd, const f64 tol, INT* info);

/* Triangular banded (TB) verification routines */
void dtbt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f64* AB, const INT ldab,
            const f64* X, const INT ldx,
            const f64* B, const INT ldb,
            f64* work, f64* resid);

void dtbt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f64* AB, const INT ldab,
            const f64 scale, const f64* cnorm, const f64 tscal,
            const f64* X, const INT ldx,
            const f64* B, const INT ldb,
            f64* work, f64* resid);

void dtbt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f64* AB, const INT ldab,
            const f64* B, const INT ldb,
            const f64* X, const INT ldx,
            const f64* XACT, const INT ldxact,
            const f64* ferr, const f64* berr,
            f64* reslts);

void dtbt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const INT n, const INT kd,
            const f64* AB, const INT ldab, f64* work, f64* rat);

/* Triangular banded matrix generation */
void dlattb(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, const INT kd,
            f64* AB, const INT ldab, f64* B,
            f64* work, INT* info, uint64_t state[static 4]);

/* Symmetric verification routines (BK variants) */
void dsyt01_3(const char* uplo, const INT n,
              const f64* const restrict A, const INT lda,
              f64* const restrict AFAC, const INT ldafac,
              f64* const restrict E,
              INT* const restrict ipiv,
              f64* const restrict C, const INT ldc,
              f64* const restrict rwork, f64* resid);

void dsyt01_aa(const char* uplo, const INT n,
               const f64* const restrict A, const INT lda,
               const f64* const restrict AFAC, const INT ldafac,
               const INT* const restrict ipiv,
               f64* const restrict C, const INT ldc,
               f64* const restrict rwork, f64* resid);

void dsyt01_rook(const char* uplo, const INT n,
                 const f64* const restrict A, const INT lda,
                 const f64* const restrict AFAC, const INT ldafac,
                 const INT* const restrict ipiv,
                 f64* const restrict C, const INT ldc,
                 f64* const restrict rwork, f64* resid);

/* TSQR verification */
void dtsqr01(const char* tssw, const INT m, const INT n, const INT mb,
             const INT nb, f64* result);

/* CS decomposition verification */
void dcsdts(const INT m, const INT p, const INT q,
            const f64* X, f64* XF, const INT ldx,
            f64* U1, const INT ldu1,
            f64* U2, const INT ldu2,
            f64* V1T, const INT ldv1t,
            f64* V2T, const INT ldv2t,
            f64* theta, INT* iwork,
            f64* work, const INT lwork,
            f64* rwork, f64* result);

#endif /* VERIFY_H */
