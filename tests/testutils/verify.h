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

/* General (DGE) verification routines */
void dget01(const int m, const int n, const double * const restrict A, const int lda,
            double * const restrict AFAC, const int ldafac, const int * const restrict ipiv,
            double * const restrict rwork, double *resid);

void dget02(const char* trans, const int m, const int n, const int nrhs,
            const double * const restrict A, const int lda, const double * const restrict X,
            const int ldx, double * const restrict B, const int ldb,
            double * const restrict rwork, double *resid);

void dget03(const int n, const double * const restrict A, const int lda,
            const double * const restrict AINV, const int ldainv, double * const restrict work,
            const int ldwork, double * const restrict rwork, double *rcond, double *resid);

void dget04(const int n, const int nrhs, const double * const restrict X, const int ldx,
            const double * const restrict XACT, const int ldxact, const double rcond,
            double *resid);

double dget06(const double rcond, const double rcondc);

void dget07(const char* trans, const int n, const int nrhs,
            const double * const restrict A, const int lda,
            const double * const restrict B, const int ldb,
            const double * const restrict X, const int ldx,
            const double * const restrict XACT, const int ldxact,
            const double * const restrict ferr, const bool chkferr,
            const double * const restrict berr, double * const restrict reslts);

void dget08(const char* trans, const int m, const int n, const int nrhs,
            const double* A, const int lda, const double* X, const int ldx,
            double* B, const int ldb, double* rwork, double* resid);

/* Banded (GB) verification routines */
void dgbt01(int m, int n, int kl, int ku,
            const double* A, int lda,
            const double* AFAC, int ldafac,
            const int* ipiv,
            double* work,
            double* resid);

void dgbt02(const char* trans, int m, int n, int kl, int ku, int nrhs,
            const double* A, int lda,
            const double* X, int ldx,
            double* B, int ldb,
            double* rwork,
            double* resid);

void dgbt05(const char* trans, int n, int kl, int ku, int nrhs,
            const double* AB, int ldab,
            const double* B, int ldb,
            const double* X, int ldx,
            const double* XACT, int ldxact,
            const double* FERR,
            const double* BERR,
            double* reslts);

/* Positive definite banded (PB) verification routines */
void dpbt01(const char* uplo, const int n, const int kd,
            const double* A, const int lda,
            double* AFAC, const int ldafac,
            double* rwork, double* resid);

void dpbt02(const char* uplo, const int n, const int kd, const int nrhs,
            const double* A, const int lda,
            const double* X, const int ldx,
            double* B, const int ldb,
            double* rwork, double* resid);

void dpbt05(const char* uplo, const int n, const int kd, const int nrhs,
            const double* AB, const int ldab,
            const double* B, const int ldb,
            const double* X, const int ldx,
            const double* XACT, const int ldxact,
            const double* ferr, const double* berr,
            double* reslts);

/* Tridiagonal (DGT) verification routines */
void dgtt01(const int n, const double * const restrict DL, const double * const restrict D,
            const double * const restrict DU, const double * const restrict DLF,
            const double * const restrict DF, const double * const restrict DUF,
            const double * const restrict DU2, const int * const restrict ipiv,
            double * const restrict work, const int ldwork, double *resid);

void dgtt02(const char* trans, const int n, const int nrhs,
            const double * const restrict DL, const double * const restrict D,
            const double * const restrict DU, const double * const restrict X, const int ldx,
            double * const restrict B, const int ldb, double *resid);

void dgtt05(const char* trans, const int n, const int nrhs,
            const double * const restrict DL, const double * const restrict D,
            const double * const restrict DU, const double * const restrict B, const int ldb,
            const double * const restrict X, const int ldx,
            const double * const restrict XACT, const int ldxact,
            const double * const restrict ferr, const double * const restrict berr,
            double * const restrict reslts);

/* Matrix generation routines */
void dlatb4(const char *path, const int imat, const int m, const int n,
            char *type, int *kl, int *ku, double *anorm, int *mode,
            double *cndnum, char *dist);

void dlatms(const int m, const int n, const char* dist, uint64_t seed,
            const char* sym, double *d, const int mode, const double cond,
            const double dmax, const int kl, const int ku, const char* pack,
            double *A, const int lda, double *work, int *info);

void dlatmt(const int m, const int n, const char* dist,
            const char* sym, double* d, const int mode,
            const double cond, const double dmax, const int rank,
            const int kl, const int ku, const char* pack,
            double* A, const int lda, double* work, int* info);

void dlatm1(const int mode, const double cond, const int irsign,
            const int idist, double* d, const int n, int* info);

void dlagge(const int m, const int n, const int kl, const int ku,
            const double* d, double* A, const int lda, uint64_t seed,
            double* work, int* info);

void dlagsy(const int n, const int k, const double* d, double* A,
            const int lda, double* work, int* info);

void dlarot(const int lrows, const int lleft, const int lright,
            const int nl, const double c, const double s,
            double* A, const int lda, double* xleft, double* xright);

void dlarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const int m, const int n, const int kl,
            const int ku, const int nrhs, const double* A, const int lda,
            double* X, const int ldx, double* B, const int ldb,
            uint64_t seed, int* info);

void dlaord(const char* job, const int n, double* X, const int incx);

/* Symmetric (SY) verification routines */
void dlavsy(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const double* const restrict A, const int lda,
            const int* const restrict ipiv, double* const restrict B, const int ldb, int* info);

void dsyt01(const char* uplo, const int n, const double* const restrict A, const int lda,
            const double* const restrict AFAC, const int ldafac, const int* const restrict ipiv,
            double* const restrict C, const int ldc, double* const restrict rwork, double* resid);

/* Cholesky (PO) verification routines */
void dpot01(const char* uplo, const int n, const double* const restrict A, const int lda,
            double* const restrict AFAC, const int ldafac, double* const restrict rwork,
            double* resid);

void dpot02(const char* uplo, const int n, const int nrhs,
            const double* const restrict A, const int lda,
            const double* const restrict X, const int ldx,
            double* const restrict B, const int ldb,
            double* const restrict rwork, double* resid);

void dpot03(const char* uplo, const int n, const double* const restrict A, const int lda,
            double* const restrict AINV, const int ldainv, double* const restrict work,
            const int ldwork, double* const restrict rwork, double* rcond, double* resid);

void dpot05(const char* uplo, const int n, const int nrhs,
            const double* const restrict A, const int lda,
            const double* const restrict B, const int ldb,
            const double* const restrict X, const int ldx,
            const double* const restrict XACT, const int ldxact,
            const double* const restrict ferr, const double* const restrict berr,
            double* const restrict reslts);

void dpot06(const char* uplo, const int n, const int nrhs,
            const double* A, const int lda, const double* X, const int ldx,
            double* B, const int ldb, double* rwork, double* resid);

/* Positive semidefinite pivoted Cholesky (PS) verification routines */
int dgennd(const int m, const int n, const double* const restrict A, const int lda);

void dpst01(const char* uplo, const int n,
            const double* const restrict A, const int lda,
            double* const restrict AFAC, const int ldafac,
            double* const restrict PERM, const int ldperm,
            const int* const restrict piv,
            double* const restrict rwork, double* resid, const int rank);

/* Packed Cholesky (PP) verification routines */
void dppt01(const char* uplo, const int n, const double* const restrict A,
            double* const restrict AFAC, double* const restrict rwork,
            double* resid);

void dppt02(const char* uplo, const int n, const int nrhs,
            const double* const restrict A,
            const double* const restrict X, const int ldx,
            double* const restrict B, const int ldb,
            double* const restrict rwork, double* resid);

void dppt03(const char* uplo, const int n, const double* const restrict A,
            const double* const restrict AINV, double* const restrict work,
            const int ldwork, double* const restrict rwork,
            double* rcond, double* resid);

void dppt05(const char* uplo, const int n, const int nrhs,
            const double* const restrict AP,
            const double* const restrict B, const int ldb,
            const double* const restrict X, const int ldx,
            const double* const restrict XACT, const int ldxact,
            const double* const restrict FERR, const double* const restrict BERR,
            double* const restrict reslts);

/* QR verification routines */
void dqrt01(const int m, const int n, const double * const restrict A,
            double * const restrict AF, double * const restrict Q, double * const restrict R,
            const int lda, double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dqrt02(const int m, const int n, const int k, const double * const restrict A,
            const double * const restrict AF, double * const restrict Q, double * const restrict R,
            const int lda, const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dqrt03(const int m, const int n, const int k, const double * const restrict AF,
            double * const restrict C, double * const restrict CC, double * const restrict Q,
            const int lda, const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dqrt04(const int m, const int n, const int nb, double * restrict result);

void dqrt05(const int m, const int n, const int l, const int nb, double * restrict result);

void dqrt01p(const int m, const int n,
             const double * const restrict A,
             double * const restrict AF,
             double * const restrict Q,
             double * const restrict R,
             const int lda,
             double * const restrict tau,
             double * const restrict work, const int lwork,
             double * const restrict rwork,
             double * restrict result);

/* LQ verification routines */
void dlqt01(const int m, const int n, const double * const restrict A,
            double * const restrict AF, double * const restrict Q, double * const restrict L,
            const int lda, double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dlqt02(const int m, const int n, const int k, const double * const restrict A,
            const double * const restrict AF, double * const restrict Q, double * const restrict L,
            const int lda, const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dlqt03(const int m, const int n, const int k, const double * const restrict AF,
            double * const restrict C, double * const restrict CC, double * const restrict Q,
            const int lda, const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dlqt04(const int m, const int n, const int nb, double * restrict result);

void dlqt05(const int m, const int n, const int l, const int nb, double * restrict result);

/* Householder reconstruction verification routines */
void dorhr_col01(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, double * restrict result);

void dorhr_col02(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, double * restrict result);

/* QL verification routines */
void dqlt01(const int m, const int n, const double * const restrict A,
            double * const restrict AF, double * const restrict Q, double * const restrict L,
            const int lda, double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dqlt02(const int m, const int n, const int k,
            const double * const restrict A, const double * const restrict AF,
            double * const restrict Q, double * const restrict L, const int lda,
            const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void dqlt03(const int m, const int n, const int k, const double * const restrict AF,
            double * const restrict C, double * const restrict CC, double * const restrict Q,
            const int lda, const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

/* RQ verification routines */
void drqt01(const int m, const int n, const double * const restrict A,
            double * const restrict AF, double * const restrict Q, double * const restrict R,
            const int lda, double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void drqt02(const int m, const int n, const int k,
            const double * const restrict A, const double * const restrict AF,
            double * const restrict Q, double * const restrict R, const int lda,
            const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

void drqt03(const int m, const int n, const int k, const double * const restrict AF,
            double * const restrict C, double * const restrict CC, double * const restrict Q,
            const int lda, const double * const restrict tau, double * const restrict work,
            const int lwork, double * const restrict rwork, double * restrict result);

/* QR with pivoting verification routines */
double dqpt01(const int m, const int n, const int k, const double* A, const double* AF,
              const int lda, const double* tau, const int* jpvt,
              double* work, const int lwork);

double dqrt11(const int m, const int k, const double* A, const int lda,
              const double* tau, double* work, const int lwork);

double dqrt12(const int m, const int n, const double* A, const int lda,
              const double* S, double* work, const int lwork);

/* RZ factorization verification routines */
double drzt01(const int m, const int n, const double* A, const double* AF,
              const int lda, const double* tau, double* work, const int lwork);

double drzt02(const int m, const int n, const double* AF, const int lda,
              const double* tau, double* work, const int lwork);

/* Least squares verification routines */
void dqrt13(const int scale, const int m, const int n,
            double* A, const int lda, double* norma);

double dqrt14(const char* trans, const int m, const int n, const int nrhs,
              const double* A, const int lda, const double* X, const int ldx,
              double* work, const int lwork);

void dqrt15(const int scale, const int rksel,
            const int m, const int n, const int nrhs,
            double* A, const int lda, double* B, const int ldb,
            double* S, int* rank, double* norma, double* normb,
            double* work, const int lwork);

void dqrt16(const char* trans, const int m, const int n, const int nrhs,
            const double* A, const int lda,
            const double* X, const int ldx,
            double* B, const int ldb,
            double* rwork, double* resid);

double dqrt17(const char* trans, const int iresid,
              const int m, const int n, const int nrhs,
              const double* A, const int lda,
              const double* X, const int ldx,
              const double* B, const int ldb,
              double* C,
              double* work, const int lwork);

/* Orthogonal random matrix generator */
void dlaror(const char* side, const char* init,
            const int m, const int n,
            double* A, const int lda,
            const uint64_t seed,
            double* X, int* info);

void dlarge(const int n, double* A, const int lda, uint64_t* seed,
            double* work, int* info);

void dlarnv_rng(const int idist, uint64_t* seed, const int n, double* x);

double dlaran_rng(uint64_t* seed);

/* Triangular verification routines */
void dtrt01(const char* uplo, const char* diag, const int n,
            const double* A, const int lda, double* AINV, const int ldainv,
            double* rcond, double* work, double* resid);

void dtrt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const double* A, const int lda,
            const double* X, const int ldx, const double* B, const int ldb,
            double* work, double* resid);

void dtrt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const double* A, const int lda,
            const double scale, const double* cnorm, const double tscal,
            const double* X, const int ldx, const double* B, const int ldb,
            double* work, double* resid);

void dtrt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const double* A, const int lda,
            const double* B, const int ldb, const double* X, const int ldx,
            const double* XACT, const int ldxact,
            const double* ferr, const double* berr, double* reslts);

void dtrt06(const double rcond, const double rcondc,
            const char* uplo, const char* diag, const int n,
            const double* A, const int lda, double* work, double* rat);

/* Triangular matrix generation */
void dlattr(const int imat, const char* uplo, const char* trans, char* diag,
            uint64_t* seed, const int n, double* A, const int lda,
            double* B, double* work, int* info);

/* Triangular packed (TP) verification routines */
void dtpt01(const char* uplo, const char* diag, const int n,
            const double* AP, double* AINVP,
            double* rcond, double* work, double* resid);

void dtpt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const double* AP, const double* X, const int ldx,
            const double* B, const int ldb,
            double* work, double* resid);

void dtpt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const double* AP, const double scale, const double* cnorm,
            const double tscal, const double* X, const int ldx,
            const double* B, const int ldb,
            double* work, double* resid);

void dtpt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const double* AP, const double* B, const int ldb,
            const double* X, const int ldx,
            const double* XACT, const int ldxact,
            const double* ferr, const double* berr,
            double* reslts);

void dtpt06(const double rcond, const double rcondc,
            const char* uplo, const char* diag, const int n,
            const double* AP, double* work, double* rat);

/* Triangular packed matrix generation */
void dlattp(const int imat, const char* uplo, const char* trans, char* diag,
            uint64_t* seed, const int n, double* AP, double* B, double* work,
            int* info);

/* Eigenvalue verification routines */
void dstech(const int n, const double* const restrict A, const double* const restrict B,
            const double* const restrict eig, const double tol,
            double* const restrict work, int* info);

void dstt21(const int n, const int kband, const double* const restrict AD,
            const double* const restrict AE, const double* const restrict SD,
            const double* const restrict SE, const double* const restrict U, const int ldu,
            double* const restrict work, double* restrict result);

void dsyt21(const int itype, const char* uplo, const int n, const int kband,
            const double* const restrict A, const int lda,
            const double* const restrict D, const double* const restrict E,
            const double* const restrict U, const int ldu,
            double* restrict V, const int ldv,
            const double* const restrict tau,
            double* const restrict work, double* restrict result);

void dsyt22(const int itype, const char* uplo, const int n, const int m,
            const int kband, const double* const restrict A, const int lda,
            const double* const restrict D, const double* const restrict E,
            const double* const restrict U, const int ldu,
            const double* const restrict V, const int ldv,
            const double* const restrict tau,
            double* const restrict work, double* restrict result);

double dsxt1(const int ijob, const double* const restrict D1, const int n1,
             const double* const restrict D2, const int n2,
             const double abstol, const double ulp, const double unfl);

/* Positive definite tridiagonal (PT) verification routines */
void dptt01(const int n, const double* const restrict D, const double* const restrict E,
            const double* const restrict DF, const double* const restrict EF,
            double* const restrict work, double* resid);

void dptt02(const int n, const int nrhs, const double* const restrict D,
            const double* const restrict E, const double* const restrict X, const int ldx,
            double* const restrict B, const int ldb, double* resid);

void dptt05(const int n, const int nrhs, const double* const restrict D,
            const double* const restrict E, const double* const restrict B, const int ldb,
            const double* const restrict X, const int ldx,
            const double* const restrict XACT, const int ldxact,
            const double* const restrict FERR, const double* const restrict BERR,
            double* const restrict reslts);

void dlaptm(const int n, const int nrhs, const double alpha,
            const double* const restrict D, const double* const restrict E,
            const double* const restrict X, const int ldx, const double beta,
            double* const restrict B, const int ldb);

/* SVD verification routines */
void dbdt01(const int m, const int n, const int kd,
            const double* const restrict A, const int lda,
            const double* const restrict Q, const int ldq,
            const double* const restrict D, const double* const restrict E,
            const double* const restrict PT, const int ldpt,
            double* const restrict work, double* resid);

void dort03(const char* rc, const int mu, const int mv, const int n,
            const int k, const double* const restrict U, const int ldu,
            const double* const restrict V, const int ldv,
            double* const restrict work, const int lwork,
            double* result, int* info);

void dbdt05(const int m, const int n, const double* const restrict A, const int lda,
            const double* const restrict S, const int ns,
            const double* const restrict U, const int ldu,
            const double* const restrict VT, const int ldvt,
            double* const restrict work, double* resid);

/* Non-symmetric eigenvalue verification routines */
void dort01(const char* rowcol, const int m, const int n,
            const double* U, const int ldu,
            double* work, const int lwork, double* resid);

void dhst01(const int n, const int ilo, const int ihi,
            const double* A, const int lda,
            const double* H, const int ldh,
            const double* Q, const int ldq,
            double* work, const int lwork, double* result);

void dget22(const char* transa, const char* transe, const char* transw,
            const int n, const double* A, const int lda,
            const double* E, const int lde,
            const double* wr, const double* wi,
            double* work, double* result);

void dlatm4(const int itype, const int n, const int nz1, const int nz2,
            const int isign, const double amagn, const double rcond,
            const double triang, const int idist,
            double* A, const int lda);

void dlatme(const int n, const char* dist, uint64_t* seed, double* D,
            const int mode, const double cond, const double dmax,
            const char* ei, const char* rsign, const char* upper,
            const char* sim, double* DS, const int modes, const double conds,
            const int kl, const int ku, const double anorm,
            double* A, const int lda, double* work, int* info);

double dlatm2(const int m, const int n, const int i, const int j,
              const int kl, const int ku, const int idist,
              const double* d, const int igrade,
              const double* dl, const double* dr,
              const int ipvtng, const int* iwork, const double sparse);

double dlatm3(const int m, const int n, const int i, const int j,
              int* isub, int* jsub, const int kl, const int ku,
              const int idist, const double* d, const int igrade,
              const double* dl, const double* dr,
              const int ipvtng, const int* iwork, const double sparse);

void dlatmr(const int m, const int n, const char* dist, const char* sym,
            double* d, const int mode, const double cond, const double dmax,
            const char* rsign, const char* grade, double* dl,
            const int model, const double condl, double* dr,
            const int moder, const double condr, const char* pivtng,
            const int* ipivot, const int kl, const int ku,
            const double sparse, const double anorm, const char* pack,
            double* A, const int lda, int* iwork, int* info);

/* GSVD verification routines */
void dgsvts3(const int m, const int p, const int n,
             const double* A, double* AF, const int lda,
             const double* B, double* BF, const int ldb,
             double* U, const int ldu,
             double* V, const int ldv,
             double* Q, const int ldq,
             double* alpha, double* beta,
             double* R, const int ldr,
             int* iwork,
             double* work, const int lwork,
             double* rwork,
             double* result);

/* Additional matrix generators (MATGEN) */

/* Hilbert matrix generator */
void dlahilb(const int n, const int nrhs,
             double* A, const int lda,
             double* X, const int ldx,
             double* B, const int ldb,
             double* work, int* info);

/* Kronecker product block matrix */
void dlakf2(const int m, const int n,
            const double* A, const int lda,
            const double* B, const double* D, const double* E,
            double* Z, const int ldz);

/* Singular value distribution */
void dlatm7(const int mode, const double cond, const int irsign,
            const int idist, double* d, const int n, const int rank,
            int* info);

/* Generalized Sylvester test matrices */
void dlatm5(const int prtype, const int m, const int n,
            double* A, const int lda,
            double* B, const int ldb,
            double* C, const int ldc,
            double* D, const int ldd,
            double* E, const int lde,
            double* F, const int ldf,
            double* R, const int ldr,
            double* L, const int ldl,
            const double alpha, int qblcka, int qblckb);

/* Generalized eigenvalue test matrices */
void dlatm6(const int type, const int n,
            double* A, const int lda, double* B,
            double* X, const int ldx, double* Y, const int ldy,
            const double alpha, const double beta,
            const double wx, const double wy,
            double* S, double* DIF);

/* Matrix parameter setup for tridiagonal/banded tests */
void dlatb5(const char* path, const int imat, const int n,
            char* type, int* kl, int* ku, double* anorm,
            int* mode, double* cndnum, char* dist);

/* QL/RQ solve helpers */
void dgeqls(const int m, const int n, const int nrhs,
            double* A, const int lda, const double* tau,
            double* B, const int ldb,
            double* work, const int lwork, int* info);

void dgerqs(const int m, const int n, const int nrhs,
            double* A, const int lda, const double* tau,
            double* B, const int ldb,
            double* work, const int lwork, int* info);

/* Packed symmetric (SP) verification routines */
void dspt01(const char* uplo, const int n, const double* A,
            const double* AFAC, const int* ipiv, double* C, const int ldc,
            double* rwork, double* resid);

/* Packed symmetric multiply (from DSPTRF factorization) */
void dlavsp(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const double* const restrict A,
            const int* const restrict ipiv,
            double* const restrict B, const int ldb, int* info);

/* Symmetric Rook multiply (from DSYTRF_ROOK factorization) */
void dlavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const int n, const int nrhs,
                 const double* const restrict A, const int lda,
                 const int* const restrict ipiv,
                 double* const restrict B, const int ldb, int* info);

/* Triangular banded matrix generation */
void dlattb(const int imat, const char* uplo, const char* trans, char* diag,
            uint64_t* seed, const int n, const int kd,
            double* AB, const int ldab, double* B,
            double* work, int* info);

#endif /* VERIFY_H */
