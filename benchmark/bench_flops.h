/**
 * @file bench_flops.h
 * @brief LAWN 41 FLOP formulas for LAPACK routines (real arithmetic).
 *
 * Derived from LAPACK++ (include/lapack/flops.hh).
 * Adapted from C++ to C (static inline, char parameters).
 *
 * Each routine has separate fmuls/fadds functions and a convenience
 * *_flops() function returning total FLOPs for real arithmetic
 * (fmuls + fadds).  For complex arithmetic, multiply fmuls by 6
 * and fadds by 2.
 */

// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BENCH_FLOPS_H
#define BENCH_FLOPS_H

/* ================================================================
 * LU factorization (getrf)
 * ================================================================ */

static inline double fmuls_getrf(double m, double n)
{
    return (m >= n)
        ? (0.5*m*n*n - 1./6*n*n*n + 0.5*m*n - 0.5*n*n + 2./3*n)
        : (0.5*n*m*m - 1./6*m*m*m + 0.5*n*m - 0.5*m*m + 2./3*m);
}

static inline double fadds_getrf(double m, double n)
{
    return (m >= n)
        ? (0.5*m*n*n - 1./6*n*n*n - 0.5*m*n + 1./6*n)
        : (0.5*n*m*m - 1./6*m*m*m - 0.5*n*m + 1./6*m);
}

static inline double getrf_flops(double m, double n)
    { return fmuls_getrf(m, n) + fadds_getrf(m, n); }

/* ================================================================
 * LU inversion (getri)
 * ================================================================ */

static inline double fmuls_getri(double n)
    { return 2./3*n*n*n + 0.5*n*n + 5./6*n; }

static inline double fadds_getri(double n)
    { return 2./3*n*n*n - 1.5*n*n + 5./6*n; }

static inline double getri_flops(double n)
    { return fmuls_getri(n) + fadds_getri(n); }

/* ================================================================
 * LU solve (getrs)
 * ================================================================ */

static inline double fmuls_getrs(double n, double nrhs)
    { return nrhs*n*n; }

static inline double fadds_getrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

static inline double getrs_flops(double n, double nrhs)
    { return fmuls_getrs(n, nrhs) + fadds_getrs(n, nrhs); }

/* ================================================================
 * Cholesky factorization (potrf)
 * ================================================================ */

static inline double fmuls_potrf(double n)
    { return 1./6*n*n*n + 0.5*n*n + 1./3*n; }

static inline double fadds_potrf(double n)
    { return 1./6*n*n*n - 1./6*n; }

static inline double potrf_flops(double n)
    { return fmuls_potrf(n) + fadds_potrf(n); }

/* ================================================================
 * Cholesky inversion (potri)
 * ================================================================ */

static inline double fmuls_potri(double n)
    { return 1./3*n*n*n + n*n + 2./3*n; }

static inline double fadds_potri(double n)
    { return 1./3*n*n*n - 0.5*n*n + 1./6*n; }

static inline double potri_flops(double n)
    { return fmuls_potri(n) + fadds_potri(n); }

/* ================================================================
 * Cholesky solve (potrs)
 * ================================================================ */

static inline double fmuls_potrs(double n, double nrhs)
    { return nrhs*n*(n + 1); }

static inline double fadds_potrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

static inline double potrs_flops(double n, double nrhs)
    { return fmuls_potrs(n, nrhs) + fadds_potrs(n, nrhs); }

/* ================================================================
 * Symmetric indefinite factorization (sytrf)
 * ================================================================ */

static inline double fmuls_sytrf(double n)
    { return 1./6*n*n*n + 0.5*n*n + 10./3*n; }

static inline double fadds_sytrf(double n)
    { return 1./6*n*n*n - 1./6*n; }

static inline double sytrf_flops(double n)
    { return fmuls_sytrf(n) + fadds_sytrf(n); }

/* ================================================================
 * QR factorization (geqrf)
 * ================================================================ */

static inline double fmuls_geqrf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3*n*n*n +   m*n + 0.5*n*n + 23./6*n)
        : (n*m*m - 1./3*m*m*m + 2*n*m - 0.5*m*m + 23./6*m);
}

static inline double fadds_geqrf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3*n*n*n + 0.5*n*n       + 5./6*n)
        : (n*m*m - 1./3*m*m*m + n*m - 0.5*m*m + 5./6*m);
}

static inline double geqrf_flops(double m, double n)
    { return fmuls_geqrf(m, n) + fadds_geqrf(m, n); }

/* ================================================================
 * QL factorization (geqlf) — same as geqrf
 * ================================================================ */

static inline double geqlf_flops(double m, double n)
    { return geqrf_flops(m, n); }

/* ================================================================
 * RQ factorization (gerqf)
 * ================================================================ */

static inline double fmuls_gerqf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3*n*n*n +   m*n + 0.5*n*n + 29./6*n)
        : (n*m*m - 1./3*m*m*m + 2*n*m - 0.5*m*m + 29./6*m);
}

static inline double fadds_gerqf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3*n*n*n + m*n - 0.5*n*n + 5./6*n)
        : (n*m*m - 1./3*m*m*m + 0.5*m*m       + 5./6*m);
}

static inline double gerqf_flops(double m, double n)
    { return fmuls_gerqf(m, n) + fadds_gerqf(m, n); }

/* ================================================================
 * LQ factorization (gelqf) — same as gerqf
 * ================================================================ */

static inline double gelqf_flops(double m, double n)
    { return gerqf_flops(m, n); }

/* ================================================================
 * Generate Q from QR (ungqr / orgqr)
 * ================================================================ */

static inline double fmuls_ungqr(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2./3*k*k*k + 2*n*k - k*k - 5./3*k; }

static inline double fadds_ungqr(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2./3*k*k*k + n*k - m*k + 1./3*k; }

static inline double ungqr_flops(double m, double n, double k)
    { return fmuls_ungqr(m, n, k) + fadds_ungqr(m, n, k); }

/* ================================================================
 * Multiply by Q (unmqr / ormqr)
 * side: 'L' = left, 'R' = right
 * ================================================================ */

static inline double fmuls_unmqr(char side, double m, double n, double k)
{
    return (side == 'L' || side == 'l')
        ? (2*n*m*k - n*k*k + 2*n*k)
        : (2*n*m*k - m*k*k + m*k + n*k - 0.5*k*k + 0.5*k);
}

static inline double fadds_unmqr(char side, double m, double n, double k)
{
    return (side == 'L' || side == 'l')
        ? (2*n*m*k - n*k*k + n*k)
        : (2*n*m*k - m*k*k + m*k);
}

static inline double unmqr_flops(char side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k) + fadds_unmqr(side, m, n, k); }

/* ================================================================
 * Hessenberg reduction (gehrd)
 * ================================================================ */

static inline double fmuls_gehrd(double n)
    { return 5./3*n*n*n + 0.5*n*n - 7./6*n; }

static inline double fadds_gehrd(double n)
    { return 5./3*n*n*n - n*n - 2./3*n; }

static inline double gehrd_flops(double n)
    { return fmuls_gehrd(n) + fadds_gehrd(n); }

/* ================================================================
 * Tridiagonal reduction (sytrd / hetrd)
 * ================================================================ */

static inline double fmuls_sytrd(double n)
    { return 2./3*n*n*n + 2.5*n*n - 1./6*n; }

static inline double fadds_sytrd(double n)
    { return 2./3*n*n*n + n*n - 8./3*n; }

static inline double sytrd_flops(double n)
    { return fmuls_sytrd(n) + fadds_sytrd(n); }

/* ================================================================
 * Bidiagonal reduction (gebrd)
 * ================================================================ */

static inline double fmuls_gebrd(double m, double n)
{
    return (m >= n)
        ? (2*m*n*n - 2./3*n*n*n + 2*n*n + 20./3*n)
        : (2*n*m*m - 2./3*m*m*m + 2*m*m + 20./3*m);
}

static inline double fadds_gebrd(double m, double n)
{
    return (m >= n)
        ? (2*m*n*n - 2./3*n*n*n + n*n - m*n + 5./3*n)
        : (2*n*m*m - 2./3*m*m*m + m*m - n*m + 5./3*m);
}

static inline double gebrd_flops(double m, double n)
    { return fmuls_gebrd(m, n) + fadds_gebrd(m, n); }

/* ================================================================
 * Triangular inversion (trtri)
 * ================================================================ */

static inline double fmuls_trtri(double n)
    { return 1./6*n*n*n + 0.5*n*n + 1./3*n; }

static inline double fadds_trtri(double n)
    { return 1./6*n*n*n - 0.5*n*n + 1./3*n; }

static inline double trtri_flops(double n)
    { return fmuls_trtri(n) + fadds_trtri(n); }

#endif /* BENCH_FLOPS_H */
