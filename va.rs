//! Kona K3 scalar arithmetic verbs translated from C to Rust.
//!
//! Original C source: https://github.com/kevinlawler/kona/blob/master/src/va.c
//! Author: Kevin Lawler
//!
//! This file contains the direct Rust translation of:
//!   times()  → _mul:  element-wise multiplication (K x * y)
//!   _dot()   → dot:   dot product / fused multiply-accumulate (+/ x * y)
//!   plus()   → plus:  element-wise addition (K x + y)
//!   minus()  → minus: element-wise subtraction (K x - y)
//!
//! Translation approach:
//!   - C macros (SCALAR_INIT, SCALAR_OP_CASE, etc.) are inlined as Rust logic
//!   - C type dispatch (if/else chains on ABS(at), ABS(bt)) preserved exactly
//!   - C void* pointer casts → Rust enum matching
//!   - C flexible array member access → Rust Vec indexing

use crate::k::{K, KData};
use crate::parallel;
// ===================================================================
// SCALAR_INIT equivalent
//
// C original (from va.c, inlined macro):
//   I at=a->t, an=a->n, bt=b->t, bn=b->n;
//   I type = MAX(ABS(at),ABS(bt));
//   P(at <= 0 && bt <= 0 && an != bn, LE)
//   P(type > 2, TE);
//   I zt=type;
//   if(MIN(at,bt) < 1) zt=-zt;
//   if(!at || !bt) zt=0;
//   if(1==zt*zt)zt*=2;
//   I zn=at>0?bn:an;
//
// This computes the output type (zt) and length (zn) for a dyadic
// scalar operation, handling type promotion and scalar extension.
// ===================================================================

struct ScalarInit {
    at: i64, // a->t
    an: i64, // a->n
    bt: i64, // b->t
    bn: i64, // b->n
    zt: i64, // output type
    zn: i64, // output length
}

fn scalar_init(a: &K, b: &K) -> ScalarInit {
    let at = a.t;
    let an = a.n;
    let bt = b.t;
    let bn = b.n;

    // 1. Initial Type
    let typ = at.abs().max(bt.abs());

    // 2. Length Check & Logic (zn)
    // Only check strict atom mismatch here. List/Atom broadcast handled below.
    if at < 0 && bt < 0 && an != bn {
        panic!("length error: atoms mismatch");
    }
    // List mismatch strict check (if both are lists)
    if at == 0 && bt == 0 && an != bn {
        panic!("length error: lists mismatch (matrices must have same rows)");
    }
    // Broadcasting Rules: if lengths differ, one must be 1.
    // If both are > 1 and differ -> Error
    if an != bn && an != 1 && bn != 1 {
        panic!("length error: incompatible shapes an={}, bn={}", an, bn);
    }
    // Output length: max of inputs
    let zn = std::cmp::max(an, bn);

    // 3. Output Type Logic (Depends on zn)
    let mut zt = typ;

    // If result is vector length (>1), force positive type
    if zn > 1 {
        zt = zt.abs();
    } else {
        // If length 1:
        // Atom + Atom -> Atom (negative)
        // Vector + Atom -> Vector (positive)
        // Vector + Vector -> Vector (positive)
        if at < 0 && bt < 0 {
            zt = -zt.abs();
        } else {
            zt = zt.abs(); // Vector of length 1
        }
    }

    // List override (t=0)
    if at == 0 || bt == 0 {
        zt = 0;
    }

    // Int op Int promotion (K3 heuristic for safety)
    // if zt.abs() == 1 {
    //     // Promote to float (2/-2)
    //     zt = if zt < 0 { -2 } else { 2 };
    // }

    assert!(zt.abs() <= 2, "type error produced invalid type");

    ScalarInit {
        at,
        an,
        bt,
        bn,
        zt,
        zn,
    }
}

// ===================================================================
// Element-wise scalar operation helper
//
// This replaces the C SCALAR_OP_CASE macro.
// C original:
//   if(2==ABS(at)&&2==ABS(bt)){ SCALAR_OP_CASE(OP,kF(z),kF(a),kF(b)) }
//   else if(2==ABS(at)&&1==ABS(bt)){ SCALAR_OP_CASE(OP_FI,kF(z),kF(a),kI(b)) }
//   ...
//
// SCALAR_OP_CASE expands to a loop:
//   if(an==bn)      DO(zn, z[i] = OP(a[i], b[i]))
//   else if(an==1)  DO(zn, z[i] = OP(a[0], b[i]))
//   else            DO(zn, z[i] = OP(a[i], b[0]))
// ===================================================================

/// Apply a dyadic scalar operation element-wise, with scalar extension.
///
/// This is the Rust equivalent of Kona's SCALAR_OP_CASE macro.
/// Handles three cases:
///   - Both arrays same length: zip and apply
///   - a is scalar: broadcast a[0] across b
///   - b is scalar: broadcast b[0] across a
fn scalar_op_f64<F>(a: &[f64], b: &[f64], an: i64, bn: i64, zn: i64, op: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Copy + Send + Sync,
{
    parallel::binary_f64(a, b, an, bn, zn, op)
}

fn scalar_op_i64<F>(a: &[i64], b: &[i64], an: i64, bn: i64, zn: i64, op: F) -> Vec<i64>
where
    F: Fn(i64, i64) -> i64 + Copy + Send + Sync,
{
    parallel::binary_i64(a, b, an, bn, zn, op)
}

// ===================================================================
// Generic dyadic scalar verb dispatcher
//
// This replaces the entire if/else type-dispatch chain in va.c.
// Every arithmetic verb (plus, minus, times, etc.) follows the same
// pattern: SCALAR_INIT, then dispatch by type pair, create result K.
// ===================================================================

fn dyadic_scalar(
    a: &K,
    b: &K,
    op_ii: fn(i64, i64) -> i64,
    op_ff: fn(f64, f64) -> f64,
    _op_fi: fn(f64, i64) -> f64,
    _op_if: fn(i64, f64) -> f64,
) -> K {
    let s = scalar_init(a, b);

    let abs_at = s.at.abs();
    let abs_bt = s.bt.abs();

    // ---------------------------------------------------------------
    // Type dispatch — direct translation of va.c's if/else chain:
    //
    //   if(2==ABS(at)&&2==ABS(bt))      → float × float
    //   else if(2==ABS(at)&&1==ABS(bt)) → float × int
    //   else if(1==ABS(at)&&2==ABS(bt)) → int × float
    //   else if(1==ABS(at)&&1==ABS(bt)) → int × int
    //   else if(0==at||0==bt)           → general list (recurse)
    // ---------------------------------------------------------------

    if abs_at == 2 && abs_bt == 2 {
        // float × float → float
        let af = a.kf_data();
        let bf = b.kf_data();
        let zf = scalar_op_f64(af, bf, s.an, s.bn, s.zn, op_ff);
        K {
            t: s.zt,
            n: s.zn,
            data: KData::Floats(zf),
        }
    } else if abs_at == 2 && abs_bt == 1 {
        // float × int → float (C: TIMES_FI(x,y) = x * I2F(y))
        let af = a.kf_data();
        let bi = b.ki_data();
        let zf = scalar_op_f64(
            af,
            // C does I2F inline; we pre-convert. Same result.
            &bi.iter().map(|&x| K::i2f(x)).collect::<Vec<f64>>(),
            s.an,
            s.bn,
            s.zn,
            op_ff,
        );
        K {
            t: s.zt,
            n: s.zn,
            data: KData::Floats(zf),
        }
    } else if abs_at == 1 && abs_bt == 2 {
        // int × float → float (C: TIMES_IF(x,y) = I2F(x) * y)
        let ai = a.ki_data();
        let bf = b.kf_data();
        let zf = scalar_op_f64(
            &ai.iter().map(|&x| K::i2f(x)).collect::<Vec<f64>>(),
            bf,
            s.an,
            s.bn,
            s.zn,
            op_ff,
        );
        K {
            t: s.zt,
            n: s.zn,
            data: KData::Floats(zf),
        }
    } else if abs_at == 1 && abs_bt == 1 {
        // int × int → int (C: SCALAR_OP_CASE(TIMES, kI(z), kI(a), kI(b)))
        let ai = a.ki_data();
        let bi = b.ki_data();
        let zi = scalar_op_i64(ai, bi, s.an, s.bn, s.zn, op_ii);
        K {
            t: s.zt,
            n: s.zn,
            data: KData::Ints(zi),
        }
    } else if s.at == 0 || s.bt == 0 {
        // General list recursion
        // If a is list, b is atom: map over a
        match (&a.data, &b.data) {
            (KData::List(la), _) if s.at == 0 && s.bt != 0 => {
                let new_list = la
                    .iter()
                    .map(|item| dyadic_scalar(item, b, op_ii, op_ff, _op_fi, _op_if))
                    .collect();
                K::from_list(new_list)
            }
            (_, KData::List(lb)) if s.at != 0 && s.bt == 0 => {
                let new_list = lb
                    .iter()
                    .map(|item| dyadic_scalar(a, item, op_ii, op_ff, _op_fi, _op_if))
                    .collect();
                K::from_list(new_list)
            }
            (KData::List(la), KData::List(lb)) => {
                if la.len() != lb.len() {
                    panic!("length mismatch in recursion");
                }
                let new_list = la
                    .iter()
                    .zip(lb.iter())
                    .map(|(item_a, item_b)| {
                        dyadic_scalar(item_a, item_b, op_ii, op_ff, _op_fi, _op_if)
                    })
                    .collect();
                K::from_list(new_list)
            }
            _ => panic!("recursion unreachable state"),
        }
    } else {
        panic!("type error: unsupported types at={}, bt={}", s.at, s.bt);
    }
}

// ===================================================================
// times() — element-wise multiplication (_mul)
//
// C original (va.c):
//   K times(K a, K b)
//   {
//     SCALAR_INIT(2)
//     K z=newK(zt,zn);U(z)
//     #define TIMES(x, y) ((x) * (y))
//     #define TIMES_FI(x, y) ((x) * I2F(y))
//     #define TIMES_IF(x, y) (I2F(x) * (y))
//     if(2==ABS(at)&&2==ABS(bt)){ SCALAR_OP_CASE(TIMES,   kF(z),kF(a),kF(b)) }
//     else if(2==ABS(at)&&1==ABS(bt)){ SCALAR_OP_CASE(TIMES_FI,kF(z),kF(a),kI(b)) }
//     else if(1==ABS(at)&&2==ABS(bt)){ SCALAR_OP_CASE(TIMES_IF,kF(z),kI(a),kF(b)) }
//     else if(1==ABS(at)&&1==ABS(bt)){ SCALAR_OP_CASE(TIMES,   kI(z),kI(a),kI(b)) }
//     else if(0==at||0==bt){ dp(&z,times,a,b); }
//     R z;
//   }
// ===================================================================

pub fn times(a: &K, b: &K) -> K {
    dyadic_scalar(
        a,
        b,
        |x, y| x * y,         // TIMES(x,y) = x * y  (int)
        |x, y| x * y,         // TIMES(x,y) = x * y  (float)
        |x, y| x * K::i2f(y), // TIMES_FI(x,y) = x * I2F(y)
        |x, y| K::i2f(x) * y, // TIMES_IF(x,y) = I2F(x) * y
    )
}

// ===================================================================
// _dot() — dot product / fused multiply-accumulate
//
// C original (va.c):
//   K _dot(K a,K b)
//   {
//     SCALAR_INIT(2);
//     I A=ABS(at),B=ABS(bt);
//     I accI=0;F accF=0.0;
//     #define DOT_F   accF+=x*y
//     #define DOT_FI  accF+=x*I2F(y)
//     #define DOT_IF  accF+=I2F(x)*y
//     #define DOT_I   accI+=x*y
//     if(2==A&&2==B){ F x,y; SCALAR_EXPR_CASE(DOT_F, F,kF(a),kF(b),x,y) }
//     else if(2==A&&1==B){ F x;I y; SCALAR_EXPR_CASE(DOT_FI,F,kF(a),kI(b),x,y) }
//     else if(1==A&&2==B){ I x;F y; SCALAR_EXPR_CASE(DOT_IF,F,kI(a),kF(b),x,y) }
//     else if(1==A&&1==B){ I x,y; SCALAR_EXPR_CASE(DOT_I, I,kI(a),kI(b),x,y) }
//     else if(0==A||0==B){
//       V p[]={0,(V)0x16};
//       K x,y=overDyad(0,p+2,(x=times(a,b))); cd(x);
//       R y;
//     }
//     R 1==ABS(zt)?Ki(accI):Kf(accF);
//   }
//
// Key insight: For flat numeric arrays, _dot does the multiply-accumulate
// INLINE in a single loop (accF += x * y). No intermediate array.
// This is the fused operation that maps directly to GPU FMA instructions.
//
// For general lists (t=0), it falls back to times() then overDyad() to sum.
// That's: +/ times(a,b) — multiply then reduce with plus.
// ===================================================================

pub fn dot(a: &K, b: &K) -> K {
    let s = scalar_init(a, b);

    let abs_a = s.at.abs();
    let abs_b = s.bt.abs();

    if abs_a == 2 && abs_b == 2 {
        let af = a.kf_data();
        let bf = b.kf_data();
        let n = s.zn as usize;
        let acc_f = dot_ff_parallel(af, bf, s.an, s.bn, n);
        K::kf(acc_f)
    } else if abs_a == 2 && abs_b == 1 {
        let af = a.kf_data();
        let bi = b.ki_data();
        let n = s.zn as usize;
        let acc_f = dot_fi_parallel(af, bi, s.an, s.bn, n);
        K::kf(acc_f)
    } else if abs_a == 1 && abs_b == 2 {
        let ai = a.ki_data();
        let bf = b.kf_data();
        let n = s.zn as usize;
        let acc_f = dot_if_parallel(ai, bf, s.an, s.bn, n);
        K::kf(acc_f)
    } else if abs_a == 1 && abs_b == 1 {
        let ai = a.ki_data();
        let bi = b.ki_data();
        let n = s.zn as usize;
        let acc_i = dot_ii_parallel(ai, bi, s.an, s.bn, n);
        K::ki(acc_i)
    } else if abs_a == 0 || abs_b == 0 {
        let product = times(a, b);
        match &product.data {
            KData::Ints(v) => K::ki(v.iter().sum()),
            KData::Floats(v) => K::kf(v.iter().sum()),
            _ => panic!("dot: cannot sum general list"),
        }
    } else {
        panic!("type error in _dot: at={}, bt={}", s.at, s.bt);
    }
}

// ===================================================================
// plus() — element-wise addition
//
// C original (va.c):
//   K plus(K a, K b)
//   {
//     SCALAR_INIT(2)
//     K z=newK(zt,zn);U(z)
//     #define PLUS(x, y) ((x) + (y))
//     #define PLUS_FI(x, y) ((x) + I2F(y))
//     #define PLUS_IF(x, y) (I2F(x) + (y))
//     ...same dispatch pattern as times...
//     R z;
//   }
// ===================================================================

#[allow(dead_code)]
pub fn plus(a: &K, b: &K) -> K {
    dyadic_scalar(
        a,
        b,
        |x, y| x + y,         // PLUS(x,y) = x + y  (int)
        |x, y| x + y,         // PLUS(x,y) = x + y  (float)
        |x, y| x + K::i2f(y), // PLUS_FI(x,y) = x + I2F(y)
        |x, y| K::i2f(x) + y, // PLUS_IF(x,y) = I2F(x) + y
    )
}

// ===================================================================
// minus() — element-wise subtraction
//
// C original (va.c):
//   K minus(K a, K b)
//   {
//     SCALAR_INIT(2)
//     K z=newK(zt,zn);U(z)
//     #define MINUS(x, y) ((x) - (y))
//     #define MINUS_FI(x, y) ((x) - I2F(y))
//     #define MINUS_IF(x, y) (I2F(x) - (y))
//     ...same dispatch pattern as times...
//     R z;
//   }
// ===================================================================

#[allow(dead_code)]
pub fn minus(a: &K, b: &K) -> K {
    dyadic_scalar(
        a,
        b,
        |x, y| x - y,         // MINUS(x,y) = x - y  (int)
        |x, y| x - y,         // MINUS(x,y) = x - y  (float)
        |x, y| x - K::i2f(y), // MINUS_FI(x,y) = x - I2F(y)
        |x, y| K::i2f(x) - y, // MINUS_IF(x,y) = I2F(x) - y
    )
}

// ===================================================================
// Core numeric and comparison primitives from va.c / vc.c
// ===================================================================

fn each_numeric_i64(x: &K, f: fn(f64) -> i64) -> K {
    match &x.data {
        KData::Ints(v) => {
            let out: Vec<i64> = v.iter().map(|&a| f(a as f64)).collect();
            K {
                t: if x.t < 0 { -1 } else { 1 },
                n: x.n,
                data: KData::Ints(out),
            }
        }
        KData::Floats(v) => {
            let out: Vec<i64> = v.iter().map(|&a| f(a)).collect();
            K {
                t: if x.t < 0 { -1 } else { 1 },
                n: x.n,
                data: KData::Ints(out),
            }
        }
        KData::List(v) => K::from_list(v.iter().map(|item| each_numeric_i64(item, f)).collect()),
    }
}

fn dyadic_float<F>(a: &K, b: &K, op: F) -> K
where
    F: Fn(f64, f64) -> f64 + Copy + Send + Sync,
{
    let s = scalar_init(a, b);
    match (&a.data, &b.data) {
        (KData::Ints(ai), KData::Ints(bi)) => {
            let av: Vec<f64> = ai.iter().map(|&x| x as f64).collect();
            let bv: Vec<f64> = bi.iter().map(|&x| x as f64).collect();
            let out = scalar_op_f64(&av, &bv, s.an, s.bn, s.zn, op);
            K {
                t: if s.zn == 1 && s.at < 0 && s.bt < 0 {
                    -2
                } else {
                    2
                },
                n: s.zn,
                data: KData::Floats(out),
            }
        }
        (KData::Ints(ai), KData::Floats(bf)) => {
            let av: Vec<f64> = ai.iter().map(|&x| x as f64).collect();
            let out = scalar_op_f64(&av, bf, s.an, s.bn, s.zn, op);
            K {
                t: if s.zn == 1 && s.at < 0 && s.bt < 0 {
                    -2
                } else {
                    2
                },
                n: s.zn,
                data: KData::Floats(out),
            }
        }
        (KData::Floats(af), KData::Ints(bi)) => {
            let bv: Vec<f64> = bi.iter().map(|&x| x as f64).collect();
            let out = scalar_op_f64(af, &bv, s.an, s.bn, s.zn, op);
            K {
                t: if s.zn == 1 && s.at < 0 && s.bt < 0 {
                    -2
                } else {
                    2
                },
                n: s.zn,
                data: KData::Floats(out),
            }
        }
        (KData::Floats(af), KData::Floats(bf)) => {
            let out = scalar_op_f64(af, bf, s.an, s.bn, s.zn, op);
            K {
                t: if s.zn == 1 && s.at < 0 && s.bt < 0 {
                    -2
                } else {
                    2
                },
                n: s.zn,
                data: KData::Floats(out),
            }
        }
        (KData::List(la), _) if a.t == 0 && b.t != 0 => {
            K::from_list(la.iter().map(|item| dyadic_float(item, b, op)).collect())
        }
        (_, KData::List(lb)) if a.t != 0 && b.t == 0 => {
            K::from_list(lb.iter().map(|item| dyadic_float(a, item, op)).collect())
        }
        (KData::List(la), KData::List(lb)) => {
            if la.len() != lb.len() {
                panic!("length error: lists mismatch");
            }
            K::from_list(
                la.iter()
                    .zip(lb.iter())
                    .map(|(x, y)| dyadic_float(x, y, op))
                    .collect(),
            )
        }
        _ => panic!("type error: expected numeric data"),
    }
}

fn dyadic_int<F>(a: &K, b: &K, op: F) -> K
where
    F: Fn(i64, i64) -> i64 + Copy + Send + Sync,
{
    let s = scalar_init(a, b);
    match (&a.data, &b.data) {
        (KData::Ints(ai), KData::Ints(bi)) => {
            let out = scalar_op_i64(ai, bi, s.an, s.bn, s.zn, op);
            K {
                t: if s.zn == 1 && s.at < 0 && s.bt < 0 {
                    -1
                } else {
                    1
                },
                n: s.zn,
                data: KData::Ints(out),
            }
        }
        (KData::List(la), _) if a.t == 0 && b.t != 0 => {
            K::from_list(la.iter().map(|item| dyadic_int(item, b, op)).collect())
        }
        (_, KData::List(lb)) if a.t != 0 && b.t == 0 => {
            K::from_list(lb.iter().map(|item| dyadic_int(a, item, op)).collect())
        }
        (KData::List(la), KData::List(lb)) => {
            if la.len() != lb.len() {
                panic!("length error: lists mismatch");
            }
            K::from_list(
                la.iter()
                    .zip(lb.iter())
                    .map(|(x, y)| dyadic_int(x, y, op))
                    .collect(),
            )
        }
        _ => dyadic_float(a, b, |x, y| op(x as i64, y as i64) as f64),
    }
}

fn dyadic_compare<F>(a: &K, b: &K, op: F) -> K
where
    F: Fn(f64, f64) -> bool + Copy + Send + Sync,
{
    let s = scalar_init(a, b);
    let bools: Vec<i64> = match (&a.data, &b.data) {
        (KData::Ints(ai), KData::Ints(bi)) => {
            let av: Vec<f64> = ai.iter().map(|&x| x as f64).collect();
            let bv: Vec<f64> = bi.iter().map(|&x| x as f64).collect();
            scalar_op_f64(
                &av,
                &bv,
                s.an,
                s.bn,
                s.zn,
                move |x, y| if op(x, y) { 1.0 } else { 0.0 },
            )
            .into_iter()
            .map(|x| x as i64)
            .collect()
        }
        (KData::Ints(ai), KData::Floats(bf)) => {
            let av: Vec<f64> = ai.iter().map(|&x| x as f64).collect();
            scalar_op_f64(
                &av,
                bf,
                s.an,
                s.bn,
                s.zn,
                move |x, y| if op(x, y) { 1.0 } else { 0.0 },
            )
            .into_iter()
            .map(|x| x as i64)
            .collect()
        }
        (KData::Floats(af), KData::Ints(bi)) => {
            let bv: Vec<f64> = bi.iter().map(|&x| x as f64).collect();
            scalar_op_f64(
                af,
                &bv,
                s.an,
                s.bn,
                s.zn,
                move |x, y| if op(x, y) { 1.0 } else { 0.0 },
            )
            .into_iter()
            .map(|x| x as i64)
            .collect()
        }
        (KData::Floats(af), KData::Floats(bf)) => scalar_op_f64(
            af,
            bf,
            s.an,
            s.bn,
            s.zn,
            move |x, y| if op(x, y) { 1.0 } else { 0.0 },
        )
        .into_iter()
        .map(|x| x as i64)
        .collect(),
        (KData::List(la), _) if a.t == 0 && b.t != 0 => {
            return K::from_list(la.iter().map(|item| dyadic_compare(item, b, op)).collect());
        }
        (_, KData::List(lb)) if a.t != 0 && b.t == 0 => {
            return K::from_list(lb.iter().map(|item| dyadic_compare(a, item, op)).collect());
        }
        (KData::List(la), KData::List(lb)) => {
            if la.len() != lb.len() {
                panic!("length error: lists mismatch");
            }
            return K::from_list(
                la.iter()
                    .zip(lb.iter())
                    .map(|(x, y)| dyadic_compare(x, y, op))
                    .collect(),
            );
        }
        _ => panic!("type error: expected comparable data"),
    };
    K {
        t: if s.zn == 1 && s.at < 0 && s.bt < 0 {
            -1
        } else {
            1
        },
        n: s.zn,
        data: KData::Ints(bools),
    }
}

#[allow(dead_code)]
pub fn power(a: &K, b: &K) -> K {
    dyadic_float(a, b, |x, y| {
        if y == 0.0 {
            1.0
        } else if x == 0.0 {
            0.0
        } else {
            x.powf(y)
        }
    })
}

#[allow(dead_code)]
pub fn divide(a: &K, b: &K) -> K {
    dyadic_float(a, b, |x, y| x / y)
}

#[allow(dead_code)]
pub fn reciprocal(x: &K) -> K {
    divide(&K::kf(1.0), x)
}

#[allow(dead_code)]
pub fn negate(x: &K) -> K {
    minus(&K::ki(0), x)
}

#[allow(dead_code)]
pub fn modulo(a: &K, b: &K) -> K {
    match (&a.data, &b.data) {
        (KData::Ints(_), KData::Ints(_)) => dyadic_int(a, b, |x, y| {
            if y == 0 {
                x
            } else {
                x - y * ((x as f64 / y as f64).floor() as i64)
            }
        }),
        _ => dyadic_float(a, b, |x, y| {
            if y == 0.0 {
                x
            } else {
                let h = x - y * (x / y).floor();
                if h.abs() > 0.0 { h } else { 0.0 }
            }
        }),
    }
}

#[allow(dead_code)]
pub fn min_and(a: &K, b: &K) -> K {
    dyadic_float(a, b, |x, y| x.min(y))
}

#[allow(dead_code)]
pub fn max_or(a: &K, b: &K) -> K {
    dyadic_float(a, b, |x, y| x.max(y))
}

#[allow(dead_code)]
pub fn floor_verb(x: &K) -> K {
    each_numeric_i64(x, |v| v.floor() as i64)
}

#[allow(dead_code)]
pub fn ceiling(x: &K) -> K {
    each_numeric_i64(x, |v| v.ceil() as i64)
}

#[allow(dead_code)]
pub fn equals(a: &K, b: &K) -> K {
    dyadic_compare(a, b, |x, y| {
        if x.is_nan() && y.is_nan() {
            true
        } else {
            (x - y).abs() == 0.0
        }
    })
}

#[allow(dead_code)]
pub fn less(a: &K, b: &K) -> K {
    dyadic_compare(a, b, |x, y| x < y)
}

#[allow(dead_code)]
pub fn more(a: &K, b: &K) -> K {
    dyadic_compare(a, b, |x, y| x > y)
}

#[allow(dead_code)]
pub fn match_k(a: &K, b: &K) -> K {
    K::ki(if k_match(a, b) { 1 } else { 0 })
}

// ===================================================================
// Neural Network Primitives (Kona-NN)
// ===================================================================

/// Matrix Multiplication (A x B)
///
/// Supports:
///   A: MxK matrix (List of K vectors, or flat array treated as matrix if needed)
///   B: KxN matrix
///
/// specific implementation for now:
///   A is List of Float Arrays (Rows)
///   B is List of Float Arrays (Columns? Or Rows?)
///   Convention: A is [M, K], B is [K, N].
///   But in K, matrices are usually Lists of Rows.
///   So B being [K, N] means B is a list of K rows, each length N?
///   Matmul: element (i, j) = dot(A[i], BT[j]) where BT is B transposed.
///   Or dot(A[i], column(B, j)).
///
///   Let's stick to strict K convention: Matrix = List of Lists (Rows).
///   To do A x B efficiently, we need B transposed (columns as rows).
///   So: `mmul(A, B)` implies we dot every row of A with every column of B.
pub fn matmul(a: &K, b: &K) -> K {
    // 1. Validate A and B are lists (matrices)
    let a_rows = match &a.data {
        KData::List(v) => v,
        _ => panic!("matmul: A must be a list (matrix)"),
    };
    let b_rows = match &b.data {
        KData::List(v) => v,
        _ => panic!("matmul: B must be a list (matrix)"),
    };

    let m = a_rows.len(); // Rows in A
    if m == 0 {
        return K::from_list(vec![]);
    }
    let k = a_rows[0].n as usize; // Columns in A

    let b_k = b_rows.len(); // Rows in B
    if b_k == 0 {
        return K::from_list(vec![]);
    }
    let n = b_rows[0].n as usize; // Columns in B

    for (idx, row) in a_rows.iter().enumerate() {
        if row.n as usize != k {
            panic!("matmul: row {} in matrix A has invalid length {} (expected {})", idx, row.n, k);
        }
    }
    for (idx, row) in b_rows.iter().enumerate() {
        if row.n as usize != n {
            panic!("matmul: row {} in matrix B has invalid length {} (expected {})", idx, row.n, n);
        }
    }

    if k != b_k {
        panic!(
            "matmul: dimension mismatch: A[{}, {}] x B[{}, {}]",
            m, k, b_k, n
        );
    }

    // Transpose B into a contiguous 1D flat array (size N * K)
    // b_transposed[j * k + x] corresponds to B[x][j]
    let mut b_transposed = vec![0.0; n * k];
    for r in 0..k {
        let row = b_rows[r].kf_data();
        for c in 0..n {
            b_transposed[c * k + r] = row[c];
        }
    }

    // Determine # threads
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    
    // Check if total operations are small, run sequentially if so
    if m * n * k < 16_384 || m < 2 {
        let mut final_rows = Vec::with_capacity(m);
        for row_k in a_rows {
            let row_a = &row_k.kf_data()[..k];
            let mut row_c = Vec::with_capacity(n);
            for j in 0..n {
                let offset = j * k;
                let col_b = &b_transposed[offset..offset + k];
                let sum: f64 = row_a.iter().zip(col_b.iter()).map(|(&x, &y)| x * y).sum();
                row_c.push(sum);
            }
            final_rows.push(K::from_floats(row_c));
        }
        return K::from_list(final_rows);
    }

    let chunk_size = (m + num_threads - 1) / num_threads;

    let c_rows = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);

        for chunk in a_rows.chunks(chunk_size) {
            let b_transposed_ref = &b_transposed;

            handles.push(s.spawn(move || {
                let mut chunk_results = Vec::with_capacity(chunk.len());
                for row_k in chunk {
                    let row_a = &row_k.kf_data()[..k];
                    let mut row_c = Vec::with_capacity(n);
                    for j in 0..n {
                        let offset = j * k;
                        let col_b = &b_transposed_ref[offset..offset + k];
                        let sum: f64 = row_a.iter().zip(col_b.iter()).map(|(&x, &y)| x * y).sum();
                        row_c.push(sum);
                    }
                    chunk_results.push(K::from_floats(row_c));
                }
                chunk_results
            }));
        }

        let mut final_rows = Vec::with_capacity(m);
        for h in handles {
            final_rows.extend(h.join().unwrap());
        }
        final_rows
    });

    K::from_list(c_rows)
}

/// Transpose a Matrix (List of Lists)
/// M x N -> N x M
pub fn transpose(x: &K) -> K {
    let rows = match &x.data {
        KData::List(v) => v,
        _ => panic!("transpose: expected matrix (list of lists)"),
    };
    let m = rows.len();
    if m == 0 {
        return K::from_list(vec![]);
    }
    
    let first_row = &rows[0];
    let n = first_row.n as usize;
    
    match &first_row.data {
        KData::Ints(_) => {
            let mut cols = vec![Vec::with_capacity(m); n];
            for i in 0..m {
                match &rows[i].data {
                    KData::Ints(row) => {
                        if row.len() != n {
                            panic!("length error: rows must have equal length");
                        }
                        for j in 0..n {
                            cols[j].push(row[j]);
                        }
                    }
                    _ => panic!("type error: expected integer matrix"),
                }
            }
            let mut new_rows = Vec::with_capacity(n);
            for col in cols {
                new_rows.push(K::from_ints(col));
            }
            K::from_list(new_rows)
        }
        KData::Floats(_) => {
            let mut cols = vec![Vec::with_capacity(m); n];
            for i in 0..m {
                match &rows[i].data {
                    KData::Floats(row) => {
                        if row.len() != n {
                            panic!("length error: rows must have equal length");
                        }
                        for j in 0..n {
                            cols[j].push(row[j]);
                        }
                    }
                    _ => panic!("type error: expected float matrix"),
                }
            }
            let mut new_rows = Vec::with_capacity(n);
            for col in cols {
                new_rows.push(K::from_floats(col));
            }
            K::from_list(new_rows)
        }
        KData::List(_) => {
            let mut cols = vec![Vec::with_capacity(m); n];
            for i in 0..m {
                match &rows[i].data {
                    KData::List(row) => {
                        if row.len() != n {
                            panic!("length error: rows must have equal length");
                        }
                        for j in 0..n {
                            cols[j].push(row[j].clone());
                        }
                    }
                    _ => panic!("type error: expected list matrix"),
                }
            }
            let mut new_rows = Vec::with_capacity(n);
            for col in cols {
                new_rows.push(K::from_list(col));
            }
            K::from_list(new_rows)
        }
    }
}

#[allow(dead_code)]
pub fn flip(x: &K) -> K {
    transpose(x)
}


/// Sigmoid Activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res = map_f64_parallel(v, |val| 1.0 / (1.0 + (-val).exp()));
            K::from_floats(res)
        }
        KData::List(rows) => {
            K::from_list(map_rows_parallel(rows, |row| sigmoid(row)))
        }
        _ => panic!("sigmoid: expected float array or matrix"),
    }
}

/// Tanh Activation
pub fn tanh(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res = map_f64_parallel(v, |val| val.tanh());
            K::from_floats(res)
        }
        KData::List(rows) => {
            K::from_list(map_rows_parallel(rows, |row| tanh(row)))
        }
        _ => panic!("tanh: expected float array or matrix"),
    }
}

/// ReLU Activation: max(0, x)
pub fn relu(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res = map_f64_parallel(v, |val| if val > 0.0 { val } else { 0.0 });
            K::from_floats(res)
        }
        KData::List(rows) => {
            K::from_list(map_rows_parallel(rows, |row| relu(row)))
        }
        _ => panic!("relu: expected float array or matrix"),
    }
}

/// Softmax: exp(x) / sum(exp(x)) (Row-wise if matrix, or global if vector)
pub fn softmax(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res = softmax_floats_parallel(v);
            K::from_floats(res)
        }
        KData::List(rows) => {
            K::from_list(map_rows_parallel(rows, |row| softmax(row)))
        }
        _ => panic!("softmax: expected float array or list of arrays"),
    }
}

/// Sum columns (reduce along axis 0) -> Row Vector
/// Used for bias gradient: db = sum(dy, axis=0)
pub fn sum_cols(x: &K) -> K {
    match &x.data {
        KData::List(rows) => {
            if rows.is_empty() {
                return K::from_list(vec![]);
            }
            let n = rows[0].n as usize;
            let mut sum = vec![0.0; n];

            for (idx, row) in rows.iter().enumerate() {
                let rf = row.kf_data();
                if rf.len() != n {
                    panic!("sum_cols: row {} has invalid length {} (expected {})", idx, rf.len(), n);
                }
                for i in 0..n {
                    sum[i] += rf[i];
                }
            }
            K::from_floats(sum)
        }
        KData::Floats(_) => x.clone(), // Already a vector
        _ => panic!("sum_cols: expected matrix or vector"),
    }
}

/// ReLU Derivative: 1 where x > 0, else 0
pub fn relu_derivative(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res = map_f64_parallel(v, |val| if val > 0.0 { 1.0 } else { 0.0 });
            K::from_floats(res)
        }
        KData::List(rows) => {
            K::from_list(map_rows_parallel(rows, |row| relu_derivative(row)))
        }
        _ => panic!("relu_derivative: expected float/matrix"),
    }
}

/// Sigmoid Backward: sig(x) * (1 - sig(x)) * dy
pub fn sigmoid_backward(x: &K, dy: &K) -> K {
    let s = sigmoid(x);
    sigmoid_backward_inner(&s, dy)
}

// Helper to avoid re-computing sigmoid(sigmoid(x))
fn sigmoid_backward_inner(s: &K, dy: &K) -> K {
    match (&s.data, &dy.data) {
        (KData::Floats(sv), KData::Floats(dv)) => {
            let res = parallel::binary_f64(sv, dv, s.n, dy.n, s.n, |sig, d| sig * (1.0 - sig) * d);
            K::from_floats(res)
        }
        (KData::List(sr), KData::List(dr)) => {
            K::from_list(zip_rows_parallel(sr, dr, |r1, r2| sigmoid_backward_inner(r1, r2)))
        }
        _ => panic!("sigmoid_backward_inner: mismatch"),
    }
}

/// Tanh Backward: (1 - tanh(x)^2) * dy
pub fn tanh_backward(x: &K, dy: &K) -> K {
    let t = tanh(x); // output of tanh
    tanh_backward_inner(&t, dy)
}

fn tanh_backward_inner(t: &K, dy: &K) -> K {
    match (&t.data, &dy.data) {
        (KData::Floats(tv), KData::Floats(dv)) => {
            let res = parallel::binary_f64(tv, dv, t.n, dy.n, t.n, |tah, d| (1.0 - tah * tah) * d);
            K::from_floats(res)
        }
        (KData::List(tr), KData::List(dr)) => {
            K::from_list(zip_rows_parallel(tr, dr, |r1, r2| tanh_backward_inner(r1, r2)))
        }
        _ => panic!("tanh_backward: mismatch"),
    }
}
// ===================================================================
// Monadic Verbs (Single argument)
// ===================================================================

// ===================================================================
// Monadic Verbs (Single argument)
// ===================================================================

pub fn first(x: &K) -> K {
    if x.t < 0 {
        return x.clone();
    }
    match &x.data {
        KData::Ints(v) => {
            if v.is_empty() {
                K::ki(0)
            } else {
                K::ki(v[0])
            }
        }
        KData::Floats(v) => {
            if v.is_empty() {
                K::kf(0.0)
            } else {
                K::kf(v[0])
            }
        }
        KData::List(v) => {
            if v.is_empty() {
                x.clone()
            } else {
                v[0].clone()
            }
        }
    }
}

pub fn reverse(x: &K) -> K {
    if x.t < 0 {
        return x.clone();
    }
    match &x.data {
        KData::Ints(v) => {
            let mut rev = v.clone();
            rev.reverse();
            K {
                t: x.t,
                n: x.n,
                data: KData::Ints(rev),
            }
        }
        KData::Floats(v) => {
            let mut rev = v.clone();
            rev.reverse();
            K {
                t: x.t,
                n: x.n,
                data: KData::Floats(rev),
            }
        }
        KData::List(v) => {
            let mut rev = v.clone();
            rev.reverse();
            K {
                t: x.t,
                n: x.n,
                data: KData::List(rev),
            }
        }
    }
}

pub fn ln(x: &K) -> K {
    match x.t.abs() {
        2 => {
            // Float
            match &x.data {
                KData::Floats(v) => {
                    let res: Vec<f64> = v.iter().map(|f| f.ln()).collect();
                    K {
                        t: x.t,
                        n: x.n,
                        data: KData::Floats(res),
                    }
                }
                _ => panic!("type error: expected floats"),
            }
        }
        1 => {
            // Int -> Promoted to Float
            match &x.data {
                KData::Ints(v) => {
                    let res: Vec<f64> = v.iter().map(|i| (*i as f64).ln()).collect();
                    let t = if x.t < 0 { -2 } else { 2 };
                    K {
                        t,
                        n: x.n,
                        data: KData::Floats(res),
                    }
                }
                _ => panic!("type error: expected ints"),
            }
        }
        0 => {
            // List (recurse)
            match &x.data {
                KData::List(v) => {
                    let res: Vec<K> = v.iter().map(|k| ln(k)).collect();
                    K {
                        t: 0,
                        n: x.n,
                        data: KData::List(res),
                    }
                }
                _ => panic!("type error: expected list"),
            }
        }
        _ => panic!("type error in ln"),
    }
}

pub fn exp(x: &K) -> K {
    match x.t.abs() {
        2 => {
            // Float
            match &x.data {
                KData::Floats(v) => {
                    let res: Vec<f64> = v.iter().map(|f| f.exp()).collect();
                    K {
                        t: x.t,
                        n: x.n,
                        data: KData::Floats(res),
                    }
                }
                _ => panic!("type error: expected floats"),
            }
        }
        1 => {
            // Int -> Promoted to Float
            match &x.data {
                KData::Ints(v) => {
                    let res: Vec<f64> = v.iter().map(|i| (*i as f64).exp()).collect();
                    let t = if x.t < 0 { -2 } else { 2 };
                    K {
                        t,
                        n: x.n,
                        data: KData::Floats(res),
                    }
                }
                _ => panic!("type error: expected ints"),
            }
        }
        0 => {
            // List (recurse)
            match &x.data {
                KData::List(v) => {
                    let res: Vec<K> = v.iter().map(|k| exp(k)).collect();
                    K {
                        t: 0,
                        n: x.n,
                        data: KData::List(res),
                    }
                }
                _ => panic!("type error: expected list"),
            }
        }
        _ => panic!("type error in exp"),
    }
}

pub fn sqrt(x: &K) -> K {
    match x.t.abs() {
        2 => {
            // Float
            match &x.data {
                KData::Floats(v) => {
                    let res: Vec<f64> = v.iter().map(|f| f.sqrt()).collect();
                    K {
                        t: x.t,
                        n: x.n,
                        data: KData::Floats(res),
                    }
                }
                _ => panic!("type error: expected floats"),
            }
        }
        1 => {
            // Int -> Promoted to Float
            match &x.data {
                KData::Ints(v) => {
                    let res: Vec<f64> = v.iter().map(|i| (*i as f64).sqrt()).collect();
                    let t = if x.t < 0 { -2 } else { 2 };
                    K {
                        t,
                        n: x.n,
                        data: KData::Floats(res),
                    }
                }
                _ => panic!("type error: expected ints"),
            }
        }
        0 => {
            // List (recurse)
            match &x.data {
                KData::List(v) => {
                    let res: Vec<K> = v.iter().map(|k| sqrt(k)).collect();
                    K {
                        t: 0,
                        n: x.n,
                        data: KData::List(res),
                    }
                }
                _ => panic!("type error: expected list"),
            }
        }
        _ => panic!("type error in sqrt"),
    }
}

/// Upper Triangle (Causal Mask)
/// Sets elements below diagonal to -inf
pub fn triu(x: &K, k: i64) -> K {
    match &x.data {
        KData::List(rows) => {
            let mut new_rows = Vec::new();
            for (i, row_k) in rows.iter().enumerate() {
                let mut new_row_data = Vec::new();
                match &row_k.data {
                    KData::Floats(vals) => {
                        for (j, val) in vals.iter().enumerate() {
                            if (j as i64) >= (i as i64) + k {
                                new_row_data.push(*val);
                            } else {
                                new_row_data.push(f64::NEG_INFINITY);
                            }
                        }
                    }
                    _ => panic!("triu: expected float matrix"),
                }
                new_rows.push(K::from_floats(new_row_data));
            }
            K::from_list(new_rows)
        }
        _ => panic!("triu: expected matrix (list of lists)"),
    }
}

/// Gather: Embedding Lookup
/// x: Indices (Int Array)
/// table: Embedding Matrix (List of Vectors)
pub fn gather(indices: &K, table: &K) -> K {
    // 1. Validate Table is a matrix
    let rows = match &table.data {
        KData::List(v) => v,
        _ => panic!("gather: table must be a list (matrix)"),
    };

    // 2. Map indices to rows
    match &indices.data {
        KData::Ints(idxs) => {
            let mut result_rows = Vec::with_capacity(idxs.len());
            for &idx in idxs {
                let i = idx as usize;
                if i >= rows.len() {
                    panic!("gather: index out of bounds {} >= {}", i, rows.len());
                }
                result_rows.push(rows[i].clone());
            }
            K::from_list(result_rows)
        }
        KData::List(list_idxs) => {
            // Batch gather (recurse)
            let res: Vec<K> = list_idxs.iter().map(|k| gather(k, table)).collect();
            K::from_list(res)
        }
        _ => panic!("gather: indices must be int array or list of int arrays"),
    }
}

// ===================================================================
// find() — search for element in array, return index
//
// C original (vq.c):
//   K find(K a, K b)
//   {
//     I at=a->t, an=a->n, bt=b->t;
//     if(-2==at && 2==bt)DO(an, if(!FC(kF(a)[i],*kF(b)))R Ki(i))
//     if(-1==at && 1==bt)DO(an, if(kI(a)[i]==*kI(b))R Ki(i))
//     R Ki(an);
//   }
//
// Returns index of first match, or array length if not found.
// Kevin's pattern: one tight DO loop, early return on match.
// ===================================================================

/// Find value b in array a, return index. Kevin's vq.c find().
pub fn find(a: &K, b: &K) -> K {
    match (&a.data, &b.data) {
        // Float array, find a float — DO(an, if(!FC(kF(a)[i],*kF(b)))R Ki(i))
        (KData::Floats(af), KData::Floats(bf)) => {
            if bf.is_empty() {
                return K::ki(af.len() as i64);
            }
            let target = bf[0];
            for (i, &v) in af.iter().enumerate() {
                if (v - target).abs() < 1e-15 {
                    return K::ki(i as i64);
                }
            }
            K::ki(af.len() as i64) // Not found → return length
        }
        // Int array, find an int — DO(an, if(kI(a)[i]==*kI(b))R Ki(i))
        (KData::Ints(ai), KData::Ints(bi)) => {
            if bi.is_empty() {
                return K::ki(ai.len() as i64);
            }
            let target = bi[0];
            for (i, &v) in ai.iter().enumerate() {
                if v == target {
                    return K::ki(i as i64);
                }
            }
            K::ki(ai.len() as i64) // Not found
        }
        // List: search through items — DO(an, if(matchI(kK(a)[i],b))R Ki(i))
        (KData::List(items), _) => {
            for (i, item) in items.iter().enumerate() {
                if k_match(item, b) {
                    return K::ki(i as i64);
                }
            }
            K::ki(items.len() as i64)
        }
        _ => panic!("find: type mismatch"),
    }
}

/// Match two K values for equality. Kevin's matchI().
fn k_match(a: &K, b: &K) -> bool {
    match (&a.data, &b.data) {
        (KData::Ints(ai), KData::Ints(bi)) => ai == bi,
        (KData::Floats(af), KData::Floats(bf)) => {
            af.len() == bf.len() && af.iter().zip(bf.iter()).all(|(x, y)| (x - y).abs() < 1e-15)
        }
        (KData::List(al), KData::List(bl)) => {
            al.len() == bl.len() && al.iter().zip(bl.iter()).all(|(x, y)| k_match(x, y))
        }
        _ => false,
    }
}

// ===================================================================
// where() — expand count vector into index vector
//
// C original (vg.c):
//   K where(K x)
//   {
//     I zn=0,y,j,t=0;
//     DO(xn,if((y=kI(x)[i])<0)continue;zn+=y)
//     K z=newK(-1,zn); U(z)
//     DO(xn, for(j=0;j<kI(x)[i];j++)kI(z)[t++]=i)
//     R z;
//   }
//
// Example: where(0 1 0 2) → (1 3 3)
//          where(3 0 1)   → (0 0 0 2)
//
// Each index i appears x[i] times in the output.
// This is used for sparse-to-dense expansion and selection.
// ===================================================================

/// Expand count vector to index vector. Kevin's vg.c where().
pub fn kona_where(x: &K) -> K {
    let data = x.ki_data();

    // First pass: count total output length — DO(xn, zn+=y)
    let mut zn: usize = 0;
    for &y in data {
        if y > 0 {
            zn += y as usize;
        }
    }

    // Second pass: fill — DO(xn, for(j=0;j<kI(x)[i];j++)kI(z)[t++]=i)
    let mut z = Vec::with_capacity(zn);
    for (i, &count) in data.iter().enumerate() {
        for _ in 0..count.max(0) {
            z.push(i as i64);
        }
    }

    K::from_ints(z)
}

// ===================================================================
// find_all() — find index of each element of b in a
//
// Kevin's dyadic ? when both args are vectors:
//   For each element in b, find its position in a.
//   Uses hash for O(1) lookup when a has a hash table.
//
// This is the vectorized version:
//   result[i] = find(a, b[i])
// ===================================================================

/// Find all: for each element in b, find its index in a.
pub fn find_all(a: &K, b: &K) -> K {
    match (&a.data, &b.data) {
        // Int vectors: build hash map for O(1) lookup per element
        (KData::Ints(ai), KData::Ints(bi)) => {
            // Build index: first occurrence of each value
            let mut map = std::collections::HashMap::with_capacity(ai.len());
            for (i, &v) in ai.iter().enumerate() {
                map.entry(v).or_insert(i);
            }
            let an = ai.len() as i64;
            // Lookup each element of b
            let result: Vec<i64> = bi
                .iter()
                .map(|&v| map.get(&v).map(|&i| i as i64).unwrap_or(an))
                .collect();
            K::from_ints(result)
        }
        // Float vectors: linear scan per element (no hash for floats)
        (KData::Floats(af), KData::Floats(bf)) => {
            let an = af.len() as i64;
            let result: Vec<i64> = bf
                .iter()
                .map(|&target| {
                    af.iter()
                        .position(|&v| (v - target).abs() < 1e-15)
                        .map(|i| i as i64)
                        .unwrap_or(an)
                })
                .collect();
            K::from_ints(result)
        }
        _ => panic!("find_all: expected matching vector types"),
    }
}

// ===================================================================
// grade_up() and grade_down() — monadic < and >
// ===================================================================

fn cmp_floats_ascending(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater, // NaN goes to the end
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

fn cmp_floats_descending(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater, // NaN goes to the end
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.total_cmp(&a),            // Reverse order for non-NaN
    }
}

fn compare_k(a: &K, b: &K) -> std::cmp::Ordering {
    match (&a.data, &b.data) {
        (KData::Ints(av), KData::Ints(bv)) => {
            for (x, y) in av.iter().zip(bv.iter()) {
                match x.cmp(y) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            av.len().cmp(&bv.len())
        }
        (KData::Floats(av), KData::Floats(bv)) => {
            for (&x, &y) in av.iter().zip(bv.iter()) {
                match cmp_floats_ascending(x, y) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            av.len().cmp(&bv.len())
        }
        (KData::List(av), KData::List(bv)) => {
            for (x, y) in av.iter().zip(bv.iter()) {
                match compare_k(x, y) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            av.len().cmp(&bv.len())
        }
        _ => a.t.cmp(&b.t),
    }
}

fn compare_k_descending(a: &K, b: &K) -> std::cmp::Ordering {
    match (&a.data, &b.data) {
        (KData::Ints(av), KData::Ints(bv)) => {
            for (x, y) in av.iter().zip(bv.iter()) {
                match y.cmp(x) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            bv.len().cmp(&av.len())
        }
        (KData::Floats(av), KData::Floats(bv)) => {
            for (&x, &y) in av.iter().zip(bv.iter()) {
                match cmp_floats_descending(x, y) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            bv.len().cmp(&av.len())
        }
        (KData::List(av), KData::List(bv)) => {
            for (x, y) in av.iter().zip(bv.iter()) {
                match compare_k_descending(x, y) {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            bv.len().cmp(&av.len())
        }
        _ => b.t.cmp(&a.t),
    }
}

/// Monadic < (grade up): returns indices that sort x ascending.
pub fn grade_up(x: &K) -> K {
    match &x.data {
        KData::Ints(v) => {
            let mut indices: Vec<i64> = (0..v.len() as i64).collect();
            indices.sort_by(|&i, &j| v[i as usize].cmp(&v[j as usize]));
            K::from_ints(indices)
        }
        KData::Floats(v) => {
            let mut indices: Vec<i64> = (0..v.len() as i64).collect();
            indices.sort_by(|&i, &j| cmp_floats_ascending(v[i as usize], v[j as usize]));
            K::from_ints(indices)
        }
        KData::List(v) => {
            let mut indices: Vec<i64> = (0..v.len() as i64).collect();
            indices.sort_by(|&i, &j| compare_k(&v[i as usize], &v[j as usize]));
            K::from_ints(indices)
        }
    }
}

/// Monadic > (grade down): returns indices that sort x descending.
pub fn grade_down(x: &K) -> K {
    match &x.data {
        KData::Ints(v) => {
            let mut indices: Vec<i64> = (0..v.len() as i64).collect();
            indices.sort_by(|&i, &j| v[j as usize].cmp(&v[i as usize]));
            K::from_ints(indices)
        }
        KData::Floats(v) => {
            let mut indices: Vec<i64> = (0..v.len() as i64).collect();
            indices.sort_by(|&i, &j| cmp_floats_descending(v[i as usize], v[j as usize]));
            K::from_ints(indices)
        }
        KData::List(v) => {
            let mut indices: Vec<i64> = (0..v.len() as i64).collect();
            indices.sort_by(|&i, &j| compare_k_descending(&v[i as usize], &v[j as usize]));
            K::from_ints(indices)
        }
    }
}

// ===================================================================
// take() — dyadic # slice portion
// ===================================================================

/// Dyadic # (take): returns first or last abs(n) elements of b, cycling.
pub fn take(a: &K, b: &K) -> K {
    let n = match &a.data {
        KData::Ints(v) if v.len() == 1 => v[0],
        KData::Floats(v) if v.len() == 1 => v[0] as i64,
        _ => panic!("type error: take left argument must be an integer or float scalar"),
    };

    let len = n.abs() as usize;
    let b_t = b.t;
    let z_t = b_t.abs();

    match &b.data {
        KData::Ints(v) => {
            if v.is_empty() {
                if len > 0 {
                    panic!("length error: cannot take from empty list");
                }
                return K {
                    t: z_t,
                    n: 0,
                    data: KData::Ints(vec![]),
                };
            }
            let mut z = Vec::with_capacity(len);
            let y_len = v.len();
            for i in 0..len {
                let idx = if n >= 0 {
                    i % y_len
                } else {
                    (y_len - (len - i) % y_len) % y_len
                };
                z.push(v[idx]);
            }
            K {
                t: z_t,
                n: len as i64,
                data: KData::Ints(z),
            }
        }
        KData::Floats(v) => {
            if v.is_empty() {
                if len > 0 {
                    panic!("length error: cannot take from empty list");
                }
                return K {
                    t: z_t,
                    n: 0,
                    data: KData::Floats(vec![]),
                };
            }
            let mut z = Vec::with_capacity(len);
            let y_len = v.len();
            for i in 0..len {
                let idx = if n >= 0 {
                    i % y_len
                } else {
                    (y_len - (len - i) % y_len) % y_len
                };
                z.push(v[idx]);
            }
            K {
                t: z_t,
                n: len as i64,
                data: KData::Floats(z),
            }
        }
        KData::List(v) => {
            if v.is_empty() {
                if len > 0 {
                    panic!("length error: cannot take from empty list");
                }
                return K {
                    t: z_t,
                    n: 0,
                    data: KData::List(vec![]),
                };
            }
            let mut z = Vec::with_capacity(len);
            let y_len = v.len();
            for i in 0..len {
                let idx = if n >= 0 {
                    i % y_len
                } else {
                    (y_len - (len - i) % y_len) % y_len
                };
                z.push(v[idx].clone());
            }
            K {
                t: z_t,
                n: len as i64,
                data: KData::List(z),
            }
        }
    }
}

/// Unique primitive: returns only unique elements in order of first appearance.
pub fn unique(x: &K) -> K {
    if x.t < 0 {
        return x.clone();
    }
    match &x.data {
        KData::Ints(v) => {
            let mut seen = std::collections::HashSet::new();
            let mut unique_v: Vec<i64> = Vec::new();
            for &val in v {
                if seen.insert(val) {
                    unique_v.push(val);
                }
            }
            K {
                t: x.t,
                n: unique_v.len() as i64,
                data: KData::Ints(unique_v),
            }
        }
        KData::Floats(v) => {
            let mut unique_v: Vec<f64> = Vec::new();
            for &val in v {
                let mut found = false;
                for &u in &unique_v {
                    if (val.is_nan() && u.is_nan()) || (val - u).abs() < 1e-15 {
                        found = true;
                        break;
                    }
                }
                if !found {
                    unique_v.push(val);
                }
            }
            K {
                t: x.t,
                n: unique_v.len() as i64,
                data: KData::Floats(unique_v),
            }
        }
        KData::List(v) => {
            let mut unique_v: Vec<K> = Vec::new();
            for item in v {
                let mut found = false;
                for u in &unique_v {
                    if k_match(item, u) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    unique_v.push(item.clone());
                }
            }
            K {
                t: x.t,
                n: unique_v.len() as i64,
                data: KData::List(unique_v),
            }
        }
    }
}


// ===================================================================
// Parallel Helper Functions
// ===================================================================

fn map_rows_parallel<F>(rows: &[K], f: F) -> Vec<K>
where
    F: Fn(&K) -> K + Copy + Send + Sync,
{
    let len = rows.len();
    if len <= 4 {
        return rows.iter().map(f).collect();
    }

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = (len + num_threads - 1) / num_threads;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for chunk in rows.chunks(chunk_size) {
            handles.push(s.spawn(move || {
                chunk.iter().map(f).collect::<Vec<K>>()
            }));
        }
        let mut final_rows = Vec::with_capacity(len);
        for h in handles {
            final_rows.extend(h.join().unwrap());
        }
        final_rows
    })
}

fn zip_rows_parallel<F>(sr: &[K], dr: &[K], f: F) -> Vec<K>
where
    F: Fn(&K, &K) -> K + Copy + Send + Sync,
{
    let len = sr.len();
    if len <= 4 {
        return sr.iter().zip(dr.iter()).map(|(r1, r2)| f(r1, r2)).collect();
    }

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = (len + num_threads - 1) / num_threads;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for (chunk_s, chunk_d) in sr.chunks(chunk_size).zip(dr.chunks(chunk_size)) {
            handles.push(s.spawn(move || {
                chunk_s.iter().zip(chunk_d.iter()).map(|(r1, r2)| f(r1, r2)).collect::<Vec<K>>()
            }));
        }
        let mut final_rows = Vec::with_capacity(len);
        for h in handles {
            final_rows.extend(h.join().unwrap());
        }
        final_rows
    })
}

fn map_f64_parallel<F>(v: &[f64], op: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Copy + Send + Sync,
{
    let len = v.len();
    let workers = parallel::worker_count(len);
    if workers == 1 {
        return v.iter().map(|&x| op(x)).collect();
    }

    let mut out = vec![0.0; len];
    let chunk = len.div_ceil(workers);

    std::thread::scope(|scope| {
        let mut chunks = out.chunks_mut(chunk);
        for i in 0..workers {
            let start = i * chunk;
            if let Some(out_chunk) = chunks.next() {
                let v_slice = &v[start..start + out_chunk.len()];
                scope.spawn(move || {
                    for j in 0..out_chunk.len() {
                        out_chunk[j] = op(v_slice[j]);
                    }
                });
            }
        }
    });

    out
}

fn softmax_floats_parallel(v: &[f64]) -> Vec<f64> {
    let len = v.len();
    let workers = parallel::worker_count(len);
    if workers == 1 {
        let max_val = v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = v.iter().map(|&val| (val - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        return exps.iter().map(|&val| val / sum).collect();
    }

    let chunk = len.div_ceil(workers);

    let max_val = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            let slice = &v[start..end];
            handles.push(scope.spawn(move || {
                slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            }));
        }
        let mut global_max = f64::NEG_INFINITY;
        for h in handles {
            global_max = global_max.max(h.join().unwrap());
        }
        global_max
    });

    let mut exps = vec![0.0; len];

    let total_sum = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut chunks = exps.chunks_mut(chunk);
        for i in 0..workers {
            let start = i * chunk;
            if let Some(out_chunk) = chunks.next() {
                let slice = &v[start..start + out_chunk.len()];
                handles.push(scope.spawn(move || {
                    let mut local_sum = 0.0;
                    for j in 0..out_chunk.len() {
                        let exp_val = (slice[j] - max_val).exp();
                        out_chunk[j] = exp_val;
                        local_sum += exp_val;
                    }
                    local_sum
                }));
            }
        }
        let mut global_sum = 0.0;
        for h in handles {
            global_sum += h.join().unwrap();
        }
        global_sum
    });

    std::thread::scope(|scope| {
        let mut chunks = exps.chunks_mut(chunk);
        for _ in 0..workers {
            if let Some(out_chunk) = chunks.next() {
                scope.spawn(move || {
                    for val in out_chunk.iter_mut() {
                        *val /= total_sum;
                    }
                });
            }
        }
    });

    exps
}

fn dot_ff_parallel(af: &[f64], bf: &[f64], an: i64, bn: i64, len: usize) -> f64 {
    let workers = parallel::worker_count(len);
    if workers == 1 {
        return dot_ff_range(af, bf, an, bn, 0, len);
    }

    let chunk = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || {
                dot_ff_range(af, bf, an, bn, start, end)
            }));
        }
        let mut total = 0.0;
        for h in handles {
            total += h.join().unwrap();
        }
        total
    })
}

fn dot_ff_range(af: &[f64], bf: &[f64], an: i64, bn: i64, start: usize, end: usize) -> f64 {
    let mut sum = 0.0;
    if an == bn {
        let af_slice = &af[start..end];
        let bf_slice = &bf[start..end];
        for i in 0..af_slice.len() {
            sum += af_slice[i] * bf_slice[i];
        }
    } else if an == 1 {
        let a0 = af[0];
        let bf_slice = &bf[start..end];
        for i in 0..bf_slice.len() {
            sum += a0 * bf_slice[i];
        }
    } else {
        let b0 = bf[0];
        let af_slice = &af[start..end];
        for i in 0..af_slice.len() {
            sum += af_slice[i] * b0;
        }
    }
    sum
}

fn dot_fi_parallel(af: &[f64], bi: &[i64], an: i64, bn: i64, len: usize) -> f64 {
    let workers = parallel::worker_count(len);
    if workers == 1 {
        return dot_fi_range(af, bi, an, bn, 0, len);
    }

    let chunk = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || {
                dot_fi_range(af, bi, an, bn, start, end)
            }));
        }
        let mut total = 0.0;
        for h in handles {
            total += h.join().unwrap();
        }
        total
    })
}

fn dot_fi_range(af: &[f64], bi: &[i64], an: i64, bn: i64, start: usize, end: usize) -> f64 {
    let mut sum = 0.0;
    if an == bn {
        let af_slice = &af[start..end];
        let bi_slice = &bi[start..end];
        for i in 0..af_slice.len() {
            sum += af_slice[i] * (bi_slice[i] as f64);
        }
    } else if an == 1 {
        let a0 = af[0];
        let bi_slice = &bi[start..end];
        for i in 0..bi_slice.len() {
            sum += a0 * (bi_slice[i] as f64);
        }
    } else {
        let b0 = bi[0] as f64;
        let af_slice = &af[start..end];
        for i in 0..af_slice.len() {
            sum += af_slice[i] * b0;
        }
    }
    sum
}

fn dot_if_parallel(ai: &[i64], bf: &[f64], an: i64, bn: i64, len: usize) -> f64 {
    let workers = parallel::worker_count(len);
    if workers == 1 {
        return dot_if_range(ai, bf, an, bn, 0, len);
    }

    let chunk = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || {
                dot_if_range(ai, bf, an, bn, start, end)
            }));
        }
        let mut total = 0.0;
        for h in handles {
            total += h.join().unwrap();
        }
        total
    })
}

fn dot_if_range(ai: &[i64], bf: &[f64], an: i64, bn: i64, start: usize, end: usize) -> f64 {
    let mut sum = 0.0;
    if an == bn {
        let ai_slice = &ai[start..end];
        let bf_slice = &bf[start..end];
        for i in 0..ai_slice.len() {
            sum += (ai_slice[i] as f64) * bf_slice[i];
        }
    } else if an == 1 {
        let a0 = ai[0] as f64;
        let bf_slice = &bf[start..end];
        for i in 0..bf_slice.len() {
            sum += a0 * bf_slice[i];
        }
    } else {
        let b0 = bf[0];
        let ai_slice = &ai[start..end];
        for i in 0..ai_slice.len() {
            sum += (ai_slice[i] as f64) * b0;
        }
    }
    sum
}

fn dot_ii_parallel(ai: &[i64], bi: &[i64], an: i64, bn: i64, len: usize) -> i64 {
    let workers = parallel::worker_count(len);
    if workers == 1 {
        return dot_ii_range(ai, bi, an, bn, 0, len);
    }

    let chunk = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || {
                dot_ii_range(ai, bi, an, bn, start, end)
            }));
        }
        let mut total = 0;
        for h in handles {
            total += h.join().unwrap();
        }
        total
    })
}

fn dot_ii_range(ai: &[i64], bi: &[i64], an: i64, bn: i64, start: usize, end: usize) -> i64 {
    let mut sum = 0;
    if an == bn {
        let ai_slice = &ai[start..end];
        let bi_slice = &bi[start..end];
        for i in 0..ai_slice.len() {
            sum += ai_slice[i] * bi_slice[i];
        }
    } else if an == 1 {
        let a0 = ai[0];
        let bi_slice = &bi[start..end];
        for i in 0..bi_slice.len() {
            sum += a0 * bi_slice[i];
        }
    } else {
        let b0 = bi[0];
        let ai_slice = &ai[start..end];
        for i in 0..ai_slice.len() {
            sum += ai_slice[i] * b0;
        }
    }
    sum
}

fn simplify_list(v: Vec<K>) -> K {
    if v.is_empty() {
        return K::from_list(v);
    }
    let first_t = v[0].t;
    if first_t == -1 {
        if v.iter().all(|item| item.t == -1) {
            let ints = v.iter().map(|item| item.ki_data()[0]).collect();
            return K::from_ints(ints);
        }
    } else if first_t == -2 {
        if v.iter().all(|item| item.t == -2) {
            let floats = v.iter().map(|item| item.kf_data()[0]).collect();
            return K::from_floats(floats);
        }
    }
    K::from_list(v)
}

pub fn over(f: impl Fn(&K, &K) -> K, init: Option<&K>, x: &K) -> K {
    if x.t < 0 {
        match init {
            Some(y) => return f(y, x),
            None => return x.clone(),
        }
    }
    match &x.data {
        KData::Ints(v) => {
            match init {
                Some(y) => {
                    let mut acc = y.clone();
                    for &val in v {
                        acc = f(&acc, &K::ki(val));
                    }
                    acc
                }
                None => {
                    if v.is_empty() {
                        panic!("length error: over with no initial value expects a non-empty list");
                    }
                    let mut acc = K::ki(v[0]);
                    for &val in &v[1..] {
                        acc = f(&acc, &K::ki(val));
                    }
                    acc
                }
            }
        }
        KData::Floats(v) => {
            match init {
                Some(y) => {
                    let mut acc = y.clone();
                    for &val in v {
                        acc = f(&acc, &K::kf(val));
                    }
                    acc
                }
                None => {
                    if v.is_empty() {
                        panic!("length error: over with no initial value expects a non-empty list");
                    }
                    let mut acc = K::kf(v[0]);
                    for &val in &v[1..] {
                        acc = f(&acc, &K::kf(val));
                    }
                    acc
                }
            }
        }
        KData::List(v) => {
            match init {
                Some(y) => {
                    let mut acc = y.clone();
                    for item in v {
                        acc = f(&acc, item);
                    }
                    acc
                }
                None => {
                    if v.is_empty() {
                        panic!("length error: over with no initial value expects a non-empty list");
                    }
                    let mut acc = v[0].clone();
                    for item in &v[1..] {
                        acc = f(&acc, item);
                    }
                    acc
                }
            }
        }
    }
}

pub fn scan(f: impl Fn(&K, &K) -> K, init: Option<&K>, x: &K) -> K {
    if x.t < 0 {
        match init {
            Some(y) => return f(y, x),
            None => return x.clone(),
        }
    }
    match &x.data {
        KData::Ints(v) => {
            match init {
                Some(y) => {
                    let mut results = Vec::with_capacity(v.len());
                    let mut acc = y.clone();
                    for &val in v {
                        acc = f(&acc, &K::ki(val));
                        results.push(acc.clone());
                    }
                    simplify_list(results)
                }
                None => {
                    if v.is_empty() {
                        panic!("length error: scan with no initial value expects a non-empty list");
                    }
                    let mut results = Vec::with_capacity(v.len());
                    let mut acc = K::ki(v[0]);
                    results.push(acc.clone());
                    for &val in &v[1..] {
                        acc = f(&acc, &K::ki(val));
                        results.push(acc.clone());
                    }
                    simplify_list(results)
                }
            }
        }
        KData::Floats(v) => {
            match init {
                Some(y) => {
                    let mut results = Vec::with_capacity(v.len());
                    let mut acc = y.clone();
                    for &val in v {
                        acc = f(&acc, &K::kf(val));
                        results.push(acc.clone());
                    }
                    simplify_list(results)
                }
                None => {
                    if v.is_empty() {
                        panic!("length error: scan with no initial value expects a non-empty list");
                    }
                    let mut results = Vec::with_capacity(v.len());
                    let mut acc = K::kf(v[0]);
                    results.push(acc.clone());
                    for &val in &v[1..] {
                        acc = f(&acc, &K::kf(val));
                        results.push(acc.clone());
                    }
                    simplify_list(results)
                }
            }
        }
        KData::List(v) => {
            match init {
                Some(y) => {
                    let mut results = Vec::with_capacity(v.len());
                    let mut acc = y.clone();
                    for item in v {
                        acc = f(&acc, item);
                        results.push(acc.clone());
                    }
                    simplify_list(results)
                }
                None => {
                    if v.is_empty() {
                        panic!("length error: scan with no initial value expects a non-empty list");
                    }
                    let mut results = Vec::with_capacity(v.len());
                    let mut acc = v[0].clone();
                    results.push(acc.clone());
                    for item in &v[1..] {
                        acc = f(&acc, item);
                        results.push(acc.clone());
                    }
                    simplify_list(results)
                }
            }
        }
    }
}

pub fn each(f: impl Fn(&K) -> K, x: &K) -> K {
    if x.t < 0 {
        return f(x);
    }
    match &x.data {
        KData::Ints(v) => {
            let res: Vec<K> = v.iter().map(|&val| f(&K::ki(val))).collect();
            simplify_list(res)
        }
        KData::Floats(v) => {
            let res: Vec<K> = v.iter().map(|&val| f(&K::kf(val))).collect();
            simplify_list(res)
        }
        KData::List(v) => {
            let res: Vec<K> = v.iter().map(|item| f(item)).collect();
            simplify_list(res)
        }
    }
}

#[allow(dead_code)]
pub fn _2m(x: &K) -> K {
    if x.t != -1 {
        panic!("type error: monadic 2: expects integer function ID");
    }
    let fno = x.ki_data()[0];
    let valence = match fno {
        101 | 102 | 103 | 105 => 1,
        104 => 2,
        _ => 1, // Default valence
    };
    K::from_list(vec![x.clone(), K::ki(valence)])
}

#[allow(dead_code)]
pub fn _2d(a: &K, b: &K) -> K {
    let fno = match a.t {
        -1 => a.ki_data()[0],
        0 => {
            let items = a.kk_data();
            if items.is_empty() || items[0].t != -1 {
                panic!("type error: expected integer function ID in projection");
            }
            items[0].ki_data()[0]
        }
        _ => panic!("type error: dyadic 2: expects integer function ID or projection"),
    };

    match crate::ffi::cfuncs(fno, b) {
        Ok(res) => res,
        Err(msg) => panic!("FFI error: {}", msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_ints() {
        let x = K::from_ints(vec![1, 2, 2, 3, 1, 4, 3]);
        let res = unique(&x);
        assert_eq!(res.t, 1);
        assert_eq!(res.ki_data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_unique_floats() {
        let x = K::from_floats(vec![1.5, 2.5, 2.5000000000000001, f64::NAN, 3.5, 1.5, f64::NAN]);
        let res = unique(&x);
        assert_eq!(res.t, 2);
        let fd = res.kf_data();
        assert_eq!(fd.len(), 4);
        assert!((fd[0] - 1.5).abs() < 1e-15);
        assert!((fd[1] - 2.5).abs() < 1e-15);
        assert!(fd[2].is_nan());
        assert!((fd[3] - 3.5).abs() < 1e-15);
    }

    #[test]
    fn test_unique_list() {
        let x = K::from_list(vec![
            K::from_ints(vec![1, 2]),
            K::from_ints(vec![3, 4]),
            K::from_ints(vec![1, 2]),
            K::from_ints(vec![5]),
        ]);
        let res = unique(&x);
        assert_eq!(res.t, 0);
        let rd = res.kk_data();
        assert_eq!(rd.len(), 3);
        assert_eq!(rd[0].ki_data(), &[1, 2]);
        assert_eq!(rd[1].ki_data(), &[3, 4]);
        assert_eq!(rd[2].ki_data(), &[5]);
    }

    #[test]
    fn test_unique_atoms() {
        let x_int = K::ki(42);
        let res_int = unique(&x_int);
        assert_eq!(res_int.t, -1);
        assert_eq!(res_int.ki_data(), &[42]);

        let x_float = K::kf(3.14);
        let res_float = unique(&x_float);
        assert_eq!(res_float.t, -2);
        assert_eq!(res_float.kf_data(), &[3.14]);
    }

    #[test]
    fn test_first() {
        // Int atom
        let a = K::ki(42);
        let res = first(&a);
        assert_eq!(res.t, -1);
        assert_eq!(res.ki_data(), &[42]);

        // Int array
        let b = K::from_ints(vec![10, 20, 30]);
        let res = first(&b);
        assert_eq!(res.t, -1);
        assert_eq!(res.ki_data(), &[10]);

        // Float array
        let c = K::from_floats(vec![1.5, 2.5]);
        let res = first(&c);
        assert_eq!(res.t, -2);
        assert_eq!(res.kf_data(), &[1.5]);

        // List
        let d = K::from_list(vec![
            K::from_ints(vec![1, 2]),
            K::from_ints(vec![3, 4])
        ]);
        let res = first(&d);
        assert_eq!(res.t, 1);
        assert_eq!(res.ki_data(), &[1, 2]);
    }

    #[test]
    fn test_reverse() {
        // Atom
        let a = K::ki(42);
        let res = reverse(&a);
        assert_eq!(res.t, -1);
        assert_eq!(res.ki_data(), &[42]);

        // Int array
        let b = K::from_ints(vec![1, 2, 3]);
        let res = reverse(&b);
        assert_eq!(res.t, 1);
        assert_eq!(res.ki_data(), &[3, 2, 1]);

        // Float array
        let c = K::from_floats(vec![1.5, 2.5]);
        let res = reverse(&c);
        assert_eq!(res.t, 2);
        assert_eq!(res.kf_data(), &[2.5, 1.5]);

        // List
        let d = K::from_list(vec![
            K::from_ints(vec![1, 2]),
            K::from_ints(vec![3, 4])
        ]);
        let res = reverse(&d);
        assert_eq!(res.t, 0);
        let rd = res.kk_data();
        assert_eq!(rd.len(), 2);
        assert_eq!(rd[0].ki_data(), &[3, 4]);
        assert_eq!(rd[1].ki_data(), &[1, 2]);
    }

    #[test]
    fn test_flip() {
        // Int list of lists
        let row1 = K::from_ints(vec![1, 2, 3]);
        let row2 = K::from_ints(vec![4, 5, 6]);
        let mat = K::from_list(vec![row1, row2]);
        let flipped = flip(&mat);
        assert_eq!(flipped.t, 0);
        let rows = flipped.kk_data();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].ki_data(), &[1, 4]);
        assert_eq!(rows[1].ki_data(), &[2, 5]);
        assert_eq!(rows[2].ki_data(), &[3, 6]);
    }

    #[test]
    fn test_over() {
        let x = K::from_ints(vec![1, 2, 3]);
        let res1 = over(plus, None, &x);
        assert_eq!(res1.t, -1);
        assert_eq!(res1.ki_data(), &[6]);

        let res2 = over(plus, Some(&K::ki(10)), &x);
        assert_eq!(res2.t, -1);
        assert_eq!(res2.ki_data(), &[16]);

        let x_f = K::from_floats(vec![1.0, 2.0, 3.0, 4.0]);
        let res3 = over(times, None, &x_f);
        assert_eq!(res3.t, -2);
        assert!((res3.kf_data()[0] - 24.0).abs() < 1e-15);
    }

    #[test]
    fn test_scan() {
        let x = K::from_ints(vec![1, 2, 3]);
        let res1 = scan(plus, None, &x);
        assert_eq!(res1.t, 1);
        assert_eq!(res1.ki_data(), &[1, 3, 6]);

        let res2 = scan(plus, Some(&K::ki(10)), &x);
        assert_eq!(res2.t, 1);
        assert_eq!(res2.ki_data(), &[11, 13, 16]);
    }

    #[test]
    fn test_each() {
        let x = K::from_ints(vec![1, 2, 3]);
        let res1 = each(negate, &x);
        assert_eq!(res1.t, 1);
        assert_eq!(res1.ki_data(), &[-1, -2, -3]);

        let x_f = K::from_floats(vec![1.0, 2.0, 4.0]);
        let res2 = each(reciprocal, &x_f);
        assert_eq!(res2.t, 2);
        assert!((res2.kf_data()[0] - 1.0).abs() < 1e-15);
        assert!((res2.kf_data()[1] - 0.5).abs() < 1e-15);
        assert!((res2.kf_data()[2] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn test_ffi_sin_cos_log() {
        // Test monadic _2m (projection)
        let fno = K::ki(101);
        let proj = _2m(&fno);
        assert_eq!(proj.t, 0);
        let p_data = proj.kk_data();
        assert_eq!(p_data.len(), 2);
        assert_eq!(p_data[0].ki_data()[0], 101);
        assert_eq!(p_data[1].ki_data()[0], 1); // Valence of sin is 1

        // Test dyadic _2d with sin (101) on atom
        let res_atom = _2d(&fno, &K::kf(0.0));
        assert_eq!(res_atom.t, -2);
        assert!((res_atom.kf_data()[0] - 0.0).abs() < 1e-15);

        // Test dyadic _2d with sin on array
        let x_arr = K::from_floats(vec![0.0, std::f64::consts::FRAC_PI_2]);
        let res_arr = _2d(&fno, &x_arr);
        assert_eq!(res_arr.t, 2);
        assert!((res_arr.kf_data()[0] - 0.0).abs() < 1e-15);
        assert!((res_arr.kf_data()[1] - 1.0).abs() < 1e-15);

        // Test cos (102) and log (103)
        let cos_res = _2d(&K::ki(102), &K::kf(0.0));
        assert!((cos_res.kf_data()[0] - 1.0).abs() < 1e-15);

        let log_res = _2d(&K::ki(103), &K::kf(std::f64::consts::E));
        assert!((log_res.kf_data()[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_ffi_add_two_sum_all() {
        // Test add_two (104) which takes 2 arguments
        let args = K::from_list(vec![K::ki(10), K::ki(20)]);
        let res = _2d(&K::ki(104), &args);
        assert_eq!(res.t, -1);
        assert_eq!(res.ki_data()[0], 30);

        // Test sum_all (105)
        let arr = K::from_ints(vec![1, 2, 3, 4]);
        let sum_res = _2d(&K::ki(105), &arr);
        assert_eq!(sum_res.t, -1);
        assert_eq!(sum_res.ki_data()[0], 10);
    }

    #[test]
    #[should_panic(expected = "FFI error: arity error: sin expects 1 argument")]
    fn test_ffi_failures() {
        // Calling sin (101) with 2 arguments should fail/panic
        let args = K::from_list(vec![K::kf(1.0), K::kf(2.0)]);
        _2d(&K::ki(101), &args);
    }
}


