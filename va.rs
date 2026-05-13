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
    if zt.abs() == 1 {
        // Promote to float (2/-2)
        zt = if zt < 0 { -2 } else { 2 };
    }

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

    // Integer and float accumulators — exactly like the C code
    let mut acc_i: i64 = 0;
    let mut acc_f: f64 = 0.0;

    // ---------------------------------------------------------------
    // SCALAR_EXPR_CASE — differs from SCALAR_OP_CASE in that it
    // accumulates into a scalar instead of writing to an output array.
    //
    // C pattern:
    //   SCALAR_EXPR_CASE(DOT_F, F, kF(a), kF(b), x, y)
    // expands to (approximately):
    //   if(an==bn)     DO(zn, x=kF(a)[i]; y=kF(b)[i]; accF+=x*y)
    //   else if(an==1) DO(zn, x=kF(a)[0]; y=kF(b)[i]; accF+=x*y)
    //   else           DO(zn, x=kF(a)[i]; y=kF(b)[0]; accF+=x*y)
    // ---------------------------------------------------------------

    if abs_a == 2 && abs_b == 2 {
        // float × float → float accumulator
        // DOT_F: accF += x * y
        let af = a.kf_data();
        let bf = b.kf_data();
        let n = s.zn as usize;

        if s.an == s.bn {
            for i in 0..n {
                acc_f += af[i] * bf[i]; // DOT_F
            }
        } else if s.an == 1 {
            let a0 = af[0];
            for i in 0..n {
                acc_f += a0 * bf[i]; // DOT_F
            }
        } else {
            let b0 = bf[0];
            for i in 0..n {
                acc_f += af[i] * b0; // DOT_F
            }
        }
        K::kf(acc_f)
    } else if abs_a == 2 && abs_b == 1 {
        // float × int → float accumulator
        // DOT_FI: accF += x * I2F(y)
        let af = a.kf_data();
        let bi = b.ki_data();
        let n = s.zn as usize;

        if s.an == s.bn {
            for i in 0..n {
                acc_f += af[i] * K::i2f(bi[i]); // DOT_FI
            }
        } else if s.an == 1 {
            let a0 = af[0];
            for i in 0..n {
                acc_f += a0 * K::i2f(bi[i]); // DOT_FI
            }
        } else {
            let b0 = K::i2f(bi[0]);
            for i in 0..n {
                acc_f += af[i] * b0; // DOT_FI
            }
        }
        K::kf(acc_f)
    } else if abs_a == 1 && abs_b == 2 {
        // int × float → float accumulator
        // DOT_IF: accF += I2F(x) * y
        let ai = a.ki_data();
        let bf = b.kf_data();
        let n = s.zn as usize;

        if s.an == s.bn {
            for i in 0..n {
                acc_f += K::i2f(ai[i]) * bf[i]; // DOT_IF
            }
        } else if s.an == 1 {
            let a0 = K::i2f(ai[0]);
            for i in 0..n {
                acc_f += a0 * bf[i]; // DOT_IF
            }
        } else {
            let b0 = bf[0];
            for i in 0..n {
                acc_f += K::i2f(ai[i]) * b0; // DOT_IF
            }
        }
        K::kf(acc_f)
    } else if abs_a == 1 && abs_b == 1 {
        // int × int → int accumulator
        // DOT_I: accI += x * y
        let ai = a.ki_data();
        let bi = b.ki_data();
        let n = s.zn as usize;

        if s.an == s.bn {
            for i in 0..n {
                acc_i += ai[i] * bi[i]; // DOT_I
            }
        } else if s.an == 1 {
            let a0 = ai[0];
            for i in 0..n {
                acc_i += a0 * bi[i]; // DOT_I
            }
        } else {
            let b0 = bi[0];
            for i in 0..n {
                acc_i += ai[i] * b0; // DOT_I
            }
        }
        K::ki(acc_i)
    } else if abs_a == 0 || abs_b == 0 {
        // General list fallback:
        //   C: y = overDyad(0, p+2, (x = times(a,b)));
        //   This is: +/ times(a,b) — multiply then sum
        let product = times(a, b);
        // Sum the product array (overDyad with plus)
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

    if k != b_k {
        panic!(
            "matmul: dimension mismatch: A[{}, {}] x B[{}, {}]",
            m, k, b_k, n
        );
    }

    // Naive MatMul: C[i][j] = sum(A[i][k] * B[k][j])
    // This is slow (O(MLN) with bad locality).
    // Better: Transpose B first? Or just iterate.
    // Given we are in Rust, let's just do the naive loops for MVP.
    // Optimization: Blocked loop later.

    // Provide B as columns for faster access?
    // Actually, accessing B[k][j] repeatedly is bad.
    // Let's transpose B into B_cols: Vec<Vec<f64>>.
    // B is [K rows of N floats].
    // B_cols should be [N columns of K floats].
    let mut b_cols = vec![Vec::with_capacity(k); n];
    for r in 0..k {
        let row = b_rows[r].kf_data(); // Assume float matrix for NN
        for c in 0..n {
            b_cols[c].push(row[c]);
        }
    }

    // Parallel MatMul using std::thread::scope (No external deps!)
    // C[i] = A[i] . B_cols

    // Determine # threads
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = (m + num_threads - 1) / num_threads;

    // Pre-allocate result vector with placeholders (unsafe or Option?)
    // Safe way: collect from threads.

    let c_rows = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);

        for chunk in a_rows.chunks(chunk_size) {
            let b_cols_ref = &b_cols; // Shared ref

            handles.push(s.spawn(move || {
                let mut chunk_results = Vec::with_capacity(chunk.len());
                for row_k in chunk {
                    let row_a = row_k.kf_data();
                    let mut row_c = Vec::with_capacity(n);
                    for j in 0..n {
                        let col_b = &b_cols_ref[j];
                        let mut sum = 0.0;
                        for x in 0..k {
                            sum += row_a[x] * col_b[x];
                        }
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
    let n = rows[0].n as usize;

    let mut cols = vec![Vec::with_capacity(m); n];

    for i in 0..m {
        if let KData::Floats(row) = &rows[i].data {
            for j in 0..n {
                cols[j].push(row[j]);
            }
        } else {
            panic!("transpose: expected float matrix");
        }
    }

    let mut new_rows = Vec::with_capacity(n);
    for col in cols {
        new_rows.push(K::from_floats(col));
    }
    K::from_list(new_rows)
}

/// Sigmoid Activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res: Vec<f64> = v.iter().map(|&val| 1.0 / (1.0 + (-val).exp())).collect();
            K::from_floats(res)
        }
        KData::List(rows) => {
            let new_rows: Vec<K> = rows.iter().map(|row| sigmoid(row)).collect();
            K::from_list(new_rows)
        }
        _ => panic!("sigmoid: expected float array or matrix"),
    }
}

/// Tanh Activation
pub fn tanh(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res: Vec<f64> = v.iter().map(|&val| val.tanh()).collect();
            K::from_floats(res)
        }
        KData::List(rows) => {
            let new_rows: Vec<K> = rows.iter().map(|row| tanh(row)).collect();
            K::from_list(new_rows)
        }
        _ => panic!("tanh: expected float array or matrix"),
    }
}

/// ReLU Activation: max(0, x)
pub fn relu(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            let res: Vec<f64> = v
                .iter()
                .map(|&val| if val > 0.0 { val } else { 0.0 })
                .collect();
            K::from_floats(res)
        }
        KData::List(rows) => {
            let new_rows: Vec<K> = rows.iter().map(|row| relu(row)).collect();
            K::from_list(new_rows)
        }
        _ => panic!("relu: expected float array or matrix"),
    }
}

/// Softmax: exp(x) / sum(exp(x)) (Row-wise if matrix, or global if vector)
pub fn softmax(x: &K) -> K {
    match &x.data {
        KData::Floats(v) => {
            // Stability: subtract max
            let max_val = v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f64> = v.iter().map(|&val| (val - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let res: Vec<f64> = exps.iter().map(|&val| val / sum).collect();
            K::from_floats(res)
        }
        KData::List(rows) => {
            // Apply softmax row-wise
            let new_rows: Vec<K> = rows.iter().map(|row| softmax(row)).collect();
            K::from_list(new_rows)
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

            for row in rows {
                let rf = row.kf_data();
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
            let res: Vec<f64> = v
                .iter()
                .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                .collect();
            K::from_floats(res)
        }
        KData::List(rows) => {
            let new_rows: Vec<K> = rows.iter().map(|row| relu_derivative(row)).collect();
            K::from_list(new_rows)
        }
        _ => panic!("relu_derivative: expected float/matrix"),
    }
}

/// Sigmoid Backward: sig(x) * (1 - sig(x)) * dy
pub fn sigmoid_backward(x: &K, dy: &K) -> K {
    // We compute sigmoid(x) first
    let s = sigmoid(x);
    // (1 - s)
    // We need 1.0 broadcasted or map.
    // Let's do it manually element-wise for speed/simplicity

    // We need to iterate s and dy in parallel (Hadamard 3-way)
    // This is specialized.
    // TODO: implement generalized 'map3'?
    // For now, let's just implement it recursively.

    match (&s.data, &dy.data) {
        (KData::Floats(sv), KData::Floats(dv)) => {
            let n = sv.len();
            let mut res = Vec::with_capacity(n);
            for i in 0..n {
                let sig = sv[i];
                // dSig = sig * (1 - sig)
                let d_sig = sig * (1.0 - sig);
                res.push(d_sig * dv[i]);
            }
            K::from_floats(res)
        }
        (KData::List(sr), KData::List(dr)) => {
            let new_rows = sr
                .iter()
                .zip(dr.iter())
                .map(|(s_row, d_row)| {
                    // We need to pass original x, but we already computed s=sigmoid(x).
                    // So we need a helper that takes s, dy.
                    // But the signature is (x, dy).
                    // Wait, if I use s here, I am recursing incorrectly if I call sigmoid_backward again.
                    // I should make a helper 'sigmoid_backward_from_output(s, dy)'.
                    // But to keep signature 'sigmoid_backward(x, dy)', I must recompute s inside.
                    // Actually, 's' is already computed at top level.
                    // If I recurse `sigmoid_backward`, it will recompute sig(x). That's fine.
                    sigmoid_backward_inner(s_row, d_row)
                })
                .collect();
            K::from_list(new_rows)
        }
        _ => panic!("sigmoid_backward: shape mismatch"),
    }
}

// Helper to avoid re-computing sigmoid(sigmoid(x))
fn sigmoid_backward_inner(s: &K, dy: &K) -> K {
    match (&s.data, &dy.data) {
        (KData::Floats(sv), KData::Floats(dv)) => {
            let n = sv.len();
            let mut res = Vec::with_capacity(n);
            for i in 0..n {
                let sig = sv[i];
                res.push(sig * (1.0 - sig) * dv[i]);
            }
            K::from_floats(res)
        }
        (KData::List(sr), KData::List(dr)) => {
            let new_rows: Vec<K> = sr
                .iter()
                .zip(dr.iter())
                .map(|(r1, r2)| sigmoid_backward_inner(r1, r2))
                .collect();
            K::from_list(new_rows)
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
            let n = tv.len();
            let mut res = Vec::with_capacity(n);
            for i in 0..n {
                let tah = tv[i];
                // dt = 1 - tanh^2
                let d_t = 1.0 - (tah * tah);
                res.push(d_t * dv[i]);
            }
            K::from_floats(res)
        }
        (KData::List(tr), KData::List(dr)) => {
            let new_rows: Vec<K> = tr
                .iter()
                .zip(dr.iter())
                .map(|(r1, r2)| tanh_backward_inner(r1, r2))
                .collect();
            K::from_list(new_rows)
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
