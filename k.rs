//! Kona K3 object system translated from C to Rust.
//!
//! Original C source: https://github.com/kevinlawler/kona
//! Author: Kevin Lawler
//!
//! The K object (struct k0 in C) is the fundamental data type in K/Kona.
//! It holds a type tag, element count, and data array.
//!
//! C definition (from ts.h):
//!   typedef struct k0{I _c,t,n;struct k0*k[1];}*K;
//!
//! Type tags (t field):
//!   0 = general list (list of K objects)
//!   1 = int array     (I = i64)
//!   2 = float array   (F = f64)
//!   3 = char array    (C = u8)
//!   4 = symbol array  (S = string)
//!  -1 = int atom
//!  -2 = float atom
//!  -3 = char atom
//!  Negative = atom (single value), Positive = array

use std::fmt;

/// K type tags — mirrors Kona's t field in struct k0
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KType {
    /// 0: General list of K objects
    List,
    /// 1: Integer array (i64 elements)
    IntArray,
    /// 2: Float array (f64 elements)
    FloatArray,
    /// 3: Char array
    CharArray,
}

/// The K object — Kona's universal data container.
///
/// C original:
///   typedef struct k0{I _c,t,n;struct k0*k[1];}*K;
///
/// _c = reference count
///  t = type tag
///  n = element count
///  k = flexible array member (data lives here)
#[derive(Clone)]
pub struct K {
    /// Type tag. Matches Kona's t field.
    /// 1 = int array, 2 = float array, -1 = int atom, -2 = float atom, 0 = general list
    pub t: i64,
    /// Number of elements. Matches Kona's n field.
    pub n: i64,
    /// The actual data — using Rust enum instead of C's void* cast tricks.
    pub data: KData,
}

/// Data payload for a K object.
///
/// In C, Kona uses kI(x) = (I*)ke(x), kF(x) = (F*)ke(x) etc.
/// to cast the flexible array member k[] to the right pointer type.
/// In Rust, we use an enum to hold typed data safely.
#[derive(Clone)]
pub enum KData {
    /// kI(x) — integer data (I = i64 in Kona)
    Ints(Vec<i64>),
    /// kF(x) — float data (F = f64 in Kona)
    Floats(Vec<f64>),
    /// kK(x) — general list of K objects
    #[allow(dead_code)]
    List(Vec<K>),
}

impl K {
    // ===================================================================
    // Constructors — equivalent to newK(t, n), Ki(x), Kf(x) in Kona
    // ===================================================================

    /// Create an integer array K object.
    /// C equivalent: newK(1, n) then fill kI(z)
    #[allow(dead_code)]
    pub fn from_ints(v: Vec<i64>) -> K {
        let n = v.len() as i64;
        K {
            t: 1, // Always array type — use ki() for atoms
            n,
            data: KData::Ints(v),
        }
    }

    /// Create a float array K object.
    /// C equivalent: newK(2, n) then fill kF(z)
    pub fn from_floats(v: Vec<f64>) -> K {
        let n = v.len() as i64;
        K {
            t: 2, // Always array type — use kf() for atoms
            n,
            data: KData::Floats(v),
        }
    }

    /// Create an integer atom.
    /// C equivalent: Ki(x) — from km.h
    pub fn ki(x: i64) -> K {
        K {
            t: -1,
            n: 1,
            data: KData::Ints(vec![x]),
        }
    }

    /// Create a float atom.
    /// C equivalent: Kf(x) — from km.h
    pub fn kf(x: f64) -> K {
        K {
            t: -2,
            n: 1,
            data: KData::Floats(vec![x]),
        }
    }

    /// Create a general list.
    /// C equivalent: newK(0, n)
    #[allow(dead_code)]
    pub fn from_list(v: Vec<K>) -> K {
        let n = v.len() as i64;
        K {
            t: 0,
            n,
            data: KData::List(v),
        }
    }

    // ===================================================================
    // Accessors — equivalent to kI(x), kF(x), kK(x) macros in Kona
    //
    // C originals (from d.h):
    //   #define ke(x) (((K)x)->k)
    //   #define kI(x) ((I*)ke(x))
    //   #define kF(x) ((F*)ke(x))
    //   #define kK(x) ((K*)ke(x))
    // ===================================================================

    /// Get integer data slice.
    /// C equivalent: kI(x) — returns I* pointer to int data
    pub fn ki_data(&self) -> &[i64] {
        match &self.data {
            KData::Ints(v) => v,
            _ => panic!("type error: expected int data, got t={}", self.t),
        }
    }

    /// Get float data slice.
    /// C equivalent: kF(x) — returns F* pointer to float data
    pub fn kf_data(&self) -> &[f64] {
        match &self.data {
            KData::Floats(v) => v,
            _ => panic!("type error: expected float data, got t={}", self.t),
        }
    }

    /// Get list data slice.
    /// C equivalent: kK(x) — returns K* pointer to K object array
    #[allow(dead_code)]
    pub fn kk_data(&self) -> &[K] {
        match &self.data {
            KData::List(v) => v,
            _ => panic!("type error: expected list data, got t={}", self.t),
        }
    }

    /// Get abs(t) — used constantly in Kona for type dispatch.
    /// C: ABS(at) where at = a->t
    #[allow(dead_code)]
    pub fn abs_t(&self) -> i64 {
        self.t.abs()
    }

    /// Check if this is an int type (atom or array).
    #[allow(dead_code)]
    pub fn is_int(&self) -> bool {
        self.abs_t() == 1
    }

    /// Check if this is a float type (atom or array).
    #[allow(dead_code)]
    pub fn is_float(&self) -> bool {
        self.abs_t() == 2
    }

    /// Check if this is a general list.
    #[allow(dead_code)]
    pub fn is_list(&self) -> bool {
        self.t == 0
    }

    /// Convert int data to f64.
    /// C equivalent: I2F(x) — used in mixed-type operations
    pub fn i2f(x: i64) -> f64 {
        x as f64
    }
}

// ===================================================================
// Debug/Display — so we can see what's inside K objects
// ===================================================================

impl fmt::Debug for K {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.data {
            KData::Ints(v) => {
                if v.len() == 1 {
                    write!(f, "{}", v[0])
                } else {
                    write!(f, "{:?}", v)
                }
            }
            KData::Floats(v) => {
                if v.len() == 1 {
                    write!(f, "{}", v[0])
                } else {
                    write!(f, "{:?}", v)
                }
            }
            KData::List(v) => {
                write!(f, "(")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ";")?;
                    }
                    write!(f, "{:?}", item)?;
                }
                write!(f, ")")
            }
        }
    }
}
