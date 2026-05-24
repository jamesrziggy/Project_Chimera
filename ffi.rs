//! Safe BCPL-style FFI registry and cfuncs dispatcher.
//!
//! Provides a safe registry where integer function IDs (`fno`) map to Rust
//! implementations. Centralizes type validation to ensure foolproof execution.

use crate::k::{K, KData};
use crate::va;

/// Unpacks argument K object. If it is a general list (type 0), returns its
/// elements. Otherwise, returns a single-item vector containing the object.
pub fn unpack_args(args: &K) -> Vec<K> {
    if args.t == 0 {
        args.kk_data().to_vec()
    } else {
        vec![args.clone()]
    }
}

/// Helper to apply a monadic f64 function element-wise on K.
/// Supports atoms, arrays of ints/floats, and lists recursively.
fn apply_monadic_f64<F>(x: &K, op: F) -> Result<K, String>
where
    F: Fn(f64) -> f64 + Copy,
{
    match x.t {
        -1 => {
            // Int atom -> Float atom
            let val = x.ki_data()[0];
            Ok(K::kf(op(val as f64)))
        }
        -2 => {
            // Float atom
            let val = x.kf_data()[0];
            Ok(K::kf(op(val)))
        }
        1 => {
            // Int array -> Float array
            let res: Vec<f64> = x.ki_data().iter().map(|&val| op(val as f64)).collect();
            Ok(K::from_floats(res))
        }
        2 => {
            // Float array
            let res: Vec<f64> = x.kf_data().iter().map(|&val| op(val)).collect();
            Ok(K::from_floats(res))
        }
        0 => {
            // General list -> recurse element-wise
            let mut res = Vec::new();
            for item in x.kk_data() {
                res.push(apply_monadic_f64(item, op)?);
            }
            Ok(K::from_list(res))
        }
        _ => Err(format!("type error: expected numeric data, got tag {}", x.t)),
    }
}

/// Pre-validation helper to verify that two K objects are compatible for addition
/// without triggering internal panics in va::plus.
fn can_add(a: &K, b: &K) -> Result<(), String> {
    let at = a.t;
    let bt = b.t;
    let an = a.n;
    let bn = b.n;

    // Supported tags are int, float, list (1, 2, -1, -2, 0)
    if !matches!(at, -2 | -1 | 0 | 1 | 2) || !matches!(bt, -2 | -1 | 0 | 1 | 2) {
        return Err(format!("type error: unsupported type for FFI addition: at={}, bt={}", at, bt));
    }

    // Check list dimensions if both are lists/arrays
    if an != bn && an != 1 && bn != 1 {
        return Err(format!("length error: incompatible shapes for FFI addition: {} vs {}", an, bn));
    }

    if at == 0 || bt == 0 {
        match (&a.data, &b.data) {
            (KData::List(la), _) if at == 0 && bt != 0 => {
                for item in la { can_add(item, b)?; }
            }
            (_, KData::List(lb)) if at != 0 && bt == 0 => {
                for item in lb { can_add(a, item)?; }
            }
            (KData::List(la), KData::List(lb)) => {
                if la.len() != lb.len() {
                    return Err(format!("length error: list elements mismatch {} vs {}", la.len(), lb.len()));
                }
                for (item_a, item_b) in la.iter().zip(lb.iter()) {
                    can_add(item_a, item_b)?;
                }
            }
            _ => return Err("corrupt K object: tag is list but data variant mismatch".to_string()),
        }
    }
    Ok(())
}

/// BCPL-style FFI dispatcher. Matches function codes to safe Rust handlers.
/// Checks arity and parameters, returning a Result to prevent panics.
pub fn cfuncs(fno: i64, args: &K) -> Result<K, String> {
    let unpacked = unpack_args(args);

    match fno {
        // 101: sin(x)
        101 => {
            if unpacked.len() != 1 {
                return Err(format!("arity error: sin expects 1 argument, got {}", unpacked.len()));
            }
            let x = &unpacked[0];
            apply_monadic_f64(x, f64::sin)
        }

        // 102: cos(x)
        102 => {
            if unpacked.len() != 1 {
                return Err(format!("arity error: cos expects 1 argument, got {}", unpacked.len()));
            }
            let x = &unpacked[0];
            apply_monadic_f64(x, f64::cos)
        }

        // 103: log(x) (natural logarithm)
        103 => {
            if unpacked.len() != 1 {
                return Err(format!("arity error: log expects 1 argument, got {}", unpacked.len()));
            }
            let x = &unpacked[0];
            apply_monadic_f64(x, f64::ln)
        }

        // 104: add_two(x, y)
        104 => {
            if unpacked.len() != 2 {
                return Err(format!("arity error: add_two expects 2 arguments, got {}", unpacked.len()));
            }
            can_add(&unpacked[0], &unpacked[1])?;
            // Delegates safely to va::plus primitive
            Ok(va::plus(&unpacked[0], &unpacked[1]))
        }

        // 105: sum_all(x) (sum reduction)
        105 => {
            if unpacked.len() != 1 {
                return Err(format!("arity error: sum_all expects 1 argument, got {}", unpacked.len()));
            }
            let x = &unpacked[0];
            match x.t {
                -1 | -2 => Ok(x.clone()), // Atom sum is the atom itself
                1 => {
                    let sum: i64 = x.ki_data().iter().sum();
                    Ok(K::ki(sum))
                }
                2 => {
                    let sum: f64 = x.kf_data().iter().sum();
                    Ok(K::kf(sum))
                }
                0 => {
                    // Try to sum elements recursively using va::plus
                    let items = x.kk_data();
                    if items.is_empty() {
                        return Ok(K::ki(0));
                    }
                    let mut acc = items[0].clone();
                    for item in &items[1..] {
                        can_add(&acc, item)?;
                        acc = va::plus(&acc, item);
                    }
                    Ok(acc)
                }
                _ => Err(format!("type error: sum_all expects numeric list/atom, got tag {}", x.t)),
            }
        }

        _ => Err(format!("value error: unknown foreign function ID {}", fno)),
    }
}
