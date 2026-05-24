//! Small parallel chunk executor for Kona-style primitives.
//!
//! This keeps the useful scheduler idea from `proc.rs`: split runnable work into
//! chunks and hand those chunks to workers. It avoids the complex Go-runtime port
//! shape and keeps primitive semantics in `va.rs`.

use std::thread;

const MIN_PARALLEL_LEN: usize = 131_072; // Increased to reduce thread overhead

pub fn worker_count(len: usize) -> usize {
    if len < MIN_PARALLEL_LEN {
        return 1;
    }

    let cpus = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    cpus.min(len)
}

pub fn binary_f64<F>(a: &[f64], b: &[f64], an: i64, bn: i64, zn: i64, op: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Copy + Send + Sync,
{
    let len = zn as usize;
    let workers = worker_count(len);

    if workers == 1 {
        let mut out = vec![0.0; len];
        binary_f64_range_mut(a, b, an, bn, 0, &mut out, op);
        return out;
    }

    let mut out = vec![0.0; len];
    let chunk = len.div_ceil(workers);

    thread::scope(|scope| {
        let mut chunks = out.chunks_mut(chunk);
        for i in 0..workers {
            let start = i * chunk;
            if let Some(out_chunk) = chunks.next() {
                scope.spawn(move || {
                    binary_f64_range_mut(a, b, an, bn, start, out_chunk, op);
                });
            }
        }
    });

    out
}

pub fn binary_i64<F>(a: &[i64], b: &[i64], an: i64, bn: i64, zn: i64, op: F) -> Vec<i64>
where
    F: Fn(i64, i64) -> i64 + Copy + Send + Sync,
{
    let len = zn as usize;
    let workers = worker_count(len);

    if workers == 1 {
        let mut out = vec![0; len];
        binary_i64_range_mut(a, b, an, bn, 0, &mut out, op);
        return out;
    }

    let mut out = vec![0; len];
    let chunk = len.div_ceil(workers);

    thread::scope(|scope| {
        let mut chunks = out.chunks_mut(chunk);
        for i in 0..workers {
            let start = i * chunk;
            if let Some(out_chunk) = chunks.next() {
                scope.spawn(move || {
                    binary_i64_range_mut(a, b, an, bn, start, out_chunk, op);
                });
            }
        }
    });

    out
}

fn binary_f64_range_mut<F>(
    a: &[f64],
    b: &[f64],
    an: i64,
    bn: i64,
    start: usize,
    out_chunk: &mut [f64],
    op: F,
) where
    F: Fn(f64, f64) -> f64 + Copy,
{
    let len = out_chunk.len();
    if an == 1 && bn == 1 {
        let x = a[0];
        let y = b[0];
        let val = op(x, y);
        for i in 0..len {
            out_chunk[i] = val;
        }
    } else if an == 1 {
        let x = a[0];
        let b_slice = &b[start..start + len];
        for i in 0..len {
            out_chunk[i] = op(x, b_slice[i]);
        }
    } else if bn == 1 {
        let y = b[0];
        let a_slice = &a[start..start + len];
        for i in 0..len {
            out_chunk[i] = op(a_slice[i], y);
        }
    } else {
        let a_slice = &a[start..start + len];
        let b_slice = &b[start..start + len];
        for i in 0..len {
            out_chunk[i] = op(a_slice[i], b_slice[i]);
        }
    }
}

fn binary_i64_range_mut<F>(
    a: &[i64],
    b: &[i64],
    an: i64,
    bn: i64,
    start: usize,
    out_chunk: &mut [i64],
    op: F,
) where
    F: Fn(i64, i64) -> i64 + Copy,
{
    let len = out_chunk.len();
    if an == 1 && bn == 1 {
        let x = a[0];
        let y = b[0];
        let val = op(x, y);
        for i in 0..len {
            out_chunk[i] = val;
        }
    } else if an == 1 {
        let x = a[0];
        let b_slice = &b[start..start + len];
        for i in 0..len {
            out_chunk[i] = op(x, b_slice[i]);
        }
    } else if bn == 1 {
        let y = b[0];
        let a_slice = &a[start..start + len];
        for i in 0..len {
            out_chunk[i] = op(a_slice[i], y);
        }
    } else {
        let a_slice = &a[start..start + len];
        let b_slice = &b[start..start + len];
        for i in 0..len {
            out_chunk[i] = op(a_slice[i], b_slice[i]);
        }
    }
}
