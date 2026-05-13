//! Small parallel chunk executor for Kona-style primitives.
//!
//! This keeps the useful scheduler idea from `proc.rs`: split runnable work into
//! chunks and hand those chunks to workers. It avoids the unsafe Go-runtime port
//! shape and keeps primitive semantics in `va.rs`.

use std::thread;

const MIN_PARALLEL_LEN: usize = 16_384;

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
        return binary_f64_range(a, b, an, bn, 0, len, op);
    }

    let chunk = len.div_ceil(workers);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);

        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || binary_f64_range(a, b, an, bn, start, end, op)));
        }

        let mut out = Vec::with_capacity(len);
        for handle in handles {
            out.extend(handle.join().expect("parallel f64 worker panicked"));
        }
        out
    })
}

pub fn binary_i64<F>(a: &[i64], b: &[i64], an: i64, bn: i64, zn: i64, op: F) -> Vec<i64>
where
    F: Fn(i64, i64) -> i64 + Copy + Send + Sync,
{
    let len = zn as usize;
    let workers = worker_count(len);

    if workers == 1 {
        return binary_i64_range(a, b, an, bn, 0, len, op);
    }

    let chunk = len.div_ceil(workers);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);

        for start in (0..len).step_by(chunk) {
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || binary_i64_range(a, b, an, bn, start, end, op)));
        }

        let mut out = Vec::with_capacity(len);
        for handle in handles {
            out.extend(handle.join().expect("parallel i64 worker panicked"));
        }
        out
    })
}

fn binary_f64_range<F>(
    a: &[f64],
    b: &[f64],
    an: i64,
    bn: i64,
    start: usize,
    end: usize,
    op: F,
) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    let mut out = Vec::with_capacity(end - start);

    for i in start..end {
        let x = if an == 1 { a[0] } else { a[i] };
        let y = if bn == 1 { b[0] } else { b[i] };
        out.push(op(x, y));
    }

    out
}

fn binary_i64_range<F>(
    a: &[i64],
    b: &[i64],
    an: i64,
    bn: i64,
    start: usize,
    end: usize,
    op: F,
) -> Vec<i64>
where
    F: Fn(i64, i64) -> i64 + Copy,
{
    let mut out = Vec::with_capacity(end - start);

    for i in start..end {
        let x = if an == 1 { a[0] } else { a[i] };
        let y = if bn == 1 { b[0] } else { b[i] };
        out.push(op(x, y));
    }

    out
}
