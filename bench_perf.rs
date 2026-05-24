mod k;
mod parallel;
mod primitives;
mod va;
mod ffi;

use k::K;
use std::time::Instant;

fn main() {
    println!("=== Project Chimera CPU Optimization Benchmark ===");

    // 1. Benchmark Dot Product
    {
        let size = 10_000_000;
        let a = K::from_floats(vec![0.5; size]);
        let b = K::from_floats(vec![2.0; size]);

        let start = Instant::now();
        let _res = va::dot(&a, &b);
        let duration = start.elapsed();
        println!("dot product (10M floats): {:?}", duration);
    }

    // 2. Benchmark Parallel Scheduler (element-wise multiplication)
    {
        let size = 10_000_000;
        let a = K::from_floats(vec![1.5; size]);
        let b = K::from_floats(vec![2.0; size]);

        let start = Instant::now();
        let _res = va::times(&a, &b);
        let duration = start.elapsed();
        println!("parallel times (10M floats): {:?}", duration);
    }

    // 3. Benchmark MatMul
    {
        // 256 x 256 matrix multiplication
        let m = 256;
        let k = 256;
        let n = 256;
        let mut a_rows = Vec::with_capacity(m);
        for _ in 0..m {
            a_rows.push(K::from_floats(vec![0.01; k]));
        }
        let a = K::from_list(a_rows);

        let mut b_rows = Vec::with_capacity(k);
        for _ in 0..k {
            b_rows.push(K::from_floats(vec![0.02; n]));
        }
        let b = K::from_list(b_rows);

        let start = Instant::now();
        let _res = va::matmul(&a, &b);
        let duration = start.elapsed();
        println!("matmul (256x256 x 256x256): {:?}", duration);
    }

    // 4. Benchmark Activations (on 10M floats)
    {
        let size = 5_000_000;
        let a = K::from_floats(vec![0.8; size]);

        let start = Instant::now();
        let _res = va::sigmoid(&a);
        let duration = start.elapsed();
        println!("sigmoid (5M floats): {:?}", duration);

        let start = Instant::now();
        let _res = va::tanh(&a);
        let duration = start.elapsed();
        println!("tanh (5M floats): {:?}", duration);

        let start = Instant::now();
        let _res = va::relu(&a);
        let duration = start.elapsed();
        println!("relu (5M floats): {:?}", duration);

        // Softmax is slower, use 1M floats or row-wise
        let start = Instant::now();
        let _res = va::softmax(&a);
        let duration = start.elapsed();
        println!("softmax (5M floats): {:?}", duration);
    }
}
