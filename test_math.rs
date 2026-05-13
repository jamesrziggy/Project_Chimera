mod k;
mod parallel;
mod primitives;
mod va;

use k::K;

fn main() {
    println!(
        "implemented primitives: {}",
        primitives::implemented_count()
    );

    let a = K::from_ints(vec![1, 2, 3]);
    let b = K::from_ints(vec![4, 5, 6]);
    println!("plus([1,2,3],[4,5,6]) = {:?}", va::plus(&a, &b));
    println!("minus([4,5,6],[1,2,3]) = {:?}", va::minus(&b, &a));
    println!("times([1,2,3],[4,5,6]) = {:?}", va::times(&a, &b));
    println!("divide([4,5,6],[1,2,3]) = {:?}", va::divide(&b, &a));
    println!("power([1,2,3],2) = {:?}", va::power(&a, &K::ki(2)));
    println!("modulo([4,5,6],3) = {:?}", va::modulo(&b, &K::ki(3)));
    println!("min_and([1,2,3],[4,5,6]) = {:?}", va::min_and(&a, &b));
    println!("max_or([1,2,3],[4,5,6]) = {:?}", va::max_or(&a, &b));
    println!("less([1,2,3],[4,5,6]) = {:?}", va::less(&a, &b));
    println!("more([4,5,6],[1,2,3]) = {:?}", va::more(&b, &a));
    println!(
        "equals([1,2,3],[1,0,3]) = {:?}",
        va::equals(&a, &K::from_ints(vec![1, 0, 3]))
    );
    println!("match([1,2,3],[1,2,3]) = {:?}", va::match_k(&a, &a));
    println!("negate([1,2,3]) = {:?}", va::negate(&a));
    println!("reciprocal([1,2,3]) = {:?}", va::reciprocal(&a));
    println!(
        "floor([1.2,-1.2,3.0]) = {:?}",
        va::floor_verb(&K::from_floats(vec![1.2, -1.2, 3.0]))
    );
    println!(
        "ceiling([1.2,-1.2,3.0]) = {:?}",
        va::ceiling(&K::from_floats(vec![1.2, -1.2, 3.0]))
    );

    let big_a = K::from_ints((0..100_000).collect());
    let big_b = K::from_ints((0..100_000).rev().collect());
    let big_times = va::times(&big_a, &big_b);
    println!(
        "parallel workers for 100000 elements: {}",
        parallel::worker_count(100_000)
    );
    println!(
        "times big first/last = {:?}/{:?}",
        big_times.ki_data()[0],
        big_times.ki_data()[99_999]
    );

    // Test Sqrt
    let x = K::from_floats(vec![4.0, 9.0, 16.0]);
    let s = va::sqrt(&x);
    println!("sqrt([4,9,16]) = {:?}", s);

    // Test Exp
    let x = K::from_floats(vec![0.0, 1.0]);
    let e = va::exp(&x);
    println!("exp([0,1]) = {:?}", e);

    // Test Triu (Mask)
    // Create 3x3 matrix
    let row1 = K::from_floats(vec![1.0, 1.0, 1.0]);
    let row2 = K::from_floats(vec![1.0, 1.0, 1.0]);
    let row3 = K::from_floats(vec![1.0, 1.0, 1.0]);
    let mat = K::from_list(vec![row1, row2, row3]);
    let masked = va::triu(&mat, 0); // Upper triangle
    println!("triu(3x3):");
    println!("{:?}", masked);
}
