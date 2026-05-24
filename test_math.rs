mod k;
mod parallel;
mod primitives;
mod va;
mod rag;
mod ffi;


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

    // Test grade_up / grade_down
    let vals_i = K::from_ints(vec![30, 10, 20]);
    println!("grade_up([30,10,20]) = {:?}", va::grade_up(&vals_i));
    println!("grade_down([30,10,20]) = {:?}", va::grade_down(&vals_i));

    let vals_f = K::from_floats(vec![f64::NAN, 30.0, f64::INFINITY, 10.0, f64::NEG_INFINITY, 20.0, f64::NAN]);
    println!("grade_up with NaNs/Infs = {:?}", va::grade_up(&vals_f));
    println!("grade_down with NaNs/Infs = {:?}", va::grade_down(&vals_f));

    // Test take
    let x = K::from_ints(vec![10, 20, 30]);
    println!("take(5, [10,20,30]) = {:?}", va::take(&K::ki(5), &x));
    println!("take(-5, [10,20,30]) = {:?}", va::take(&K::ki(-5), &x));
    println!("take(2, [10,20,30]) = {:?}", va::take(&K::ki(2), &x));
    println!("take(-2, [10,20,30]) = {:?}", va::take(&K::ki(-2), &x));

    // Test first, reverse, flip
    let arr = K::from_ints(vec![10, 20, 30]);
    println!("first([10,20,30]) = {:?}", va::first(&arr));
    println!("reverse([10,20,30]) = {:?}", va::reverse(&arr));
    let mat_ints = K::from_list(vec![
        K::from_ints(vec![1, 2, 3]),
        K::from_ints(vec![4, 5, 6]),
    ]);
    println!("flip([[1,2,3],[4,5,6]]) = {:?}", va::flip(&mat_ints));

    // Test unique
    let dup_ints = K::from_ints(vec![1, 2, 2, 3, 1, 4, 3]);
    println!("unique([1,2,2,3,1,4,3]) = {:?}", va::unique(&dup_ints));
    let dup_floats = K::from_floats(vec![1.5, 2.5, 2.5, f64::NAN, 3.5, 1.5, f64::NAN]);
    println!("unique([1.5,2.5,2.5,NaN,3.5,1.5,NaN]) = {:?}", va::unique(&dup_floats));
    let dup_list = K::from_list(vec![
        K::from_ints(vec![1, 2]),
        K::from_ints(vec![3, 4]),
        K::from_ints(vec![1, 2]),
        K::from_ints(vec![5]),
    ]);
    println!("unique([[1,2],[3,4],[1,2],[5]]) = {:?}", va::unique(&dup_list));
    println!("unique(atom 42) = {:?}", va::unique(&K::ki(42)));

    // Test FFI primitives
    let fno_sin = K::ki(101);
    let proj_sin = va::_2m(&fno_sin);
    println!("FFI projection monadic 2: 101 = {:?}", proj_sin);
    let ffi_res = va::_2d(&proj_sin, &K::kf(0.0));
    println!("FFI call dyadic _2d (sin 0.0) = {:?}", ffi_res);
    let ffi_res_arr = va::_2d(&fno_sin, &K::from_floats(vec![0.0, 1.5707963267948966]));
    println!("FFI call dyadic _2d (sin [0, pi/2]) = {:?}", ffi_res_arr);

    // Test RAG Retrieval
    use rag::{Document, RetrievalPipeline};
    let docs = vec![
        Document { id: 1, content: "Rust is a systems programming language".to_string() },
        Document { id: 2, content: "Kona is an interpreter for the K programming language".to_string() },
        Document { id: 3, content: "Retrieve relevant context for RAG pipelines".to_string() },
    ];
    let pipeline = RetrievalPipeline::new(docs);
    let results = pipeline.retrieve("K interpreter language", 2);
    println!("RAG Retrieve results for 'K interpreter language':");
    for (doc, score) in results {
        println!("  Doc {}: '{}' (Score: {:.4})", doc.id, doc.content, score);
    }
}

