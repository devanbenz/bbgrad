use bbgrad::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

fn main() {
    let data = TensorData::new(
        TensorDtype::Float64,
        TensorDataInner::<f64>::NdArray(ArrayD::<f64>::random(
            IxDyn(&[200, 100, 250]),
            StandardNormal,
        )),
    );
    let t1 = Tensor::new(data, None);
    let data2 = TensorData::new(
        TensorDtype::Float64,
        TensorDataInner::<f64>::NdArray(ArrayD::<f64>::random(
            IxDyn(&[200, 100, 250]),
            StandardNormal,
        )),
    );
    let t2 = Tensor::new(data2, None);

    let _ = t1.clone() + t2.clone();
    println!(
        "t1 [[10, 10, 10]] before add={}",
        t1.ndarray()[[10, 10, 10]]
    );
    println!(
        "t2 [[10, 10, 10]] before add={}\n",
        t2.ndarray()[[10, 10, 10]]
    );

    let mut times = Vec::new();
    for _ in 0..100 {
        let t1c = t1.clone();
        let t2c = t2.clone();
        let start = std::time::Instant::now();
        let _t3 = t1c + t2c;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!("Adding 5 million vectors:");
    println!("Best:   {:.3} ms", times[0]);
    println!("Avg:    {:.3} ms\n", avg);

    let t3 = t1 + t2;

    println!("tensor size={}", t3.size());
    println!("tensor shape {:?}", t3.shape());
    println!("t3 [[10, 10, 10]] after add={}", t3.ndarray()[[10, 10, 10]]);
}
