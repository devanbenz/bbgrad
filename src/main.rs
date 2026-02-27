use bbgrad::autograd::ops_impl::{MatMul, Sigmoid};
use bbgrad::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};

fn main() {
    let input_data = TensorData::new(
        TensorDtype::Float64,
        TensorDataInner::List(vec![0.9, 0.1, 0.8, 0.2]),
    );
    let t1 = Tensor::new(input_data, Some(&[2, 2]));
    t1.reshape(&[4]);
    println!("{}", t1);
    let weights = TensorData::new(
        TensorDtype::Float64,
        TensorDataInner::List(vec![1., -1., 1., -1., 1., 1., -1., -1.]),
    );
    let weight_tensor = Tensor::new(weights, Some(&[2, 4]));
    println!("{}", weight_tensor);

    let weighted_sum = weight_tensor.matmul(&t1);
    println!("{}\n", weighted_sum);

    let weighted_sum = weighted_sum + (-0.5f64);
    println!("{}\n", weighted_sum);

    let weighted_sum = weighted_sum.sigmoid();
    println!("{}\n", weighted_sum);

    weighted_sum.graph();
}
