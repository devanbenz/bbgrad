use bbgrad::autograd::nn::{NeuralNet, NeuralNetBuilder};
use bbgrad::autograd::ops_impl::{MatMul, Sigmoid};
use bbgrad::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};

fn main() {
    let input_data = TensorData::new(
        TensorDtype::Float64,
        TensorDataInner::List(vec![0.9, 0.1, 0.8, 0.2]),
    );
    let t1 = Tensor::new(input_data, Some(&[2, 2]));
    t1.reshape(&[4]);
    println!("tensor 1 (inputs): {}\n", t1);
    let weights = TensorData::new(
        TensorDtype::Float64,
        TensorDataInner::List(vec![1., 1., 0.0, 0.0, 1., 1., -1.2, -1.5]),
    );
    let weight_tensor = Tensor::new(weights, Some(&[2, 4]));
    println!("tensor 2 (weights): {}\n", weight_tensor);

    let weighted_sum = weight_tensor.matmul(&t1);
    println!("weighted sum: {}\n", weighted_sum);

    let weighted_sum = weighted_sum + (-0.5f64);
    println!("weighted sum with bias of -0.5: {}\n", weighted_sum);

    let weighted_sum = weighted_sum.sigmoid();
    println!("weighted sum with sigmoid squish: {}\n", weighted_sum);

    let loss = weighted_sum.loss(0, 1.0);
    println!("loss: {loss}");

    let nn: NeuralNet<f64> = NeuralNetBuilder::new()
        .with_layer(vec![1, 16], vec![1, 16])
        .with_layer(vec![1, 16], vec![1, 16])
        .build();
}
