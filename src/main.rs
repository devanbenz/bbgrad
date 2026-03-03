use bbgrad::autograd::nn::PerceptronBuilder;
use bbgrad::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;

fn main() {
    let perceptron = PerceptronBuilder::new((-1.0, 1.0))
        .with_layer(784)
        .with_layer(16)
        .with_layer(16)
        .with_layer(10)
        .build();

    let input_tensor = Tensor::new(
        TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::NdArray(ArrayD::random(IxDyn(&[784, 1]), perceptron.distribution())),
        ),
        None,
    );
    let loss = perceptron.forward(input_tensor).loss(5, 10, 1.0f64);
    println!("{loss}");
}
