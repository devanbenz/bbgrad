use bbgrad::autograd::nn::PerceptronBuilder;
use bbgrad::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;

fn main() {
    let mut data = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("/Users/devan/Documents/OSS/mnist/mnist_test.csv")
        .unwrap();

    let perceptron = PerceptronBuilder::new((-1.0, 1.0))
        .with_layer(784)
        .with_layer(16)
        .with_layer(16)
        .with_layer(10)
        .build();

    for record in data.records() {
        let r = record
            .unwrap()
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();

        let label = r[0] as usize;
        let input_tensor = Tensor::new(
            TensorData::new(
                TensorDtype::Float64,
                TensorDataInner::NdArray(
                    ArrayD::from_shape_vec(IxDyn(&[784, 1]), r[1..].to_owned())
                        .expect("Cannot create array from input"),
                ),
            ),
            None,
        );
        let i = input_tensor / 255.0f64;
        let t = perceptron.forward(i);
        t.graph();
    }
}
