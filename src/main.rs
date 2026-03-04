use bbgrad::autograd::nn::PerceptronBuilder;
use bbgrad::autograd::tensor_builder::FloatTensorBuilder;
use ndarray::{ArrayD, IxDyn};

fn main() {
    let input_file = std::env::args().nth(1).expect("Usage: bbgrad <input csv>");
    let mut data = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(input_file)
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
        let array = ArrayD::from_shape_vec(IxDyn(&[784, 1]), r[1..].to_owned()).unwrap() / 255.0f64;
        let input_tensor = FloatTensorBuilder::new()
            .with_ndarray(array)
            .with_grad(false)
            .build();
        let t = perceptron.forward(input_tensor);
        t.graph();
        let loss = t.loss(label, 10, 1f64);
        println!("loss: {}", loss);
    }
}
