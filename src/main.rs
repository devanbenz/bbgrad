use bbgrad::autograd::nn::PerceptronBuilder;
use bbgrad::autograd::tensor_builder::FloatTensorBuilder;
use ndarray::{ArrayD, IxDyn};
use ndarray_stats::QuantileExt;

fn main() {
    let mut count = 0.0;
    let mut correct_prediction = 0.0;

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
        count += 1.;
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

        let mut target_vec = vec![0.0f64; 10];
        target_vec[label] = 1.0;
        let target_arr = ArrayD::from_shape_vec(IxDyn(&[10, 1]), target_vec).unwrap();
        let target_tensor = FloatTensorBuilder::new()
            .with_ndarray(target_arr)
            .with_grad(false)
            .build();

        let t = perceptron.forward(input_tensor);
        let prediction = t.ndarray().argmax().expect("couldn't get argmax");
        if prediction[0] == label {
            correct_prediction += 1.;
        }
        let loss = t.loss(&target_tensor);
        loss.backward(None);
        println!("\n\n");
        loss.graph();
    }
}
