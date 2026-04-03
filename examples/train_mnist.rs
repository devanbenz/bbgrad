use std::time;

use bbgrad::autograd::nn::PerceptronBuilder;
use bbgrad::autograd::tensor_builder::FloatTensorBuilder;
use ndarray::{ArrayD, IxDyn};
use ndarray_stats::QuantileExt;

fn main() {
    let mut count = 0.0;
    let mut correct_prediction = 0.0;

    let input_file = std::env::args().nth(1).unwrap();
    let train_file = std::env::args().nth(2).unwrap();
    let epoch_count = std::env::args().nth(3).unwrap().parse::<i32>().unwrap();
    let learning_rate = std::env::args().nth(4).unwrap().parse::<f64>().unwrap();
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

        let prediction = perceptron.predict(input_tensor);
        let prediction = prediction.ndarray().argmax().expect("cannot get argmax")[0];
        if prediction == label {
            correct_prediction += 1.;
        }
    }
    println!("Pre-training accuracy: {}", correct_prediction / count);

    println!("Begin training...");
    let t = time::Instant::now();
    for epoch in 1..epoch_count {
        let mut epoch_loss = 0.0;
        let mut record_n = 0.0;
        let mut train_data = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(&train_file)
            .unwrap();

        for record in train_data.records() {
            let r = record
                .unwrap()
                .iter()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<f64>>();

            let label = r[0] as usize;
            let array =
                ArrayD::from_shape_vec(IxDyn(&[784, 1]), r[1..].to_owned()).unwrap() / 255.0f64;
            let input_tensor = FloatTensorBuilder::new()
                .with_ndarray(array)
                .with_grad(false)
                .build();
            let mut arr = [0.0; 10];
            arr[label] = 1.00;
            let exp_array = ArrayD::from_shape_vec(IxDyn(&[10, 1]), arr.to_vec()).unwrap();
            let exp_tensor = FloatTensorBuilder::new()
                .with_ndarray(exp_array)
                .with_grad(false)
                .build();

            epoch_loss += perceptron.train(input_tensor, exp_tensor, learning_rate);
            record_n += 1.0;
        }
        println!("epoch: {}, processed: {} records", epoch, record_n);
        println!("loss: {}", epoch_loss / record_n);
    }
    println!("Training time: {}s", t.elapsed().as_secs_f64());

    count = 0.0;
    correct_prediction = 0.0;
    data.seek(csv::Position::new()).expect("cannot seek");
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

        let prediction = perceptron.predict(input_tensor);
        let prediction = prediction.ndarray().argmax().expect("cannot get argmax")[0];
        if prediction == label {
            correct_prediction += 1.;
        }
    }
    println!("Post-training accuracy: {}", correct_prediction / count);
}
