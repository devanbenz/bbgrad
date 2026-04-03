use super::tensor_builder::FloatTensorBuilder;
use crate::autograd::ops_impl::{MatMul, Sigmoid, Softmax};
use crate::autograd::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct Perceptron {
    random_dist: Uniform<f64>,
    layers: Vec<usize>,
    weights: Vec<Tensor<f64>>,
    biases: Vec<Tensor<f64>>,
}

impl Perceptron {
    pub fn predict(&self, input: Tensor<f64>) -> Tensor<f64> {
        self.forward(input)
    }

    pub fn train(
        &self,
        input: Tensor<f64>,
        expected_output: Tensor<f64>,
        learning_rate: f64,
    ) -> f64 {
        for w in &self.weights {
            w.zero_grad();
        }
        for b in &self.biases {
            b.zero_grad();
        }

        let prediction = self.forward(input);

        let loss = prediction.loss(&expected_output);
        let loss_val = loss.ndarray().mean().unwrap();
        loss.backward(None);

        for w in &self.weights {
            if let Some(grad) = w.grad() {
                let updated = &w.ndarray() - &(&grad.ndarray() * learning_rate);
                w.update_data(updated);
            }
        }
        for b in &self.biases {
            if let Some(grad) = b.grad() {
                let updated = &b.ndarray() - &(&grad.ndarray() * learning_rate);
                b.update_data(updated);
            }
        }

        loss_val
    }

    pub fn curr_flat_loss(&self, input: Tensor<f64>, expected_output: Tensor<f64>) -> f64 {
        let prediction = self.forward(input);
        let loss = prediction.loss(&expected_output).ndarray().mean();
        assert!(loss.is_some());

        loss.unwrap()
    }

    pub fn forward(&self, input: Tensor<f64>) -> Tensor<f64> {
        let mut activation = input;

        for (idx, weight_tensor) in self.weights.iter().enumerate() {
            if idx == self.weights.len() - 1 {
                activation =
                    (weight_tensor.matmul(&activation) + self.biases[idx].clone()).softmax();
            } else {
                activation =
                    (weight_tensor.matmul(&activation) + self.biases[idx].clone()).sigmoid();
            }
        }

        activation
    }

    pub fn distribution(&self) -> &Uniform<f64> {
        &self.random_dist
    }
}

pub struct PerceptronBuilder {
    nn: Perceptron,
}

impl PerceptronBuilder {
    pub fn new(rand_dist: (f64, f64)) -> Self {
        let rd = Uniform::new(rand_dist.0, rand_dist.1).unwrap();
        Self {
            nn: Perceptron {
                layers: vec![],
                random_dist: rd,
                weights: vec![],
                biases: vec![],
            },
        }
    }

    pub fn with_layer(mut self, size: usize) -> PerceptronBuilder {
        self.nn.layers.push(size);
        self
    }

    pub fn build(mut self) -> Perceptron {
        for i in 0..self.nn.layers.len() - 1 {
            let rows = self.nn.layers[i + 1];
            let cols = self.nn.layers[i];

            let weights = FloatTensorBuilder::new()
                .with_ndarray(ArrayD::<f64>::random(
                    IxDyn(&[rows, cols]),
                    &self.nn.random_dist,
                ))
                .with_grad(true)
                .build();

            let biases = FloatTensorBuilder::new()
                .with_ndarray(ArrayD::zeros(IxDyn(&[rows, 1])))
                .with_grad(true)
                .build();

            self.nn.weights.push(weights);
            self.nn.biases.push(biases);
        }

        self.nn
    }
}

pub struct RNN {
    random_dist: Uniform<f64>,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    input_to_hidden_weights: Tensor<f64>,
    hidden_to_hidden_weights: Tensor<f64>,
    hidden_to_output_weights: Tensor<f64>,
    hidden_bias: Tensor<f64>,
    output_bias: Tensor<f64>,
}

impl RNN {
    pub fn init_hidden(&self) -> Tensor<f64> {
        FloatTensorBuilder::new()
            .with_ndarray(ArrayD::zeros(IxDyn(&[self.hidden_size, 1])))
            .build()
    }

    pub fn forward_step(
        &self,
        input: Tensor<f64>,
        hidden_state: Tensor<f64>,
    ) -> (Tensor<f64>, Tensor<f64>) {
        let new_hidden_state = (self.input_to_hidden_weights.matmul(&input)
            + self.hidden_to_hidden_weights.matmul(&hidden_state)
            + self.hidden_bias.clone())
        .sigmoid();

        let output = (self.hidden_to_output_weights.matmul(&new_hidden_state)
            + self.output_bias.clone())
        .softmax();

        (new_hidden_state, output)
    }

    pub fn forward_sequence(&self, inputs: Vec<Tensor<f64>>) -> (Tensor<f64>, Vec<Tensor<f64>>) {
        let mut hidden_state = self.init_hidden();
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let (new_hidden_state, output) = self.forward_step(input, hidden_state);
            hidden_state = new_hidden_state;
            outputs.push(output);
        }

        (hidden_state, outputs)
    }

    pub fn predict(&self, inputs: Vec<Tensor<f64>>) -> Vec<Tensor<f64>> {
        let (_final_hidden_state, outputs) = self.forward_sequence(inputs);
        outputs
    }

    pub fn forward_and_loss(
        &self,
        inputs: Vec<Tensor<f64>>,
        expected_outputs: Vec<Tensor<f64>>,
    ) -> f64 {
        let sequence_len = expected_outputs.len() as f64;
        let (_final_hidden_state, outputs) = self.forward_sequence(inputs);

        let mut total_loss_val = 0.0;
        for (output, expected) in outputs.iter().zip(expected_outputs.iter()) {
            let loss = output.loss(expected);
            total_loss_val += loss.ndarray().mean().unwrap();
            loss.backward(None);
        }

        total_loss_val / sequence_len
    }

    pub fn zero_all_grad(&self) {
        for param in self.params() {
            param.zero_grad();
        }
    }

    pub fn update_params(&self, learning_rate: f64, divisor: f64) {
        let grad_clip = 5.0;
        for param in self.params() {
            if let Some(grad) = param.grad() {
                let clipped_grad = grad.ndarray().mapv(|x| x.clamp(-grad_clip, grad_clip));
                let scaled_grad = &clipped_grad / divisor;
                let updated = &param.ndarray() - &(&scaled_grad * learning_rate);
                param.update_data(updated);
            }
        }
    }

    pub fn curr_flat_loss(
        &self,
        inputs: Vec<Tensor<f64>>,
        expected_outputs: Vec<Tensor<f64>>,
    ) -> f64 {
        let sequence_len = expected_outputs.len() as f64;
        let (_final_hidden_state, outputs) = self.forward_sequence(inputs);

        let mut total_loss_val = 0.0;
        for (output, expected) in outputs.iter().zip(expected_outputs.iter()) {
            let loss = output.loss(expected).ndarray().mean();
            assert!(loss.is_some());
            total_loss_val += loss.unwrap();
        }

        total_loss_val / sequence_len
    }

    fn params(&self) -> Vec<&Tensor<f64>> {
        vec![
            &self.input_to_hidden_weights,
            &self.hidden_to_hidden_weights,
            &self.hidden_to_output_weights,
            &self.hidden_bias,
            &self.output_bias,
        ]
    }

    pub fn distribution(&self) -> &Uniform<f64> {
        &self.random_dist
    }
}

pub struct RNNBuilder {
    random_dist: Uniform<f64>,
    input_size: Option<usize>,
    hidden_size: Option<usize>,
    output_size: Option<usize>,
}

impl RNNBuilder {
    pub fn new(rand_dist: (f64, f64)) -> Self {
        let rd = Uniform::new(rand_dist.0, rand_dist.1).unwrap();
        Self {
            random_dist: rd,
            input_size: None,
            hidden_size: None,
            output_size: None,
        }
    }

    pub fn with_input_size(mut self, input_size: usize) -> RNNBuilder {
        self.input_size = Some(input_size);
        self
    }

    pub fn with_hidden_size(mut self, hidden_size: usize) -> RNNBuilder {
        self.hidden_size = Some(hidden_size);
        self
    }

    pub fn with_output_size(mut self, output_size: usize) -> RNNBuilder {
        self.output_size = Some(output_size);
        self
    }

    pub fn build(self) -> RNN {
        let input_size = self.input_size.expect("input_size is required");
        let hidden_size = self.hidden_size.expect("hidden_size is required");
        let output_size = self.output_size.expect("output_size is required");

        let input_to_hidden_weights = FloatTensorBuilder::new()
            .with_ndarray(ArrayD::<f64>::random(
                IxDyn(&[hidden_size, input_size]),
                &self.random_dist,
            ))
            .with_grad(true)
            .build();

        let hidden_to_hidden_weights = FloatTensorBuilder::new()
            .with_ndarray(ArrayD::<f64>::random(
                IxDyn(&[hidden_size, hidden_size]),
                &self.random_dist,
            ))
            .with_grad(true)
            .build();

        let hidden_to_output_weights = FloatTensorBuilder::new()
            .with_ndarray(ArrayD::<f64>::random(
                IxDyn(&[output_size, hidden_size]),
                &self.random_dist,
            ))
            .with_grad(true)
            .build();

        let hidden_bias = FloatTensorBuilder::new()
            .with_ndarray(ArrayD::zeros(IxDyn(&[hidden_size, 1])))
            .with_grad(true)
            .build();

        let output_bias = FloatTensorBuilder::new()
            .with_ndarray(ArrayD::zeros(IxDyn(&[output_size, 1])))
            .with_grad(true)
            .build();

        RNN {
            random_dist: self.random_dist,
            input_size,
            hidden_size,
            output_size,
            input_to_hidden_weights,
            hidden_to_hidden_weights,
            hidden_to_output_weights,
            hidden_bias,
            output_bias,
        }
    }
}
