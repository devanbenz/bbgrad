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

    pub fn train(&self, input: Tensor<f64>, expected_output: Tensor<f64>, learning_rate: f64) {
        for w in &self.weights {
            w.zero_grad();
        }
        for b in &self.biases {
            b.zero_grad();
        }

        let prediction = self.forward(input);

        let loss = prediction.loss(&expected_output);
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

    pub fn backward(&self) {
        todo!()
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
