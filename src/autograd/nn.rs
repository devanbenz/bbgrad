use crate::autograd::ForwardType;
use crate::autograd::ops_impl::{MatMul, Sigmoid};
use crate::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand_distr::uniform::SampleUniform;

pub struct Perceptron<T: ForwardType + SampleUniform> {
    random_dist: Uniform<T>,
    layers: Vec<usize>,
    weights: Vec<Tensor<T>>,
    biases: Vec<Tensor<T>>,
}

impl<T: ForwardType + SampleUniform> Perceptron<T> {
    pub fn forward(&self, input: Tensor<T>) -> Tensor<T> {
        let mut activation = input;

        for weight_tensor in self.weights.iter() {
            activation = weight_tensor.matmul(&activation).sigmoid();
        }

        activation
    }

    pub fn distribution(&self) -> &Uniform<T> {
        &self.random_dist
    }
}

pub struct PerceptronBuilder<T: ForwardType + SampleUniform> {
    nn: Perceptron<T>,
}

impl<T: ForwardType + SampleUniform> PerceptronBuilder<T> {
    pub fn new(rand_dist: (T, T)) -> Self {
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

    pub fn with_layer(mut self, size: usize) -> PerceptronBuilder<T> {
        let weights = self.nn.layers.push(size);
        self
    }

    pub fn build(mut self) -> Perceptron<T> {
        for i in 0..self.nn.layers.len() - 1 {
            let rows = self.nn.layers[i + 1];
            let cols = self.nn.layers[i];

            let weights = Tensor::new(
                TensorData::new(
                    TensorDtype::Float64,
                    TensorDataInner::NdArray(ArrayD::random(
                        IxDyn(&[rows, cols]),
                        &self.nn.random_dist,
                    )),
                ),
                None,
            );

            let biases: Tensor<T> = Tensor::new(
                TensorData::new(
                    TensorDtype::Float64,
                    TensorDataInner::NdArray(ArrayD::zeros(IxDyn(&[1, cols]))),
                ),
                None,
            );
            self.nn.weights.push(weights);
            self.nn.biases.push(biases);
        }

        self.nn
    }
}
