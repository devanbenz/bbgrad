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
}

impl<T: ForwardType + SampleUniform> Perceptron<T> {
    pub fn forward(&self, input: Tensor<T>) -> Tensor<T> {
        let mut activation = input;

        for i in 0..self.layers.len() - 1 {
            let rows = self.layers[i + 1];
            let cols = self.layers[i];

            let weights = Tensor::new(
                TensorData::new(
                    TensorDtype::Float64,
                    TensorDataInner::NdArray(ArrayD::random(
                        IxDyn(&[rows, cols]),
                        &self.random_dist,
                    )),
                ),
                None,
            );

            activation = weights.matmul(&activation).sigmoid();
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
            },
        }
    }

    pub fn with_layer(mut self, size: usize) -> PerceptronBuilder<T> {
        self.nn.layers.push(size);
        self
    }

    pub fn build(self) -> Perceptron<T> {
        self.nn
    }
}
