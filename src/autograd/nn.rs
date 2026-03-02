use crate::autograd::ForwardType;
use crate::autograd::ops_impl::{MatMul, Sigmoid};
use crate::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
use ndarray::{ArrayD, IxDyn};

pub struct Perceptron<T: ForwardType> {
    layers: Vec<(Tensor<T>, Tensor<T>)>,
}

impl<T: ForwardType> Perceptron<T> {
    pub fn forward(&self, input: Tensor<T>) -> Tensor<T> {
        for layer in self.layers {}
        let weight_tensor = Tensor::new(weights, Some(&[2, 4]));
        println!("tensor 2 (weights): {}\n", weight_tensor);

        let weighted_sum = weight_tensor.matmul(&t1);
        println!("weighted sum: {}\n", weighted_sum);

        let weighted_sum = weighted_sum + (-0.5f64);
        println!("weighted sum with bias of -0.5: {}\n", weighted_sum);

        let weighted_sum = weighted_sum.sigmoid();
        println!("weighted sum with sigmoid squish: {}\n", weighted_sum);
    }
}

impl<T: ForwardType> Default for Perceptron<T> {
    fn default() -> Self {
        Self { layers: vec![] }
    }
}

pub struct PerceptronBuilder<T: ForwardType> {
    nn: Perceptron<T>,
}

impl<T: ForwardType> PerceptronBuilder<T> {
    pub fn new() -> Self {
        PerceptronBuilder {
            nn: Perceptron::default(),
        }
    }

    pub fn with_layer(
        mut self,
        weight_shape: Vec<usize>,
        bias_shape: Vec<usize>,
    ) -> PerceptronBuilder<T> {
        let weight_tensor = Tensor::new(
            TensorData::new(
                TensorDtype::Float64,
                TensorDataInner::NdArray(ArrayD::zeros(IxDyn(&weight_shape))),
            ),
            None,
        );
        let bias_tensor = Tensor::new(
            TensorData::new(
                TensorDtype::Float64,
                TensorDataInner::NdArray(ArrayD::zeros(IxDyn(&bias_shape))),
            ),
            None,
        );
        self.nn.layers.push((weight_tensor, bias_tensor));
        self
    }

    pub fn build(self) -> Perceptron<T> {
        self.nn
    }
}
