use crate::autograd::ForwardType;
use crate::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
use ndarray::{ArrayD, IxDyn};

pub struct NeuralNet<T: ForwardType> {
    layers: Vec<(Tensor<T>, Tensor<T>)>,
}

impl<T: ForwardType> NeuralNet<T> {}

impl<T: ForwardType> Default for NeuralNet<T> {
    fn default() -> Self {
        Self { layers: vec![] }
    }
}

pub struct NeuralNetBuilder<T: ForwardType> {
    nn: NeuralNet<T>,
}

impl<T: ForwardType> NeuralNetBuilder<T> {
    pub fn new() -> Self {
        NeuralNetBuilder {
            nn: NeuralNet::default(),
        }
    }

    pub fn with_layer(
        mut self,
        weight_shape: Vec<usize>,
        bias_shape: Vec<usize>,
    ) -> NeuralNetBuilder<T> {
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

    pub fn build(self) -> NeuralNet<T> {
        self.nn
    }
}
