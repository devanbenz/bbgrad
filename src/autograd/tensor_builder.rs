use ndarray::{ArrayBase, IxDyn, OwnedRepr};

use super::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};

pub struct FloatTensorBuilder {
    ndarray: Option<ArrayBase<OwnedRepr<f64>, IxDyn, f64>>,
    requires_grad: bool,
}

impl FloatTensorBuilder {
    pub fn new() -> Self {
        Self {
            ndarray: None,
            requires_grad: false,
        }
    }

    pub fn with_ndarray(mut self, ndarray: ArrayBase<OwnedRepr<f64>, IxDyn, f64>) -> Self {
        self.ndarray = Some(ndarray);
        self
    }

    pub fn with_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn build(self) -> Tensor<f64> {
        let ndarray = self.ndarray.expect("ndarray is required to build a FloatTensor");
        let data = TensorData::new(TensorDtype::Float64, TensorDataInner::NdArray(ndarray));
        let tensor = Tensor::new(data, None);
        tensor.set_requires_grad(self.requires_grad);
        tensor
    }
}
