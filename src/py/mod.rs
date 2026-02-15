use pyo3::{pyclass, pymethods};

use crate::autograd::tensor::{Tensor, TensorData, TensorDataInner};

#[pyo3::pymodule]
#[pyo3(name = "bbgrad")]
mod pytensor {
    use pyo3::prelude::*;

    use super::PythonTensor;

    #[pyfunction]
    fn tensor(data: Vec<f64>) -> PythonTensor {
        PythonTensor::new(data)
    }
}

#[pyclass]
struct PythonTensor {
    inner: Tensor<f64>,
}

#[pymethods]
impl PythonTensor {
    #[new]
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            inner: Tensor::new(
                TensorData::new(
                    crate::autograd::tensor::TensorDtype::Float64,
                    TensorDataInner::List(data),
                ),
                None,
            ),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.clone().shape()
    }
}
