#![allow(dead_code)]

use super::ops::{TensorDiv, TensorMul, TensorPow, TensorSub};
use super::tensor::{Tensor, TensorData, TensorDataInner};
use super::{ForwardType, ops::TensorAdd};
use crate::autograd::ops_impl::Pow;
use ndarray::{ArrayD, IxDyn};

pub trait Backward {
    type OutGrad;
    type Node;
    type Output;

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output;
}

impl<T: ForwardType> Backward for TensorAdd<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a + b) wrt a is 1
        // derivative of (a + b) wrt b is 1
        (out_grade.clone(), out_grade.clone())
    }
}

impl<T: ForwardType> Backward for TensorSub<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a - b) wrt a is 1
        // derivative of (a - b) wrt b is -1
        (out_grade.clone(), -out_grade.clone())
    }
}

impl<T: ForwardType> Backward for TensorDiv<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, _out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        todo!()
    }
}

impl<T: ForwardType> Backward for TensorMul<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        assert_eq!(node.inputs().len(), 2);
        let a = node.inputs();
        let a = a.first().unwrap();
        let b = node.inputs();
        let b = b.get(1).unwrap();

        // derivative of (a * b) wrt a is 1 * b
        // derivative of (a * b) wrt b is 1 * a
        (out_grade.clone() * b.clone(), out_grade.clone() * a.clone())
    }
}

impl<T: ForwardType> Backward for TensorPow<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        // Pow is a unary operation: f(x) = x^n where n = self.power
        // d/dx (x^n) = n * x^(n-1)
        let inputs = node.inputs();
        assert_eq!(inputs.len(), 1);
        let x = inputs.first().unwrap();

        let n: T = T::from(self.power).unwrap();
        let n_arr = ArrayD::from_elem(IxDyn(&x.shape()), n);
        let n_tensor = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(n_arr)),
            None,
        );

        let grad_x = out_grade * n_tensor * x.clone().pow(self.power - 1);

        let zeros_arr = ArrayD::zeros(IxDyn(&x.shape()));
        let zeros_tensor = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(zeros_arr)),
            None,
        );
        (grad_x, zeros_tensor)
    }
}

#[cfg(test)]
mod backward_tests {
    use crate::autograd::tensor::{TensorData, TensorDataInner, TensorDtype};

    use super::*;

    fn build_tensors() -> (Tensor<f64>, Tensor<f64>) {
        let data = TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::List(vec![1., 2., 3., 4.]),
        );
        let tensor = Tensor::new(data, Some(&[2, 2]));
        let data2 = TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::List(vec![1., 2., 3., 4.]),
        );
        let tensor2 = Tensor::new(data2, Some(&[2, 2]));
        (tensor, tensor2)
    }

    #[test]
    fn backward_add() {}
}
