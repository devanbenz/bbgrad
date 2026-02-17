#![allow(dead_code)]

use super::ops::{TensorMul, TensorPow, TensorSub};
use super::{ops::TensorAdd, tensor::Tensor};
use crate::autograd::ops_impl::Pow;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Mul, Neg};

pub(crate) trait Backward {
    type OutGrad;
    type Node;
    type Output;

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output;
}

impl<T: Clone + Debug> Backward for TensorAdd<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a + b) wrt a is 1
        // derivative of (a + b) wrt b is 1
        (out_grade.clone(), out_grade.clone())
    }
}

impl<T: Clone + Debug + Zero + num_traits::One + Neg<Output = T> + 'static> Backward
    for TensorSub<T>
{
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a - b) wrt a is 1
        // derivative of (a - b) wrt b is -1
        (out_grade.clone(), -out_grade.clone())
    }
}

impl<T: Clone + Debug + Zero + Mul<Output = T> + num_traits::One + 'static> Backward
    for TensorMul<T>
{
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

impl<
    T: Clone
        + Debug
        + Zero
        + num_traits::One
        + Mul<Output = T>
        + num_traits::Pow<i32, Output = T>
        + 'static,
> Backward for TensorPow<T>
where
    i32: Mul<Tensor<T>, Output = Tensor<T>>,
{
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        assert_eq!(node.inputs().len(), 2);
        let a = node.inputs();
        let a = a.first().unwrap();
        let b = node.inputs();
        let b = b.get(1).unwrap();

        // derivative of (a * b) wrt a is out_grad * (x * b^x) * a^x-1
        // derivative of (a * b) wrt b is out_grad * (x * a^x) * b^x-1
        let a = out_grade.clone()
            * (self.power * b.clone().pow(self.power))
            * a.clone().pow(self.power - 1);
        let b = out_grade.clone()
            * (self.power * a.clone().pow(self.power))
            * b.clone().pow(self.power - 1);

        (a, b)
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
