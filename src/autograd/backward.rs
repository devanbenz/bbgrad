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

impl<T: Clone + Debug + Zero + Neg<Output = T> + 'static> Backward for TensorSub<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a - b) wrt a is 1
        // derivative of (a - b) wrt b is -1
        (out_grade.clone(), -out_grade.clone())
    }
}

impl<T: Clone + Debug + Zero + Mul<Output = T> + 'static> Backward for TensorMul<T> {
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

impl<T: Clone + Debug + Zero + Mul<Output = T> + num_traits::Pow<i32, Output = T> + 'static>
    Backward for TensorPow<T>
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
        let a = out_grade.clone() * b.clone().pow(self.power) * a.clone().pow(self.power - 1);

        (out_grade.clone(), out_grade.clone())
    }
}
