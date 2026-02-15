#![allow(dead_code)]

use super::ops::TensorMul;
use super::{ops::TensorAdd, tensor::Tensor};
use std::fmt::Debug;

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

impl<T: Clone + Debug> Backward for TensorMul<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        todo!()
    }
}
