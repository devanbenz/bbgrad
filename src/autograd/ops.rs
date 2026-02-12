#![allow(dead_code)]

use ndarray::linalg;
use std::fmt::Debug;
use std::marker::Copy;
use std::ops::{self, Sub};
use std::vec;

use ndarray_rand::rand_distr::num_traits::{self, Zero};

use crate::autograd::backward::{
    Forward, TensorAdd, TensorDiv, TensorMatMul, TensorMul, TensorNeg, TensorSub,
};

use super::backward::{TensorPow, TensorSum};
use super::tensor::Tensor;

trait Sum {
    type Output;

    fn sum(&self) -> Self::Output;
}

trait Pow {
    type Output;
    type Exp;

    fn pow(&self, exp: Self::Exp) -> Self::Output;
}

impl<T> ops::Add for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorAdd::new().call(vec![self, rhs])
    }
}

impl<T> ops::Sub for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Sub<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorSub::new().call(vec![self, rhs])
    }
}

impl<T> ops::Mul for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Mul<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorMul::new().call(vec![self, rhs])
    }
}

impl<T> ops::Div for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Div<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorDiv::new().call(vec![self, rhs])
    }
}

impl<T> ops::Neg for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Neg<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        TensorNeg::new().call(vec![self])
    }
}

impl<T> linalg::Dot<Tensor<T>> for Tensor<T>
where
    T: Clone
        + Debug
        + ops::Add<Output = T>
        + ops::Neg<Output = T>
        + 'static
        + num_traits::identities::One
        + Sub<Output = T>
        + Copy
        + std::ops::Div<Output = T>
        + Zero,
{
    type Output = Tensor<T>;

    fn dot(&self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorMatMul::new().call(vec![self.to_owned(), rhs.to_owned()])
    }
}

impl<T: Clone + Debug + Zero + 'static + num_traits::Pow<i32, Output = T>> Pow for Tensor<T> {
    type Output = Tensor<T>;
    type Exp = i32;

    fn pow(&self, exp: Self::Exp) -> Self::Output {
        TensorPow::new(exp).call(vec![self.to_owned()])
    }
}

impl<T: Clone + Debug + Zero + 'static> Sum for Tensor<T> {
    type Output = Tensor<T>;

    fn sum(&self) -> Self::Output {
        TensorSum::new().call(vec![self.to_owned()])
    }
}

#[cfg(test)]
mod tests {

    use ndarray::linalg::Dot;

    use crate::autograd::ops::{Pow, Sum};
    use crate::autograd::tensor::{Tensor, TensorData, TensorDtype, TensorInner};

    fn build_tensors() -> (Tensor<f64>, Tensor<f64>) {
        let data = TensorData::new(
            TensorDtype::Float64,
            TensorInner::List(vec![1., 2., 3., 4.]),
        );
        let tensor = Tensor::new(data, Some(&[2, 2]));
        let data2 = TensorData::new(
            TensorDtype::Float64,
            TensorInner::List(vec![1., 2., 3., 4.]),
        );
        let tensor2 = Tensor::new(data2, Some(&[2, 2]));
        (tensor, tensor2)
    }

    #[test]
    fn test_tensor_add() {
        let (tensor, tensor2) = build_tensors();
        let t3 = tensor + tensor2;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 2f64);
    }

    #[test]
    fn test_tensor_sub() {
        let (tensor, tensor2) = build_tensors();
        let t3 = tensor - tensor2;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 0f64);
    }

    #[test]
    fn test_tensor_mul() {
        let (tensor, tensor2) = build_tensors();
        let t3 = tensor * tensor2;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 1f64);
        assert_eq!(t3.ndarray()[[1, 0]], 9f64);
    }

    #[test]
    fn test_tensor_div() {
        let (tensor, tensor2) = build_tensors();
        let t3 = tensor + tensor2.clone();
        let t4 = t3 / tensor2;
        assert_eq!(t4.shape(), &[2, 2]);
        assert_eq!(t4.ndarray()[[0, 0]], 2f64);
    }

    #[test]
    fn test_tensor_neg() {
        let (_, tensor2) = build_tensors();
        let t3 = -tensor2;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], -1f64);
    }

    #[test]
    fn test_tensor_matmul() {
        let (tensor, tensor2) = build_tensors();
        let t3 = tensor.dot(&tensor2);
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 7f64);
        assert_eq!(t3.ndarray()[[1, 0]], 15f64);
    }

    #[test]
    fn test_tensor_sum() {
        let (tensor, _) = build_tensors();
        let sum = tensor.sum();
        assert_eq!(sum.shape(), &[1]);
        assert_eq!(sum.data[0], (1 + 2 + 3 + 4) as f64);
    }

    #[test]
    fn test_tensor_pow() {
        let (tensor, _) = build_tensors();
        let sum = tensor.pow(2);
        assert_eq!(sum.shape(), &[2, 2]);
        assert_eq!(sum.data[[1, 0]], 9_f64);
    }
}
