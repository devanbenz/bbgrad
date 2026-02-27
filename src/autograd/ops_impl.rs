#![allow(dead_code)]

use std::ops::{self, Add, Div, Mul};
use std::vec;

use crate::autograd::forward::Forward;
use crate::autograd::ops::{
    TensorAdd, TensorBroadcastTo, TensorDiv, TensorMatMul, TensorMul, TensorNeg, TensorPow,
    TensorReshape, TensorSub, TensorSum, TensorTranspose,
};
use crate::impl_tensor_op;

use super::ForwardType;
use super::ops::{TensorScalarAdd, TensorScalarDiv, TensorScalarMul, TensorSigmoid};
use super::tensor::Tensor;

trait Sum {
    type Output;

    fn sum(&self) -> Self::Output;
}

pub trait Pow {
    type Output;
    type Exp;

    fn pow(&self, exp: Self::Exp) -> Self::Output;
}

pub trait MatMul {
    type Output;

    fn matmul(&self, rhs: &Self::Output) -> Self::Output;
}

pub trait BroastcastTo {
    type Output;
    type Shape;

    fn broadcast_to(&self, exp: Self::Shape) -> Self::Output;
}

trait Reshape {
    type Output;
    type Shape;

    fn reshape(&self, exp: Self::Shape) -> Self::Output;
}

trait Transpose {
    type Output;
    type Shape;

    fn transpose(&self, exp: Self::Shape) -> Self::Output;
}

pub trait Sigmoid {
    type Output;

    fn sigmoid(&self) -> Self::Output;
}

impl<T: ForwardType> ops::Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorAdd::new().call(vec![self, rhs])
    }
}

impl<T: ForwardType> ops::Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorSub::new().call(vec![self, rhs])
    }
}

impl<T: ForwardType> ops::Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorMul::new().call(vec![self, rhs])
    }
}

impl<T: ForwardType> ops::Div for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorDiv::new().call(vec![self, rhs])
    }
}

impl<T: ForwardType> ops::Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        TensorNeg::new().call(vec![self])
    }
}

impl<T: ForwardType> ndarray::linalg::Dot<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn dot(&self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorMatMul::new().call(vec![self.to_owned(), rhs.to_owned()])
    }
}

impl<T: ForwardType> MatMul for Tensor<T> {
    type Output = Tensor<T>;

    fn matmul(&self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        TensorMatMul::new().call(vec![self.to_owned(), rhs.to_owned()])
    }
}

impl<T: ForwardType> Pow for Tensor<T> {
    type Output = Tensor<T>;
    type Exp = i32;

    fn pow(&self, exp: Self::Exp) -> Self::Output {
        TensorPow::new(exp).call(vec![self.to_owned()])
    }
}

impl<T: ForwardType> Sum for Tensor<T> {
    type Output = Tensor<T>;

    fn sum(&self) -> Self::Output {
        TensorSum::new().call(vec![self.to_owned()])
    }
}

impl<T: ForwardType> BroastcastTo for Tensor<T> {
    type Output = Tensor<T>;
    type Shape = Vec<usize>;

    fn broadcast_to(&self, shape: Self::Shape) -> Self::Output {
        TensorBroadcastTo::new(shape).call(vec![self.to_owned()])
    }
}

impl<T: ForwardType> Reshape for Tensor<T> {
    type Output = Tensor<T>;
    type Shape = Vec<usize>;

    fn reshape(&self, shape: Self::Shape) -> Self::Output {
        TensorReshape::new(shape).call(vec![self.to_owned()])
    }
}

impl<T: ForwardType> Transpose for Tensor<T> {
    type Output = Tensor<T>;
    type Shape = Vec<usize>;

    fn transpose(&self, shape: Self::Shape) -> Self::Output {
        TensorTranspose::new(shape).call(vec![self.to_owned()])
    }
}

impl<T: ForwardType> Sigmoid for Tensor<T> {
    type Output = Tensor<T>;

    fn sigmoid(&self) -> Self::Output {
        TensorSigmoid::new().call(vec![self.to_owned()])
    }
}

impl_tensor_op!(Add, add, TensorScalarAdd);
impl_tensor_op!(Mul, mul, TensorScalarMul);
impl_tensor_op!(Div, div, TensorScalarDiv);

#[cfg(test)]
mod tests {
    use crate::autograd::ops_impl::{BroastcastTo, MatMul, Pow, Sum, Transpose};
    use crate::autograd::tensor::{Tensor, TensorData, TensorDataInner, TensorDtype};
    use ndarray::linalg::Dot;

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
        let t4 = tensor.matmul(&tensor2);
        assert_eq!(t4.shape(), &[2, 2]);
        assert_eq!(t4.ndarray()[[0, 0]], 7f64);
        assert_eq!(t4.ndarray()[[1, 0]], 15f64);
    }

    #[test]
    fn test_tensor_sum() {
        let (tensor, _) = build_tensors();
        let sum = tensor.sum();
        assert_eq!(sum.shape(), &[1]);
        assert_eq!(sum.ndarray()[0], (1 + 2 + 3 + 4) as f64);
    }

    #[test]
    fn test_tensor_pow() {
        let (tensor, _) = build_tensors();
        let sum = tensor.pow(2);
        assert_eq!(sum.shape(), &[2, 2]);
        assert_eq!(sum.ndarray()[[1, 0]], 9_f64);
    }

    #[test]
    fn test_tensor_broadcast_to() {
        let (tensor, _) = build_tensors();
        let sum = tensor.broadcast_to(vec![2, 2, 2]);
        assert_eq!(sum.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = Tensor::new(
            TensorData::new(
                TensorDtype::Float64,
                TensorDataInner::List(vec![1., 2., 3., 4.]),
            ),
            None,
        );
        tensor.reshape(&[2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_transpose() {
        let (tensor, _) = build_tensors();
        let sum = tensor.transpose(vec![2, 2]);
        assert_eq!(sum.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_scalar_add() {
        let (tensor, _) = build_tensors();
        let t3 = 1_f64 + tensor.clone();
        let t4 = tensor + 1_f64;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 2f64);
        assert_eq!(t4.shape(), &[2, 2]);
        assert_eq!(t4.ndarray()[[0, 0]], 2f64);
    }

    #[test]
    fn test_tensor_scalar_sub() {
        todo!()
    }

    #[test]
    fn test_tensor_scalar_mul() {
        let (tensor, _) = build_tensors();
        let t3 = tensor.clone() * 10_f64;
        let t4 = 10_f64 * tensor;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 10f64);
        assert_eq!(t3.ndarray()[[1, 0]], 30f64);
        assert_eq!(t4.shape(), &[2, 2]);
        assert_eq!(t4.ndarray()[[0, 0]], 10f64);
        assert_eq!(t4.ndarray()[[1, 0]], 30f64);
    }

    #[test]
    fn test_tensor_scalar_div() {
        let (tensor, tensor2) = build_tensors();
        let t3 = tensor + tensor2.clone();
        let t4 = t3.clone() / 10_f64;
        let t5 = t3 / 10_f64;
        assert_eq!(t4.shape(), &[2, 2]);
        assert_eq!(t4.ndarray()[[0, 0]], 0.2f64);
        assert_eq!(t5.shape(), &[2, 2]);
        assert_eq!(t5.ndarray()[[0, 0]], 0.2f64);
    }
}
