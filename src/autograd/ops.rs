use std::fmt::Debug;
use std::marker::{Copy, PhantomData};
use std::ops::{self, Sub};
use std::vec;

use ndarray::{ArrayBase, ArrayD, Ix1, Ix2, IxDyn, LinalgScalar, OwnedRepr, linalg};
use ndarray_rand::rand_distr::num_traits::{self, Zero};

use crate::autograd::backward::{
    Forward, TensorAdd, TensorDiv, TensorMatMul, TensorMul, TensorNeg, TensorSub,
};
use crate::autograd::tensor::{TensorBuilder, TensorData, TensorInner, TensorOp};

use super::tensor::Tensor;

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
        let combined_data = &self.data - &rhs.data;
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data));
        let inputs = vec![self, rhs];
        TensorBuilder::new(data, None)
            .input(inputs)
            .op(TensorOp::Sub(TensorSub {
                marker: PhantomData,
            }))
            .build()
    }
}

impl<T> ops::Mul for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Mul<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        let combined_data = &self.data * &rhs.data;
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data));
        let inputs = vec![self, rhs];
        TensorBuilder::new(data, None)
            .input(inputs)
            .op(TensorOp::Mul(TensorMul {
                marker: PhantomData,
            }))
            .build()
    }
}

impl<T> ops::Div for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Div<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        let combined_data = &self.data / &rhs.data;
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data));
        let inputs = vec![self, rhs];
        TensorBuilder::new(data, None)
            .input(inputs)
            .op(TensorOp::Div(TensorDiv {
                marker: PhantomData,
            }))
            .build()
    }
}

impl<T> ops::Neg for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Neg<Output = T> + 'static + Zero,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let combined_data = &self.data.clone().neg();
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data.to_owned()));
        TensorBuilder::new(data, None)
            .op(TensorOp::Neg(TensorNeg {
                marker: PhantomData,
            }))
            .build()
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
        let combined_data = dot_dyn(&self.data, &rhs.data);
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data));
        let inputs = vec![self.to_owned(), rhs.to_owned()];
        TensorBuilder::new(data, None)
            .input(inputs)
            .op(TensorOp::MatMul(TensorMatMul {
                marker: PhantomData,
            }))
            .build()
    }
}

// TODO: Support higher dimension vectors > (2, 2)
fn dot_dyn<T>(
    a: &ArrayBase<OwnedRepr<T>, IxDyn, T>,
    b: &ArrayBase<OwnedRepr<T>, IxDyn, T>,
) -> ArrayBase<OwnedRepr<T>, IxDyn, T>
where
    T: LinalgScalar,
{
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    match (a_ndim, b_ndim) {
        (1, 1) => {
            let a_view = a.view().into_dimensionality::<Ix1>().unwrap();
            let b_view = b.view().into_dimensionality::<Ix1>().unwrap();
            let scalar = a_view.dot(&b_view);
            ArrayD::from_elem(IxDyn(&[]), scalar)
        }
        (2, 1) => {
            let a_view = a.view().into_dimensionality::<Ix2>().unwrap();
            let b_view = b.view().into_dimensionality::<Ix1>().unwrap();
            a_view.dot(&b_view).into_dyn().into_owned()
        }
        (1, 2) => {
            let a_view = a.view().into_dimensionality::<Ix1>().unwrap();
            let b_view = b.view().into_dimensionality::<Ix2>().unwrap();
            a_view.dot(&b_view).into_dyn().into_owned()
        }
        (2, 2) => {
            let a_view = a.view().into_dimensionality::<Ix2>().unwrap();
            let b_view = b.view().into_dimensionality::<Ix2>().unwrap();
            a_view.dot(&b_view).into_dyn().into_owned()
        }
        _ => todo!(),
    }
}

#[cfg(test)]
mod tests {

    use ndarray::linalg::Dot;

    use crate::autograd::tensor::{self, Tensor, TensorData, TensorDtype, TensorInner};

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
}
