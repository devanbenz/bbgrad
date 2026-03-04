use ndarray::{ArrayBase, ArrayD, Ix1, Ix2, IxDyn, OwnedRepr};
use std::marker::PhantomData;

use super::ForwardType;

#[derive(Debug, Clone)]
pub enum TensorOp<T: ForwardType> {
    Add(TensorAdd<T>),
    Sub(TensorSub<T>),
    Div(TensorDiv<T>),
    Mul(TensorMul<T>),
    Neg(TensorNeg<T>),
    MatMul(TensorMatMul<T>),
    Sum(TensorSum<T>),
    BroadcastTo(TensorBroadcastTo<T>),
    Pow(TensorPow<T>),
    Transpose(TensorTranspose<T>),
    Reshape(TensorReshape<T>),
    Log(TensorLog<T>),
    Exp(TensorExp<T>),
    Relu(TensorRelu<T>),
    Sigmoid(TensorSigmoid<T>),
    Tanh(TensorTanh<T>),
    Sqrt(TensorSqrt<T>),
    Softmax(TensorSoftmax<T>),
    ScalarMul(TensorScalarMul<T>),
    ScalarAdd(TensorScalarAdd<T>),
    ScalarDiv(TensorScalarDiv<T>),
}

#[derive(Debug, Clone)]
pub struct TensorAdd<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorAdd<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSub<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorSub<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorDiv<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorDiv<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMul<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorMul<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorNeg<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorNeg<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMatMul<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorMatMul<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

// TODO: Support higher dimension vectors > (2, 2)
pub(crate) fn dot_dyn<T: ForwardType>(
    a: &ArrayBase<OwnedRepr<T>, IxDyn, T>,
    b: &ArrayBase<OwnedRepr<T>, IxDyn, T>,
) -> ArrayBase<OwnedRepr<T>, IxDyn, T> {
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

#[derive(Debug, Clone)]
pub struct TensorSum<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: ForwardType> TensorSum<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorBroadcastTo<T: ForwardType> {
    pub(crate) marker: PhantomData<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T: ForwardType> TensorBroadcastTo<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorPow<T: ForwardType> {
    marker: PhantomData<T>,
    pub(crate) power: i32,
}

impl<T: ForwardType> TensorPow<T> {
    pub(crate) fn new(power: i32) -> Self {
        Self {
            marker: PhantomData,
            power,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorTranspose<T: ForwardType> {
    marker: PhantomData<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T: ForwardType> TensorTranspose<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorReshape<T: ForwardType> {
    marker: PhantomData<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T: ForwardType> TensorReshape<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorLog<T: ForwardType> {
    marker: PhantomData<T>,
    pub(crate) rhs: T,
}

impl<T: ForwardType> TensorLog<T> {
    pub(crate) fn new(rhs: T) -> Self {
        Self {
            marker: PhantomData,
            rhs,
        }
    }
}

// TensorExp is e^x
#[derive(Debug, Clone)]
pub struct TensorExp<T: ForwardType> {
    marker: PhantomData<T>,
}

impl<T: ForwardType> TensorExp<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorRelu<T: ForwardType> {
    marker: PhantomData<T>,
}

impl<T: ForwardType> TensorRelu<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSigmoid<T: ForwardType> {
    marker: PhantomData<T>,
}

impl<T: ForwardType> TensorSigmoid<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorTanh<T: ForwardType> {
    marker: PhantomData<T>,
}

impl<T: ForwardType> TensorTanh<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSqrt<T: ForwardType> {
    marker: PhantomData<T>,
}

impl<T: ForwardType> TensorSqrt<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSoftmax<T: ForwardType> {
    marker: PhantomData<T>,
}

impl<T: ForwardType> TensorSoftmax<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorScalarMul<T: ForwardType> {
    marker: PhantomData<T>,
    scalar: T,
}

impl<T: ForwardType> TensorScalarMul<T> {
    pub(crate) fn new(scalar: T) -> Self {
        Self {
            marker: PhantomData,
            scalar,
        }
    }

    pub(crate) fn scalar(&self) -> T {
        self.scalar
    }
}

#[derive(Debug, Clone)]
pub struct TensorScalarAdd<T: ForwardType> {
    marker: PhantomData<T>,
    scalar: T,
}

impl<T: ForwardType> TensorScalarAdd<T> {
    pub(crate) fn new(scalar: T) -> Self {
        Self {
            marker: PhantomData,
            scalar,
        }
    }

    pub(crate) fn scalar(&self) -> T {
        self.scalar
    }
}

#[derive(Debug, Clone)]
pub struct TensorScalarDiv<T: ForwardType> {
    marker: PhantomData<T>,
    scalar: T,
}

impl<T: ForwardType> TensorScalarDiv<T> {
    pub(crate) fn new(scalar: T) -> Self {
        Self {
            marker: PhantomData,
            scalar,
        }
    }

    pub(crate) fn scalar(&self) -> T {
        self.scalar
    }
}
