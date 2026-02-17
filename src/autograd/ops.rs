use ndarray::{ArrayBase, ArrayD, Ix1, Ix2, IxDyn, LinalgScalar, OwnedRepr};
use num_traits::Zero;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone)]
pub enum TensorOp<T: Debug + Clone> {
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
    ScalarMul(TensorScalarMul<T>),
    ScalarAdd(TensorScalarAdd<T>),
    ScalarDiv(TensorScalarDiv<T>),
}

#[derive(Debug, Clone)]
pub struct TensorAdd<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + Add<Output = T> + Zero + 'static> TensorAdd<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSub<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + Sub<Output = T> + Zero + 'static> TensorSub<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorDiv<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + Div<Output = T> + Zero + 'static> TensorDiv<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMul<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + Mul<Output = T> + Zero + 'static> TensorMul<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorNeg<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + Neg<Output = T> + Zero + 'static> TensorNeg<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMatMul<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + LinalgScalar + Zero + 'static> TensorMatMul<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

// TODO: Support higher dimension vectors > (2, 2)
pub(crate) fn dot_dyn<T>(
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

#[derive(Debug, Clone)]
pub struct TensorSum<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

impl<T: Clone + Debug + Zero + 'static> TensorSum<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorBroadcastTo<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T: Clone + Debug + Zero + 'static> TensorBroadcastTo<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorPow<T: Debug + Clone> {
    marker: PhantomData<T>,
    pub(crate) power: i32,
}

impl<T: Clone + Debug + Zero + 'static> TensorPow<T> {
    pub(crate) fn new(power: i32) -> Self {
        Self {
            marker: PhantomData,
            power,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorTranspose<T: Debug + Clone> {
    marker: PhantomData<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T: Clone + Debug + Zero + 'static> TensorTranspose<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorReshape<T: Debug + Clone> {
    marker: PhantomData<T>,
    pub(crate) shape: Vec<usize>,
}

impl<T: Clone + Debug + Zero + 'static> TensorReshape<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorLog<T: Debug + Clone> {
    marker: PhantomData<T>,
    pub(crate) rhs: T,
}

impl<T: Clone + Debug + Zero + 'static> TensorLog<T> {
    pub(crate) fn new(rhs: T) -> Self {
        Self {
            marker: PhantomData,
            rhs,
        }
    }
}

// TensorExp is e^x
#[derive(Debug, Clone)]
pub struct TensorExp<T: Debug + Clone> {
    marker: PhantomData<T>,
}

impl<T: Clone + Debug + Zero + 'static> TensorExp<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorRelu<T: Debug + Clone> {
    marker: PhantomData<T>,
}

impl<T: Clone + Debug + Zero + 'static> TensorRelu<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSigmoid<T: Debug + Clone> {
    marker: PhantomData<T>,
}

impl<T: Clone + Debug + Zero + 'static> TensorSigmoid<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorTanh<T: Debug + Clone> {
    marker: PhantomData<T>,
}

impl<T: Clone + Debug + Zero + 'static> TensorTanh<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSqrt<T: Debug + Clone> {
    marker: PhantomData<T>,
}

impl<T: Clone + Debug + Zero + 'static> TensorSqrt<T> {
    pub(crate) fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorScalarMul<T> {
    marker: PhantomData<T>,
    scalar: T,
}

impl<T: Clone + Debug + Zero + 'static> TensorScalarMul<T> {
    pub(crate) fn new(scalar: T) -> Self {
        Self {
            marker: PhantomData,
            scalar,
        }
    }

    pub(crate) fn scalar(&self) -> T {
        self.scalar.clone()
    }
}

#[derive(Debug, Clone)]
pub struct TensorScalarAdd<T> {
    marker: PhantomData<T>,
    scalar: T,
}

impl<T: Clone + Debug + Zero + 'static> TensorScalarAdd<T> {
    pub(crate) fn new(scalar: T) -> Self {
        Self {
            marker: PhantomData,
            scalar,
        }
    }

    pub(crate) fn scalar(&self) -> T {
        self.scalar.clone()
    }
}

#[derive(Debug, Clone)]
pub struct TensorScalarDiv<T> {
    marker: PhantomData<T>,
    scalar: T,
}

impl<T: Clone + Debug + Zero + 'static> TensorScalarDiv<T> {
    pub(crate) fn new(scalar: T) -> Self {
        Self {
            marker: PhantomData,
            scalar,
        }
    }

    pub(crate) fn scalar(&self) -> T {
        self.scalar.clone()
    }
}
