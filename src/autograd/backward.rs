#![allow(dead_code)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use ndarray::{ArrayBase, ArrayD, Ix1, Ix2, IxDyn, LinalgScalar, OwnedRepr};
use num_traits::{Float, Pow, Zero};

use super::tensor::{Tensor, TensorBuilder, TensorData, TensorInner};

//BroadcastTo
//Pow
//Transpose
//Reshape
//Log
//Exp
//Relu
//Sigmoid
//Tanh
//Sqrt

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
}

pub(crate) trait Backward {
    type OutGrad;
    type Node;
    type Output;

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output;
}

pub(crate) trait Forward<T>
where
    T: Debug + Clone + Add<Output = T> + Zero + 'static,
{
    fn call(&self, inputs: Vec<Tensor<T>>) -> Tensor<T> {
        let requires_grad = inputs.iter().any(|t| t.requires_grad());
        let dtype = inputs[0].dtype();
        let forward_output =
            self.forward(inputs.iter().map(|f| f.data.clone()).collect::<Vec<
                ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>,
            >>());
        let mut output_tensor = TensorBuilder::new(
            TensorData::new(dtype, TensorInner::NdArray(forward_output)),
            None,
        );
        if requires_grad {
            output_tensor.input(inputs);
            output_tensor.op(self.operation());
        }

        output_tensor.build()
    }

    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>;

    fn operation(&self) -> TensorOp<T>;
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

impl<T: Clone + Debug + Add<Output = T> + Zero + 'static> Forward<T> for TensorAdd<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 2, "add input requires lenght of 2");
        assert_eq!(inputs[0].ndim(), inputs[1].ndim());
        &inputs[0] + &inputs[1]
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Add(TensorAdd::new())
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

impl<T: Clone + Debug + Sub<Output = T> + Zero + 'static> Forward<T> for TensorSub<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 2, "add input requires lenght of 2");
        assert_eq!(inputs[0].ndim(), inputs[1].ndim());
        &inputs[0] - &inputs[1]
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Sub(TensorSub::new())
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

impl<T: Clone + Debug + Div<Output = T> + Zero + 'static> Forward<T> for TensorDiv<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 2, "add input requires lenght of 2");
        assert_eq!(inputs[0].ndim(), inputs[1].ndim());
        &inputs[0] / &inputs[1]
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Div(TensorDiv::new())
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

impl<T: Clone + Debug + Mul<Output = T> + Zero + 'static> Forward<T> for TensorMul<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 2, "add input requires lenght of 2");
        assert_eq!(inputs[0].ndim(), inputs[1].ndim());
        &inputs[0] * &inputs[1]
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Mul(TensorMul::new())
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

impl<T: Clone + Debug + Neg<Output = T> + Zero + 'static> Forward<T> for TensorNeg<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "neg input requires lenght of 1");
        -inputs[0].clone()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Neg(TensorNeg::new())
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

impl<T: Clone + Debug + LinalgScalar + Zero + 'static> Forward<T> for TensorMatMul<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 2, "matmul input requires length of 2");
        dot_dyn(&inputs[0], &inputs[1])
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::MatMul(TensorMatMul::new())
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

impl<T: Clone + Debug + Zero + 'static> Forward<T> for TensorSum<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "matmul input requires length of 2");
        let sum = inputs[0].sum();
        ArrayD::from_elem(IxDyn(&[1]), sum)
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Sum(TensorSum::new())
    }
}

#[derive(Debug, Clone)]
pub struct TensorBroadcastTo<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
    shape: Vec<usize>,
}

impl<T: Clone + Debug + Zero + 'static> TensorBroadcastTo<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

impl<T: Clone + Debug + Zero + 'static> Forward<T> for TensorBroadcastTo<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "broadcast_to input requires length of 1");
        let data = inputs[0].broadcast(IxDyn(&self.shape));
        assert!(data.is_some());
        data.unwrap().to_owned()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::BroadcastTo(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct TensorPow<T: Debug + Clone> {
    marker: PhantomData<T>,
    power: i32,
}

impl<T: Clone + Debug + Zero + 'static> TensorPow<T> {
    pub(crate) fn new(power: i32) -> Self {
        Self {
            marker: PhantomData,
            power,
        }
    }
}

impl<T: Clone + Debug + Pow<i32, Output = T> + Zero + 'static> Forward<T> for TensorPow<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "pow input requires length of 1");
        inputs[0].mapv(|x| x.pow(self.power))
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Pow(TensorPow::new(self.power))
    }
}

#[derive(Debug, Clone)]
pub struct TensorTranspose<T: Debug + Clone> {
    marker: PhantomData<T>,
    shape: Vec<usize>,
}

impl<T: Clone + Debug + Zero + 'static> TensorTranspose<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

impl<T: Clone + Debug + Zero + 'static> Forward<T> for TensorTranspose<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "transpose input requires length of 1");
        // TODO: this should return a view not an owned tensor
        inputs[0].t().to_owned()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Transpose(TensorTranspose::new(self.clone().shape))
    }
}

#[derive(Debug, Clone)]
pub struct TensorReshape<T: Debug + Clone> {
    marker: PhantomData<T>,
    shape: Vec<usize>,
}

impl<T: Clone + Debug + Zero + 'static> TensorReshape<T> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        Self {
            marker: PhantomData,
            shape,
        }
    }
}

impl<T: Clone + Debug + Zero + 'static> Forward<T> for TensorReshape<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "broadcast_to input requires length of 1");
        let data = inputs[0].to_shape(IxDyn(&self.shape));
        assert!(data.is_ok());
        data.unwrap().to_owned()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Reshape(TensorReshape::new(self.clone().shape))
    }
}

#[derive(Debug, Clone)]
pub struct TensorLog<T: Debug + Clone> {
    marker: PhantomData<T>,
    rhs: T,
}

impl<T: Clone + Debug + Zero + 'static> TensorLog<T> {
    pub(crate) fn new(rhs: T) -> Self {
        Self {
            marker: PhantomData,
            rhs,
        }
    }
}

impl<T: Clone + Debug + Float + 'static> Forward<T> for TensorLog<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "log input requires length of 1");
        inputs[0].log(self.rhs)
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Log(TensorLog::new(self.rhs))
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

impl<T: Clone + Debug + Float + 'static> Forward<T> for TensorExp<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "broadcast_to input requires length of 1");
        inputs[0].exp()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Exp(TensorExp::new())
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

impl<T: Clone + Debug + Float + 'static> Forward<T> for TensorRelu<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "broadcast_to input requires length of 1");
        inputs[0].mapv(|x| x.max(T::zero()))
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Relu(TensorRelu::new())
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

impl<T: Clone + Debug + Neg + Add<T, Output = T> + Pow<T, Output = T> + Float + 'static> Forward<T>
    for TensorSigmoid<T>
{
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "sigmoid input requires length of 1");
        inputs[0].mapv(|x| T::one() / (T::one() + (-x).exp()))
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Sigmoid(TensorSigmoid::new())
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

impl<T: Clone + Debug + Float + 'static> Forward<T> for TensorTanh<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "tanh input requires length of 1");
        inputs[0].tanh()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Tanh(TensorTanh::new())
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

impl<T: Clone + Debug + Float + 'static> Forward<T> for TensorSqrt<T> {
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        // TODO: This should be a view not an owned tensor, but for now, we will make it owned
        assert_eq!(inputs.len(), 1, "sqrt input requires length of 1");
        inputs[0].sqrt()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::Sqrt(TensorSqrt::new())
    }
}
