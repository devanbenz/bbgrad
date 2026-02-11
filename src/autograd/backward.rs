#![allow(dead_code)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Add;

use num_traits::Zero;

use super::tensor::{Tensor, TensorBuilder, TensorData, TensorInner, TensorOp};

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

#[derive(Debug, Clone)]
pub struct TensorDiv<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct TensorMul<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct TensorNeg<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct TensorMatMul<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct TensorSum<T: Debug + Clone> {
    pub(crate) marker: PhantomData<T>,
}
