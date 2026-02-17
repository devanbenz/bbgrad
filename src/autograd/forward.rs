#![allow(dead_code)]

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::ops::{TensorScalarAdd, TensorScalarDiv, TensorScalarMul};
use super::tensor::{Tensor, TensorBuilder, TensorData, TensorDataInner};
use crate::autograd::ops::{
    TensorAdd, TensorBroadcastTo, TensorDiv, TensorExp, TensorLog, TensorMatMul, TensorMul,
    TensorNeg, TensorOp, TensorPow, TensorRelu, TensorReshape, TensorSigmoid, TensorSqrt,
    TensorSub, TensorSum, TensorTanh, TensorTranspose, dot_dyn,
};
use ndarray::{ArrayBase, ArrayD, IxDyn, LinalgScalar, OwnedRepr};
use num_traits::{Float, Pow, Zero};

pub(crate) trait Forward<T>
where
    T: Debug + Clone + Add<Output = T> + Zero + num_traits::One + 'static,
{
    fn call(&self, inputs: Vec<Tensor<T>>) -> Tensor<T> {
        let requires_grad = inputs.iter().any(|t| t.requires_grad());
        let dtype = inputs[0].dtype();
        let forward_output =
            self.forward(inputs.iter().map(|f| f.ndarray().clone()).collect::<Vec<
                ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>,
            >>());
        let mut output_tensor = TensorBuilder::new(
            TensorData::new(dtype, TensorDataInner::NdArray(forward_output)),
            None,
        );
        if requires_grad {
            output_tensor.inputs(inputs);
            output_tensor.op(self.operation());
        }

        output_tensor.build()
    }

    fn forward(
        &self,
        inputs: Vec<ArrayBase<OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ArrayBase<OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>;

    fn operation(&self) -> TensorOp<T>;
}

impl<T: Clone + Debug + Add<Output = T> + Zero + num_traits::One + 'static> Forward<T>
    for TensorAdd<T>
{
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

impl<T: Clone + Debug + Sub<Output = T> + Zero + num_traits::One + 'static> Forward<T>
    for TensorSub<T>
{
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

impl<T: Clone + Debug + Div<Output = T> + Zero + num_traits::One + 'static> Forward<T>
    for TensorDiv<T>
{
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

impl<T: Clone + Debug + Mul<Output = T> + Zero + num_traits::One + 'static> Forward<T>
    for TensorMul<T>
{
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

impl<T: Clone + Debug + Neg<Output = T> + Zero + num_traits::One + 'static> Forward<T>
    for TensorNeg<T>
{
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

impl<T: Clone + Debug + LinalgScalar + Zero + num_traits::One + 'static> Forward<T>
    for TensorMatMul<T>
{
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

impl<T: Clone + Debug + Zero + num_traits::One + 'static> Forward<T> for TensorSum<T> {
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

impl<T: Clone + Debug + Zero + num_traits::One + 'static> Forward<T> for TensorBroadcastTo<T> {
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

impl<T: Clone + Debug + Pow<i32, Output = T> + Zero + num_traits::One + 'static> Forward<T>
    for TensorPow<T>
{
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

impl<T: Clone + Debug + Zero + num_traits::One + 'static> Forward<T> for TensorTranspose<T> {
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

impl<T: Clone + Debug + Zero + num_traits::One + 'static> Forward<T> for TensorReshape<T> {
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

impl<T: Clone + Debug + Add<Output = T> + Zero + num_traits::One + ndarray::ScalarOperand + 'static>
    Forward<T> for TensorScalarAdd<T>
{
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "add scalar input requires length of 1");
        &inputs[0] + self.scalar()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::ScalarAdd(self.clone())
    }
}

impl<T: Clone + Debug + Mul<Output = T> + Zero + num_traits::One + ndarray::ScalarOperand + 'static>
    Forward<T> for TensorScalarMul<T>
{
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "mul scalar input requires length of 1");
        &inputs[0] * self.scalar()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::ScalarMul(self.clone())
    }
}

impl<T: Clone + Debug + Div<Output = T> + Zero + num_traits::One + ndarray::ScalarOperand + 'static>
    Forward<T> for TensorScalarDiv<T>
{
    fn forward(
        &self,
        inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>, T> {
        assert_eq!(inputs.len(), 1, "div scalar input requires length of 1");
        &inputs[0] / self.scalar()
    }

    fn operation(&self) -> TensorOp<T> {
        TensorOp::ScalarDiv(self.clone())
    }
}
