#![allow(dead_code)]

use super::ops::{
    TensorDiv, TensorMatMul, TensorMul, TensorPow, TensorScalarAdd, TensorSigmoid, TensorSoftmax,
    TensorSub, TensorSum, dot_dyn,
};
use super::tensor::{Tensor, TensorData, TensorDataInner};
use super::{ForwardType, ops::TensorAdd};
use ndarray::{ArrayD, IxDyn};

pub trait Backward {
    type OutGrad;
    type Node;
    type Output;

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output;
}

impl<T: ForwardType> Backward for TensorAdd<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a + b) wrt a is 1
        // derivative of (a + b) wrt b is 1
        (out_grade.clone(), out_grade.clone())
    }
}

impl<T: ForwardType> Backward for TensorSub<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        // derivative of (a - b) wrt a is 1
        // derivative of (a - b) wrt b is -1
        (out_grade.clone(), -out_grade.clone())
    }
}

impl<T: ForwardType> Backward for TensorDiv<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, _out_grade: Self::OutGrad, _node: Self::Node) -> Self::Output {
        todo!()
    }
}

impl<T: ForwardType> Backward for TensorMul<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        let inputs = node.inputs();
        assert_eq!(inputs.len(), 2);
        let a = inputs.first().unwrap();
        let b = inputs.get(1).unwrap();

        // derivative of (a * b) wrt a is 1 * b
        // derivative of (a * b) wrt b is 1 * a

        let grad_a_arr = &out_grade.ndarray() * &b.ndarray();
        let grad_b_arr = &out_grade.ndarray() * &a.ndarray();
        let grad_a = Tensor::new(
            TensorData::new(a.dtype(), TensorDataInner::NdArray(grad_a_arr)),
            None,
        );
        let grad_b = Tensor::new(
            TensorData::new(b.dtype(), TensorDataInner::NdArray(grad_b_arr)),
            None,
        );
        (grad_a, grad_b)
    }
}

impl<T: ForwardType> Backward for TensorPow<T> {
    type OutGrad = Tensor<T>;

    type Node = Tensor<T>;

    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        // Pow is a unary operation: f(x) = x^n where n = self.power
        // d/dx (x^n) = n * x^(n-1)

        let inputs = node.inputs();
        assert_eq!(inputs.len(), 1);
        let x = inputs.first().unwrap();

        let n: T = T::from(self.power).unwrap();
        let x_arr = x.ndarray();
        let x_pow = x_arr.mapv(|v| v.pow(self.power - 1));
        let grad_arr = &out_grade.ndarray() * &(&x_pow * n);
        let grad_x = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(grad_arr)),
            None,
        );

        let zeros_arr = ArrayD::zeros(IxDyn(&x.shape()));
        let zeros_tensor = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(zeros_arr)),
            None,
        );
        (grad_x, zeros_tensor)
    }
}

impl<T: ForwardType> Backward for TensorSum<T> {
    type OutGrad = Tensor<T>;
    type Node = Tensor<T>;
    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        // d/dx sum(x) = 1 for each element
        // Broadcast the upstream gradient (shape [1]) to the input shape
        let inputs = node.inputs();
        assert_eq!(inputs.len(), 1);
        let x = inputs.first().unwrap();

        let upstream_val = out_grade.ndarray();
        let grad_arr = ArrayD::from_elem(IxDyn(&x.shape()), upstream_val[IxDyn(&[0])]);
        let grad_x = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(grad_arr)),
            None,
        );

        let zeros_arr = ArrayD::zeros(IxDyn(&x.shape()));
        let zeros = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(zeros_arr)),
            None,
        );
        (grad_x, zeros)
    }
}

impl<T: ForwardType> Backward for TensorMatMul<T> {
    type OutGrad = Tensor<T>;
    type Node = Tensor<T>;
    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        let inputs = node.inputs();
        assert_eq!(inputs.len(), 2);
        let x = inputs.first().unwrap();
        let y = &inputs[1];

        // Z = X @ Y
        // dL/dX = dL/dZ @ Y^T
        // dL/dY = X^T @ dL/dZ

        let upstream = out_grade.ndarray();
        let y_arr = y.ndarray();
        let x_arr = x.ndarray();
        let y_t = y_arr.t().to_owned().into_dyn().into_owned();
        let x_t = x_arr.t().to_owned().into_dyn().into_owned();
        let grad_x_arr = dot_dyn(&upstream, &y_t);
        let grad_y_arr = dot_dyn(&x_t, &upstream);
        let grad_x = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(grad_x_arr)),
            None,
        );
        let grad_y = Tensor::new(
            TensorData::new(y.dtype(), TensorDataInner::NdArray(grad_y_arr)),
            None,
        );
        (grad_x, grad_y)
    }
}

impl<T: ForwardType> Backward for TensorSigmoid<T> {
    type OutGrad = Tensor<T>;
    type Node = Tensor<T>;
    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

        let node_arr = node.ndarray();
        let ones_arr = ArrayD::from_elem(IxDyn(&node.shape()), T::one());
        let sigmoid_grad = &node_arr * &(&ones_arr - &node_arr);
        let grad_arr = &out_grade.ndarray() * &sigmoid_grad;
        let grad_x = Tensor::new(
            TensorData::new(node.dtype(), TensorDataInner::NdArray(grad_arr)),
            None,
        );

        let zeros_arr = ArrayD::zeros(IxDyn(&node.shape()));
        let zeros_tensor = Tensor::new(
            TensorData::new(node.dtype(), TensorDataInner::NdArray(zeros_arr)),
            None,
        );
        (grad_x, zeros_tensor)
    }
}

impl<T: ForwardType> Backward for TensorSoftmax<T> {
    type OutGrad = Tensor<T>;
    type Node = Tensor<T>;
    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        // softmax'(x_i) = y_i * (dL/dy_i - sum(dL/dy * y))
        // where y = softmax(x)
        // node is the softmax output (y)
        let upstream = out_grade.ndarray();
        let y = node.ndarray();

        // sum(dL/dy * y)
        let dot = (&upstream * &y).sum();

        // grad_x = y * (upstream - dot)
        let grad_arr = &y * &(&upstream - dot);
        let grad_x = Tensor::new(
            TensorData::new(node.dtype(), TensorDataInner::NdArray(grad_arr)),
            None,
        );

        let zeros_arr = ArrayD::zeros(IxDyn(&node.shape()));
        let zeros = Tensor::new(
            TensorData::new(node.dtype(), TensorDataInner::NdArray(zeros_arr)),
            None,
        );
        (grad_x, zeros)
    }
}

impl<T: ForwardType> Backward for TensorScalarAdd<T> {
    type OutGrad = Tensor<T>;
    type Node = Tensor<T>;
    type Output = (Tensor<T>, Tensor<T>);

    fn backward(&self, out_grade: Self::OutGrad, node: Self::Node) -> Self::Output {
        // d/dx (x + scalar) = 1, so gradient passes through unchanged
        let inputs = node.inputs();
        assert_eq!(inputs.len(), 1);
        let x = inputs.first().unwrap();

        let zeros_arr = ArrayD::zeros(IxDyn(&x.shape()));
        let zeros = Tensor::new(
            TensorData::new(x.dtype(), TensorDataInner::NdArray(zeros_arr)),
            None,
        );
        (out_grade, zeros)
    }
}

#[cfg(test)]
mod backward_tests {
    use crate::autograd::tensor::{TensorData, TensorDataInner, TensorDtype};

    use super::*;

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
    fn backward_mul() {
        let (t1, t2) = build_tensors();
        let t3 = t1 * t2;
        t3.backward(None);
        t3.graph();
    }
}
