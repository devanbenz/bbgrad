use bbgrad::autograd::ops_impl::BroastcastTo;
use bbgrad::autograd::tensor::{Tensor, TensorData, TensorDtype, TensorInner};

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

fn main() {
    let (tensor, _) = build_tensors();
    let sum = tensor.broadcast_to(vec![2, 2, 2]);
    assert_eq!(sum.shape(), &[2, 2, 2]);
}
