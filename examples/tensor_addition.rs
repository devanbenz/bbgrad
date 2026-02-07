use bbgrad::tensor::{Tensor, TensorData, TensorDtype, TensorInner};

fn main() {
    let data = TensorData::new(
        TensorDtype::Float64,
        TensorInner::List(vec![1., 2., 3., 4.]),
    );
    let tensor = Tensor::new(data, Some(&[4]));
    let data2 = TensorData::new(
        TensorDtype::Float64,
        TensorInner::List(vec![1., 2., 3., 4.]),
    );
    let tensor2 = Tensor::new(data2, Some(&[4]));
    let t3 = tensor + tensor2;
    println!("{:?}", t3);
}
