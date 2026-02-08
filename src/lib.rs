pub mod tensor;

#[cfg(test)]
mod tests {

    use crate::tensor::tensor::{Tensor, TensorData, TensorDtype, TensorInner};

    #[test]
    fn test_tensor_addition() {
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
        let t3 = tensor + tensor2;
        assert_eq!(t3.shape(), &[2, 2]);
        assert_eq!(t3.ndarray()[[0, 0]], 2f64);
    }
}
