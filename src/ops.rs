use std::fmt::Debug;
use std::ops;

use crate::tensor::{Tensor, TensorData};

impl<T> ops::Add for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        let combined_data = &self.data + &rhs.data;
        let data = TensorData::new(
            self.dtype(),
            crate::tensor::TensorInner::NdArray(combined_data),
        );
        let inputs = vec![self, rhs];
        Tensor::with_inputs(data, inputs, None)
    }
}
