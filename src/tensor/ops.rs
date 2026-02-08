use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops;

use crate::tensor::tensor::{
    TensorAdd, TensorBuilder, TensorData, TensorInner, TensorOp, TensorSub,
};

use super::tensor::Tensor;

impl<T> ops::Add for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        let combined_data = &self.data + &rhs.data;
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data));
        let inputs = vec![self, rhs];
        TensorBuilder::new(data, None)
            .op(TensorOp::Add(TensorAdd {
                marker: PhantomData,
            }))
            .input(inputs)
            .build()
    }
}

impl<T> ops::Sub for Tensor<T>
where
    T: Clone + Debug + ops::Add<Output = T> + ops::Sub<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        let combined_data = &self.data - &rhs.data;
        let data = TensorData::new(self.dtype(), TensorInner::NdArray(combined_data));
        let inputs = vec![self, rhs];
        TensorBuilder::new(data, None)
            .input(inputs)
            .op(TensorOp::Sub(TensorSub {
                marker: PhantomData,
            }))
            .build()
    }
}

impl ops::FnOnce for TensorOp<T: Debug + Clone> {
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output {
        todo!()
    }
}
