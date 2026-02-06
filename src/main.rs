#![allow(dead_code)]

use std::fmt::{Debug, Display};
use std::mem::offset_of;
use std::rc::Rc;

use ndarray::{ArrayBase, IxDyn, OwnedRepr};

#[derive(Debug)]
enum TensorDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
enum TensorDtype {
    Float64(f64),
    Float32(f32),
    Int64(i64),
    Int32(i32),
}

#[derive(Debug)]
enum TensorInner {
    List(Vec<TensorDtype>),
    Scalar(TensorDtype),
    Tensor(Tensor),
    NdArray(ArrayBase<OwnedRepr<TensorDtype>, IxDyn, TensorDtype>),
}

struct TensorData {
    inner: TensorInner,
    dtype: TensorDtype,
}

impl TensorData {
    pub fn new(dtype: TensorDtype, inner: TensorInner) -> Self {
        Self { inner, dtype }
    }
}

#[derive(Debug)]
struct Tensor {
    data: ArrayBase<OwnedRepr<TensorDtype>, IxDyn, TensorDtype>,
    grad: Option<i32>,
    requires_grad: bool,
    input: Option<Rc<Tensor>>,
    device: TensorDevice,
    dtype: TensorDtype,
}

impl Tensor {
    pub fn new(shape: &[usize], data: TensorData) -> Self {
        let arr: ArrayBase<_, IxDyn, _> = match data.inner {
            TensorInner::List(items) => ArrayBase::from_shape_vec(shape, items).unwrap(),
            TensorInner::Scalar(item) => ArrayBase::from_shape_vec(shape, vec![item]).unwrap(),
            TensorInner::Tensor(tensor) => tensor.data,
            TensorInner::NdArray(array_base) => {
                if array_base.shape() != shape {
                    let a = array_base.to_shape(shape).unwrap().to_owned();
                    ArrayBase::from(a)
                } else {
                    ArrayBase::from(array_base)
                }
            }
        };
        Self {
            data: arr,
            grad: None,
            requires_grad: false,
            input: None,
            device: TensorDevice::Cpu,
            dtype: data.dtype,
        }
    }

    pub fn backward() {
        todo!()
    }
}

impl std::ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::{Any, TypeId};

    #[test]
    fn tensor_test() {
        let tensor = Tensor::new(
            &[2, 2],
            TensorData::new(TensorDtype::Float64(_), vec![1., 2., 3., 4.]),
        );
        println!("{:?}", tensor);
        let tensor2 = Tensor::new(&[2, 2], TensorData::Tensor(tensor));
        println!("{:?}", tensor2.data.type_id());
    }
}
