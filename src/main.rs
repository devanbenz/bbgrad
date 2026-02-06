#![allow(dead_code)]

use std::any::TypeId;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use ndarray::{ArrayBase, IxDyn, OwnedRepr};

#[derive(Debug)]
enum TensorDevice {
    Cpu,
    Gpu,
}

#[derive(Debug)]
enum TensorOp {
    Add,
    Sub,
}

#[derive(Debug, Clone)]
enum TensorDtype {
    Float64,
    Float32,
    Int64,
    Int32,
}

#[derive(Debug)]
enum TensorInner<T>
where
    T: Clone + Debug,
{
    List(Vec<T>),
    Scalar(T),
    Tensor(Tensor<T>),
    NdArray(ArrayBase<OwnedRepr<T>, IxDyn, T>),
}

struct TensorData<T>
where
    T: Clone + Debug,
{
    inner: TensorInner<T>,
    dtype: TensorDtype,
}

impl<T> TensorData<T>
where
    T: Clone + Debug + 'static,
{
    pub fn new(dtype: TensorDtype, inner: TensorInner<T>) -> Self {
        match dtype {
            TensorDtype::Float64 => {
                assert_eq!(TypeId::of::<f64>(), TypeId::of::<T>())
            }
            TensorDtype::Float32 => {
                assert_eq!(TypeId::of::<f32>(), TypeId::of::<T>())
            }
            TensorDtype::Int64 => {
                assert_eq!(TypeId::of::<i64>(), TypeId::of::<T>())
            }
            TensorDtype::Int32 => {
                assert_eq!(TypeId::of::<i32>(), TypeId::of::<T>())
            }
        }
        Self { inner, dtype }
    }
}

#[derive(Debug)]
struct Tensor<T>
where
    T: Clone + Debug,
{
    data: ArrayBase<OwnedRepr<T>, IxDyn, T>,
    grad: Option<i32>,
    requires_grad: bool,
    input: Option<Rc<Tensor<T>>>,
    device: TensorDevice,
    dtype: TensorDtype,
    op: Option<TensorOp>,
}

impl<T> Tensor<T>
where
    T: Clone + Debug,
{
    // TODO: Right now I require a shape for tensors, it would be nice to implement
    // something like a Vec<NestedList<T>> for this and infer the shape similar to
    // what numpy can do. There is overhead to this though, I need to see how python handles
    // this, my assumption is that numpy has some specialized C code for this.
    pub fn new(shape: &[usize], data: TensorData<T>) -> Self {
        let arr: ArrayBase<_, IxDyn, _> = match data.inner {
            TensorInner::List(items) => ArrayBase::from_shape_vec(shape, items).unwrap(),
            TensorInner::Scalar(item) => ArrayBase::from_shape_vec(shape, vec![item]).unwrap(),
            TensorInner::Tensor(tensor) => tensor.data,
            TensorInner::NdArray(array_base) => {
                if array_base.shape() != shape {
                    array_base.to_shape(shape).unwrap().to_owned()
                } else {
                    array_base
                }
            }
        };
        Self {
            data: arr,
            grad: None,
            requires_grad: true,
            input: None,
            device: TensorDevice::Cpu,
            dtype: data.dtype,
            op: None,
        }
    }

    pub fn backward() {
        todo!()
    }
}

impl<T> Display for Tensor<T>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn tensor_data_test() {
        let _ = TensorData::new(TensorDtype::Float64, TensorInner::Scalar(15f64));
        let _ = TensorData::new(
            TensorDtype::Float64,
            TensorInner::Scalar("Should panic!".to_string()),
        );
    }

    #[test]
    fn tensor_test() {
        let _data = TensorData::new(TensorDtype::Float64, TensorInner::Scalar(15f64));
    }
}
