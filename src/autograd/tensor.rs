#![allow(dead_code)]

use super::ops::TensorOp;
use ndarray::{ArcArray, ArrayBase, ArrayD, IxDyn, OwnedRepr};
use ndarray_rand::rand_distr::num_traits;
use std::ops::Add;
use std::sync::{Arc, LockResult, Mutex, RwLock, RwLockReadGuard};
use std::{
    any::TypeId,
    fmt::{Debug, Display},
};

#[derive(Debug, Clone)]
pub enum TensorDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorDtype {
    Float64,
    Float32,
    Int64,
    Int32,
}

#[derive(Debug)]
pub enum TensorDataInner<T>
where
    T: Clone + Debug,
{
    List(Vec<T>),
    Scalar(T),
    Tensor(TensorInner<T>),
    NdArray(ArrayBase<OwnedRepr<T>, IxDyn, T>),
}

pub struct TensorData<T>
where
    T: Clone + Debug + Add<Output = T>,
{
    inner: TensorDataInner<T>,
    dtype: TensorDtype,
}

impl<T> TensorData<T>
where
    T: Clone + Debug + 'static + Add<Output = T>,
{
    pub fn new(dtype: TensorDtype, inner: TensorDataInner<T>) -> Self {
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

#[derive(Debug, Clone)]
pub struct Tensor<T: Clone + Debug> {
    inner: Arc<RwLock<TensorInner<T>>>,
    inputs: Arc<RwLock<Vec<Tensor<T>>>>,
}

impl<T: Clone + Debug + Add<Output = T> + num_traits::Zero> Tensor<T> {
    pub fn new(data: TensorData<T>, shape: Option<&[usize]>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(TensorInner::new(data, shape))),
            inputs: Arc::new(RwLock::new(vec![])),
        }
    }

    // This method does not return a reference to the inputs, it's a copy.
    pub fn inputs(&self) -> Vec<Tensor<T>> {
        match self.inputs.read() {
            Ok(val) => val.to_vec(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn backward(&self) {
        todo!();
    }

    pub fn shape(&self) -> Vec<usize> {
        match self.inner.read() {
            Ok(val) => val.data.shape().to_vec(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn dtype(&self) -> TensorDtype {
        match self.inner.read() {
            Ok(val) => val.dtype.clone(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn requires_grad(&self) -> bool {
        match self.inner.read() {
            Ok(val) => val.requires_grad,
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn ndim(&self) -> usize {
        match self.inner.read() {
            Ok(val) => val.data.ndim(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn size(&self) -> usize {
        match self.inner.read() {
            Ok(val) => val.data.len(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn device(&self) -> TensorDevice {
        match self.inner.read() {
            Ok(val) => val.device.clone(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn ndarray(&self) -> ArrayBase<OwnedRepr<T>, IxDyn, T> {
        match self.inner.read() {
            Ok(val) => val.data.clone(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn detach(&self) -> Tensor<T> {
        Tensor {
            inner: self.inner.clone(),
            inputs: self.inputs.clone(),
        }
    }
}

// TODO: Rename this to TensorInner and wrap it in an Arc within a new Tensor type
#[derive(Debug, Clone)]
pub struct TensorInner<T>
where
    T: Clone + Debug,
{
    pub data: ArrayBase<OwnedRepr<T>, IxDyn, T>,
    grad: Option<ArcArray<T, IxDyn>>,
    requires_grad: bool,
    device: TensorDevice,
    dtype: TensorDtype,
    op: Option<TensorOp<T>>,
}

// TODO: Support nested lists
impl<T> TensorInner<T>
where
    T: Clone + Debug + Add<Output = T> + num_traits::identities::Zero,
{
    // TODO: Right now I require a shape for tensors, it would be nice to implement
    // something like a Vec<NestedList<T>> for this and infer the shape similar to
    // what numpy can do. There is overhead to this though, I need to see how python handles
    // this, my assumption is that numpy has some specialized C code for this.
    pub fn new(data: TensorData<T>, shape: Option<&[usize]>) -> Self {
        // TODO: Need to check shape vs data to ensure that shape matches the data
        // length we are attempting to make an ArrayBase.
        let arr: ArrayBase<_, IxDyn, _> = match data.inner {
            TensorDataInner::List(items) => {
                if let Some(sh) = shape {
                    let _shape_len: usize = sh.iter().sum();
                    ArrayBase::from_shape_vec(sh, items).unwrap()
                } else {
                    ArrayD::from_shape_vec(IxDyn(&[items.len()]), items).unwrap()
                }
            }
            TensorDataInner::Scalar(item) => {
                assert!(shape.is_some());
                assert_eq!(
                    shape.unwrap(),
                    &[1],
                    "When passing a scalar datatype you need to make sure shape is &[1]"
                );
                ArrayBase::from_shape_vec(shape.unwrap(), vec![item]).unwrap()
            }
            TensorDataInner::Tensor(tensor) => tensor.data,
            TensorDataInner::NdArray(array_base) => {
                if let Some(shape) = shape {
                    if array_base.shape() != shape {
                        array_base.to_shape(shape).unwrap().to_owned()
                    } else {
                        array_base
                    }
                } else {
                    array_base
                }
            }
        };
        Self {
            // TODO: Some of these datatypes should probably be of the RC
            // variant so I don't have to throw clones all over the place.
            data: arr,
            grad: None,
            requires_grad: true,
            device: TensorDevice::Cpu,
            dtype: data.dtype,
            op: None,
        }
    }
}

impl<T> Display for Tensor<T>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner.read().unwrap().data)
    }
}

impl<T: num_traits::Zero + Clone> num_traits::Zero for Tensor<T>
where
    T: Clone + Debug + 'static,
{
    fn zero() -> Self {
        let data = ArrayD::<T>::zeros(IxDyn(&[]));
        Tensor::new(
            TensorData::new(TensorDtype::Float64, TensorDataInner::NdArray(data)),
            None,
        )
    }

    fn is_zero(&self) -> bool {
        self.inner.read().unwrap().data.iter().all(|x| x.is_zero())
    }
}

pub struct TensorBuilder<T>
where
    T: Clone + Debug + num_traits::identities::Zero,
{
    tensor: Tensor<T>,
}

impl<T> TensorBuilder<T>
where
    T: Clone + Debug + Add<Output = T> + num_traits::identities::Zero,
{
    pub fn new(data: TensorData<T>, shape: Option<&[usize]>) -> Self {
        Self {
            tensor: Tensor::new(data, shape),
        }
    }

    pub fn device(&mut self, device: TensorDevice) -> &mut TensorBuilder<T> {
        self.tensor.inner.write().unwrap().device = device;
        self
    }

    pub fn inputs(&mut self, input: Vec<Tensor<T>>) -> &mut TensorBuilder<T> {
        match self.tensor.inputs.write() {
            Ok(mut val) => {
                *val = input;
            }
            Err(err) => panic!("{:?}", err),
        }
        self
    }

    pub fn op(&mut self, op: TensorOp<T>) -> &mut TensorBuilder<T> {
        self.tensor.inner.write().unwrap().op = Some(op);
        self
    }

    pub fn build(self) -> Tensor<T> {
        self.tensor
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use super::*;

    #[test]
    #[should_panic]
    fn tensor_data_test() {
        let _ = TensorData::new(TensorDtype::Float64, TensorDataInner::Scalar(15f64));
        // Should panic with i32 as datatype for inner
        let _ = TensorData::new(TensorDtype::Float64, TensorDataInner::Scalar(10i32));
    }

    #[test]
    fn tensor_test() {
        let data = TensorData::new(TensorDtype::Float64, TensorDataInner::Scalar(15f64));
        let _tensor = Tensor::new(data, Some(&[1]));
        let data = TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::List(vec![1., 2., 3., 4.]),
        );
        let _tensor = Tensor::new(data, Some(&[2, 2]));
        let data = TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::<f64>::NdArray(ArrayD::<f64>::zeros(IxDyn(&[3, 4, 5]))),
        );
        let tensor = Tensor::new(data, None);
        tensor.inner.clone().write().unwrap().data[[2, 2, 2]] += 0.5;
    }
}
