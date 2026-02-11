#![allow(dead_code)]

use std::any::TypeId;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Add;

use ndarray::{ArcArray, ArrayBase, ArrayD, IxDyn, OwnedRepr};

use super::backward::{
    TensorAdd, TensorDiv, TensorMatMul, TensorMul, TensorNeg, TensorSub, TensorSum,
};

#[derive(Debug, Clone)]
pub enum TensorDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
pub enum TensorOp<T: Debug + Clone> {
    Add(TensorAdd<T>),
    Sub(TensorSub<T>),
    Div(TensorDiv<T>),
    Mul(TensorMul<T>),
    Neg(TensorNeg<T>),
    MatMul(TensorMatMul<T>),
    Sum(TensorSum<T>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorDtype {
    Float64,
    Float32,
    Int64,
    Int32,
}

#[derive(Debug)]
pub enum TensorInner<T>
where
    T: Clone + Debug,
{
    List(Vec<T>),
    Scalar(T),
    Tensor(Tensor<T>),
    NdArray(ArrayBase<OwnedRepr<T>, IxDyn, T>),
}

pub struct TensorData<T>
where
    T: Clone + Debug + Add<Output = T>,
{
    inner: TensorInner<T>,
    dtype: TensorDtype,
}

impl<T> TensorData<T>
where
    T: Clone + Debug + 'static + Add<Output = T>,
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

// TODO: Rename this to TensorInner and wrap it in an Arc within a new Tensor type
#[derive(Debug, Clone)]
pub struct Tensor<T>
where
    T: Clone + Debug,
{
    pub data: ArrayBase<OwnedRepr<T>, IxDyn, T>,
    grad: Option<ArcArray<T, IxDyn>>,
    requires_grad: bool,
    input: Option<Vec<Tensor<T>>>,
    device: TensorDevice,
    dtype: TensorDtype,
    op: Option<TensorOp<T>>,
}

// TODO: Support nested lists
impl<T> Tensor<T>
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
            TensorInner::List(items) => {
                assert!(shape.is_some());
                let _shape_len: usize = shape.unwrap().iter().sum();
                ArrayBase::from_shape_vec(shape.unwrap(), items).unwrap()
            }
            TensorInner::Scalar(item) => {
                assert!(shape.is_some());
                assert_eq!(
                    shape.unwrap(),
                    &[1],
                    "When passing a scalar datatype you need to make sure shape is &[1]"
                );
                ArrayBase::from_shape_vec(shape.unwrap(), vec![item]).unwrap()
            }
            TensorInner::Tensor(tensor) => tensor.data,
            TensorInner::NdArray(array_base) => {
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
            input: None,
            device: TensorDevice::Cpu,
            dtype: data.dtype,
            op: None,
        }
    }

    pub fn with_inputs(
        data: TensorData<T>,
        inputs: Vec<Tensor<T>>,
        shape: Option<&[usize]>,
    ) -> Self {
        let mut tensor = Tensor::new(data, shape);
        tensor.input = Some(inputs);
        tensor
    }

    pub fn backward(&self) {
        todo!();
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn dtype(&self) -> TensorDtype {
        self.dtype.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn device(&self) -> TensorDevice {
        self.device.clone()
    }

    pub fn ndarray(&self) -> ArrayBase<OwnedRepr<T>, IxDyn, T> {
        self.data.clone()
    }

    pub fn detach(&self) -> Tensor<T> {
        Tensor {
            data: self.data.clone(),
            grad: self.grad.clone(),
            requires_grad: false,
            input: self.input.clone(),
            device: self.device.clone(),
            dtype: self.dtype.clone(),
            op: self.op.clone(),
        }
    }

    pub fn sum(self) -> Tensor<T>
    where
        T: 'static,
    {
        let sum = self.data.sum();
        let data = TensorData::new(self.dtype(), TensorInner::Scalar(sum.to_owned()));
        TensorBuilder::new(data, Some(&[1]))
            .op(TensorOp::Sum(TensorSum {
                marker: PhantomData,
            }))
            .build()
    }

    // TODO: This should be a view not an owned tensor, but for now, we will make it owned
    pub fn broadcast_to(&self, shape: &[usize]) -> Tensor<T>
    where
        T: 'static,
    {
        let new_arr = self.data.broadcast(shape);
        let data = TensorData::new(
            self.dtype(),
            TensorInner::NdArray(new_arr.unwrap().to_owned()),
        );
        TensorBuilder::new(data, None)
            .op(TensorOp::Sum(TensorSum {
                marker: PhantomData,
            }))
            .build()
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

impl<T: num_traits::Zero + Clone> num_traits::Zero for Tensor<T>
where
    T: Clone + Debug + 'static,
{
    fn zero() -> Self {
        // Create a scalar (0-dim) tensor holding zero
        let data = ArrayD::<T>::zeros(IxDyn(&[]));
        Tensor::new(
            TensorData::new(TensorDtype::Float64, TensorInner::NdArray(data)),
            None,
        )
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
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
        self.tensor.device = device;
        self
    }

    pub fn input(&mut self, input: Vec<Tensor<T>>) -> &mut TensorBuilder<T> {
        self.tensor.input = Some(input);
        self
    }

    pub fn op(&mut self, op: TensorOp<T>) -> &mut TensorBuilder<T> {
        self.tensor.op = Some(op);
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
        let _ = TensorData::new(TensorDtype::Float64, TensorInner::Scalar(15f64));
        // Should panic with i32 as datatype for inner
        let _ = TensorData::new(TensorDtype::Float64, TensorInner::Scalar(10i32));
    }

    #[test]
    fn tensor_test() {
        let data = TensorData::new(TensorDtype::Float64, TensorInner::Scalar(15f64));
        let _tensor = Tensor::new(data, Some(&[1]));
        let data = TensorData::new(
            TensorDtype::Float64,
            TensorInner::List(vec![1., 2., 3., 4.]),
        );
        let _tensor = Tensor::new(data, Some(&[2, 2]));
        let data = TensorData::new(
            TensorDtype::Float64,
            TensorInner::<f64>::NdArray(ArrayD::<f64>::zeros(IxDyn(&[3, 4, 5]))),
        );
        let mut tensor = Tensor::new(data, None);
        tensor.data[[2, 2, 2]] += 0.5;
    }
}
