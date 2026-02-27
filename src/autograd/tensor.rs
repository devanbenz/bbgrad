#![allow(dead_code)]

use crate::autograd::ForwardType;
use crate::autograd::backward::Backward;

use super::ops::TensorOp;
use ndarray::{ArcArray, ArrayBase, ArrayD, IxDyn, OwnedRepr};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::{any::TypeId, fmt::Display};

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
pub enum TensorDataInner<T: ForwardType> {
    List(Vec<T>),
    Scalar(T),
    Tensor(TensorInner<T>),
    NdArray(ArrayBase<OwnedRepr<T>, IxDyn, T>),
}

pub struct TensorData<T: ForwardType> {
    inner: TensorDataInner<T>,
    dtype: TensorDtype,
}

impl<T: ForwardType> TensorData<T> {
    pub fn new(dtype: TensorDtype, inner: TensorDataInner<T>) -> Self {
        Self { inner, dtype }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T: ForwardType> {
    inner: Arc<RwLock<TensorInner<T>>>,
    inputs: Arc<RwLock<Vec<Tensor<T>>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
}

impl<T: ForwardType> Tensor<T> {
    pub fn new(data: TensorData<T>, shape: Option<&[usize]>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(TensorInner::new(data, shape))),
            inputs: Arc::new(RwLock::new(vec![])),
            grad: Arc::new(RwLock::new(None)),
        }
    }

    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    // This method does not return a reference to the inputs, it's a copy.
    pub fn inputs(&self) -> Vec<Tensor<T>> {
        match self.inputs.read() {
            Ok(val) => val.to_vec(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn backward(&self, grad: Option<Tensor<T>>) {
        let mut grads: HashMap<usize, Tensor<T>> = HashMap::new();
        if grad.is_none() {
            let arr = ArrayD::ones(self.shape());
            grads.insert(
                self.id(),
                Tensor::new(
                    TensorData::new(self.dtype().clone(), TensorDataInner::NdArray(arr)),
                    None,
                ),
            );
        }
        let mut dfs_queue: VecDeque<Tensor<T>> = VecDeque::new();
        let mut _visited: HashSet<usize> = HashSet::new();
        dfs_queue.push_back(self.clone());
        while let Some(node) = dfs_queue.pop_front() {
            if let Some(operation) = node.op() {
                let _output = match operation {
                    TensorOp::Add(tensor_add) => tensor_add.backward(node.clone(), node),
                    TensorOp::Sub(tensor_sub) => tensor_sub.backward(node.clone(), node),
                    TensorOp::Div(tensor_div) => tensor_div.backward(node.clone(), node),
                    TensorOp::Mul(tensor_mul) => tensor_mul.backward(node.clone(), node),
                    TensorOp::Neg(_tensor_neg) => todo!(),
                    TensorOp::MatMul(_tensor_mat_mul) => todo!(),
                    TensorOp::Sum(_tensor_sum) => todo!(),
                    TensorOp::BroadcastTo(_tensor_broadcast_to) => todo!(),
                    TensorOp::Pow(tensor_pow) => tensor_pow.backward(node.clone(), node),
                    TensorOp::Transpose(_tensor_transpose) => todo!(),
                    TensorOp::Reshape(_tensor_reshape) => todo!(),
                    TensorOp::Log(_tensor_log) => todo!(),
                    TensorOp::Exp(_tensor_exp) => todo!(),
                    TensorOp::Relu(_tensor_relu) => todo!(),
                    TensorOp::Sigmoid(_tensor_sigmoid) => todo!(),
                    TensorOp::Tanh(_tensor_tanh) => todo!(),
                    TensorOp::Sqrt(_tensor_sqrt) => todo!(),
                    TensorOp::ScalarMul(_tensor_scalar_mul) => todo!(),
                    TensorOp::ScalarAdd(_tensor_scalar_add) => todo!(),
                    TensorOp::ScalarDiv(_tensor_scalar_div) => todo!(),
                };
            }
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self.inner.read() {
            Ok(val) => val.data.shape().to_vec(),
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn op(&self) -> Option<TensorOp<T>> {
        match self.inner.read() {
            Ok(val) => val.op.clone(),
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

    // TODO: maybe this is not right, I'm just shaping the data and reassigning it
    pub fn reshape(&self, shape: &[usize]) {
        match self.inner.write() {
            Ok(mut inner) => {
                let new_shape = inner.data.to_shape(shape).unwrap();
                inner.data = new_shape.to_owned();
            }
            Err(err) => panic!("{err}"),
        }
    }

    pub fn detach(&self) -> Tensor<T> {
        Tensor {
            inner: self.inner.clone(),
            inputs: self.inputs.clone(),
            grad: self.grad.clone(),
        }
    }

    pub fn graph(&self) {
        fn op_name<T: ForwardType>(op: &TensorOp<T>) -> &str {
            match op {
                TensorOp::Add(_) => "Add",
                TensorOp::Sub(_) => "Sub",
                TensorOp::Div(_) => "Div",
                TensorOp::Mul(_) => "Mul",
                TensorOp::Neg(_) => "Neg",
                TensorOp::MatMul(_) => "MatMul",
                TensorOp::Sum(_) => "Sum",
                TensorOp::BroadcastTo(_) => "BroadcastTo",
                TensorOp::Pow(_) => "Pow",
                TensorOp::Transpose(_) => "Transpose",
                TensorOp::Reshape(_) => "Reshape",
                TensorOp::Log(_) => "Log",
                TensorOp::Exp(_) => "Exp",
                TensorOp::Relu(_) => "Relu",
                TensorOp::Sigmoid(_) => "Sigmoid",
                TensorOp::Tanh(_) => "Tanh",
                TensorOp::Sqrt(_) => "Sqrt",
                TensorOp::ScalarMul(_) => "ScalarMul",
                TensorOp::ScalarAdd(_) => "ScalarAdd",
                TensorOp::ScalarDiv(_) => "ScalarDiv",
            }
        }

        fn print_graph<T: ForwardType>(
            tensor: &Tensor<T>,
            prefix: &str,
            is_last: bool,
            is_root: bool,
            visited: &mut HashSet<usize>,
        ) {
            let connector = if is_root {
                ""
            } else if is_last {
                "└── "
            } else {
                "├── "
            };

            let (shape_str, op_str) = {
                let inner = tensor.inner.read().unwrap();
                let shape_str = format!("{:?}", inner.data.shape());
                let op_str = match &inner.op {
                    Some(op) => format!(" ({})", op_name(op)),
                    None => String::new(),
                };
                (shape_str, op_str)
            };

            let grad_str = match &tensor.grad.read() {
                Ok(g) => {
                    if let Some(ref v) = **g {
                        format!(" grad={:?}", v.grad.read().unwrap())
                    } else {
                        String::new()
                    }
                }
                Err(_) => String::new(),
            };

            let id = tensor.id();
            if visited.contains(&id) {
                println!(
                    "{}{}Tensor {}{}{} *",
                    prefix, connector, shape_str, op_str, grad_str
                );
                return;
            }
            visited.insert(id);

            println!(
                "{}{}Tensor {}{}{}",
                prefix, connector, shape_str, op_str, grad_str
            );

            let child_prefix = if is_root {
                String::new()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };

            let inputs = tensor.inputs();
            for (i, input) in inputs.iter().enumerate() {
                let is_last_child = i == inputs.len() - 1;
                print_graph(input, &child_prefix, is_last_child, false, visited);
            }
        }

        let mut visited = HashSet::new();
        print_graph(self, "", false, true, &mut visited);
    }
}

// TODO: Rename this to TensorInner and wrap it in an Arc within a new Tensor type
#[derive(Debug, Clone)]
pub struct TensorInner<T: ForwardType> {
    pub data: ArrayBase<OwnedRepr<T>, IxDyn, T>,
    grad: Option<ArcArray<T, IxDyn>>,
    requires_grad: bool,
    device: TensorDevice,
    dtype: TensorDtype,
    op: Option<TensorOp<T>>,
}

// TODO: Support nested lists
impl<T: ForwardType> TensorInner<T> {
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

impl<T: ForwardType> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner.read().unwrap().data)
    }
}

impl<T: ForwardType> num_traits::Zero for Tensor<T> {
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
    T: ForwardType,
{
    tensor: Tensor<T>,
}

impl<T: ForwardType> TensorBuilder<T> {
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
    use ndarray::linalg::Dot;

    use crate::autograd::ops_impl::{MatMul, Sigmoid};

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
    fn tensor_data_test() {
        let _ = TensorData::new(TensorDtype::Float64, TensorDataInner::Scalar(15f64));
        // Should panic with f32 as datatype when Float64 is specified
        let _ = TensorData::new(TensorDtype::Float64, TensorDataInner::Scalar(10f32));
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
        let t2 = tensor + 1_f64;

        let t3 = t2.clone();
        assert_eq!(t2.id(), t3.id());
        let t4 = t3.clone() * t2.clone();
        let t4_inputs = t4.inputs.clone().read().unwrap().to_vec();
        assert_eq!(t4_inputs[0].id(), t3.id());
        assert_eq!(t4_inputs[1].id(), t2.id());
    }

    #[test]
    fn tensor_forward_calc_test() {
        let input_data = TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::List(vec![0.9, 0.1, 0.8, 0.2]),
        );
        let t1 = Tensor::new(input_data, Some(&[2, 2]));
        t1.reshape(&[4]);
        let weights = TensorData::new(
            TensorDtype::Float64,
            TensorDataInner::List(vec![1., -1., 1., -1., 1., 1., -1., -1.]),
        );
        let weight_tensor = Tensor::new(weights, Some(&[2, 4]));
        let weighted_sum = weight_tensor.matmul(&t1);
        let weighted_sum = weighted_sum + (-0.5f64);
        let weighted_sum = weighted_sum.sigmoid();
        assert_eq!(format!("{:.2}", weighted_sum.ndarray()[0]), "0.71");
    }
}
