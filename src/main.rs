#![allow(dead_code, unused_imports)]

use std::any::Any;
use std::rc::Rc;

use ndarray::{Array, Array1, ArrayBase, OwnedRepr, Shape};

enum Dtype {
    Float32(f32),
}

struct Tensor<T, D> {
    data: ArrayBase<OwnedRepr<T>, D, T>,
    grad: Option<i32>,
    requires_grad: bool,
    input: Option<Rc<Tensor<T, D>>>,
}

impl<T, D> Tensor<T, D> {
    pub fn new(data: ArrayBase<OwnedRepr<T>, D, T>) -> Self {
        Self {
            data,
            grad: None,
            requires_grad: false,
            input: None,
        }
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_test() {}
}
