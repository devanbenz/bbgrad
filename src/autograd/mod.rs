pub mod backward;
pub mod forward;
pub mod nn;
pub mod ops;
pub mod ops_impl;
pub(crate) mod scalar_ops_macro;
pub mod tensor;

use std::fmt::Debug;
use std::ops::{Mul, Neg, SubAssign};

use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::{Float, Pow};

pub trait ForwardType:
    Debug + Float + LinalgScalar + ScalarOperand + Pow<i32, Output = Self> + SubAssign
{
}

impl<T> ForwardType for T where
    T: Debug + Float + LinalgScalar + ScalarOperand + Pow<i32, Output = T> + SubAssign
{
}

pub trait BackwardType:
    Clone
    + Debug
    + num_traits::Zero
    + num_traits::One
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Pow<i32, Output = Self>
    + 'static
{
}

impl<T> BackwardType for T where
    T: Clone
        + Debug
        + num_traits::Zero
        + num_traits::One
        + Neg<Output = T>
        + Mul<Output = T>
        + Pow<i32, Output = T>
        + 'static
{
}
