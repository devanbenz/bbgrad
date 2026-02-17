#[macro_export]
macro_rules! impl_tensor_op {
    ($trait:ident, $method:ident, $op_struct:ident) => {
        impl_tensor_op!(@scalar_lhs $trait, $method, $op_struct,
            [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
        );
        impl_tensor_op!(@scalar_rhs $trait, $method, $op_struct,
            [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64]
        );
    };
    (@scalar_lhs $trait:ident, $method:ident, $op_struct:ident, [$($scalar:ty),*]) => {
        $(
            impl $trait<Tensor<$scalar>> for $scalar {
                type Output = Tensor<$scalar>;

                fn $method(self, rhs: Tensor<$scalar>) -> Tensor<$scalar> {
                    $op_struct::new(self).call(vec![rhs])
                }
            }
        )*
    };
    (@scalar_rhs $trait:ident, $method:ident, $op_struct:ident, [$($scalar:ty),*]) => {
        $(
            impl $trait<$scalar> for Tensor<$scalar> {
                type Output = Tensor<$scalar>;

                fn $method(self, rhs: $scalar) -> Tensor<$scalar> {
                    $op_struct::new(rhs).call(vec![self])
                }
            }
        )*
    };
}
