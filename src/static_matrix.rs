use std::ops::{Deref, DerefMut};

use crate::static_vector::{StaticRowVector, StaticVector, StaticRowVectorf32};

macro_rules! impl_static_matrix {
    ($name:ident, $row_vector:ident, $data_type:ident) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name<const N: usize, const M: usize>([$row_vector<M>; N]);

        impl<const N: usize, const M: usize> Deref for $name<N, M> {
            type Target = [$row_vector<M>; N];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<const N: usize, const M: usize> DerefMut for $name<N, M> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<const N: usize, const M: usize> Default for $name<N, M> {
            fn default() -> Self {
                // weird thing to do, but arrays of know sizes can only be initialized with Copy types
                // so to get around that, we create an array of unit type that we map to a $row_vector
                // this will be optimized away by the compiler, because the array of unit type is 0 sized
                [(); N].map(|_| $row_vector::default()).into()
            }
        }

        impl<const N: usize, const M: usize> $name<N, M> {
            pub fn new() -> Self {
                Self::default()
            }

            pub fn new_filled(value: $data_type) -> Self {
                // same thing as Default::default()
                [(); N].map(|_| $row_vector::new_filled(value)).into()
            }

            pub fn size(&self) -> (usize, usize) {
                (N, M)
            }
        }

        impl<const N: usize, const M: usize> From<[$row_vector<M>; N]> for $name<N, M> {
            fn from(data: [$row_vector<M>; N]) -> Self {
                $name(data)
            }
        }

        impl<const N: usize, const M: usize> From<[[$data_type; M]; N]> for $name<N, M> {
            fn from(data: [[$data_type; M]; N]) -> Self {
                data.map($row_vector::from).into()
            }
        }

        
    };
}

impl_static_matrix!(StaticMatrix, StaticRowVector, f64);
impl_static_matrix!(StaticMatrixf32, StaticRowVectorf32, f32);


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let matrix: StaticMatrix<3, 3> = StaticMatrix::default();
        assert_eq!(matrix.size(), (3, 3));
    }
}
