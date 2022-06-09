use std::ops::{Deref, DerefMut};

use crate::static_vector::{StaticRowVector, StaticVector};

macro_rules! impl_static_matrix {
    ($name:ident, $row_vector:ident) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name<const N: usize, const M: usize, T = f64>([$row_vector<M, T>; N]);

        impl<const N: usize, const M: usize, T> Deref for $name<N, M, T> {
            type Target = [$row_vector<M, T>; N];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<const N: usize, const M: usize, T> DerefMut for $name<N, M, T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<const N: usize, const M: usize, T: Default> Default for $name<N, M, T> {
            fn default() -> Self {
                // weird thing to do, but arrays of know sizes can only be initialized with Copy types
                // so to get around that, we create an array of unit type that we map to a $row_vector
                // this will be optimized away by the compiler, because the array of unit type is 0 sized
                [(); N].map(|_| $row_vector::default()).into()
            }
        }

        impl<const N: usize, const M: usize, T: Clone> $name<N, M, T> {
            pub fn new_filled(value: T) -> Self {
                // same thing as Default::default()
                [(); N]
                    .map(|_| $row_vector::new_filled(value.clone()))
                    .into()
            }

            pub fn size(&self) -> (usize, usize) {
                // most useless function but ok
                (N, M)
            }
        }

        impl<const N: usize, const M: usize, T> From<[$row_vector<M, T>; N]> for $name<N, M, T> {
            fn from(data: [$row_vector<M, T>; N]) -> Self {
                $name(data)
            }
        }

        impl<const N: usize, const M: usize, T> From<[[T; M]; N]> for $name<N, M, T> {
            fn from(data: [[T; M]; N]) -> Self {
                data.map($row_vector::from).into()
            }
        }
    };
}

impl_static_matrix!(StaticMatrix, StaticRowVector);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let matrix: StaticMatrix<3, 3> = StaticMatrix::default();
        assert_eq!(matrix.size(), (3, 3));
    }
}
