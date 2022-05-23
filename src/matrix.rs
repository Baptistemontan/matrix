use std::ops::{Deref, Index, IndexMut};

use crate::vector::{RowVector, RowVectorf32, Vector};

macro_rules! impl_matrix {
    ($name:ident, $row_vector:ident, $data_type:ident) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name(Vec<$row_vector>);

        impl Deref for $name {
            type Target = [$row_vector];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        // keeping this implementation for documentation on why it is not implemented
        // impl DerefMut for $name {
        //     fn deref_mut(&mut self) -> &mut Self::Target {
        //         // DerefMut is a BAD THING to implement, the entire system rely on the fact that
        //         // each row is the same size, but here we allow the user to override one of the row with
        //         // a vector of a different size
        //         // BAD, VERY BAD
        //         &mut self.0
        //     }
        // }

        impl Index<(usize, usize)> for $name {
            type Output = $data_type;

            fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
                &self.0[row][col]
            }
        }

        impl IndexMut<(usize, usize)> for $name {
            fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
                &mut self.0[row][col]
            }
        }

        impl FromIterator<$row_vector> for $name {
            fn from_iter<T: IntoIterator<Item = $row_vector>>(iter: T) -> Self {
                let data = iter.into_iter().collect();
                $name(data)
            }
        }

        impl From<Vec<$row_vector>> for $name {
            fn from(data: Vec<$row_vector>) -> Self {
                $name(data)
            }
        }

        impl<const N: usize> From<[$row_vector; N]> for $name {
            fn from(data: [$row_vector; N]) -> Self {
                $name(data.into_iter().collect())
            }
        }

        impl<const N: usize, const M: usize> From<[[$data_type; M]; N]> for $name {
            fn from(data: [[$data_type; M]; N]) -> Self {
                $name(data.into_iter().map($row_vector::from).collect())
            }
        }

        impl $name {
            pub fn new(rows: usize, cols: usize) -> Self {
                $name(vec![$row_vector::new(cols); rows])
            }

            pub fn new_filled(rows: usize, cols: usize, value: $data_type) -> Self {
                $name(vec![$row_vector::new_filled(cols, value); rows])
            }

            pub fn size(&self) -> (usize, usize) {
                (self.0.len(), self.0[0].len())
            }
        }
    };
}

impl_matrix!(Matrix, RowVector, f64);
impl_matrix!(Matrixf32, RowVectorf32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let matrix: Matrix = Matrix::new(3, 3);
        assert_eq!(matrix.size(), (3, 3));
    }
}
