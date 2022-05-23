use std::ops::{Deref, Index, IndexMut};

use crate::vector::{RowVector, Vector};

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix(Vec<RowVector>);

impl Deref for Matrix {
    type Target = [RowVector];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// keeping this implementation for documentation on why it is not implemented
// impl DerefMut for Matrix {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         // DerefMut is a BAD THING to implement, the entire system rely on the fact that
//         // each row is the same size, but here we allow the user to override one of the row with
//         // a vector of a different size
//         // BAD, VERY BAD
//         &mut self.0
//     }
// }

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.0[row][col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.0[row][col]
    }
}

impl FromIterator<RowVector> for Matrix {
    fn from_iter<T: IntoIterator<Item = RowVector>>(iter: T) -> Self {
        let data = iter.into_iter().collect();
        Matrix(data)
    }
}

impl From<Vec<RowVector>> for Matrix {
    fn from(data: Vec<RowVector>) -> Self {
        Matrix(data)
    }
}

impl<const N:usize> From<[RowVector; N]> for Matrix {
    fn from(data: [RowVector; N]) -> Self {
        Matrix(data.into_iter().collect())
    }
}

impl<const N:usize, const M: usize> From<[[f64; M]; N]> for Matrix {
    fn from(data: [[f64; M]; N]) -> Self {
        Matrix(data.into_iter().map(|row| RowVector::from(row)).collect())
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix(vec![RowVector::new(cols); rows])
    }

    pub fn new_filled(rows: usize, cols: usize, value: f64) -> Self {
        Matrix(vec![RowVector::new_filled(cols, value); rows])
    }

    pub fn size(&self) -> (usize, usize) {
        (self.0.len(), self.0[0].len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let matrix: Matrix = Matrix::new(3, 3);
        assert_eq!(matrix.size(), (3, 3));
    }
}
