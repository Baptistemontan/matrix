use std::ops::{Deref, DerefMut};

use crate::vector::{RowVector, Vector};

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix(Vec<RowVector>);

impl Deref for Matrix {
    type Target = [RowVector];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // DerefMut is a BAD THING to implement, the entire system rely on the fact that
        // each row is the same size, but here we allow the user to override one of the row with
        // a vector of a different size
        // BAD, VERY BAD
        // but eh, f*ck it
        &mut self.0
    }
}

impl FromIterator<RowVector> for Matrix {
    fn from_iter<T: IntoIterator<Item = RowVector>>(iter: T) -> Self {
        let data = iter.into_iter().collect();
        Matrix(data)
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
