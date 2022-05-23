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
}

#[cfg(test)]
mod tests {
    use super::*;
}
