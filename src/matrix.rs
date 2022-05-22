use std::ops::{Deref, DerefMut};

use crate::vector::RowVector;

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

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    use super::*;
}
