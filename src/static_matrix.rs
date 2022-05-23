use std::ops::{Deref, DerefMut};

use crate::static_vector::{StaticRowVector, StaticVector};

#[derive(Debug, PartialEq, Clone)]
pub struct StaticMatrix<const N: usize, const M: usize>([StaticRowVector<M>; N]);

impl<const N: usize, const M: usize> Deref for StaticMatrix<N, M> {
    type Target = [StaticRowVector<M>; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize, const M: usize> DerefMut for StaticMatrix<N, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize, const M: usize> Default for StaticMatrix<N, M> {
    fn default() -> Self {
        // weird thing to do, but arrays of know sizes can only be initialized with Copy types
        // so to get around that, we create an array of unit type that we map to a StaticRowVector
        // this will be optimized away by the compiler, because the array of unit type is 0 sized
        [(); N].map(|_| StaticRowVector::default()).into()
    }
}

impl<const N: usize, const M: usize> StaticMatrix<N, M> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_filled(value: f64) -> Self {
        // same thing as Default::default()
        [(); N].map(|_| StaticRowVector::new_filled(value)).into()
    }
}

impl<const N: usize, const M: usize> From<[StaticRowVector<M>; N]> for StaticMatrix<N, M> {
    fn from(data: [StaticRowVector<M>; N]) -> Self {
        StaticMatrix(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
