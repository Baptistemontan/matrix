use std::{
    error::Error,
    fmt::Display,
    ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{dot_product::DotProduct, matrix::Matrix};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorSizeMismatchError(usize, usize);

impl Display for VectorSizeMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Vector size mismatch: expected size {} but found size {}.",
            self.0, self.1
        )
    }
}

impl Error for VectorSizeMismatchError {}

type Result<T> = std::result::Result<T, VectorSizeMismatchError>;

pub trait Vector<T: Copy + Default + DivAssign>:
    Sized + DerefMut<Target = [T]> + From<Vec<T>> + FromIterator<T> + Into<Vec<T>> + Clone
{
    type TransposeTo: Vector<T>;

    fn transpose(self) -> Self::TransposeTo {
        self.into().into()
    }

    fn new(size: usize) -> Self {
        vec![T::default(); size].into()
    }

    fn new_filled(size: usize, value: T) -> Self {
        vec![value; size].into()
    }

    fn try_combine<F: FnMut(T, T) -> T>(
        &self,
        other: &Self,
        mut combiner: F,
    ) -> Result<Self> {
        if self.len() != other.len() {
            Err(VectorSizeMismatchError(self.len(), other.len()))
        } else {
            Ok(self
                .iter()
                .zip(other.iter())
                .map(|(x, y)| combiner(*x, *y))
                .collect())
        }
    }

    fn combine<F: FnMut(T, T) -> T>(&self, other: &Self, combiner: F) -> Self {
        self.try_combine(other, combiner).unwrap()
    }

    fn try_combine_mut<F: FnMut(&mut T, T)>(
        &mut self,
        other: &Self,
        mut combiner: F,
    ) -> Result<()> {
        if self.len() != other.len() {
            Err(VectorSizeMismatchError(self.len(), other.len()))
        } else {
            self.iter_mut()
                .zip(other.iter())
                .for_each(|(x, y)| combiner(x, *y));
            Ok(())
        }
    }

    fn combine_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, combiner: F) {
        self.try_combine_mut(other, combiner).unwrap();
    }

    fn map<F: FnMut(T) -> T>(&self, map_fn: F) -> Self {
        self.iter().copied().map(map_fn).collect()
    }

    fn map_mut<F: FnMut(&mut T)>(&mut self, map_fn: F) {
        self.iter_mut().for_each(map_fn);
    }

    fn norme(&self) -> T;

    fn normalize(&mut self) {
        let norme = self.norme();
        self.map_mut(|x| *x /= norme);
    }

    fn clone_normalized(&self) -> Self {
        let mut v = self.clone();
        v.normalize();
        v
    }
}

macro_rules! impl_vector {
    ($name:ident, $transpose_to:ident, $data_type:ident) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name(Vec<$data_type>);

        impl Deref for $name {
            type Target = [$data_type];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl From<Vec<$data_type>> for $name {
            fn from(data: Vec<$data_type>) -> Self {
                $name(data)
            }
        }

        impl<const N: usize> From<[$data_type; N]> for $name {
            fn from(data: [$data_type; N]) -> Self {
                $name(data.into())
            }
        }

        impl FromIterator<$data_type> for $name {
            fn from_iter<I: IntoIterator<Item = $data_type>>(iter: I) -> Self {
                let data: Vec<$data_type> = iter.into_iter().collect();
                data.into()
            }
        }

        impl Into<Vec<$data_type>> for $name {
            fn into(self) -> Vec<$data_type> {
                self.0
            }
        }

        impl Vector<$data_type> for $name {
            type TransposeTo = $transpose_to;

            fn norme(&self) -> $data_type {
                self.dot_product(self).unwrap().sqrt()
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self.map_mut(|x| *x = x.neg());
                self
            }
        }

        impl Neg for &$name {
            type Output = $name;

            fn neg(self) -> Self::Output {
                self.map($data_type::neg)
            }
        }

        impl AddAssign<&$name> for $name {
            fn add_assign(&mut self, other: &Self) {
                self.combine_mut(other, $data_type::add_assign)
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                *self += &other;
            }
        }

        impl Add<&$name> for $name {
            type Output = Self;

            fn add(mut self, other: &Self) -> Self::Output {
                self += other;
                self
            }
        }

        impl Add for $name {
            type Output = Self;

            fn add(mut self, other: Self) -> Self::Output {
                self += other;
                self
            }
        }

        impl Add for &$name {
            type Output = $name;

            fn add(self, other: Self) -> Self::Output {
                self.combine(other, $data_type::add)
            }
        }

        impl SubAssign<&$name> for $name {
            fn sub_assign(&mut self, other: &Self) {
                self.combine_mut(other, $data_type::sub_assign)
            }
        }

        impl SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                *self -= &other;
            }
        }

        impl Sub<&$name> for $name {
            type Output = Self;

            fn sub(mut self, other: &Self) -> Self::Output {
                self -= other;
                self
            }
        }

        impl Sub for $name {
            type Output = Self;

            fn sub(mut self, other: Self) -> Self::Output {
                self -= other;
                self
            }
        }

        impl Sub for &$name {
            type Output = $name;

            fn sub(self, other: Self) -> Self::Output {
                self.combine(other, $data_type::sub)
            }
        }

        impl MulAssign<$data_type> for $name {
            fn mul_assign(&mut self, scalar: $data_type) {
                self.map_mut(|x| *x *= scalar)
            }
        }

        impl Mul<$data_type> for $name {
            type Output = Self;

            fn mul(mut self, scalar: $data_type) -> Self::Output {
                self *= scalar;
                self
            }
        }

        impl Mul<$name> for $data_type {
            type Output = $name;

            fn mul(self, mut vector: $name) -> Self::Output {
                vector *= self;
                vector
            }
        }

        impl Mul<$data_type> for &$name {
            type Output = $name;

            fn mul(self, scalar: $data_type) -> Self::Output {
                self.map(|x| x * scalar)
            }
        }

        impl Mul<&$name> for $data_type {
            type Output = $name;

            fn mul(self, vector: &$name) -> Self::Output {
                vector * self
            }
        }

        impl DivAssign<$data_type> for $name {
            fn div_assign(&mut self, scalar: $data_type) {
                self.map_mut(|x| *x /= scalar)
            }
        }

        impl Div<$data_type> for $name {
            type Output = Self;

            fn div(mut self, scalar: $data_type) -> Self::Output {
                self /= scalar;
                self
            }
        }

        impl Div<$name> for $data_type {
            type Output = $name;

            fn div(self, mut vector: $name) -> Self::Output {
                vector /= self;
                vector
            }
        }

        impl Div<$data_type> for &$name {
            type Output = $name;

            fn div(self, scalar: $data_type) -> Self::Output {
                self.map(|x| x / scalar)
            }
        }

        impl Div<&$name> for $data_type {
            type Output = $name;

            fn div(self, vector: &$name) -> Self::Output {
                vector / self
            }
        }

        impl DotProduct for &$name {
            type Output = Result<$data_type>;

            fn dot_product(self, other: Self) -> Self::Output {
                if self.len() != other.len() {
                    Err(VectorSizeMismatchError(self.len(), other.len()))
                } else {
                    Ok(self.iter().zip(other.iter()).map(|(x, y)| x * y).sum())
                }
            }
        }
    };
}

impl_vector!(RowVector, ColumnVector, f64);
impl_vector!(ColumnVector, RowVector, f64);
impl_vector!(RowVectorf32, ColumnVectorf32, f32);
impl_vector!(ColumnVectorf32, RowVectorf32, f32);

impl DotProduct<&ColumnVector> for &RowVector {
    type Output = Result<f64>;

    fn dot_product(self, col: &ColumnVector) -> Self::Output {
        if self.len() != col.len() {
            Err(VectorSizeMismatchError(self.len(), col.len()))
        } else {
            Ok(self.iter().zip(col.iter()).map(|(x, y)| x * y).sum())
        }
    }
}

impl DotProduct<&RowVector> for &ColumnVector {
    type Output = Matrix;

    fn dot_product(self, row: &RowVector) -> Self::Output {
        self.iter().copied().map(|x| row * x).collect()
    }
}

impl DotProduct<&ColumnVectorf32> for &RowVectorf32 {
    type Output = Result<f32>;

    fn dot_product(self, col: &ColumnVectorf32) -> Self::Output {
        if self.len() != col.len() {
            Err(VectorSizeMismatchError(self.len(), col.len()))
        } else {
            Ok(self.iter().zip(col.iter()).map(|(x, y)| x * y).sum())
        }
    }
}

// impl DotProduct<&RowVectorf32> for &ColumnVectorf32 {
//     type Output = Matrixf32;

//     fn dot_product(self, row: &RowVectorf32) -> Self::Output {
//         self.iter().copied().map(|x| row * x).collect()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let row = RowVector::from([1.0, 2.0, 3.0]);
        let col = ColumnVector::from([1.0, 2.0, 3.0]);
        let transposed = row.transpose();
        assert_eq!(transposed, col);

        let row = RowVector::from([1.0, 2.0, 3.0]);
        let col = ColumnVector::from([1.0, 2.0, 3.0]);
        let transposed = col.transpose();
        assert_eq!(transposed, row);
    }

    #[test]
    fn test_add_row_owned() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        let sum = x + y;
        assert_eq!(sum, RowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_col_owned() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        let sum = x + y;
        assert_eq!(sum, ColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_row_ref() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        let sum = &x + &y;
        assert_eq!(sum, RowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_col_ref() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        let sum = &x + &y;
        assert_eq!(sum, ColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_row_owned() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        x += y;
        assert_eq!(x, RowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_col_owned() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        x += y;
        assert_eq!(x, ColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_row_ref() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        x += &y;
        assert_eq!(x, RowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_col_ref() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        x += &y;
        assert_eq!(x, ColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_sub_row_owned() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        let diff = x - y;
        assert_eq!(diff, RowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_col_owned() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        let diff = x - y;
        assert_eq!(diff, ColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_row_ref() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        let diff = &x - &y;
        assert_eq!(diff, RowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_col_ref() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        let diff = &x - &y;
        assert_eq!(diff, ColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_row_owned() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        x -= y;
        assert_eq!(x, RowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_col_owned() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        x -= y;
        assert_eq!(x, ColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_row_ref() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([2.0, 7.0, 1.0]);
        x -= &y;
        assert_eq!(x, RowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_col_ref() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([2.0, 7.0, 1.0]);
        x -= &y;
        assert_eq!(x, ColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_mul_row_owned() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = x * y;
        assert_eq!(prod, RowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_col_owned() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = x * y;
        assert_eq!(prod, ColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_row_ref() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = &x * y;
        assert_eq!(prod, RowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_col_ref() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = &x * y;
        assert_eq!(prod, ColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_assign_row() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x *= y;
        assert_eq!(x, RowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_assign_col() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x *= y;
        assert_eq!(x, ColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_div_row_owned() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = x / y;
        assert_eq!(quot, RowVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_col_owned() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = x / y;
        assert_eq!(quot, ColumnVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_row_ref() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = &x / y;
        assert_eq!(quot, RowVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_col_ref() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = &x / y;
        assert_eq!(quot, ColumnVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_assign_row() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x /= y;
        assert_eq!(x, RowVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_assign_col() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x /= y;
        assert_eq!(x, ColumnVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_neg_row_owned() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let neg = -x;
        assert_eq!(neg, RowVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_col_owned() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let neg = -x;
        assert_eq!(neg, ColumnVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_row_ref() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let neg = -&x;
        assert_eq!(neg, RowVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_col_ref() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let neg = -&x;
        assert_eq!(neg, ColumnVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_indexing_row() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_indexing_col() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_indexing_row_mut() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        x[0] = 10.0;
        x[1] = 20.0;
        x[2] = 30.0;
        assert_eq!(x, RowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_indexing_col_mut() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        x[0] = 10.0;
        x[1] = 20.0;
        x[2] = 30.0;
        assert_eq!(x, ColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    #[should_panic]
    fn test_add_different_size_row() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([1.0, 2.0]);
        let _z = x + y;
    }

    #[test]
    #[should_panic]
    fn test_add_different_size_col() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 2.0]);
        let _z = x + y;
    }

    #[test]
    #[should_panic]
    fn test_sub_different_size_row() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([1.0, 2.0]);
        let _z = x - y;
    }

    #[test]
    #[should_panic]
    fn test_sub_different_size_col() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 2.0]);
        let _z = x - y;
    }

    #[test]
    #[should_panic]
    fn test_add_assign_different_size_row() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([1.0, 2.0]);
        x += y;
    }

    #[test]
    #[should_panic]
    fn test_add_assign_different_size_col() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 2.0]);
        x += y;
    }

    #[test]
    #[should_panic]
    fn test_sub_assign_different_size_row() {
        let mut x = RowVector::from([1.0, 2.0, 3.0]);
        let y = RowVector::from([1.0, 2.0]);
        x -= y;
    }

    #[test]
    #[should_panic]
    fn test_sub_assign_different_size_col() {
        let mut x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 2.0]);
        x -= y;
    }

    #[test]
    fn test_dot_product() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 0.0, 2.0]);
        let dot = x.dot_product(&y).unwrap();
        assert_eq!(dot, 7.0);
    }

    #[test]
    #[should_panic]
    fn test_dot_product_different_size() {
        let x = ColumnVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 0.0]);
        let _dot = x.dot_product(&y).unwrap();
    }

    #[test]
    fn test_dot_product_row_col() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 0.0, 2.0]);
        let dot = x.dot_product(&y).unwrap();
        assert_eq!(dot, 7.0);
    }

    #[test]
    #[should_panic]
    fn test_dot_product_row_col_different_size() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([1.0, 0.0]);
        let _dot = x.dot_product(&y).unwrap();
    }

    #[test]
    fn test_dot_product_col_row() {
        let x = RowVector::from([1.0, 2.0, 3.0]);
        let y = ColumnVector::from([3.0, 2.0]);
        let mat = y.dot_product(&x);
        let expected_matrix = Matrix::from([[3.0, 6.0, 9.0], [2.0, 4.0, 6.0]]);
        assert_eq!(mat, expected_matrix);
    }
}
