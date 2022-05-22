use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorOpError {
    SizeMismatch(usize, usize),
}

type Result<T> = std::result::Result<T, VectorOpError>;

pub trait Vector:
    Sized + DerefMut<Target = [f64]> + From<Vec<f64>> + FromIterator<f64> + Into<Vec<f64>>
{
    type TransposeTo: Vector;

    fn transpose(self) -> Self::TransposeTo {
        self.into().into()
    }

    fn new(size: usize) -> Self {
        vec![0.0; size].into()
    }

    fn new_filled(size: usize, value: f64) -> Self {
        vec![value; size].into()
    }

    fn try_combine<F: FnMut(f64, f64) -> f64>(
        &self,
        other: &Self,
        mut combiner: F,
    ) -> Result<Self> {
        if self.len() != other.len() {
            Err(VectorOpError::SizeMismatch(self.len(), other.len()))
        } else {
            Ok(self
                .iter()
                .zip(other.iter())
                .map(|(x, y)| combiner(*x, *y))
                .collect())
        }
    }

    fn combine<F: FnMut(f64, f64) -> f64>(&self, other: &Self, combiner: F) -> Self {
        self.try_combine(other, combiner).unwrap()
    }

    fn try_combine_mut<F: FnMut(&mut f64, f64)>(
        &mut self,
        other: &Self,
        mut combiner: F,
    ) -> Result<()> {
        if self.len() != other.len() {
            Err(VectorOpError::SizeMismatch(self.len(), other.len()))
        } else {
            self.iter_mut()
                .zip(other.iter())
                .for_each(|(x, y)| combiner(x, *y));
            Ok(())
        }
    }

    fn combine_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, combiner: F) {
        self.try_combine_mut(other, combiner).unwrap();
    }

    fn map<F: FnMut(f64) -> f64>(&self, map_fn: F) -> Self {
        self.iter().copied().map(map_fn).collect()
    }

    fn map_mut<F: FnMut(&mut f64)>(&mut self, map_fn: F) {
        self.iter_mut().for_each(map_fn);
    }

    fn norme(&self) -> f64 {
        todo!()
    }
}

macro_rules! impl_vector {
    ($name:ident, $transpose_to:ident) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name(Vec<f64>);

        impl Deref for $name {
            type Target = [f64];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl From<Vec<f64>> for $name {
            fn from(data: Vec<f64>) -> Self {
                $name(data)
            }
        }

        impl FromIterator<f64> for $name {
            fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
                let data: Vec<f64> = iter.into_iter().collect();
                data.into()
            }
        }

        impl Into<Vec<f64>> for $name {
            fn into(self) -> Vec<f64> {
                self.0
            }
        }

        impl Vector for $name {
            type TransposeTo = $transpose_to;
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self *= -1.0;
                self
            }
        }

        impl Neg for &$name {
            type Output = $name;

            fn neg(self) -> Self::Output {
                self.map(f64::neg)
            }
        }

        impl AddAssign<&$name> for $name {
            fn add_assign(&mut self, other: &Self) {
                self.combine_mut(other, f64::add_assign)
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
                self.combine(other, f64::add)
            }
        }

        impl SubAssign<&$name> for $name {
            fn sub_assign(&mut self, other: &Self) {
                self.combine_mut(other, f64::sub_assign)
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
                self.combine(other, f64::sub)
            }
        }

        impl MulAssign<f64> for $name {
            fn mul_assign(&mut self, scalar: f64) {
                self.map_mut(|x| *x *= scalar)
            }
        }

        impl Mul<f64> for $name {
            type Output = Self;

            fn mul(mut self, scalar: f64) -> Self::Output {
                self *= scalar;
                self
            }
        }

        impl Mul<$name> for f64 {
            type Output = $name;

            fn mul(self, mut vector: $name) -> Self::Output {
                vector *= self;
                vector
            }
        }

        impl Mul<f64> for &$name {
            type Output = $name;

            fn mul(self, scalar: f64) -> Self::Output {
                self.map(|x| x * scalar)
            }
        }

        impl Mul<&$name> for f64 {
            type Output = $name;

            fn mul(self, vector: &$name) -> Self::Output {
                vector * self
            }
        }

        impl DivAssign<f64> for $name {
            fn div_assign(&mut self, scalar: f64) {
                self.map_mut(|x| *x /= scalar)
            }
        }

        impl Div<f64> for $name {
            type Output = Self;

            fn div(mut self, scalar: f64) -> Self::Output {
                self /= scalar;
                self
            }
        }

        impl Div<$name> for f64 {
            type Output = $name;

            fn div(self, mut vector: $name) -> Self::Output {
                vector /= self;
                vector
            }
        }

        impl Div<f64> for &$name {
            type Output = $name;

            fn div(self, scalar: f64) -> Self::Output {
                self.map(|x| x / scalar)
            }
        }

        impl Div<&$name> for f64 {
            type Output = $name;

            fn div(self, vector: &$name) -> Self::Output {
                vector / self
            }
        }
    };
}

impl_vector!(RowVector, ColumnVector);
impl_vector!(ColumnVector, RowVector);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let row = RowVector::from(vec![1.0, 2.0, 3.0]);
        let col = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let transposed = row.transpose();
        assert_eq!(transposed, col);

        let row = RowVector::from(vec![1.0, 2.0, 3.0]);
        let col = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let transposed = col.transpose();
        assert_eq!(transposed, row);
    }

    #[test]
    fn test_add_row_owned() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        let sum = x + y;
        assert_eq!(sum, RowVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_col_owned() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        let sum = x + y;
        assert_eq!(sum, ColumnVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_row_ref() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        let sum = &x + &y;
        assert_eq!(sum, RowVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_col_ref() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        let sum = &x + &y;
        assert_eq!(sum, ColumnVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_row_owned() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        x += y;
        assert_eq!(x, RowVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_col_owned() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        x += y;
        assert_eq!(x, ColumnVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_row_ref() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        x += &y;
        assert_eq!(x, RowVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_col_ref() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        x += &y;
        assert_eq!(x, ColumnVector::from(vec![3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_sub_row_owned() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        let diff = x - y;
        assert_eq!(diff, RowVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_col_owned() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        let diff = x - y;
        assert_eq!(diff, ColumnVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_row_ref() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        let diff = &x - &y;
        assert_eq!(diff, RowVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_col_ref() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        let diff = &x - &y;
        assert_eq!(diff, ColumnVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_row_owned() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        x -= y;
        assert_eq!(x, RowVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_col_owned() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        x -= y;
        assert_eq!(x, ColumnVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_row_ref() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![2.0, 7.0, 1.0]);
        x -= &y;
        assert_eq!(x, RowVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_col_ref() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![2.0, 7.0, 1.0]);
        x -= &y;
        assert_eq!(x, ColumnVector::from(vec![-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_mul_row_owned() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = x * y;
        assert_eq!(prod, RowVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_col_owned() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = x * y;
        assert_eq!(prod, ColumnVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_row_ref() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = &x * y;
        assert_eq!(prod, RowVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_col_ref() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = &x * y;
        assert_eq!(prod, ColumnVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_assign_row() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        x *= y;
        assert_eq!(x, RowVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_assign_col() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        x *= y;
        assert_eq!(x, ColumnVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_div_row_owned() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = x / y;
        assert_eq!(quot, RowVector::from(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_col_owned() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = x / y;
        assert_eq!(quot, ColumnVector::from(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_row_ref() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = &x / y;
        assert_eq!(quot, RowVector::from(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_col_ref() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = &x / y;
        assert_eq!(quot, ColumnVector::from(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_assign_row() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        x /= y;
        assert_eq!(x, RowVector::from(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_assign_col() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = 10.0;
        x /= y;
        assert_eq!(x, ColumnVector::from(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_neg_row_owned() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let neg = -x;
        assert_eq!(neg, RowVector::from(vec![-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_col_owned() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let neg = -x;
        assert_eq!(neg, ColumnVector::from(vec![-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_row_ref() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let neg = -&x;
        assert_eq!(neg, RowVector::from(vec![-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_col_ref() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let neg = -&x;
        assert_eq!(neg, ColumnVector::from(vec![-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_indexing_row() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_indexing_col() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_indexing_row_mut() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        x[0] = 10.0;
        x[1] = 20.0;
        x[2] = 30.0;
        assert_eq!(x, RowVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_indexing_col_mut() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        x[0] = 10.0;
        x[1] = 20.0;
        x[2] = 30.0;
        assert_eq!(x, ColumnVector::from(vec![10.0, 20.0, 30.0]));
    }

    #[test]
    #[should_panic]
    fn test_add_different_size_row() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![1.0, 2.0]);
        let _z = x + y;
    }

    #[test]
    #[should_panic]
    fn test_add_different_size_col() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![1.0, 2.0]);
        let _z = x + y;
    }

    #[test]
    #[should_panic]
    fn test_sub_different_size_row() {
        let x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![1.0, 2.0]);
        let _z = x - y;
    }

    #[test]
    #[should_panic]
    fn test_sub_different_size_col() {
        let x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![1.0, 2.0]);
        let _z = x - y;
    }

    #[test]
    #[should_panic]
    fn test_add_assign_different_size_row() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![1.0, 2.0]);
        x += y;
    }

    #[test]
    #[should_panic]
    fn test_add_assign_different_size_col() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![1.0, 2.0]);
        x += y;
    }

    #[test]
    #[should_panic]
    fn test_sub_assign_different_size_row() {
        let mut x = RowVector::from(vec![1.0, 2.0, 3.0]);
        let y = RowVector::from(vec![1.0, 2.0]);
        x -= y;
    }

    #[test]
    #[should_panic]
    fn test_sub_assign_different_size_col() {
        let mut x = ColumnVector::from(vec![1.0, 2.0, 3.0]);
        let y = ColumnVector::from(vec![1.0, 2.0]);
        x -= y;
    }
}
