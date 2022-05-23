use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{dot_product::DotProduct, static_matrix::StaticMatrix};

pub trait StaticVector<const N: usize>:
    Sized + DerefMut<Target = [f64; N]> + Clone + From<[f64; N]> + Into<[f64; N]> + Default
{
    type TransposeTo: StaticVector<N>;

    fn transpose(self) -> Self::TransposeTo {
        self.into().into()
    }

    fn new() -> Self {
        Self::default()
    }

    fn new_filled(value: f64) -> Self {
        [value; N].into()
    }

    fn combine<F: FnMut(f64, f64) -> f64>(&self, other: &Self, mut combiner: F) -> Self {
        let mut i = 0;
        self.deref()
            .map(|x| {
                let val = combiner(x, other[i]);
                i += 1;
                val
            })
            .into()
    }

    fn combine_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut combiner: F) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| {
            combiner(a, *b);
        });
    }

    fn map<F: FnMut(f64) -> f64>(&self, map_fn: F) -> Self {
        self.deref().map(map_fn).into()
    }

    fn map_mut<F: FnMut(&mut f64)>(&mut self, map_fn: F) {
        self.iter_mut().for_each(map_fn);
    }

    fn norme(&self) -> f64;

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
    ($name:ident, $transpose_to:ident) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name<const N: usize>([f64; N]);

        impl $name<3> {
            pub fn cross_product(&self, other: &Self) -> Self {
                let Self([a1, a2, a3]) = self;
                let Self([b1, b2, b3]) = other;
                Self([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])
            }
        }

        impl<const N: usize> Default for $name<N> {
            // can't derive Default for [f64; N]
            fn default() -> Self {
                Self([0.0; N])
            }
        }

        impl<const N: usize> Deref for $name<N> {
            type Target = [f64; N];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<const N: usize> DerefMut for $name<N> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<const N: usize> From<[f64; N]> for $name<N> {
            fn from(data: [f64; N]) -> Self {
                $name(data)
            }
        }

        impl<const N: usize> Into<[f64; N]> for $name<N> {
            fn into(self) -> [f64; N] {
                self.0
            }
        }

        impl<const N: usize> StaticVector<N> for $name<N> {
            type TransposeTo = $transpose_to<N>;

            fn norme(&self) -> f64 {
                self.dot_product(self).sqrt()
            }
        }

        impl<const N: usize> Neg for $name<N> {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self.map_mut(|x| *x = x.neg());
                self
            }
        }

        impl<const N: usize> Neg for &$name<N> {
            type Output = $name<N>;

            fn neg(self) -> Self::Output {
                self.map(f64::neg)
            }
        }

        impl<const N: usize> AddAssign<&Self> for $name<N> {
            fn add_assign(&mut self, other: &Self) {
                self.combine_mut(other, f64::add_assign)
            }
        }

        impl<const N: usize> AddAssign for $name<N> {
            fn add_assign(&mut self, other: Self) {
                *self += &other;
            }
        }

        impl<const N: usize> Add<&Self> for $name<N> {
            type Output = Self;

            fn add(mut self, other: &Self) -> Self::Output {
                self += other;
                self
            }
        }

        impl<const N: usize> Add for $name<N> {
            type Output = Self;

            fn add(mut self, other: Self) -> Self::Output {
                self += other;
                self
            }
        }

        impl<const N: usize> Add for &$name<N> {
            type Output = $name<N>;

            fn add(self, other: Self) -> Self::Output {
                self.combine(other, f64::add)
            }
        }

        impl<const N: usize> SubAssign<&Self> for $name<N> {
            fn sub_assign(&mut self, other: &Self) {
                self.combine_mut(other, f64::sub_assign)
            }
        }

        impl<const N: usize> SubAssign for $name<N> {
            fn sub_assign(&mut self, other: Self) {
                *self -= &other;
            }
        }

        impl<const N: usize> Sub<&Self> for $name<N> {
            type Output = Self;

            fn sub(mut self, other: &Self) -> Self::Output {
                self -= other;
                self
            }
        }

        impl<const N: usize> Sub for $name<N> {
            type Output = Self;

            fn sub(mut self, other: Self) -> Self::Output {
                self -= other;
                self
            }
        }

        impl<const N: usize> Sub for &$name<N> {
            type Output = $name<N>;

            fn sub(self, other: Self) -> Self::Output {
                self.combine(other, f64::sub)
            }
        }

        impl<const N: usize> MulAssign<f64> for $name<N> {
            fn mul_assign(&mut self, scalar: f64) {
                self.map_mut(|x| *x *= scalar)
            }
        }

        impl<const N: usize> Mul<f64> for $name<N> {
            type Output = Self;

            fn mul(mut self, scalar: f64) -> Self::Output {
                self *= scalar;
                self
            }
        }

        impl<const N: usize> Mul<$name<N>> for f64 {
            type Output = $name<N>;

            fn mul(self, mut vector: $name<N>) -> Self::Output {
                vector *= self;
                vector
            }
        }

        impl<const N: usize> Mul<f64> for &$name<N> {
            type Output = $name<N>;

            fn mul(self, scalar: f64) -> Self::Output {
                self.map(|x| x * scalar)
            }
        }

        impl<const N: usize> Mul<&$name<N>> for f64 {
            type Output = $name<N>;

            fn mul(self, vector: &$name<N>) -> Self::Output {
                vector * self
            }
        }

        impl<const N: usize> DivAssign<f64> for $name<N> {
            fn div_assign(&mut self, scalar: f64) {
                self.map_mut(|x| *x /= scalar)
            }
        }

        impl<const N: usize> Div<f64> for $name<N> {
            type Output = Self;

            fn div(mut self, scalar: f64) -> Self::Output {
                self /= scalar;
                self
            }
        }

        impl<const N: usize> Div<$name<N>> for f64 {
            type Output = $name<N>;

            fn div(self, mut vector: $name<N>) -> Self::Output {
                vector /= self;
                vector
            }
        }

        impl<const N: usize> Div<f64> for &$name<N> {
            type Output = $name<N>;

            fn div(self, scalar: f64) -> Self::Output {
                self.map(|x| x / scalar)
            }
        }

        impl<const N: usize> Div<&$name<N>> for f64 {
            type Output = $name<N>;

            fn div(self, vector: &$name<N>) -> Self::Output {
                vector / self
            }
        }

        impl<const N: usize> DotProduct for &$name<N> {
            type Output = f64;

            fn dot_product(self, other: Self) -> Self::Output {
                self.iter().zip(other.iter()).map(|(x, y)| x * y).sum()
            }
        }
    };
}

impl_vector!(StaticRowVector, StaticColumnVector);
impl_vector!(StaticColumnVector, StaticRowVector);

impl<const N: usize> DotProduct<&StaticColumnVector<N>> for &StaticRowVector<N> {
    type Output = f64;

    fn dot_product(self, col: &StaticColumnVector<N>) -> Self::Output {
        self.iter().zip(col.iter()).map(|(x, y)| x * y).sum()
    }
}

impl<const N: usize, const M: usize> DotProduct<&StaticRowVector<M>> for &StaticColumnVector<N> {
    type Output = StaticMatrix<N, M>;

    fn dot_product(self, row: &StaticRowVector<M>) -> Self::Output {
        self.deref().map(|x| row * x).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let row = StaticRowVector::from([1.0, 2.0, 3.0]);
        let col = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let transposed = row.transpose();
        assert_eq!(transposed, col);

        let row = StaticRowVector::from([1.0, 2.0, 3.0]);
        let col = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let transposed = col.transpose();
        assert_eq!(transposed, row);
    }

    #[test]
    fn test_add_row_owned() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        let sum = x + y;
        assert_eq!(sum, StaticRowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_col_owned() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        let sum = x + y;
        assert_eq!(sum, StaticColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_row_ref() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        let sum = &x + &y;
        assert_eq!(sum, StaticRowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_col_ref() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        let sum = &x + &y;
        assert_eq!(sum, StaticColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_row_owned() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        x += y;
        assert_eq!(x, StaticRowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_col_owned() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        x += y;
        assert_eq!(x, StaticColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_row_ref() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        x += &y;
        assert_eq!(x, StaticRowVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_add_assign_col_ref() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        x += &y;
        assert_eq!(x, StaticColumnVector::from([3.0, 9.0, 4.0]));
    }

    #[test]
    fn test_sub_row_owned() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        let diff = x - y;
        assert_eq!(diff, StaticRowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_col_owned() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        let diff = x - y;
        assert_eq!(diff, StaticColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_row_ref() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        let diff = &x - &y;
        assert_eq!(diff, StaticRowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_col_ref() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        let diff = &x - &y;
        assert_eq!(diff, StaticColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_row_owned() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        x -= y;
        assert_eq!(x, StaticRowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_col_owned() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        x -= y;
        assert_eq!(x, StaticColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_row_ref() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticRowVector::from([2.0, 7.0, 1.0]);
        x -= &y;
        assert_eq!(x, StaticRowVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_sub_assign_col_ref() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([2.0, 7.0, 1.0]);
        x -= &y;
        assert_eq!(x, StaticColumnVector::from([-1.0, -5.0, 2.0]));
    }

    #[test]
    fn test_mul_row_owned() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = x * y;
        assert_eq!(prod, StaticRowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_col_owned() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = x * y;
        assert_eq!(prod, StaticColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_row_ref() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = &x * y;
        assert_eq!(prod, StaticRowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_col_ref() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let prod = &x * y;
        assert_eq!(prod, StaticColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_assign_row() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x *= y;
        assert_eq!(x, StaticRowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_mul_assign_col() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x *= y;
        assert_eq!(x, StaticColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_div_row_owned() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = x / y;
        assert_eq!(quot, StaticRowVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_col_owned() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = x / y;
        assert_eq!(quot, StaticColumnVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_row_ref() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = &x / y;
        assert_eq!(quot, StaticRowVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_col_ref() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        let quot = &x / y;
        assert_eq!(quot, StaticColumnVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_assign_row() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x /= y;
        assert_eq!(x, StaticRowVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_div_assign_col() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = 10.0;
        x /= y;
        assert_eq!(x, StaticColumnVector::from([0.1, 0.2, 0.3]));
    }

    #[test]
    fn test_neg_row_owned() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let neg = -x;
        assert_eq!(neg, StaticRowVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_col_owned() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let neg = -x;
        assert_eq!(neg, StaticColumnVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_row_ref() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let neg = -&x;
        assert_eq!(neg, StaticRowVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_neg_col_ref() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let neg = -&x;
        assert_eq!(neg, StaticColumnVector::from([-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_indexing_row() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_indexing_col() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_indexing_row_mut() {
        let mut x = StaticRowVector::from([1.0, 2.0, 3.0]);
        x[0] = 10.0;
        x[1] = 20.0;
        x[2] = 30.0;
        assert_eq!(x, StaticRowVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_indexing_col_mut() {
        let mut x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        x[0] = 10.0;
        x[1] = 20.0;
        x[2] = 30.0;
        assert_eq!(x, StaticColumnVector::from([10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_dot_product() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([1.0, 0.0, 2.0]);
        let dot = x.dot_product(&y);
        assert_eq!(dot, 7.0);
    }

    #[test]
    fn test_dot_product_row_col() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([1.0, 0.0, 2.0]);
        let dot = x.dot_product(&y);
        assert_eq!(dot, 7.0);
    }

    #[test]
    fn test_dot_product_col_row() {
        let x = StaticRowVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([3.0, 2.0]);
        let mat = y.dot_product(&x);
        assert_eq!(
            mat,
            StaticMatrix::from([
                StaticRowVector::from([3.0, 6.0, 9.0]),
                StaticRowVector::from([2.0, 4.0, 6.0])
            ])
        );
    }

    #[test]
    fn test_cross_product() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([1.0, 0.0, 2.0]);
        let cross = x.cross_product(&y);
        assert_eq!(cross, StaticColumnVector::from([4.0, 1.0, -2.0]));
    }
}
