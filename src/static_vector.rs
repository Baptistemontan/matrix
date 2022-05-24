use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{dot_product::DotProduct, static_matrix::StaticMatrix};

pub trait StaticVector<const N: usize, T: Clone>:
    Sized + DerefMut<Target = [T; N]> + Clone + From<[T; N]> + Into<[T; N]>
{
    type TransposeTo: StaticVector<N, T>;

    fn transpose(self) -> Self::TransposeTo {
        self.into().into()
    }

    fn new_filled(value: T) -> Self {
        [(); N].map(|_| value.clone()).into()
    }

    fn combine<F: FnMut(T, T) -> T>(&self, other: &Self, mut combiner: F) -> Self {
        let mut i = 0;
        self.deref()
            .clone()
            .map(|x| {
                let val = combiner(x, other[i].clone());
                i += 1;
                val
            })
            .into()
    }

    fn combine_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, mut combiner: F) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| {
            combiner(a, b.clone());
        });
    }

    fn map<F: FnMut(T) -> T>(&self, map_fn: F) -> Self {
        self.deref().clone().map(map_fn).into()
    }

    fn map_mut<F: FnMut(&mut T)>(&mut self, map_fn: F) {
        self.iter_mut().for_each(map_fn);
    }
}

pub trait NormalizeVector<const N: usize, T: DivAssign + Clone>: StaticVector<N, T> {
    fn norme(&self) -> T;

    fn normalize(&mut self) {
        let norme = self.norme();
        self.map_mut(|x| *x /= norme.clone());
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
        pub struct $name<const N: usize, T>([T; N]);

        impl<T: Mul<Output = T> + Sub<Output = T> + Clone> $name<3, T> {
            pub fn cross_product(&self, other: &Self) -> Self {
                let [a1, a2, a3] = self.0.clone();
                let [b1, b2, b3] = other.0.clone();
                Self([
                    a2.clone() * b3.clone() - a3.clone() * b2.clone(),
                    a3 * b1.clone() - a1.clone() * b3,
                    a1 * b2 - a2 * b1,
                ])
            }
        }

        impl<const N: usize, T: Default> Default for $name<N, T> {
            // can't derive Default for [T; N]
            fn default() -> Self {
                [(); N].map(|_| T::default()).into()
            }
        }

        impl<const N: usize, T> Deref for $name<N, T> {
            type Target = [T; N];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<const N: usize, T> DerefMut for $name<N, T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<const N: usize, T> From<[T; N]> for $name<N, T> {
            fn from(data: [T; N]) -> Self {
                $name(data)
            }
        }

        impl<const N: usize, T> Into<[T; N]> for $name<N, T> {
            fn into(self) -> [T; N] {
                self.0
            }
        }

        impl<const N: usize, T: Clone> StaticVector<N, T> for $name<N, T> {
            type TransposeTo = $transpose_to<N, T>;
        }

        impl<const N: usize, T: Neg<Output = T> + Clone> Neg for $name<N, T> {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self.map_mut(|x| *x = x.clone().neg());
                self
            }
        }

        impl<const N: usize, T: Neg<Output = T> + Clone> Neg for &$name<N, T> {
            type Output = $name<N, T>;

            fn neg(self) -> Self::Output {
                StaticVector::map(&self, T::neg)
            }
        }

        impl<const N: usize, T: AddAssign + Clone> AddAssign<&Self> for $name<N, T> {
            fn add_assign(&mut self, other: &Self) {
                self.combine_mut(other, T::add_assign)
            }
        }

        impl<const N: usize, T: AddAssign + Clone> AddAssign for $name<N, T> {
            fn add_assign(&mut self, other: Self) {
                *self += &other;
            }
        }

        impl<const N: usize, T: AddAssign + Clone> Add<&Self> for $name<N, T> {
            type Output = Self;

            fn add(mut self, other: &Self) -> Self::Output {
                self += other;
                self
            }
        }

        impl<const N: usize, T: AddAssign + Clone> Add for $name<N, T> {
            type Output = Self;

            fn add(mut self, other: Self) -> Self::Output {
                self += other;
                self
            }
        }

        impl<const N: usize, T: Add<Output = T> + Clone> Add for &$name<N, T> {
            type Output = $name<N, T>;

            fn add(self, other: Self) -> Self::Output {
                self.combine(other, T::add)
            }
        }

        impl<const N: usize, T: SubAssign + Clone> SubAssign<&Self> for $name<N, T> {
            fn sub_assign(&mut self, other: &Self) {
                self.combine_mut(other, T::sub_assign)
            }
        }

        impl<const N: usize, T: SubAssign + Clone> SubAssign for $name<N, T> {
            fn sub_assign(&mut self, other: Self) {
                *self -= &other;
            }
        }

        impl<const N: usize, T: SubAssign + Clone> Sub<&Self> for $name<N, T> {
            type Output = Self;

            fn sub(mut self, other: &Self) -> Self::Output {
                self -= other;
                self
            }
        }

        impl<const N: usize, T: SubAssign + Clone> Sub for $name<N, T> {
            type Output = Self;

            fn sub(mut self, other: Self) -> Self::Output {
                self -= other;
                self
            }
        }

        impl<const N: usize, T: Sub<Output = T> + Clone> Sub for &$name<N, T> {
            type Output = $name<N, T>;

            fn sub(self, other: Self) -> Self::Output {
                self.combine(other, T::sub)
            }
        }

        impl<const N: usize, T: MulAssign + Clone> MulAssign<T> for $name<N, T> {
            fn mul_assign(&mut self, scalar: T) {
                self.map_mut(|x| *x *= scalar.clone())
            }
        }

        impl<const N: usize, T: MulAssign + Clone> Mul<T> for $name<N, T> {
            type Output = Self;

            fn mul(mut self, scalar: T) -> Self::Output {
                self *= scalar;
                self
            }
        }

        impl<const N: usize> Mul<$name<N, f64>> for f64 {
            type Output = $name<N, f64>;

            fn mul(self, mut vector: $name<N, f64>) -> Self::Output {
                vector *= self;
                vector
            }
        }

        impl<const N: usize> Mul<$name<N, f32>> for f32 {
            type Output = $name<N, f32>;

            fn mul(self, mut vector: $name<N, f32>) -> Self::Output {
                vector *= self;
                vector
            }
        }

        impl<const N: usize, T: Mul<Output = T> + Clone> Mul<T> for &$name<N, T> {
            type Output = $name<N, T>;

            fn mul(self, scalar: T) -> Self::Output {
                self.map(|x| x * scalar.clone())
            }
        }

        impl<const N: usize> Mul<&$name<N, f64>> for f64 {
            type Output = $name<N, f64>;

            fn mul(self, vector: &$name<N, f64>) -> Self::Output {
                vector * self
            }
        }

        impl<const N: usize> Mul<&$name<N, f32>> for f32 {
            type Output = $name<N, f32>;

            fn mul(self, vector: &$name<N, f32>) -> Self::Output {
                vector * self
            }
        }

        impl<const N: usize, T: DivAssign + Clone> DivAssign<T> for $name<N, T> {
            fn div_assign(&mut self, scalar: T) {
                self.map_mut(|x| *x /= scalar.clone())
            }
        }

        impl<const N: usize, T: DivAssign + Clone> Div<T> for $name<N, T> {
            type Output = Self;

            fn div(mut self, scalar: T) -> Self::Output {
                self /= scalar;
                self
            }
        }

        impl<const N: usize> Div<$name<N, f64>> for f64 {
            type Output = $name<N, f64>;

            fn div(self, mut vector: $name<N, f64>) -> Self::Output {
                vector /= self;
                vector
            }
        }

        impl<const N: usize> Div<$name<N, f32>> for f32 {
            type Output = $name<N, f32>;

            fn div(self, mut vector: $name<N, f32>) -> Self::Output {
                vector /= self;
                vector
            }
        }

        impl<const N: usize, T: Div<Output = T> + Clone> Div<T> for &$name<N, T> {
            type Output = $name<N, T>;

            fn div(self, scalar: T) -> Self::Output {
                self.map(|x| x / scalar.clone())
            }
        }

        impl<const N: usize> Div<&$name<N, f64>> for f64 {
            type Output = $name<N, f64>;

            fn div(self, vector: &$name<N, f64>) -> Self::Output {
                vector / self
            }
        }

        impl<const N: usize> Div<&$name<N, f32>> for f32 {
            type Output = $name<N, f32>;

            fn div(self, vector: &$name<N, f32>) -> Self::Output {
                vector / self
            }
        }

        impl<const N: usize, T: Mul<Output = T> + Clone + Sum> DotProduct for &$name<N, T> {
            type Output = T;

            fn dot_product(self, other: Self) -> Self::Output {
                self.iter()
                    .zip(other.iter())
                    .map(|(x, y)| x.clone() * y.clone())
                    .sum()
            }
        }
    };
}

impl_vector!(StaticRowVector, StaticColumnVector);
impl_vector!(StaticColumnVector, StaticRowVector);

impl<const N: usize, T: Clone + Mul<Output = T> + Sum> DotProduct<&StaticColumnVector<N, T>>
    for &StaticRowVector<N, T>
{
    type Output = T;

    fn dot_product(self, col: &StaticColumnVector<N, T>) -> Self::Output {
        self.iter()
            .zip(col.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .sum()
    }
}

// impl<const N: usize, const M: usize, T> DotProduct<&StaticRowVector<M, T>> for &StaticColumnVector<N, T> {
//     type Output = StaticMatrix<N, M, T>;

//     fn dot_product(self, row: &StaticRowVector<M, T>) -> Self::Output {
//         self.deref().map(|x| row * x).into()
//     }
// }

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

    // #[test]
    // fn test_dot_product_col_row() {
    //     let x = StaticRowVector::from([1.0, 2.0, 3.0]);
    //     let y = StaticColumnVector::from([3.0, 2.0]);
    //     let mat = y.dot_product(&x);
    //     assert_eq!(mat, StaticMatrix::from([[3.0, 6.0, 9.0], [2.0, 4.0, 6.0]]));
    // }

    #[test]
    fn test_cross_product() {
        let x = StaticColumnVector::from([1.0, 2.0, 3.0]);
        let y = StaticColumnVector::from([1.0, 0.0, 2.0]);
        let cross = x.cross_product(&y);
        assert_eq!(cross, StaticColumnVector::from([4.0, 1.0, -2.0]));
    }
}
