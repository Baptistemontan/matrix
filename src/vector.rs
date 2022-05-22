use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorOpError {
    SizeMismatch(usize, usize),
}

type Result<T> = std::result::Result<T, VectorOpError>;

pub trait Vector: Sized + DerefMut<Target = [f64]> + From<Vec<f64>> + FromIterator<f64> {
    type TransposeTo: Vector;

    fn to_vec(self) -> Vec<f64>;

    fn transpose(self) -> Self::TransposeTo {
        self.to_vec().into()
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

        impl Vector for $name {
            type TransposeTo = $transpose_to;

            fn to_vec(self) -> Vec<f64> {
                self.0
            }
        }

        // Add + Sub + Neg + Mul<f64> + Div<f64>
        // all of them Assign

        // Neg
        impl Neg for $name {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self *= -1.0;
                self
            }
        }

        // AddAssign
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

        // Add
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

        // AddAssign
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

        // Add
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

        // MulAssign
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

        // DivAssign
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
