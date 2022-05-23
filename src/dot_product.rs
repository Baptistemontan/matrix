pub trait DotProduct<Rhs = Self> {
    type Output;

    fn dot_product(self, other: Rhs) -> Self::Output;
}
