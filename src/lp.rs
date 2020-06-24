//! L<sup>p</sup> spaces.

use crate::coords::Coordinates;

use num_traits::real::Real;
use num_traits::zero;

/// A point in L<sup>1</sup> space.
pub use crate::taxi::Taxicab as L1;

/// Compute the L<sup>1</sup> distance between two points.
pub use crate::taxi::taxicab_distance as l1_distance;

/// A point in L<sup>2</sup> space.
pub use crate::euclid::Euclidean as L2;
/// An L<sup>2</sup> distance.
pub use crate::euclid::EuclideanDistance as L2Distance;

/// Compute the L<sup>2</sup> distance between two points.
pub use crate::euclid::euclidean_distance as l2_distance;

/// A point in L<sup>∞</sup> space.
pub use crate::chebyshev::Chebyshev as Linf;

/// Compute the L<sup>∞</sup> distance between two points.
pub use crate::chebyshev::chebyshev_distance as linf_distance;

/// Compute the [L<sup>p</sup> distance] between two points.
///
/// [L<sup>p</sup> distance]: https://en.wikipedia.org/wiki/Lp_space
pub fn lp_distance<T, U>(p: T::Value, x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    debug_assert!(x.dims() == y.dims());

    let mut sum: T::Value = zero();
    for i in 0..x.dims() {
        sum += (x.coord(i) - y.coord(i)).abs().powf(p);
    }

    sum.powf(p.recip())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp_distance() {
        assert_eq!(l1_distance(&[0.0, 0.0], &[3.0, 4.0]), 7.0);
        assert_eq!(l2_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
        assert!(lp_distance(3.0, &[0.0, 0.0], &[3.0, 4.0]) < 5.0);
        assert_eq!(linf_distance(&[0.0, 0.0], &[3.0, 4.0]), 4.0);
    }
}
