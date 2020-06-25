//! [Hamming space](https://en.wikipedia.org/wiki/Hamming_space).

use crate::distance::{Metric, Proximity};

use num_traits::PrimInt;

/// A point in Hamming space.
///
/// This wrapper equips any integer with the [Hamming distance] metric.
///
/// [Hamming distance]: https://en.wikipedia.org/wiki/Hamming_distance
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hamming<T>(pub T);

impl<T> Hamming<T> {
    /// Wrap a point.
    pub fn new(point: T) -> Self {
        Self(point)
    }

    /// Unwrap a point.
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// Compute the Hamming distance between two integers.
pub fn hamming_distance<T: PrimInt>(x: T, y: T) -> i32 {
    (x ^ y).count_ones() as i32
}

/// The hamming distance function.
impl<T: PrimInt> Proximity for Hamming<T> {
    type Distance = i32;

    fn distance(&self, other: &Self) -> Self::Distance {
        hamming_distance(self.0, other.0)
    }
}

impl<T: PrimInt> Proximity<T> for Hamming<T> {
    type Distance = i32;

    fn distance(&self, other: &T) -> Self::Distance {
        hamming_distance(self.0, *other)
    }
}

impl<T: PrimInt> Proximity<Hamming<T>> for T {
    type Distance = i32;

    fn distance(&self, other: &Hamming<T>) -> Self::Distance {
        hamming_distance(*self, other.0)
    }
}

/// Hamming distance is a metric.
impl<T: PrimInt> Metric for Hamming<T> {}

impl<T: PrimInt> Metric<T> for Hamming<T> {}

impl<T: PrimInt> Metric<Hamming<T>> for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        assert_eq!(hamming_distance(0, 0xFFFFFFFFu32), 32);

        assert_eq!(Hamming(0xFFFFFFFFu32).distance(&Hamming(0xAAAAAAAAu32)), 16);
        assert_eq!(Hamming(0x55555555u32).distance(&0xAAAAAAAAu32), 32);
        assert_eq!(0xDEADBEEFu32.distance(&Hamming(0xACABACABu32)), 10);
    }
}
