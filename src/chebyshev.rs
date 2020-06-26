//! [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance).

use crate::coords::{Coordinates, CoordinateMetric, CoordinateProximity};
use crate::distance::{Metric, Proximity};

use num_traits::{zero, Signed};

/// A point in Chebyshev space.
///
/// This wrapper equips any [coordinate space] with the [Chebyshev distance] metric.
///
/// [coordinate space]: [Coordinates]
/// [Chebyshev distance]: https://en.wikipedia.org/wiki/Chebyshev_distance
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Chebyshev<T>(pub T);

impl<T> Chebyshev<T> {
    /// Wrap a point.
    pub fn new(point: T) -> Self {
        Self(point)
    }

    /// Unwrap a point.
    pub fn inner(&self) -> &T {
        &self.0
    }

    /// Unwrap a point.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Coordinates> Coordinates for Chebyshev<T> {
    type Value = T::Value;

    fn dims(&self) -> usize {
        self.0.dims()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.0.coord(i)
    }
}

/// Compute the Chebyshev distance between two points.
///
/// ```math
/// \begin{aligned}
/// \mathrm{chebyshev\_distance}(x, y) &= \|x - y\|_\infty \\
/// &= \max_i |x_i - y_i|
/// \end{aligned}
/// ```
pub fn chebyshev_distance<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
{
    debug_assert!(x.dims() == y.dims());

    let mut max = zero();

    for i in 0..x.dims() {
        let diff = (x.coord(i) - y.coord(i)).abs();
        if diff > max {
            max = diff;
        }
    }

    max
}

/// The Chebyshev distance function.
impl<T: Coordinates> Proximity for Chebyshev<T> {
    type Distance = T::Value;

    fn distance(&self, other: &Self) -> Self::Distance {
        chebyshev_distance(self, other)
    }
}

impl<T: Coordinates> Proximity<T> for Chebyshev<T> {
    type Distance = T::Value;

    fn distance(&self, other: &T) -> Self::Distance {
        chebyshev_distance(self, other)
    }
}

impl<T: Coordinates> Proximity<Chebyshev<T>> for T {
    type Distance = T::Value;

    fn distance(&self, other: &Chebyshev<T>) -> Self::Distance {
        chebyshev_distance(self, other)
    }
}

/// Chebyshev distance is a metric.
impl<T: Coordinates> Metric for Chebyshev<T> {}

impl<T: Coordinates> Metric<T> for Chebyshev<T> {}

impl<T: Coordinates> Metric<Chebyshev<T>> for T {}

impl<T: Coordinates> CoordinateProximity<T::Value> for Chebyshev<T> {
    type Distance = T::Value;

    fn distance_to_coords(&self, coords: &[T::Value]) -> Self::Distance {
        chebyshev_distance(self, coords)
    }
}

impl<T: Coordinates> CoordinateMetric<T::Value> for Chebyshev<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        assert_eq!(chebyshev_distance(&[-3, 4], &[4, -3]), 7);

        assert_eq!(Chebyshev([-3, 4]).distance(&Chebyshev([4, -3])), 7);
        assert_eq!(Chebyshev([-3, 4]).distance(&[4, -3]), 7);
        assert_eq!([-3, 4].distance(&Chebyshev([4, -3])), 7);
    }
}
