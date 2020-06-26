//! [Taxicab (Manhattan) distance](https://en.wikipedia.org/wiki/Taxicab_geometry).

use crate::coords::{Coordinates, CoordinateMetric, CoordinateProximity};
use crate::distance::{Metric, Proximity};

use num_traits::{zero, Signed};

/// A point in taxicab space.
///
/// This wrapper equips any [coordinate space] with the [taxicab distance metric].
///
/// [coordinate space]: [Coordinates]
/// [taxicab distance metric]: https://en.wikipedia.org/wiki/Taxicab_geometry
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Taxicab<T>(pub T);

impl<T> Taxicab<T> {
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

impl<T: Coordinates> Coordinates for Taxicab<T> {
    type Value = T::Value;

    fn dims(&self) -> usize {
        self.0.dims()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.0.coord(i)
    }
}

/// Compute the taxicab distance between two points.
///
/// ```math
/// \begin{aligned}
/// \mathrm{taxicab\_distance}(x, y) &= \|x - y\|_1 \\
/// &= \sum_i |x_i - y_i|
/// \end{aligned}
/// ```
pub fn taxicab_distance<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
{
    debug_assert!(x.dims() == y.dims());

    let mut sum = zero();
    for i in 0..x.dims() {
        sum += (x.coord(i) - y.coord(i)).abs();
    }

    sum
}

/// The taxicab distance function.
impl<T: Coordinates> Proximity for Taxicab<T> {
    type Distance = T::Value;

    fn distance(&self, other: &Self) -> Self::Distance {
        taxicab_distance(self, other)
    }
}

impl<T: Coordinates> Proximity<T> for Taxicab<T> {
    type Distance = T::Value;

    fn distance(&self, other: &T) -> Self::Distance {
        taxicab_distance(self, other)
    }
}

impl<T: Coordinates> Proximity<Taxicab<T>> for T {
    type Distance = T::Value;

    fn distance(&self, other: &Taxicab<T>) -> Self::Distance {
        taxicab_distance(self, other)
    }
}

/// Taxicab distance is a metric.
impl<T: Coordinates> Metric for Taxicab<T> {}

impl<T: Coordinates> Metric<T> for Taxicab<T> {}

impl<T: Coordinates> Metric<Taxicab<T>> for T {}

impl<T: Coordinates> CoordinateProximity<T::Value> for Taxicab<T> {
    type Distance = T::Value;

    fn distance_to_coords(&self, coords: &[T::Value]) -> Self::Distance {
        taxicab_distance(self, coords)
    }
}

impl<T: Coordinates> CoordinateMetric<T::Value> for Taxicab<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        assert_eq!(taxicab_distance([-3, 4], [4, -3]), 14);

        assert_eq!(Taxicab([-3, 4]).distance(&Taxicab([4, -3])), 14);
        assert_eq!(Taxicab([-3, 4]).distance(&[4, -3]), 14);
        assert_eq!([-3, 4].distance(&Taxicab([4, -3])), 14);
    }
}
