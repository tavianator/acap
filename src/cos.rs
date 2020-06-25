//! [Cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity).

use crate::coords::Coordinates;
use crate::distance::{Metric, Proximity};

use num_traits::real::Real;
use num_traits::{one, zero};

/// Compute the [cosine *similarity*] between two points.
///
/// This is not suitable for implementing [`Proximity::distance()`] because the result is reversed
///
/// [cosine *similarity*]: https://en.wikipedia.org/wiki/Cosine_similarity
/// [`Proximity::distance()`]: Proximity#method.distance
pub fn cosine_similarity<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    debug_assert!(x.dims() == y.dims());

    let mut dot: T::Value = zero();
    let mut xx: T::Value = zero();
    let mut yy: T::Value = zero();

    for i in 0..x.dims() {
        let xi = x.coord(i);
        let yi = y.coord(i);
        dot += xi * yi;
        xx += xi * xi;
        yy += yi * yi;
    }

    dot / (xx * yy).sqrt()
}

/// Compute the [cosine distance] between two points.
///
/// [cosine distance]: https://en.wikipedia.org/wiki/Cosine_similarity
pub fn cosine_distance<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    let one: T::Value = one();
    one - cosine_similarity(x, y)
}

/// Equips any [coordinate space] with the [cosine distance] function.
///
/// [coordinate space]: [Coordinates]
/// [cosine distance]: https://en.wikipedia.org/wiki/Cosine_similarity
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cosine<T>(pub T);

impl<T> Proximity for Cosine<T>
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &Self) -> Self::Distance {
        cosine_distance(&self.0, &other.0)
    }
}

impl<T> Proximity<T> for Cosine<T>
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &T) -> Self::Distance {
        cosine_distance(&self.0, other)
    }
}

impl<T> Proximity<Cosine<T>> for T
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &Cosine<T>) -> Self::Distance {
        cosine_distance(self, &other.0)
    }
}

/// Compute the [angular distance] between two points.
///
/// [angular distance]: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
pub fn angular_distance<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    cosine_similarity(x, y).acos()
}

/// Equips any [coordinate space] with the [angular distance] metric.
///
/// [coordinate space]: [Coordinates]
/// [angular distance]: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Angular<T>(pub T);

impl<T> Proximity for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &Self) -> Self::Distance {
        cosine_distance(&self.0, &other.0)
    }
}

impl<T> Proximity<T> for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &T) -> Self::Distance {
        angular_distance(&self.0, other)
    }
}

impl<T> Proximity<Angular<T>> for T
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &Angular<T>) -> Self::Distance {
        angular_distance(self, &other.0)
    }
}

/// Angular distance is a metric.
impl<T> Metric for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
{}

/// Angular distance is a metric.
impl<T> Metric<T> for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
{}

/// Angular distance is a metric.
impl<T> Metric<Angular<T>> for T
where
    T: Coordinates,
    T::Value: Real,
{}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine() {
        assert_eq!(cosine_distance([3.0, 4.0], [3.0, 4.0]), 0.0);
        assert_eq!(cosine_distance([3.0, 4.0], [-4.0, 3.0]), 1.0);
        assert_eq!(cosine_distance([3.0, 4.0], [-3.0, -4.0]), 2.0);
        assert_eq!(cosine_distance([3.0, 4.0], [4.0, -3.0]), 1.0);
    }

    #[test]
    fn test_angular() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

        assert_eq!(angular_distance([3.0, 4.0], [3.0, 4.0]), 0.0);

        assert!((angular_distance([3.0, 4.0], [-4.0, 3.0]) - FRAC_PI_2).abs() < 1.0e-9);
        assert!((angular_distance([3.0, 4.0], [-3.0, -4.0]) - PI).abs() < 1.0e-9);
        assert!((angular_distance([3.0, 4.0], [4.0, -3.0]) - FRAC_PI_2).abs() < 1.0e-9);

        assert!((angular_distance([0.0, 1.0], [1.0, 1.0]) - FRAC_PI_4).abs() < 1.0e-9);
    }
}

