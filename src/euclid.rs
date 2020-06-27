//! [Euclidean space](https://en.wikipedia.org/wiki/Euclidean_space).

use crate::coords::{Coordinates, CoordinateMetric, CoordinateProximity};
use crate::distance::{Distance, Metric, Proximity, Value};

use num_traits::zero;

use std::cmp::Ordering;
use std::convert::TryFrom;

/// A point in Euclidean space.
///
/// This wrapper equips any [coordinate space] with the [Euclidean distance] metric.
///
/// [coordinate space]: Coordinates
/// [Euclidean distance]: euclidean_distance
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Euclidean<T>(pub T);

impl<T> Euclidean<T> {
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

impl<T: Coordinates> Coordinates for Euclidean<T> {
    type Value = T::Value;

    fn dims(&self) -> usize {
        self.0.dims()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.0.coord(i)
    }
}

/// Compute the [Euclidean distance] between two points.
///
/// ```math
/// \begin{aligned}
/// \mathrm{euclidean\_distance}(x, y) &= \|x - y\|_2 \\
/// &= \sqrt{\sum_i (x_i - y_i)^2}
/// \end{aligned}
/// ```
///
/// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
pub fn euclidean_distance<T, U>(x: T, y: U) -> EuclideanDistance<T::Value>
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
{
    debug_assert!(x.dims() == y.dims());

    let mut sum = zero();
    for i in 0..x.dims() {
        let diff = x.coord(i) - y.coord(i);
        sum += diff * diff;
    }

    EuclideanDistance::from_squared(sum)
}

/// The Euclidean distance function.
impl<T> Proximity for Euclidean<T>
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{
    type Distance = EuclideanDistance<T::Value>;

    fn distance(&self, other: &Self) -> Self::Distance {
        euclidean_distance(self, other)
    }
}

impl<T> Proximity<T> for Euclidean<T>
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{
    type Distance = EuclideanDistance<T::Value>;

    fn distance(&self, other: &T) -> Self::Distance {
        euclidean_distance(self, other)
    }
}

impl<T> Proximity<Euclidean<T>> for T
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{
    type Distance = EuclideanDistance<T::Value>;

    fn distance(&self, other: &Euclidean<T>) -> Self::Distance {
        euclidean_distance(self, other)
    }
}

/// Euclidean distance is a metric.
impl<T> Metric for Euclidean<T>
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{}

impl<T> Metric<T> for Euclidean<T>
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{}

impl<T> Metric<Euclidean<T>> for T
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{}

impl<T> CoordinateProximity<T::Value> for Euclidean<T>
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{
    type Distance = EuclideanDistance<T::Value>;

    fn distance_to_coords(&self, coords: &[T::Value]) -> Self::Distance {
        euclidean_distance(self, coords)
    }
}

impl<T> CoordinateMetric<T::Value> for Euclidean<T>
where
    T: Coordinates,
    EuclideanDistance<T::Value>: Distance,
{}

/// A [Euclidean distance].
///
/// This type stores the squared value of the Euclidean distance, to avoid computing expensive
/// square roots until absolutely necessary.
///
///     # use acap::distance::Distance;
///     # use acap::euclid::EuclideanDistance;
///     # use std::convert::TryFrom;
///     let a = EuclideanDistance::try_from(3).unwrap();
///     let b = EuclideanDistance::try_from(4).unwrap();
///     let c = EuclideanDistance::from_squared(a.squared_value() + b.squared_value());
///     assert!(a < c && b < c);
///     assert_eq!(c.value(), 5.0f32);
///
/// [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct EuclideanDistance<T>(T);

impl<T: Value> EuclideanDistance<T> {
    /// Creates a `EuclideanDistance` from an already-squared value.
    pub fn from_squared(value: T) -> Self {
        debug_assert!(!value.is_negative());
        Self(value)
    }

    /// Get the squared distance value.
    pub fn squared_value(self) -> T {
        self.0
    }
}

/// Error type for failed conversions from negative numbers to [`EuclideanDistance`].
#[derive(Debug)]
pub struct NegativeDistanceError;

/// Implement EuclideanDistance for a floating-point type.
macro_rules! float_distance {
    ($f:ty) => {
        impl TryFrom<$f> for EuclideanDistance<$f> {
            type Error = NegativeDistanceError;

            #[inline]
            fn try_from(value: $f) -> Result<Self, Self::Error> {
                if value >= 0.0 {
                    Ok(Self(value * value))
                } else {
                    Err(NegativeDistanceError)
                }
            }
        }

        impl From<EuclideanDistance<$f>> for $f {
            #[inline]
            fn from(value: EuclideanDistance<$f>) -> $f {
                value.0.sqrt()
            }
        }

        impl PartialOrd<$f> for EuclideanDistance<$f> {
            #[inline]
            fn partial_cmp(&self, other: &$f) -> Option<Ordering> {
                if let Ok(rhs) = Self::try_from(*other) {
                    self.partial_cmp(&rhs)
                } else {
                    Some(Ordering::Greater)
                }
            }
        }

        impl PartialOrd<EuclideanDistance<$f>> for $f {
            #[inline]
            fn partial_cmp(&self, other: &EuclideanDistance<$f>) -> Option<Ordering> {
                if let Ok(lhs) = EuclideanDistance::try_from(*self) {
                    lhs.partial_cmp(other)
                } else {
                    Some(Ordering::Less)
                }
            }
        }

        impl PartialEq<$f> for EuclideanDistance<$f> {
            #[inline]
            fn eq(&self, other: &$f) -> bool {
                self.partial_cmp(other) == Some(Ordering::Equal)
            }
        }

        impl PartialEq<EuclideanDistance<$f>> for $f {
            #[inline]
            fn eq(&self, other: &EuclideanDistance<$f>) -> bool {
                self.partial_cmp(other) == Some(Ordering::Equal)
            }
        }

        impl Distance for EuclideanDistance<$f> {
            type Value = $f;
        }
    }
}

float_distance!(f32);
float_distance!(f64);

/// Implement EuclideanDistance for an integer type.
macro_rules! int_distance {
    ($i:ty, $f:ty, $ff:ty) => {
        impl TryFrom<$i> for EuclideanDistance<$i> {
            type Error = NegativeDistanceError;

            #[inline]
            fn try_from(value: $i) -> Result<Self, Self::Error> {
                if value >= 0 {
                    Ok(Self(value * value))
                } else {
                    Err(NegativeDistanceError)
                }
            }
        }

        impl From<EuclideanDistance<$i>> for $f {
            #[inline]
            fn from(value: EuclideanDistance<$i>) -> Self {
                (value.0 as $ff).sqrt() as $f
            }
        }

        impl PartialOrd<$i> for EuclideanDistance<$i> {
            #[inline]
            fn partial_cmp(&self, other: &$i) -> Option<Ordering> {
                if let Ok(rhs) = Self::try_from(*other) {
                    self.partial_cmp(&rhs)
                } else {
                    Some(Ordering::Greater)
                }
            }
        }

        impl PartialOrd<EuclideanDistance<$i>> for $i {
            #[inline]
            fn partial_cmp(&self, other: &EuclideanDistance<$i>) -> Option<Ordering> {
                if let Ok(lhs) = EuclideanDistance::try_from(*self) {
                    lhs.partial_cmp(other)
                } else {
                    Some(Ordering::Less)
                }
            }
        }

        impl PartialEq<$i> for EuclideanDistance<$i> {
            #[inline]
            fn eq(&self, other: &$i) -> bool {
                self.partial_cmp(other) == Some(Ordering::Equal)
            }
        }

        impl PartialEq<EuclideanDistance<$i>> for $i {
            #[inline]
            fn eq(&self, other: &EuclideanDistance<$i>) -> bool {
                self.partial_cmp(other) == Some(Ordering::Equal)
            }
        }

        impl PartialOrd<$f> for EuclideanDistance<$i> {
            #[inline]
            fn partial_cmp(&self, other: &$f) -> Option<Ordering> {
                if *other >= 0.0 {
                    let lhs = self.0 as $ff;
                    let mut rhs = *other as $ff;
                    rhs *= rhs;
                    lhs.partial_cmp(&rhs)
                } else {
                    Some(Ordering::Greater)
                }
            }
        }

        impl PartialOrd<EuclideanDistance<$i>> for $f {
            #[inline]
            fn partial_cmp(&self, other: &EuclideanDistance<$i>) -> Option<Ordering> {
                if *other >= 0.0 {
                    let mut lhs = *self as $ff;
                    lhs *= lhs;
                    let rhs = other.0 as $ff;
                    lhs.partial_cmp(&rhs)
                } else {
                    Some(Ordering::Greater)
                }
            }
        }

        impl PartialEq<$f> for EuclideanDistance<$i> {
            #[inline]
            fn eq(&self, other: &$f) -> bool {
                self.partial_cmp(other) == Some(Ordering::Equal)
            }
        }

        impl PartialEq<EuclideanDistance<$i>> for $f {
            #[inline]
            fn eq(&self, other: &EuclideanDistance<$i>) -> bool {
                self.partial_cmp(other) == Some(Ordering::Equal)
            }
        }

        impl Distance for EuclideanDistance<$i> {
            type Value = $f;
        }
    }
}

int_distance!(i16, f32, f32);
int_distance!(i32, f32, f64);
int_distance!(i64, f64, f64);
int_distance!(isize, f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i32() {
        let five = euclidean_distance([0, 0], [3, 4]);
        assert_eq!(five, EuclideanDistance::from_squared(25));
        assert_eq!(five, 5.0f32);

        let thirteen = Euclidean([0, 0]).distance(&Euclidean([5, 12]));
        assert_eq!(thirteen, EuclideanDistance::from_squared(169));
        assert_eq!(thirteen, 13.0f32);

        assert!(five < thirteen);
        assert!(five < 13);
        assert!(5 < thirteen);
        assert!(-5 < thirteen);
    }

    #[test]
    fn test_f64() {
        let five = euclidean_distance([0.0, 0.0], [3.0, 4.0]);
        assert_eq!(five, EuclideanDistance::from_squared(25.0));
        assert_eq!(five, 5.0);

        let thirteen = Euclidean([0.0, 0.0]).distance(&Euclidean([5.0, 12.0]));
        assert_eq!(thirteen, EuclideanDistance::from_squared(169.0));
        assert_eq!(thirteen, 13.0);

        assert!(five < thirteen);
        assert!(five < 13.0);
        assert!(5.0 < thirteen);
        assert!(-5.0 < thirteen);
    }
}
