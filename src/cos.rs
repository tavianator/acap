//! [Cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity).

use crate::coords::Coordinates;
use crate::distance::{Distance, Metric, Proximity, Value};

use num_traits::real::Real;
use num_traits::{one, zero};

use core::cmp::Ordering;

/// Compute the [cosine *similarity*] between two points.
///
/// Use [cosine_distance] instead if you are implementing [`Proximity::distance()`].
///
/// ```math
/// \begin{aligned}
/// \mathrm{cosine\_similarity}(x, y) &= \frac{x \cdot y}{\|x\| \|y\|} \\
/// &= \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}} \\
/// &= \cos \theta
/// \end{aligned}
/// ```
///
/// [cosine *similarity*]: https://en.wikipedia.org/wiki/Cosine_similarity
/// [`Proximity::distance()`]: Proximity#tymethod.distance
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
/// ```math
/// \begin{aligned}
/// \mathrm{cosine\_distance}(x, y) &= 1 - \mathrm{cosine\_similarity}(x, y) \\
/// &= 1 - \frac{x \cdot y}{\|x\| \|y\|} \\
/// &= 1 - \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}} \\
/// &= 1 - \cos \theta
/// \end{aligned}
/// ```
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
/// [coordinate space]: Coordinates
/// [cosine distance]: cosine_distance
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

/// Compute the [cosine *similarity*] between two pre-normalized (unit magnitude) points.
///
/// Use [`prenorm_cosine_distance()`] instead if you are implementing [`Proximity::distance()`].
///
/// ```math
/// \begin{aligned}
/// \mathrm{prenorm\_cosine\_similarity}(x, y) &= x \cdot y \\
/// &= \sum_i x_i y_i \\
/// &= \cos \theta
/// \end{aligned}
/// ```
///
/// [cosine *similarity*]: https://en.wikipedia.org/wiki/Cosine_similarity
/// [`Proximity::distance()`]: Proximity#tymethod.distance
pub fn prenorm_cosine_similarity<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    debug_assert!(x.dims() == y.dims());

    let mut dot: T::Value = zero();

    for i in 0..x.dims() {
        dot += x.coord(i) * y.coord(i);
    }

    dot
}

/// Compute the [cosine distance] between two pre-normalized (unit magnitude) points.
///
/// ```math
/// \begin{aligned}
/// \mathrm{prenorm\_cosine\_distance}(x, y) &= 1 - \mathrm{prenorm\_cosine\_similarity}(x, y) \\
/// &= 1 - x \cdot y \\
/// &= 1 - \sum_i x_i y_i \\
/// &= 1 - \cos \theta
/// \end{aligned}
/// ```
///
/// [cosine distance]: https://en.wikipedia.org/wiki/Cosine_similarity
pub fn prenorm_cosine_distance<T, U>(x: T, y: U) -> T::Value
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    let one: T::Value = one();
    one - prenorm_cosine_similarity(x, y)
}

/// Equips any [coordinate space] with the [cosine distance] function for pre-normalized (unit
/// magnitude) points.
///
/// [coordinate space]: Coordinates
/// [cosine distance]: prenorm_cosine_distance
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrenormCosine<T>(pub T);

impl<T> Proximity for PrenormCosine<T>
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &Self) -> Self::Distance {
        prenorm_cosine_distance(&self.0, &other.0)
    }
}

impl<T> Proximity<T> for PrenormCosine<T>
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &T) -> Self::Distance {
        prenorm_cosine_distance(&self.0, other)
    }
}

impl<T> Proximity<PrenormCosine<T>> for T
where
    T: Coordinates,
    T::Value: Real,
{
    type Distance = T::Value;

    fn distance(&self, other: &PrenormCosine<T>) -> Self::Distance {
        prenorm_cosine_distance(self, &other.0)
    }
}

/// Compute the [angular distance] between two points.
///
/// ```math
/// \begin{aligned}
/// \mathrm{angular\_distance}(x, y) &= \arccos(\mathrm{cosine\_similarity}(x, y)) \\
/// &= \arccos \left( \frac{x \cdot y}{\|x\| \|y\|} \right) \\
/// &= \arccos \left( \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}} \right) \\
/// &= \theta
/// \end{aligned}
/// ```
///
/// [angular distance]: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
pub fn angular_distance<T, U>(x: T, y: U) -> AngularDistance<T::Value>
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    AngularDistance::from_cos(cosine_similarity(x, y))
}

/// Equips any [coordinate space] with the [angular distance] metric.
///
/// [coordinate space]: Coordinates
/// [angular distance]: angular_distance
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Angular<T>(pub T);

impl<T> Proximity for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{
    type Distance = AngularDistance<T::Value>;

    fn distance(&self, other: &Self) -> Self::Distance {
        angular_distance(&self.0, &other.0)
    }
}

impl<T> Proximity<T> for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{
    type Distance = AngularDistance<T::Value>;

    fn distance(&self, other: &T) -> Self::Distance {
        angular_distance(&self.0, other)
    }
}

impl<T> Proximity<Angular<T>> for T
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{
    type Distance = AngularDistance<T::Value>;

    fn distance(&self, other: &Angular<T>) -> Self::Distance {
        angular_distance(self, &other.0)
    }
}

/// Angular distance is a metric.
impl<T> Metric for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{}

/// Angular distance is a metric.
impl<T> Metric<T> for Angular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{}

/// Angular distance is a metric.
impl<T> Metric<Angular<T>> for T
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{}

/// Compute the [angular distance] between two points.
///
/// ```math
/// \begin{aligned}
/// \mathrm{prenorm\_angular\_distance}(x, y) &= \arccos(\mathrm{prenorm\_cosine\_similarity}(x, y)) \\
/// &= \arccos(x \cdot y) \\
/// &= \arccos \left( \sum_i x_i y_i \right) \\
/// &= \theta
/// \end{aligned}
/// ```
///
/// [angular distance]: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
pub fn prenorm_angular_distance<T, U>(x: T, y: U) -> AngularDistance<T::Value>
where
    T: Coordinates,
    U: Coordinates<Value = T::Value>,
    T::Value: Real,
{
    AngularDistance::from_cos(prenorm_cosine_similarity(x, y))
}

/// Equips any [coordinate space] with the [angular distance] metric for pre-normalized (unit
/// magnitude) points.
///
/// [coordinate space]: Coordinates
/// [angular distance]: prenorm_angular_distance
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrenormAngular<T>(pub T);

impl<T> Proximity for PrenormAngular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{
    type Distance = AngularDistance<T::Value>;

    fn distance(&self, other: &Self) -> Self::Distance {
        prenorm_angular_distance(&self.0, &other.0)
    }
}

impl<T> Proximity<T> for PrenormAngular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{
    type Distance = AngularDistance<T::Value>;

    fn distance(&self, other: &T) -> Self::Distance {
        prenorm_angular_distance(&self.0, other)
    }
}

impl<T> Proximity<PrenormAngular<T>> for T
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{
    type Distance = AngularDistance<T::Value>;

    fn distance(&self, other: &PrenormAngular<T>) -> Self::Distance {
        prenorm_angular_distance(self, &other.0)
    }
}

/// Angular distance is a metric.
impl<T> Metric for PrenormAngular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{}

/// Angular distance is a metric.
impl<T> Metric<T> for PrenormAngular<T>
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{}

/// Angular distance is a metric.
impl<T> Metric<PrenormAngular<T>> for T
where
    T: Coordinates,
    T::Value: Real,
    AngularDistance<T::Value>: Distance,
{}

/// An [angular distance].
///
/// This type stores the cosine of the angle, to avoid computing the expensive trancendental
/// `acos()` function until absolutely necessary.
///
///     # use acap::distance::Distance;
///     # use acap::cos::AngularDistance;
///     let zero = AngularDistance::from_cos(1.0);
///     let pi_2 = AngularDistance::from_cos(0.0);
///     let pi = AngularDistance::from_cos(-1.0);
///     assert!(zero < pi_2 && pi_2 < pi);
///
/// [angular distance]: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AngularDistance<T>(T);

impl<T: Real + Value> AngularDistance<T> {
    /// Creates an `AngularDistance` from the cosine of an angle.
    pub fn from_cos(value: T) -> Self {
        Self(value)
    }

    /// Get the cosine of this angle.
    pub fn cos(self) -> T {
        self.0
    }
}

impl<T: PartialOrd> PartialOrd for AngularDistance<T> {
    fn partial_cmp(&self, other: &AngularDistance<T>) -> Option<Ordering> {
        // acos() is decreasing, so swap the comparison order
        other.0.partial_cmp(&self.0)
    }
}

/// Error type for failed conversions from angles outside of `$[0, \pi]$` to [`AngularDistance`].
#[derive(Debug)]
pub struct InvalidAngleError;

macro_rules! impl_distance {
    ($f:ident) => {
        impl TryFrom<$f> for AngularDistance<$f> {
            type Error = InvalidAngleError;

            #[inline]
            fn try_from(value: $f) -> Result<Self, Self::Error> {
                if value >= 0.0 && value <= core::$f::consts::PI {
                    Ok(Self(value.cos()))
                } else {
                    Err(InvalidAngleError)
                }
            }
        }

        impl From<AngularDistance<$f>> for $f {
            #[inline]
            fn from(value: AngularDistance<$f>) -> $f {
                value.0.acos()
            }
        }

        impl PartialOrd<$f> for AngularDistance<$f> {
            #[inline]
            fn partial_cmp(&self, other: &$f) -> Option<Ordering> {
                self.value().partial_cmp(other)
            }
        }

        impl PartialOrd<AngularDistance<$f>> for $f {
            #[inline]
            fn partial_cmp(&self, other: &AngularDistance<$f>) -> Option<Ordering> {
                self.partial_cmp(&other.value())
            }
        }

        impl PartialEq<$f> for AngularDistance<$f> {
            #[inline]
            fn eq(&self, other: &$f) -> bool {
                self.value() == *other
            }
        }

        impl PartialEq<AngularDistance<$f>> for $f {
            #[inline]
            fn eq(&self, other: &AngularDistance<$f>) -> bool {
                *self == other.value()
            }
        }

        impl Distance for AngularDistance<$f> {
            type Value = $f;
        }
    };
}

impl_distance!(f32);
impl_distance!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, SQRT_2};

    #[test]
    fn test_cosine() {
        assert_eq!(cosine_distance([3.0, 4.0], [3.0, 4.0]), 0.0);
        assert_eq!(cosine_distance([3.0, 4.0], [-4.0, 3.0]), 1.0);
        assert_eq!(cosine_distance([3.0, 4.0], [-3.0, -4.0]), 2.0);
        assert_eq!(cosine_distance([3.0, 4.0], [4.0, -3.0]), 1.0);
    }

    #[test]
    fn test_prenorm_cosine() {
        assert_eq!(prenorm_cosine_distance([0.6, 0.8], [0.6, 0.8]), 0.0);
        assert_eq!(prenorm_cosine_distance([0.6, 0.8], [-0.8, 0.6]), 1.0);
        assert_eq!(prenorm_cosine_distance([0.6, 0.8], [-0.6, -0.8]), 2.0);
        assert_eq!(prenorm_cosine_distance([0.6, 0.8], [0.8, -0.6]), 1.0);
    }

    #[test]
    fn test_angular() {
        let zero = angular_distance([3.0, 4.0], [3.0, 4.0]);
        let pi_4 = Angular([0.0, 1.0]).distance(&Angular([1.0, 1.0]));
        let pi_2 = Angular([3.0, 4.0]).distance(&[-4.0, 3.0]);
        let pi = [3.0, 4.0].distance(&Angular([-3.0, -4.0]));

        assert_eq!(zero.cos(), 1.0);
        assert_eq!(pi_2.cos(), 0.0);
        assert_eq!(pi.cos(), -1.0);

        assert_eq!(zero, 0.0);

        assert!(zero < pi_4);
        assert!(zero < pi_2);
        assert!(zero < pi);

        assert!(pi_4 < pi_2);
        assert!(pi_4 < pi);

        assert!(pi_2 < pi);

        assert!(FRAC_PI_4 < pi_2);
        assert!(pi_2 > FRAC_PI_4);

        assert!(pi_2 < PI);
        assert!(PI > pi_2);

        assert!((pi_4.value() - FRAC_PI_4).abs() < 1.0e-9);
        assert!((pi_2.value() - FRAC_PI_2).abs() < 1.0e-9);
        assert!((pi.value() - PI).abs() < 1.0e-9);
    }

    #[test]
    fn test_prenorm_angular() {
        let sqrt_2_inv = 1.0 / SQRT_2;

        let zero = prenorm_angular_distance([0.6, 0.8], [0.6, 0.8]);
        let pi_4 = PrenormAngular([0.0, 1.0]).distance(&PrenormAngular([sqrt_2_inv, sqrt_2_inv]));
        let pi_2 = PrenormAngular([0.6, 0.8]).distance(&[-0.8, 0.6]);
        let pi = [0.6, 0.8].distance(&PrenormAngular([-0.6, -0.8]));

        assert_eq!(zero.cos(), 1.0);
        assert_eq!(pi_2.cos(), 0.0);
        assert_eq!(pi.cos(), -1.0);

        assert_eq!(zero, 0.0);

        assert!(zero < pi_4);
        assert!(zero < pi_2);
        assert!(zero < pi);

        assert!(pi_4 < pi_2);
        assert!(pi_4 < pi);

        assert!(pi_2 < pi);

        assert!(FRAC_PI_4 < pi_2);
        assert!(pi_2 > FRAC_PI_4);

        assert!(pi_2 < PI);
        assert!(PI > pi_2);

        assert!((pi_4.value() - FRAC_PI_4).abs() < 1.0e-9);
        assert!((pi_2.value() - FRAC_PI_2).abs() < 1.0e-9);
        assert!((pi.value() - PI).abs() < 1.0e-9);
    }
}
