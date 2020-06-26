//! Abstract notions of distance.

use num_traits::{Num, NumAssign, Signed};

/// A number type suitable for distance values.
///
/// This trait is automatically implemented for all types that support the required operations.
pub trait Value: Copy + Num + NumAssign + Signed + PartialOrd {}

/// Blanket [Value] implementation.
impl<T: Num + NumAssign + Signed + Copy + PartialOrd> Value for T {}

/// A distance between two points.
///
/// An implementation may be an actual numerical distance, or an [order embedding] of the true
/// distance.  This allows for optimizations whenever distances can be compared more efficiently
/// than their exact values can be computed, as is the case for [Euclidean distance].  Implementors
/// must satisfy, for all distances `$x$` and `$y$`:
///
/// ```math
/// \begin{aligned}
/// x.\mathrm{value}() &< y.\mathrm{value}() & &\iff& x.\mathrm{value}() &< y \\
/// & & &\iff& x &< y.\mathrm{value}() \\
/// & & &\iff& x &< y
/// \end{aligned}
/// ```
///
/// [order embedding]: https://en.wikipedia.org/wiki/Order_embedding
/// [Euclidean distance]: crate::euclid::EuclideanDistance
pub trait Distance
where
    Self: Copy,
    Self: Into<<Self as Distance>::Value>,
    Self: PartialOrd<<Self as Distance>::Value>,
    <Self as Distance>::Value: PartialOrd<Self>,
    Self: PartialOrd,
{
    /// The type of actual numerical distances.
    type Value: Value;

    /// Get the real numerical value of this distance.
    fn value(self) -> Self::Value {
        self.into()
    }
}

/// Any numerical distance value can be a [Distance].
impl<T: Value> Distance for T {
    type Value = T;
}

/// A space with some notion of distance between points.
///
/// Distances in this space don't need to obey any particular rules like symmetry or the [triangle
/// inequality].  However, spaces that satisfy those rules, at least approximately, often allow for
/// more accurate and efficient searches.
///
/// Type parameters:
///
/// * `T`: The type to compare against.
///
/// [triangle inequality]: https://en.wikipedia.org/wiki/Triangle_inequality
pub trait Proximity<T: ?Sized = Self> {
    /// The type that represents distances.
    type Distance: Distance;

    /// Calculate the distance between this point and another one.
    fn distance(&self, other: &T) -> Self::Distance;
}

// See https://github.com/rust-lang/rust/issues/38078
/// Shorthand for `K::Distance::Value`.
pub type DistanceValue<K, V = K> = <<K as Proximity<V>>::Distance as Distance>::Value;

/// Blanket [Proximity] implementation for references.
impl<'k, 'v, K: Proximity<V>, V> Proximity<&'v V> for &'k K {
    type Distance = K::Distance;

    fn distance(&self, other: &&'v V) -> Self::Distance {
        (*self).distance(*other)
    }
}

/// Marker trait for [metric spaces].
///
/// A metric must be symmetric and obey the [triangle inequality].  More precisely, let `$x$`,
/// `$y$`, and `$z$` be any elements of a metric space, and let
/// `$d(x, y) = x.\mathrm{distance}(y).\mathrm{value}()$`.  Then the following rules must hold:
///
/// ```math
/// \begin{aligned}
/// d(x, x) &= 0 \\
/// d(x, y) &= d(y, x) & \text{(symmetry)} \\
/// d(x, z) &\le d(x, y) + d(y, z) & \text{(triangle inequality)}
/// \end{aligned}
/// ```
///
/// Those conditions also imply the following condition:
///
/// ```math
/// \begin{aligned}
/// d(x, y) &\ge \rlap{0}\phantom{d(x, y) + d(y, z)} & \text{\phantom{(triangle inequality)}\llap{(non-negativity)}}
/// \end{aligned}
/// ```
/// Because we do not prohibit `$d(x, y) = 0$` for distinct `$x$` and `$y$`, these spaces are more
/// properly known as [pseudometric spaces].  This distinction is usually unimportant.
///
/// [metric spaces]: https://en.wikipedia.org/wiki/Metric_space
/// [triangle inequality]: https://en.wikipedia.org/wiki/Triangle_inequality
/// [pseudometric spaces]: https://en.wikipedia.org/wiki/Pseudometric_space
pub trait Metric<T: ?Sized = Self>: Proximity<T> {}

/// Blanket [Metric] implementation for references.
impl<'k, 'v, K: Metric<V>, V> Metric<&'v V> for &'k K {}
