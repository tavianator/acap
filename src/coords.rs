//! [Coordinate spaces](https://en.wikipedia.org/wiki/Cartesian_coordinate_system).

use crate::distance::Value;

/// A coordinate space.
pub trait Coordinates {
    /// The type of individual coordinates.
    type Value: Value;

    /// Get the number of dims this point has.
    fn dims(&self) -> usize;

    /// Get the `i`th coordinate of this point.
    fn coord(&self, i: usize) -> Self::Value;

    /// Create a vector with this point's coordinates as values.
    fn as_vec(&self) -> Vec<Self::Value> {
        let len = self.dims();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(self.coord(i));
        }
        vec
    }
}

/// [`Coordinates`] implementation for slices.
impl<T: Value> Coordinates for [T] {
    type Value = T;

    fn dims(&self) -> usize {
        self.len()
    }

    fn coord(&self, i: usize) -> T {
        self[i]
    }
}

/// [`Coordinates`] implementation for arrays.
macro_rules! array_coordinates {
    ($n:expr) => {
        impl<T: Value> Coordinates for [T; $n] {
            type Value = T;

            fn dims(&self) -> usize {
                $n
            }

            fn coord(&self, i: usize) -> T {
                self[i]
            }
        }
    };
}

array_coordinates!(1);
array_coordinates!(2);
array_coordinates!(3);
array_coordinates!(4);
array_coordinates!(5);
array_coordinates!(6);
array_coordinates!(7);
array_coordinates!(8);

/// [`Coordinates`] implemention for vectors.
impl<T: Value> Coordinates for Vec<T> {
    type Value = T;

    fn dims(&self) -> usize {
        self.len()
    }

    fn coord(&self, i: usize) -> T {
        self[i]
    }
}

/// Blanket [`Coordinates`] implementation for references.
impl<T: ?Sized + Coordinates> Coordinates for &T {
    type Value = T::Value;

    fn dims(&self) -> usize {
        (*self).dims()
    }

    fn coord(&self, i: usize) -> Self::Value {
        (*self).coord(i)
    }
}
