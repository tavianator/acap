//! As Close As Possible — [nearest neighbor search] in Rust.
//!
//! # Overview
//!
//! The notion of distances between points is captured by the [`Proximity`] trait.  Its
//! [`distance()`] method returns a [`Distance`], from which the actual numerical distance may be
//! retrieved with [`value()`].  These layers of abstraction allow `acap` to work generically with
//! different distance functions over different types.
//!
//! There are no restrictions on the distances computed by a [`Proximity`].  For example, they don't
//! have to be symmetric, subadditive, or even positive.  Implementations that do have these
//! desirable properties will additionally implement the [`Metric`] marker trait.  This distinction
//! allows `acap` to support a wide variety of useful metric and non-metric distances.
//!
//! As a concrete example, consider `Euclidean<[i32; 2]>`.  The [`Euclidean`] wrapper equips any
//! type that has [coordinates] with the [Euclidean distance] function as its [`Proximity`]
//! implementation:
//!
//!     use acap::distance::Proximity;
//!     use acap::euclid::Euclidean;
//!
//!     let a = Euclidean([3, 4]);
//!     let b = Euclidean([7, 7]);
//!     assert_eq!(a.distance(&b), 5);
//!
//! In this case, `distance()` doesn't return a number directly; as an optimization, it returns a
//! [`EuclideanDistance`] wrapper.  This wrapper stores the squared value of the distance, to avoid
//! computing square roots until absolutely necessary.  Still, it transparently supports comparisons
//! with numerical values:
//!
//!     # use acap::distance::Proximity;
//!     # use acap::euclid::Euclidean;
//!     # let a = Euclidean([3, 4]);
//!     # let b = Euclidean([7, 7]);
//!     use acap::distance::Distance;
//!
//!     let d = a.distance(&b);
//!     assert!(d > 4 && d < 6);
//!     assert_eq!(d, 5);
//!     assert_eq!(d.value(), 5.0f32);
//!
//! For finding the nearest neighbors to a point from a set of other points, the
//! [`NearestNeighbors`] trait provides a uniform interface to [many different similarity search
//! data structures].  One such structure is the [vantage-point tree], available in `acap` as
//! [`VpTree`]:
//!
//!     # use acap::euclid::Euclidean;
//!     use acap::vp::VpTree;
//!
//!     let tree = VpTree::balanced(vec![
//!         Euclidean([3, 4]),
//!         Euclidean([5, 12]),
//!         Euclidean([8, 15]),
//!         Euclidean([7, 24]),
//!     ]);
//!
//! [`VpTree`] implements [`NearestNeighbors`], which has a [`nearest()`] method that returns an
//! optional [`Neighbor`].  The [`Neighbor`] struct holds the actual neighbor it found, and the
//! distance it was from the target:
//!
//!     # use acap::euclid::Euclidean;
//!     # use acap::vp::VpTree;
//!     use acap::knn::NearestNeighbors;
//!
//!     # let tree = VpTree::balanced(
//!     #     vec![Euclidean([3, 4]), Euclidean([5, 12]), Euclidean([8, 15]), Euclidean([7, 24])]
//!     # );
//!     let nearest = tree.nearest(&[7, 7]).unwrap();
//!     assert_eq!(nearest.item, &Euclidean([3, 4]));
//!     assert_eq!(nearest.distance, 5);
//!
//! [`NearestNeighbors`] also provides the [`nearest_within()`], [`k_nearest()`], and
//! [`k_nearest_within()`] methods which find up to `k` neighbors within a possible threshold.
//!
//! It can be expensive to compute nearest neighbors exactly, especially in high dimensions.
//! For performance reasons, [`NearestNeighbors`] implementations are allowed to return approximate
//! results.  Many implementations have a speed/accuracy tradeoff which can be tuned.  Those
//! implementations which always return exact results will also implement the [`ExactNeighbors`]
//! marker trait.  For example, a [`VpTree`] will be exact when the [`Proximity`] function is a
//! [`Metric`].
//!
//! # Examples
//!
//! ## Searching without owning
//!
//! Since [`Proximity`] has a blanket implementation for references, you can store references in a
//! nearest neighbor index instead of having it hold the data itself:
//!
//!     use acap::euclid::Euclidean;
//!     use acap::knn::NearestNeighbors;
//!     use acap::vp::VpTree;
//!
//!     let points = vec![
//!         Euclidean([3, 4]),
//!         Euclidean([5, 12]),
//!         Euclidean([8, 15]),
//!         Euclidean([7, 24]),
//!     ];
//!
//!     let tree = VpTree::balanced(points.iter());
//!
//!     let nearest = tree.nearest(&&[7, 7]).unwrap();
//!     assert!(std::ptr::eq(*nearest.item, &points[0]));
//!
//! ## Custom distance functions
//!
//! See the [`Proximity`] documentation.
//!
//! [nearest neighbor search]: https://en.wikipedia.org/wiki/Nearest_neighbor_search
//! [`distance()`]: distance::Proximity#tymethod.distance
//! [`value()`]: distance::Distance#method.value
//! [coordinates]: coords::Coordinates
//! [Euclidean distance]: https://en.wikipedia.org/wiki/Euclidean_distance
//! [`NearestNeighbors`]: knn::NearestNeighbors
//! [many different similarity search data structures]: knn::NearestNeighbors#implementors
//! [vantage-point tree]: https://en.wikipedia.org/wiki/Vantage-point_tree
//! [`VpTree`]: vp::VpTree
//! [`Neighbor`]: knn::Neighbor
//! [`nearest()`]: knn::NearestNeighbors#method.nearest
//! [`k_nearest()`]: knn::NearestNeighbors#method.k_nearest
//! [`nearest_within()`]: knn::NearestNeighbors#method.nearest_within
//! [`k_nearest_within()`]: knn::NearestNeighbors#method.k_nearest_within
//! [`ExactNeighbors`]: knn::ExactNeighbors

#![warn(rust_2018_idioms)]

#![no_std]

extern crate alloc;

pub mod chebyshev;
pub mod coords;
pub mod cos;
pub mod distance;
pub mod euclid;
pub mod exhaustive;
pub mod hamming;
pub mod kd;
pub mod knn;
pub mod lp;
pub mod taxi;
pub mod vp;

mod util;

pub use coords::Coordinates;
pub use distance::{Distance, Metric, Proximity};
pub use euclid::{euclidean_distance, Euclidean, EuclideanDistance};
pub use knn::{ExactNeighbors, NearestNeighbors, Neighbor};
