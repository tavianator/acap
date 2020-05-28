//! As Close As Possible — [nearest neighbor search] in Rust.
//!
//! [nearest neighbor search]: https://en.wikipedia.org/wiki/Nearest_neighbor_search

pub mod coords;
pub mod distance;
pub mod euclid;

pub use coords::Coordinates;
pub use distance::{Distance, Metric, Proximity};
pub use euclid::{euclidean_distance, Euclidean, EuclideanDistance};
