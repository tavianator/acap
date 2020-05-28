//! As Close As Possible — [nearest neighbor search] in Rust.
//!
//! [nearest neighbor search]: https://en.wikipedia.org/wiki/Nearest_neighbor_search

pub mod coords;
pub mod distance;
pub mod euclid;

pub use coords::Coordinates;
pub use distance::{Distance, Metric, Proximity};
pub use euclid::{euclidean_distance, Euclidean, EuclideanDistance};

use std::cmp::Ordering;
use std::convert::TryInto;
use std::collections::BinaryHeap;

/// A nearest neighbor.
#[derive(Clone, Copy, Debug)]
pub struct Neighbor<V, D> {
    /// The neighbor itself.
    pub item: V,
    /// The distance from the target to this neighbor.
    pub distance: D,
}

impl<V, D> Neighbor<V, D> {
    /// Create a new Neighbor.
    pub fn new(item: V, distance: D) -> Self {
        Self { item, distance }
    }
}

impl<V1, D1, V2, D2> PartialEq<Neighbor<V2, D2>> for Neighbor<V1, D1>
where
    V1: PartialEq<V2>,
    D1: PartialEq<D2>,
{
    fn eq(&self, other: &Neighbor<V2, D2>) -> bool {
        self.item == other.item && self.distance == other.distance
    }
}

/// Accumulates nearest neighbor search results.
///
/// Type parameters:
///
/// * `K`: The type of the search target (the "key" type)
/// * `V`: The type of neighbors this contains (the "value" type)
///
/// Neighborhood implementations keep track of the current search radius and accumulate the results,
/// work which would otherwise have to be duplicated for every nearest neighbor search algorithm.
/// They also serve as a customization point, allowing for functionality to be injected into any
/// [NearestNeighbors] implementation (for example, filtering the result set or limiting the number
/// of neighbors considered).
pub trait Neighborhood<K: Proximity<V>, V> {
    /// Returns the target of the nearest neighbor search.
    fn target(&self) -> K;

    /// Check whether a distance is within the current search radius.
    fn contains<D>(&self, distance: D) -> bool
    where
        D: PartialOrd<K::Distance>;

    /// Consider a new candidate neighbor.
    ///
    /// Returns `self.target().distance(item)`.
    fn consider(&mut self, item: V) -> K::Distance;
}

/// A candidate nearest neighbor found during a search.
#[derive(Debug)]
struct Candidate<V, D>(Neighbor<V, D>);

impl<V, D: Distance> Candidate<V, D> {
    fn new<K>(target: K, item: V) -> Self
    where
        K: Proximity<V, Distance = D>,
    {
        let distance = target.distance(&item);
        Self(Neighbor::new(item, distance))
    }
}

impl<V, D: PartialOrd> PartialOrd for Candidate<V, D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.distance.partial_cmp(&other.0.distance)
    }
}

impl<V, D: PartialOrd> Ord for Candidate<V, D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("Unordered distances found during nearest neighbor search")
    }
}

impl<V, D: PartialEq> PartialEq for Candidate<V, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}

impl<V, D: PartialEq> Eq for Candidate<V, D> {}

/// A [Neighborhood] with at most one result.
#[derive(Debug)]
struct SingletonNeighborhood<K, V, D> {
    /// The search target.
    target: K,
    /// The current threshold distance.
    threshold: Option<D>,
    /// The current nearest neighbor, if any.
    candidate: Option<Candidate<V, D>>,
}

impl<K, V, D> SingletonNeighborhood<K, V, D> {
    /// Create a new singleton neighborhood.
    ///
    /// * `target`: The search target.
    /// * `threshold`: The maximum allowable distance.
    fn new(target: K, threshold: Option<D>) -> Self {
        Self {
            target,
            threshold,
            candidate: None,
        }
    }

    /// Convert this result into an optional neighbor.
    fn into_option(self) -> Option<Neighbor<V, D>> {
        self.candidate.map(|c| c.0)
    }
}

impl<K, V> Neighborhood<K, V> for SingletonNeighborhood<K, V, K::Distance>
where
    K: Copy + Proximity<V>,
{
    fn target(&self) -> K {
        self.target
    }

    fn contains<D>(&self, distance: D) -> bool
    where
        D: PartialOrd<K::Distance>,
    {
        self.threshold.map_or(true, |t| distance <= t)
    }

    fn consider(&mut self, item: V) -> K::Distance {
        let candidate = Candidate::new(self.target, item);
        let distance = candidate.0.distance;

        if self.contains(distance) {
            self.threshold = Some(distance);
            self.candidate = Some(candidate);
        }

        distance
    }
}

/// A [Neighborhood] of up to `k` results, using a binary heap.
#[derive(Debug)]
struct HeapNeighborhood<K, V, D> {
    /// The target of the nearest neighbor search.
    target: K,
    /// The number of nearest neighbors to find.
    k: usize,
    /// The current threshold distance to the farthest result.
    threshold: Option<D>,
    /// A max-heap of the best candidates found so far.
    heap: BinaryHeap<Candidate<V, D>>,
}

impl<K, V, D: PartialOrd> HeapNeighborhood<K, V, D> {
    /// Create a new singleton neighborhood.
    ///
    /// * `target`: The search target.
    /// * `k`: The number of nearest neighbors to find.
    /// * `threshold`: The maximum allowable distance.
    fn new(target: K, k: usize, threshold: Option<D>) -> Self {
        Self {
            target,
            k,
            threshold,
            heap: BinaryHeap::new(),
        }
    }

    /// Convert this result into an optional neighbor.
    fn into_vec(self) -> Vec<Neighbor<V, D>> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| c.0)
            .collect()
    }
}

impl<K, V> Neighborhood<K, V> for HeapNeighborhood<K, V, K::Distance>
where
    K: Copy + Proximity<V>,
{
    fn target(&self) -> K {
        self.target
    }

    fn contains<D>(&self, distance: D) -> bool
    where
        D: PartialOrd<K::Distance>,
    {
        self.k > 0 && self.threshold.map_or(true, |t| distance <= t)
    }

    fn consider(&mut self, item: V) -> K::Distance {
        let candidate = Candidate::new(self.target, item);
        let distance = candidate.0.distance;

        if self.contains(distance) {
            let heap = &mut self.heap;

            if heap.len() == self.k {
                heap.pop();
            }

            heap.push(candidate);

            if heap.len() == self.k {
                self.threshold = heap.peek().map(|c| c.0.distance)
            }
        }

        distance
    }
}

/// A [nearest neighbor search] index.
///
/// Type parameters:
///
/// * `K`: The type of the search target (the "key" type)
/// * `V`: The type of the returned neighbors (the "value" type)
///
/// In general, exact nearest neighbor searches may be prohibitively expensive due to the [curse of
/// dimensionality].  Therefore, NearestNeighbor implementations are allowed to give approximate
/// results.  The marker trait [ExactNeighbors] denotes implementations which are guaranteed to give
/// exact results.
///
/// [nearest neighbor search]: https://en.wikipedia.org/wiki/Nearest_neighbor_search
/// [curse of dimensionality]: https://en.wikipedia.org/wiki/Curse_of_dimensionality
pub trait NearestNeighbors<K: Proximity<V>, V = K> {
    /// Returns the nearest neighbor to `target` (or `None` if this index is empty).
    fn nearest(&self, target: &K) -> Option<Neighbor<&V, K::Distance>> {
        self.search(SingletonNeighborhood::new(target, None))
            .into_option()
    }

    /// Returns the nearest neighbor to `target` within the distance `threshold`, if one exists.
    fn nearest_within<D>(&self, target: &K, threshold: D) -> Option<Neighbor<&V, K::Distance>>
    where
        D: TryInto<K::Distance>,
    {
        if let Ok(distance) = threshold.try_into() {
            self.search(SingletonNeighborhood::new(target, Some(distance)))
                .into_option()
        } else {
            None
        }
    }

    /// Returns the up to `k` nearest neighbors to `target`.
    fn k_nearest(&self, target: &K, k: usize) -> Vec<Neighbor<&V, K::Distance>> {
        self.search(HeapNeighborhood::new(target, k, None))
            .into_vec()
    }

    /// Returns the up to `k` nearest neighbors to `target` within the distance `threshold`.
    fn k_nearest_within<D>(&self, target: &K, k: usize, threshold: D) -> Vec<Neighbor<&V, K::Distance>>
    where
        D: TryInto<K::Distance>,
    {
        if let Ok(distance) = threshold.try_into() {
            self.search(HeapNeighborhood::new(target, k, Some(distance)))
                .into_vec()
        } else {
            Vec::new()
        }
    }

    /// Search for nearest neighbors and add them to a neighborhood.
    fn search<'k, 'v, N>(&'v self, neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>;
}

/// Marker trait for [NearestNeighbors] implementations that always return exact results.
pub trait ExactNeighbors<K: Proximity<V>, V = K>: NearestNeighbors<K, V> {}

#[cfg(test)]
pub mod tests {
    use super::*;

    type Point = Euclidean<[f32; 3]>;

    /// Test a [NearestNeighbors] implementation.
    pub fn test_nearest_neighbors<T, F>(from_iter: F)
    where
        T: NearestNeighbors<Point>,
        F: Fn(Vec<Point>) -> T,
    {
        test_empty(&from_iter);
        test_pythagorean(&from_iter);
    }

    fn test_empty<T, F>(from_iter: &F)
    where
        T: NearestNeighbors<Point>,
        F: Fn(Vec<Point>) -> T,
    {
        let points = Vec::new();
        let index = from_iter(points);
        let target = Euclidean([0.0, 0.0, 0.0]);
        assert_eq!(index.nearest(&target), None);
        assert_eq!(index.nearest_within(&target, 1.0), None);
        assert!(index.k_nearest(&target, 0).is_empty());
        assert!(index.k_nearest(&target, 3).is_empty());
        assert!(index.k_nearest_within(&target, 0, 1.0).is_empty());
        assert!(index.k_nearest_within(&target, 3, 1.0).is_empty());
    }

    fn test_pythagorean<T, F>(from_iter: &F)
    where
        T: NearestNeighbors<Point>,
        F: Fn(Vec<Point>) -> T,
    {
        let points = vec![
            Euclidean([3.0, 4.0, 0.0]),
            Euclidean([5.0, 0.0, 12.0]),
            Euclidean([0.0, 8.0, 15.0]),
            Euclidean([1.0, 2.0, 2.0]),
            Euclidean([2.0, 3.0, 6.0]),
            Euclidean([4.0, 4.0, 7.0]),
        ];
        let index = from_iter(points);
        let target = Euclidean([0.0, 0.0, 0.0]);

        assert_eq!(
            index.nearest(&target).expect("No nearest neighbor found"),
            Neighbor::new(&Euclidean([1.0, 2.0, 2.0]), 3.0)
        );

        assert_eq!(index.nearest_within(&target, 2.0), None);
        assert_eq!(
            index.nearest_within(&target, 4.0).expect("No nearest neighbor found within 4.0"),
            Neighbor::new(&Euclidean([1.0, 2.0, 2.0]), 3.0)
        );

        assert!(index.k_nearest(&target, 0).is_empty());
        assert_eq!(
            index.k_nearest(&target, 3),
            vec![
                Neighbor::new(&Euclidean([1.0, 2.0, 2.0]), 3.0),
                Neighbor::new(&Euclidean([3.0, 4.0, 0.0]), 5.0),
                Neighbor::new(&Euclidean([2.0, 3.0, 6.0]), 7.0),
            ]
        );

        assert!(index.k_nearest(&target, 0).is_empty());
        assert_eq!(
            index.k_nearest_within(&target, 3, 6.0),
            vec![
                Neighbor::new(&Euclidean([1.0, 2.0, 2.0]), 3.0),
                Neighbor::new(&Euclidean([3.0, 4.0, 0.0]), 5.0),
            ]
        );
        assert_eq!(
            index.k_nearest_within(&target, 3, 8.0),
            vec![
                Neighbor::new(&Euclidean([1.0, 2.0, 2.0]), 3.0),
                Neighbor::new(&Euclidean([3.0, 4.0, 0.0]), 5.0),
                Neighbor::new(&Euclidean([2.0, 3.0, 6.0]), 7.0),
            ]
        );
    }
}
