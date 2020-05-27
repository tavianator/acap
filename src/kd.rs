//! k-d trees.

use crate::coords::{Coordinates, CoordinateMetric, CoordinateProximity};
use crate::distance::{Metric, Proximity};
use crate::util::Ordered;
use crate::{ExactNeighbors, NearestNeighbors, Neighborhood};

use std::iter::FromIterator;
use std::ops::Deref;

/// A node in a k-d tree.
#[derive(Debug)]
struct KdNode<T> {
    /// The vantage point itself.
    item: T,
    /// The left subtree, if any.
    left: Option<Box<Self>>,
    /// The right subtree, if any.
    right: Option<Box<Self>>,
}

impl<T: Coordinates> KdNode<T> {
    /// Create a new KdNode.
    fn new(item: T) -> Self {
        Self {
            item,
            left: None,
            right: None,
        }
    }

    /// Create a balanced tree.
    fn balanced<I: IntoIterator<Item = T>>(items: I) -> Option<Self> {
        let mut nodes: Vec<_> = items
            .into_iter()
            .map(Self::new)
            .map(Box::new)
            .map(Some)
            .collect();

        Self::balanced_recursive(&mut nodes, 0)
            .map(|node| *node)
    }

    /// Create a balanced subtree.
    fn balanced_recursive(nodes: &mut [Option<Box<Self>>], level: usize) -> Option<Box<Self>> {
        if nodes.is_empty() {
            return None;
        }

        nodes.sort_by_cached_key(|x| Ordered::new(x.as_ref().unwrap().item.coord(level)));

        let (left, right) = nodes.split_at_mut(nodes.len() / 2);
        let (node, right) = right.split_first_mut().unwrap();
        let mut node = node.take().unwrap();

        let next = (level + 1) % node.item.dims();
        node.left = Self::balanced_recursive(left, next);
        node.right = Self::balanced_recursive(right, next);

        Some(node)
    }

    /// Push a new item into this subtree.
    fn push(&mut self, item: T, level: usize) {
        let next = (level + 1) % item.dims();

        if item.coord(level) <= self.item.coord(level) {
            if let Some(left) = &mut self.left {
                left.push(item, next);
            } else {
                self.left = Some(Box::new(Self::new(item)));
            }
        } else {
            if let Some(right) = &mut self.right {
                right.push(item, next);
            } else {
                self.right = Some(Box::new(Self::new(item)));
            }
        }
    }
}

/// Marker trait for [Proximity] implementations that are compatible with k-d trees.
pub trait KdProximity<V: ?Sized = Self>
where
    Self: Coordinates<Value = V::Value>,
    Self: Proximity<V>,
    Self: CoordinateProximity<V::Value, Distance = <Self as Proximity<V>>::Distance>,
    V: Coordinates,
{}

/// Blanket [KdProximity] implementation.
impl<K, V> KdProximity<V> for K
where
    K: Coordinates<Value = V::Value>,
    K: Proximity<V>,
    K: CoordinateProximity<V::Value, Distance = <K as Proximity<V>>::Distance>,
    V: Coordinates,
{}

/// Marker trait for [Metric] implementations that are compatible with k-d tree.
pub trait KdMetric<V: ?Sized = Self>
where
    Self: KdProximity<V>,
    Self: Metric<V>,
    Self: CoordinateMetric<V::Value>,
    V: Coordinates,
{}

/// Blanket [KdMetric] implementation.
impl<K, V> KdMetric<V> for K
where
    K: KdProximity<V>,
    K: Metric<V>,
    K: CoordinateMetric<V::Value>,
    V: Coordinates,
{}

trait KdSearch<K, V, N>: Copy
where
    K: KdProximity<V>,
    V: Coordinates + Copy,
    N: Neighborhood<K, V>,
{
    /// Get this node's item.
    fn item(self) -> V;

    /// Get the left subtree.
    fn left(self) -> Option<Self>;

    /// Get the right subtree.
    fn right(self) -> Option<Self>;

    /// Recursively search for nearest neighbors.
    fn search(self, level: usize, closest: &mut [V::Value], neighborhood: &mut N) {
        let item = self.item();
        neighborhood.consider(item);

        let target = neighborhood.target();

        if target.coord(level) <= item.coord(level) {
            self.search_near(self.left(), level, closest, neighborhood);
            self.search_far(self.right(), level, closest, neighborhood);
        } else {
            self.search_near(self.right(), level, closest, neighborhood);
            self.search_far(self.left(), level, closest, neighborhood);
        }
    }

    /// Search the subtree closest to the target.
    fn search_near(self, near: Option<Self>, level: usize, closest: &mut [V::Value], neighborhood: &mut N) {
        if let Some(near) = near {
            let next = (level + 1) % self.item().dims();
            near.search(next, closest, neighborhood);
        }
    }

    /// Search the subtree farthest from the target.
    fn search_far(self, far: Option<Self>, level: usize, closest: &mut [V::Value], neighborhood: &mut N) {
        if let Some(far) = far {
            // Update the closest possible point
            let item = self.item();
            let target = neighborhood.target();
            let saved = std::mem::replace(&mut closest[level], item.coord(level));
            if neighborhood.contains(target.distance_to_coords(closest)) {
                let next = (level + 1) % item.dims();
                far.search(next, closest, neighborhood);
            }
            closest[level] = saved;
        }
    }
}

impl<'a, K, V, N> KdSearch<K, &'a V, N> for &'a KdNode<V>
where
    K: KdProximity<&'a V>,
    V: Coordinates,
    N: Neighborhood<K, &'a V>,
{
    fn item(self) -> &'a V {
        &self.item
    }

    fn left(self) -> Option<Self> {
        self.left.as_ref().map(Box::deref)
    }

    fn right(self) -> Option<Self> {
        self.right.as_ref().map(Box::deref)
    }
}

/// A [k-d tree](https://en.wikipedia.org/wiki/K-d_tree).
#[derive(Debug)]
pub struct KdTree<T> {
    root: Option<KdNode<T>>,
}

impl<T: Coordinates> KdTree<T> {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            root: None,
        }
    }

    /// Create a balanced tree out of a sequence of items.
    pub fn balanced<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self {
            root: KdNode::balanced(items),
        }
    }

    /// Rebalance this k-d tree.
    pub fn balance(&mut self) {
        let mut nodes = Vec::new();
        if let Some(root) = self.root.take() {
            nodes.push(Some(Box::new(root)));
        }

        let mut i = 0;
        while i < nodes.len() {
            let node = nodes[i].as_mut().unwrap();
            let inside = node.left.take();
            let outside = node.right.take();
            if inside.is_some() {
                nodes.push(inside);
            }
            if outside.is_some() {
                nodes.push(outside);
            }

            i += 1;
        }

        self.root = KdNode::balanced_recursive(&mut nodes, 0)
            .map(|node| *node);
    }

    /// Push a new item into the tree.
    ///
    /// Inserting elements individually tends to unbalance the tree.  Use [KdTree::balanced] if
    /// possible to create a balanced tree from a batch of items.
    pub fn push(&mut self, item: T) {
        if let Some(root) = &mut self.root {
            root.push(item, 0);
        } else {
            self.root = Some(KdNode::new(item));
        }
    }
}

impl<T: Coordinates> Extend<T> for KdTree<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        if self.root.is_some() {
            for item in items {
                self.push(item);
            }
        } else {
            self.root = KdNode::balanced(items);
        }
    }
}

impl<T: Coordinates> FromIterator<T> for KdTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self::balanced(items)
    }
}

/// An iterator that moves values out of a k-d tree.
#[derive(Debug)]
pub struct IntoIter<T> {
    stack: Vec<KdNode<T>>,
}

impl<T> IntoIter<T> {
    fn new(node: Option<KdNode<T>>) -> Self {
        Self {
            stack: node.into_iter().collect(),
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.stack.pop().map(|node| {
            if let Some(left) = node.left {
                self.stack.push(*left);
            }
            if let Some(right) = node.right {
                self.stack.push(*right);
            }
            node.item
        })
    }
}

impl<T> IntoIterator for KdTree<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.root)
    }
}

impl<K, V> NearestNeighbors<K, V> for KdTree<V>
where
    K: KdProximity<V>,
    V: Coordinates,
{
    fn search<'k, 'v, N>(&'v self, mut neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>,
    {
        if let Some(root) = &self.root {
            let mut closest = neighborhood.target().as_vec();
            root.search(0, &mut closest, &mut neighborhood);
        }
        neighborhood
    }
}

impl<K, V> ExactNeighbors<K, V> for KdTree<V>
where
    K: KdMetric<V>,
    V: Coordinates,
{}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::test_nearest_neighbors;

    #[test]
    fn test_kd_tree() {
        test_nearest_neighbors(KdTree::from_iter);
    }

    #[test]
    fn test_unbalanced_kd_tree() {
        test_nearest_neighbors(|points| {
            let mut tree = KdTree::new();
            for point in points {
                tree.push(point);
            }
            tree
        });
    }
}
