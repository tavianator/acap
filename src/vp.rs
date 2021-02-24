//! [Vantage-point trees](https://en.wikipedia.org/wiki/Vantage-point_tree).

use crate::distance::{Distance, DistanceValue, Metric, Proximity};
use crate::util::Ordered;
use crate::{ExactNeighbors, NearestNeighbors, Neighborhood};

use num_traits::zero;

use std::fmt::{self, Debug, Formatter};
use std::iter::{Extend, FromIterator};
use std::ops::Deref;

/// A node in a VP tree.
#[derive(Debug)]
struct VpNode<T, R = DistanceValue<T>> {
    /// The vantage point itself.
    item: T,
    /// The radius of this node.
    radius: R,
    /// The subtree inside the radius, if any.
    inside: Option<Box<Self>>,
    /// The subtree outside the radius, if any.
    outside: Option<Box<Self>>,
}

impl<T: Proximity> VpNode<T> {
    /// Create a new VpNode.
    fn new(item: T) -> Self {
        Self {
            item,
            radius: zero(),
            inside: None,
            outside: None,
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

        Self::balanced_recursive(&mut nodes)
            .map(|node| *node)
    }

    /// Create a balanced subtree.
    fn balanced_recursive(nodes: &mut [Option<Box<Self>>]) -> Option<Box<Self>> {
        if let Some((node, children)) = nodes.split_first_mut() {
            let mut node = node.take().unwrap();
            children.sort_by_cached_key(|x| Ordered::new(node.distance_to_box(x)));

            let (inside, outside) = children.split_at_mut(children.len() / 2);
            if let Some(last) = inside.last() {
                node.radius = node.distance_to_box(last).value();
            }

            node.inside = Self::balanced_recursive(inside);
            node.outside = Self::balanced_recursive(outside);

            Some(node)
        } else {
            None
        }
    }

    /// Get the distance between to boxed nodes.
    fn distance_to_box(&self, child: &Option<Box<Self>>) -> T::Distance {
        self.item.distance(&child.as_ref().unwrap().item)
    }

    /// Push a new item into this subtree.
    fn push(&mut self, item: T) {
        match (&mut self.inside, &mut self.outside) {
            (None, None) => {
                self.outside = Some(Box::new(Self::new(item)));
            }
            (Some(inside), Some(outside)) => {
                if self.item.distance(&item) <= self.radius {
                    inside.push(item);
                } else {
                    outside.push(item);
                }
            }
            _ => {
                let node = Box::new(Self::new(item));
                let other = self.inside.take().xor(self.outside.take()).unwrap();

                let r1 = self.item.distance(&node.item);
                let r2 = self.item.distance(&other.item);

                if r1 <= r2 {
                    self.radius = r2.into();
                    self.inside = Some(node);
                    self.outside = Some(other);
                } else {
                    self.radius = r1.into();
                    self.inside = Some(other);
                    self.outside = Some(node);
                }
            }
        }
    }
}

trait VpSearch<K, V, N>: Copy
where
    K: Proximity<V, Distance = V::Distance>,
    V: Proximity,
    N: Neighborhood<K, V>,
{
    /// Get the vantage point of this node.
    fn item(self) -> V;

    /// Get the radius of this node.
    fn radius(self) -> DistanceValue<V>;

    /// Get the inside subtree.
    fn inside(self) -> Option<Self>;

    /// Get the outside subtree.
    fn outside(self) -> Option<Self>;

    /// Recursively search for nearest neighbors.
    fn search(self, neighborhood: &mut N) {
        let distance = neighborhood.consider(self.item()).into();

        if distance <= self.radius() {
            self.search_inside(distance, neighborhood);
            self.search_outside(distance, neighborhood);
        } else {
            self.search_outside(distance, neighborhood);
            self.search_inside(distance, neighborhood);
        }
    }

    /// Search the inside subtree.
    fn search_inside(self, distance: DistanceValue<V>, neighborhood: &mut N) {
        if let Some(inside) = self.inside() {
            if neighborhood.contains(distance - self.radius()) {
                inside.search(neighborhood);
            }
        }
    }

    /// Search the outside subtree.
    fn search_outside(self, distance: DistanceValue<V>, neighborhood: &mut N) {
        if let Some(outside) = self.outside() {
            if neighborhood.contains(self.radius() - distance) {
                outside.search(neighborhood);
            }
        }
    }
}

impl<'a, K, V, N> VpSearch<K, &'a V, N> for &'a VpNode<V>
where
    K: Proximity<&'a V, Distance = V::Distance>,
    V: Proximity,
    N: Neighborhood<K, &'a V>,
{
    fn item(self) -> &'a V {
        &self.item
    }

    fn radius(self) -> DistanceValue<V> {
        self.radius
    }

    fn inside(self) -> Option<Self> {
        self.inside.as_ref().map(Box::deref)
    }

    fn outside(self) -> Option<Self> {
        self.outside.as_ref().map(Box::deref)
    }
}

/// A [vantage-point tree](https://en.wikipedia.org/wiki/Vantage-point_tree).
pub struct VpTree<T: Proximity> {
    root: Option<VpNode<T>>,
}

impl<T: Proximity> VpTree<T> {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Create a balanced tree out of a sequence of items.
    pub fn balanced<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self {
            root: VpNode::balanced(items),
        }
    }

    /// Iterate over the items stored in this tree.
    pub fn iter(&self) -> Iter<T> {
        (&self).into_iter()
    }

    /// Rebalance this VP tree.
    pub fn balance(&mut self) {
        let mut nodes = Vec::new();
        if let Some(root) = self.root.take() {
            nodes.push(Some(Box::new(root)));
        }

        let mut i = 0;
        while i < nodes.len() {
            let node = nodes[i].as_mut().unwrap();
            let inside = node.inside.take();
            let outside = node.outside.take();
            if inside.is_some() {
                nodes.push(inside);
            }
            if outside.is_some() {
                nodes.push(outside);
            }

            i += 1;
        }

        self.root = VpNode::balanced_recursive(&mut nodes)
            .map(|node| *node);
    }

    /// Push a new item into the tree.
    ///
    /// Inserting elements individually tends to unbalance the tree.  Use [VpTree::balanced] if
    /// possible to create a balanced tree from a batch of items.
    pub fn push(&mut self, item: T) {
        if let Some(root) = &mut self.root {
            root.push(item);
        } else {
            self.root = Some(VpNode::new(item));
        }
    }
}

// Can't derive(Debug) due to https://github.com/rust-lang/rust/issues/26925
impl<T> Debug for VpTree<T>
where
    T: Proximity + Debug,
    DistanceValue<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("VpTree")
            .field("root", &self.root)
            .finish()
    }
}

impl<T: Proximity> Default for VpTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Proximity> Extend<T> for VpTree<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        if self.root.is_some() {
            for item in items {
                self.push(item);
            }
        } else {
            self.root = VpNode::balanced(items);
        }
    }
}

impl<T: Proximity> FromIterator<T> for VpTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self::balanced(items)
    }
}

/// An iterator that moves values out of a VP tree.
pub struct IntoIter<T: Proximity> {
    stack: Vec<VpNode<T>>,
}

impl<T: Proximity> IntoIter<T> {
    fn new(node: Option<VpNode<T>>) -> Self {
        Self {
            stack: node.into_iter().collect(),
        }
    }
}

impl<T> Debug for IntoIter<T>
where
    T: Proximity + Debug,
    DistanceValue<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("stack", &self.stack)
            .finish()
    }
}

impl<T: Proximity> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.stack.pop().map(|node| {
            if let Some(inside) = node.inside {
                self.stack.push(*inside);
            }
            if let Some(outside) = node.outside {
                self.stack.push(*outside);
            }
            node.item
        })
    }
}

impl<T: Proximity> IntoIterator for VpTree<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.root)
    }
}

/// An iterator over the values in a VP tree.
pub struct Iter<'a, T: Proximity> {
    stack: Vec<&'a VpNode<T>>,
}

impl<'a, T: Proximity> Iter<'a, T> {
    fn new(node: &'a Option<VpNode<T>>) -> Self {
        Self {
            stack: node.as_ref().into_iter().collect(),
        }
    }
}

impl<'a, T> Debug for Iter<'a, T>
where
    T: Proximity + Debug,
    DistanceValue<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("Iter")
            .field("stack", &self.stack)
            .finish()
    }
}

impl<'a, T: Proximity> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.stack.pop().map(|node| {
            if let Some(inside) = &node.inside {
                self.stack.push(inside);
            }
            if let Some(outside) = &node.outside {
                self.stack.push(outside);
            }
            &node.item
        })
    }
}

impl<'a, T: Proximity> IntoIterator for &'a VpTree<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(&self.root)
    }
}

impl<K, V> NearestNeighbors<K, V> for VpTree<V>
where
    K: Proximity<V, Distance = V::Distance>,
    V: Proximity,
{
    fn search<'k, 'v, N>(&'v self, mut neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>,
    {
        if let Some(root) = &self.root {
            root.search(&mut neighborhood);
        }
        neighborhood
    }
}

impl<K, V> ExactNeighbors<K, V> for VpTree<V>
where
    K: Metric<V, Distance = V::Distance>,
    V: Metric,
{}

/// A node in a flat VP tree.
#[derive(Debug)]
struct FlatVpNode<T, R = DistanceValue<T>> {
    /// The vantage point itself.
    item: T,
    /// The radius of this node.
    radius: R,
    /// The size of the inside subtree.
    inside_len: usize,
}

impl<T: Proximity> FlatVpNode<T> {
    /// Create a new FlatVpNode.
    fn new(item: T) -> Self {
        Self {
            item,
            radius: zero(),
            inside_len: 0,
        }
    }

    /// Create a balanced tree.
    fn balanced<I: IntoIterator<Item = T>>(items: I) -> Vec<Self> {
        let mut nodes: Vec<_> = items
            .into_iter()
            .map(Self::new)
            .collect();

        Self::balance_recursive(&mut nodes);

        nodes
    }

    /// Create a balanced subtree.
    fn balance_recursive(nodes: &mut [Self]) {
        if let Some((node, children)) = nodes.split_first_mut() {
            children.sort_by_cached_key(|x| Ordered::new(node.item.distance(&x.item)));

            let (inside, outside) = children.split_at_mut(children.len() / 2);
            if let Some(last) = inside.last() {
                node.radius = node.item.distance(&last.item).into();
            }

            node.inside_len = inside.len();

            Self::balance_recursive(inside);
            Self::balance_recursive(outside);
        }
    }
}

impl<'a, K, V, N> VpSearch<K, &'a V, N> for &'a [FlatVpNode<V>]
where
    K: Proximity<&'a V, Distance = V::Distance>,
    V: Proximity,
    N: Neighborhood<K, &'a V>,
{
    fn item(self) -> &'a V {
        &self[0].item
    }

    fn radius(self) -> DistanceValue<V> {
        self[0].radius
    }

    fn inside(self) -> Option<Self> {
        let end = self[0].inside_len + 1;
        if end > 1 {
            Some(&self[1..end])
        } else {
            None
        }
    }

    fn outside(self) -> Option<Self> {
        let start = self[0].inside_len + 1;
        if start < self.len() {
            Some(&self[start..])
        } else {
            None
        }
    }
}

/// A [vantage-point tree] stored as a flat array.
///
/// A FlatVpTree is always balanced and usually more efficient than a [VpTree], but doesn't support
/// dynamic updates.
///
/// [vantage-point tree]: https://en.wikipedia.org/wiki/Vantage-point_tree
pub struct FlatVpTree<T: Proximity> {
    nodes: Vec<FlatVpNode<T>>,
}

impl<T: Proximity> FlatVpTree<T> {
    /// Create a balanced tree out of a sequence of items.
    pub fn balanced<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self {
            nodes: FlatVpNode::balanced(items),
        }
    }

    /// Iterate over the items stored in this tree.
    pub fn iter(&self) -> FlatIter<T> {
        (&self).into_iter()
    }
}

impl<T> Debug for FlatVpTree<T>
where
    T: Proximity + Debug,
    DistanceValue<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("FlatVpTree")
            .field("nodes", &self.nodes)
            .finish()
    }
}

impl<T: Proximity> FromIterator<T> for FlatVpTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self::balanced(items)
    }
}

/// An iterator that moves values out of a flat VP tree.
pub struct FlatIntoIter<T: Proximity>(std::vec::IntoIter<FlatVpNode<T>>);

impl<T> Debug for FlatIntoIter<T>
where
    T: Proximity + Debug,
    DistanceValue<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_tuple("FlatIntoIter")
            .field(&self.0)
            .finish()
    }
}

impl<T: Proximity> Iterator for FlatIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.next().map(|n| n.item)
    }
}

impl<T: Proximity> IntoIterator for FlatVpTree<T> {
    type Item = T;
    type IntoIter = FlatIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        FlatIntoIter(self.nodes.into_iter())
    }
}

/// An iterator over the values in a flat VP tree.
pub struct FlatIter<'a, T: Proximity>(std::slice::Iter<'a, FlatVpNode<T>>);

impl<'a, T> Debug for FlatIter<'a, T>
where
    T: Proximity + Debug,
    DistanceValue<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_tuple("FlatIter")
            .field(&self.0)
            .finish()
    }
}

impl<'a, T: Proximity> Iterator for FlatIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.0.next().map(|n| &n.item)
    }
}

impl<'a, T: Proximity> IntoIterator for &'a FlatVpTree<T> {
    type Item = &'a T;
    type IntoIter = FlatIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        FlatIter(self.nodes.iter())
    }
}

impl<K, V> NearestNeighbors<K, V> for FlatVpTree<V>
where
    K: Proximity<V, Distance = V::Distance>,
    V: Proximity,
{
    fn search<'k, 'v, N>(&'v self, mut neighborhood: N) -> N
    where
        K: 'k,
        V: 'v,
        N: Neighborhood<&'k K, &'v V>,
    {
        if !self.nodes.is_empty() {
            self.nodes.as_slice().search(&mut neighborhood);
        }
        neighborhood
    }
}

impl<K, V> ExactNeighbors<K, V> for FlatVpTree<V>
where
    K: Metric<V, Distance = V::Distance>,
    V: Metric,
{}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::test_exact_neighbors;

    #[test]
    fn test_vp_tree() {
        test_exact_neighbors(VpTree::from_iter);
    }

    #[test]
    fn test_unbalanced_vp_tree() {
        test_exact_neighbors(|points| {
            let mut tree = VpTree::new();
            for point in points {
                tree.push(point);
            }
            tree
        });
    }

    #[test]
    fn test_flat_vp_tree() {
        test_exact_neighbors(FlatVpTree::from_iter);
    }
}
