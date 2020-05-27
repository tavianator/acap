//! Benchmark for various NearestNeighbors implementations.

use acap::euclid::Euclidean;
use acap::exhaustive::ExhaustiveSearch;
use acap::kd::KdTree;
use acap::vp::{FlatVpTree, VpTree};
use acap::NearestNeighbors;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use std::iter::FromIterator;

type Point = Euclidean<[f32; 3]>;

/// Generates a spiral used as the benchmark data set.
fn spiral() -> Vec<Point> {
    let mut points = Vec::new();

    let size = 1000;
    let turns = 10.0;
    for i in 0..size {
        let y = 2.0 * (i as f32) / (size as f32) - 1.0;
        let m = (1.0 - y * y).sqrt();
        let theta = turns * y * std::f32::consts::PI;
        let (sin, cos) = theta.sin_cos();
        let x = m * cos * cos;
        let z = m * sin * cos;
        points.push(Euclidean([x, y, z]));
    }

    points
}

fn bench_from_iter(c: &mut Criterion) {
    let points = black_box(spiral());

    let mut group = c.benchmark_group("from_iter");
    group.bench_function("ExhaustiveSearch", |b| b.iter(|| ExhaustiveSearch::from_iter(points.clone())));
    group.bench_function("VpTree", |b| b.iter(|| VpTree::from_iter(points.clone())));
    group.bench_function("FlatVpTree", |b| b.iter(|| FlatVpTree::from_iter(points.clone())));
    group.bench_function("KdTree", |b| b.iter(|| KdTree::from_iter(points.clone())));
    group.finish();
}

fn bench_nearest_neighbors(c: &mut Criterion) {
    let points = black_box(spiral());
    let target = black_box(Euclidean([0.0, 0.0, 0.0]));

    let exhaustive = ExhaustiveSearch::from_iter(points.clone());
    let vp_tree = VpTree::from_iter(points.clone());
    let flat_vp_tree = FlatVpTree::from_iter(points.clone());
    let kd_tree = KdTree::from_iter(points.clone());

    let mut nearest = c.benchmark_group("NearestNeighbors::nearest");
    nearest.bench_function("ExhaustiveSearch", |b| b.iter(|| exhaustive.nearest(&target)));
    nearest.bench_function("VpTree", |b| b.iter(|| vp_tree.nearest(&target)));
    nearest.bench_function("FlatVpTree", |b| b.iter(|| flat_vp_tree.nearest(&target)));
    nearest.bench_function("KdTree", |b| b.iter(|| kd_tree.nearest(&target)));
    nearest.finish();

    let mut nearest_within = c.benchmark_group("NearestNeighbors::nearest_within");
    nearest_within.bench_function("ExhaustiveSearch", |b| b.iter(|| exhaustive.nearest_within(&target, 0.1)));
    nearest_within.bench_function("VpTree", |b| b.iter(|| vp_tree.nearest_within(&target, 0.1)));
    nearest_within.bench_function("FlatVpTree", |b| b.iter(|| flat_vp_tree.nearest_within(&target, 0.1)));
    nearest_within.bench_function("KdTree", |b| b.iter(|| kd_tree.nearest_within(&target, 0.1)));
    nearest_within.finish();

    let mut k_nearest = c.benchmark_group("NearestNeighbors::k_nearest");
    k_nearest.bench_function("ExhaustiveSearch", |b| b.iter(|| exhaustive.k_nearest(&target, 3)));
    k_nearest.bench_function("VpTree", |b| b.iter(|| vp_tree.k_nearest(&target, 3)));
    k_nearest.bench_function("FlatVpTree", |b| b.iter(|| flat_vp_tree.k_nearest(&target, 3)));
    k_nearest.bench_function("KdTree", |b| b.iter(|| kd_tree.k_nearest(&target, 3)));
    k_nearest.finish();

    let mut k_nearest_within = c.benchmark_group("NearestNeighbors::k_nearest_within");
    k_nearest_within.bench_function("ExhaustiveSearch", |b| b.iter(|| exhaustive.k_nearest_within(&target, 3, 0.1)));
    k_nearest_within.bench_function("VpTree", |b| b.iter(|| vp_tree.k_nearest_within(&target, 3, 0.1)));
    k_nearest_within.bench_function("FlatVpTree", |b| b.iter(|| flat_vp_tree.k_nearest_within(&target, 3, 0.1)));
    k_nearest_within.bench_function("KdTree", |b| b.iter(|| kd_tree.k_nearest_within(&target, 3, 0.1)));
    k_nearest_within.finish();
}

criterion_group!(benches, bench_from_iter, bench_nearest_neighbors);
criterion_main!(benches);
