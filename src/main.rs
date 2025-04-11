use core::fmt;
use core::ops::{Index, Range};
use std::collections::HashSet;

struct Xoroshiro {
    low: u64,
    high: u64,
}

impl Xoroshiro {
    fn new(low: u64, high: u64) -> Self {
        if low == 0 && high == 0 {
            Self {
                low: 0x9e3779b97f4a7c15,
                high: 0x6a09e667f3bcc909,
            }
        } else {
            Self { low, high }
        }
    }

    fn next(&mut self) -> u64 {
        let Self { low, mut high } = *self;
        let mid = low.wrapping_add(high).rotate_left(17).wrapping_add(low);
        high ^= low;
        self.low = low.rotate_left(49) ^ high ^ (high << 21);
        self.high = high.rotate_left(28);
        mid
    }
}

fn mix_stafford_13(mut seed: u64) -> u64 {
    seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d049bb133111eb);
    seed ^ (seed >> 31)
}

struct BedrockFloorNoise {
    floor_seed_low: u64,
    floor_seed_high: u64,
}

impl BedrockFloorNoise {
    fn from_world_seed(world_seed: u64) -> Self {
        let base_seed_low = world_seed ^ 0x6a09e667f3bcc909;
        let base_seed_high = base_seed_low.wrapping_add(0x9e3779b97f4a7c15);

        let mut world_prng = Xoroshiro::new(
            mix_stafford_13(base_seed_low),
            mix_stafford_13(base_seed_high),
        );
        let world_seed_low = world_prng.next();
        let world_seed_high = world_prng.next();

        // md5 of "minecraft:bedrock_floor"
        let (const_seed_low, const_seed_high) = (0xbbf7928b7bf1d285, 0xc4dc7cf90e1b3b94);

        let mut floor_prng = Xoroshiro::new(
            const_seed_low ^ world_seed_low,
            const_seed_high ^ world_seed_high,
        );
        Self {
            floor_seed_low: floor_prng.next(),
            floor_seed_high: floor_prng.next(),
        }
    }

    fn is_bedrock(&self, x: i32, y: i32, z: i32) -> bool {
        if y <= -64 {
            true
        } else if y >= -59 {
            false
        } else {
            let block_key = hash_three(x, y, z);
            let mut block_prng =
                Xoroshiro::new(block_key ^ self.floor_seed_low, self.floor_seed_high);

            // equivalent to
            // let boundary = map_via_lerp(y as f64, -64.0, -59.0, 1.0, 0.0);
            // (block_prng.next() >> 40) as f64 * 2.0f64.powi(-24) < boundary

            static BOUNDARIES: [u64; 4] =
                [13421773 << 40, 10066330 << 40, 6710887 << 40, 3355444 << 40];
            block_prng.next() < BOUNDARIES[(y + 63) as usize]
        }
    }

    fn is_interior(&self, x: i32, z: i32) -> bool {
        !self.is_bedrock(x, -63, z) && !self.is_bedrock(x, -62, z)
    }

    fn get_column_type(&self, x: i32, z: i32) -> ColumnType {
        // .. -> Interior
        // .# -> Wall
        // ## -> Wall
        // #.# -> Wall
        // #.. -> Hazard
        if self.is_bedrock(x, -62, z) {
            ColumnType::Wall
        } else {
            if self.is_bedrock(x, -63, z) {
                if self.is_bedrock(x, -61, z) {
                    ColumnType::Wall
                } else {
                    ColumnType::Hazard
                }
            } else {
                ColumnType::Interior
            }
        }
    }
}

enum ColumnType {
    Interior,
    Wall,
    Hazard,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Interior => write!(f, "."),
            Self::Wall => write!(f, "#"),
            Self::Hazard => write!(f, "!"),
        }
    }
}

// fn map_via_lerp(value: f64, from_old: f64, to_old: f64, from_new: f64, to_new: f64) -> f64 {
//     let normalized = (value - from_old) / (to_old - from_old);
//     from_new + normalized * (to_new - from_new)
// }

fn hash_three(x: i32, y: i32, z: i32) -> u64 {
    let mut l: i64 = (x.wrapping_mul(3129871) as i64) ^ (y as i64) ^ ((z as i64) * 116129781);
    l = l.wrapping_mul(l.wrapping_mul(42317861).wrapping_add(11));
    (l >> 16) as u64
}

const WORLD_BORDER: i32 = 29_999_984;

struct InteriorRowSegmentIterator<'a> {
    x: i32,
    z: i32,
    noise: &'a BedrockFloorNoise,
}

impl<'a> InteriorRowSegmentIterator<'a> {
    fn new(z: i32, noise: &'a BedrockFloorNoise) -> Self {
        Self {
            x: -WORLD_BORDER,
            z,
            noise,
        }
    }
}

impl Iterator for InteriorRowSegmentIterator<'_> {
    type Item = Range<i32>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.x <= WORLD_BORDER && !self.noise.is_interior(self.x, self.z) {
            self.x += 1;
        }
        if self.x > WORLD_BORDER {
            return None;
        }
        let x_start = self.x;
        while self.x <= WORLD_BORDER && self.noise.is_interior(self.x, self.z) {
            self.x += 1;
        }
        let x_end = self.x;
        Some(x_start..x_end)
    }
}

#[derive(Clone, Debug)]
struct RegionInfo {
    size: u32,
}

struct Graph<T> {
    nodes: Vec<GraphNode<T>>,
}

struct GraphNode<T> {
    value: T,
    adjacent_nodes: Vec<u32>,
}

impl<T> Graph<T> {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn add_node(&mut self, value: T) -> u32 {
        self.nodes.push(GraphNode {
            value,
            adjacent_nodes: Vec::new(),
        });
        assert!(self.nodes.len() - 1 < u32::MAX as usize, "too many nodes"); // u32::MAX is used as a sentinel value by component explorer
        (self.nodes.len() - 1) as u32
    }

    fn add_edge(&mut self, u: u32, v: u32) {
        self.nodes[u as usize].adjacent_nodes.push(v);
        self.nodes[v as usize].adjacent_nodes.push(u);
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn new_component_explorer(&self) -> ComponentExplorer<'_, T> {
        ComponentExplorer {
            graph: self,
            component_ids: vec![u32::MAX; self.nodes.len()],
            n_explored_components: 0,
        }
    }
}

struct ComponentExplorer<'a, T> {
    graph: &'a Graph<T>,
    component_ids: Vec<u32>,
    n_explored_components: u32,
}

enum ExplorationStatus {
    New(Vec<u32>),
    Old,
}

impl<'a, T> ComponentExplorer<'a, T> {
    fn explore(&mut self, node_id: u32) -> (u32, ExplorationStatus) {
        let component_id = self.component_ids[node_id as usize];
        if component_id != u32::MAX {
            return (component_id, ExplorationStatus::Old);
        }

        fn dfs<T>(
            graph: &Graph<T>,
            component_ids: &mut [u32],
            nodes_in_component: &mut Vec<u32>,
            current_node_id: u32,
            component_id: u32,
        ) {
            component_ids[current_node_id as usize] = component_id;
            nodes_in_component.push(current_node_id);
            for neighbor_id in &graph.nodes[current_node_id as usize].adjacent_nodes {
                if component_ids[*neighbor_id as usize] == u32::MAX {
                    dfs(graph, component_ids, nodes_in_component, *neighbor_id, component_id);
                }
            }
        }

        let mut nodes_in_component = Vec::new();

        let component_id = self.n_explored_components;
        self.n_explored_components += 1;

        dfs(
            &self.graph,
            &mut self.component_ids,
            &mut nodes_in_component,
            node_id,
            component_id,
        );

        (component_id, ExplorationStatus::New(nodes_in_component))
    }
}

impl<T> Index<u32> for Graph<T> {
    type Output = T;

    fn index(&self, index: u32) -> &T {
        &self.nodes[index as usize].value
    }
}

fn enumerate_interior_regions(
    noise: &BedrockFloorNoise,
    mut callback: impl FnMut((i32, i32), RegionInfo),
) {
    let mut cell_is_interior: Vec<bool> = vec![false; WORLD_BORDER as usize * 2 + 1];

    // Sorted by key, and the first occurences of each value are 0, 1, 2, ...
    let mut cell_regions: Vec<(i32, u32)> = Vec::new();

    let mut regions: Vec<RegionInfo> = Vec::new();

    for z in -WORLD_BORDER..=-WORLD_BORDER + 10 {
        let mut graph: Graph<(RegionInfo, Range<i32>)> = Graph::new();
        for (x, region_id) in &cell_regions {
            if *region_id == graph.len() as u32 {
                // First occurence of region. The range does not really mean anything, but lets us
                // obtain the x later when finalizing the node.
                graph.add_node((regions[*region_id as usize].clone(), *x..x + 1));
            }
        }

        let mut next_cell_is_interior: Vec<bool> = vec![false; WORLD_BORDER as usize * 2 + 1];
        let mut cell_region_iter = cell_regions.into_iter().peekable();
        for segment in InteriorRowSegmentIterator::new(z, &noise) {
            let mut size = segment.len() as u32;

            // Find existing regions corresponding to this segment
            let mut regions_to_merge_with = Vec::new();
            for x in segment.clone() {
                next_cell_is_interior[(x + WORLD_BORDER) as usize] = true;
                if !cell_is_interior[(x + WORLD_BORDER) as usize] {
                    continue;
                }
                // Merge with region above
                while cell_region_iter.peek().is_some_and(|(top_x, _)| *top_x < x) {
                    cell_region_iter.next();
                }
                if cell_region_iter
                    .peek()
                    .is_some_and(|(top_x, _)| *top_x == x)
                {
                    let top_region = cell_region_iter.next().unwrap().1;
                    regions_to_merge_with.push(top_region);
                } else {
                    // Isolated region of size 1 we are not otherwise interested in
                    size += 1;
                }
            }

            if size == 1 && regions_to_merge_with.len() == 0 {
                // An isolated region of size 1 we're not interested in storing info for
                continue;
            }

            let bottom_node = graph.add_node((RegionInfo { size }, segment));
            for top_node in regions_to_merge_with {
                graph.add_edge(top_node, bottom_node);
            }
        }

        let mut next_cell_regions: Vec<(i32, u32)> = Vec::new();
        let mut next_regions: Vec<RegionInfo> = Vec::new();

        let mut components = graph.new_component_explorer();

        for bottom_node in regions.len()..graph.len() {
            let bottom_node = bottom_node as u32;

            let (region_id, status) = components.explore(bottom_node);
            if let ExplorationStatus::New(nodes) = status {
                next_regions.push(RegionInfo {
                    size: nodes.iter().map(|node| graph[*node].0.size).sum(),
                });
            }

            let (_, segment) = &graph[bottom_node];
            for x in segment.clone() {
                next_cell_regions.push((x, region_id));
            }
        }

        for top_node in 0..regions.len() {
            let top_node = top_node as u32;

            if let (_, ExplorationStatus::New(_)) = components.explore(top_node) {
                // A lone top node, finalize it
                let (region_info, segment) = &graph[top_node];
                callback((segment.start, z - 1), region_info.clone());
            }
        }

        cell_is_interior = next_cell_is_interior;
        cell_regions = next_cell_regions;
        regions = next_regions;

        println!(
            "{} cell->region maps, {} regions",
            cell_regions.len(),
            regions.len()
        );
    }
}

fn main() {
    let noise = BedrockFloorNoise::from_world_seed(-972064012444369952i64 as u64);

    let mut best_coords = (0, 0);
    let mut best_size = 0;

    // for z in -30000000..-30000000 + 5 {
    //     for x in -29994691 - 5..-29994691 + 5 {
    //         print!("{}", noise.get_column_type(x, z));
    //     }
    //     println!();
    // }

    enumerate_interior_regions(&noise, |start_coords, region_info| {
        if region_info.size <= best_size {
            return;
        }

        fn dfs(
            noise: &BedrockFloorNoise,
            visited: &mut HashSet<(i32, i32)>,
            interior_count: &mut u32,
            (x, z): (i32, i32),
        ) -> Result<(), ()> {
            visited.insert((x, z));

            let ty = if (-WORLD_BORDER..=WORLD_BORDER).contains(&x)
                && (-WORLD_BORDER..=WORLD_BORDER).contains(&z)
            {
                noise.get_column_type(x, z)
            } else {
                ColumnType::Wall
            };
            match ty {
                ColumnType::Interior => {}
                ColumnType::Wall => return Ok(()),
                ColumnType::Hazard => return Err(()),
            }
            *interior_count += 1;

            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let x1 = x + dx;
                let z1 = z + dz;
                if !visited.contains(&(x1, z1)) {
                    dfs(noise, visited, interior_count, (x1, z1))?;
                }
            }
            Ok(())
        }

        let mut visited = HashSet::new();
        let mut interior_count = 0;
        if dfs(&noise, &mut visited, &mut interior_count, start_coords).is_err() {
            // Hazard encountered
            return;
        }

        println!(
            "found {} (alleged {}) at {:?}",
            interior_count, region_info.size, start_coords,
        );
        assert_eq!(interior_count, region_info.size);

        best_coords = start_coords;
        best_size = region_info.size;
    });
}
