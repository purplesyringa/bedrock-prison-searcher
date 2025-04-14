use std::time::Instant;
use core::fmt;
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

#[derive(Clone)]
struct ComponentInfo {
    size: u32,
    frontier_size: u32,
}

fn enumerate_interior_regions(
    noise: &BedrockFloorNoise,
    mut callback: impl FnMut((i32, i32), u32),
) {
    // This stores component ID for each interior cell. `0` means this is an exterior cell.
    let mut cell_component: Vec<u32> = vec![0; WORLD_BORDER as usize * 2 + 1];

    // Component info. Unallocated components are denoted by frontier size 0.
    let mut component_info: Vec<ComponentInfo> = vec![
        ComponentInfo {
            size: 0,
            frontier_size: 0,
        };
        WORLD_BORDER as usize + 2
    ];
    let mut component_allocator_ptr = 1;

    for z in -WORLD_BORDER..=-WORLD_BORDER + 10 {
        let mut left = 0;

        for x in -WORLD_BORDER..=WORLD_BORDER {
            let i = (x + WORLD_BORDER) as usize;

            let up = cell_component[i];

            let current = if noise.is_interior(x, z) {
                if left == 0 && up == 0 {
                    // Create a new current component
                    while component_info[component_allocator_ptr].frontier_size != 0 {
                        component_allocator_ptr += 1;
                        if component_allocator_ptr == component_info.len() {
                            component_allocator_ptr = 1;
                        }
                    }
                    let component_id = component_allocator_ptr as u32;
                    component_allocator_ptr += 1;
                    if component_allocator_ptr == component_info.len() {
                        component_allocator_ptr = 1;
                    }

                    component_info[component_id as usize] = ComponentInfo {
                        size: 1,
                        frontier_size: 1,
                    };

                    component_id
                } else if up == 0 {
                    // Reuse left component
                    component_info[left as usize].size += 1;
                    component_info[left as usize].frontier_size += 1;
                    left
                } else if left == up {
                    // Reuse left=up component
                    component_info[left as usize].size += 1;
                    left
                } else {
                    // Reuse up component
                    component_info[up as usize].size += 1;

                    if left != 0 {
                        // Merge left into up
                        let left_size = component_info[left as usize].size;

                        component_info[up as usize].size += left_size;
                        component_info[up as usize].frontier_size += component_info[left as usize].frontier_size;

                        // Remap
                        for x1 in (x - left_size as i32).max(-WORLD_BORDER)
                            ..(x - 1 + left_size as i32).min(WORLD_BORDER)
                        {
                            let i1 = (x1 + WORLD_BORDER) as usize;
                            if cell_component[i1] == left {
                                cell_component[i1] = up;
                            }
                        }

                        // Drop left
                        component_info[left as usize].frontier_size = 0;
                    }

                    up
                }
            } else {
                if up != 0 {
                    component_info[up as usize].frontier_size -= 1;
                    if component_info[up as usize].frontier_size == 0 {
                        // Finalize and drop up
                        callback((x, z - 1), component_info[up as usize].size);
                    }
                }

                0
            };

            cell_component[i] = current;
            left = current;
        }
    }
}

fn main() {
    let noise = BedrockFloorNoise::from_world_seed(-972064012444369952i64 as u64);

    let mut best_coords = (0, 0);
    let mut best_size = 0;

    // for z in -29999983 - 5..-29999983 + 5 {
    //     for x in -25202376 - 5..-25202376 + 5 {
    //         print!("{}", noise.get_column_type(x, z));
    //     }
    //     println!();
    // }

    let start_instant = Instant::now();

    enumerate_interior_regions(&noise, |start_coords, size| {
        if size <= best_size {
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
            "[{:?}] found {} (alleged {}) at {:?}",
            start_instant.elapsed(), interior_count, size, start_coords,
        );
        assert_eq!(interior_count, size);

        best_coords = start_coords;
        best_size = size;
    });
}
