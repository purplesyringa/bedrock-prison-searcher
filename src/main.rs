#![feature(portable_simd)]

use core::fmt;
use std::simd::{
    cmp::{SimdOrd, SimdPartialOrd},
    i32x4, i32x8, i64x8, mask64x8,
    num::SimdInt,
    simd_swizzle, u32x8, u32x16, u64x8,
};
use std::time::Instant;

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
        assert!((-63..=-60).contains(&y));

        let block_key = hash_three(x, y, z);
        let mut block_prng = Xoroshiro::new(block_key ^ self.floor_seed_low, self.floor_seed_high);

        // equivalent to
        // let boundary = map_via_lerp(y as f64, -64.0, -59.0, 1.0, 0.0);
        // (block_prng.next() >> 40) as f64 * 2.0f64.powi(-24) < boundary

        static BOUNDARIES: [u64; 4] =
            [13421773 << 40, 10066330 << 40, 6710887 << 40, 3355444 << 40];
        block_prng.next() < BOUNDARIES[(y + 63) as usize]
    }

    fn is_bedrock_vec(&self, x: i32x8, y: i32, z: i32x8) -> mask64x8 {
        assert!((-63..=-60).contains(&y));

        let block_key = hash_three_vec(x, y, z);

        let low = block_key ^ u64x8::splat(self.floor_seed_low);
        let high = u64x8::splat(self.floor_seed_high);

        assert_ne!(
            self.floor_seed_high, 0,
            "Cannot handle this rare case soundly"
        );

        let sum = low + high;
        let block_prng_next = ((sum << 17) | (sum >> 47)) + low;

        static BOUNDARIES: [u64; 4] =
            [13421773 << 40, 10066330 << 40, 6710887 << 40, 3355444 << 40];
        block_prng_next.simd_lt(u64x8::splat(BOUNDARIES[(y + 63) as usize]))
    }

    fn is_interior(&self, (x, z): (i32, i32)) -> bool {
        !self.is_bedrock(x, -63, z) && !self.is_bedrock(x, -62, z)
    }

    fn is_interior_vec(&self, x: i32x8, z: i32x8) -> mask64x8 {
        !self.is_bedrock_vec(x, -63, z) & !self.is_bedrock_vec(x, -62, z)
    }

    fn get_column_type(&self, (x, z): (i32, i32)) -> ColumnType {
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

#[derive(Clone, Copy, Debug)]
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
    let mut l = (x.wrapping_mul(3129871) as i64) ^ (y as i64) ^ ((z as i64) * 116129781);
    l = l.wrapping_mul(l.wrapping_mul(42317861).wrapping_add(11));
    (l >> 16) as u64
}

fn hash_three_vec(x: i32x8, y: i32, z: i32x8) -> u64x8 {
    let mut l = (x * i32x8::splat(3129871)).cast::<i64>()
        ^ i64x8::splat(y as i64)
        ^ (z.cast::<i64>() * i64x8::splat(116129781));
    l = l * (l * i64x8::splat(42317861) + i64x8::splat(11));
    (l >> 16).cast::<u64>()
}

const WORLD_BORDER: i32 = 29_999_984;
const SEARCH_RADIUS: i32 = 4;

fn enumerate_diagonals(noise: &BedrockFloorNoise, mut callback: impl FnMut((i32, i32))) {
    let mut total: u64 = 0;
    for i in (-2 * WORLD_BORDER..=2 * WORLD_BORDER).step_by(2 * SEARCH_RADIUS as usize) {
        total += (2 * WORLD_BORDER - i.abs()) as u64;
    }

    let mut current = 0;
    for i in (-2 * WORLD_BORDER..=2 * WORLD_BORDER)
        .step_by(2 * SEARCH_RADIUS as usize)
        .take(10000)
    {
        // Main diagonals
        let mut j_min = -WORLD_BORDER - i.min(0);
        let j_max = WORLD_BORDER - i.max(0);

        current += (j_max - j_min) as u64;
        if i % 2003 == 0 {
            println!("{}% checked", current as f32 / total as f32 * 100.0);
        }

        while j_min + 7 <= j_max {
            let x = i32x8::splat(j_min) + i32x8::from([0, 1, 2, 3, 4, 5, 6, 7]);
            let z = i32x8::splat(i) + x;
            let mut bitmask = noise.is_interior_vec(x, z).to_bitmask();
            while bitmask != 0 {
                let dj = bitmask.trailing_zeros() as i32;
                callback((j_min + dj, i + j_min + dj));
                bitmask &= bitmask - 1;
            }
            j_min += 8;
        }
        while j_min <= j_max {
            let coords0 = (j_min, i + j_min);
            if noise.is_interior(coords0) {
                callback(coords0);
            }
            j_min += 1;
        }

        // Anti-diagonals
        let j_min = -WORLD_BORDER + i.max(0);
        let j_max = WORLD_BORDER + i.min(0);
        for j_rem in 0..SEARCH_RADIUS {
            if (j_min + j_rem) % SEARCH_RADIUS == 0 {
                continue;
            }
            let mut j_min = j_min + j_rem;
            while j_min + 7 * SEARCH_RADIUS <= j_max {
                let x = i32x8::splat(j_min)
                    + i32x8::from([0, 1, 2, 3, 4, 5, 6, 7].map(|dj| dj * SEARCH_RADIUS));
                let z = i32x8::splat(i) - x;
                let mut bitmask = noise.is_interior_vec(x, z).to_bitmask();
                while bitmask != 0 {
                    let dj = bitmask.trailing_zeros() as i32 * SEARCH_RADIUS;
                    callback((j_min + dj, i - j_min - dj));
                    bitmask &= bitmask - 1;
                }
                j_min += 8 * SEARCH_RADIUS;
            }
            while j_min <= j_max {
                let coords0 = (j_min, i - j_min);
                if noise.is_interior(coords0) {
                    callback(coords0);
                }
                j_min += SEARCH_RADIUS;
            }
        }
    }
}

struct CoordSet {
    values: [u32; 32],
}

impl CoordSet {
    fn new() -> Self {
        Self { values: [0; 32] }
    }

    fn insert(&mut self, (x, z): (i32, i32)) -> bool {
        let row = &mut self.values[z as usize & 31];
        let bit = x as usize & 31;
        let ret = (*row >> bit) & 1 == 0;
        *row |= 1 << bit;
        ret
    }

    #[inline(always)]
    fn has_empty_row(&self) -> bool {
        self.values.contains(&0)
    }

    fn has_empty_column(&self) -> bool {
        // LLVM can autovectorize this
        //     self.values.iter().copied().fold(0, |a, b| a | b) != u32::MAX
        // just fine, but when combined with a call to has_empty_row(), it leads to an abomination
        // for no good reason. Vectorize this manually.
        let v = u32x16::from_slice(&self.values[..16]) | u32x16::from_slice(&self.values[16..]);
        let v = simd_swizzle!(v, [0, 1, 2, 3, 4, 5, 6, 7])
            | simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15]);
        let v = simd_swizzle!(v, [0, 1, 2, 3]) | simd_swizzle!(v, [4, 5, 6, 7]);
        let v = simd_swizzle!(v, [0, 1]) | simd_swizzle!(v, [2, 3]);
        (v[0] | v[1]) != u32::MAX
    }
}

struct ComponentWalker {
    large_stack_x: [i32; 8 + 32 * 32],
    large_stack_z: [i32; 8 + 32 * 32],
}

impl ComponentWalker {
    fn new() -> Self {
        Self {
            // First 8 elements need to be outside world border
            large_stack_x: [i32::MIN; 8 + 32 * 32],
            large_stack_z: [0; 8 + 32 * 32],
        }
    }

    #[inline(always)]
    fn get_component_size_ignoring_hazards(
        &mut self,
        noise: &BedrockFloorNoise,
        (x0, z0): (i32, i32),
    ) -> Option<usize> {
        let mut visited = CoordSet::new();
        visited.insert((x0, z0));

        let mut stack_size = 0;

        let neigh_x = i32x4::splat(x0) + i32x4::from([-1, 1, 0, 0]);
        let neigh_z = i32x4::splat(z0) + i32x4::from([0, 0, -1, 1]);
        neigh_x.copy_to_slice(&mut self.large_stack_x[8..]);
        neigh_z.copy_to_slice(&mut self.large_stack_z[8..]);

        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            visited.insert((x0 + dx, z0 + dz));
        }

        let mut x = neigh_x.resize(i32::MIN);
        let mut z = neigh_z.resize(0);

        let mut interior_count = 1;

        loop {
            let in_world_mask = ((x + i32x8::splat(WORLD_BORDER))
                .cast::<u32>()
                .simd_max((z + i32x8::splat(WORLD_BORDER)).cast::<u32>())
                .simd_le(u32x8::splat(2 * WORLD_BORDER as u32)))
            .cast();

            let interior_mask = noise.is_interior_vec(x, z);

            let mut interior_bitmask = (interior_mask & in_world_mask).to_bitmask();
            while interior_bitmask != 0 {
                let index = interior_bitmask.trailing_zeros() as usize;
                let (x, z) = (x[index], z[index]);
                interior_bitmask &= interior_bitmask - 1;

                interior_count += 1;

                for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (x1, z1) = (x + dx, z + dz);
                    if visited.insert((x1, z1)) {
                        self.large_stack_x[8 + stack_size] = x1;
                        self.large_stack_z[8 + stack_size] = z1;
                        stack_size += 1;
                    }
                }
            }

            if stack_size == 0 {
                break;
            }

            x = i32x8::from_slice(&self.large_stack_x[stack_size..]);
            z = i32x8::from_slice(&self.large_stack_z[stack_size..]);
            stack_size = stack_size.max(8) - 8;
        }

        // `visited` loops at 32, so we want to ensure we haven't accidentally mistaken a far away
        // block for being visited. The simplest way to ensure this hasn't happened is to check that
        // there's an empty row or column, which naturally acts as a barrier of propagation; if no
        // such barriers exists, we enter the sad path. This hasn't ever been triggered yet, so I
        // haven't implemented it, but it would require switching `visited` to a `HashSet`. We run
        // the assert only if the interior count is large enough for performance reasons.
        if interior_count >= 16 {
            assert!(
                visited.has_empty_row() && visited.has_empty_column(),
                "Possibly lost some blocks"
            );
        }

        Some(interior_count)
    }

    #[inline(always)]
    fn component_borders_hazards(
        &mut self,
        noise: &BedrockFloorNoise,
        (x0, z0): (i32, i32),
    ) -> bool {
        let mut visited = CoordSet::new();
        visited.insert((x0, z0));

        let mut stack_size = 1;
        self.large_stack_x[8] = x0;
        self.large_stack_z[8] = z0;

        while stack_size > 0 {
            stack_size -= 1;
            let x = self.large_stack_x[8 + stack_size];
            let z = self.large_stack_z[8 + stack_size];

            let ty = if x.abs() <= WORLD_BORDER && z.abs() <= WORLD_BORDER {
                noise.get_column_type((x, z))
            } else {
                ColumnType::Wall
            };
            match ty {
                ColumnType::Interior => {}
                ColumnType::Wall => continue,
                ColumnType::Hazard => return true,
            }

            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (x1, z1) = (x + dx, z + dz);
                // No need for bounds check on visited because `get_component_size_ignoring_hazards`
                // has already performed it.
                if visited.insert((x1, z1)) {
                    self.large_stack_x[8 + stack_size] = x1;
                    self.large_stack_z[8 + stack_size] = z1;
                    stack_size += 1;
                }
            }
        }

        false
    }
}

fn main() {
    let noise = BedrockFloorNoise::from_world_seed(-972064012444369952i64 as u64);

    let mut best_coords = (0, 0);
    let mut best_size = 0;

    // for z in -29963411 - 10..-29963411 + 10 {
    //     for x in -29999605 - 10..-29999605 + 10 {
    //         print!("{}", noise.get_column_type((x, z)));
    //     }
    //     println!();
    // }

    let start_instant = Instant::now();

    println!(
        "Searching for components >= {} (might find smaller ones as well, but not all of them)",
        (SEARCH_RADIUS - 1) * SEARCH_RADIUS * 2 + 2
    );

    let mut walker = ComponentWalker::new();

    enumerate_diagonals(
        &noise,
        #[inline(always)]
        |coords0| {
            let Some(size) = walker.get_component_size_ignoring_hazards(&noise, coords0) else {
                return;
            };
            if size <= best_size || walker.component_borders_hazards(&noise, coords0) {
                return;
            }

            println!(
                "[{:?}] found {} at {:?}",
                start_instant.elapsed(),
                size,
                coords0,
            );

            best_coords = coords0;
            best_size = size;
        },
    );
}
