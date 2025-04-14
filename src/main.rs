use std::time::Instant;
use core::fmt;

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

    fn is_interior(&self, (x, z): (i32, i32)) -> bool {
        !self.is_bedrock(x, -63, z) && !self.is_bedrock(x, -62, z)
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
const SEARCH_RADIUS: i32 = 4;

fn enumerate_diagonals(mut callback: impl FnMut((i32, i32))) {
    for z0 in (-WORLD_BORDER..=-WORLD_BORDER + 100).step_by(SEARCH_RADIUS as usize * 2) {
        for (x, dz) in (-WORLD_BORDER..=WORLD_BORDER).zip((0..=SEARCH_RADIUS).chain((0..SEARCH_RADIUS).rev()).cycle()) {
            if dz > 0 {
                callback((x, z0 + dz - 1));
            }
            if dz != SEARCH_RADIUS {
                callback((x, z0 + 2 * SEARCH_RADIUS - 1 - dz));
            }
        }
    }
}

struct CoordSet {
    values: [u32; 32],
}

impl CoordSet {
    fn new() -> Self {
        Self {
            values: [0; 32],
        }
    }

    fn insert(&mut self, (x, z): (i32, i32)) -> bool {
        let row = &mut self.values[z as usize & 31];
        let bit = x as usize & 31;
        let ret = (*row >> bit) & 1 == 0;
        *row |= 1 << bit;
        ret
    }
}

fn main() {
    let noise = BedrockFloorNoise::from_world_seed(-972064012444369952i64 as u64);

    let mut best_coords = (0, 0);
    let mut best_size = 0;

    // for z in -29993432 - 10..-29993432 + 10 {
    //     for x in -20877857 - 10..-20877857 + 10 {
    //         print!("{}", noise.get_column_type(x, z));
    //     }
    //     println!();
    // }

    let start_instant = Instant::now();
    let mut queue = [(0, 0); 32 * 32];

    enumerate_diagonals(#[inline(always)] |coords0| {
        if !noise.is_interior(coords0) {
            return;
        }

        let (x0, z0) = coords0;

        let mut visited = CoordSet::new();
        let mut ptr = 1;
        visited.insert(coords0);
        queue[0] = coords0;

        let mut interior_count = 0;

        while ptr > 0 {
            ptr -= 1;
            let (x, z) = queue[ptr];

            interior_count += 1;

            // We want all neighbours to be at most 15 blocks away from start so that the whole
            // component spans at most 31 blocks. Subtract 1 from the boundaries because we're
            // checking (x, z), not the neighbours themselves.
            assert!((x0 - x + 14) as u32 <= 28 && (z0 - z + 14) as u32 <= 28, "Out of bounds");

            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (x1, z1) = (x + dx, z + dz);
                let coords1 = (x1, z1);

                if !visited.insert(coords1) {
                    continue;
                }

                let ty1 = if (-WORLD_BORDER..=WORLD_BORDER).contains(&x1)
                    && (-WORLD_BORDER..=WORLD_BORDER).contains(&z1)
                {
                    noise.get_column_type(coords1)
                } else {
                    ColumnType::Wall
                };
                match ty1 {
                    ColumnType::Interior => {}
                    ColumnType::Wall => continue,
                    ColumnType::Hazard => return,
                }

                queue[ptr] = coords1;
                ptr += 1;
            }
        }

        if interior_count > best_size {
            println!(
                "[{:?}] found {} at {:?}",
                start_instant.elapsed(), interior_count, coords0,
            );

            best_coords = coords0;
            best_size = interior_count;
        }
    });
}
