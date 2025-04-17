# Bedrock prison searcher

This is motivated by [Bamboo Bot's video](https://www.youtube.com/watch?v=m85D_RKJWUQ) on making Minecraft prisons out of bedrock blocks which are naturally generated on the bedrock floor. The floor is generated randomly, and certain configurations are locally inescapable (e.g. air surrounded by two block high bedrock walls).

I am interested in finding *large* naturally generated prisons. [The original bedrock generator](https://github.com/Developer-Mike/minecraft-bedrock-generator) is written in Java and is unsuitable for brute-force, so the prison in the original video is only about 2000 blocks away from spawn. This project is written in Rust, uses AVX-512 to vectorize the bedrock floor computation, DFS-style flood fill, and performs a rough first scan to filter out blocks that don't have a chance to be part of a large connected component.

You'll need a Zen 4 processor for the brute-force to be efficient. It takes 1 month to scan a world seed (within world borders) in single-thread. Multi-thread support is not implemented, but is easy to add if necessary; the algorithm scales just fine. No checkpoint mechanism is implemented, but that is also easy to add if necessary.

I haven't completed the full search for any seed yet; the largest prison I've found spans 25 blocks and is located at `28939509 -62 -28625251` on seed `-972064012444369952`. In its current state, the program only searches for prisons of size 26 and larger; I haven't encountered any such prisons yet.

To compile the code, change the world seed in `main.rs` and run `RUSTFLAGS="-C target-cpu=znver4" cargo build --release`.

The code contains two assertions which have a small, but non-zero chance of firing:

- "Cannot handle this rare case soundly": for estimated ~1 world seed, the random generator Minecraft uses behaves slightly unusually and is a bit slower to simulate. While handling this case is easy, it slightly reduces performance, so I haven't implemented it, but you can do that yourself if you find a world seed which triggers this assertion. I have been unable to find such a seed; that can easily be another brute-force topic.

- "Possibly lost some blocks": This triggers for very large and narrow components. I have not encountered such components in the world seed I've tested the program on. Again, this can be handled with a fallback path, but I haven't seen the need yet.

While the code is currently tuned to working performantly with AVX-512 on Zen 4, earlier commits contain alternative implementations that work better without AVX-512 or on Intel processors: feel free to experiment to find a better configuration.

You should have already realized I don't this project very seriously by the ugly UX and unimplemented features. Feel free to build on this; I believe the CPU implementation to be near optimal, but switching to GPU might provide some benefits, I just don't have the necessary skills.

The code in this repository is published under the MIT license.
