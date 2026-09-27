//! The fast SHA-256 (ring) must equal the reference (sha2 crate) byte for byte: identities in every
//! sidecar/manifest/resume key depend on it.
use kokoro::engine::{sha256_bytes, sha256_bytes_sha2, sha256_file};

#[path = "support/paths.rs"]
mod paths;

#[test]
fn ring_sha256_equals_sha2_and_known_file_hash() {
    let mut x = 0x9e3779b97f4a7c15u64;
    let data: Vec<u8> = (0..3_000_000).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x as u8
    }).collect();
    for n in [0usize, 1, 55, 56, 63, 64, 65, 127, 128, 1000, 4096, 65537, 1 << 20, 3_000_000] {
        assert_eq!(sha256_bytes(&data[..n]), sha256_bytes_sha2(&data[..n]), "len {n}");
    }
    assert_eq!(sha256_bytes(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let w = paths::model_dir().join("kokoro-v1_0.pth");
    assert_eq!(sha256_file(&w).unwrap(), "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4");
}
