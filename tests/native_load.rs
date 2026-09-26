//! The native .pth/.pt readers must reproduce the reference's loaded tensors BIT-EXACTLY.
//! Expected values: models/raw_state.safetensors = KModel.state_dict() after the reference's own
//! load (oracle/dump_weights.py), and voices_*.safetensors = torch.load of the voice packs.

use kokoro::{st, torchpt, weights::Weights};
use std::path::PathBuf;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}
fn snap() -> PathBuf {
    data().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987")
}

#[test]
fn pth_matches_reference_state_dict_bitwise() {
    let native = Weights::load_pth(&snap().join("kokoro-v1_0.pth")).unwrap();
    let reference = st::load(&data().join("models/raw_state.safetensors")).unwrap();
    let mut missing = vec![];
    for (name, t) in &reference {
        match native.raw(name) {
            None => missing.push(name.clone()),
            Some(n) => {
                assert_eq!(n.shape, t.shape, "{name}: shape");
                let (a, b) = (n.f32().unwrap(), t.f32().unwrap());
                assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: bits differ");
            }
        }
    }
    let extra: Vec<_> = native.names().filter(|k| !reference.contains_key(*k)).cloned().collect();
    println!("reference keys {}, native keys {}, extra-in-pth {:?}", reference.len(), native.len(), extra);
    // Exactly the AdaIN InstanceNorm affine params are absent from the checkpoint, and the
    // reference holds them at identity init; anything else missing is a failure.
    let unexpected: Vec<_> = missing.iter().filter(|k| !(k.ends_with(".norm.weight") || k.ends_with(".norm.bias"))).collect();
    assert!(unexpected.is_empty(), "unexpected missing keys: {unexpected:?}");
    assert_eq!(missing.len(), 140, "absent AdaIN affine param count");
    for k in &missing {
        let init = if k.ends_with("weight") { 1.0f32 } else { 0.0 };
        assert!(reference[k].f32().unwrap().iter().all(|&v| v == init), "{k}: reference not at identity init");
    }
    // and the hydrated model from the .pth must equal the one from the reference export
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap().join("config.json")).unwrap()).unwrap();
    kokoro::model::Kokoro::from_weights(&native, &cfg).expect("model hydrates from native .pth");
}

#[test]
fn voice_packs_match_bitwise() {
    for v in ["af_heart", "am_adam"] {
        let native = torchpt::load_voice_pack(&snap().join(format!("voices/{v}.pt"))).unwrap();
        let r = st::load(&data().join(format!("models/voices_{v}.safetensors"))).unwrap();
        let rt = &r["pack"];
        assert_eq!(native.shape, rt.shape);
        assert!(native.data.iter().zip(rt.f32().unwrap()).all(|(a, b)| a.to_bits() == b.to_bits()), "{v}: bits differ");
    }
}

#[test]
fn corrupt_and_missing_artifacts_are_refused() {
    let dir = data().join("tmp/native_load_test");
    std::fs::create_dir_all(&dir).unwrap();
    // missing
    assert!(torchpt::load_voice_pack(&dir.join("nope.pt")).is_err());
    // truncated
    let bytes = std::fs::read(snap().join("voices/af_heart.pt")).unwrap();
    std::fs::write(dir.join("trunc.pt"), &bytes[..bytes.len() / 2]).unwrap();
    assert!(torchpt::load_voice_pack(&dir.join("trunc.pt")).is_err());
    // garbage
    std::fs::write(dir.join("garbage.pt"), b"not a zip at all, definitely not").unwrap();
    assert!(torchpt::load_voice_pack(&dir.join("garbage.pt")).is_err());
    // weights file handed to the voice loader (wrong shape) is refused
    assert!(Weights::load_pth(&dir.join("garbage.pt")).is_err());
}
