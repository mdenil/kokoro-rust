//! Weight store: raw tensors keyed by the reference module's state_dict names.
//! Weight-normalized convolutions are resolved here (hydrate-once), from weight_g/weight_v.

use crate::ops;
use crate::st::{self, TensorMap};
use anyhow::{bail, Context, Result};
use std::path::Path;

pub struct Weights {
    map: TensorMap,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self { map: st::load(path)? })
    }

    /// Load directly from the upstream PyTorch checkpoint (kokoro-v1_0.pth), no conversion step.
    pub fn load_pth(path: &Path) -> Result<Self> {
        let pt = crate::torchpt::load_kmodel_checkpoint(path)?;
        let map = pt
            .into_iter()
            .map(|(k, t)| (k, st::Tensor { shape: t.shape, data: st::Data::F32(t.data) }))
            .collect();
        Ok(Self { map })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn raw(&self, name: &str) -> Option<&st::Tensor> {
        self.map.get(name)
    }

    pub fn from_map(map: TensorMap) -> Self {
        Self { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.map.get(name).with_context(|| format!("missing tensor {name}"))?.shape)
    }

    /// Fetch an f32 tensor, asserting its exact shape.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.map.get(name).with_context(|| format!("missing tensor {name}"))?;
        if t.shape != shape {
            bail!("tensor {name}: shape {:?}, expected {:?}", t.shape, shape);
        }
        Ok(t.f32()?.to_vec())
    }

    /// Weight for `prefix`: `prefix.weight` if present, else weight_norm(weight_v, weight_g).
    pub fn conv_weight(&self, prefix: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let plain = format!("{prefix}.weight");
        if self.map.contains_key(&plain) {
            return self.get(&plain, shape);
        }
        let v = self.get(&format!("{prefix}.weight_v"), shape)?;
        let g = self.get(&format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
        Ok(ops::weight_norm(&v, &g))
    }

    /// AdaIN1d's InstanceNorm1d(affine=True) parameters are absent from kokoro-v1_0.pth (the
    /// upstream enables affine only as an ONNX-export workaround); the reference's non-strict
    /// load leaves them at torch's deterministic init (weight 1, bias 0). Only these names may
    /// default; every other missing tensor stays a hard error.
    pub fn get_instance_norm_affine(&self, name: &str, c: usize, init: f32) -> Result<Vec<f32>> {
        if self.map.contains_key(name) {
            return self.get(name, &[c]);
        }
        if !(name.ends_with(".norm.weight") || name.ends_with(".norm.bias")) {
            bail!("missing tensor {name}");
        }
        Ok(vec![init; c])
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}
