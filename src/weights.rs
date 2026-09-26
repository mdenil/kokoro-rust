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

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}
