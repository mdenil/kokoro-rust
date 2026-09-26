//! Minimal safetensors reader: u64-LE header length, JSON directory, raw LE payload.
//! Only the dtypes this project uses (F32, I64) are accepted; anything else is refused.

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn f32(&self) -> Result<&[f32]> {
        match &self.data {
            Data::F32(v) => Ok(v),
            _ => bail!("tensor is not f32"),
        }
    }

    pub fn i64(&self) -> Result<&[i64]> {
        match &self.data {
            Data::I64(v) => Ok(v),
            _ => bail!("tensor is not i64"),
        }
    }
}

pub type TensorMap = BTreeMap<String, Tensor>;

pub fn load(path: &Path) -> Result<TensorMap> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parsing safetensors {}", path.display()))
}

pub fn parse(bytes: &[u8]) -> Result<TensorMap> {
    ensure!(bytes.len() >= 8, "file too short for header length");
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    ensure!(8 + hlen <= bytes.len(), "header length {hlen} exceeds file size");
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])?;
    let payload = &bytes[8 + hlen..];
    let obj = header.as_object().context("header is not a JSON object")?;

    let mut out = TensorMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().context("missing dtype")?;
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .context("missing shape")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize).context("bad shape entry"))
            .collect::<Result<_>>()?;
        let offs = info["data_offsets"].as_array().context("missing data_offsets")?;
        ensure!(offs.len() == 2, "bad data_offsets for {name}");
        let (s, e) = (
            offs[0].as_u64().context("bad offset")? as usize,
            offs[1].as_u64().context("bad offset")? as usize,
        );
        ensure!(s <= e && e <= payload.len(), "offsets out of range for {name}");
        let raw = &payload[s..e];
        let numel: usize = shape.iter().product();
        let data = match dtype {
            "F32" => {
                ensure!(raw.len() == numel * 4, "size mismatch for {name}");
                Data::F32(
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            }
            "I64" => {
                ensure!(raw.len() == numel * 8, "size mismatch for {name}");
                Data::I64(
                    raw.chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            }
            other => bail!("unsupported dtype {other} for {name}"),
        };
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}
