//! Native reader for PyTorch zip checkpoints (`torch.save`, new zipfile format) holding f32
//! tensors: kokoro-v1_0.pth (dict of state dicts) and voices/*.pt (a single tensor).
//!
//! Deliberately restricted: STORED (uncompressed) zip entries only, a small pickle opcode set,
//! and exactly the globals `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and
//! `torch.FloatStorage`. Anything else is refused rather than guessed — pickle is never
//! executed, only interpreted.

use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

struct Zip<'a> {
    bytes: &'a [u8],
    entries: HashMap<String, (usize, usize)>, // name -> (data offset, size)
}

fn u16le(b: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(b.get(o..o + 2).context("zip truncated")?.try_into().unwrap()))
}
fn u32le(b: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(o..o + 4).context("zip truncated")?.try_into().unwrap()))
}

impl<'a> Zip<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        ensure!(bytes.len() >= 22, "not a zip file");
        let lo = bytes.len().saturating_sub(22 + 65535);
        let eocd = (lo..=bytes.len() - 22)
            .rev()
            .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .context("zip end-of-central-directory not found")?;
        let n = u16le(bytes, eocd + 10)? as usize;
        let cd_off = u32le(bytes, eocd + 16)? as usize;
        ensure!(cd_off != 0xFFFF_FFFF, "zip64 archives are not supported");
        let mut entries = HashMap::new();
        let mut p = cd_off;
        for _ in 0..n {
            ensure!(u32le(bytes, p)? == 0x0201_4b50, "bad central directory entry");
            let method = u16le(bytes, p + 10)?;
            let csize = u32le(bytes, p + 20)? as usize;
            let usize_ = u32le(bytes, p + 24)? as usize;
            let nlen = u16le(bytes, p + 28)? as usize;
            let xlen = u16le(bytes, p + 30)? as usize;
            let clen = u16le(bytes, p + 32)? as usize;
            let lho = u32le(bytes, p + 42)? as usize;
            let name = std::str::from_utf8(bytes.get(p + 46..p + 46 + nlen).context("zip truncated")?)?.to_string();
            ensure!(method == 0, "zip entry {name} is compressed (method {method}); only STORED is supported");
            ensure!(csize == usize_ && csize != 0xFFFF_FFFF, "zip entry {name}: unsupported sizes");
            ensure!(u32le(bytes, lho)? == 0x0403_4b50, "bad local header for {name}");
            let data = lho + 30 + u16le(bytes, lho + 26)? as usize + u16le(bytes, lho + 28)? as usize;
            ensure!(data + csize <= bytes.len(), "zip entry {name} out of range");
            entries.insert(name, (data, csize));
            p += 46 + nlen + xlen + clen;
        }
        Ok(Self { bytes, entries })
    }

    fn get(&self, name: &str) -> Result<&'a [u8]> {
        let &(o, n) = self.entries.get(name).with_context(|| format!("zip entry {name} missing"))?;
        Ok(&self.bytes[o..o + n])
    }
}

#[derive(Clone, Debug)]
enum V {
    None,
    Bool,
    Int(i64),
    Str(String),
    Tuple(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    Storage(String),
    Tensor { key: String, offset: usize, shape: Vec<usize>, stride: Vec<usize> },
    Mark,
}

fn as_usize_vec(v: &V) -> Result<Vec<usize>> {
    match v {
        V::Tuple(xs) => xs
            .iter()
            .map(|x| match x {
                V::Int(i) if *i >= 0 => Ok(*i as usize),
                _ => bail!("expected non-negative int"),
            })
            .collect(),
        _ => bail!("expected tuple of ints"),
    }
}

fn unpickle(data: &[u8]) -> Result<V> {
    let mut stack: Vec<V> = vec![];
    let mut memo: HashMap<u32, V> = HashMap::new();
    let mut i = 0usize;
    let rd = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = data.get(*i..*i + n).context("pickle truncated")?;
        *i += n;
        Ok(s)
    };
    let pop_mark = |stack: &mut Vec<V>| -> Result<Vec<V>> {
        let pos = stack.iter().rposition(|v| matches!(v, V::Mark)).context("MARK not found")?;
        let items = stack.split_off(pos + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = *data.get(i).context("pickle ended without STOP")?;
        i += 1;
        match op {
            0x80 => {
                rd(&mut i, 1)?; // PROTO
            }
            b'}' => stack.push(V::Dict(vec![])),
            b')' => stack.push(V::Tuple(vec![])),
            b']' => stack.push(V::Tuple(vec![])), // EMPTY_LIST (treated as a sequence)
            b'N' => stack.push(V::None),
            0x88 => stack.push(V::Bool),
            0x89 => stack.push(V::Bool),
            b'(' => stack.push(V::Mark),
            b'X' => {
                let n = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as usize;
                stack.push(V::Str(std::str::from_utf8(rd(&mut i, n)?)?.to_string()));
            }
            b'J' => stack.push(V::Int(i32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap()) as i64)),
            b'K' => stack.push(V::Int(rd(&mut i, 1)?[0] as i64)),
            b'M' => stack.push(V::Int(u16::from_le_bytes(rd(&mut i, 2)?.try_into().unwrap()) as i64)),
            b'q' => {
                let k = rd(&mut i, 1)?[0] as u32;
                memo.insert(k, stack.last().context("BINPUT on empty stack")?.clone());
            }
            b'r' => {
                let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap());
                memo.insert(k, stack.last().context("LONG_BINPUT on empty stack")?.clone());
            }
            b'h' => {
                let k = rd(&mut i, 1)?[0] as u32;
                stack.push(memo.get(&k).context("BINGET miss")?.clone());
            }
            b'j' => {
                let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into().unwrap());
                stack.push(memo.get(&k).context("LONG_BINGET miss")?.clone());
            }
            b'c' => {
                let rest = &data[i..];
                let a = rest.iter().position(|&b| b == b'\n').context("bad GLOBAL")?;
                let b = rest[a + 1..].iter().position(|&b| b == b'\n').context("bad GLOBAL")?;
                let module = std::str::from_utf8(&rest[..a])?;
                let name = std::str::from_utf8(&rest[a + 1..a + 1 + b])?;
                i += a + b + 2;
                let g = format!("{module}.{name}");
                ensure!(
                    matches!(g.as_str(), "collections.OrderedDict" | "torch._utils._rebuild_tensor_v2" | "torch.FloatStorage"),
                    "refusing unsupported pickle global {g}"
                );
                stack.push(V::Global(g));
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(V::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                ensure!(stack.len() >= n, "TUPLE{n} underflow");
                let items = stack.split_off(stack.len() - n);
                stack.push(V::Tuple(items));
            }
            b'Q' => {
                let pid = stack.pop().context("BINPERSID underflow")?;
                match pid {
                    V::Tuple(t) if t.len() == 5 => {
                        match (&t[0], &t[1], &t[2]) {
                            (V::Str(kind), V::Global(g), V::Str(key)) if kind == "storage" && g == "torch.FloatStorage" => {
                                stack.push(V::Storage(key.clone()))
                            }
                            _ => bail!("unsupported persistent id {t:?}"),
                        }
                    }
                    other => bail!("unsupported persistent id {other:?}"),
                }
            }
            b'R' => {
                let args = stack.pop().context("REDUCE underflow")?;
                let f = stack.pop().context("REDUCE underflow")?;
                match (f, args) {
                    (V::Global(g), V::Tuple(_)) if g == "collections.OrderedDict" => stack.push(V::Dict(vec![])),
                    (V::Global(g), V::Tuple(a)) if g == "torch._utils._rebuild_tensor_v2" => {
                        ensure!(a.len() >= 4, "_rebuild_tensor_v2 args");
                        let key = match &a[0] {
                            V::Storage(k) => k.clone(),
                            _ => bail!("tensor without storage"),
                        };
                        let offset = match &a[1] {
                            V::Int(o) if *o >= 0 => *o as usize,
                            _ => bail!("bad storage offset"),
                        };
                        stack.push(V::Tensor { key, offset, shape: as_usize_vec(&a[2])?, stride: as_usize_vec(&a[3])? });
                    }
                    (f, _) => bail!("unsupported REDUCE of {f:?}"),
                }
            }
            b'b' => {
                stack.pop().context("BUILD underflow")?; // state (e.g. OrderedDict._metadata): ignored
            }
            b's' => {
                let v = stack.pop().context("SETITEM underflow")?;
                let k = stack.pop().context("SETITEM underflow")?;
                match stack.last_mut() {
                    Some(V::Dict(d)) => d.push((k, v)),
                    _ => bail!("SETITEM on non-dict"),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                ensure!(items.len() % 2 == 0, "SETITEMS odd count");
                match stack.last_mut() {
                    Some(V::Dict(d)) => {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                    _ => bail!("SETITEMS on non-dict"),
                }
            }
            b'.' => return stack.pop().context("empty stack at STOP"),
            other => bail!("unsupported pickle opcode 0x{other:02x} at {}", i - 1),
        }
    }
}

/// A loaded f32 tensor (contiguous row-major).
#[derive(Clone, Debug)]
pub struct PtTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

struct Loader<'a> {
    zip: Zip<'a>,
    prefix: String,
    storages: HashMap<String, Vec<f32>>,
}

impl<'a> Loader<'a> {
    fn new(bytes: &'a [u8]) -> Result<(Self, V)> {
        let zip = Zip::parse(bytes)?;
        let pkl = zip.entries.keys().find(|k| k.ends_with("/data.pkl")).context("no data.pkl in checkpoint")?.clone();
        let prefix = pkl.trim_end_matches("data.pkl").to_string();
        if let Ok(bo) = zip.get(&format!("{prefix}byteorder")) {
            ensure!(bo == b"little", "unsupported byteorder {:?}", String::from_utf8_lossy(bo));
        }
        let root = unpickle(zip.get(&pkl)?)?;
        Ok((Self { zip, prefix, storages: HashMap::new() }, root))
    }

    fn tensor(&mut self, v: &V) -> Result<PtTensor> {
        let V::Tensor { key, offset, shape, stride } = v else { bail!("expected tensor, got {v:?}") };
        let numel: usize = shape.iter().product();
        // Fast path: a row-major contiguous view is a plain byte range of its storage (same values
        // as the generic strided walk below, without the per-element index arithmetic).
        let mut contiguous = true;
        let mut expect = 1usize;
        for d in (0..shape.len()).rev() {
            if shape[d] != 1 && stride[d] != expect {
                contiguous = false;
            }
            expect *= shape[d];
        }
        if contiguous && !self.storages.contains_key(key) {
            let raw = self.zip.get(&format!("{}data/{key}", self.prefix))?;
            ensure!(raw.len() % 4 == 0, "storage {key} size not a multiple of 4");
            let bytes = raw.get(4 * offset..4 * (offset + numel)).with_context(|| format!("tensor view exceeds storage {key}"))?;
            let data = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            return Ok(PtTensor { shape: shape.clone(), data });
        }
        if !self.storages.contains_key(key) {
            let raw = self.zip.get(&format!("{}data/{key}", self.prefix))?;
            ensure!(raw.len() % 4 == 0, "storage {key} size not a multiple of 4");
            let vals = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            self.storages.insert(key.clone(), vals);
        }
        let st = &self.storages[key];
        let mut out = Vec::with_capacity(numel);
        let nd = shape.len();
        let mut idx = vec![0usize; nd];
        for _ in 0..numel {
            let off = offset + idx.iter().zip(stride).map(|(i, s)| i * s).sum::<usize>();
            out.push(*st.get(off).with_context(|| format!("tensor view exceeds storage {key}"))?);
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(PtTensor { shape: shape.clone(), data: out })
    }
}

fn key_str(v: &V) -> Result<&str> {
    match v {
        V::Str(s) => Ok(s),
        other => bail!("non-string dict key {other:?}"),
    }
}

/// Load kokoro-v1_0.pth: {submodule: state_dict} -> flat "submodule.param" map. A leading
/// "module." inside a sub-state-dict is stripped, as KModel.__init__ does on its fallback path.
pub fn load_kmodel_checkpoint(path: &Path) -> Result<BTreeMap<String, PtTensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (mut ld, root) = Loader::new(&bytes)?;
    let V::Dict(top) = root else { bail!("checkpoint root is not a dict") };
    let mut out = BTreeMap::new();
    for (k, sd) in &top {
        let sub = key_str(k)?;
        let V::Dict(items) = sd else { bail!("{sub} is not a state dict") };
        for (pk, t) in items {
            let name = key_str(pk)?;
            let name = name.strip_prefix("module.").unwrap_or(name);
            out.insert(format!("{sub}.{name}"), ld.tensor(t)?);
        }
    }
    Ok(out)
}

/// Load a voice pack (a single tensor, expected [510, 1, 256]).
pub fn load_voice_pack(path: &Path) -> Result<PtTensor> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (mut ld, root) = Loader::new(&bytes)?;
    let t = ld.tensor(&root)?;
    ensure!(t.shape.len() == 3 && t.shape[1] == 1 && t.shape[2] == 256, "unexpected voice pack shape {:?}", t.shape);
    Ok(t)
}
