//! OOV fallback: misaki 0.9.4 `EspeakFallback` over phonemizer-fork's EspeakBackend(en-us,
//! preserve_punctuation=True, with_stress=True, tie='^'), driving libespeak-ng through its C API.
//!
//! libespeak-ng (GPL-3.0-or-later) is a separately installed system library, loaded at runtime with
//! dlopen (default: `libespeak-ng.so.1` through the dynamic loader's search path) together with its
//! language data (default: the library's own data location; `ESPEAK_DATA_PATH` is honoured by the
//! library). It is never built into this binary or run as a subprocess. A missing or unusable
//! library is a hard error. Pronunciations of fallback words follow the installed version; the
//! version and the hashes of the library and the en-us data it actually loaded are part of the
//! frontend identity (and so of the resume key).

use super::g2p::Fallback;
use super::MToken;
use anyhow::{bail, ensure, Context, Result};
use fancy_regex::Regex;
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Library name looked up through the dynamic loader when no explicit library is given.
pub const DEFAULT_LIBRARY: &str = "libespeak-ng.so.1";
/// phonemizer's tie mode (espeakPHONEMES_TIE) needs espeak-ng >= 1.49.
pub const MIN_VERSION: (u32, u32) = (1, 49);
pub const INSTALL_HINT: &str = "install eSpeak NG from your distribution (Debian/Ubuntu: sudo apt install libespeak-ng1), or pass --espeak-lib / --espeak-data";
/// phonemizer resolves language "en-us" to this voice identifier.
const VOICE: &str = "gmw/en-US";
const PHONEMIZER_MARKS: &str = ";:,.!?¡¿—…\"«»“”(){}[]";
/// Files of espeak-ng-data that en-us phonemization reads (hashed into the identity).
const EN_US_DATA: [&str; 6] = ["phontab", "phonindex", "phondata", "intonations", "en_dict", "lang/gmw/en-US"];
/// espeakINITIALIZE_DONT_EXIT: report initialization errors instead of calling exit().
const INITIALIZE_DONT_EXIT: c_int = 0x8000;
/// AUDIO_OUTPUT_SYNCHRONOUS, as phonemizer.
const AUDIO_OUTPUT_SYNCHRONOUS: c_int = 0x02;

type TextToPhonemes = unsafe extern "C" fn(*mut *const c_void, c_int, c_int) -> *const c_char;

/// Which libespeak-ng and language data to load. `None` = the system defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EspeakSource {
    /// shared library file (default: `libespeak-ng.so.1` via the dynamic loader)
    pub library: Option<PathBuf>,
    /// directory CONTAINING `espeak-ng-data/` (default: the library's own data location)
    pub data: Option<PathBuf>,
}

struct Api {
    _lib: libloading::Library,
    text_to_phonemes: TextToPhonemes,
}

/// One process-wide espeak instance (the C library keeps global state; calls are serialized).
pub struct Espeak {
    api: Mutex<Api>,
    pub source: EspeakSource,
    /// version reported by espeak_Info
    pub version: String,
    /// the library file actually mapped into the process
    pub library: PathBuf,
    pub library_sha256: String,
    /// the espeak-ng-data directory the library selected after initialization
    pub data_dir: PathBuf,
    /// sha256 over the en-us data files (EN_US_DATA: names and contents)
    pub data_sha256: String,
    marks_re: Regex,
    syllabic_re: Regex,
}

static INSTANCE: OnceLock<std::result::Result<Espeak, String>> = OnceLock::new();

#[repr(C)]
struct DlInfo {
    dli_fname: *const c_char,
    dli_fbase: *mut c_void,
    dli_sname: *const c_char,
    dli_saddr: *mut c_void,
}

extern "C" {
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// "1.51", "1.52.0", "1.51.1-dev ..." -> (major, minor)
fn parse_version(v: &str) -> Option<(u32, u32)> {
    let mut it = v.split(|c: char| !c.is_ascii_digit());
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

impl Espeak {
    /// Load (once per process) libespeak-ng and its data. Later calls must name the same source.
    pub fn get(source: &EspeakSource) -> Result<&'static Espeak> {
        let r = INSTANCE.get_or_init(|| Self::load(source).map_err(|e| format!("{e:#}")));
        match r {
            Ok(e) => {
                ensure!(e.source == *source, "espeak-ng already loaded from {} / {}", e.library.display(), e.data_dir.display());
                Ok(e)
            }
            Err(msg) => bail!("{msg}"),
        }
    }

    /// One line describing the backend (used in the frontend identity).
    pub fn describe(&self) -> String {
        format!("espeak-ng {} [lib sha256 {}, en-us data sha256 {}]", self.version, &self.library_sha256[..16], &self.data_sha256[..16])
    }

    #[allow(unsafe_code)]
    fn load(source: &EspeakSource) -> Result<Espeak> {
        let lib_name: &std::ffi::OsStr = match &source.library {
            Some(p) => {
                ensure!(p.is_file(), "espeak-ng library not found: {} ({INSTALL_HINT})", p.display());
                p.as_os_str()
            }
            None => DEFAULT_LIBRARY.as_ref(),
        };
        if let Some(d) = &source.data {
            ensure!(d.join("espeak-ng-data/phontab").is_file(), "espeak-ng-data (with phontab) not found under {} (--espeak-data names the directory CONTAINING espeak-ng-data)", d.display());
        }
        // SAFETY: loading a shared library runs its initializers; libespeak-ng's symbols are used
        // with their documented C signatures (speak_lib.h).
        let lib = unsafe { libloading::Library::new(lib_name) }.map_err(|e| anyhow::anyhow!("cannot load the eSpeak NG library {}: {e}; {INSTALL_HINT}", lib_name.to_string_lossy()))?;
        let sym = |name: &str| format!("{} has no symbol {name}: not a usable libespeak-ng ({INSTALL_HINT})", lib_name.to_string_lossy());
        let (version, library, data_dir, t2p) = unsafe {
            let init: libloading::Symbol<unsafe extern "C" fn(c_int, c_int, *const c_char, c_int) -> c_int> = lib.get(b"espeak_Initialize\0").with_context(|| sym("espeak_Initialize"))?;
            let info: libloading::Symbol<unsafe extern "C" fn(*mut *const c_char) -> *const c_char> = lib.get(b"espeak_Info\0").with_context(|| sym("espeak_Info"))?;
            let set_voice: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> c_int> = lib.get(b"espeak_SetVoiceByName\0").with_context(|| sym("espeak_SetVoiceByName"))?;
            let t2p: libloading::Symbol<TextToPhonemes> = lib.get(b"espeak_TextToPhonemes\0").with_context(|| sym("espeak_TextToPhonemes"))?;
            let t2p: TextToPhonemes = *t2p;
            // the file the loader actually mapped (the default name is resolved by the loader)
            let mut dl = DlInfo { dli_fname: std::ptr::null(), dli_fbase: std::ptr::null_mut(), dli_sname: std::ptr::null(), dli_saddr: std::ptr::null_mut() };
            ensure!(dladdr(t2p as *const c_void, &mut dl) != 0 && !dl.dli_fname.is_null(), "cannot locate the loaded eSpeak NG library file");
            let library = PathBuf::from(CStr::from_ptr(dl.dli_fname).to_string_lossy().into_owned());
            let version = CStr::from_ptr(info(std::ptr::null_mut())).to_string_lossy().into_owned();
            match parse_version(&version) {
                Some(v) if v >= MIN_VERSION => {}
                _ => bail!("eSpeak NG {version} at {} is not supported (need >= {}.{}; {INSTALL_HINT})", library.display(), MIN_VERSION.0, MIN_VERSION.1),
            }
            let path = source.data.as_ref().map(|d| CString::new(d.as_os_str().as_encoded_bytes())).transpose()?;
            // with DONT_EXIT a data error is printed by the library and initialization continues;
            // the selected data directory is verified below
            ensure!(init(AUDIO_OUTPUT_SYNCHRONOUS, 0, path.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()), INITIALIZE_DONT_EXIT) > 0, "espeak_Initialize failed");
            let mut data: *const c_char = std::ptr::null();
            info(&mut data);
            ensure!(!data.is_null(), "eSpeak NG did not report its data directory");
            let data_dir = PathBuf::from(CStr::from_ptr(data).to_string_lossy().into_owned());
            ensure!(data_dir.join("phontab").is_file(), "eSpeak NG language data not found (library {} looked in {}); {INSTALL_HINT}", library.display(), data_dir.display());
            let voice = CString::new(VOICE)?;
            ensure!(set_voice(voice.as_ptr()) == 0, "eSpeak NG voice {VOICE} not available in {} ({INSTALL_HINT})", data_dir.display());
            (version, library, data_dir, t2p)
        };
        let library_sha256 = sha256_hex(&std::fs::read(&library).with_context(|| format!("hashing {}", library.display()))?);
        let mut h = Sha256::new();
        for f in EN_US_DATA {
            let bytes = std::fs::read(data_dir.join(f)).with_context(|| format!("eSpeak NG data file {} missing ({INSTALL_HINT})", data_dir.join(f).display()))?;
            h.update(f.as_bytes());
            h.update(Sha256::digest(&bytes));
        }
        let data_sha256: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        let class: String = PHONEMIZER_MARKS.chars().map(|c| regex_escape(c)).collect();
        let e = Espeak {
            api: Mutex::new(Api { _lib: lib, text_to_phonemes: t2p }),
            source: source.clone(),
            version,
            library,
            library_sha256,
            data_dir,
            data_sha256,
            marks_re: Regex::new(&format!(r"(\s*[{class}]+\s*)+"))?,
            syllabic_re: Regex::new(r"(\S)\x{0329}")?,
        };
        // usability probe: an ordinary word must phonemize
        let probe = e.fallback("hello")?;
        ensure!(probe.as_deref().map_or(false, |p| p.chars().any(char::is_alphabetic)), "eSpeak NG {} produced no phonemes for a probe word (library {}, data {})", e.version, e.library.display(), e.data_dir.display());
        Ok(e)
    }

    /// phonemizer EspeakWrapper.text_to_phonemes(text, tie=True): IPA with U+0361 ties, all clauses.
    #[allow(unsafe_code)]
    fn text_to_phonemes(&self, text: &str) -> Result<String> {
        let api = self.api.lock().unwrap();
        let c = CString::new(text).context("NUL in text")?;
        let mut ptr: *const c_void = c.as_ptr() as *const c_void;
        let mode: c_int = 0x02 | (0x01 << 7) | (('\u{0361}' as c_int) << 8);
        let mut parts = vec![];
        while !ptr.is_null() {
            // SAFETY: ptr points into `c` (NUL-terminated) or is advanced by espeak within it /
            // set to NULL at the end; the returned buffer is copied before the next call.
            let out = unsafe { (api.text_to_phonemes)(&mut ptr, 1, mode) };
            if !out.is_null() {
                let s = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
                if !s.is_empty() {
                    parts.push(s);
                }
            }
        }
        Ok(parts.join(" "))
    }

    /// EspeakBackend._postprocess_line (strip=False, word separator ' ', tie '^', keep-flags).
    fn postprocess_line(line: &str) -> String {
        let line = super::pystr::strip(line).replace('\n', " ").replace("  ", " ");
        let line = collapse_underscores(&line).replace("_ ", " ");
        if line.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for word in line.split(' ') {
            out.push_str(&super::pystr::strip(word).replace('\u{0361}', "^"));
            out.push(' ');
        }
        out
    }

    /// Punctuation._preserve_line for one input line (index 0).
    fn preserve(&self, line: &str) -> (Vec<String>, Vec<(String, char)>) {
        let matches: Vec<(usize, usize)> = self.marks_re.find_iter(line).filter_map(|m| m.ok()).map(|m| (m.start(), m.end())).collect();
        if matches.is_empty() {
            return (vec![line.to_string()], vec![]);
        }
        if matches.len() == 1 && matches[0] == (0, line.len()) {
            return (vec![], vec![(line.to_string(), 'A')]);
        }
        let mut marks = vec![];
        for (k, &(a, b)) in matches.iter().enumerate() {
            let g = &line[a..b];
            let mut pos = 'I';
            if k == 0 && line.starts_with(g) {
                pos = 'B';
            } else if k == matches.len() - 1 && line.ends_with(g) {
                pos = 'E';
            }
            marks.push((g.to_string(), pos));
        }
        let mut out = vec![];
        let mut rest = line.to_string();
        for (m, _) in &marks {
            let split: Vec<&str> = rest.split(m.as_str()).collect();
            out.push(split[0].to_string());
            rest = split[1..].join(m);
        }
        out.push(rest);
        (out.into_iter().filter(|s| !s.is_empty()).collect(), marks)
    }

    /// Punctuation.restore (sep.word ' ', strip False); all marks carry line index 0.
    fn restore(mut text: Vec<String>, mut marks: Vec<(String, char)>) -> Vec<String> {
        let mut out = vec![];
        let mut pos = 0usize;
        while !text.is_empty() || !marks.is_empty() {
            if marks.is_empty() {
                for mut line in text.drain(..) {
                    if !line.ends_with(' ') {
                        line.push(' ');
                    }
                    out.push(line);
                }
            } else if text.is_empty() {
                out.push(marks.iter().map(|m| m.0.as_str()).collect::<String>());
                marks.clear();
            } else if pos == 0 {
                let (mark, position) = marks.remove(0);
                if text[0].ends_with(' ') {
                    text[0].pop();
                }
                match position {
                    'B' => text[0] = format!("{mark}{}", text[0]),
                    'E' => {
                        let t = text.remove(0);
                        out.push(format!("{t}{mark}{}", if mark.ends_with(' ') { "" } else { " " }));
                        pos += 1;
                    }
                    'A' => {
                        out.push(format!("{mark}{}", if mark.ends_with(' ') { "" } else { " " }));
                        pos += 1;
                    }
                    _ => {
                        if text.len() == 1 {
                            text[0].push_str(&mark);
                        } else {
                            let first = text.remove(0);
                            text[0] = format!("{first}{mark}{}", text[0]);
                        }
                    }
                }
            } else {
                out.push(text.remove(0));
                pos += 1;
            }
        }
        out
    }

    /// EspeakBackend.phonemize([text]) -> list of strings.
    pub fn phonemize(&self, text: &str) -> Result<Vec<String>> {
        let (chunks, marks) = self.preserve(text);
        let mut phonemized = vec![];
        for c in chunks {
            phonemized.push(Self::postprocess_line(&self.text_to_phonemes(&c)?));
        }
        Ok(Self::restore(phonemized, marks))
    }

    /// misaki EspeakFallback.__call__ (American English, version None).
    pub fn fallback(&self, text: &str) -> Result<Option<String>> {
        let ps = self.phonemize(text)?;
        let Some(first) = ps.first() else { return Ok(None) };
        let mut ps = super::pystr::strip(first).to_string();
        for (old, new) in e2m() {
            ps = ps.replace(old, new);
        }
        ps = self.syllabic_re.replace_all(&ps, "ᵊ$1").into_owned().replace('\u{0329}', "");
        ps = ps.replace("o^ʊ", "O").replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ").replace("ɪə", "iə").replace('ː', "");
        ps = ps.replace('o', "ɔ");
        ps = ps.replace('ɾ', "T").replace('ʔ', "t");
        Ok(Some(ps.replace('^', "")))
    }
}

impl Fallback for Espeak {
    fn g2p(&self, tk: &MToken) -> Result<(Option<String>, Option<i32>)> {
        // an internal espeak failure fails the input line (never a silently dropped word)
        Ok(match self.fallback(&tk.text).with_context(|| format!("eSpeak NG fallback on {:?}", tk.text))? {
            Some(p) => (Some(p), Some(2)),
            None => (None, None),
        })
    }
}

/// misaki EspeakFallback.E2M: insertion order, stable-sorted by descending key length (code points).
fn e2m() -> &'static [(&'static str, &'static str)] {
    static T: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    T.get_or_init(|| {
        let mut v = vec![
            ("ʔˌn\u{0329}", "ʔn"),
            ("ʔn\u{0329}", "ʔn"),
            ("a^ɪ", "I"),
            ("a^ʊ", "W"),
            ("d^ʒ", "ʤ"),
            ("e^ɪ", "A"),
            ("e", "A"),
            ("t^ʃ", "ʧ"),
            ("ɔ^ɪ", "Y"),
            ("ə^l", "ᵊl"),
            ("ʲo", "jo"),
            ("ʲə", "jə"),
            ("ʲ", ""),
            ("ɚ", "əɹ"),
            ("r", "ɹ"),
            ("x", "k"),
            ("ç", "k"),
            ("ɐ", "ə"),
            ("ɬ", "l"),
            ("\u{0303}", ""),
        ];
        v.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
        v
    })
}

fn collapse_underscores(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev = false;
    for c in s.chars() {
        if c == '_' {
            if !prev {
                out.push(c);
            }
            prev = true;
        } else {
            out.push(c);
            prev = false;
        }
    }
    out
}

fn regex_escape(c: char) -> String {
    if "\\^$.|?*+()[]{}-&~#".contains(c) || c.is_whitespace() {
        format!("\\{c}")
    } else {
        c.to_string()
    }
}
