//! OOV fallback: misaki 0.9.4 `EspeakFallback` over phonemizer-fork's EspeakBackend(en-us,
//! preserve_punctuation=True, with_stress=True, tie='^'), driving libespeak-ng through its C API.
//!
//! libespeak-ng (GPL-3.0-or-later) is an EXTERNAL runtime dependency loaded with dlopen from an
//! explicit path (default: the pinned 1.52.0 copy staged by scripts/stage_espeak.sh). It is never
//! built into this binary, never a subprocess, and a missing/mismatched library is a hard error.

use super::g2p::Fallback;
use super::MToken;
use anyhow::{bail, ensure, Context, Result};
use fancy_regex::Regex;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// espeak-ng version the production reference ships (espeakng_loader 0.2.4).
pub const PINNED_VERSION: &str = "1.52.0";
/// phonemizer resolves language "en-us" to this voice identifier on the pinned data.
const VOICE: &str = "gmw/en-US";
const PHONEMIZER_MARKS: &str = ";:,.!?¡¿—…\"«»“”(){}[]";

type TextToPhonemes = unsafe extern "C" fn(*mut *const c_void, c_int, c_int) -> *const c_char;

struct Api {
    _lib: libloading::Library,
    text_to_phonemes: TextToPhonemes,
}

/// One process-wide espeak instance (the C library keeps global state; calls are serialized).
pub struct Espeak {
    api: Mutex<Api>,
    pub version: String,
    pub library: PathBuf,
    pub data_dir: PathBuf,
    marks_re: Regex,
    syllabic_re: Regex,
}

static INSTANCE: OnceLock<std::result::Result<Espeak, String>> = OnceLock::new();

/// Default location of the pinned library + data (explicit paths override it).
pub fn default_dir() -> PathBuf {
    let root = std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into());
    PathBuf::from(root).join("frontend/espeak-ng-1.52.0")
}

impl Espeak {
    /// Load (once per process) libespeak-ng from `library` with voice data `data_dir`
    /// (the directory CONTAINING `espeak-ng-data`). Later calls must name the same paths.
    pub fn get(library: &Path, data_dir: &Path) -> Result<&'static Espeak> {
        let r = INSTANCE.get_or_init(|| Self::load(library, data_dir).map_err(|e| format!("{e:#}")));
        match r {
            Ok(e) => {
                ensure!(e.library == library && e.data_dir == data_dir, "espeak already loaded from {} / {}", e.library.display(), e.data_dir.display());
                Ok(e)
            }
            Err(msg) => bail!("{msg}"),
        }
    }

    #[allow(unsafe_code)]
    fn load(library: &Path, data_dir: &Path) -> Result<Espeak> {
        ensure!(library.is_file(), "espeak-ng library not found: {} (run scripts/stage_espeak.sh or pass --espeak-lib)", library.display());
        ensure!(data_dir.join("espeak-ng-data").is_dir(), "espeak-ng-data not found under {}", data_dir.display());
        // SAFETY: loading a shared library runs its initializers; the path is an explicit,
        // pinned libespeak-ng whose symbols are used with their documented C signatures.
        let lib = unsafe { libloading::Library::new(library) }.with_context(|| format!("dlopen {}", library.display()))?;
        let (version, t2p) = unsafe {
            let init: libloading::Symbol<unsafe extern "C" fn(c_int, c_int, *const c_char, c_int) -> c_int> = lib.get(b"espeak_Initialize\0")?;
            let info: libloading::Symbol<unsafe extern "C" fn(*mut *const c_char) -> *const c_char> = lib.get(b"espeak_Info\0")?;
            let set_voice: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> c_int> = lib.get(b"espeak_SetVoiceByName\0")?;
            let t2p: libloading::Symbol<TextToPhonemes> = lib.get(b"espeak_TextToPhonemes\0")?;
            let path = CString::new(data_dir.as_os_str().as_encoded_bytes())?;
            // AUDIO_OUTPUT_SYNCHRONOUS (0x02), as phonemizer
            ensure!(init(0x02, 0, path.as_ptr(), 0) > 0, "espeak_Initialize failed");
            let v = CStr::from_ptr(info(std::ptr::null_mut())).to_string_lossy().into_owned();
            let voice = CString::new(VOICE)?;
            ensure!(set_voice(voice.as_ptr()) == 0, "espeak_SetVoiceByName({VOICE}) failed");
            (v, *t2p)
        };
        ensure!(version.starts_with(PINNED_VERSION), "espeak-ng version {version} != pinned {PINNED_VERSION} (pronunciations would differ from the reference)");
        let class: String = PHONEMIZER_MARKS.chars().map(|c| regex_escape(c)).collect();
        Ok(Espeak {
            api: Mutex::new(Api { _lib: lib, text_to_phonemes: t2p }),
            version,
            library: library.to_path_buf(),
            data_dir: data_dir.to_path_buf(),
            marks_re: Regex::new(&format!(r"(\s*[{class}]+\s*)+"))?,
            syllabic_re: Regex::new(r"(\S)\x{0329}")?,
        })
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
        let line = line.trim().replace('\n', " ").replace("  ", " ");
        let line = collapse_underscores(&line).replace("_ ", " ");
        if line.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for word in line.split(' ') {
            out.push_str(&word.trim().replace('\u{0361}', "^"));
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
        let mut ps = first.trim().to_string();
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
    fn g2p(&self, tk: &MToken) -> (Option<String>, Option<i32>) {
        match self.fallback(&tk.text) {
            Ok(Some(p)) => (Some(p), Some(2)),
            Ok(None) => (None, None),
            // an internal espeak failure must not silently drop words
            Err(e) => panic!("espeak fallback failed on a token: {e:#}"),
        }
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
