//! Native English text frontend (port of the production misaki 0.9.4 en.G2P path;
//! docs/frontend/MISAKI_EN_SPEC.md). Work in progress: F1 components land with differential tests.

#[allow(unsafe_code)]
pub mod espeak;
pub mod g2p;
pub mod lexicon;
pub mod num2words;
pub mod pipeline;
pub mod pystr;
#[rustfmt::skip]
mod pyunicode;
pub mod spacy_tag;
pub mod spacy_tok;

/// misaki MToken (token.py) with the `_` underscore fields flattened.
#[derive(Clone, Debug, Default)]
pub struct MToken {
    pub text: String,
    pub tag: String,
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub rating: Option<i32>,
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<String>,
    pub num_flags: String,
    pub prespace: bool,
}

/// One spaCy token as the frontend consumes it (text, trailing whitespace, fine-grained tag).
#[derive(Clone, Debug)]
pub struct SpacyToken {
    pub text: String,
    pub ws: String,
    pub tag: String,
}
