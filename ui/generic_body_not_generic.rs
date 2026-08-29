// A function generic over a type or const parameter is compiled once per
// distinct argument set the crate calls it with. When most of its body never
// touches the parameter, every copy repeats that part unchanged; a thin
// generic shim that converts its argument and calls one non-generic inner
// function compiles it once. A body that works on its parameter throughout, a
// single instantiation, a lifetime parameter, or a body too small to split is
// not flagged.

// Flagged: only `bytes.as_ref()` and the drop of `bytes` depend on `B`; the
// loop is the same machine code in all three instantiations `main` makes.
pub fn checksum<B: AsRef<[u8]>>(bytes: B) -> u32 {
    let src = bytes.as_ref();
    let mut acc = 17u32;
    let mut run = 0u32;
    for byte in src {
        let v = u32::from(*byte);
        acc = acc.wrapping_mul(31).wrapping_add(v);
        if v & 1 == 0 {
            run += 1;
        } else {
            run = 0;
        }
        acc ^= run << 3;
    }
    acc.rotate_left(7) ^ (src.len() as u32)
}

pub struct Framed<T> {
    pub payload: T,
    pub header: [u8; 4],
    pub declared_len: u32,
}

impl<T> Framed<T> {
    // Flagged: `T` is the struct's parameter, not the method's, but the
    // method is still compiled once per `T`. It never touches `payload`, and
    // `self.header` is a `[u8; 4]` whatever `T` is, so no statement depends.
    pub fn header_word(&self) -> u32 {
        let header = self.header;
        let declared = self.declared_len;
        let mut word = 0u32;
        let mut shift = 0u32;
        for b in header {
            word |= u32::from(b) << shift;
            shift += 8;
        }
        let padded = (declared + 3) & !3;
        if word > padded {
            word - padded
        } else {
            padded.wrapping_sub(word).wrapping_mul(3)
        }
    }
}

// Flagged: a const parameter counts like a type parameter. `N` bounds the
// loop and the index and nothing else.
pub fn fold_block<const N: usize>(block: [u8; N]) -> u32 {
    let mut lo = 1u32;
    let mut hi = 0u32;
    let mut i = 0usize;
    while i < N {
        lo = (lo + u32::from(block[i])) % 65521;
        hi = (hi + lo) % 65521;
        i += 1;
    }
    let folded = (hi << 16) | lo;
    if folded % 2 == 0 { folded / 2 } else { folded.wrapping_mul(3) + 1 }
}

// Flagged: never called at a concrete type directly, but `relay` is, twice,
// and each instantiation of `relay` instantiates this.
pub fn digits<S: AsRef<str>>(text: S) -> u32 {
    let s = text.as_ref();
    let mut total = 0u32;
    let mut weight = 1u32;
    let mut groups = 0u32;
    for ch in s.bytes() {
        if ch.is_ascii_digit() {
            total = total.wrapping_add(u32::from(ch - b'0') * weight);
            weight = weight.wrapping_mul(10);
        } else {
            if weight != 1 {
                groups += 1;
            }
            weight = 1;
        }
    }
    (total ^ weight.rotate_right(5)).wrapping_add(groups)
}

// Fine: two statements per instantiation; this is the shim shape the lint
// asks for.
pub fn relay<S: AsRef<str>>(text: S) -> u32 {
    digits(text).wrapping_add(1)
}

// Fine: called twice, both times with `&str`, so there is one instantiation
// and nothing is duplicated.
pub fn vowels<S: AsRef<str>>(text: S) -> u32 {
    let s = text.as_ref();
    let mut n = 0u32;
    let mut last_was = false;
    for ch in s.bytes() {
        let is = matches!(ch, b'a' | b'e' | b'i' | b'o' | b'u');
        if is && !last_was {
            n += 2;
        } else if is {
            n += 1;
        }
        last_was = is;
    }
    n.saturating_sub(1)
}

// Fine: every statement moves, compares or copies a `T`; there is nothing to
// hoist out.
pub fn largest<T: PartialOrd + Copy>(items: &[T], floor: T) -> T {
    let mut best = floor;
    for item in items {
        if *item > best {
            best = *item;
        }
    }
    best
}

// Fine: two instantiations, but the shared part is a multiply and an add
// (under `generic-body-not-generic-min-statements`).
pub fn padded_len<B: AsRef<[u8]>>(bytes: B) -> usize {
    bytes.as_ref().len() * 2 + 1
}

// Fine: already split. The generic part is one call; the work is in
// `spread_inner`, which is compiled once.
pub fn spread<B: AsRef<[u8]>>(bytes: B) -> u32 {
    spread_inner(bytes.as_ref())
}

fn spread_inner(src: &[u8]) -> u32 {
    let mut min = u32::from(u8::MAX);
    let mut max = 0u32;
    for byte in src {
        let v = u32::from(*byte);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if src.is_empty() { 0 } else { (max - min) * 4 + max }
}

// Fine: a lifetime parameter is erased before codegen, so every call shares
// one body however many there are.
pub fn trailing_spaces<'a>(line: &'a str) -> &'a str {
    let bytes = line.as_bytes();
    let mut end = bytes.len();
    let mut seen = 0u32;
    while end > 0 && bytes[end - 1] == b' ' {
        end -= 1;
        seen += 1;
    }
    if seen > 4 { line } else { &line[..end] }
}

// Quiet: the arithmetic lives in a closure, and closures are not measured,
// though each instantiation of `weigh` compiles its own copy of this one.
pub fn weigh<B: AsRef<[u8]>>(bytes: B) -> u32 {
    let step = |acc: u32, b: &u8| {
        let v = u32::from(*b);
        let mixed = acc.rotate_left(5) ^ v;
        let bumped = if v > 0x7f {
            mixed.wrapping_add(v * 3)
        } else {
            mixed.wrapping_sub(v + 11)
        };
        let folded = (bumped >> 16) ^ (bumped & 0xffff);
        if folded % 7 == 0 { folded | 1 } else { folded.wrapping_mul(40503) }
    };
    bytes.as_ref().iter().fold(5381, step)
}

pub struct Sealed<T> {
    pub token: T,
    pub table: [u8; 8],
    pub seed: u32,
    pub word: u32,
}

// Flagged: nothing in `main` calls `deref` by name. `sealed.leading_zeros()`
// reaches it through the auto-deref rustc inserts before the method call and
// `&other` through a deref coercion, and each is an instantiation.
impl<T> std::ops::Deref for Sealed<T> {
    type Target = u32;
    fn deref(&self) -> &u32 {
        let mut acc = self.seed;
        let mut odd = 0u32;
        for b in self.table {
            let v = u32::from(b);
            acc = acc.rotate_left(3) ^ v;
            if v % 2 == 1 {
                odd += 1;
            }
        }
        let pick = (acc ^ odd) % 3;
        if pick == 0 {
            &self.seed
        } else if pick == 1 {
            &self.word
        } else if odd > 4 {
            &self.seed
        } else {
            &self.word
        }
    }
}

pub struct Ledger<T> {
    pub tag: T,
    pub counts: [u32; 6],
    pub total: u32,
}

// Flagged: no expression anywhere calls `drop`; the two `Ledger`s `main`
// builds run it when they go out of scope, one instantiation each.
impl<T> Drop for Ledger<T> {
    fn drop(&mut self) {
        let mut sum = 0u32;
        let mut peak = 0u32;
        let mut at = 0u32;
        let mut i = 0u32;
        for c in self.counts {
            sum = sum.wrapping_add(c);
            if c > peak {
                peak = c;
                at = i;
            }
            i += 1;
        }
        let spread = peak.saturating_sub(sum / 6);
        self.total = if spread > at { sum ^ spread } else { sum.wrapping_mul(at + 1) };
    }
}

// Quiet: the same body as `checksum`, at two concrete types, but a macro
// wrote the function and the lint does not measure what a macro expands to.
macro_rules! make_summer {
    ($name:ident) => {
        pub fn $name<B: AsRef<[u8]>>(bytes: B) -> u32 {
            let src = bytes.as_ref();
            let mut acc = 17u32;
            let mut run = 0u32;
            for byte in src {
                let v = u32::from(*byte);
                acc = acc.wrapping_mul(31).wrapping_add(v);
                if v & 1 == 0 {
                    run += 1;
                } else {
                    run = 0;
                }
                acc ^= run << 3;
            }
            acc.rotate_left(7) ^ (src.len() as u32)
        }
    };
}

make_summer!(macro_checksum);

// Quiet: the same body as `fold_block`, at two array lengths, but both uses
// are `const` initializers, evaluated at compile time and never compiled
// into the binary.
pub const fn const_fold<const N: usize>(block: [u8; N]) -> u32 {
    let mut lo = 1u32;
    let mut hi = 0u32;
    let mut i = 0usize;
    while i < N {
        lo = (lo + block[i] as u32) % 65521;
        hi = (hi + lo) % 65521;
        i += 1;
    }
    let folded = (hi << 16) | lo;
    if folded % 2 == 0 { folded / 2 } else { folded.wrapping_mul(3) + 1 }
}

pub const FOLD_TWO: u32 = const_fold([1u8, 2]);
pub const FOLD_THREE: u32 = const_fold([1u8, 2, 3]);

fn main() {
    let _ = checksum("abc");
    let _ = checksum(vec![1u8, 2, 3]);
    let _ = checksum([9u8; 4]);
    let small = Framed { payload: 1u8, header: [1, 2, 3, 4], declared_len: 6 };
    let wide = Framed { payload: "wide", header: [4, 3, 2, 1], declared_len: 9 };
    let _ = small.header_word() + wide.header_word();
    let _ = fold_block([1u8, 2]);
    let _ = fold_block([1u8, 2, 3]);
    let _ = relay("12a3");
    let _ = relay(String::from("45"));
    let _ = vowels("aei");
    let _ = vowels("xyz");
    let _ = largest(&[1u8, 9, 3], 0);
    let _ = largest(&[1.5f32, 0.5], 0.0);
    let _ = padded_len("ab");
    let _ = padded_len([0u8; 2]);
    let _ = spread("spread");
    let _ = spread(vec![7u8]);
    let _ = trailing_spaces("a  ");
    let _ = trailing_spaces("b   ");
    let _ = weigh("w");
    let _ = weigh([1u8]);
    let sealed = Sealed { token: 3u8, table: [1, 2, 3, 4, 5, 6, 7, 8], seed: 5, word: 9 };
    let other = Sealed { token: "t", table: [8, 7, 6, 5, 4, 3, 2, 1], seed: 2, word: 4 };
    let coerced: &u32 = &other;
    let _ = sealed.leading_zeros() + coerced;
    let _small_ledger = Ledger { tag: 1u8, counts: [1, 2, 3, 4, 5, 6], total: 0 };
    let _wide_ledger = Ledger { tag: "wide", counts: [6, 5, 4, 3, 2, 1], total: 0 };
    let _ = macro_checksum("m");
    let _ = macro_checksum([0u8; 3]);
    let _ = FOLD_TWO + FOLD_THREE;
}
