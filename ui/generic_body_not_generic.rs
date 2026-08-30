// A function generic over a type or const parameter is compiled once per
// distinct argument set the crate calls it with. The lint reports one only
// when it can point at a stretch of the body -- a single-entry, single-exit
// run of MIR blocks -- that names no parameter, reads only values whose types
// name none, and hands on only such values: that stretch is the same code in
// every copy, so a non-generic fn taking those values can hold it, the generic
// fn keeps its signature and calls it, and the only cost is the call. What
// depends on the parameter may come before the stretch (`bytes.as_ref()`) and
// after it (the drop of `bytes`); a `panic!` inside it is not a way out of it;
// a stretch inside a loop costs a call per pass and the note says so. A body
// that touches its parameter every few statements, a stretch shorter than
// `generic-body-not-generic-min-statements` counting only the statements
// written by hand (R1: what a macro expands to moves with the stretch but
// does not make it worth naming), a body whose every line reads through
// `&self` where `Self` holds the parameter, a single instantiation, a
// lifetime parameter, `#[inline(always)]`, a closure or a macro-written fn is
// not flagged. Nor is a stretch that could not be moved out as it stands, or
// not for the price of one call: one that would take a value each copy knows
// as a constant (R2: `width_of(TAG)`, which folds in place and would not
// behind a call), one entered only under a branch on such a constant (R4:
// each copy keeps one arm, so the arm is not code every copy carries), one
// that would hand back a borrow of a local it makes itself (R3), one whose
// arguments would borrow one another at the call (R6: `inner(rec, name)`
// with `name = &rec.name`), one whose hand-back would keep an argument
// borrowed while the body touches that argument again (R7), and one a
// compiler-made drop flag would have to cross (R5; such a flag is never
// listed either). The R-numbers are the refusal rules the comments below
// cite. Each comment names the stretch expected (its first and last source
// line), what it takes and what it yields, or says why there is none.

// Flagged: only `bytes.as_ref()` before the loop and the drop of `bytes` after
// the last line depend on `B`. The stretch is `let mut acc = 17u32;` through
// `.rotate_left(7)` -- the loop and the tail, which ends in a call so the
// value returned is made before the block that drops `bytes` -- the same in
// all three instantiations `main` makes; it takes `src: &[u8]` and yields the
// `u32` returned. All the fn wants from `B` is that `&[u8]`, and the help
// says it could take one.
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
    (acc ^ (src.len() as u32)).rotate_left(7)
}

pub struct Framed<T> {
    pub payload: T,
    pub header: [u8; 4],
    pub declared_len: u32,
}

impl<T> Framed<T> {
    // Flagged: `T` is the struct's parameter, not the method's, but the
    // method is still compiled once per `T`. The two reads through `&self`
    // come first -- where a field sits inside `Framed<T>` depends on `T` --
    // and are hoisted into locals, the second through a call so the block
    // they sit in ends with them; everything from `let mut word = 0u32;`
    // through `.wrapping_mul(3)` and the return works on a `[u8; 4]` and a
    // `u32`. The stretch takes `header: [u8; 4]` and `declared: u32` and
    // yields the `u32` returned.
    pub fn header_word(&self) -> u32 {
        let header = self.header;
        let declared = u32::from_be(self.declared_len);
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

// Flagged: a const parameter counts like a type parameter. `N` is the length
// of `block` and nothing else; past `as_slice` the body reads a `&[u8]`. The
// stretch is `let mut lo = 1u32;` through the last line and the return; it
// takes `bytes: &[u8]` and yields the `u32` returned. (Indexing `block[i]`
// under `while i < N` would not do: the test against `N` and the array's type
// bring `N` into every pass of the loop.)
pub fn fold_block<const N: usize>(block: [u8; N]) -> u32 {
    let bytes = block.as_slice();
    let mut lo = 1u32;
    let mut hi = 0u32;
    let mut i = 0usize;
    while i < bytes.len() {
        lo = (lo + u32::from(bytes[i])) % 65521;
        hi = (hi + lo) % 65521;
        i += 1;
    }
    let folded = (hi << 16) | lo;
    if folded % 2 == 0 { folded / 2 } else { folded.wrapping_mul(3) + 1 }
}

// Flagged: never called at a concrete type directly, but `relay` is, twice,
// and each instantiation of `relay` instantiates this. `#[inline]` does not
// excuse it -- the hint is not a demand -- but the note mentions it. The
// stretch is `let mut total = 0u32;` through `.wrapping_add(groups)`, before
// the drop of `text`; it takes `s: &str` and yields the `u32` returned.
#[inline]
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

pub trait HasName {
    fn name(&self) -> &str;
}

pub struct Ann;
pub struct Bob;
pub struct Cy;

impl HasName for Ann {
    fn name(&self) -> &str {
        "ann"
    }
}

impl HasName for Bob {
    fn name(&self) -> &str {
        "Bob"
    }
}

impl HasName for Cy {
    fn name(&self) -> &str {
        "cy"
    }
}

// Flagged, and told the sharper thing: all `tally` wants from `X` is the
// `&str` that `HasName::name` returns, so it could take the `&str`. The
// stretch is `let bytes = s.as_bytes();` through `.wrapping_add(..)`, before
// the drop of `x`; it takes `s: &str` and yields the `u32` returned. Three
// instantiations.
pub fn tally<X: HasName>(x: X) -> u32 {
    let s = x.name();
    let bytes = s.as_bytes();
    let mut h = 2166136261u32;
    let mut upper = 0u32;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        h = (h ^ u32::from(b)).wrapping_mul(16777619);
        if b.is_ascii_uppercase() {
            upper += 1;
        }
        i += 1;
    }
    (h ^ upper).wrapping_add(bytes.len() as u32)
}

// Flagged: `WIDE` is tested once, in the middle, which cuts the body in two.
// The first loop with the two `let`s before it is one stretch -- it starts at
// the fn's first statement, so there is no prefix, and `src` is the fn's own
// argument, whose type names no parameter -- and is the one reported: `let
// mut acc = 1u32;` through `i += 1;`, taking `src: &[u8]` and yielding `acc:
// u32`. The second loop with the tail and the return is another, long enough
// to report on its own but shorter, so the note counts it as one more: it
// starts where the two sides of `if WIDE` meet again, so every copy runs it
// whichever way its `WIDE` went. Only the one line inside the `if` belongs to
// one copy and not the other (contrast `arm_sum`).
pub fn wide_sum<const WIDE: bool>(src: &[u8]) -> u32 {
    let mut acc = 1u32;
    let mut i = 0usize;
    while i < src.len() {
        let v = u32::from(src[i]);
        acc = acc.rotate_left(5) ^ v;
        acc = acc.wrapping_mul(3).wrapping_add(v >> 1);
        if acc & 0x10 == 0 {
            acc = acc.wrapping_add(i as u32);
        }
        i += 1;
    }
    if WIDE {
        acc = acc.swap_bytes();
    }
    let mut folded = acc;
    let mut j = 0usize;
    while j < src.len() {
        folded = folded.wrapping_add(u32::from(src[j]) << (j % 3));
        j += 2;
    }
    (folded ^ (src.len() as u32)).wrapping_add(7)
}

// Quiet (R4, entry under a const-derived branch): the `else` arm is the loop
// of `wide_sum` and then some -- well past the minimum, reading only `src`,
// yielding `acc` -- but which arm runs is `FLAG`, a constant in each copy:
// after constant propagation `arm_sum::<true>` keeps the first arm and
// `arm_sum::<false>` the second, so that stretch is code one copy carries,
// not both, and moving it out shares nothing. What the copies do share, the
// line before the test and the line after the arms rejoin, is too short.
pub fn arm_sum<const FLAG: bool>(src: &[u8]) -> u32 {
    let mut acc = 3u32;
    if FLAG {
        acc ^= src.len() as u32;
    } else {
        let mut i = 0usize;
        while i < src.len() {
            let v = u32::from(src[i]);
            acc = acc.rotate_left(5) ^ v;
            acc = acc.wrapping_mul(3).wrapping_add(v >> 1);
            if acc & 0x10 == 0 {
                acc = acc.wrapping_add(i as u32);
            }
            acc ^= acc >> 7;
            i += 1;
        }
        acc = acc.swap_bytes();
    }
    acc.wrapping_add(7)
}

#[derive(Clone, Copy)]
pub enum Coding {
    Plain,
    Hex,
    Sum,
    Xor,
}

pub const fn coding_of(tag: u8) -> Coding {
    match tag {
        0 => Coding::Plain,
        1 => Coding::Hex,
        2 => Coding::Sum,
        _ => Coding::Xor,
    }
}

// Quiet (R4, arm of a `match` on a const-derived value): the `Hex` arm reads
// only `src`, hands back the `u32` the fn returns, and is thirty statements
// written by hand -- but the `match` is on `coding_of(TAG)`, which names no
// parameter in its type and is a constant in each copy all the same: every
// copy keeps the one arm its `TAG` selects and drops the other three, so the
// arm is code one copy in four carries. The block that switches is itself
// clean (`coding` is a plain enum); what marks it is where its value came
// from.
pub fn encode_arm<const TAG: u8>(src: &[u8]) -> u32 {
    let coding = coding_of(TAG);
    match coding {
        Coding::Plain => {
            let mut acc = 0u32;
            for &b in src {
                acc = acc.wrapping_mul(31).wrapping_add(u32::from(b));
            }
            acc
        }
        Coding::Hex => {
            let mut acc = 1u32;
            for &b in src {
                acc = acc.rotate_left(4) ^ u32::from(b >> 4);
                acc = acc.rotate_left(4) ^ u32::from(b & 15);
            }
            acc
        }
        Coding::Sum => src.iter().map(|&b| u32::from(b)).sum::<u32>().wrapping_mul(7),
        Coding::Xor => {
            let mut acc = 0xffu32;
            for &b in src {
                acc ^= u32::from(b);
                acc = acc.rotate_right(1);
            }
            acc
        }
    }
}

pub const fn width_of(tag: u8) -> usize {
    match tag {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8,
    }
}

// Quiet (R2, a const-derived value among what the stretch would take; bun
// `to_bun_string_comptime<const ENCODING>`): everything past the first line
// reads only `src` and `width`, and `width` is a plain `usize` -- but its
// value is `width_of(TAG)`, a different constant in each copy, so
// `chunks(width)`, the shifts and the multiply all fold to constants per
// copy. Behind a call taking `width: usize` they are computed at run time in
// every copy: that is not the same code at the cost of a call, so it is not
// offered. The loop is the only stretch here that reaches the minimum, and
// `width` is live into it.
pub fn encode_as<const TAG: u8>(src: &[u8]) -> u32 {
    let width = width_of(TAG);
    let mut acc = 0u32;
    let mut n = 0u32;
    for chunk in src.chunks(width) {
        let mut word = (width as u32).wrapping_mul(0x9e37_79b9);
        for &b in chunk {
            word = word.rotate_left(width as u32 + 3) ^ u32::from(b);
        }
        acc = acc.rotate_left(width as u32).wrapping_add(word);
        acc ^= acc >> (17 - width as u32);
        n = n.wrapping_add(width as u32);
    }
    acc ^ n
}

// Flagged (R2 does not reach it): `TAG` goes into one byte of `buf` and the
// rest of the body fills the bytes after it from `src` and from each other --
// a header stamped, then the record. That store puts the constant in memory;
// `buf` is run-time data with a constant somewhere in it, not a value each
// copy knows, and nothing past the store folds per copy. The stretch after it
// is every copy's, `buf` goes in as `&mut`, and it is offered.
pub fn stamp_tag<const TAG: u8>(src: &[u8; 4]) -> [u8; 16] {
    let mut buf = [0u8; 16];
    buf[3] = TAG;
    buf[4] = src[0];
    buf[5] = src[1];
    buf[6] = src[2];
    buf[7] = src[3];
    buf[8] = buf[4] ^ buf[5];
    buf[9] = buf[6] ^ buf[7];
    buf[10] = buf[8].wrapping_add(buf[9]);
    buf[11] = buf[10].rotate_left(3);
    buf[12] = buf[11] ^ buf[4];
    buf[13] = buf[12].wrapping_mul(31);
    buf[14] = buf[13] ^ buf[5];
    buf[15] = buf[14].wrapping_add(buf[6]);
    buf[0] = buf[15];
    buf
}

// Flagged, the same stretch: here the per-copy constant is the index, not
// the byte stored. `buf[TAG as usize] = 7` reads the constant to find the
// place and stores a plain `7` there; where a value goes says nothing about
// what it was computed from, so `buf` is no more a constant of the
// instantiation than in `stamp_tag`.
pub fn stamp_at<const TAG: u8>(src: &[u8; 4]) -> [u8; 16] {
    let mut buf = [0u8; 16];
    buf[TAG as usize] = 7;
    buf[4] = src[0];
    buf[5] = src[1];
    buf[6] = src[2];
    buf[7] = src[3];
    buf[8] = buf[4] ^ buf[5];
    buf[9] = buf[6] ^ buf[7];
    buf[10] = buf[8].wrapping_add(buf[9]);
    buf[11] = buf[10].rotate_left(3);
    buf[12] = buf[11] ^ buf[4];
    buf[13] = buf[12].wrapping_mul(31);
    buf[14] = buf[13] ^ buf[5];
    buf[15] = buf[14].wrapping_add(buf[6]);
    buf[0] = buf[15];
    buf
}

pub trait Sink {
    fn put(&mut self, byte: u8);
}

pub struct Count(pub u32);
pub struct Last(pub u8);

impl Sink for Count {
    fn put(&mut self, _byte: u8) {
        self.0 += 1;
    }
}

impl Sink for Last {
    fn put(&mut self, byte: u8) {
        self.0 = byte;
    }
}

// Flagged: `sink.put` runs at the top of every iteration, so the stretch is
// the rest of the loop body -- `let v = u32::from(b);` through `^
// 0x5bd1e995;` -- entered after `put` returns and left for the loop's next
// `next()`; the inner fn would be called once per byte and the note says so.
// It takes `acc: u32` and `b: u8` and yields `acc: u32`.
pub fn drain<S: Sink>(sink: &mut S, src: &[u8]) -> u32 {
    let mut acc = 5381u32;
    for &b in src {
        sink.put(b);
        let v = u32::from(b);
        acc = acc.rotate_left(5).wrapping_add(v);
        acc ^= v << 8;
        if v & 1 == 1 {
            acc = acc.wrapping_mul(33);
        } else {
            acc = acc.wrapping_sub(v >> 1);
        }
        acc = acc.rotate_right(v & 7) ^ 0x5bd1e995;
    }
    acc
}

// Flagged: the `panic!` sits inside the stretch. A block that panics has
// predecessors like any other and must be as parameter-free as the rest, but
// control never leaves it forward, so it is not a second way out: the inner
// fn panics where this one did. The stretch is `let mut acc = 1u32;` through
// `.rotate_right(3)`, before the drop of `bytes`; it takes `src: &[u8]` and
// yields the `u32` returned.
pub fn strict_sum<B: AsRef<[u8]>>(bytes: B) -> u32 {
    let src = bytes.as_ref();
    let mut acc = 1u32;
    let mut zeros = 0u32;
    for byte in src {
        let v = u32::from(*byte);
        if v == 0 {
            zeros += 1;
            if zeros > 3 {
                panic!("more than three zero bytes");
            }
        }
        acc = acc.wrapping_mul(37).wrapping_add(v);
        acc ^= acc >> 11;
    }
    (acc ^ zeros).rotate_right(3)
}

// Flagged: one loop inside another. The inner `while` is a single-entry,
// single-exit stretch on its own and long enough; the outer loop that holds
// it is another, and the largest wins: `let mut acc = 0x9e3779b9u32;`
// through `.rotate_left(5)`, before the drop of `bytes`, taking `src: &[u8]`
// and yielding the `u32` returned.
pub fn lattice<B: AsRef<[u8]>>(bytes: B) -> u32 {
    let src = bytes.as_ref();
    let mut acc = 0x9e3779b9u32;
    let mut i = 0usize;
    while i < src.len() {
        let v = u32::from(src[i]);
        let mut bit = 0u32;
        while bit < 8 {
            if (v >> bit) & 1 == 1 {
                acc = acc.wrapping_mul(31) ^ bit;
            } else {
                acc = acc.rotate_left(3).wrapping_add(bit);
            }
            acc ^= acc >> 7;
            bit += 1;
        }
        acc ^= v;
        i += 1;
    }
    (acc ^ (src.len() as u32)).rotate_left(5)
}

// Quiet: `sink` is written to every few statements, so no stretch between two
// `put`s is long enough, though most statements depend on nothing generic.
pub fn render<S: Sink>(sink: &mut S, mut n: u32) -> u32 {
    n = n.wrapping_mul(2654435761);
    let hi = (n >> 24) as u8;
    sink.put(hi);
    n ^= n >> 13;
    let mid = (n >> 8) as u8;
    sink.put(mid);
    n = n.rotate_left(9).wrapping_add(40503);
    let lo = n as u8;
    sink.put(lo);
    n ^= u32::from(hi) | (u32::from(lo) << 16);
    sink.put((n % 251) as u8);
    n.count_ones() + u32::from(mid)
}

// Quiet: `f` is called in the middle of every iteration; the arithmetic on
// either side of the call is a stretch of its own, each too short.
pub fn scan<F: FnMut(u8)>(src: &[u8], mut f: F) -> u32 {
    let mut acc = 0u32;
    let mut prev = 0u8;
    for &b in src {
        let delta = b.wrapping_sub(prev);
        acc = acc.rotate_left(3) ^ u32::from(delta);
        f(delta);
        prev = b;
        acc = acc.wrapping_add(u32::from(b) << 2);
        if acc & 1 == 1 {
            acc ^= 0x9e3779b9;
        }
    }
    acc
}

// Quiet: `UP` is tested every few lines, so the body is a row of short
// stretches between the tests, none long enough.
pub fn stepped<const UP: bool>(mut n: u32) -> u32 {
    n = n.wrapping_mul(31).rotate_left(3);
    if UP {
        n = n.wrapping_add(7);
    }
    n ^= n >> 5;
    n = n.wrapping_mul(17);
    if UP {
        n = n.swap_bytes();
    }
    n = n.wrapping_sub(n >> 11) | 1;
    if UP {
        n ^= 0xa5a5;
    }
    n.wrapping_mul(3).rotate_right(2)
}

pub struct Wrap<T> {
    pub inner: T,
    pub len: u32,
    pub cap: u32,
}

impl<T> Wrap<T> {
    // Quiet: every statement is `u32` arithmetic, and none of them names `T`,
    // so a stretch could start at the fn's first statement with nothing
    // before it -- but every one of them reads a field through `&self`, the
    // one value such a stretch would have to be handed, and `&Wrap<T>` names
    // `T` (where `len` and `cap` sit inside `Wrap<T>` depends on it). No
    // stretch that leaves `self` out is long enough.
    pub fn slack(&self) -> u32 {
        let mut n = self.cap.wrapping_sub(self.len);
        n = n.wrapping_mul(3) ^ self.cap;
        n = n.rotate_left(self.len & 15);
        n = n.wrapping_add(self.cap >> 2);
        n ^= self.len.count_ones();
        n = n.wrapping_mul(self.cap | 1);
        n = n.wrapping_sub(self.len >> 3);
        n.rotate_right(self.cap & 7)
    }
}

// Quiet: `#[inline(always)]` asks for a copy of the body at every call site;
// a call to one shared inner fn is what the author ruled out. Without the
// attribute the loop and the tail would be reported like `checksum`'s.
#[inline(always)]
pub fn mix<B: AsRef<[u8]>>(bytes: B) -> u32 {
    let src = bytes.as_ref();
    let mut h = 0x811c9dc5u32;
    let mut i = 0usize;
    while i < src.len() {
        h = (h ^ u32::from(src[i])).wrapping_mul(0x01000193);
        h ^= h >> 15;
        h = h.rotate_left(1).wrapping_add(i as u32);
        i += 1;
    }
    h.wrapping_add(src.len() as u32).rotate_left(11)
}

// Fine: called twice, both times with `&str`, so there is one instantiation
// and nothing is duplicated, though the loop would qualify.
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
// `&other` through a deref coercion, and each is an instantiation. The reads
// through `&self` at the top and the borrows of its fields at the bottom
// depend on `T`; the stretch between them is `let mut odd = 0u32;` through
// `|| odd > 4;`, taking `table: [u8; 8]` and `acc: u32` and yielding `pick:
// bool`.
impl<T> std::ops::Deref for Sealed<T> {
    type Target = u32;
    fn deref(&self) -> &u32 {
        let table = self.table;
        let mut acc = self.seed.rotate_left(1);
        let mut odd = 0u32;
        for b in table {
            let v = u32::from(b);
            acc = acc.rotate_left(3) ^ v;
            if v % 2 == 1 {
                odd += 1;
            }
        }
        let pick = (acc ^ odd) % 3 == 1 || odd > 4;
        if pick { &self.word } else { &self.seed }
    }
}

pub struct Record {
    pub name: Vec<u8>,
    pub count: u32,
    pub total: u32,
    pub sent: u32,
    pub flags: u32,
}

// Quiet (R6, two of the inner fn's arguments conflict at the call): between
// the two `put` calls nothing names `S`, the stretch is long enough, and it
// takes `rec: &mut Record`, `name: &[u8]` and `weight`. But `name` *is*
// `&rec.name`: one body may write `rec.count` while `rec.name` is borrowed,
// two arguments may not -- `inner(rec, name, weight)` is E0502. The shorter
// stretch from `rec.count = ..` on does not read `name` at all, and is no
// better: `name` is still held across it for the `put` after.
pub fn recount<S: Sink>(sink: &mut S, rec: &mut Record, weight: u32) {
    let name: &[u8] = &rec.name;
    sink.put(rec.sent as u8);
    let len = name.len() as u32;
    let first = u32::from(name.first().copied().unwrap_or(0));
    let last = u32::from(name.last().copied().unwrap_or(0));
    rec.count = rec.count.wrapping_add(1);
    rec.total = rec.total.wrapping_add(len.wrapping_mul(weight));
    let mix = (first << 8 | last).rotate_left(rec.count & 31);
    rec.flags ^= mix;
    rec.flags = rec.flags.wrapping_mul(0x9e37_79b9) ^ weight;
    if rec.flags & 1 == 0 {
        rec.total = rec.total.rotate_left(3);
    } else {
        rec.count = rec.count.wrapping_add(len & 3);
    }
    rec.flags = rec.flags.wrapping_add(rec.total ^ rec.count);
    sink.put(last as u8);
    sink.put(name.len() as u8);
    rec.sent += 1;
}

// Quiet (R6 again, the borrow never read inside the stretch): `name` is made
// before the first `put` and read only by the last, so it is not among what
// the stretch from `rec.count = ..` through `rec.flags = ..` takes -- that is
// `rec: &mut Record`, `len` and `weight` -- but it is held across it, and
// `inner(rec, len, weight)` with `name = &rec.name` live is E0502 where the
// one body writing `rec.count` beside it was fine.
pub fn restamp<S: Sink>(sink: &mut S, rec: &mut Record, weight: u32) {
    let name: &[u8] = &rec.name;
    let len = name.len() as u32;
    sink.put(len as u8);
    rec.count = rec.count.wrapping_add(1);
    rec.total = rec.total.wrapping_add(len.wrapping_mul(weight));
    let mix = (len << 8 | weight).rotate_left(rec.count & 31);
    rec.flags ^= mix;
    rec.flags = rec.flags.wrapping_mul(0x9e37_79b9) ^ weight;
    if rec.flags & 1 == 0 {
        rec.total = rec.total.rotate_left(3);
    } else {
        rec.count = rec.count.wrapping_add(len & 3);
    }
    rec.flags = rec.flags.wrapping_add(rec.total ^ rec.count);
    sink.put(name.first().copied().unwrap_or(0));
    rec.sent += 1;
}

pub struct Cursor {
    pub data: Vec<u8>,
    pub pos: usize,
    pub reads: u32,
    pub sum: u32,
}

// Quiet (R7, a hand-back keeps one of the inner fn's arguments borrowed past
// the call): from `let start` to the first `put` nothing names `S`, the
// stretch is long enough, and it takes `cur: &mut Cursor` and `want` and
// hands back `sum` and `chunk: &[u8]` -- and `chunk` is `&cur.data[..]`. In
// one body `cur.reads += 1` beside a live borrow of `cur.data` is two places
// of `*cur`; once `chunk` comes back from `inner(cur, want)`, whose signature
// can only tie it to all of `*cur` (mutably: the inner fn wrote `cur.sum`
// through the same reference), `cur.reads += 1` before the last `put` is
// E0503. A stretch starting after `let chunk` would take `chunk` beside `cur`
// (R6), and what is left either side is too short.
pub fn read_into<S: Sink>(sink: &mut S, cur: &mut Cursor, want: usize) {
    let start = cur.pos.min(cur.data.len());
    let end = cur.data.len().min(start + want);
    let chunk: &[u8] = &cur.data[start..end];
    let mut sum = cur.sum;
    for &b in chunk {
        sum = sum.rotate_left(5) ^ u32::from(b);
    }
    cur.sum = sum;
    cur.pos = end;
    let padded = end - start < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    sink.put(sum as u8);
    cur.reads += 1;
    sink.put(chunk.len() as u8);
}

pub trait Push {
    fn push(&mut self, bytes: &[u8]);
}

impl Push for Vec<u8> {
    fn push(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

impl Push for u32 {
    fn push(&mut self, bytes: &[u8]) {
        *self = self.wrapping_add(bytes.len() as u32);
    }
}

// Flagged, short of the R7 stretch (the bun fixture's `read_into`, body
// verbatim): as in `read_into` the largest stretch -- `let start` through
// `cur.reads += 1` -- hands back `chunk`, a borrow of `cur.data`, and the
// body touches `cur` again while `chunk` lives, so R7 refuses it. But here
// ten lines come before `let chunk`, and they are a stretch of their own:
// `let start` through `i += 1;` takes `cur: &mut Cursor` and `want` and
// yields `start`, `take`, `end` and `sum`, plain integers all, so nothing
// stays borrowed past the call. That prefix is what is reported. R7 removes
// a candidate, not the fn: the largest stretch that survives every rule is
// offered, and `read_into` above is quiet only because what survives there
// is too short.
pub fn read_scanned<W: Push>(out: &mut W, cur: &mut Cursor, want: usize) {
    let start = cur.pos.min(cur.data.len());
    let avail = cur.data.len() - start;
    let take = avail.min(want).min(64);
    let end = start + take;
    let mut sum = cur.sum;
    let mut i = start;
    while i < end {
        sum = sum.rotate_left(5) ^ u32::from(cur.data[i]);
        i += 1;
    }
    cur.sum = sum;
    cur.pos = end;
    let chunk: &[u8] = &cur.data[start..end];
    let padded = take < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    cur.reads += 1;
    out.push(chunk);
    if padded {
        out.push(b"\0");
    }
}

// Quiet (R7, the hand-back put away first): the same stretch as `read_into`,
// but past the first `put` the body files `chunk` in `keep`, which outlives
// it, before bumping `cur.reads`. `chunk` itself is dead by then; the loan it
// carried is not -- `keep` holds it, for all of `'c` -- so once `chunk` comes
// back from `inner(cur, want)` tied to all of `*cur`, `cur.reads += 1` is
// E0503 just the same. What a callee handed a place to leave the hand-back
// in does with it is not followed: from the `push` on the loan is taken as
// held.
pub fn read_keep<'c, S: Sink>(sink: &mut S, cur: &'c mut Cursor, want: usize, keep: &mut Vec<&'c [u8]>) {
    let start = cur.pos.min(cur.data.len());
    let end = cur.data.len().min(start + want);
    let chunk: &[u8] = &cur.data[start..end];
    let mut sum = cur.sum;
    for &b in chunk {
        sum = sum.rotate_left(5) ^ u32::from(b);
    }
    cur.sum = sum;
    cur.pos = end;
    let padded = end - start < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    sink.put(sum as u8);
    keep.push(chunk);
    cur.reads += 1;
    sink.put(cur.reads as u8);
}

pub struct Kept<'k> {
    pub last: &'k [u8],
}

// Quiet (R7, the hand-back stored through a pointer first): as `read_keep`,
// with `out.last = chunk` -- a store through `out: &mut Kept` -- in place of
// the `push`. No local of the body holds the loan after it; `*out` does.
pub fn read_store<'c, S: Sink>(sink: &mut S, cur: &'c mut Cursor, want: usize, out: &mut Kept<'c>) {
    let start = cur.pos.min(cur.data.len());
    let end = cur.data.len().min(start + want);
    let chunk: &[u8] = &cur.data[start..end];
    let mut sum = cur.sum;
    for &b in chunk {
        sum = sum.rotate_left(5) ^ u32::from(b);
    }
    cur.sum = sum;
    cur.pos = end;
    let padded = end - start < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    sink.put(sum as u8);
    out.last = chunk;
    cur.reads += 1;
    sink.put(cur.reads as u8);
}

// Quiet (R7, a shared loan and a later write through a `Box`): from `let
// start` to the first `put` the stretch only reads `cur`, so the inner fn
// would take `&Box<Cursor>` (or `&Cursor`) and hand back `chunk` tied to it by
// a shared loan -- under which `cur.reads += 1` while `chunk` lives is E0506.
// In this MIR that write goes through the raw pointer `ElaborateBoxDerefs`
// copies out of the `Box`, not through `cur` by name; it is a write through
// `cur` all the same. What follows `let chunk` is too short on its own, and
// so is what precedes it. (With `sink.put(cur.reads as u8)` in place of the
// increment the stretch is reported, and rightly: a read sits fine beside a
// shared loan.)
pub fn peek_boxed<S: Sink>(sink: &mut S, mut cur: Box<Cursor>, want: usize) -> Box<Cursor> {
    let start = cur.pos.min(cur.data.len());
    let end = cur.data.len().min(start + want);
    let chunk: &[u8] = &cur.data[start..end];
    let mut sum = cur.sum;
    for &b in chunk {
        sum = sum.rotate_left(5) ^ u32::from(b);
    }
    sink.put(sum as u8);
    cur.reads += 1;
    sink.put(chunk.len() as u8);
    cur
}

pub struct Src {
    pub buf: Vec<u8>,
    pub pos: usize,
}

pub struct Parser<'a> {
    pub src: &'a mut Src,
    pub depth: u32,
    pub sum: u32,
}

// Quiet (R7, an exclusive loan earned through a nested `&mut`; bun
// `JSXTag::parse`): from `let start` to the first `put` the stretch writes
// `p.src.pos` -- through the `&mut Src` behind `p`, which `Derefer` reaches by
// copying that `&mut` into a temporary and writing through the copy -- so the
// inner fn must take `p: &mut Parser`, and `tok`, borrowed through the same
// path, comes back tied to all of `*p` exclusively: the mere read `p.depth`
// while `tok` lives is E0503. Taking `p: &Parser` instead is E0594 inside the
// inner fn. A stretch starting after `let tok` would take `tok` beside `p`
// (R6).
pub fn next_token<S: Sink>(sink: &mut S, p: &mut Parser<'_>, want: usize) {
    let start = p.src.pos.min(p.src.buf.len());
    let end = p.src.buf.len().min(start + want);
    let tok: &[u8] = &p.src.buf[start..end];
    let mut sum = p.sum;
    for &b in tok {
        sum = sum.rotate_left(5) ^ u32::from(b);
    }
    p.src.pos = end;
    let padded = end - start < want;
    if padded {
        sum ^= 0xa5a5_a5a5;
    }
    sink.put(sum as u8);
    sink.put(p.depth as u8);
    sink.put(tok.len() as u8);
}

// Flagged, two lines shorter than the blocks allow (the checked-arithmetic
// tail is trimmed): from `let start` through `cur.reads += 1` nothing names
// `S`. In this MIR `cur.pos = end; cur.reads += 1` is one block -- the store,
// then the checked `(u32, bool)` pair and the assert on its overflow bit --
// with the store of the sum sharing the next block with `sink.put`, which
// does name `S`. Ending on that block the stretch would hand back "the `(u32,
// bool)` from `cur.reads += 1`" for the outer fn to finish the addition with,
// which no one would write; the search gives the block back whole instead,
// `cur.pos = end` with it, so the stretch ends at the `if`, takes `cur: &mut
// Cursor` and `want`, and yields `start`, `end` and `padded`.
pub fn refill<S: Sink>(sink: &mut S, cur: &mut Cursor, want: usize) {
    let start = cur.pos.min(cur.data.len());
    let avail = cur.data.len() - start;
    let take = avail.min(want).min(64);
    let end = start + take;
    let mut sum = cur.sum;
    let mut i = start;
    while i < end {
        sum = sum.rotate_left(5) ^ u32::from(cur.data[i]);
        i += 1;
    }
    cur.sum = sum;
    let padded = take < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    cur.pos = end;
    cur.reads += 1;
    sink.put((end - start) as u8);
    if padded {
        sink.put(0);
    }
}

// Flagged, the same stretch: two `+=` in a row are two such blocks, the
// second holding the store of the first's sum beside its own pair. The
// second goes back for handing back its pair; that leaves the first's pair
// crossing the edge, so the first goes back too, and the stretch again ends
// at the `if` and yields `start`, `end` and `padded`.
pub fn refill_twice<S: Sink>(sink: &mut S, cur: &mut Cursor, want: usize) {
    let start = cur.pos.min(cur.data.len());
    let avail = cur.data.len() - start;
    let take = avail.min(want).min(64);
    let end = start + take;
    let mut sum = cur.sum;
    let mut i = start;
    while i < end {
        sum = sum.rotate_left(5) ^ u32::from(cur.data[i]);
        i += 1;
    }
    cur.sum = sum;
    cur.pos = end;
    let padded = take < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    cur.sum += 3;
    cur.reads += 1;
    sink.put((end - start) as u8);
    if padded {
        sink.put(0);
    }
}

// Flagged, the same stretch: `let total = cur.reads + 1` copies `cur.reads`
// into a temporary before computing the pair, in the pair's block, and
// `total` itself is assigned in the next, beside `sink.put`. The block goes
// back whole, copy and all, rather than have the stretch yield "the `(u32,
// bool)` from `cur.reads + 1`".
pub fn refill_total<S: Sink>(sink: &mut S, cur: &mut Cursor, want: usize) -> u32 {
    let start = cur.pos.min(cur.data.len());
    let avail = cur.data.len() - start;
    let take = avail.min(want).min(64);
    let end = start + take;
    let mut sum = cur.sum;
    let mut i = start;
    while i < end {
        sum = sum.rotate_left(5) ^ u32::from(cur.data[i]);
        i += 1;
    }
    cur.sum = sum;
    cur.pos = end;
    let padded = take < want;
    if padded {
        cur.sum ^= 0xa5a5_a5a5;
    }
    let total = cur.reads + 1;
    sink.put((end - start) as u8);
    if padded {
        sink.put(0);
    }
    total
}

fn consume(text: String) -> u32 {
    text.len() as u32
}

// Quiet (R5, given away on one path, dropped to some effect on the other):
// between the two `put` calls nothing names `S` and the run is long enough,
// but it drops `guard` only when `acc` is even; when it is odd the fn holds
// the lock to its end and releases it there, behind the drop flag that `if`
// clears. An inner fn taking `guard` by value would release the lock on its
// own return, before the second `put`. Either side of the `if` alone is too
// short.
pub fn settle<S: Sink>(sink: &mut S, lock: &std::sync::Mutex<u32>, seed: u32) -> u32 {
    let guard = lock.lock().unwrap();
    let base = *guard;
    sink.put(base as u8);
    let mut acc = seed ^ base;
    acc = acc.rotate_left(5).wrapping_mul(31);
    acc ^= acc >> 7;
    acc = acc.wrapping_add(base | 1);
    if acc & 1 == 0 {
        drop(guard);
    }
    acc = acc.wrapping_mul(2654435761);
    acc ^= acc >> 13;
    acc = acc.rotate_left(9).wrapping_add(seed);
    sink.put(acc as u8);
    acc
}

// Quiet (R5, a drop flag among what the stretch would hand back): `text` is
// made only when `acc` is even and dropped at the fn's end behind the flag
// that records it, which the run between the two `put` calls sets. The flag
// is a `bool` nobody wrote and `text` a `String` that run may not have made;
// an inner fn can return neither. Made on both paths, `text` would be handed
// back like any other value. Either side of the `if` alone is too short.
pub fn label<S: Sink>(sink: &mut S, seed: u32) -> u32 {
    let text: String;
    sink.put(seed as u8);
    let mut acc = seed;
    acc = acc.rotate_left(5).wrapping_mul(31);
    acc ^= acc >> 7;
    acc = acc.wrapping_add(seed | 1);
    if acc & 1 == 0 {
        text = String::from("even");
        acc ^= text.len() as u32;
    }
    acc = acc.wrapping_mul(2654435761);
    acc ^= acc >> 13;
    acc = acc.rotate_left(9).wrapping_add(seed);
    sink.put(acc as u8);
    acc
}

// Flagged (R5, a drop flag among what the stretch would take): `text` is
// given away or kept before the `put`, and its drop at the fn's end tests the
// flag that says which. A stretch running on to the `return` would hold that
// test, take the flag, and drop a `text` that may be gone; the one reported
// stops short of it: `acc = acc.rotate_left(5)..` through `.rotate_right(seed
// & 7)`, taking `seed: u32` and `acc: u32` and yielding the `u32` returned.
pub fn unlabel<S: Sink>(sink: &mut S, seed: u32, keep: bool) -> u32 {
    let text = String::from("kept");
    let mut acc = seed;
    if !keep {
        acc ^= consume(text);
    }
    sink.put(acc as u8);
    acc = acc.rotate_left(5).wrapping_mul(31);
    acc ^= acc >> 7;
    acc = acc.wrapping_add(seed | 1);
    acc = acc.rotate_right(3).wrapping_sub(40503);
    acc ^= acc >> 11;
    acc = acc.wrapping_mul(2654435761);
    acc ^= acc >> 13;
    acc = acc.rotate_left(9).wrapping_add(seed);
    acc = acc.wrapping_sub(acc >> 5) | 1;
    acc ^= acc >> 3;
    acc.wrapping_mul(16777619).rotate_right(seed & 7)
}

// Flagged (R5, the flag itself never listed): `text` is made before the first
// `put` and given away inside the stretch when `acc` is even; all the fn does
// with it afterwards is drop it, behind the flag the stretch clears, and a
// `String`'s drop only frees memory, so the inner fn may take `text` by value
// and drop it itself. The stretch is `let mut acc = seed;` through the `}`
// after `acc ^= 0x9e3779b9;`, taking `seed: u32` and `text: String` and
// yielding `acc: u32` -- and no `bool`: the flag has no name and no source to
// give it.
pub fn handoff<S: Sink>(sink: &mut S, seed: u32) -> u32 {
    let text = String::from("payload");
    sink.put(seed as u8);
    let mut acc = seed;
    acc = acc.rotate_left(5).wrapping_mul(31);
    acc ^= acc >> 7;
    acc = acc.wrapping_add(seed | 1);
    if acc & 1 == 0 {
        acc ^= consume(text);
    }
    acc = acc.wrapping_mul(2654435761);
    acc = acc.rotate_left(9).wrapping_add(seed);
    if acc & 2 == 0 {
        acc ^= 0x9e3779b9;
    }
    sink.put(acc as u8);
    acc
}

pub trait Emit {
    fn emit(&mut self, bytes: &[u8]);
}

impl Emit for Count {
    fn emit(&mut self, bytes: &[u8]) {
        self.0 += bytes.len() as u32;
    }
}

impl Emit for Last {
    fn emit(&mut self, bytes: &[u8]) {
        if let Some(&b) = bytes.last() {
            self.0 = b;
        }
    }
}

// Quiet (R3, a hand-back that borrows a local the stretch makes; bun
// `send_data`'s `content_to_compress`, `readdir_*`'s `name_to_copy`): from
// `seed.wrapping_mul` through `&buf[..n]` nothing names `E` and the run is
// long enough, but what it hands the call after it is `view`, a borrow of
// `buf`, and `buf` is made inside the stretch: the inner fn would return a
// reference into its own frame. The part before `buf` is a handful of
// statements, so no shorter stretch qualifies and nothing is offered.
pub fn own_view<E: Emit>(out: &mut E, seed: u32) {
    let x = seed.wrapping_mul(0x9e37_79b9);
    let y = x.rotate_left(7) ^ seed;
    let z = y.swap_bytes().wrapping_add(x);
    let n = (z & 7) as usize;
    let buf = [
        x as u8,
        y as u8,
        z as u8,
        (x ^ y) as u8,
        (y & z) as u8,
        (x | z) as u8,
        !(x as u8),
        (x ^ y ^ z) as u8,
    ];
    let view = &buf[..n];
    out.emit(view);
}

// Quiet (R3, the address parked where it outlives the stretch): the same
// arithmetic and the same `buf`, but the borrow of it goes into `parts`, and
// `parts` is still read by the loop after the stretch. `Vec::push` is handed
// `&buf[..n]` beside `&mut parts` -- a place that can keep an address -- so
// whether `parts` is made inside the stretch (and handed back holding the
// borrow) or before it (and lent to the inner fn, which would push a borrow
// of its own local into it), `buf` would not live long enough. Nothing
// shorter qualifies, so nothing is offered.
pub fn parked_view<E: Emit>(out: &mut E, seed: u32) {
    let mut parts: Vec<&[u8]> = Vec::with_capacity(2);
    let x = seed.wrapping_mul(0x9e37_79b9);
    let y = x.rotate_left(7) ^ seed;
    let z = y.swap_bytes().wrapping_add(x);
    let n = (z & 7) as usize;
    let buf = [
        x as u8,
        y as u8,
        z as u8,
        (x ^ y) as u8,
        (y & z) as u8,
        (x | z) as u8,
        !(x as u8),
        (x ^ y ^ z) as u8,
    ];
    parts.push(&buf[..n]);
    parts.push(b";");
    for part in &parts {
        out.emit(part);
    }
}

fn stash<'a>(slot: std::rc::Rc<std::cell::Cell<Option<&'a [u8]>>>, view: &'a [u8]) {
    slot.set(Some(view));
}

// Quiet (R3, the address parked through a handle the call is given whole):
// `parked_view` again, but the place the borrow of `buf` is left in arrives
// as an `Rc` moved into `stash` -- nothing `&mut` about it, and the callee
// owns the `Rc`. What it does not own is the `Cell` the `Rc` points at:
// `slot` is the other handle, read after the stretch, so the inner fn (from
// `Rc::new` through `stash(..)`, handing back `slot`) would return a way to a
// borrow of its own `buf`. A pointer inside an argument that the argument
// does not own outright is a place like any `&mut`, and nothing is offered.
pub fn rc_parked<E: Emit>(out: &mut E, seed: u32) {
    let slot = std::rc::Rc::new(std::cell::Cell::new(None));
    let slot2 = std::rc::Rc::clone(&slot);
    let x = seed.wrapping_mul(0x9e37_79b9);
    let y = x.rotate_left(7) ^ seed;
    let z = y.swap_bytes().wrapping_add(x);
    let n = (z & 7) as usize;
    let buf = [
        x as u8,
        y as u8,
        z as u8,
        (x ^ y) as u8,
        (y & z) as u8,
        (x | z) as u8,
        !(x as u8),
        (x ^ y ^ z) as u8,
    ];
    stash(slot2, &buf[..n]);
    if let Some(v) = slot.get() {
        out.emit(v);
    }
}

pub struct Both<'a, 'b> {
    pub v: &'a [u8],
    pub p: &'b mut Vec<&'a [u8]>,
}

impl Both<'_, '_> {
    fn go(self) {
        self.p.push(self.v);
    }
}

// Quiet (R3, the carrier and the place in one argument): `parked_view` once
// more, with `&buf[..n]` and `&mut parts` put into one `Both` before the call
// that joins them, so `go` is handed a single argument and no rule about a
// second one sees the store. The struct literal is where the two meet, and
// that is refused as the call would have been: the stretch from
// `Vec::with_capacity` through `.go()` would hand back `parts` holding a
// borrow of the `buf` it made. Nothing shorter qualifies.
pub fn single_arg_park<E: Emit>(out: &mut E, seed: u32) {
    let mut parts: Vec<&[u8]> = Vec::with_capacity(2);
    let x = seed.wrapping_mul(0x9e37_79b9);
    let y = x.rotate_left(7) ^ seed;
    let z = y.swap_bytes().wrapping_add(x);
    let n = (z & 7) as usize;
    let buf = [
        x as u8,
        y as u8,
        z as u8,
        (x ^ y) as u8,
        (y & z) as u8,
        (x | z) as u8,
        !(x as u8),
        (x ^ y ^ z) as u8,
    ];
    Both {
        v: &buf[..n],
        p: &mut parts,
    }
    .go();
    parts.push(b";");
    for part in &parts {
        out.emit(part);
    }
}

fn push_pair<'a>(pair: (&'a [u8], &mut Vec<&'a [u8]>)) {
    pair.1.push(pair.0);
}

// Quiet (R3, the same through a tuple): the pair `(&buf[..n], &mut parts)`
// is the one argument, and building it is what is refused.
pub fn tuple_arg_park<E: Emit>(out: &mut E, seed: u32) {
    let mut parts: Vec<&[u8]> = Vec::with_capacity(2);
    let x = seed.wrapping_mul(0x9e37_79b9);
    let y = x.rotate_left(7) ^ seed;
    let z = y.swap_bytes().wrapping_add(x);
    let n = (z & 7) as usize;
    let buf = [
        x as u8,
        y as u8,
        z as u8,
        (x ^ y) as u8,
        (y & z) as u8,
        (x | z) as u8,
        !(x as u8),
        (x ^ y ^ z) as u8,
    ];
    push_pair((&buf[..n], &mut parts));
    parts.push(b";");
    for part in &parts {
        out.emit(part);
    }
}

fn advance(cur: &mut &mut [u8], bytes: &[u8]) {
    let (head, tail) = std::mem::take(cur).split_at_mut(bytes.len());
    head.copy_from_slice(bytes);
    *cur = tail;
}

// Flagged, for less than it first looks (R3 then R6 refuse the larger
// stretches; bun `print_json`, ConsoleObject.rs -- `iso_string_buf` /
// `cursor` / `out_buf`): from `[b' '; 32]` to the line before
// `out.emit(shown)` nothing names `E`. But that stretch hands on `shown`,
// which borrows `buf`, and `buf` is made inside it -- a fn cannot return a
// borrow of its own local. Start one line later and `buf` goes in by
// reference instead, beside `cursor`, which *is* a `&mut` into `buf`:
// `inner(&buf, cursor, ..)` does not borrow-check. What is left is the
// stretch from `start - len` on, where `cursor` is dead: it takes `buf` (by
// reference), `start` and `len`, trims, and hands back `end`, `begin` and
// the `&[u8]` that `shown` reborrows -- a borrow of something it was handed,
// which is fine. The fill before it is a few statements short of a stretch
// of its own.
pub fn trim_stamp<E: Emit>(out: &mut E, secs: u32, frac: u32) -> usize {
    let mut buf = [b' '; 32];
    let mut cursor = &mut buf[4..];
    let start = cursor.len();
    advance(&mut cursor, &[b'0' | secs as u8, b'.', frac as u8, b'0']);
    let len = cursor.len();
    let n = start - len;
    let text = &buf[..4 + n];
    let mut end = text.len();
    while end > 0 && text[end - 1] == b' ' {
        end -= 1;
    }
    let mut begin = 0;
    while begin < end && text[begin] == b' ' {
        begin += 1;
    }
    if end - begin > 2 && text[end - 1] == b'0' && text[end - 2] == b'0' {
        end -= 2;
    }
    let shown = &text[begin..end];
    out.emit(shown);
    end - begin
}

pub struct FrameHeader {
    pub length: u32,
    pub kind: u8,
    pub flags: u8,
    pub stream: u32,
}

impl FrameHeader {
    // Flagged (R3's call rule, narrowed by what the other argument can hold;
    // bun `FrameHeader::write`, h2_frame_parser.rs, 9 copies): everything up
    // to `out.emit(&buf)` fills a `[u8; 9]` from `&self` and hands `buf` back
    // by value. On the way `copy_from_slice` is handed `&mut buf[5..9]`
    // beside `&self.stream.to_be_bytes()` -- two addresses in one call, which
    // a rule counting address-holding operands would refuse -- but a `&[u8]`
    // is nowhere a callee could leave the first, so nothing of `buf` outlives
    // the stretch and it is offered: it takes `self: &FrameHeader` and yields
    // `buf: [u8; 9]`.
    pub fn write<E: Emit>(&self, out: &mut E) -> usize {
        let mut buf = [0u8; 9];
        buf[0] = (self.length >> 16) as u8;
        buf[1] = (self.length >> 8) as u8;
        buf[2] = self.length as u8;
        buf[3] = self.kind;
        buf[4] = self.flags;
        buf[5..9].copy_from_slice(&self.stream.to_be_bytes());
        out.emit(&buf);
        buf.len()
    }
}

pub struct Stats {
    pub dots: u32,
    pub longest: usize,
}

// Flagged, for less than it first looks (R3 then R6 refuse the larger
// stretches; the same bun `print_json` shape as `trim_stamp`, with the
// `write!`s it has there): from `[0u8; 40]` to the line before
// `out.emit(text)` nothing names `E`, but that stretch hands on `text`, a
// borrow of the `buf` it makes. One line later `buf` goes in by reference,
// beside `cursor`, a `&mut` into it; and each `write!` hands `cursor` to a
// call beside `Arguments` holding the addresses of the numbers it prints. What
// is left is the arithmetic from `let start` through `stats.dots = ..`: it
// takes `stats`, `secs`, `frac` and the `&mut [u8]` that `cursor` reborrows,
// and hands back `cursor`, `start`, the seven numbers the `write!`s print and
// the `u32` stored to `dots` (the store shares a block with the first
// `write!`).
pub fn print_stamp<E: Emit>(out: &mut E, stats: &mut Stats, secs: u64, frac: u32) {
    use std::io::Write as _;
    let mut buf = [0u8; 40];
    let mut cursor = &mut buf[..];
    let start = cursor.len();
    let days = secs / 86_400;
    let rem = secs - days * 86_400;
    let hours = (rem / 3_600) as u32;
    let minutes = ((rem / 60) % 60) as u32;
    let seconds = (rem % 60) as u32;
    let year_ish = 1970 + (days * 400 / 146_097) as u32;
    let day_ish = (days - u64::from(year_ish - 1970) * 365) as u32 % 366;
    let millis = frac / 1_000_000;
    let micros = (frac / 1_000) % 1_000;
    let packed = (hours << 12) | (minutes << 6) | seconds;
    stats.dots = stats.dots.wrapping_add(packed ^ day_ish ^ micros);
    let _ = write!(cursor, "{year_ish:04}-{day_ish:03}T{hours:02}:{minutes:02}:{seconds:02}");
    if frac != 0 {
        let _ = write!(cursor, ".{millis:03}{micros:03}");
    }
    let _ = write!(cursor, "Z");
    let n = start - cursor.len();
    let text = &buf[..n];
    stats.longest = stats.longest.max(n);
    out.emit(text);
}

pub struct Settings<M> {
    pub marker: M,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub gamma: u32,
    pub seed: u32,
    pub limit: u32,
    pub floor: u32,
    pub scale: u32,
}

// Flagged (many unnamed hand-backs are awkward, not a reason to refuse; bun
// `IntFormat::new`, the `PosixSpawnOptions` builders): the eight initializers
// are `u32` arithmetic on `base` and name nothing generic, and together they
// are long enough. What they make is values only the `Settings<M>` literal,
// which does name `M`, puts together -- so the inner fn returns them (an
// array, or a plain struct), which compiles and costs one call. The stretch
// is `width: ..` through `.rotate_left(19)` on the `scale` line -- the `^
// 0x58` and the move of `marker` share a block with the literal -- taking
// `base: u32` and yielding eight `u32`s, none of which the author named: the
// note names seven by the field they land in and the last, not yet a field's
// value, by its expression.
pub fn settings_for<M>(marker: M, base: u32) -> Settings<M> {
    Settings {
        width: base.wrapping_mul(3).rotate_left(2) ^ 0x51,
        height: base.wrapping_mul(5).rotate_left(3) ^ 0x52,
        depth: base.wrapping_mul(7).rotate_left(5) ^ 0x53,
        gamma: base.wrapping_mul(11).rotate_left(7) ^ 0x54,
        seed: base.wrapping_mul(13).rotate_left(11) ^ 0x55,
        limit: base.wrapping_mul(17).rotate_left(13) ^ 0x56,
        floor: base.wrapping_mul(19).rotate_left(17) ^ 0x57,
        scale: base.wrapping_mul(23).rotate_left(19) ^ 0x58,
        marker,
    }
}

pub struct Printer {
    pub col: u32,
    pub out: Vec<u8>,
    pub errors: u32,
}

impl Printer {
    fn fail(&mut self) -> u32 {
        self.errors += 1;
        self.errors
    }

    // Flagged, smaller than it looks (a temporary the author did not name is
    // a hand-back like any other; bun css `Printer::write_str`, 26 copies):
    // `s.as_ref()` before, and the drop of `s` on *each* of the two ways out,
    // depend on `S`, so the stretch can hold neither `return`: it is
    // `self.col = ..` through the `push` and the `len()` call, it takes
    // `self: &mut Printer` and `s: &[u8]`, and what it yields is one
    // expression temporary -- the `usize` from `self.out.len()` that the `if`
    // compares (`h` is dead by then). The split is `if inner(self, s) > 4096`.
    pub fn write_str<S: AsRef<[u8]>>(&mut self, s: S) -> Result<(), u32> {
        let s = s.as_ref();
        self.col = self.col.wrapping_add(s.len() as u32);
        let mut h = self.col;
        for &b in s {
            h = (h ^ u32::from(b)).wrapping_mul(0x01000193);
        }
        self.out.extend_from_slice(s);
        self.out.push((h & 0x7f) as u8);
        if self.out.len() > 4096 {
            return Err(self.fail());
        }
        Ok(())
    }
}

pub struct Ledger<T> {
    pub tag: T,
    pub counts: [u32; 6],
    pub total: u32,
}

// Flagged: no expression anywhere calls `drop`; the two `Ledger`s `main`
// builds run it when they go out of scope, one instantiation each. The reads
// through `&mut self` come first and the write to `self.total` last; the
// stretch between is `let mut sum = 0u32;` through the `let total = if ..`
// line, taking `counts: [u32; 6]` and `floor: u32` and yielding `total: u32`.
impl<T> Drop for Ledger<T> {
    fn drop(&mut self) {
        let counts = self.counts;
        let floor = self.total.min(60);
        let mut sum = 0u32;
        let mut peak = 0u32;
        let mut at = 0u32;
        let mut i = 0u32;
        for c in counts {
            sum = sum.wrapping_add(c);
            if c > peak {
                peak = c;
                at = i;
            }
            i += 1;
        }
        let spread = peak.saturating_sub(sum / 6).wrapping_add(floor);
        let total = if spread > at { sum ^ spread } else { sum.wrapping_mul(at + 1) };
        self.total = total;
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
            (acc ^ (src.len() as u32)).rotate_left(7)
        }
    };
}

make_summer!(macro_checksum);

pub const VERBOSE: bool = true;

pub struct Scope {
    pub tag: &'static str,
    pub on: bool,
}

impl Scope {
    pub fn visible(&self) -> bool {
        self.on
    }
    pub fn log(&self, args: std::fmt::Arguments<'_>) {
        if self.on {
            eprintln!("{args}");
        }
    }
}

pub static REQUESTS: Scope = Scope { tag: "req", on: true };

// A scoped logging macro: gated on a constant, branching before
// `format_args!` so each argument is evaluated once. One line of it is some
// fifty MIR statements, none of which names a parameter, all of which the
// macro wrote.
macro_rules! trace {
    ($scope:path, $fmt:expr $(, $arg:expr)* $(,)?) => {
        if VERBOSE && $scope.visible() {
            if $scope.tag.len() > 2 {
                $scope.log(format_args!(concat!("\x1b[2m[{}]\x1b[0m ", $fmt, "{}"), $scope.tag, $($arg,)* "\n"));
            } else {
                $scope.log(format_args!(concat!("[{}] ", $fmt, "{}"), $scope.tag, $($arg,)* "\n"));
            }
        }
    };
}

pub struct Conn<const TLS: bool> {
    pub id: u32,
    pub open: bool,
}

impl<const TLS: bool> Conn<TLS> {
    // Quiet (R1, macro mass): every other line reads through `&mut self`, a
    // `&mut Conn<TLS>`, so the one run of statements a stretch could be made
    // of is what `trace!` expands to -- well past the minimum, one way in and
    // one way out, taking nothing and yielding nothing -- and there is no
    // source in it to move. Only statements written by hand count toward
    // `generic-body-not-generic-min-statements`; these were written by the
    // macro.
    pub fn close(&mut self) -> u32 {
        trace!(REQUESTS, "close");
        if !self.open {
            return self.id;
        }
        self.open = false;
        self.id.rotate_left(if TLS { 3 } else { 5 })
    }
}

// Quiet: the same shape as `fold_block`, at two array lengths, but both uses
// are `const` initializers, evaluated at compile time and never compiled
// into the binary.
pub const fn const_fold<const N: usize>(block: [u8; N]) -> u32 {
    let bytes = block.as_slice();
    let mut lo = 1u32;
    let mut hi = 0u32;
    let mut i = 0usize;
    while i < bytes.len() {
        lo = (lo + bytes[i] as u32) % 65521;
        hi = (hi + lo) % 65521;
        i += 1;
    }
    let folded = (hi << 16) | lo;
    if folded % 2 == 0 { folded / 2 } else { folded.wrapping_mul(3) + 1 }
}

pub const FOLD_TWO: u32 = const_fold([1u8, 2]);
pub const FOLD_THREE: u32 = const_fold([1u8, 2, 3]);

fn evens(src: &[u8]) -> impl Iterator<Item = u8> + '_ {
    src.iter().copied().filter(|b| b & 1 == 0)
}

// Quiet (a value the stretch would take has a type no signature can spell):
// `let mut acc = 17u32;` through `.rotate_left(7)` is the same in both copies
// and long enough, but it reads `it`, made before `sink.put(0)` and so handed
// in -- and `it` is the `impl Iterator` of `evens`, which the body sees as
// `Filter<Copied<Iter<u8>>, {closure}>`: free of `S`, yet nothing an inner fn
// could declare a parameter as. What lies either side of `it.next()` is under
// the threshold on its own. (Made after `sink.put(0)` instead, `it` would be
// the inner fn's own local and the stretch would take `src`.)
pub fn fold_evens<S: Sink>(sink: &mut S, src: &[u8]) -> u32 {
    let mut it = evens(src);
    sink.put(0);
    let mut acc = 17u32;
    let mut run = 0u32;
    while let Some(b) = it.next() {
        let v = u32::from(b);
        acc = acc.wrapping_mul(31).wrapping_add(v);
        run = if v & 2 == 0 { run + 1 } else { 0 };
    }
    acc ^= run << 3;
    acc = (acc ^ run).rotate_left(7);
    sink.put(acc as u8);
    acc
}

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
    let _ = tally(Ann) + tally(Bob) + tally(Cy);
    let _ = wide_sum::<true>(b"wide") + wide_sum::<false>(b"narrow");
    let _ = arm_sum::<true>(b"this") + arm_sum::<false>(b"that");
    let _ = encode_arm::<0>(b"p") + encode_arm::<1>(b"h") + encode_arm::<2>(b"s");
    let _ = encode_arm::<3>(b"x");
    let _ = encode_as::<0>(b"ab") ^ encode_as::<1>(b"cd") ^ encode_as::<2>(b"ef");
    let _ = encode_as::<3>(b"gh");
    let _ = stamp_tag::<0>(b"abcd")[0] ^ stamp_tag::<1>(b"cdef")[0];
    let _ = stamp_at::<2>(b"abcd")[0] ^ stamp_at::<3>(b"cdef")[0];
    let mut count = Count(0);
    let mut last = Last(0);
    let _ = drain(&mut count, b"abc") + drain(&mut last, b"de");
    let _ = strict_sum("s") + strict_sum([1u8, 0]);
    let _ = lattice("l") + lattice(vec![3u8]);
    let _ = render(&mut count, 3) + render(&mut last, 4);
    let _ = scan(b"up", |b| count.0 += u32::from(b));
    let _ = scan(b"down", |b| last.0 = b);
    let _ = stepped::<true>(5) + stepped::<false>(6);
    let narrow = Wrap { inner: 1u8, len: 2, cap: 8 };
    let broad = Wrap { inner: "w", len: 3, cap: 9 };
    let _ = narrow.slack() + broad.slack();
    let _ = mix("m") + mix([0u8; 5]);
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
    let mut rec = Record { name: b"rec".to_vec(), count: 0, total: 0, sent: 0, flags: 0 };
    recount(&mut count, &mut rec, 2);
    recount(&mut last, &mut rec, 3);
    restamp(&mut count, &mut rec, 4);
    restamp(&mut last, &mut rec, 5);
    let mut cur = Cursor { data: vec![1, 2, 3], pos: 0, reads: 0, sum: 0 };
    read_into(&mut count, &mut cur, 2);
    read_into(&mut last, &mut cur, 9);
    {
        let mut v: Vec<u8> = Vec::new();
        let mut k = 0u32;
        read_scanned(&mut v, &mut cur, 2);
        read_scanned(&mut k, &mut cur, 9);
        last.0 ^= v.len() as u8 ^ k as u8;
    }
    {
        let mut a = Cursor { data: vec![1, 2, 3], pos: 0, reads: 0, sum: 0 };
        let mut b = Cursor { data: vec![4, 5], pos: 0, reads: 0, sum: 0 };
        let mut keep: Vec<&[u8]> = Vec::new();
        read_keep(&mut count, &mut a, 2, &mut keep);
        read_keep(&mut last, &mut b, 9, &mut keep);
        last.0 ^= keep.len() as u8;
    }
    {
        let mut a = Cursor { data: vec![1, 2, 3], pos: 0, reads: 0, sum: 0 };
        let mut b = Cursor { data: vec![4, 5], pos: 0, reads: 0, sum: 0 };
        let mut kept = Kept { last: &[] };
        read_store(&mut count, &mut a, 2, &mut kept);
        read_store(&mut last, &mut b, 9, &mut kept);
        last.0 ^= kept.last.len() as u8;
    }
    let boxed = Box::new(Cursor { data: vec![1, 2, 3], pos: 0, reads: 0, sum: 0 });
    let boxed = peek_boxed(&mut count, boxed, 2);
    let _ = peek_boxed(&mut last, boxed, 9);
    let mut src = Src { buf: vec![1, 2, 3], pos: 0 };
    let mut parser = Parser { src: &mut src, depth: 1, sum: 0 };
    next_token(&mut count, &mut parser, 2);
    next_token(&mut last, &mut parser, 9);
    refill(&mut count, &mut cur, 2);
    refill(&mut last, &mut cur, 9);
    refill_twice(&mut count, &mut cur, 2);
    refill_twice(&mut last, &mut cur, 9);
    let _ = refill_total(&mut count, &mut cur, 2) + refill_total(&mut last, &mut cur, 9);
    let lock = std::sync::Mutex::new(5u32);
    let _ = settle(&mut count, &lock, 1) + settle(&mut last, &lock, 2);
    let _ = label(&mut count, 3) + label(&mut last, 4);
    let _ = unlabel(&mut count, 5, true) + unlabel(&mut last, 6, false);
    let _ = handoff(&mut count, 7) + handoff(&mut last, 8);
    own_view(&mut count, 3);
    own_view(&mut last, 5);
    parked_view(&mut count, 3);
    parked_view(&mut last, 5);
    rc_parked(&mut count, 3);
    rc_parked(&mut last, 5);
    single_arg_park(&mut count, 3);
    single_arg_park(&mut last, 5);
    tuple_arg_park(&mut count, 3);
    tuple_arg_park(&mut last, 5);
    let _ = trim_stamp(&mut count, 7, 9) + trim_stamp(&mut last, 1, 0);
    let header = FrameHeader { length: 3, kind: 1, flags: 0, stream: 9 };
    let _ = header.write(&mut count) + header.write(&mut last);
    let mut stats = Stats { dots: 0, longest: 0 };
    print_stamp(&mut count, &mut stats, 12, 5);
    print_stamp(&mut last, &mut stats, 3, 0);
    let _ = settings_for((), 3).width + settings_for(7u8, 5).height + settings_for("m", 9).depth;
    let mut printer = Printer { col: 0, out: Vec::new(), errors: 0 };
    let _ = printer.write_str("abc");
    let _ = printer.write_str(b"abc");
    let _ = printer.write_str(vec![1u8, 2]);
    let _small_ledger = Ledger { tag: 1u8, counts: [1, 2, 3, 4, 5, 6], total: 0 };
    let _wide_ledger = Ledger { tag: "wide", counts: [6, 5, 4, 3, 2, 1], total: 0 };
    let _ = macro_checksum("m");
    let _ = macro_checksum([0u8; 3]);
    let mut plain = Conn::<false> { id: 3, open: true };
    let mut secure = Conn::<true> { id: 5, open: true };
    let _ = plain.close() + secure.close();
    let _ = FOLD_TWO + FOLD_THREE;
    let _ = fold_evens(&mut count, b"even") + fold_evens(&mut last, b"odd");
}
