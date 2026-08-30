//! MIR machinery shared by the body-level analyses, with no opinion about
//! what any of it means for a lint.
//!
//! It answers six questions about a body:
//!
//! * which MIR to read at all (`mir_for`: the pre-optimization body, so `?`
//!   is still a `Try::branch` call and every aggregate is intact);
//! * what a place names, as an [`Atom`] -- a local plus its leading field
//!   path -- and how faithfully (`place_info`, [`Exactness`]);
//! * which branches a block is control-dependent on (`build_cfg`,
//!   `post_dominators`, `control_deps`), over a CFG in which blocks that
//!   cannot reach a `return` do not exist, so nothing is "decided" by an
//!   `assert!`;
//! * what a branch switches on (`switch_operand_atoms`);
//! * which panics the body reaches without calling anything (`assert_panics`);
//! * how the blocks connect when nothing is pruned ([`FlowGraph`]): each
//!   block's normal successors and predecessors, a virtual EXIT that every
//!   `return` leads to and no panic does, dominators, post-dominators, which
//!   blocks a branch decides, and whether one block can reach another -- the
//!   substrate for carving a single-entry, single-exit run of blocks out of a
//!   body.
//!
//! What the answers mean is the caller's business: `ctor_flow` combines them
//! into "does a failure exit depend on a stored field", `unchecked_input_len`
//! and `variant_flow` use only `mir_for` and trace the body their own way,
//! `forbidden_reach` turns `assert_panics` into call-graph edges, and
//! `generic_body_not_generic` searches a `FlowGraph` for the stretch of a
//! generic fn that could be a non-generic fn of its own.

use std::collections::{HashSet, VecDeque};

use rustc_data_structures::graph::dominators::{Dominators, dominators};
use rustc_data_structures::graph::{DirectedGraph, Predecessors, StartNode, Successors};
use rustc_hir::LangItem;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::{
    AssertKind, BasicBlock, BasicBlockData, Body, Local, Operand, Place, ProjectionElem,
    START_BLOCK, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

// ── MIR access ───────────────────────────────────────────────────────────────

pub(crate) fn mir_for<'tcx>(tcx: TyCtxt<'tcx>, def: LocalDefId) -> Option<MirRef<'tcx>> {
    if !tcx.def_kind(def).is_fn_like() || !tcx.is_mir_available(def.to_def_id()) {
        return None;
    }
    // The pre-optimization body keeps `?` as `Try::branch` calls and every
    // aggregate intact regardless of the build's opt level. It is stolen once
    // `optimized_mir` runs, which nothing before codegen asks for; fall back
    // if some other driver did.
    let steal = tcx.mir_drops_elaborated_and_const_checked(def);
    if steal.is_stolen() {
        Some(MirRef::Opt(tcx.optimized_mir(def.to_def_id())))
    } else {
        Some(MirRef::Steal(steal.borrow()))
    }
}

pub(crate) enum MirRef<'tcx> {
    Steal(rustc_data_structures::sync::MappedReadGuard<'tcx, Body<'tcx>>),
    Opt(&'tcx Body<'tcx>),
}

impl<'tcx> std::ops::Deref for MirRef<'tcx> {
    type Target = Body<'tcx>;
    fn deref(&self) -> &Body<'tcx> {
        match self {
            MirRef::Steal(g) => g,
            MirRef::Opt(b) => b,
        }
    }
}

// ── panics with no call ──────────────────────────────────────────────────────

/// The lang item rustc calls when `kind`'s assertion fires.
///
/// [`AssertKind::panic_function`] is this same mapping and is what codegen
/// uses, so delegating to it keeps every kind in step with rustc as the enum
/// grows. It `bug!`s on exactly two kinds, and deliberately: their panics take
/// runtime arguments (the length and the index; the required and found
/// alignment), so codegen names those lang items itself instead of asking.
/// Naming them here is what makes this total, and calling `panic_function`
/// bare would ICE on the commonest kind of all.
fn assert_panic_lang_item(kind: &AssertKind<Operand<'_>>) -> LangItem {
    match kind {
        AssertKind::BoundsCheck { .. } => LangItem::PanicBoundsCheck,
        AssertKind::MisalignedPointerDereference { .. } => {
            LangItem::PanicMisalignedPointerDereference
        }
        other => other.panic_function(),
    }
}

/// Every panic `body` reaches through an `Assert` terminator rather than a
/// call, as the lang item it invokes and the span that provoked it.
///
/// These are the panics rustc lowers during MIR building -- a bounds check, an
/// arithmetic overflow, a division or remainder by zero -- and there is no
/// call in HIR for any of them, at any spelling, so a walk over expressions
/// cannot see them however it is written.
///
/// **Not every one of these is present in every build.** The overflow kinds
/// exist only where `-C overflow-checks` is on (the debug default, and what
/// `#[rustc_inherit_overflow_checks]` propagates); division and remainder by
/// zero and the bounds check are emitted unconditionally. So an absent
/// overflow assert means the profile did not ask for one, not that the
/// arithmetic cannot overflow.
pub(crate) fn assert_panics<'a>(body: &'a Body<'_>) -> impl Iterator<Item = (LangItem, Span)> + 'a {
    body.basic_blocks.iter().filter_map(|data| {
        let term = data.terminator.as_ref()?;
        let TerminatorKind::Assert { msg, .. } = &term.kind else {
            return None;
        };
        Some((assert_panic_lang_item(msg), term.source_info.span))
    })
}

// ── places as atoms ──────────────────────────────────────────────────────────

/// A local plus the leading run of field projections: `_3.1.0`. Anything past
/// the first deref/index/downcast is folded into the prefix before it.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Atom {
    pub(crate) local: Local,
    pub(crate) path: Vec<u32>,
}

/// Path component for "some element of": indexing conflates positions but
/// keeps the container distinct from its elements, so a length check on a
/// slice is not a check on the element later stored out of it.
pub(crate) const ANY_ELEM: u32 = u32::MAX;

impl Atom {
    pub(crate) fn whole(local: Local) -> Self {
        Atom {
            local,
            path: Vec::new(),
        }
    }
    pub(crate) fn extended(&self, tail: &[u32]) -> Self {
        // `node = node.next` in a loop composes without bound; past a few
        // levels the distinction stops mattering, so the path saturates.
        const MAX_PATH: usize = 6;
        let mut path = self.path.clone();
        let room = MAX_PATH.saturating_sub(path.len());
        path.extend_from_slice(&tail[..tail.len().min(room)]);
        Atom {
            local: self.local,
            path,
        }
    }
    pub(crate) fn overlaps(&self, other: &Atom) -> bool {
        self.local == other.local && self.path.iter().zip(&other.path).all(|(a, b)| a == b)
    }
    /// `self` (a decision atom) reads `stored` or a part of it. The reverse,
    /// a decision on the whole of something only part of which is stored
    /// (`lexer.next()?` then `log: lexer.log`), is not evidence about the part.
    pub(crate) fn inspects(&self, stored: &Atom) -> bool {
        self.local == stored.local && self.path.starts_with(&stored.path)
    }
}

/// How well `PlaceInfo::atom` names what the projection actually read.
///
/// A `Downcast` is absorbing: it ends the exact field path and no later
/// projection puts it back, so "payload" and "exact" cannot hold at once and
/// every caller below branches payload-first. Three states, not four.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exactness {
    /// Pure field access: the atom is exact.
    Exact,
    /// A `Downcast` appeared: this reads a variant payload of the atom.
    VariantPayload,
    /// Some other projection: the atom names more than was read.
    Inexact,
}

impl Exactness {
    /// A projection this does not model. It can only lose precision, and it
    /// cannot un-see a `Downcast`.
    fn blur(self) -> Self {
        match self {
            Exactness::Exact | Exactness::Inexact => Exactness::Inexact,
            Exactness::VariantPayload => Exactness::VariantPayload,
        }
    }
}

pub(crate) struct PlaceInfo {
    pub(crate) atom: Atom,
    pub(crate) exactness: Exactness,
    pub(crate) index_locals: Vec<Local>,
}

pub(crate) fn place_info(place: Place<'_>) -> PlaceInfo {
    let mut path = Vec::new();
    let mut exactness = Exactness::Exact;
    let mut index_locals = Vec::new();
    for elem in place.projection.iter() {
        match elem {
            ProjectionElem::Field(f, _) if exactness == Exactness::Exact => path.push(f.as_u32()),
            ProjectionElem::Field(..) => {}
            // `(*r).f`: the reference is the value for slicing purposes, so
            // a deref neither ends the field path nor makes it inexact.
            ProjectionElem::Deref => {}
            ProjectionElem::Downcast(..) => exactness = Exactness::VariantPayload,
            ProjectionElem::Index(v) => {
                index_locals.push(v);
                if exactness == Exactness::Exact {
                    path.push(ANY_ELEM);
                }
            }
            ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. }
                if exactness == Exactness::Exact =>
            {
                path.push(ANY_ELEM);
            }
            _ => exactness = exactness.blur(),
        }
    }
    PlaceInfo {
        atom: Atom {
            local: place.local,
            path,
        },
        exactness,
        index_locals,
    }
}

// ── control dependence ───────────────────────────────────────────────────────

/// The body's CFG over normal (non-unwind) edges, with every block that
/// cannot reach a `return` pruned: a panic, abort or `unreachable!()` is
/// "does not happen" here, otherwise everything after `assert!(x)` would be
/// control-dependent on `x`.
pub(crate) struct Cfg {
    succs: IndexVec<BasicBlock, Vec<BasicBlock>>,
    /// The virtual exit node, one past the last block.
    exit: BasicBlock,
}

/// The block's terminator hands control back to the caller: a `return`, or
/// the `become` that is a call and a return in one.
fn leaves_fn(data: &BasicBlockData<'_>) -> bool {
    matches!(
        data.terminator.as_ref().map(|t| &t.kind),
        Some(TerminatorKind::Return | TerminatorKind::TailCall { .. })
    )
}

/// Normal (non-unwind) successors of a non-cleanup block; none otherwise.
fn raw_successors(body: &Body<'_>, data: &BasicBlockData<'_>) -> Vec<BasicBlock> {
    let Some(term) = data.terminator.as_ref().filter(|_| !data.is_cleanup) else {
        return Vec::new();
    };
    // Unwind targets are always cleanup blocks, so this leaves the normal edges.
    let mut v: Vec<_> = term
        .successors()
        .filter(|b| !body.basic_blocks[*b].is_cleanup)
        .collect();
    v.sort();
    v.dedup();
    v
}

pub(crate) fn build_cfg(body: &Body<'_>) -> Cfg {
    let exit = BasicBlock::from_usize(body.basic_blocks.len());
    let raw: IndexVec<BasicBlock, Vec<BasicBlock>> = body
        .basic_blocks
        .iter()
        .map(|data| raw_successors(body, data))
        .collect();
    let mut can_return: IndexVec<BasicBlock, bool> =
        body.basic_blocks.iter().map(leaves_fn).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in raw.indices() {
            if !can_return[b] && raw[b].iter().any(|s| can_return[*s]) {
                can_return[b] = true;
                changed = true;
            }
        }
    }
    let succs = raw
        .iter()
        .map(|r| {
            let live: Vec<BasicBlock> = r.iter().copied().filter(|s| can_return[*s]).collect();
            if live.is_empty() { vec![exit] } else { live }
        })
        .collect();
    Cfg { succs, exit }
}

type Bits = DenseBitSet<BasicBlock>;
/// Post-dominator sets over blocks plus the virtual exit.
pub(crate) type Pdoms = IndexVec<BasicBlock, Bits>;

pub(crate) fn post_dominators(cfg: &Cfg) -> Pdoms {
    let size = cfg.exit.as_usize() + 1;
    let mut pdom = IndexVec::from_elem_n(Bits::new_filled(size), size);
    pdom[cfg.exit] = Bits::new_empty(size);
    pdom[cfg.exit].insert(cfg.exit);
    let mut changed = true;
    while changed {
        changed = false;
        for (b, succs) in cfg.succs.iter_enumerated() {
            let mut acc = Bits::new_filled(size);
            for &s in succs {
                acc.intersect(&pdom[s]);
            }
            acc.insert(b);
            if acc != pdom[b] {
                pdom[b] = acc;
                changed = true;
            }
        }
    }
    pdom
}

/// Branch blocks that `t` is directly control-dependent on, in block order:
/// `t` does not post-dominate them but does post-dominate one of their
/// successors, so their outcome sends control to `t` or away from it. A
/// branch that only decides whether one of those is reached (an early
/// `return Ok` guard in front of it) is not among them; `control_deps` has
/// those too.
pub(crate) fn direct_control_deps(cfg: &Cfg, pdom: &Pdoms, t: BasicBlock) -> Vec<BasicBlock> {
    cfg.succs
        .iter_enumerated()
        .filter(|(a, succs)| {
            let strictly = pdom[*a].contains(t) && *a != t;
            succs.len() >= 2 && !strictly && succs.iter().any(|s| pdom[*s].contains(t))
        })
        .map(|(a, _)| a)
        .collect()
}

/// Branch blocks that `target` is transitively control-dependent on,
/// innermost first.
pub(crate) fn control_deps(cfg: &Cfg, pdom: &Pdoms, target: BasicBlock) -> Vec<BasicBlock> {
    let mut seen = HashSet::new();
    let mut q = VecDeque::from([target]);
    let mut deps = Vec::new();
    while let Some(t) = q.pop_front() {
        for a in direct_control_deps(cfg, pdom, t) {
            if seen.insert(a) {
                deps.push(a);
                q.push_back(a);
            }
        }
    }
    deps
}

pub(crate) fn switch_operand_atoms(body: &Body<'_>, bb: BasicBlock) -> Vec<Atom> {
    match &body.basic_blocks[bb].terminator().kind {
        TerminatorKind::SwitchInt { discr, .. } => discr
            .place()
            .map(|p| {
                let info = place_info(p);
                let mut v: Vec<Atom> = info.index_locals.into_iter().map(Atom::whole).collect();
                v.push(info.atom);
                v
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

// ── the unpruned flow graph ──────────────────────────────────────────────────
//
// `Cfg` is the graph for asking what decides whether a block runs, and there
// a block that cannot reach a `return` is best treated as not existing. A
// caller that wants to lift a run of blocks out of the body whole needs the
// opposite reading of the same edges: a panicking arm inside the run is code
// that moves with it, so it stays in the graph, with its predecessors, to be
// vetted like any other block. What it must not be is a way *out* of the run.
// So here a block that returns has the one successor EXIT, and a block that
// diverges -- a call that never comes back, an `abort`, an `Unreachable` after
// real statements -- has none: control that enters it leaves the fn from
// there, which constrains nothing about where the run hands control back and
// post-dominates nothing. It follows that a block from which no path returns
// has no post-dominator at all, and a caller walking `ipdom_chain` from it
// gets an empty chain. That loses the inside of a panic arm as a place to
// start a region and nothing else; wiring the diverging blocks to EXIT instead
// would make EXIT the nearest post-dominator of every block with a `panic!`
// under it, and the chain would step over every region smaller than the rest
// of the fn.
//
// Two kinds of block are left out of the graph altogether, edges and all.
// Unwind cleanup, as everywhere in this module: a cleanup block only ever
// leads to more cleanup, so dropping them changes no dominance between the
// blocks that remain. And the empty `unreachable` block: every `match` on an
// enum has an `otherwise` edge to one, and `SimplifyCfg` has already folded
// all of a body's into a single shared block, so keeping it would hand every
// `match` in the body a common successor whose other predecessors lie outside
// any region one of them heads. It holds no code, and the edge to it is, like
// an unwind edge, a way control provably does not go.
//
// Dominators and post-dominators both come from rustc's Lengauer-Tarjan
// routine, run forwards from the entry block and backwards from EXIT over the
// same pair of edge lists, so building the graph costs a few passes over the
// edges and every dominance query after that is constant-time; `reaches` is
// one breadth-first walk per call. Nothing here is quadratic in blocks, which
// the bit-set fixpoint `post_dominators` runs over `Cfg` is -- harmless at the
// sizes `ctor_flow` reads, and not something to run on a body of five
// thousand blocks.

/// The body's CFG over normal edges with nothing pruned, plus a virtual EXIT
/// node one past the last block. The note above says what is a node and what
/// is an edge; in short, a returning block's only successor is EXIT, a
/// diverging block has none, and unwind cleanup and the shared empty
/// `unreachable` block are not in the graph.
pub(crate) struct FlowGraph {
    /// Indexed by block and by EXIT. A block not in the graph has no edges
    /// either way.
    succs: IndexVec<BasicBlock, Vec<BasicBlock>>,
    /// `succs` inverted, so `preds[exit]` is the returning blocks.
    preds: IndexVec<BasicBlock, Vec<BasicBlock>>,
    /// Blocks on the normal path, and EXIT.
    nodes: Bits,
    exit: BasicBlock,
    dom: Dominators<BasicBlock>,
    /// Dominators of the same edges reversed and rooted at EXIT, which is to
    /// say post-dominators.
    pdom: Dominators<BasicBlock>,
}

/// One direction of a [`FlowGraph`]'s edges and a root to walk them from:
/// the shape rustc's dominator routine asks for. Forwards from the entry
/// block it yields dominators; the same two lists swapped and rooted at EXIT
/// yield post-dominators.
struct Edges<'g> {
    out: &'g IndexSlice<BasicBlock, Vec<BasicBlock>>,
    into: &'g IndexSlice<BasicBlock, Vec<BasicBlock>>,
    root: BasicBlock,
}

impl DirectedGraph for Edges<'_> {
    type Node = BasicBlock;
    fn num_nodes(&self) -> usize {
        self.out.len()
    }
}

impl StartNode for Edges<'_> {
    fn start_node(&self) -> BasicBlock {
        self.root
    }
}

impl Successors for Edges<'_> {
    fn successors(&self, node: BasicBlock) -> impl Iterator<Item = BasicBlock> {
        self.out[node].iter().copied()
    }
}

impl Predecessors for Edges<'_> {
    fn predecessors(&self, node: BasicBlock) -> impl Iterator<Item = BasicBlock> {
        self.into[node].iter().copied()
    }
}

impl FlowGraph {
    pub(crate) fn new(body: &Body<'_>) -> Self {
        let exit = BasicBlock::from_usize(body.basic_blocks.len());
        let size = exit.as_usize() + 1;
        let mut nodes = Bits::new_empty(size);
        for (b, data) in body.basic_blocks.iter_enumerated() {
            let shared_unreachable = data.terminator.is_some() && data.is_empty_unreachable();
            if !data.is_cleanup && !shared_unreachable {
                nodes.insert(b);
            }
        }
        nodes.insert(exit);
        let mut succs: IndexVec<BasicBlock, Vec<BasicBlock>> =
            IndexVec::from_elem_n(Vec::new(), size);
        let mut preds: IndexVec<BasicBlock, Vec<BasicBlock>> =
            IndexVec::from_elem_n(Vec::new(), size);
        for (b, data) in body.basic_blocks.iter_enumerated() {
            if !nodes.contains(b) {
                continue;
            }
            let out = &mut succs[b];
            if leaves_fn(data) {
                out.push(exit);
            } else {
                out.extend(
                    raw_successors(body, data)
                        .into_iter()
                        .filter(|s| nodes.contains(*s)),
                );
            }
            for &s in out.iter() {
                preds[s].push(b);
            }
        }
        let dom = dominators(&Edges {
            out: &succs,
            into: &preds,
            root: START_BLOCK,
        });
        let pdom = dominators(&Edges {
            out: &preds,
            into: &succs,
            root: exit,
        });
        FlowGraph {
            succs,
            preds,
            nodes,
            exit,
            dom,
            pdom,
        }
    }

    /// The virtual node every `return` leads to, one past the last block. It
    /// indexes nothing in the body.
    pub(crate) fn exit(&self) -> BasicBlock {
        self.exit
    }

    /// `b` is a node: a block on the normal path (not unwind cleanup, not
    /// the shared empty `unreachable`), or EXIT.
    pub(crate) fn contains(&self, b: BasicBlock) -> bool {
        self.nodes.contains(b)
    }

    /// Where control can go next from `b` without unwinding: blocks, or EXIT
    /// alone when `b` returns. Empty for a diverging block, for EXIT, and for
    /// a block not in the graph.
    pub(crate) fn succs(&self, b: BasicBlock) -> &[BasicBlock] {
        &self.succs[b]
    }

    /// The nodes `b` is a successor of; for EXIT, the returning blocks.
    pub(crate) fn preds(&self, b: BasicBlock) -> &[BasicBlock] {
        &self.preds[b]
    }

    /// `b`'s nearest strict dominator; `None` for the entry block and for a
    /// node no path from it reaches.
    pub(crate) fn idom(&self, b: BasicBlock) -> Option<BasicBlock> {
        if self.nodes.contains(b) {
            self.dom.immediate_dominator(b)
        } else {
            None
        }
    }

    /// Some path of normal edges leads from `b` to a `return`.
    pub(crate) fn can_return(&self, b: BasicBlock) -> bool {
        self.nodes.contains(b) && self.pdom.is_reachable(b)
    }

    /// `b`'s nearest strict post-dominator: the first node every returning
    /// path from `b` meets again. `None` for EXIT, for a block from which no
    /// path returns, and for a block not in the graph.
    pub(crate) fn ipdom(&self, b: BasicBlock) -> Option<BasicBlock> {
        if self.nodes.contains(b) {
            self.pdom.immediate_dominator(b)
        } else {
            None
        }
    }

    /// `b`'s strict post-dominators nearest first, ending with EXIT; empty
    /// exactly when `ipdom(b)` is `None`. At most one item per node.
    pub(crate) fn ipdom_chain(&self, b: BasicBlock) -> impl Iterator<Item = BasicBlock> + '_ {
        std::iter::successors(self.ipdom(b), |&x| self.ipdom(x))
    }

    /// The blocks whose running `b`'s terminator decides: those directly
    /// control-dependent on `b`, which is every node from each successor of
    /// `b` up the post-dominator tree to, and short of, `b`'s own nearest
    /// post-dominator -- the arm a branch picks runs until the arms meet
    /// again, and where they meet runs either way. In block order, each once;
    /// EXIT is never among them. Empty for a block with one successor (that
    /// successor is where its "arms" meet) and for a node not in the graph. A
    /// successor from which no path returns is decided by `b` and, having no
    /// post-dominator to climb to, ends its walk there; when `b` itself has
    /// none, each arm is climbed as far as it goes. `b` is among its own when
    /// an arm leads back round to it (a loop's test decides whether the test
    /// runs again). What a block decided by `b` decides in turn is not
    /// included: that closure is `control_deps`' reading over `Cfg`, and
    /// here the caller's to take.
    pub(crate) fn decides(&self, b: BasicBlock) -> Vec<BasicBlock> {
        let succs = self.succs(b);
        if succs.len() < 2 {
            return Vec::new();
        }
        let join = self.ipdom(b);
        let mut decided = Bits::new_empty(self.succs.len());
        for &s in succs {
            let mut node = Some(s);
            while let Some(n) = node
                && node != join
                && n != self.exit
                && decided.insert(n)
            {
                node = self.ipdom(n);
            }
        }
        decided.iter().collect()
    }

    /// Some path of normal edges, possibly empty, leads from `from` to `to`
    /// (either may be EXIT). One breadth-first walk, so a call costs at most a
    /// pass over the edges reachable from `from`.
    pub(crate) fn reaches(&self, from: BasicBlock, to: BasicBlock) -> bool {
        let mut seen = Bits::new_empty(self.succs.len());
        let mut queue = VecDeque::from([from]);
        seen.insert(from);
        while let Some(b) = queue.pop_front() {
            if b == to {
                return true;
            }
            for &s in &self.succs[b] {
                if seen.insert(s) {
                    queue.push_back(s);
                }
            }
        }
        false
    }
}
