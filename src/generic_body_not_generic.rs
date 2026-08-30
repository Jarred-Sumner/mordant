//! `generic_body_not_generic`: a generic function this crate stamps out at
//! several concrete argument sets, with a stretch of its body that is the
//! same code in every copy and could move, as it stands, into a non-generic
//! function called once from where it stood.
//!
//! **What a finding proves.** The stretch it names is a set of whole basic
//! blocks of the function's pre-optimization MIR (the body `mir_flow::mir_for`
//! serves the pack's other MIR lints) with one way in and one way out:
//! control enters it only at its first block, and every edge that leaves it
//! -- unwind edges aside, as everywhere in this lint -- lands on one and the
//! same block, or else the stretch runs on to the function's `return`. A
//! block that panics or otherwise never continues may sit inside it: it
//! leaves nowhere. No statement or terminator in the stretch names a type or
//! const parameter (a `T::default()`, a `size_of::<T>()`, a cast to
//! `[u8; N]`), and none reads, writes, borrows or drops a place whose type
//! involves one. Every local the stretch reads before writing it -- what the
//! code before it hands in -- and every local it writes that is still read
//! after it -- what it hands back -- has a type no parameter appears in:
//! those are the inner function's arguments and results, and the finding
//! lists them by name and type. So the stretch, moved into a function of its
//! own that takes the first set and returns the second, has no type or const
//! parameters and is compiled once; the generic function calls it where the
//! stretch stood and keeps its signature, and its callers keep their type
//! checking. Nothing goes through `dyn`, no branch is added, and the one
//! instruction each instantiation gains is the call. What does depend on the
//! parameters stays where it is: before the stretch (`let src =
//! bytes.as_ref()`, `let s: &str = x.name()`, `let header = self.header`
//! out of a `&Framed<T>`) and after it (the drop of the by-value `bytes`,
//! the `T::from(acc)` built from a `u32` the stretch yields, the return).
//!
//! **Why it runs by default, and what it still cannot see.** That the
//! stretch is the same code in every copy and moves out verbatim for one
//! call, with the edited program still compiling, is proven from the code,
//! so every finding is actionable as written; what stays out of reach is
//! the bytes it saves, which depend on inlining, the opt level, LTO and
//! identical-code folding, and the instantiation count, which is a floor
//! (see the census paragraph below).
//!
//! **What it stays quiet on.** A function is reported only with a stretch
//! in hand, so duplication the search cannot frame as one goes unreported,
//! and that is the intended side to err on. A body whose parameter-free
//! statements are interleaved with dependent ones -- a `W: Write` written to
//! every few lines, an `F: FnMut(u8)` called inside the hot loop, a `const
//! VERBOSE: bool` tested at every step, arithmetic whose every few results
//! are folded straight into a `T` -- has no stretch of the required size,
//! and sharing what it has would take a `&mut dyn Write`, a fn pointer, a
//! run-time flag or a call per seam: a slower program, not the same one
//! compiled once. One dependent line is not interleaving: a const flag
//! tested once mid-body only splits the body in two, and the larger side is
//! reported if it qualifies -- the side before the test or after the join,
//! that is, which every copy runs; what sits inside an arm of `if FLAG` or of
//! a `match` on `coding_of(TAG)` is never a stretch, however long, because
//! each copy keeps only the arm its own constant selects and the others have
//! nothing there to share; arithmetic that only feeds a `T` built after it
//! is a stretch that yields the `u32`s the `T` is built from, with the
//! `T::from(..)` after it like the drop of `bytes`, and is reported. Also
//! unreported: a stretch under
//! `generic-body-not-generic-min-statements` counted statements, counting
//! toward that minimum only the ones written by hand -- what a `log!` line
//! expands to is fifty statements in every copy and no source anyone could
//! move, so it adds to a stretch's printed size but cannot make one; one that
//! would need two exits (an early `return` or `?` out of its middle); one
//! whose inputs would include the generic value itself (a `&self` on
//! `Framed<T>` read only for its `[u8; 4]` header is still a `&Framed<T>`:
//! copy the field into a local first and the stretch starts after that
//! line); one whose inputs would include a value each copy knows as a
//! constant (`let width = width_of(TAG)` on a `const TAG: u8`, `L::WIDTH *
//! 2`, and whatever is computed from them: the `chunks(width)` and the
//! shifts by `width` after it fold to constants in every copy, and would not
//! behind a call taking `width: usize`); one whose results would include a
//! borrow of a local the stretch itself makes or consumes (`let view =
//! &buf[..n]` handed to the sink after it, with `buf` filled inside it: a
//! function cannot return a reference into its own frame, so the stretch
//! reported ends before the borrow); one
//! that gives away on one path a value whose drop on the other does more
//! than free memory (`if done { drop(guard) }` on a lock guard the function
//! otherwise holds to its end), since the call would take that drop with it;
//! and the clean statements that share a basic block with the dependent
//! statement just before or just after the stretch, since the search works
//! in whole blocks and that block is lost to it. A stretch inside a loop of
//! the generic function is reported, and the
//! finding says the call would then run once per iteration. Functions whose
//! attributes make a call something other than a free change
//! (`#[inline(always)]`, `#[track_caller]`, `#[target_feature]`, `#[naked]`)
//! and `async` and `gen` functions are never candidates: see `splittable` and
//! `builds_coroutine`.
//! A borrow of a constant expression is judged by the expression, not by the
//! parameters rustc files it under; the one shape still judged blind -- a
//! promoted constant whose own body loads another constant of the
//! function's -- errs toward dependent, which can cut a stretch short but
//! never report one that is not there.
//!
//! **Why the census walks the bodies itself.** The instantiation count comes
//! from walking every body in the crate, reading the callee and generic
//! arguments typeck recorded at each call or fn-item use and at each
//! `Deref` it inserted on the way to a method or field (an adjustment on the
//! expression, not an expression of its own), reading the type of every
//! value the body drops off its MIR `Drop` terminators (HIR has no node for
//! a drop; the body is the pre-optimization one `mir_flow::mir_for` serves
//! the pack's other MIR lints), and propagating all of it
//! through generic callers to a fixpoint (`g::<u32>` calling `f::<T>`
//! instantiates `f::<u32>`, and dropping its `D<T>` instantiates
//! `<D<u32> as Drop>::drop`). rustc's own answer,
//! `collect_and_partition_mono_items`, is not asked: it drives the whole
//! monomorphization collector, which demands `optimized_mir` for every
//! reachable body, and that steals the `mir_drops_elaborated_and_const_checked`
//! body `mir_for` reads (see the comment there) out from under every other
//! lint in the pack; nor is codegen's collector something a check-build lint
//! pass should force. The walk sees less than the collector -- calls through
//! `dyn`, fn pointers and other crates' bodies are invisible (a `Vec<D<u8>>`
//! drops its elements inside `Vec`'s own `Drop`, `mem::drop(d)` inside
//! `core`), and `const` and `static` initializers are skipped whole, though
//! a fn pointer made in one can be called at run time -- so the count it
//! reports is a floor.
//!
//! **Why everything happens in `check_crate_post`.** `Registrar::add` wants
//! `for<'tcx> LateLintPass<'tcx> + 'static`, so the pass struct cannot hold
//! a `GenericArgsRef<'tcx>`. `check_fn` records only which local functions
//! are worth searching; the block classifier, the region search, the census
//! and the report all run at the end with `'tcx` locals.

use std::collections::VecDeque;
use std::ops::ControlFlow;

use clippy_utils::source::snippet_opt;
use clippy_utils::visitors::for_each_expr_without_closures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::work_queue::WorkQueue;
use rustc_errors::Diag;
use rustc_hir::attrs::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, ClosureKind, ConstContext, Expr, ExprKind, FnDecl, LangItem, Node};
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    self, BasicBlock, CallReturnPlaces, Local, Location, Operand, Place, RETURN_PLACE, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::adjustment::{Adjust, DerefAdjustKind};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, GenericArg, GenericArgKind, GenericArgsRef, GenericParamDefKind, Ty, TyCtxt,
    TypeSuperVisitable, TypeVisitableExt, TypeVisitor, TypeckResults,
};
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::MaybeLiveLocals;
use rustc_span::{BytePos, ExpnKind, Span, Spanned};

use crate::MordantConfig;
use crate::baseline::{emit_hir_then, join};
use crate::mir_flow::{FlowGraph, mir_for};

rustc_session::declare_lint! {
    /// Flags a generic function (type or const parameters, its own or its
    /// `impl`'s) that this crate instantiates with at least
    /// `generic-body-not-generic-min-instantiations` (default 2) distinct
    /// concrete argument sets, and whose body, read as pre-optimization MIR,
    /// holds a stretch of at least `generic-body-not-generic-min-statements`
    /// counted statements (default 24) that is the same in every one of them
    /// and could be lifted out as it stands: control enters it at one point
    /// and leaves it for one point (or runs on to the `return`), nothing in
    /// it names a parameter or touches a place whose type involves one, and
    /// every value it takes from the code before it and every value it
    /// leaves for the code after it has a type free of the parameters. rustc
    /// compiles the whole body once per argument set, so that stretch is
    /// emitted that many times over, and it need not be: moved into a
    /// non-generic inner function that takes those values and returns those
    /// results, called from where it stood, it is compiled once. The generic
    /// function keeps its signature, its callers keep their type checking,
    /// no `dyn` and no branch is introduced, and what each copy pays is one
    /// call. The finding underlines the stretch, lists what it takes by name
    /// and type and what it yields, and points at one of the instantiating
    /// call sites; when the stretch sits inside a loop of the generic
    /// function it says so, since the call is then paid once per iteration.
    ///
    /// The code around the stretch may depend on the parameters, and
    /// usually does: the `bytes.as_ref()` that turns a `B: AsRef<[u8]>` into
    /// the `&[u8]` the stretch reads comes before it, the drop of `bytes` and
    /// the return after it. When that is all the function does with a
    /// generic argument -- turn it once, on the way in, into a value of a
    /// concrete type through a trait method (`let s: &str = x.name()`) and
    /// drop it on the way out -- a second help line says it could take that
    /// value instead. A method of a trait, or of an impl of one, does not
    /// get that line: its signature is the trait's, not its own to change.
    ///
    /// A statement is counted when it does something at run time: storage
    /// markers, plain jumps, returns and unwinding edges are not. Toward the
    /// minimum it counts only if it was written by hand: the statements a
    /// macro expands to (a `trace!` line, the `Arguments` a `write!` builds)
    /// are in the stretch and in its printed size, but a stretch is not
    /// reported on the strength of them, since there is no source there to
    /// move; what a `for` loop or a `?` desugars to counts as written, and
    /// so do the tokens handed to a macro (`f(x)` in `log!("{}", f(x))`).
    /// It depends
    /// on a parameter when it names one anywhere -- a call to `T::default`,
    /// a `size_of::<T>()`, a cast to `[u8; N]`, a test of `i < N` -- or
    /// reads or writes a place whose type, or the type of a field it goes
    /// through, involves one. A place's type is the type of what it names,
    /// not of the variable it starts from: moving a `T`, dereferencing a
    /// `&T`, dropping a `Vec<T>` and reading `self.inner.len` through an
    /// `inner: Wrap<T>` are dependent, while reading a `[u8; 4]` field of
    /// `self` where `Self` is `Framed<T>`, indexing a `[u8; N]` and
    /// arithmetic on a `u32` local are not. A borrow of a constant
    /// expression is judged by the expression: rustc files every such
    /// constant under the generic function with all its parameters, but only
    /// one whose expression names a parameter is built again for each
    /// instantiation, so `&[1, 2, 3]` is independent and `&T::ZERO` and `&N`
    /// are dependent. The stretch asks one thing the statement test does
    /// not: its inputs and outputs are whole locals, and `self` is a
    /// `&Framed<T>` whatever is read through it, so a read of `self.header`
    /// cannot be the stretch's own first line. Copy or borrow the field into
    /// a local (`let header = self.header;`) and the stretch starts after
    /// that line, which is where dependent code belongs anyway: before the
    /// stretch (`path.as_ref()`, `&block[..]` on a `[u8; N]`) or after it
    /// (the drop of a by-value generic argument, a `T::from(acc)` on what
    /// the stretch yields, the return).
    ///
    /// The stretch is whole basic blocks with one entry and one exit, and a
    /// function is reported only with one in hand, so some duplication goes
    /// unreported: a clean run shorter than the threshold; one with an early
    /// `return` or `?` out of its middle, which is two exits; the clean
    /// statements that share a block with the dependent one just before or
    /// just after the run. A body whose independent statements are
    /// interleaved with dependent ones -- a `W: Write` written to throughout,
    /// a callback called inside the loop, a const flag tested every few
    /// lines, a `T` built up step by step between the independent lines --
    /// is quiet on purpose: sharing those statements would cost a `dyn`
    /// call, a branch or a call per seam, not one call. A flag tested once
    /// mid-body splits it into two stretches, before the test and after the
    /// join, and the larger is reported if it qualifies; the inside of an arm
    /// chosen by a const parameter (`if FLAG { .. }`, an arm of `match
    /// coding_of(TAG)`) is never one, since each copy keeps only the arm its
    /// own constant selects and there is nothing there the other copies
    /// share; independent work that only feeds a `T` built after it
    /// is a stretch that yields what the `T` is built from, and is reported
    /// like any other. There is
    /// no minimum share of the body: a 900-statement stretch of a
    /// 3000-statement function is worth more than a 30-statement stretch of
    /// a 40-statement one, and the bodies a share gate would keep out have
    /// no stretch to report anyway.
    ///
    /// Instantiations are counted from this crate's own calls and fn-item
    /// uses, followed through generic callers: `open::<&str>` and
    /// `open::<PathBuf>` are two, and so are `load::<A>` and `load::<B>` when
    /// `load<T>` is only ever called from `run<T>` and `run` is called with
    /// both. The calls the compiler writes for you count too: a `Deref` or
    /// `DerefMut` impl reached through `w.field` or `w.method()` on the
    /// wrapper, or through `&w` coerced to a reference to its target, and a
    /// `Drop` impl run where a value that owns one -- directly, in a field,
    /// an element, a box or a closure capture -- goes out of scope. Uses
    /// evaluated at compile time -- in a `const` or `static` initializer, an
    /// array length, a `const {}` block -- are not counted, since nothing is
    /// compiled for them. Calls through `dyn`, fn pointers and other crates'
    /// code are not seen -- that includes the elements a `Vec` drops and the
    /// value `mem::drop` takes -- so an exported function used only
    /// downstream stays quiet and the count printed is a floor, not the
    /// total.
    ///
    /// Not reported: functions where a call is not a free change --
    /// `#[inline(always)]`, which asks for a copy of the body at every call
    /// site; `#[track_caller]`, whose panics would name the shim's line from
    /// inside a plain inner fn; `#[target_feature]`, whose inner fn would be
    /// compiled without the features; `#[naked]` -- and `async` and `gen`
    /// functions, whose code is a state machine rather than a stretch an
    /// inner fn could hold. A plain `#[inline]` function is reported, with a
    /// note: the hint is not a demand, but the inner fn must not inherit it,
    /// or every codegen unit that calls it compiles its own copy again.
    ///
    /// Runs by default, because that the stretch is the same code in every
    /// copy and moves out verbatim for one call with the program still
    /// compiling is shown from the code; what it cannot see is the bytes
    /// that saves, which depend on inlining, opt level, LTO and symbol
    /// folding, and instantiations outside this crate, so the count it
    /// prints is a floor.
    pub GENERIC_BODY_NOT_GENERIC,
    Warn,
    "a generic fn with a stretch of body that is the same in every instantiation, compiled once per instantiation"
}

pub struct GenericBodyNotGeneric {
    min_statements: usize,
    min_instantiations: usize,
    /// Local generic functions with a body of their own, in visit order.
    /// Everything about them that borrows `'tcx` is computed in
    /// `check_crate_post`.
    candidates: Vec<LocalDefId>,
}

rustc_session::impl_lint_pass!(GenericBodyNotGeneric => [GENERIC_BODY_NOT_GENERIC]);

impl GenericBodyNotGeneric {
    pub fn new(config: &MordantConfig) -> Self {
        Self {
            min_statements: config.generic_body_not_generic_min_statements,
            min_instantiations: config.generic_body_not_generic_min_instantiations,
            candidates: Vec::new(),
        }
    }
}

/// A function the census tracks instantiations of: a fn item or method (not
/// a closure, whose parameters are its parent's and whose parent is the item
/// a reader would split; not a constructor) with at least one type or const
/// parameter in scope, its own or inherited from the `impl` or trait it sits
/// in. Lifetime parameters alone never count: they are erased before codegen
/// and duplicate nothing. Whether it was written by hand does not matter
/// here: a macro- or derive-generated generic fn is never reported, but the
/// concrete argument sets it is called with still flow through its body to
/// the hand-written generic fns it calls (`#[derive(Clone)]` on `Foo<T>`
/// calling a hand-written `impl<T> Clone for Bar<T>`), so it must have an
/// entry for `propagate` to follow.
fn generic_fn(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
    tcx.def_kind(def).is_fn_like()
        && !tcx.is_closure_like(def.to_def_id())
        && tcx.generics_of(def).requires_monomorphization(tcx)
}

/// A function this lint can say something about: one the census tracks,
/// written by hand rather than by a macro, so there is source to split.
fn eligible(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
    generic_fn(tcx, def) && !tcx.def_span(def).from_expansion()
}

/// Attributes that make "the same code, behind one call" untrue, read off
/// what codegen will see. `#[inline(always)]` (and rustc's own
/// `#[rustc_force_inline]`) asks for a copy of the body at every call site:
/// an out-of-line call into a shared inner fn is exactly the change its
/// author refused, so the fn is not reported. `#[track_caller]` makes every
/// panic in the body report the caller's location; moved into a plain inner
/// fn they would report the shim's line instead, and an inner fn that keeps
/// them (`#[track_caller]` too) is passed a hidden `&Location` -- more than
/// the one call the help promises. `#[target_feature]` code moved to an
/// inner fn is compiled without the features unless the attribute moves with
/// it, and then the call is `unsafe` or needs the same guard. A `#[naked]`
/// body is one `naked_asm!` block and admits no call at all. Plain
/// `#[inline]` is a hint, not a demand: the fn is reported, with a note (see
/// `inline_note`). The ABI, `const` and `unsafe` need nothing: the ABI is
/// the outer signature's, which stays; a `const fn`'s inner fn is `const`
/// too at no cost in codegen; an `unsafe` inner fn compiles the same.
fn splittable(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
    let attrs = tcx.codegen_fn_attrs(def);
    !attrs.inline.always()
        && !attrs
            .flags
            .intersects(CodegenFnAttrFlags::TRACK_CALLER | CodegenFnAttrFlags::NAKED)
        && attrs.target_features.is_empty()
}

/// An `async fn`, `gen fn` or `async gen fn`: its HIR body is the coroutine
/// it desugars to, and its own MIR only moves the arguments into that
/// coroutine's state and returns it -- one statement, dependent on every
/// parameter through the coroutine's type. The code a reader sees lives in
/// the coroutine body, which is closure-like (its parameters are the fn's,
/// `generic_fn` excludes it, `check_fn` sees it as `FnKind::Closure`) and is
/// lowered to a state machine whose blocks are resume points, not a stretch
/// an inner fn could hold. Neither def is a candidate.
fn builds_coroutine(body: &Body<'_>) -> bool {
    matches!(
        body.value.kind,
        ExprKind::Closure(closure) if matches!(closure.kind, ClosureKind::Coroutine(_))
    )
}

/// The clause a finding on an `#[inline]` fn carries, `None` for any other.
/// The hint does not excuse the fn -- rustc is free to ignore it, and a body
/// this size it often does -- but the fix has a trap worth naming: an inner
/// fn that inherits the attribute is instantiated in every codegen unit that
/// calls it, generic or not, and nothing is shared after all.
fn inline_note(tcx: TyCtxt<'_>, def: LocalDefId) -> Option<String> {
    matches!(tcx.codegen_fn_attrs(def).inline, InlineAttr::Hint).then(|| {
        format!(
            "`{}` is `#[inline]`; leave the inner fn without it, or every codegen unit that \
             calls it compiles its own copy again and nothing is shared",
            tcx.def_path_str(def)
        )
    })
}

// ── the block classifier ─────────────────────────────────────────────────────
//
// The body read is the one `mir_for` serves: the pre-optimization MIR
// (`MirPhase::Runtime(PostCleanup)`: drops elaborated, borrowck's false edges
// already removed), before inlining has copied callees in and before any pass
// has folded anything away, so what is counted is what the fn itself says.
// It is read a basic block at a time, because the region search works in
// whole blocks: each block gets how many of its statements and terminator
// are counted and how many of those are dependent, and the body's two totals
// are the sums. Every statement and terminator in a block that is not unwind
// cleanup counts once, except the ones that are bookkeeping rather than code.
// Cleanup blocks are the drops run while a panic unwinds; they mirror the
// normal path's drops and would count every one of them twice, so they are
// recorded empty and flagged, and are never part of a region.
//
// A counted item is *dependent* when a type or const parameter appears
// anywhere in it -- a cast's target type, an aggregate's or a callee's
// generic arguments, a constant, the type a `Field` projection spells -- or
// in the type of any place it reads or writes (`_1: T` moved into a call,
// `*_2` where `_2: &T`, a `Vec<T>` local dropped). A place's type is the type
// of what it names, not of the local it starts from, so `(*_1).header` with
// `_1: &Framed<T>` is a `[u8; 4]` and independent: the statement is about the
// header, not the payload. The types written on the way there still count:
// `(*_1).inner.len` through `inner: Wrap<T>` spells `Wrap<T>` in its middle
// projection and is dependent, while an `Index` projection spells no type,
// so `_1[_3]` with `_1: [u8; N]` is a `u8` and independent. Everything else
// counted is *independent*: the same statement over the same types at every
// instantiation, which is what a region is made of. Made of, not made by:
// that a run of clean blocks can move out behind one call is the region
// search's finding, not the classifier's -- one way in, one way out, and
// every local the run takes from the code before it or leaves for the code
// after it of a type no parameter appears in. That last test is on whole
// locals, so the independent `(*_1).header` above still cannot open a
// region: `_1`, the `&Framed<T>`, would be its input. Copied out first
// (`_2 = (*_1).header` in a block before the region), `_2: [u8; 4]` can be.
//
// One statement is asked differently, because asked whole it gives the wrong
// answer: the load of a promoted constant. A borrow of a constant expression
// -- `&[1, 2, 3]`, `&T::ZERO`, `&N` -- is lifted out of the body into a small
// `promoted[k]` body of its own, evaluated at compile time, and what stays
// behind is `_n = const promoted[k]`, a `Const::Unevaluated` naming the fn
// with its whole identity argument list whatever the expression was, so its
// flags say every parameter, always. (A string literal is not one of these:
// it is a `Const::Val`, already bytes, and names nothing.) Whether the value
// is one allocation every instantiation shares or one built per argument set
// is decided by the promoted's own body: `[1, 2, 3]` names no parameter in
// any local or statement, `T::ZERO` and `N` do -- and `&T::ZERO` is a `&i32`
// either way, so the constant's type alone cannot tell. So for exactly that
// statement shape, the only one promotion writes, `promoted_load_names_param`
// puts the same whole-item test to the constant's type and to every local,
// statement and terminator of `tcx.promoted_mir(def)[k]` instead of to the
// identity arguments. That errs the safe way: a promoted that itself loads a
// nested constant filed under the fn stays dependent. Each promoted is loaded
// by one statement, so each small body is read once; and one that passes
// evaluates to the same bytes under any arguments, so the borrow moved into a
// non-generic inner fn is promoted there to the same constant. Anything the
// shape does not match exactly -- another body's promoted copied in by
// inlining, a load some later pass rewrote -- is asked whole as before and
// stays dependent.
//
// What the uncounted items mean for a region (a run of whole blocks lifted
// into a non-generic inner fn) follows from the same two tests and needs no
// rule of its own. A `StorageLive`/`StorageDead` of a local whose type names
// a parameter is uncounted and does not make its block dependent: the marker
// emits nothing, and when the block moves it simply stays behind in the
// outer fn with the local it marks. A `Drop` terminator is counted and
// `PlaceParams` reads the dropped place's type, so dropping a `T`, a
// `Vec<T>` or a field reached through one is dependent and can never sit
// inside a region. `Return`, `Goto` and the unwind edges are uncounted and
// clean: where control goes is the region search's business, not the
// classifier's.

/// The counted statements and terminators of one body: how many there are,
/// and how many of those no type or const parameter appears in. Both are
/// sums over the body's `BlockFacts`; they are the two numbers a finding
/// prints and the cheap gate that skips the region search for a body that
/// cannot hold a region of `min-statements` independent items at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BodyMeasure {
    pub(crate) independent: usize,
    pub(crate) total: usize,
}

/// One basic block's share of the measure: how many of its statements and
/// its terminator are counted, how many of those a type or const parameter
/// appears in, and how many of them the author wrote. A block is *clean* --
/// may be part of a region -- when it is not unwind cleanup and nothing
/// counted in it is dependent; a cleanup block is recorded with all counts
/// zero so the body totals skip it, and flagged so the region search does
/// too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockFacts {
    pub(crate) counted: u32,
    /// Non-zero is what the region search calls dirty.
    pub(crate) dependent: u32,
    /// The counted items whose span is hand-written (`hand_written`): what
    /// the block puts toward `min-statements`. Never above `counted`; for a
    /// clean block, all of them independent. One `log!` line can be fifty
    /// MIR statements that name no parameter, and a region made of nothing
    /// else is code in every copy but not source anyone could move, so the
    /// size a region must reach is measured in these and the size a finding
    /// prints in `counted`.
    pub(crate) hand_written: u32,
    pub(crate) cleanup: bool,
}

impl BlockFacts {
    /// The block may be a region member: reachable on the normal path and
    /// the same code at every instantiation.
    pub(crate) fn clean(self) -> bool {
        !self.cleanup && self.dependent == 0
    }

    /// The counted items that do not depend on a parameter: what the block
    /// adds to a region's size.
    pub(crate) fn independent(self) -> u32 {
        self.counted - self.dependent
    }
}

/// Whether the MIR item at `span` was written by hand: its syntax context is
/// the root one, or it was made by a desugaring or an AST pass (a `for`
/// loop's `into_iter()` and `next()`, a `?`'s match, an `.await`) out of
/// code that was, judged by where that desugaring was applied. What a macro
/// wrote is not, however deep: the fifty statements of a `trace!` line, the
/// `Arguments` a `write!` builds. Tokens the author passed to a macro keep
/// their own context, so the `compute(x)` in `log!("{}", compute(x))` is
/// hand-written and the formatting around it is not. This is the walk
/// `Span::source_callsite` makes, stopped at the first macro instead of
/// carried through it; `body_position` is the other reading of the same
/// chain, which carries a macro's items out to where the invocation stands.
fn hand_written(span: Span) -> bool {
    let mut ctxt = span.ctxt();
    loop {
        if ctxt.is_root() {
            return true;
        }
        let expn = ctxt.outer_expn_data();
        match expn.kind {
            ExpnKind::Root => return true,
            ExpnKind::Macro(..) => return false,
            ExpnKind::Desugaring(_) | ExpnKind::AstPass(_) => ctxt = expn.call_site.ctxt(),
        }
    }
}

/// Statements that do something at run time. Storage markers, fake reads,
/// user type ascriptions, coverage counters and the like are bookkeeping
/// that emits nothing in any instantiation.
fn counted_statement(statement: &Statement<'_>) -> bool {
    !matches!(
        statement.kind,
        StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Nop
            | StatementKind::FakeRead(_)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::BackwardIncompatibleDropHint { .. }
    )
}

/// Terminators that do something at run time: calls, drops, switches,
/// asserts, inline asm. The ones that only name where control goes next --
/// plain jumps, the return, the unwind edges, and the false edges borrowck
/// adds for loops and match guards -- are not counted.
fn counted_terminator(terminator: &Terminator<'_>) -> bool {
    !matches!(
        terminator.kind,
        TerminatorKind::Goto { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
    )
}

/// Whether any place a statement or terminator reads or writes has a type a
/// parameter appears in. The item's own embedded types (casts, constants,
/// generic arguments, the field types a projection goes through) are asked
/// of the item as a whole; what that cannot see is the type a place ends at
/// when nothing in it spells one -- a bare `_2 = move _1`, a deref `*_3` --
/// because a local's type lives in the body's declarations, which is why
/// this needs the body. Only the projected type is checked, deliberately not
/// the base local's (no `super_place`): `(*_1).header` is about the header.
struct PlaceParams<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    found: bool,
}

impl<'tcx> Visitor<'tcx> for PlaceParams<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, _: PlaceContext, _: Location) {
        if place
            .ty(&self.body.local_decls, self.tcx)
            .ty
            .has_non_region_param()
        {
            self.found = true;
        }
    }
}

impl<'tcx> PlaceParams<'_, 'tcx> {
    /// The two ways a statement can depend on the parameters, asked apart:
    /// whether the statement itself names one (a promoted load judged by its
    /// own body, see above), and whether a place it touches has a type one
    /// appears in.
    fn statement_params(&mut self, statement: &Statement<'tcx>, at: Location) -> (bool, bool) {
        self.found = false;
        self.visit_statement(statement, at);
        let names_param = promoted_load_names_param(self.tcx, self.body, statement)
            .unwrap_or_else(|| statement.has_non_region_param());
        (names_param, self.found)
    }

    /// The same for a terminator.
    fn terminator_params(&mut self, terminator: &Terminator<'tcx>, at: Location) -> (bool, bool) {
        self.found = false;
        self.visit_terminator(terminator, at);
        (terminator.has_non_region_param(), self.found)
    }

    /// Whether a counted statement is dependent: a parameter appears in the
    /// statement itself or in the type of a place it touches. The one test
    /// the classifier and the sharper help both put, so they cannot disagree
    /// about an item.
    fn dependent_statement(&mut self, statement: &Statement<'tcx>, at: Location) -> bool {
        let (names_param, typed_place) = self.statement_params(statement, at);
        names_param || typed_place
    }

    /// The same for a counted terminator.
    fn dependent_terminator(&mut self, terminator: &Terminator<'tcx>, at: Location) -> bool {
        let (names_param, typed_place) = self.terminator_params(terminator, at);
        names_param || typed_place
    }

    /// Whether a statement is dependent *only* by naming a parameter: in a
    /// constant (`N`, `L::WIDTH`, a `&N` promoted per argument set), a
    /// callee's generic arguments (`width_of(TAG)` on a const `TAG`,
    /// `size_of::<T>()`), a cast's or aggregate's type -- while every place
    /// it reads and writes has a type free of them. What such a statement
    /// makes is a value rustc knows per instantiation: the seed of
    /// `const_derived`. One that also reads a generic place (`x.name()`,
    /// `bytes.as_ref()`) makes a run-time value and is not one; one that
    /// writes a generic place makes a local no region can name anyway.
    fn const_only_statement(&mut self, statement: &Statement<'tcx>, at: Location) -> bool {
        let (names_param, typed_place) = self.statement_params(statement, at);
        names_param && !typed_place
    }

    /// The same for a terminator: a call whose callee or arguments name a
    /// parameter only in constants and generic arguments.
    fn const_only_terminator(&mut self, terminator: &Terminator<'tcx>, at: Location) -> bool {
        let (names_param, typed_place) = self.terminator_params(terminator, at);
        names_param && !typed_place
    }
}

/// Whether `statement`, if it is the load of one of `body`'s own promoted
/// constants (`_n = const promoted[k]`), names a type or const parameter --
/// judged by the constant's type and the promoted's own body rather than by
/// the identity arguments the constant carries, which name every parameter
/// regardless (see above). `None` for any other statement, which the caller
/// asks whole. The place assigned is left to `PlaceParams` like every place.
fn promoted_load_names_param<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    statement: &Statement<'tcx>,
) -> Option<bool> {
    let (_, mir::Rvalue::Use(mir::Operand::Constant(constant), _)) = statement.kind.as_assign()?
    else {
        return None;
    };
    let mir::Const::Unevaluated(unevaluated, ty) = constant.const_ else {
        return None;
    };
    let index = unevaluated.promoted?;
    let def = body.source.def_id();
    if unevaluated.def != def {
        return None;
    }
    // `promoted_mir` is the query codegen reads these bodies through: it
    // makes sure borrowck, the one reader of the fn's promoteds as built, has
    // run, then lowers them (a handful of statements each) to runtime MIR
    // once, cached. Nothing `mir_for` reads is stolen by it.
    let promoted = tcx.promoted_mir(def).get(index)?;
    Some(
        ty.has_non_region_param()
            || promoted
                .local_decls
                .iter()
                .any(|local| local.ty.has_non_region_param())
            || promoted.basic_blocks.iter().any(|data| {
                data.statements.iter().any(|s| s.has_non_region_param())
                    || data
                        .terminator
                        .as_ref()
                        .is_some_and(|t| t.has_non_region_param())
            }),
    )
}

/// Classifies every basic block of `body` (indexed like `body.basic_blocks`)
/// and sums the blocks into the body's measure. One pass over the
/// statements, the same cost the whole-body count had.
pub(crate) fn classify<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
) -> (IndexVec<BasicBlock, BlockFacts>, BodyMeasure) {
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let facts: IndexVec<BasicBlock, BlockFacts> = body
        .basic_blocks
        .iter_enumerated()
        .map(|(block, data)| {
            if data.is_cleanup {
                return BlockFacts {
                    cleanup: true,
                    ..BlockFacts::default()
                };
            }
            let mut facts = BlockFacts::default();
            let mut count = |dependent: bool, span: Span| {
                facts.counted += 1;
                if dependent {
                    facts.dependent += 1;
                }
                if hand_written(span) {
                    facts.hand_written += 1;
                }
            };
            let at = |statement_index| Location {
                block,
                statement_index,
            };
            for (index, statement) in data.statements.iter().enumerate() {
                if !counted_statement(statement) {
                    continue;
                }
                count(
                    places.dependent_statement(statement, at(index)),
                    statement.source_info.span,
                );
            }
            if let Some(terminator) = &data.terminator
                && counted_terminator(terminator)
            {
                count(
                    places.dependent_terminator(terminator, at(data.statements.len())),
                    terminator.source_info.span,
                );
            }
            facts
        })
        .collect();
    let measure = facts.iter().fold(
        BodyMeasure {
            independent: 0,
            total: 0,
        },
        |sum, block| BodyMeasure {
            independent: sum.independent + block.independent() as usize,
            total: sum.total + block.counted as usize,
        },
    );
    (facts, measure)
}

/// The source span of every counted statement and terminator in one block,
/// in block order: what a region's reported span is the hull of. Where a
/// span that comes from an expansion stands in the body, and whether it
/// stands there at all, is `body_position`'s to say; this only says which
/// items the count saw.
pub(crate) fn counted_spans<'a>(
    body: &'a mir::Body<'_>,
    block: BasicBlock,
) -> impl Iterator<Item = Span> + 'a {
    let data = &body.basic_blocks[block];
    // A cleanup block was classified empty; it yields nothing here either.
    let live = !data.is_cleanup;
    let statements = data
        .statements
        .iter()
        .filter(move |s| live && counted_statement(s))
        .map(|s| s.source_info.span);
    let terminator = data
        .terminator
        .as_ref()
        .filter(|t| live && counted_terminator(t))
        .map(|t| t.source_info.span);
    statements.chain(terminator)
}

// ── region inputs and outputs ────────────────────────────────────────────────
//
// A region R is a set of whole basic blocks with one way in, its entry E,
// and one way out: every normal edge that leaves it lands on the same block
// X, or R holds the fn's `return` and leaves nowhere. Moving R into an inner
// fn replaces those blocks with one call whose return edge goes to X,
// `let (r1, r2) = inner(p1, p2);`, and what has to cross that call is read
// off the locals R names:
//
// * it *takes* every local R may read before it has written it -- the
//   upward-exposed uses, which is liveness run backward over R's blocks
//   alone from an empty boundary -- and every local it hands back without
//   writing it on every path to X, because the value handed back on such a
//   path is the one that came in; seeding the boundary with the hand-backs
//   instead of nothing finds both in one pass;
// * it *hands back* every local R may change -- assign, receive a call's
//   result in, borrow mutably, drop -- that the rest of the body may still
//   read once control stands at X, which is the same liveness run over the
//   whole body and asked at X's first statement. When R holds the `return`
//   the only thing read past it is `_0`.
//
// Both runs follow normal edges only (`live_in`). rustc's `MaybeLiveLocals`
// would also follow the unwind edge of every call past X into the cleanup
// blocks, and find there reads that are no business of the inner fn's: the
// flag-guarded drop of a local R has already consumed, and the drop flag
// itself. The outer fn's cleanup drops the outer fn's locals whichever side
// of the call they ended up on, so nothing read only there crosses it.
//
// Every other local R names is the inner fn's own and lives and dies in its
// frame. So no local R names may have a type that spells a type or const
// parameter: not one it takes (rule 4 of the contract), not one it hands
// back (rule 5), and not a private one either, since the inner fn declares
// it. A clean block can still name such a local -- `(*_1).header` with
// `_1: &Framed<T>` reads a `[u8; 4]` and no statement in it is dependent --
// and that is precisely the region this refuses: the inner fn cannot take a
// `&Framed<T>`, so the read of `header` has to happen before the call, in
// the shim. A write through a deref, `(*_1).len = ..`, changes what `_1`
// points at and not `_1`, so it makes `_1` something R takes, not something
// it hands back; the storage markers of a local are not a mention of it (the
// marker stays behind in the outer fn and costs nothing); and the `return`
// terminator's read of `_0` is the outer fn's, not R's.
//
// A type free of the parameters is not yet a value free of them. `let width =
// width_of(TAG)` on a `const TAG: u8`, `L::WIDTH * 2`, `size_of::<T>()` are
// `usize`s, and dependent items, so they sit before E -- but what they make is
// a constant each copy knows, and so is everything the body computes from it
// (`width as u32`, `src.chunks(width)`), which `const_derived` follows use to
// def through the body. Inside the copy, R folds around such a value: the
// chunking by 2, the shift by a literal. Behind a call that takes it as an
// argument it folds nothing, and what moved out is slower code than any copy
// had, not the same code once; so R may not take one. (It cannot make one: the
// item that would is dependent and never inside R.)
//
// Two things a signature cannot say are judged besides. A value the inner fn
// takes by value is the inner fn's to drop, so what the outer fn still does
// with it past X has to come to nothing: a local R gives away on one path and
// keeps on another is dropped past X behind a drop flag R clears, and that is
// the same program only when the drop frees memory and does nothing else
// (rustc's own line for closure captures, `has_significant_drop`) and nothing
// past X reads the local otherwise (`let left = pair.0` in R with `pair.1`
// read after it: the inner fn cannot both take `pair` and leave it). The flag
// itself has no source form and is never listed; one R assigns is let past X
// only when every drop it guards is of a local given away like that (not one
// R may or may not have *made*: that it would have to hand back), and one R
// tests that the code before E last set means R drops what may not exist,
// and refuses the region. And the inner fn cannot be handed a value that
// something else it is handed borrows: a local R moves or drops goes in by
// value, and one R writes, or writes and mutably borrows through when it is
// a `Box` or a `&mut`, goes in as `&mut` -- at E, where the original touched
// it only somewhere inside R, so a reference made from it before E and read
// inside R, which the original allowed once that reference was done with,
// becomes one param borrowing another (`inner(v, s)` with `s =
// v.as_slice()`: E0505; `inner(&mut buf, line)` with `line = &buf[..n]`:
// E0502). The other way about too: a param R only reads cannot go in beside
// one holding an exclusive borrow of it (`inner(out, &self)` with `out = &mut
// self.buf`). And the reference need not be read in R at all: one made
// before E and read only past X is held across the call just the same, where
// the original let R touch a place disjoint from the one it borrows (`let
// name = &rec.name; .. rec.count += 1 ..; out.push(name)`: fine in one body,
// E0502 as `inner(rec)`). Either way it is live at E, so it is a local live
// at E or hides in one: the params' addresses are followed through the rest
// of the body the way the next paragraph follows addresses through R -- a
// borrow counting when the borrow checker would hold it against the local,
// that is of its own storage or through `Box`es and `&mut`s but not past a
// shared reference or a raw pointer, and an exclusive borrow remembered as
// such through everything computed from it -- and the region is refused when
// a local live at E may carry the address of a param that goes in by value
// or as `&mut`, or carry an exclusive borrow of any param.
//
// Whether a local goes in by value or behind a reference is otherwise the
// writer's choice and changes nothing computed here, and for one kind of
// local the choice is free of consequence: a param R only reads, writes or
// borrows goes behind a reference, and every borrow R takes of it is a
// reborrow that points where it pointed before the split. Every other local
// R takes the address of is the inner fn's to hold, or is wanted back: a
// param R moves out of or drops (it goes in by value), a local R makes for
// itself (private to the inner fn, or handed back by value at X), and a
// param R hands back (it goes in as `&mut`, and a shared borrow of it handed
// back alongside keeps the outer fn from the very read that put it among the
// hand-backs). A borrow R takes of one of *those* points into the inner fn's
// frame or through its `&mut`: if R can leave the address somewhere the body
// looks after X, the split does not borrow-check (or, through a raw pointer,
// dangles) -- `let view = &buf[..n]` on a `buf` R filled, with `view` handed
// to the sink after X, is an inner fn returning a borrow of its local. The
// original program already bounds this: no borrow outlives its local's
// storage, so a local whose `StorageLive` and `StorageDead` all sit inside R
// cannot be pointed at from outside R. Only an addressed local whose storage
// outlasts R is followed, value by value, through R's assignments and calls,
// and the region is refused when something that may carry its address is
// live at X, is stored through a pointer, is passed to a call beside an
// argument the callee could store it through, or is put into one value (a
// struct, a tuple, a closure's captures) together with such a place, which
// is that call's two arguments arriving as one. Whether an argument is such
// a place is asked of its type (`may_take_address`), not of whether it too
// holds an address: `buf[5..9].copy_from_slice(&x.to_be_bytes())` hands the
// callee `&mut buf[..]` beside a `&[u8]`, which is nowhere to leave anything,
// and a rule that refused any second address-holding operand would refuse
// every buffer filled from a slice; `parts.push(&buf[..n])` hands it beside
// a `&mut Vec<&[u8]>`, which is somewhere, and is refused whether `parts` is
// R's own (it comes back holding the borrow) or lent to R (the inner fn
// would push a borrow of its local into it). The type is asked rather than
// whether the argument can reach storage that outlives R (a live-in, a
// hand-back, anything derived from one) for what each keeps, not because
// they agree: by reach, `buf.extend_from_slice(src)` with `src` a live-in is
// a carrier beside something that outlives R, and refused; by type, a
// `&[u8]` is nowhere and it stays. The price of going by type is that what
// the place leads to goes unasked, so the type has to answer for the worst
// of it: a pointer the argument does not own outright -- the `NonNull` in
// an `Rc` or an `Arc` moved into the call, whose other handles the body
// still holds -- counts as a place like a `&mut` to the same thing, and a
// carrier and a place built into one value count as joined there and then,
// since the one argument they arrive in later shows the callee's store to
// no rule about a second. `MaybeBorrowedLocals` is not the tool: it forgets
// a borrow only at `StorageDead`, so a `v.push(..)` in R on a `v` the body
// reads after X -- the shape this lint exists for -- would refuse every
// region it sits in.
//
// A param left where it is has one consequence after all, when a borrow R
// takes of it -- of its own storage, or through it when it is a `Box` or a
// `&mut` -- is handed back: in one body `let chunk = &cur.data[a..b]`
// borrows one place inside `*cur` and leaves `cur.reads += 1` free to happen
// while `chunk` lives; `let chunk = inner(cur)` borrows all of `*cur` for as
// long as `chunk` lives, since a signature can only tie the result to the
// whole argument. When the loan the inner fn takes is exclusive -- the param
// goes in as `&mut` (R writes it, or writes through it), or the hand-back is
// not plainly a shared reference to address-free data and so may itself hold
// a `&mut` reached through the param -- any touch of the param while the
// hand-back (or what the body copies it into) is still live is E0503/E0499;
// when it is shared, only a write, a move, a drop or a `&mut` of the param
// is (E0506/E0505/E0502). So each hand-back that may carry an address tied
// to a param is followed from X on, statement by statement, and the region is
// refused at the first such touch; coming round to E again with the
// hand-back live is a touch too, since the call takes the param once more.
// "Touch" and "live" are read the way the borrow checker will read them, not
// the way the locals spell them: a write through the pointer a lowering
// copied out of the param (`Derefer`'s copy of a `&mut` behind it,
// `ElaborateBoxDerefs`' raw pointer into a `Box`) is a write through the
// param, in R (the loan is then exclusive) as past X (it is a touch); and
// once the body past X has put the hand-back through a pointer or into a
// call beside somewhere the callee could file it (`parts.push(chunk)`), the
// loan is held by something no local answers for, and is taken as held from
// there to the end.

/// What the whole body says about its locals, computed once per fn and
/// asked once per candidate region: where each may still be read along
/// normal edges (`live_in` over every non-cleanup block), which blocks open
/// and close its storage, which locals are drop flags, and which hold a
/// value that is a constant of the instantiation.
pub(crate) struct LocalFacts {
    /// Live on entry to each non-cleanup block, unwinding disregarded.
    live: FxHashMap<BasicBlock, DenseBitSet<Local>>,
    storage: IndexVec<Local, Storage>,
    /// The non-cleanup blocks: what a region is cut from, and what the
    /// addresses of its params are followed through.
    normal: DenseBitSet<BasicBlock>,
    /// Locals whose value each instantiation knows as a constant: computed,
    /// directly or through other locals, from an item that depends on the
    /// parameters only by naming one, or from a literal chosen by a branch on
    /// such a value (`const_derived`). A region that takes one folds around
    /// it in every copy and would not behind a call.
    const_derived: DenseBitSet<Local>,
    /// `ElaborateDrops`' flags, as near as the body shows them: a `bool` no
    /// user named, past the arguments, only ever assigned a constant. The
    /// temporary an `if` on a `matches!` or a `&&` argument goes through has
    /// the same shape and is taken for one, which can cost a region the
    /// blocks that set it and nothing else.
    drop_flags: DenseBitSet<Local>,
    /// Each test of a drop flag with the local whose drop it guards: the
    /// `otherwise` edge of a `switchInt` on the flag lands on a `Drop`. `None`
    /// when it lands on anything else (a discriminant switch, another flag),
    /// which no rule here tries to see through.
    flag_guards: Vec<(Local, Option<Local>)>,
}

/// The non-cleanup blocks holding a local's storage markers, and whether any
/// of them closes it. Arguments and the return place have none: their
/// storage is the whole call.
#[derive(Default)]
struct Storage {
    blocks: Vec<BasicBlock>,
    dies: bool,
}

impl LocalFacts {
    /// `const_derived` is the set of that name, which the search has already
    /// read off the body for the branches on it (`decided_blocks`).
    pub(crate) fn new(body: &mir::Body<'_>, const_derived: &DenseBitSet<Local>) -> Self {
        let mut storage = IndexVec::from_fn_n(|_| Storage::default(), body.local_decls.len());
        let mut normal = DenseBitSet::new_empty(body.basic_blocks.len());
        // By this phase `local_info` is cleared; what the user named is what
        // has debuginfo.
        let mut drop_flags = DenseBitSet::new_empty(body.local_decls.len());
        for (local, decl) in body.local_decls.iter_enumerated().skip(body.arg_count + 1) {
            if decl.ty.is_bool() {
                drop_flags.insert(local);
            }
        }
        for var in &body.var_debug_info {
            if let mir::VarDebugInfoContents::Place(place) = var.value {
                drop_flags.remove(place.local);
            }
        }
        for (block, data) in body.basic_blocks.iter_enumerated() {
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind
                    && !assign.0.is_indirect()
                    && !matches!(assign.1, Rvalue::Use(Operand::Constant(_), _))
                {
                    drop_flags.remove(assign.0.local);
                }
            }
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::Call { destination, .. } = &terminator.kind
                && !destination.is_indirect()
            {
                drop_flags.remove(destination.local);
            }
            if data.is_cleanup {
                continue;
            }
            normal.insert(block);
            for statement in &data.statements {
                match statement.kind {
                    StatementKind::StorageLive(local) => storage[local].blocks.push(block),
                    StatementKind::StorageDead(local) => {
                        storage[local].blocks.push(block);
                        storage[local].dies = true;
                    }
                    _ => {}
                }
            }
        }
        let mut flag_guards = Vec::new();
        for data in body.basic_blocks.iter() {
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind
                && let Some(place) = discr.place()
                && !place.is_indirect()
                && drop_flags.contains(place.local)
            {
                let guarded = match &body.basic_blocks[targets.otherwise()].terminator {
                    Some(Terminator {
                        kind: TerminatorKind::Drop { place, .. },
                        ..
                    }) if !place.is_indirect() => Some(place.local),
                    _ => None,
                };
                flag_guards.push((place.local, guarded));
            }
        }
        let mut returned = DenseBitSet::new_empty(body.local_decls.len());
        returned.insert(RETURN_PLACE);
        let live = live_in(body, &normal, &returned);
        Self {
            live,
            storage,
            normal,
            const_derived: const_derived.clone(),
            drop_flags,
            flag_guards,
        }
    }

    /// The locals the body may still read, along normal edges, once control
    /// stands at the first statement of `block`.
    fn live_at(&self, block: BasicBlock) -> DenseBitSet<Local> {
        match self.live.get(&block) {
            Some(live) => live.clone(),
            None => DenseBitSet::new_empty(self.storage.len()),
        }
    }

    /// The local's storage opens and closes inside `blocks`, so no borrow of
    /// it is usable outside them: the original body borrow-checked, and a
    /// borrow never outlives its local's `StorageDead`. A local with no
    /// closing marker on a normal path (an argument, the return place, one
    /// that dies only during unwinding) is live for the whole call and never
    /// qualifies.
    fn storage_within(&self, local: Local, blocks: &DenseBitSet<BasicBlock>) -> bool {
        let storage = &self.storage[local];
        storage.dies && storage.blocks.iter().all(|&b| blocks.contains(b))
    }
}

/// The locals whose value is a constant of the instantiation rather than of
/// the fn. The seeds are the locals assigned, or a call's destination, in an
/// item that names a type or const parameter without touching a place whose
/// type does (`PlaceParams::const_only_statement`): `let width =
/// width_of(TAG)` on a const `TAG`, `L::WIDTH * 2`, `size_of::<T>()`, `N as
/// u32`, a borrow of `N` promoted per argument set. From those it runs
/// forward, use to def, to a fixpoint: a local assigned, or returned into by
/// a call, from anything that reads a marked local is marked too (`width as
/// u32`, `src.chunks(width)`, the iterator over that, an `acc` shifted by
/// `width`). In each copy rustc knows every such value and folds what is
/// computed from it -- the `chunks(2)`, the shift by a literal, the multiply
/// -- so a region that takes one is not the same code in every copy for the
/// price of a call: behind the call it is one body doing at run time what
/// each copy had for free, and `region_io` refuses it. An item that reads a
/// generic place as well (`x.name()`, `bytes.as_ref()`, `Vec::<T>::len(&v)`)
/// yields a run-time value and seeds nothing; the length of a `&[u8]` unsized
/// from a `[u8; N]` is let through the same way, which is the recall side to
/// err on for a positive the fixtures keep. Only a local's own value is
/// followed (`assigned_local`): a store to the local whole or to a field of
/// it marks it, the field being a scalar of its own that folds like a local;
/// a store through a pointer or into a callee (`buf.push(L::TAG)`) puts the
/// constant in memory, where no copy folds it, and marks nothing; and a store
/// to one element of an array (`buf[3] = L::TAG`, `buf[L::OFF] = 7`) leaves
/// the array run-time data with a constant somewhere in it and marks nothing
/// either -- the stretch that fills the rest of `buf` is every copy's. What
/// is followed is what the value is computed from, the right-hand side and a
/// call's callee and arguments, never the place it is stored to: the index in
/// `buf[L::OFF] = 7` is a use of a marked local that says nothing about `7`.
///
/// A constant reaches a local by control as well as by data: `let width =
/// match TAG { 0 => 1, 1 => 2, _ => 4 }`, `if WIDE { 8 } else { 4 }` -- the
/// idiom, and `width_of(TAG)` written out -- is a `switchInt` on `TAG` whose
/// arms are `_w = const 1_usize`, `_w = const 2_usize`, .. and name no
/// parameter, meeting again before the loop that reads `_w`. Which arm a copy
/// keeps is that copy's constant, so past the join `_w` holds one value per
/// copy exactly as the call's result did. Those are the second kind of seed
/// (`decided_constants`): a local given a value that reads no local, in a
/// block that runs or not by a constant of the instantiation
/// (`decided_blocks`). The two feed each other -- a local so marked may be
/// switched on further down, and what that switch decides may assign more
/// literals -- so they are run in turn until neither grows, which is a round
/// or two: each round has to mark a local the last did not. Both come back,
/// the locals for `region_io` and the blocks for the search's choice of entry.
fn const_derived<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    flow: &FlowGraph,
) -> (DenseBitSet<Local>, DenseBitSet<BasicBlock>) {
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let mut derived = DenseBitSet::new_empty(body.local_decls.len());
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup {
            continue;
        }
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            if let StatementKind::Assign(assign) = &statement.kind
                && let Some(local) = assigned_local(assign.0)
                && places.const_only_statement(statement, at(index))
            {
                derived.insert(local);
            }
        }
        if let Some(terminator) = &data.terminator
            && let TerminatorKind::Call { destination, .. } = &terminator.kind
            && let Some(local) = assigned_local(*destination)
            && places.const_only_terminator(terminator, at(data.statements.len()))
        {
            derived.insert(local);
        }
    }
    loop {
        if !derived.is_empty() {
            propagate_derived(body, &mut derived);
        }
        let decided = decided_blocks(body, flow, &derived);
        if !decided_constants(body, &decided, &mut derived) {
            return (derived, decided);
        }
    }
}

/// The seeds `decided` adds to `derived`: every local assigned, or returned
/// into by a call, in a block of `decided` from an item that reads no local at
/// all -- a literal, an aggregate of literals, a call on constant arguments.
/// Past the point the arms rejoin such a local is the literal of whichever arm
/// the copy kept: a constant per copy. What an arm computes from a run-time
/// value is left alone (`acc = acc.swap_bytes()` under `if WIDE`: past the
/// join `acc` is whatever the bytes were, in either copy), so the stretch
/// after a const flag tested mid-body still reports; a value computed from a
/// marked one is `propagate_derived`'s to mark, here as anywhere. A
/// `RuntimeChecks` operand is the session's constant, not the copy's, and
/// seeds nothing. Whether `derived` grew.
fn decided_constants(
    body: &mir::Body<'_>,
    decided: &DenseBitSet<BasicBlock>,
    derived: &mut DenseBitSet<Local>,
) -> bool {
    let every = DenseBitSet::new_filled(body.local_decls.len());
    let mut grew = false;
    for block in decided.iter() {
        let data = &body.basic_blocks[block];
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            if let StatementKind::Assign(assign) = &statement.kind
                && let Some(local) = assigned_local(assign.0)
                && !derived.contains(local)
                && !matches!(assign.1, Rvalue::Use(Operand::RuntimeChecks(_), _))
                && !reads_any(&every, |uses| uses.visit_rvalue(&assign.1, at(index)))
            {
                grew |= derived.insert(local);
            }
        }
        if let Some(terminator) = &data.terminator
            && let TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &terminator.kind
            && let Some(local) = assigned_local(*destination)
            && !derived.contains(local)
            && !call_reads_any(&every, func, args, at(data.statements.len()))
        {
            grew |= derived.insert(local);
        }
    }
    grew
}

/// Runs `derived` forward, use to def, to its fixpoint: a local assigned, or
/// returned into by a call, a value that reads a marked local is marked. The
/// value is the right-hand side, or the callee and arguments; the place it
/// goes to is not asked, so an index into the destination (`buf[width] = b`)
/// marks nothing.
fn propagate_derived(body: &mir::Body<'_>, derived: &mut DenseBitSet<Local>) {
    // Each pass can only mark more locals, and there are finitely many: a
    // pass that marks none is the fixpoint. Blocks are numbered roughly with
    // the flow, so a chain of assignments settles in one pass and only values
    // carried round a back edge ask for another.
    loop {
        let mut grew = false;
        for (block, data) in body.basic_blocks.iter_enumerated() {
            if data.is_cleanup {
                continue;
            }
            let at = |statement_index| Location {
                block,
                statement_index,
            };
            for (index, statement) in data.statements.iter().enumerate() {
                if let StatementKind::Assign(assign) = &statement.kind
                    && let Some(local) = assigned_local(assign.0)
                    && !derived.contains(local)
                    && reads_any(derived, |uses| uses.visit_rvalue(&assign.1, at(index)))
                {
                    grew |= derived.insert(local);
                }
            }
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } = &terminator.kind
                && let Some(local) = assigned_local(*destination)
                && !derived.contains(local)
                && call_reads_any(derived, func, args, at(data.statements.len()))
            {
                grew |= derived.insert(local);
            }
        }
        if !grew {
            break;
        }
    }
}

/// The local a store to `place` gives its value, as `const_derived` follows
/// values: the place's own local, written whole or in a field. `None` for a
/// store through a pointer and for one element of an array -- the array stays
/// run-time data whatever the element holds.
fn assigned_local(place: Place<'_>) -> Option<Local> {
    place
        .projection
        .iter()
        .all(|elem| {
            !matches!(
                elem,
                mir::ProjectionElem::Deref
                    | mir::ProjectionElem::Index(_)
                    | mir::ProjectionElem::ConstantIndex { .. }
                    | mir::ProjectionElem::Subslice { .. }
            )
        })
        .then_some(place.local)
}

/// Whether what `visit` walks uses a local in `of` (`UsesAny`).
fn reads_any(of: &DenseBitSet<Local>, visit: impl FnOnce(&mut UsesAny<'_>)) -> bool {
    let mut uses = UsesAny { of, found: false };
    visit(&mut uses);
    uses.found
}

/// Whether a call's callee or arguments -- what its result is computed from,
/// as against where it goes -- use a local in `of`.
fn call_reads_any<'tcx>(
    of: &DenseBitSet<Local>,
    func: &Operand<'tcx>,
    args: &[Spanned<Operand<'tcx>>],
    at: Location,
) -> bool {
    reads_any(of, |uses| {
        uses.visit_operand(func, at);
        for arg in args {
            uses.visit_operand(&arg.node, at);
        }
    })
}

/// What a non-generic inner fn holding a region would take and hand back,
/// both in declaration order (arguments first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RegionIo {
    /// Read in the region before the region writes them, or handed back
    /// without being written on every path: the inner fn's parameters. One
    /// the region moves or drops goes in by value, one it hands back too as
    /// `&mut`, any other as the writer likes.
    pub(crate) params: Vec<Local>,
    /// Possibly changed in the region and still read after it along a
    /// normal edge: the inner fn's results. `_0` when the region returns
    /// from the fn and assigns it.
    pub(crate) returns: Vec<Local>,
}

/// The parameters and results of an inner fn holding the region `blocks`
/// (whole, non-cleanup basic blocks; `entry` the one block entered from
/// outside; every normal edge out landing on `exit`, or `exit == None` and
/// the region holding the `return`), or `None` when the region cannot become
/// one: a local it names has a type that mentions a type or const parameter
/// (contract rules 4 and 5, and the inner fn's private locals); a param or
/// result has a type no signature can spell (`nameable`); a local it
/// gives away is still read past `exit`, or dropped there to some effect; a
/// drop flag it sets guards anything but such a local, or it tests one set
/// before it; a param it takes is a constant of the instantiation
/// (`const_derived`); a param it takes by value or as `&mut` may be borrowed by a
/// local live across it, or any param exclusively borrowed by one; or the
/// address of a local the inner fn would hold -- one it takes by value, one
/// it makes, one it hands back -- may outlive the call; or a hand-back that
/// borrows a param it leaves in place is still live where the body touches
/// that param again (`hand_back_holds_param`).
pub(crate) fn region_io<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    locals: &LocalFacts,
    blocks: &DenseBitSet<BasicBlock>,
    entry: BasicBlock,
    exit: Option<BasicBlock>,
) -> Option<RegionIo> {
    debug_assert!(blocks.contains(entry) && exit.is_none_or(|x| !blocks.contains(x)));
    let names = RegionLocals::of(tcx, body, blocks);
    // Rules (4) and (5): what goes in and what comes out are locals R names,
    // and so is everything the inner fn would declare for itself. One test
    // covers all three.
    if names
        .named
        .iter()
        .any(|local| body.local_decls[local].ty.has_non_region_param())
    {
        return None;
    }
    let live_out = match exit {
        Some(exit) => locals.live_at(exit),
        None => {
            let mut returned = DenseBitSet::new_empty(body.local_decls.len());
            returned.insert(RETURN_PLACE);
            returned
        }
    };
    // Given away in R on some path, not made afresh by R, and still read
    // past X: the inner fn takes it and the outer fn is left without. That
    // is the same program when all the outer fn did with it past X was drop
    // what R had not consumed, and the drop only frees memory.
    if let Some(exit) = exit {
        let typing_env = body.typing_env(tcx);
        for local in names.moved.iter() {
            if live_out.contains(local)
                && !names.written.contains(local)
                && (body.local_decls[local]
                    .ty
                    .has_significant_drop(tcx, typing_env)
                    || read_past(body, exit, local))
            {
                return None;
            }
        }
    }
    let mut returns = live_out;
    returns.intersect(&names.written);
    // A flag R sets and the body tests past X: fine when what it guards went
    // into the inner fn by value and stayed (the test just above passed it),
    // not when it guards a local R may or may not have made or remade.
    for flag in returns.iter().filter(|&l| locals.drop_flags.contains(l)) {
        let mut tests = locals
            .flag_guards
            .iter()
            .filter(|&&(f, _)| f == flag)
            .peekable();
        if tests.peek().is_none()
            || tests.any(|&(_, guarded)| {
                guarded.is_none_or(|v| !names.moved.contains(v) || names.written.contains(v))
            })
        {
            return None;
        }
    }
    returns.subtract(&locals.drop_flags);
    let params = exposed(body, blocks, entry, &returns);
    // What goes in and what comes out the inner fn declares in its
    // signature; free of the parameters is not yet a type it can write there.
    if params
        .iter()
        .chain(returns.iter())
        .any(|local| !nameable(body.local_decls[local].ty))
    {
        return None;
    }
    // A flag R tests that the code before E last set: R drops, on some path,
    // a local that code may not have made. (A flag R sets before every test
    // is not live at E, and one it only passes through is not in the seed.)
    if params.iter().any(|local| locals.drop_flags.contains(local)) {
        return None;
    }
    // A param whose value each copy knows as a constant (`let width =
    // width_of(TAG)`, and everything computed from it): inside the copy R
    // folds around it, behind a call taking it at run time it would not, so
    // what moved out would not be the code the copies had.
    if params
        .iter()
        .any(|local| locals.const_derived.contains(local))
    {
        return None;
    }
    // A param the call cannot be handed while a borrow tied to it is still
    // held: one that goes in by value or as `&mut` while another local may
    // hold any borrow tied to it, or one that goes in at all while another
    // local may hold an *exclusive* borrow tied to it -- E0505, E0499 or
    // E0502 at the call, where the original only touched the param once the
    // borrow had been read its last, or only ever touched a place disjoint
    // from the one borrowed (`let name = &rec.name; .. rec.count += 1 ..;
    // out.push(name)` is two fields of one body, and `inner(rec)` with `name`
    // held across it is not). What holds the borrow is live at E -- a param
    // when R reads it, any other local the body reads past X when R does
    // not -- so every local live at E is asked: the addresses are followed
    // through the part of the body outside R that some path carries to E
    // (what happens where no path leads back to E cannot be held at E), and
    // where one cannot be followed, any such local that could hold one is
    // taken to. When no local live at E could, there is nothing to ask.
    let mut across = locals.live_at(entry);
    across.union(&params);
    if across
        .iter()
        .any(|local| can_carry(body.local_decls[local].ty))
    {
        let mut outside = reaching(body, entry);
        outside.intersect(&locals.normal);
        outside.subtract(blocks);
        let mut claimed = params.clone();
        claimed.intersect(&names.exclusive);
        if !claimed.is_empty() {
            let tied = carriers(tcx, body, &outside, &claimed);
            if tied.escaped || across.iter().any(|local| tied.any.contains(local)) {
                return None;
            }
        }
        let mut shared = params.clone();
        shared.subtract(&names.exclusive);
        if !shared.is_empty() {
            let tied = carriers(tcx, body, &outside, &shared);
            if tied.escaped_exclusive || across.iter().any(|local| tied.exclusive.contains(local)) {
                return None;
            }
        }
    }
    // Every addressed local the inner fn would hold in its frame or behind a
    // `&mut` it must give back: all but the params it leaves where they are.
    // Storage contained in R settles most of them without following a value.
    let mut framed = names.addressed.clone();
    for local in names.addressed.iter() {
        let by_ref =
            params.contains(local) && !names.moved.contains(local) && !returns.contains(local);
        if by_ref || locals.storage_within(local, blocks) {
            framed.remove(local);
        }
    }
    if !framed.is_empty() {
        let carriers = carriers(tcx, body, blocks, &framed);
        if carriers.escaped || returns.iter().any(|local| carriers.any.contains(local)) {
            return None;
        }
    }
    // A param the inner fn leaves where it is can still be kept from the
    // outer fn past the call: a hand-back that points into it, or into what
    // it points at, borrows it -- all of it, where the original borrowed one
    // place inside it -- for as long as the hand-back lives (`let chunk =
    // &cur.data[a..b]; cur.pos = b; .. X .. cur.reads += 1; out.push(chunk)`
    // is two fields of `*cur` in one body and E0503 once `chunk` comes back
    // from `inner(cur)`).
    if let Some(exit) = exit {
        let past = PastExit {
            tcx,
            body,
            locals,
            region: blocks,
            entry,
            exit,
        };
        if past.hand_back_holds_param(&names.exclusive, &params, &returns) {
            return None;
        }
    }
    Some(RegionIo {
        params: params.iter().collect(),
        returns: returns.iter().collect(),
    })
}

/// What R does to each local it names, off one walk of its blocks.
struct RegionLocals {
    /// Read, written or borrowed anywhere in R. Storage markers do not name
    /// a local, and neither does the `return` terminator.
    named: DenseBitSet<Local>,
    /// Possibly changed by R: assigned, a call's or an asm block's
    /// destination, mutably borrowed, dropped. A write through a deref
    /// changes what the local points at, not the local.
    written: DenseBitSet<Local>,
    /// Moved out of or dropped somewhere in R, whole or in part: goes into
    /// the inner fn by value or not at all.
    moved: DenseBitSet<Local>,
    /// Its own storage borrowed in R (`&x`, `&mut x.field`, `&raw const x`),
    /// as opposed to something it points at (`&(*x).field`).
    addressed: DenseBitSet<Local>,
    /// What the call must be handed by value or as `&mut`, so that no other
    /// argument may hold a borrow tied to it: everything in `written` and
    /// `moved`, and a `Box` or `&mut` local R writes or mutably borrows
    /// *through* (`(*m).len = ..`, `&mut (*m).buf`) -- through a shared
    /// reference or a raw pointer the pointer is copied and nothing of the
    /// kind is needed. `Derefer` has put every `Deref` first, so the local's
    /// own type is the pointer's; and where the write goes through the
    /// pointer a lowering copied out of the local (`Derefer`'s `_t = copy
    /// (*p).src; (*_t).pos = ..` for a `&mut` behind `p`, `ElaborateBoxDerefs`'
    /// raw pointer out of a `Box`'s insides) it is laid at the local's door,
    /// which is where the borrow checker lays it (`tied_copy`).
    exclusive: DenseBitSet<Local>,
    /// Written or mutably borrowed *through*, whatever its type: the base of
    /// an indirect place under a mutating use. Folded into `exclusive` (by
    /// way of the lowerings' pointer copies, and for owning locals only) once
    /// the walk is done.
    through: DenseBitSet<Local>,
    /// Whether each local's type answers for what it points at
    /// (`owns_pointee`), looked up as places are visited.
    owning: DenseBitSet<Local>,
}

impl RegionLocals {
    fn of<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &mir::Body<'tcx>,
        blocks: &DenseBitSet<BasicBlock>,
    ) -> Self {
        let empty = DenseBitSet::new_empty(body.local_decls.len());
        let mut owning = empty.clone();
        for (local, decl) in body.local_decls.iter_enumerated() {
            if owns_pointee(decl.ty) {
                owning.insert(local);
            }
        }
        let mut names = RegionLocals {
            named: empty.clone(),
            written: empty.clone(),
            moved: empty.clone(),
            addressed: empty.clone(),
            exclusive: empty.clone(),
            through: empty,
            owning,
        };
        // Each pointer a lowering copied out of a local, with that local: a
        // write through the copy is a write through the local.
        let mut copies: Vec<(Local, Local)> = Vec::new();
        for block in blocks.iter() {
            let data = &body.basic_blocks[block];
            let at = |statement_index| Location {
                block,
                statement_index,
            };
            for (index, statement) in data.statements.iter().enumerate() {
                names.visit_statement(statement, at(index));
                copies.extend(tied_copy(tcx, body, statement));
            }
            if let Some(terminator) = &data.terminator
                && !matches!(terminator.kind, TerminatorKind::Return)
            {
                names.visit_terminator(terminator, at(data.statements.len()));
            }
        }
        // A copy of a copy (`(*(*p).a).b`) leads back in two steps; the
        // copies are few and each pass can only add.
        loop {
            let mut grew = false;
            for &(copy, of) in &copies {
                if names.through.contains(copy) {
                    grew |= names.through.insert(of);
                }
            }
            if !grew {
                break;
            }
        }
        for local in names.through.iter() {
            if names.owning.contains(local) {
                names.exclusive.insert(local);
            }
        }
        names.exclusive.union(&names.written);
        names.exclusive.union(&names.moved);
        names
    }
}

/// The statement's destination and source locals when it is one of the
/// pointer copies a lowering makes to reach through a local (`read_tied`):
/// `Derefer`'s `_t = copy (*p).f` of a `&mut` (a `CopyForDeref`, or a plain
/// copy once later passes have rewritten it), `ElaborateBoxDerefs`' `_t =
/// copy (b.0.0) as *const T`. What is then done through `_t` the borrow
/// checker holds against `p` or `b`.
fn tied_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    statement: &Statement<'tcx>,
) -> Option<(Local, Local)> {
    let (dest, rvalue) = statement.kind.as_assign()?;
    if dest.is_indirect() {
        return None;
    }
    let (Rvalue::CopyForDeref(place)
    | Rvalue::Use(Operand::Copy(place), _)
    | Rvalue::Cast(_, Operand::Copy(place), _)) = rvalue
    else {
        return None;
    };
    read_tied(tcx, body, *place).then_some((dest.local, place.local))
}

/// `root` and every local of `blocks` holding a pointer a lowering copied
/// out of it, or out of one of those in turn (`tied_copy`): the locals a use
/// through which is a use through `root`.
fn tied_copies<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
    root: Local,
) -> DenseBitSet<Local> {
    let copies: Vec<(Local, Local)> = blocks
        .iter()
        .flat_map(|block| &body.basic_blocks[block].statements)
        .filter_map(|statement| tied_copy(tcx, body, statement))
        .collect();
    let mut found = DenseBitSet::new_empty(body.local_decls.len());
    found.insert(root);
    loop {
        let mut grew = false;
        for &(copy, of) in &copies {
            if found.contains(of) {
                grew |= found.insert(copy);
            }
        }
        if !grew {
            break;
        }
    }
    found
}

impl<'tcx> Visitor<'tcx> for RegionLocals {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        use MutatingUseContext as M;
        use NonMutatingUseContext as N;
        if !place.is_indirect() {
            let local = place.local;
            match context {
                PlaceContext::MutatingUse(
                    M::Store | M::SetDiscriminant | M::AsmOutput | M::Call | M::Yield | M::Retag,
                ) => {
                    self.written.insert(local);
                }
                PlaceContext::MutatingUse(M::Borrow | M::RawBorrow) => {
                    self.written.insert(local);
                    self.addressed.insert(local);
                }
                PlaceContext::MutatingUse(M::Drop) => {
                    self.written.insert(local);
                    self.moved.insert(local);
                }
                PlaceContext::NonMutatingUse(N::Move) => {
                    self.moved.insert(local);
                }
                PlaceContext::NonMutatingUse(N::SharedBorrow | N::FakeBorrow | N::RawBorrow) => {
                    self.addressed.insert(local);
                }
                PlaceContext::MutatingUse(M::Projection)
                | PlaceContext::NonMutatingUse(
                    N::Inspect | N::Copy | N::PlaceMention | N::Projection,
                )
                | PlaceContext::NonUse(_) => {}
            }
        } else if matches!(context, PlaceContext::MutatingUse(_)) {
            self.through.insert(place.local);
        }
        // The base local and any `Index` local land in `visit_local`.
        self.super_place(place, context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if context.is_use() {
            self.named.insert(local);
        }
    }
}

/// Liveness over `blocks` alone, backward to a fixpoint, along normal edges:
/// past their boundary (an edge to a non-cleanup block outside `blocks`, a
/// `return`) exactly `boundary` is live, unwind edges lead nowhere the
/// blocks care about (the outer fn's cleanup drops the outer fn's locals
/// either way; for the whole body, nothing a caller can see), and a block
/// that cannot leave (a `panic!`, an `Unreachable`) starts from nothing. The
/// transfer function is rustc's own, call-return effect included; only the
/// `return` terminator is skipped, since its read of `_0` is the boundary's
/// to declare. The result is what is live on entry to each of `blocks`.
fn live_in(
    body: &mir::Body<'_>,
    blocks: &DenseBitSet<BasicBlock>,
    boundary: &DenseBitSet<Local>,
) -> FxHashMap<BasicBlock, DenseBitSet<Local>> {
    let empty = DenseBitSet::new_empty(body.local_decls.len());
    let preds = body.basic_blocks.predecessors();
    let mut start: FxHashMap<BasicBlock, DenseBitSet<Local>> =
        blocks.iter().map(|block| (block, empty.clone())).collect();
    // Later blocks first: a backward problem converges fastest fed roughly
    // against the flow, and MIR numbers its blocks roughly with it.
    let mut queue: WorkQueue<BasicBlock> = WorkQueue::with_none(body.basic_blocks.len());
    let mut order: Vec<BasicBlock> = blocks.iter().collect();
    order.reverse();
    for block in order {
        queue.insert(block);
    }
    let mut state = empty;
    while let Some(block) = queue.pop() {
        let data = &body.basic_blocks[block];
        state.clear();
        if let Some(terminator) = &data.terminator {
            let mut leaves = matches!(terminator.kind, TerminatorKind::Return);
            for next in terminator.successors() {
                let data = &body.basic_blocks[next];
                // Neither unwinding nor the shared empty `unreachable` block
                // (every enum `match`'s `otherwise` edge) is a way out:
                // control does not go there, nothing is live there, and the
                // region search does not count that edge as an exit either.
                if data.is_cleanup || (data.terminator.is_some() && data.is_empty_unreachable()) {
                    continue;
                }
                match start.get(&next) {
                    Some(live) => {
                        state.union(live);
                    }
                    None => leaves = true,
                }
            }
            if leaves {
                state.union(boundary);
            }
            let at = Location {
                block,
                statement_index: data.statements.len(),
            };
            match &terminator.kind {
                TerminatorKind::Return => {}
                kind => {
                    // A destination is written only on the edge that comes
                    // back, which is the one edge kept here.
                    if let TerminatorKind::Call {
                        target: Some(_),
                        destination,
                        ..
                    } = kind
                    {
                        MaybeLiveLocals.apply_call_return_effect(
                            &mut state,
                            block,
                            CallReturnPlaces::Call(*destination),
                        );
                    } else if let TerminatorKind::InlineAsm {
                        targets, operands, ..
                    } = kind
                        && !targets.is_empty()
                    {
                        MaybeLiveLocals.apply_call_return_effect(
                            &mut state,
                            block,
                            CallReturnPlaces::InlineAsm(operands),
                        );
                    }
                    MaybeLiveLocals::transfer_function(&mut state).visit_terminator(terminator, at);
                }
            }
        }
        for (index, statement) in data.statements.iter().enumerate().rev() {
            MaybeLiveLocals::transfer_function(&mut state).visit_statement(
                statement,
                Location {
                    block,
                    statement_index: index,
                },
            );
        }
        // States only grow, so a union that changes nothing is a fixpoint
        // for this block.
        if let Some(known) = start.get_mut(&block)
            && known.union(&state)
        {
            for &pred in &preds[block] {
                if blocks.contains(pred) {
                    queue.insert(pred);
                }
            }
        }
    }
    start
}

/// The blocks some path leads from to `to`, itself included, along any edge.
/// One backward walk over the predecessor lists rustc keeps.
fn reaching(body: &mir::Body<'_>, to: BasicBlock) -> DenseBitSet<BasicBlock> {
    let preds = body.basic_blocks.predecessors();
    let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
    seen.insert(to);
    let mut pending = vec![to];
    while let Some(block) = pending.pop() {
        for &pred in &preds[block] {
            if seen.insert(pred) {
                pending.push(pred);
            }
        }
    }
    seen
}

/// What is live at R's entry when exactly `returns` is live past its
/// boundary: what the inner fn takes.
fn exposed(
    body: &mir::Body<'_>,
    blocks: &DenseBitSet<BasicBlock>,
    entry: BasicBlock,
    returns: &DenseBitSet<Local>,
) -> DenseBitSet<Local> {
    live_in(body, blocks, returns)
        .remove(&entry)
        .unwrap_or_else(|| DenseBitSet::new_empty(body.local_decls.len()))
}

/// Whether anything the body can reach from `from` along normal edges reads
/// `local` other than to drop it -- the `Drop` itself, or the discriminant
/// read an enum's elaborated drop opens with (asked only of a local partly
/// moved out of by then, whose discriminant nothing else may read). Flow
/// within the reachable blocks is not followed: a local reassigned past
/// `from` and then read counts as read.
fn read_past(body: &mir::Body<'_>, from: BasicBlock, local: Local) -> bool {
    let mut of = DenseBitSet::new_empty(body.local_decls.len());
    of.insert(local);
    let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut queue = VecDeque::from([from]);
    seen.insert(from);
    while let Some(block) = queue.pop_front() {
        let data = &body.basic_blocks[block];
        let mut uses = UsesAny {
            of: &of,
            found: false,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            if let StatementKind::Assign(assign) = &statement.kind
                && let Rvalue::Discriminant(place) = &assign.1
                && !place.is_indirect()
                && place.local == local
            {
                continue;
            }
            uses.visit_statement(
                statement,
                Location {
                    block,
                    statement_index: index,
                },
            );
        }
        if let Some(terminator) = &data.terminator {
            let own_drop = matches!(
                &terminator.kind,
                TerminatorKind::Drop { place, .. } if !place.is_indirect() && place.local == local
            );
            if !own_drop {
                uses.visit_terminator(
                    terminator,
                    Location {
                        block,
                        statement_index: data.statements.len(),
                    },
                );
            }
            if uses.found {
                return true;
            }
            for next in terminator.successors() {
                if !body.basic_blocks[next].is_cleanup && seen.insert(next) {
                    queue.push_back(next);
                }
            }
        } else if uses.found {
            return true;
        }
    }
    false
}

/// Whether a fn signature can spell this type: not when a closure, a
/// coroutine, a fn item or an opaque `impl Trait` appears anywhere in it --
/// the `impl Iterator` a non-generic helper returns, revealed in the body as
/// `Filter<Copied<Iter<u8>>, {closure}>`, or the `f` of `let f = helper;`.
/// Such a type may be free of every parameter, and a local of it the inner fn
/// makes and drops for itself is fine (it is inferred), but one the inner fn
/// takes or hands back has nothing to be declared as: an `impl Iterator`
/// argument is a type parameter by another name.
fn nameable(ty: Ty<'_>) -> bool {
    !ty.walk().any(|arg| {
        matches!(
            arg.as_type().map(|ty| ty.kind()),
            Some(
                ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Coroutine(..)
                    | ty::CoroutineWitness(..)
                    | ty::FnDef(..)
                    | ty::Alias(ty::AliasTy {
                        kind: ty::Opaque { .. },
                        ..
                    })
            )
        )
    })
}

/// Whether a value of this type can hold the address of a local: a
/// reference, or anything with a lifetime in it (regions are erased in a
/// body but still there -- `Iter<'_, u8>`, a closure capturing `&x`, a
/// `dyn Trait`), or a raw pointer, anywhere inside it. A type that hides a
/// raw pointer behind no lifetime (`Vec<u8>`, a hand-made `struct P(*mut
/// u8)`) is taken to own what it points at; getting a local's address into
/// one takes `unsafe` and a cast this does not follow.
fn may_hold_address(ty: Ty<'_>) -> bool {
    ty.walk().any(|arg| match arg.kind() {
        GenericArgKind::Lifetime(_) => true,
        GenericArgKind::Type(ty) => ty.is_raw_ptr(),
        GenericArgKind::Const(_) => false,
    })
}

/// Whether a callee handed a value of this type is thereby handed a place to
/// leave an address in that outlasts the call: a `&mut` to something that
/// can hold one; a raw pointer to such a thing, bare or inside the value;
/// past a shared `&`, only an `UnsafeCell` (a `Cell`, a `Mutex`) around
/// something that can; or a type whose inside is not on view (a trait
/// object, an `impl Trait`) -- anywhere within it, the fields of structs and
/// enums included, since `struct W<'a>(&'a mut Vec<&'a u8>)` shows only the
/// lifetime. What the callee owns outright -- the value itself, and what a
/// `Box` in it holds -- dies with it or comes back as its result, which the
/// caller follows. A raw pointer in it is another matter: the value does
/// not answer for what that points at, and as a rule something else can see
/// it too -- the other handles of an `Rc<Cell<Option<&T>>>` or an
/// `Arc<Mutex<Vec<&T>>>`, whoever else holds a hand-made
/// `struct H<'a>(NonNull<Option<&'a T>>)` -- so it counts as a place the way
/// a `&mut` to its pointee would, whether the value came by value or behind
/// a `&` (the pointer copies out either way). The pointee is judged as
/// `may_hold_address` judges it, so the `NonNull<u8>` under a `Vec` or a
/// `String` is nowhere, and so is anything else showing no lifetime; a fn
/// pointer is code, not a place. `fmt::Arguments` is let through by name:
/// it points at the pieces and the `[Argument<'_>]` that `format_args!` laid
/// out in the caller's own statement, and nothing safe writes through it. So
/// a `&[u8]`, a `usize`, a `Range`, a `&Vec<&str>` or a `fmt::Arguments`
/// next to the address is nowhere to leave it, and those are what sits
/// beside `&mut buf` in the calls this lint most wants to keep
/// (`buf.extend_from_slice(src)`, `write!(buf, ..)`); a `&mut Vec<&u8>`, a
/// `&Cell<Option<&T>>` or an `Rc<Cell<Option<&T>>>` is somewhere.
fn may_take_address<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    // Each type with whether what it leads to can still be written: not once
    // a shared reference has been crossed, until an `UnsafeCell` reopens it.
    let mut pending = vec![(ty, true)];
    let mut seen: FxHashSet<(Ty<'tcx>, bool)> = FxHashSet::default();
    // Types nest finitely but instantiate without bound (`S<T>` holding a
    // `Box<S<(T, T)>>`); nothing real gets near the cap.
    let mut budget = 4096u32;
    while let Some((ty, writable)) = pending.pop() {
        if !seen.insert((ty, writable)) {
            continue;
        }
        budget -= 1;
        if budget == 0 {
            return true;
        }
        match *ty.kind() {
            // Storage the value does not own, reached through no `&`: a
            // shared `&` around the pointer freezes the pointer, not what it
            // points at.
            ty::RawPtr(pointee, _) => {
                if may_hold_address(pointee) {
                    return true;
                }
            }
            ty::Ref(_, pointee, mutability) if writable && mutability.is_mut() => {
                if may_hold_address(pointee) {
                    return true;
                }
            }
            ty::Ref(_, pointee, _) => {
                if may_hold_address(pointee) {
                    if !pointee.is_freeze(tcx, typing_env) {
                        return true;
                    }
                    pending.push((pointee, false));
                }
            }
            ty::Adt(def, args) => {
                if let Some(boxed) = ty.boxed_ty() {
                    // Owned through the pointer, so judged as a field would
                    // be -- except that `is_freeze` on whatever holds the
                    // `Box` did not look inside it.
                    if !writable && !boxed.is_freeze(tcx, typing_env) {
                        return true;
                    }
                    pending.push((boxed, writable));
                } else if !tcx.is_lang_item(def.did(), LangItem::FormatArguments) {
                    for field in def.all_fields() {
                        match tcx.try_normalize_erasing_regions(typing_env, field.ty(tcx, args)) {
                            Ok(ty) => pending.push((ty, writable)),
                            Err(_) => return true,
                        }
                    }
                }
            }
            ty::Tuple(tys) => pending.extend(tys.iter().map(|ty| (ty, writable))),
            ty::Array(element, _) | ty::Slice(element) | ty::Pat(element, _) => {
                pending.push((element, writable));
            }
            ty::Closure(_, args) => {
                pending.extend(
                    args.as_closure()
                        .upvar_tys()
                        .iter()
                        .map(|ty| (ty, writable)),
                );
            }
            ty::Alias(..) => {
                match tcx.try_normalize_erasing_regions(typing_env, ty::Unnormalized::new_wip(ty)) {
                    Ok(normal) if normal != ty => pending.push((normal, writable)),
                    _ => return true,
                }
            }
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never
            | ty::FnDef(..)
            | ty::FnPtr(..) => {}
            // A trait object, a coroutine, a parameter: not on view.
            _ => return true,
        }
    }
    false
}

/// Whether a local of this type answers for what it points at: giving it
/// away, or dropping it, ends every borrow taken through it. A `Box` and a
/// `&mut` do; a shared reference and a raw pointer are `Copy` paths to
/// storage that outlives them, and the borrow checker does not even track a
/// borrow taken through one against the pointer (`places_conflict`).
fn owns_pointee(ty: Ty<'_>) -> bool {
    !(ty.is_raw_ptr() || ty.ref_mutability() == Some(mir::Mutability::Not))
}

/// Whether a borrow of `place` is tied to `place.local` the way the borrow
/// checker ties it: of the local's own storage (`&x`, `&mut x.f`), or of
/// something reached from it through `Box`es and `&mut`s only (`&**boxed`,
/// `&mut *m`, `&(*v)[..]` on `v: &mut Vec<u8>`). Moving the local while what
/// came of such a borrow is still held is E0505; `&*r` on `r: &String`
/// leaves `r` free to go.
fn borrow_tied<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>, place: Place<'tcx>) -> bool {
    place.iter_projections().all(|(base, elem)| {
        !matches!(elem, mir::ProjectionElem::Deref) || owns_pointee(base.ty(body, tcx).ty)
    })
}

/// Whether a plain read of `place` copies out a pointer tied to
/// `place.local` the same way. User code cannot write one -- a `&mut` does
/// not copy, and a shared reference copied out of a field (`pair.1`, `s.f`
/// through `s: &mut S`) borrows nothing of `pair` or `s` -- but two
/// lowerings this body has been through do: `Derefer` splits `(*(*m).f).g`
/// into a copy of the `&mut` at `(*m).f` and a borrow through the copy, and
/// `ElaborateBoxDerefs` reaches through a `Box` by copying the raw pointer
/// out of its insides (`boxed.0.0`) or, behind a reference, the `Box` whole.
fn read_tied<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>, place: Place<'tcx>) -> bool {
    if place.projection.is_empty() {
        return false;
    }
    if body.local_decls[place.local].ty.boxed_ty().is_some() {
        return true;
    }
    let read = place.ty(body, tcx).ty;
    place.is_indirect()
        && borrow_tied(tcx, body, place)
        && (read.ref_mutability() == Some(mir::Mutability::Mut) || read.boxed_ty().is_some())
}

/// Whether a local of this type could be a carrier: it can hold an address
/// (`may_hold_address`), or it is a `Box` -- which shows no lifetime and no
/// raw pointer, and a user's `Box` owns what it points at, but the
/// `Box`-typed temporary `Derefer` copies out from behind a reference to
/// reach through it owns nothing and points where the original does.
fn can_carry(ty: Ty<'_>) -> bool {
    may_hold_address(ty) || ty.boxed_ty().is_some()
}

/// What `carriers` found about the locals it was asked about (`of`).
struct Carriers {
    /// May hold an address tied to one of `of`: a borrow tied to such a
    /// local (`borrow_tied`), a pointer a lowering copied out of one
    /// (`read_tied`), or anything computed from either.
    any: DenseBitSet<Local>,
    /// Those among `any` whose address came of an exclusive borrow -- a
    /// `&mut` or `&raw mut` tied to one of `of`, taken directly or through a
    /// carrier, the `&mut` `Derefer` copied out of one, or anything computed
    /// from those. While one is held the local stays claimed: not even a read
    /// of it may stand beside.
    exclusive: DenseBitSet<Local>,
    /// Some such address got to where locals no longer account for it:
    /// stored through a pointer, passed to a call beside an argument the
    /// callee could store it through (`v` in `v.push(&x)`), put into one
    /// value together with such a place (`Both { x: &x, v: &mut v }`, a
    /// closure over both), or into an asm block, a tail call or a yield.
    escaped: bool,
    /// The same, of an address that came of an exclusive borrow.
    escaped_exclusive: bool,
}

/// What one rvalue, operand, statement or terminator picks up, given the
/// locals asked about and the carriers found so far: whether it carries an
/// address tied to one of `of` at all, and whether one that came of an
/// exclusive borrow.
struct Picks<'a, 'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    of: &'a DenseBitSet<Local>,
    found: &'a Carriers,
    carries: bool,
    exclusive: bool,
}

impl<'a, 'mir, 'tcx> Picks<'a, 'mir, 'tcx> {
    fn of(
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        of: &'a DenseBitSet<Local>,
        found: &'a Carriers,
        visit: impl FnOnce(&mut Self),
    ) -> (bool, bool) {
        let mut picks = Picks {
            tcx,
            body,
            of,
            found,
            carries: false,
            exclusive: false,
        };
        visit(&mut picks);
        (picks.carries, picks.exclusive)
    }
}

impl<'tcx> Visitor<'tcx> for Picks<'_, '_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        use MutatingUseContext as M;
        use NonMutatingUseContext as N;
        let exclusive_borrow =
            matches!(context, PlaceContext::MutatingUse(M::Borrow | M::RawBorrow));
        if self.of.contains(place.local) {
            let (carries, exclusive) = match context {
                _ if exclusive_borrow => {
                    let tied = borrow_tied(self.tcx, self.body, *place);
                    (tied, tied)
                }
                PlaceContext::NonMutatingUse(N::SharedBorrow | N::FakeBorrow | N::RawBorrow) => {
                    (borrow_tied(self.tcx, self.body, *place), false)
                }
                PlaceContext::NonMutatingUse(N::Copy | N::Move | N::Inspect) => {
                    let tied = read_tied(self.tcx, self.body, *place);
                    let unique = place.ty(self.body, self.tcx).ty.ref_mutability()
                        == Some(mir::Mutability::Mut);
                    (tied, tied && unique)
                }
                _ => (false, false),
            };
            self.carries |= carries;
            self.exclusive |= exclusive;
        }
        // An exclusive borrow through a carrier -- `&mut *p` on the pointer
        // `ElaborateBoxDerefs` read out of a `Box` -- claims what the carrier
        // leads to, whatever the carrier was.
        if exclusive_borrow && place.is_indirect() && self.found.any.contains(place.local) {
            self.carries = true;
            self.exclusive = true;
        }
        // The base local and any `Index` local land in `visit_local`.
        self.super_place(place, context, location);
    }

    fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
        self.carries |= self.found.any.contains(local);
        self.exclusive |= self.found.exclusive.contains(local);
    }
}

/// The locals of `blocks` that may hold the address of something one of `of`
/// answers for, followed by value: a borrow tied to such a local, or a
/// pointer a lowering copied out of it, starts a carrier, and so does any
/// assignment or call result computed from a carrier, when the destination's
/// type can hold an address at all (a `len()` read through the borrow is a
/// `usize` and carries nothing); which of those trace back to an exclusive
/// borrow; and whether any got to where locals no longer account for it. The
/// caller refuses on the carriers and escapes it minds (for R's own locals,
/// any escape, and a carrier live at X: `returns` holds every local R writes
/// that is, and every carrier is written in R). Erring toward "carried" costs
/// a region; erring the other way would report a split that does not compile.
fn carriers<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
    of: &DenseBitSet<Local>,
) -> Carriers {
    let typing_env = body.typing_env(tcx);
    let may_carry =
        |place: &Place<'tcx>| !place.is_indirect() && can_carry(body.local_decls[place.local].ty);
    // An operand moved in whole is the taker's to drop or hand back, and when
    // its type shows no lifetime and no pointer -- a bare type parameter, as a
    // rule, which `may_take_address` cannot see into -- no local of this body
    // has a way to what was put in it.
    let gone = |operand: &Operand<'tcx>| {
        matches!(operand, Operand::Move(place)
            if place.projection.is_empty()
                && !may_hold_address(body.local_decls[place.local].ty))
    };
    // Whether an operand hands whoever gets it a place to leave an address
    // in that this walk does not otherwise see into.
    let somewhere = |operand: &Operand<'tcx>| {
        !gone(operand) && may_take_address(tcx, typing_env, operand.ty(&body.local_decls, tcx))
    };
    let empty = DenseBitSet::new_empty(body.local_decls.len());
    let mut found = Carriers {
        any: empty.clone(),
        exclusive: empty,
        escaped: false,
        escaped_exclusive: false,
    };
    // Each pass can only add carriers, and there are finitely many locals
    // whose type can carry: a pass that adds none is the fixpoint.
    loop {
        let mut grew = false;
        for block in blocks.iter() {
            let data = &body.basic_blocks[block];
            for statement in &data.statements {
                match &statement.kind {
                    StatementKind::Assign(assign) => {
                        let (dest, rvalue) = &**assign;
                        let (carries, exclusive) = Picks::of(tcx, body, of, &found, |p| {
                            p.visit_rvalue(rvalue, Location::START);
                        });
                        // One value holding both an address and a place to
                        // leave it in is the call rule's two arguments made
                        // one, and whoever is handed it -- `Both { v:
                        // &buf[..n], p: &mut parts }.go()`, a closure over
                        // `&buf` and `&mut parts` called -- can join them
                        // where no second argument shows. The local holding
                        // it is a carrier, but what its place leads to
                        // (`parts`) is not, so the join itself is the escape:
                        // a struct, tuple or closure built from a carrier and,
                        // in another operand, a place; or either one written
                        // into a field beside the other.
                        let built = match rvalue {
                            Rvalue::Aggregate(_, operands) if carries && operands.len() > 1 => {
                                let holds: Vec<bool> = operands
                                    .iter()
                                    .map(|held| {
                                        Picks::of(tcx, body, of, &found, |p| {
                                            p.visit_operand(held, Location::START);
                                        })
                                        .0
                                    })
                                    .collect();
                                operands.iter().enumerate().any(|(index, operand)| {
                                    holds
                                        .iter()
                                        .enumerate()
                                        .any(|(other, &carried)| carried && other != index)
                                        && somewhere(operand)
                                })
                            }
                            _ => false,
                        };
                        let beside = !dest.projection.is_empty()
                            && !dest.is_indirect()
                            && (carries || found.any.contains(dest.local))
                            && may_take_address(tcx, typing_env, body.local_decls[dest.local].ty);
                        if built || beside {
                            found.escaped = true;
                            found.escaped_exclusive |=
                                exclusive || found.exclusive.contains(dest.local);
                        }
                        if !carries {
                            continue;
                        }
                        if dest.is_indirect() {
                            found.escaped = true;
                            found.escaped_exclusive |= exclusive;
                        } else if may_carry(dest) {
                            grew |= found.any.insert(dest.local);
                            if exclusive {
                                grew |= found.exclusive.insert(dest.local);
                            }
                        }
                    }
                    // `copy_nonoverlapping` moves bytes between pointees: a
                    // carrier in it can land anywhere.
                    StatementKind::Intrinsic(_) => {
                        let (carries, exclusive) = Picks::of(tcx, body, of, &found, |p| {
                            p.visit_statement(statement, Location::START);
                        });
                        if carries {
                            found.escaped = true;
                            found.escaped_exclusive |= exclusive;
                        }
                    }
                    _ => {}
                }
            }
            let Some(terminator) = &data.terminator else {
                continue;
            };
            match &terminator.kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } => {
                    let pick = |operand: &Operand<'tcx>| {
                        Picks::of(tcx, body, of, &found, |p| {
                            p.visit_operand(operand, Location::START);
                        })
                    };
                    let by_func = pick(func);
                    let by_arg: Vec<(bool, bool)> =
                        args.iter().map(|arg| pick(&arg.node)).collect();
                    let carries = by_arg.iter().any(|&(carries, _)| carries);
                    let exclusive = by_arg.iter().any(|&(_, exclusive)| exclusive);
                    if by_func.0 {
                        found.escaped = true;
                        found.escaped_exclusive |= by_func.1;
                    }
                    if !carries {
                        continue;
                    }
                    // A callee handed an address and, in some other argument,
                    // a place an address can be left in may join them; handed
                    // nothing of the kind, it can only give the address back
                    // through its result. (One argument that is both was
                    // refused when it was put together, above.)
                    let parked = args.iter().enumerate().any(|(index, arg)| {
                        by_arg
                            .iter()
                            .enumerate()
                            .any(|(other, &(carried, _))| carried && other != index)
                            && somewhere(&arg.node)
                    });
                    if parked || destination.is_indirect() {
                        found.escaped = true;
                        found.escaped_exclusive |= exclusive;
                    }
                    if may_carry(destination) {
                        grew |= found.any.insert(destination.local);
                        if exclusive {
                            grew |= found.exclusive.insert(destination.local);
                        }
                    }
                }
                TerminatorKind::TailCall { .. }
                | TerminatorKind::InlineAsm { .. }
                | TerminatorKind::Yield { .. } => {
                    let (carries, exclusive) = Picks::of(tcx, body, of, &found, |p| {
                        p.visit_terminator(terminator, Location::START);
                    });
                    if carries {
                        found.escaped = true;
                        found.escaped_exclusive |= exclusive;
                    }
                }
                // The rest read their operands (a switch, an assert, a drop)
                // or name no local at all, and put nothing anywhere new.
                _ => {}
            }
        }
        if !grew {
            break;
        }
    }
    found
}

/// Whether a value of this type can hold a borrow the borrow checker holds
/// against the place it came from: a lifetime appears somewhere in it. A raw
/// pointer holds an address (`may_hold_address`) but no loan -- nothing stops
/// the place it points at being touched while it lives -- and a `usize` read
/// through a borrow holds neither.
fn holds_loan(ty: Ty<'_>) -> bool {
    ty.walk()
        .any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(_)))
}

/// Whether a hand-back of this type, tied to a param the region only reads,
/// ties it by a shared loan and no more: a shared reference to data that
/// holds no address of its own, so no `&mut` reached through the param can
/// sit inside it. Anything else -- a `&mut [u8]`, an `Iter<'_, T>`, a struct
/// with a lifetime, a `&Vec<&mut u8>` -- is taken to claim the param whole.
fn shared_loan(ty: Ty<'_>) -> bool {
    matches!(*ty.kind(), ty::Ref(_, pointee, mir::Mutability::Not) if !may_hold_address(pointee))
}

/// The body past a region's exit X, as the hand-back rule reads it: for a
/// hand-back that may point into a param the inner fn leaves in place (or
/// into what that param points at), whether the outer fn touches the param
/// again while the hand-back still lives. See the last paragraph of the note
/// above `LocalFacts`.
struct PastExit<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    locals: &'a LocalFacts,
    /// R's blocks.
    region: &'a DenseBitSet<BasicBlock>,
    /// R's entry E: the one block of R control can come back to from X, and
    /// where it does, the call takes the params again.
    entry: BasicBlock,
    exit: BasicBlock,
}

/// Whether one statement or terminator touches `local` in a way a loan still
/// held on it forbids: with the loan exclusive, any use at all; with it
/// shared, a write, a move, a drop or an exclusive borrow, of the local or
/// of something reached through it (`p.count += 1`, `(*p).reads += 1`,
/// `consume(p)`, `&mut p.buf`) -- through it directly or through a pointer a
/// lowering copied out of it to get there (`through`: `(*_t).reads = ..` with
/// `_t` the raw pointer `ElaborateBoxDerefs` took out of a `Box` param, or
/// the `&mut` `Derefer` copied from behind a `&mut` one). Storage markers are
/// no use of it.
struct Touches<'a> {
    local: Local,
    /// `local` and the locals holding a pointer copied out of it
    /// (`tied_copies` over the blocks walked).
    through: &'a DenseBitSet<Local>,
    exclusive: bool,
    found: bool,
}

impl<'tcx> Visitor<'tcx> for Touches<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if place.local == self.local || place.is_indirect() && self.through.contains(place.local) {
            self.found |= match context {
                PlaceContext::NonUse(_) => false,
                PlaceContext::MutatingUse(_)
                | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => true,
                PlaceContext::NonMutatingUse(_) => self.exclusive,
            };
        }
        // An `Index` local lands in `visit_local`, as a read.
        self.super_place(place, context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if local == self.local && self.exclusive && context.is_use() {
            self.found = true;
        }
    }
}

impl PastExit<'_, '_> {
    /// Whether some hand-back in `returns` may carry an address tied to a
    /// param -- a borrow of the param's own storage, or one reached through
    /// it when it is a `Box` or a `&mut`, or anything computed from either
    /// (`carriers`, seeded with the param and run over R) -- and the body,
    /// from X on, touches that param again where the hand-back, or a local
    /// past X that took its value over, is still live. What counts as a
    /// touch goes by the loan the call would hold on the param for the
    /// hand-back's sake (`Touches`): exclusive when the param is one the call
    /// must be handed by value or as `&mut` (`exclusive`: R writes, moves or
    /// mutably borrows it, or writes through it) or when the hand-back is not
    /// plainly a shared reference to address-free data (`shared_loan`);
    /// shared otherwise. A hand-back whose type holds no lifetime (a raw
    /// pointer, a length) holds no loan and is not followed. A param R also
    /// hands back is not left in place, and a borrow of its storage handed
    /// back beside it was refused before this is asked (`framed`); one
    /// reached through it lands here, under the exclusive rule, since R
    /// wrote it. Where the body past X puts the hand-back somewhere the
    /// locals no longer account for -- through a pointer (`*slot = chunk`,
    /// `out.last = chunk` behind `out: &mut Holder`), or into a call beside
    /// an argument the callee could file it in (`parts.push(chunk)` with
    /// `parts: &mut Vec<&[u8]>`; `write_all(line)` beside a `&mut File` is
    /// nowhere to file it) -- the loan lives as long as whatever took it,
    /// which is not followed: from there on it is taken as held, to the end
    /// of the body and round through R again (`pinned`).
    fn hand_back_holds_param(
        &self,
        exclusive: &DenseBitSet<Local>,
        params: &DenseBitSet<Local>,
        returns: &DenseBitSet<Local>,
    ) -> bool {
        let body = self.body;
        let hands: Vec<Local> = returns
            .iter()
            .filter(|&local| holds_loan(body.local_decls[local].ty))
            .collect();
        if hands.is_empty() {
            return false;
        }
        // The blocks past X and whether they lead round to E, walked once
        // the first pair needs them; most regions have no such pair.
        let mut past: Option<(DenseBitSet<BasicBlock>, bool)> = None;
        let mut seed = DenseBitSet::new_empty(body.local_decls.len());
        for param in params.iter() {
            seed.clear();
            seed.insert(param);
            let tied = carriers(self.tcx, body, self.region, &seed);
            for &hand in &hands {
                if hand == param || !tied.any.contains(hand) {
                    continue;
                }
                let claims = exclusive.contains(param) || !shared_loan(body.local_decls[hand].ty);
                let (blocks, reenters) = past.get_or_insert_with(|| self.reach());
                if self.touched_while_held(blocks, *reenters, param, hand, claims) {
                    return true;
                }
            }
        }
        false
    }

    /// The non-cleanup blocks control can stand in once it has left R for X,
    /// X included, short of entering R again -- which, R having one way in,
    /// it can only do at E -- and whether some edge out of them does enter R
    /// again.
    fn reach(&self) -> (DenseBitSet<BasicBlock>, bool) {
        let body = self.body;
        let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
        let mut reenters = false;
        seen.insert(self.exit);
        let mut pending = vec![self.exit];
        while let Some(block) = pending.pop() {
            let Some(terminator) = &body.basic_blocks[block].terminator else {
                continue;
            };
            for next in terminator.successors() {
                if body.basic_blocks[next].is_cleanup {
                    continue;
                }
                if self.region.contains(next) {
                    reenters = true;
                } else if seen.insert(next) {
                    pending.push(next);
                }
            }
        }
        (seen, reenters)
    }

    /// Whether, in the blocks past X, some statement or terminator touches
    /// `param` (as `Touches` has it for this kind of loan) at a point where
    /// `hand` or a local that took its value over is live, or past a point
    /// where one of those got away (`pinned`); or, the loan being exclusive,
    /// control comes round to E with the loan still held either way, where
    /// the call claims `param` again.
    fn touched_while_held(
        &self,
        past: &DenseBitSet<BasicBlock>,
        reenters: bool,
        param: Local,
        hand: Local,
        exclusive: bool,
    ) -> bool {
        let body = self.body;
        let held = taken_over(body, past, hand);
        let (pinned, pinned_reenters) = self.pinned(past, &held);
        let through = tied_copies(self.tcx, body, past, param);
        // Whether anything held is live where control arrives at `block`.
        let held_at = |block: BasicBlock| {
            self.locals
                .live
                .get(&block)
                .is_some_and(|live| held.iter().any(|local| live.contains(local)))
        };
        if exclusive && (pinned_reenters || reenters && held_at(self.entry)) {
            return true;
        }
        for block in past.iter() {
            let pinned = pinned.contains(block);
            // Nothing held is live where control arrives, and nothing got
            // away before it: whatever of it is read further down the block
            // was made afresh there, and is not the hand-back any more.
            if !pinned && !held_at(block) {
                continue;
            }
            let data = &body.basic_blocks[block];
            let live = (!pinned).then(|| live_before(body, self.locals, block, &held));
            let held_before = |index: usize| live.as_ref().is_none_or(|live| live[index]);
            let at = |statement_index| Location {
                block,
                statement_index,
            };
            let mut touches = Touches {
                local: param,
                through: &through,
                exclusive,
                found: false,
            };
            for (index, statement) in data.statements.iter().enumerate() {
                if !held_before(index) {
                    continue;
                }
                touches.visit_statement(statement, at(index));
                if touches.found {
                    return true;
                }
            }
            if let Some(terminator) = &data.terminator
                && held_before(data.statements.len())
            {
                touches.visit_terminator(terminator, at(data.statements.len()));
                if touches.found {
                    return true;
                }
            }
        }
        false
    }

    /// The blocks of `past` in which the loan must be taken as held at every
    /// point whatever the liveness of `held` says: each block where a held
    /// value gets to where the locals no longer account for it (`escapes`)
    /// and every block of `past` control can reach from one -- when it can
    /// reach E, that is round through R and out at X again, so all of them
    /// -- together with whether it can reach E. (In the escaping block itself
    /// the points before the escape have a held local live anyway: the one
    /// about to escape.)
    fn pinned(
        &self,
        past: &DenseBitSet<BasicBlock>,
        held: &DenseBitSet<Local>,
    ) -> (DenseBitSet<BasicBlock>, bool) {
        let body = self.body;
        let mut pinned = DenseBitSet::new_empty(body.basic_blocks.len());
        let mut pending: Vec<BasicBlock> = past
            .iter()
            .filter(|&block| escapes(self.tcx, body, block, held))
            .collect();
        for &block in &pending {
            pinned.insert(block);
        }
        let mut reenters = false;
        while let Some(block) = pending.pop() {
            let Some(terminator) = &body.basic_blocks[block].terminator else {
                continue;
            };
            for mut next in terminator.successors() {
                if self.region.contains(next) {
                    reenters = true;
                    // Through R, which lets go of nothing the body past X
                    // put away, and out at X.
                    next = self.exit;
                }
                if past.contains(next) && pinned.insert(next) {
                    pending.push(next);
                }
            }
        }
        (pinned, reenters)
    }
}

/// Whether `block` puts the loan a local of `held` holds where the locals no
/// longer account for it, so that nothing this file follows says when it is
/// let go: a held value stored through a pointer into a place that can hold
/// a loan (`*slot = chunk`, `(*out).last = &chunk[1..]`; not `*n =
/// chunk.len()`), moved about by an intrinsic, passed to a call as the callee
/// or beside an argument the callee is thereby handed a place to file it in
/// (`parts.push(chunk)` with `parts: &mut Vec<&[u8]>`, `sink.take(chunk)`
/// with `sink: &mut S` and `S` not on view -- `may_take_address`, as
/// `carriers` asks it, but with no allowance for an argument moved in whole:
/// what an opaque callee owns it may still hold the loan in), received from
/// one through a pointer, or handed to an asm block, a yield or a tail call.
/// A held local's `len()` or first byte going anywhere carries no loan
/// (`holds_loan` of the operand).
fn escapes<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    block: BasicBlock,
    held: &DenseBitSet<Local>,
) -> bool {
    let data = &body.basic_blocks[block];
    let carries = |operand: &Operand<'tcx>| {
        holds_loan(operand.ty(&body.local_decls, tcx))
            && reads_any(held, |uses| uses.visit_operand(operand, Location::START))
    };
    for statement in &data.statements {
        let escaped = match &statement.kind {
            StatementKind::Assign(assign) => {
                let (dest, rvalue) = &**assign;
                dest.is_indirect()
                    && holds_loan(dest.ty(body, tcx).ty)
                    && reads_any(held, |uses| uses.visit_rvalue(rvalue, Location::START))
            }
            StatementKind::Intrinsic(_) => reads_any(held, |uses| {
                uses.visit_statement(statement, Location::START)
            }),
            _ => false,
        };
        if escaped {
            return true;
        }
    }
    let Some(terminator) = &data.terminator else {
        return false;
    };
    match &terminator.kind {
        TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } => {
            if carries(func) {
                return true;
            }
            let by_arg: Vec<bool> = args.iter().map(|arg| carries(&arg.node)).collect();
            if !by_arg.contains(&true) {
                return false;
            }
            if destination.is_indirect() && holds_loan(destination.ty(body, tcx).ty) {
                return true;
            }
            let typing_env = body.typing_env(tcx);
            args.iter().enumerate().any(|(index, arg)| {
                by_arg
                    .iter()
                    .enumerate()
                    .any(|(other, &carried)| carried && other != index)
                    && may_take_address(tcx, typing_env, arg.node.ty(&body.local_decls, tcx))
            })
        }
        TerminatorKind::TailCall { .. }
        | TerminatorKind::InlineAsm { .. }
        | TerminatorKind::Yield { .. } => reads_any(held, |uses| {
            uses.visit_terminator(terminator, Location::START)
        }),
        _ => false,
    }
}

/// `hand`, and every local of `blocks` that may take its value over there:
/// assigned, or received from a call, out of operands that name one already
/// in the set, into a local whose type can hold a loan (`let rest =
/// &chunk[1..]`, `let s = str::from_utf8(chunk)`, `let pair = (chunk, n)`;
/// not `chunk.len()`). The loan a hand-back holds lives as long as any of
/// them -- and past where one of them is put through a pointer or into a
/// callee's keeping, which `escapes` answers for, as long as the body. A
/// fixpoint over the blocks; each pass can only add locals.
fn taken_over(
    body: &mir::Body<'_>,
    blocks: &DenseBitSet<BasicBlock>,
    hand: Local,
) -> DenseBitSet<Local> {
    let mut held = DenseBitSet::new_empty(body.local_decls.len());
    held.insert(hand);
    let mut added: Vec<Local> = Vec::new();
    loop {
        for block in blocks.iter() {
            let data = &body.basic_blocks[block];
            let takes = |dest: &Place<'_>, held: &DenseBitSet<Local>| {
                !dest.is_indirect()
                    && !held.contains(dest.local)
                    && holds_loan(body.local_decls[dest.local].ty)
            };
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind
                    && takes(&assign.0, &held)
                {
                    let mut uses = UsesAny {
                        of: &held,
                        found: false,
                    };
                    uses.visit_rvalue(&assign.1, Location::START);
                    if uses.found {
                        added.push(assign.0.local);
                    }
                }
            }
            if let Some(terminator) = &data.terminator
                && let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } = &terminator.kind
                && takes(destination, &held)
            {
                let mut uses = UsesAny {
                    of: &held,
                    found: false,
                };
                uses.visit_operand(func, Location::START);
                for arg in args {
                    uses.visit_operand(&arg.node, Location::START);
                }
                if uses.found {
                    added.push(destination.local);
                }
            }
        }
        if added.is_empty() {
            break;
        }
        for local in added.drain(..) {
            held.insert(local);
        }
    }
    held
}

/// For each statement of `block` and then its terminator, whether any of
/// `held` is live just before it along normal edges: rustc's liveness
/// transfer run backward through the block from what `LocalFacts` has live
/// on entry to its normal successors, the way `live_in` runs it (a call's
/// destination is written on the edge that comes back; the `return` reads
/// `_0`; unwinding leads nowhere that counts).
fn live_before(
    body: &mir::Body<'_>,
    locals: &LocalFacts,
    block: BasicBlock,
    held: &DenseBitSet<Local>,
) -> Vec<bool> {
    let data = &body.basic_blocks[block];
    let any_held = |state: &DenseBitSet<Local>| held.iter().any(|local| state.contains(local));
    let mut state = DenseBitSet::new_empty(body.local_decls.len());
    let mut live = vec![false; data.statements.len() + 1];
    if let Some(terminator) = &data.terminator {
        for next in terminator.successors() {
            if let Some(there) = locals.live.get(&next) {
                state.union(there);
            }
        }
        let at = Location {
            block,
            statement_index: data.statements.len(),
        };
        match &terminator.kind {
            TerminatorKind::Return => {
                state.insert(RETURN_PLACE);
            }
            kind => {
                if let TerminatorKind::Call {
                    target: Some(_),
                    destination,
                    ..
                } = kind
                {
                    MaybeLiveLocals.apply_call_return_effect(
                        &mut state,
                        block,
                        CallReturnPlaces::Call(*destination),
                    );
                } else if let TerminatorKind::InlineAsm {
                    targets, operands, ..
                } = kind
                    && !targets.is_empty()
                {
                    MaybeLiveLocals.apply_call_return_effect(
                        &mut state,
                        block,
                        CallReturnPlaces::InlineAsm(operands),
                    );
                }
                MaybeLiveLocals::transfer_function(&mut state).visit_terminator(terminator, at);
            }
        }
        live[data.statements.len()] = any_held(&state);
    }
    for (index, statement) in data.statements.iter().enumerate().rev() {
        MaybeLiveLocals::transfer_function(&mut state).visit_statement(
            statement,
            Location {
                block,
                statement_index: index,
            },
        );
        live[index] = any_held(&state);
    }
    live
}

// ── the region search ────────────────────────────────────────────────────────
//
// What is looked for is a region R of the body's flow graph (`FlowGraph`:
// normal edges only, a `return` is an edge to the virtual EXIT, a diverging
// block has no successor, unwind cleanup is not there) with an entry block E
// and an exit node X, X a block or EXIT, such that
//
//   (1) every block of R is clean: not cleanup, no counted item in it
//       dependent, not left by a `become` (which the flow graph sends to
//       EXIT like a `return`, but which never assigns `_0` and cannot keep
//       its callee's signature inside an inner fn), and -- folded in here
//       because it can only get worse as R grows -- no local it names has a
//       type a parameter appears in, which is `region_io`'s first refusal
//       put block by block (`fit`);
//   (2) R has one way in: every predecessor of a block of R other than E is
//       in R, and E itself is the fn's entry block or has a predecessor
//       outside R (`entries == 0` in `Grower::single_entry`);
//   (3) R has one way out: every successor of a block of R is in R or is X;
//       a `return` is the successor EXIT, so it is allowed exactly when X is
//       EXIT, and a block that diverges has no successor and constrains
//       nothing (`Grower::single_exit`);
//   (4) every local R reads before writing has a parameter-free type, and
//   (5) so does every local it writes that is read after X -- both answered
//       by `region_io`, which also refuses what a signature cannot say: a
//       param whose value is a constant of the instantiation, a drop flag
//       crossing the call, a value given away that the body still drops, a
//       by-value param another param borrows, and an address of the inner
//       fn's own locals outliving it (see that section);
//   (6) the uncounted items need no rule (see the classifier's note);
//   (7) R's hand-written counted items (`BlockFacts::hand_written`) number
//       at least `min_statements` (`Grower::checkpoint`). Its size as
//       reported is all its counted items; what a macro wrote is in R and
//       moves with it, but is not what makes R worth naming.
//
// **Why such an R is one call.** Take the inner fn whose parameters are R's
// upward-exposed locals and whose body is R's blocks, E first, with every
// edge to X replaced by a `return` of the tuple of R's live-out locals (when
// X is EXIT those edges are the fn's own `return`s and the tuple is `_0`).
// By (1), (4) and (5) nothing in that fn -- no statement, no local -- names a
// type or const parameter, so it has none and is compiled once. In the outer
// fn, delete R's blocks and put in their place one block: `let (rets) =
// inner(params)` with return edge to X. Control that used to enter R entered
// at E, by (2), and now enters the call; control that used to leave R left
// for X, by (3), and now returns to X; a block of R that panicked or never
// returned does the same inside the inner fn. Every local the deleted blocks
// read from the code before them is a parameter, every local the code after
// them reads from the deleted blocks is assigned from the tuple, and every
// other local the blocks touched is the inner fn's own. Unwinding out of the
// inner fn drops the inner fn's locals and then, through the call's unwind
// edge, the outer fn's, which is the set the original cleanup blocks dropped.
// The outer fn's signature is untouched, so its callers are too; the price is
// the call, and when X can reach E again (`in_loop`) that price is paid per
// iteration, which the finding says.
//
// **How the candidates are enumerated.** If R is valid with exit X a block,
// every path from E to EXIT leaves R, and by (3) it leaves through X: X
// strictly post-dominates E. And given E and a strict post-dominator X, a
// valid R is exactly the set of blocks reachable from E without entering X --
// (3) forces every such block in, (2) keeps every other block out, since a
// block not reachable from E that way could only be entered from outside. So
// the candidates are the pairs (E, X) with X on E's immediate-post-dominator
// chain X1 = ipdom(E), X2 = ipdom(X1), .. EXIT, and for each pair one set
// R(E, Xk) to vet, which no choice is involved in. The sets nest: every path
// from E to Xk+1 meets Xk first (Xk post-dominates E, Xk+1 does not
// post-dominate Xk's alternatives), so R(E, Xk+1) is R(E, Xk), plus Xk, plus
// what Xk reaches short of Xk+1, and one breadth-first walk from E, paused at
// each Xk and resumed from it, yields the whole chain's sets in the time of
// the largest. A block failing (1) that the walk meets is in that set and in
// every later one, so the chain ends there; (2) can fail at Xk and hold again
// at Xk+2 (a loop entered at E whose back edge source joins R only once X has
// moved past the loop), so it ends nothing; (3) and (7) likewise. Rules (2)
// and (3) are kept as two counters updated per edge as blocks join -- edges
// into R that do not go through E, edges out of R -- so vetting a pair costs
// a look at X's predecessors, not a pass over R.
//
// Every reachable clean block that some path carries to a `return`, and that
// every copy enters -- not one that runs only when a branch on a constant of
// the instantiation goes its way (`decided_blocks`: an arm of `if FLAG`, of
// `match coding_of(TAG)`), since after constant propagation such a block is
// in the copies whose constant chose it and gone from the rest, and what it
// heads is neither the same in every copy nor shared by moving it -- is tried
// as E, in reverse post-order, and the largest valid region wins
// (hand-written size, the measure rule (7) uses, then the earlier E). Two
// filters keep that from being a walk per block. A valid R lies inside E's
// dominator subtree (a block of R reached around E would be entered from
// outside R) and stops at the first unfit block down every branch of it, so
// the hand-written mass of that pruned subtree,
// summed bottom-up once for all E, bounds any region E can head; an E whose
// bound is under `min_statements` or not above the best size so far is
// skipped unwalked. And `region_io` -- a liveness fixpoint over R plus one
// over the body and an address walk over the rest of it, the one step here
// whose cost grows with locals as well as blocks -- is not asked per pair:
// the walk records which Xk passed (1)(2)(3)(7) and their sizes, and only
// afterwards are those that would beat the best put to it -- the largest
// first, which nearly always passes, and when it is refused a bisection down
// the chain for the largest that is not (`Search::probe`), all entries of a
// round drawing on one budget of such runs (`IO_BUDGET`) so that a body whose
// candidates keep failing cannot have a set of them per entry. So a fn costs
// one classification pass, two dominator computations, at most one walk per
// surviving E over the blocks its region could hold, and a bounded number of
// liveness runs; the body-wide liveness is built only once some pair has
// passed the structural rules. What the search returns is a region valid
// under every rule and, unless refusals along a chain fail to nest or the
// budget ran out, the largest there is. A
// body without `min_statements` hand-written items among all its clean blocks
// is turned away before the flow graph is built, and one without them among
// its fit blocks straight after.
//
// `other_regions` is the same search run again with the winner's blocks
// struck out, up to `OTHER_REGIONS_CAP` more times, each further winner
// struck out in turn: a `const FLAG: bool` tested once mid-body leaves a
// region either side of the test, and the finding can say the fix does not
// end with the one it names. Statement-level trimming of E's or X's block is
// not attempted: a clean statement that shares a block with a dependent one
// is lost to the region, which the lint's doc says. The one trim made goes
// the other way and is in whole blocks too (`Search::trim`): a last member
// that ends by computing the checked half of an addition finished past X is
// given back to the outer fn, the clean statements sharing it included, so
// the finding does not ask anyone to hand a `(u32, bool)` across a call.

/// A stretch of the body that could be a non-generic fn of its own, called
/// once from where it stood: whole basic blocks satisfying rules (1)-(7)
/// above, with what the call would pass and receive.
#[derive(Clone, Debug)]
pub(crate) struct Region {
    /// The one block control enters the region at.
    pub(crate) entry: BasicBlock,
    /// The one node every edge out of the region lands on; `None` when that
    /// node is EXIT, i.e. the region runs to the fn's `return`.
    pub(crate) exit: Option<BasicBlock>,
    /// The member blocks, indexed like `body.basic_blocks`.
    pub(crate) blocks: DenseBitSet<BasicBlock>,
    /// Counted statements and terminators in the members, all independent:
    /// the size a finding prints. At least `min_statements` of them are
    /// hand-written (rule (7)), usually all.
    pub(crate) size: usize,
    /// What the inner fn takes: locals read in the region before the region
    /// writes them, in declaration order (`region_io`).
    pub(crate) params: Vec<Local>,
    /// What it hands back: locals the region may change that are read after
    /// it; `_0` when `exit` is `None` and the region assigns it.
    pub(crate) returns: Vec<Local>,
    /// The exit can reach the entry again: the call replacing the region
    /// runs once per iteration of an enclosing loop, not once per call.
    pub(crate) in_loop: bool,
    /// Further valid regions of at least `min_statements`, disjoint from
    /// this one and from each other, found by searching again without them.
    /// Counting stops at `OTHER_REGIONS_CAP`.
    pub(crate) other_regions: usize,
}

/// How many further disjoint regions `best_region` looks for after the
/// winner. Each is a search of its own; three more is enough to say "and it
/// is not the only one" without turning one finding into a survey.
const OTHER_REGIONS_CAP: usize = 3;

/// The most candidates one round of the search puts to `region_io`, over
/// all its entry blocks together. Each run is a liveness fixpoint over the
/// candidate and an address walk over the rest of the body, so a bound per
/// entry would still let a body whose candidates keep failing the borrow
/// and address checks ask for a set of runs per entry block -- quadratic in
/// the body. An entry costs one run when its largest candidate passes, as it
/// nearly always does, and about `1 + log2(candidates)` when it does not
/// (`Search::probe`); a round that has spent the budget keeps the best it
/// has, so what the cap can cost is a region, never a wrong one.
const IO_BUDGET: usize = 32;

/// Whether any local a block's statements or terminator *use* is in `of`.
/// Storage markers are not uses, and neither is the `return` terminator's
/// read of `_0` -- the two exemptions `region_io` makes, made the same way,
/// so that the block-level test here refuses exactly the blocks whose
/// presence would make `region_io` refuse the region.
struct UsesAny<'a> {
    of: &'a DenseBitSet<Local>,
    found: bool,
}

impl<'tcx> Visitor<'tcx> for UsesAny<'_> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if context.is_use() && self.of.contains(local) {
            self.found = true;
        }
    }
}

/// Rule (1) as the search applies it, per block: in the flow graph (so not
/// cleanup and not the shared empty `unreachable`), nothing counted in it
/// dependent, no local it uses typed with a parameter, and not left by
/// `become` -- a tail call needs the caller's own signature, so it cannot
/// move into an inner fn with a different one. Blocks struck out by an
/// earlier round of the search are kept apart, in `excluded`, so this is
/// computed once.
fn fit_blocks<'tcx>(
    body: &mir::Body<'tcx>,
    facts: &IndexSlice<BasicBlock, BlockFacts>,
    flow: &FlowGraph,
) -> DenseBitSet<BasicBlock> {
    let mut generic = DenseBitSet::new_empty(body.local_decls.len());
    for (local, decl) in body.local_decls.iter_enumerated() {
        if decl.ty.has_non_region_param() {
            generic.insert(local);
        }
    }
    let mut fit = DenseBitSet::new_empty(body.basic_blocks.len());
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if !flow.contains(block) || !facts[block].clean() {
            continue;
        }
        if let Some(terminator) = &data.terminator
            && matches!(terminator.kind, TerminatorKind::TailCall { .. })
        {
            continue;
        }
        let mut uses = UsesAny {
            of: &generic,
            found: false,
        };
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            uses.visit_statement(statement, at(index));
        }
        if let Some(terminator) = &data.terminator
            && !matches!(terminator.kind, TerminatorKind::Return)
        {
            uses.visit_terminator(terminator, at(data.statements.len()));
        }
        if !uses.found {
            fit.insert(block);
        }
    }
    fit
}

/// The blocks that run, or not, according to a constant of the
/// instantiation: those control-dependent, directly or through further
/// branches, on a `SwitchInt` whose discriminant is a constant naming a
/// parameter (`if FLAG` on a `const FLAG: bool`) or a local whose value each
/// copy knows (`const_derived`: the discriminant of `coding_of(TAG)`, `N >
/// 4`, `size_of::<T>() == 8`). No region may be *entered* at one. Its blocks
/// are clean and the classifier cannot tell them from any others, but after
/// constant propagation each copy keeps only the arm its own constant selects:
/// an arm of `match coding_of(TAG)` is code the copies sharing one `TAG`
/// carry and the rest do not, so it is not the same stretch in every copy,
/// moving it out shares nothing between copies that chose differently, and
/// the finding's count of copies would be a number the lint cannot know
/// without evaluating the branch per argument set. Where the arms meet again
/// the code runs whichever way the branch went and is every copy's once more
/// -- a block that post-dominates the branch is not decided by it -- so a
/// const flag tested mid-body still leaves the stretch after the join to
/// report, and a region entered before the branch may hold the whole `match`,
/// arms and all, if its blocks are otherwise fit: what it takes in is then the
/// derived discriminant, which `region_io` refuses on its own account. The
/// switch block itself is usually dependent (`_5 = const FLAG` sits in it) but
/// need not be (`match coding` a block after the call that made `coding`), so
/// this is read off the flow graph, not off the classifier: each such switch
/// decides the blocks from each of its successors up the post-dominator tree
/// short of where its arms rejoin (`FlowGraph::decides`), and a branch among
/// those passes the decision on to what it decides in turn, to a fixpoint.
fn decided_blocks(
    body: &mir::Body<'_>,
    flow: &FlowGraph,
    derived: &DenseBitSet<Local>,
) -> DenseBitSet<BasicBlock> {
    // What each copy knows the value of: a constant that names a parameter,
    // or a place that reads a derived local (the local switched on, or one
    // indexing it). A `RuntimeChecks` operand asks the session, not the
    // instantiation, and is the same in every copy.
    let per_copy = |discr: &Operand<'_>| match discr {
        Operand::Constant(constant) => constant.const_.has_non_region_param(),
        Operand::Copy(place) | Operand::Move(place) => {
            let mut uses = UsesAny {
                of: derived,
                found: false,
            };
            uses.visit_place(
                place,
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect),
                Location::START,
            );
            uses.found
        }
        Operand::RuntimeChecks(_) => false,
    };
    let mut decided = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut pending: Vec<BasicBlock> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|&(block, data)| {
            flow.contains(block)
                && matches!(
                    &data.terminator,
                    Some(Terminator { kind: TerminatorKind::SwitchInt { discr, .. }, .. })
                        if per_copy(discr)
                )
        })
        .map(|(block, _)| block)
        .collect();
    // Each branch is expanded once: the first time it is found to be derived
    // or decided. What it decides is marked, and any block among that with a
    // choice of successors is a branch whose every arm is decided too.
    let mut expanded = DenseBitSet::new_empty(body.basic_blocks.len());
    while let Some(branch) = pending.pop() {
        if !expanded.insert(branch) {
            continue;
        }
        for block in flow.decides(branch) {
            if decided.insert(block) && flow.succs(block).len() >= 2 {
                pending.push(block);
            }
        }
    }
    decided
}

/// For every block, an upper bound on the hand-written size of any valid
/// region it could head -- the size rule (7) and the search's ordering go
/// by: the hand-written items of the fit blocks in its dominator subtree,
/// each branch of the subtree cut at its first unfit block. (A
/// region headed by E lies in E's subtree by rule (2); a block below an
/// unfit dominator D is reached from E only through D, since E strictly
/// dominates D.) One bottom-up pass over the dominator tree.
fn subtree_mass(
    facts: &IndexSlice<BasicBlock, BlockFacts>,
    flow: &FlowGraph,
    fit: &DenseBitSet<BasicBlock>,
    n: usize,
) -> IndexVec<BasicBlock, usize> {
    let mut children: IndexVec<BasicBlock, Vec<BasicBlock>> = IndexVec::from_elem_n(Vec::new(), n);
    for block in (0..n).map(BasicBlock::from_usize) {
        if let Some(parent) = flow.idom(block)
            && parent != block
            && parent.as_usize() < n
        {
            children[parent].push(block);
        }
    }
    let mut mass: IndexVec<BasicBlock, usize> = IndexVec::from_elem_n(0, n);
    // Post-order over the dominator tree without recursion: a block is
    // summed once all its children have been.
    let mut stack: Vec<(BasicBlock, bool)> = vec![(mir::START_BLOCK, false)];
    while let Some((block, summed_children)) = stack.pop() {
        if summed_children {
            if fit.contains(block) {
                mass[block] = facts[block].hand_written as usize
                    + children[block].iter().map(|&c| mass[c]).sum::<usize>();
            }
        } else {
            stack.push((block, true));
            stack.extend(children[block].iter().map(|&c| (c, false)));
        }
    }
    mass
}

/// One structurally valid candidate met along an entry's chain: rules
/// (1)(2)(3)(7) hold for the first `members` blocks the walk collected, with
/// exit `exit`.
struct Checkpoint {
    exit: BasicBlock,
    members: usize,
    /// Counted items in the members: what the finding prints.
    size: usize,
    /// The hand-written ones among them: what rule (7) measured and what
    /// candidates are ranked by.
    written: usize,
}

/// The state of one walk from an entry block: the region collected so far
/// and the two edge counters rules (2) and (3) are read off. Reused across
/// entries; `reset` clears exactly what the last walk touched.
struct Grower<'a> {
    flow: &'a FlowGraph,
    facts: &'a IndexSlice<BasicBlock, BlockFacts>,
    fit: &'a DenseBitSet<BasicBlock>,
    in_region: DenseBitSet<BasicBlock>,
    /// The region's blocks in the order they joined, entry first. Every
    /// prefix that ends at a checkpoint is a candidate region.
    members: Vec<BasicBlock>,
    /// Counted items over the members, and the hand-written ones among
    /// them (`BlockFacts::hand_written`). The first is the region's size as
    /// reported; the second is the size rule (7) asks `min_statements` of,
    /// so that a region made of what one macro line expands to, which has
    /// no source to move, does not pass on the strength of it.
    size: usize,
    written: usize,
    /// Edges from outside the region into a member other than the entry.
    entries: usize,
    /// Edges from a member to a node outside the region (EXIT included).
    exits: usize,
    queue: VecDeque<BasicBlock>,
}

impl<'a> Grower<'a> {
    fn new(
        flow: &'a FlowGraph,
        facts: &'a IndexSlice<BasicBlock, BlockFacts>,
        fit: &'a DenseBitSet<BasicBlock>,
        n: usize,
    ) -> Self {
        Grower {
            flow,
            facts,
            fit,
            // One past the blocks, so EXIT can be asked about and is never in.
            in_region: DenseBitSet::new_empty(n + 1),
            members: Vec::new(),
            size: 0,
            written: 0,
            entries: 0,
            exits: 0,
            queue: VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        for &block in &self.members {
            self.in_region.remove(block);
        }
        self.members.clear();
        self.size = 0;
        self.written = 0;
        self.entries = 0;
        self.exits = 0;
        self.queue.clear();
    }

    fn entry(&self) -> BasicBlock {
        self.members[0]
    }

    /// Adds a fit block and accounts for its edges: an edge from a member
    /// becomes internal (one exit fewer), an edge from outside is a way in
    /// unless this is the entry; an edge to a member closes a way in, an edge
    /// to anything else is a way out. A self-edge is neither.
    fn add(&mut self, block: BasicBlock) {
        debug_assert!(self.fit.contains(block) && !self.in_region.contains(block));
        let is_entry = self.members.is_empty();
        self.in_region.insert(block);
        self.members.push(block);
        self.size += self.facts[block].counted as usize;
        self.written += self.facts[block].hand_written as usize;
        for &pred in self.flow.preds(block) {
            if pred == block {
                continue;
            }
            if self.in_region.contains(pred) {
                self.exits -= 1;
            } else if !is_entry {
                self.entries += 1;
            }
        }
        let entry = self.entry();
        for &succ in self.flow.succs(block) {
            if succ == block {
                continue;
            }
            if self.in_region.contains(succ) {
                // Edges into the entry were never counted as ways in.
                if succ != entry {
                    self.entries -= 1;
                }
            } else {
                self.exits += 1;
            }
        }
        self.queue.push_back(block);
    }

    /// Runs the walk on from whatever is queued without entering `barrier`
    /// (or EXIT, which is not a block). `false` when it met an unfit or
    /// struck-out block: that block is in this candidate and in every larger
    /// one from this entry.
    fn grow_short_of(&mut self, barrier: BasicBlock, excluded: &DenseBitSet<BasicBlock>) -> bool {
        let exit = self.flow.exit();
        while let Some(block) = self.queue.pop_front() {
            for &succ in self.flow.succs(block) {
                if succ == barrier || succ == exit || self.in_region.contains(succ) {
                    continue;
                }
                if !self.fit.contains(succ) || excluded.contains(succ) {
                    return false;
                }
                self.add(succ);
            }
        }
        true
    }

    /// Rule (2). The entry's own clause can only fail for an unreachable
    /// entry, which is never tried; it is checked because it is stated.
    fn single_entry(&self) -> bool {
        let entry = self.entry();
        self.entries == 0
            && (entry == mir::START_BLOCK
                || self
                    .flow
                    .preds(entry)
                    .iter()
                    .any(|&p| !self.in_region.contains(p)))
    }

    /// Rule (3): every edge out of the region lands on `exit`. The edges out
    /// are counted; the ones landing on `exit` are its predecessors inside;
    /// the two agree exactly when nothing lands elsewhere. A `return` inside
    /// the region is an edge to EXIT and so counts against any other exit.
    fn single_exit(&self, exit: BasicBlock) -> bool {
        let landing = self
            .flow
            .preds(exit)
            .iter()
            .filter(|&&p| self.in_region.contains(p))
            .count();
        self.exits == landing
    }

    /// The candidate ending at the current frontier, if rules (2), (3) and
    /// (7) hold for it -- (7) put to the hand-written items, not to all of
    /// them. That `exit` is not itself a member is what the walk
    /// guarantees (see the note on nesting); it is asked again because
    /// `single_exit` would count edges into a member as landings.
    fn checkpoint(&self, exit: BasicBlock, min_statements: usize) -> Option<Checkpoint> {
        (self.written >= min_statements
            && !self.in_region.contains(exit)
            && self.single_entry()
            && self.single_exit(exit))
        .then_some(Checkpoint {
            exit,
            members: self.members.len(),
            size: self.size,
            written: self.written,
        })
    }

    /// Walks `entry`'s post-dominator chain and returns every candidate that
    /// passed the structural rules, smallest first (sizes do not decrease
    /// along the chain).
    fn walk(
        &mut self,
        entry: BasicBlock,
        excluded: &DenseBitSet<BasicBlock>,
        min_statements: usize,
    ) -> Vec<Checkpoint> {
        self.reset();
        self.add(entry);
        let mut passed = Vec::new();
        let mut previous: Option<BasicBlock> = None;
        let chain: Vec<BasicBlock> = self.flow.ipdom_chain(entry).collect();
        for exit in chain {
            // The last barrier is inside this candidate: it is reachable
            // (it post-dominates the entry) and no longer the place to stop.
            if let Some(previous) = previous
                && !self.in_region.contains(previous)
            {
                if !self.fit.contains(previous) || excluded.contains(previous) {
                    break;
                }
                self.add(previous);
            }
            if !self.grow_short_of(exit, excluded) {
                break;
            }
            passed.extend(self.checkpoint(exit, min_statements));
            previous = Some(exit);
        }
        passed
    }

    /// The first `members` blocks of the walk as a set.
    fn blocks(&self, members: usize) -> DenseBitSet<BasicBlock> {
        let mut blocks = DenseBitSet::new_empty(self.facts.len());
        for &block in &self.members[..members] {
            blocks.insert(block);
        }
        blocks
    }
}

/// A region that passed every rule, before `in_loop` and `other_regions`
/// are filled in.
struct Found {
    entry: BasicBlock,
    exit: BasicBlock,
    blocks: DenseBitSet<BasicBlock>,
    /// As on `Checkpoint`: all counted items, and the hand-written ones.
    size: usize,
    written: usize,
    io: RegionIo,
}

/// The search proper: the per-fn tables every round reads, and the
/// body-wide liveness `region_io` wants, built the first time a candidate
/// gets that far and kept for the rounds after.
struct Search<'a, 'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    flow: &'a FlowGraph,
    grower: Grower<'a>,
    mass: IndexVec<BasicBlock, usize>,
    /// Locals whose value is a constant of the instantiation
    /// (`const_derived`): read here for the branches on them, and handed on
    /// to `LocalFacts` for the region inputs among them.
    derived: DenseBitSet<Local>,
    /// Blocks that run only when a branch on such a constant goes one way
    /// (`decided_blocks`): never a region's entry.
    decided: DenseBitSet<BasicBlock>,
    locals: Option<LocalFacts>,
    min_statements: usize,
}

impl Search<'_, '_, '_> {
    /// One round: the best region valid under all seven rules the search
    /// finds among blocks that are fit and not `excluded`, by hand-written
    /// size and then by the earlier entry in reverse post-order. That is the
    /// largest such region outright unless `probe`'s bisection stepped over
    /// it on the way down from a refused candidate, or the round ran through
    /// `IO_BUDGET` first; either way it is a region that passed every rule.
    fn round(&mut self, excluded: &DenseBitSet<BasicBlock>) -> Option<Found> {
        let mut best: Option<Found> = None;
        let mut budget = IO_BUDGET;
        for &entry in self.body.basic_blocks.reverse_postorder() {
            if budget == 0 {
                break;
            }
            // A region needs an exit: EXIT itself at the least, so some path
            // from the entry must return. Blocks in a diverging arm head
            // nothing. And it must be entered in every copy: a block that
            // runs only when a branch on a per-copy constant goes its way
            // heads code some copies have and the others dropped.
            if !self.grower.fit.contains(entry)
                || excluded.contains(entry)
                || self.decided.contains(entry)
                || !self.flow.can_return(entry)
            {
                continue;
            }
            let ceiling = self.mass[entry];
            if ceiling < self.min_statements || best.as_ref().is_some_and(|b| ceiling <= b.written)
            {
                continue;
            }
            let passed = self.grower.walk(entry, excluded, self.min_statements);
            // Sizes do not decrease along the chain, so the candidates that
            // would beat the best so far are a tail of the list.
            let floor = best.as_ref().map_or(0, |b| b.written);
            let from = passed.partition_point(|c| c.written <= floor);
            if let Some(found) = self.probe(entry, &passed[from..], &mut budget) {
                best = Some(found);
            }
        }
        best
    }

    /// Puts one entry's structurally valid `candidates` (smallest first,
    /// every one beating the best so far) to `region_io`: the largest first,
    /// and when that is refused a bisection over the rest -- a pass moves the
    /// search up the list, a refusal down -- keeping the largest that
    /// passed. Refusals mostly nest the way the candidates do, since a larger
    /// region names, moves and borrows every local a smaller one does, and
    /// where they nest the result is the largest candidate that passes; where
    /// they do not (a larger region that swallows a local's whole storage is
    /// free of it again) the result is still a candidate that passed every
    /// rule, which is all a finding needs, and the largest was asked first.
    /// At most `1 + log2(len)` runs, each charged to `budget`; when that is
    /// spent the probe stops with what it has.
    fn probe(
        &mut self,
        entry: BasicBlock,
        candidates: &[Checkpoint],
        budget: &mut usize,
    ) -> Option<Found> {
        let mut found = None;
        let (mut low, mut high) = (0, candidates.len());
        let mut pick = high.checked_sub(1)?;
        while low < high && *budget > 0 {
            *budget -= 1;
            let candidate = &candidates[pick];
            let blocks = self.grower.blocks(candidate.members);
            let exit = (candidate.exit != self.flow.exit()).then_some(candidate.exit);
            let locals = self
                .locals
                .get_or_insert_with(|| LocalFacts::new(self.body, &self.derived));
            // Rules (4) and (5), the drop flags, the per-copy constants, and
            // the two address checks.
            match region_io(self.tcx, self.body, locals, &blocks, entry, exit) {
                Some(io) => {
                    found = Some(Found {
                        entry,
                        exit: candidate.exit,
                        blocks,
                        size: candidate.size,
                        written: candidate.written,
                        io,
                    });
                    low = pick + 1;
                }
                None => high = pick,
            }
            pick = low + (high - low) / 2;
        }
        found
    }

    /// Takes back off the end of a found region the blocks that are there
    /// only because the search stops at whole blocks. A `count += 1` or `let
    /// n = a + b` just before the exit ends, in this MIR, a block that
    /// computes the checked-arithmetic pair and asserts on its overflow bit,
    /// with the assignment of the sum in the block after -- the exit, when
    /// that block is dependent. Kept, such a block makes the inner fn hand
    /// back "the `(u32, bool)` from `count += 1`" for the outer fn to finish
    /// the addition with, which no one would write; dropped, the addition
    /// stays whole in the outer fn. The pair seldom has the block to itself:
    /// the operands it reads are copied into temporaries first (`let n =
    /// cur.reads + 1`), and a `+=` that follows a plain statement -- a store,
    /// a comparison, the store of the previous `+=`'s sum -- is appended to
    /// that statement's block. The block goes back whole all the same, the
    /// clean statements in it with the pair, as the ones sharing X's block
    /// already are (the module note); what the finding then prints is a
    /// stretch a little shorter than it might be instead of a hand-back no
    /// one can act on. So while the region's one way out runs through a
    /// single member that is not the entry, that member ends in such an
    /// assert on a temporary the region as it now stands hands back
    /// (`checked_step`), and it can be spared without going under
    /// `min_statements`, it becomes the new exit and `region_io` is asked
    /// again: the pair's operands, or the previous `+=`'s pair, may now cross
    /// the edge instead, and the next member is judged against those. Rules
    /// (2) and (3) survive by construction -- every edge that left the region
    /// left from that member, so every edge out of what remains lands on it,
    /// and it may lead back only to the entry. A step `region_io` refuses is
    /// not taken, and the region stays as the step before left it.
    fn trim(&self, mut found: Found) -> Found {
        let Some(locals) = &self.locals else {
            return found;
        };
        let facts = self.grower.facts;
        loop {
            let mut landers = self
                .flow
                .preds(found.exit)
                .iter()
                .copied()
                .filter(|&p| found.blocks.contains(p));
            let (Some(last), None) = (landers.next(), landers.next()) else {
                break;
            };
            let spared = facts[last].hand_written as usize;
            if last == found.entry
                || found.written - spared < self.min_statements
                || !checked_step(self.body, last, &found.io.returns)
                || self
                    .flow
                    .succs(last)
                    .iter()
                    .any(|&s| s != found.entry && s != last && found.blocks.contains(s))
            {
                break;
            }
            let mut blocks = found.blocks.clone();
            blocks.remove(last);
            let Some(io) = region_io(
                self.tcx,
                self.body,
                locals,
                &blocks,
                found.entry,
                Some(last),
            ) else {
                break;
            };
            found = Found {
                entry: found.entry,
                exit: last,
                blocks,
                size: found.size - facts[last].counted as usize,
                written: found.written - spared,
                io,
            };
        }
        found
    }
}

/// Whether `block` ends by asserting on the overflow bit of a
/// checked-arithmetic pair it computes -- `_9 = AddWithOverflow(..)` then
/// `assert(!_9.1)` -- into a temporary the source never named that is among
/// a region's hand-backs (`returns`). What else the block does is not looked
/// at: `Search::trim` gives it back whole.
fn checked_step(body: &mir::Body<'_>, block: BasicBlock, returns: &[Local]) -> bool {
    let data = &body.basic_blocks[block];
    let Some(Terminator {
        kind: TerminatorKind::Assert { cond, msg, .. },
        ..
    }) = &data.terminator
    else {
        return false;
    };
    if !matches!(**msg, mir::AssertKind::Overflow(..)) {
        return false;
    }
    let Some(tested) = cond.place() else {
        return false;
    };
    let pair = tested.local;
    if !returns.contains(&pair) || local_name(body, pair).is_some() {
        return false;
    }
    data.statements.iter().any(|statement| {
        matches!(
            &statement.kind,
            StatementKind::Assign(assign)
                if assign.0.as_local() == Some(pair)
                    && matches!(assign.1, Rvalue::BinaryOp(op, _) if op.is_overflowing())
        )
    })
}

/// The largest stretch of `body` that could move into a non-generic inner fn
/// for the price of one call -- rules (1) to (7) above -- or `None` when no
/// set of whole blocks with at least `min_statements` hand-written counted
/// items qualifies. `facts` is `classify`'s per-block verdict on the same
/// body.
pub(crate) fn best_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    facts: &IndexSlice<BasicBlock, BlockFacts>,
    min_statements: usize,
) -> Option<Region> {
    let n = body.basic_blocks.len();
    // The cheap turn-away: not enough hand-written items among the clean
    // blocks of the whole body, however they are arranged. `classify`'s
    // measure bounds this from above, but a caller need not have checked,
    // and a body whose clean mass is one logging macro's expansion is turned
    // away here, before a flow graph is built for it.
    let clean_total: usize = facts
        .iter()
        .filter(|f| f.clean())
        .map(|f| f.hand_written as usize)
        .sum();
    if clean_total < min_statements {
        return None;
    }
    let flow = FlowGraph::new(body);
    let fit = fit_blocks(body, facts, &flow);
    let fit_total: usize = fit.iter().map(|b| facts[b].hand_written as usize).sum();
    if fit_total < min_statements {
        return None;
    }
    // Which locals are constants of the instantiation is wanted twice: now,
    // for the branches on them that decide which blocks may head a region,
    // and by `region_io` for the inputs among them; and the decided blocks
    // feed back into it (a literal assigned in one is such a constant), so
    // the two come out of one fixpoint. It is a few passes over the
    // statements, read once here and handed on.
    let (derived, decided) = const_derived(tcx, body, &flow);
    let mut search = Search {
        tcx,
        body,
        flow: &flow,
        grower: Grower::new(&flow, facts, &fit, n),
        mass: subtree_mass(facts, &flow, &fit, n),
        derived,
        decided,
        locals: None,
        min_statements,
    };
    let mut excluded = DenseBitSet::new_empty(n);
    // The winner, less a checked-arithmetic tail worth nothing to the inner
    // fn (`Search::trim`). The runners-up are only counted, so they keep
    // theirs; and the tail the winner gave back is struck out with it, or a
    // tail long enough would come back as a runner-up handing back the same
    // pair.
    let best = search.round(&excluded)?;
    excluded.union(&best.blocks);
    let best = search.trim(best);
    // The runners-up: strike the winner out and ask again. The mass bound
    // was summed over blocks now excluded, so it only loosens, which a bound
    // may do.
    let mut other_regions = 0;
    while other_regions < OTHER_REGIONS_CAP
        && let Some(other) = search.round(&excluded)
    {
        other_regions += 1;
        excluded.union(&other.blocks);
    }
    let mut region = Region {
        entry: best.entry,
        exit: (best.exit != flow.exit()).then_some(best.exit),
        blocks: best.blocks,
        size: best.size,
        params: best.io.params,
        returns: best.io.returns,
        in_loop: false,
        other_regions,
    };
    // A block exit that leads back round to the entry puts the region, and
    // the call that would replace it, inside a loop of the outer fn. EXIT
    // leads nowhere.
    region.in_loop = region
        .exit
        .is_some_and(|exit| flow.reaches(exit, region.entry));
    Some(region)
}

// ── the instantiation census ─────────────────────────────────────────────────

/// Something a body does that instantiates generic fns.
#[derive(Clone, Copy)]
enum Use<'tcx> {
    /// A call or fn-item use: the callee as typeck resolved it (a trait
    /// item, not yet the impl's) with its whole argument list.
    Fn(DefId, GenericArgsRef<'tcx>),
    /// A value of this type dropped. Its drop glue calls `Drop::drop` on
    /// every part of the value that has one; `glue_drops` names the parts.
    Drop(Ty<'tcx>),
}

impl<'tcx> Use<'tcx> {
    fn has_non_region_param(self) -> bool {
        match self {
            Use::Fn(_, args) => args.has_non_region_param(),
            Use::Drop(ty) => ty.has_non_region_param(),
        }
    }

    /// The use as the copy of its body compiled for `caller_args` makes it.
    fn instantiate(self, tcx: TyCtxt<'tcx>, caller_args: GenericArgsRef<'tcx>) -> Option<Self> {
        let env = ty::TypingEnv::fully_monomorphized();
        Some(match self {
            Use::Fn(callee, args) => Use::Fn(
                callee,
                tcx.try_instantiate_and_normalize_erasing_regions(
                    caller_args,
                    env,
                    ty::EarlyBinder::bind(args),
                )
                .ok()?,
            ),
            Use::Drop(dropped) => Use::Drop(
                tcx.try_instantiate_and_normalize_erasing_regions(
                    caller_args,
                    env,
                    ty::EarlyBinder::bind(dropped),
                )
                .ok()?,
            ),
        })
    }

    fn nesting(self, nesting: &mut Nesting<'tcx>) -> usize {
        match self {
            Use::Fn(_, args) => nesting.of_args(args),
            Use::Drop(ty) => nesting.of(ty.into()),
        }
    }
}

/// A use inside a generic body that still mentions the caller's parameters:
/// it is made once per concrete argument set of `caller`.
struct Propagating<'tcx> {
    /// The typeck root the use sits in (a closure's enclosing fn).
    caller: LocalDefId,
    used: Use<'tcx>,
    site: Span,
}

/// One instantiation: a local generic fn and a concrete argument set
/// (regions erased, normalized, no parameters left) its body is compiled
/// for.
type Instantiation<'tcx> = (LocalDefId, GenericArgsRef<'tcx>);

#[derive(Default)]
struct Census<'tcx> {
    /// Local generic fn -> the concrete argument sets it is instantiated
    /// with here. Both levels in insertion order, which follows the source,
    /// so the order `propagate` works in and which use the note names do not
    /// depend on hash order.
    concrete: FxIndexMap<LocalDefId, FxIndexSet<GenericArgsRef<'tcx>>>,
    /// One use per fn that instantiates it, for the note: the lowest span
    /// wins, and on a tie (two sets derived through one call in a generic
    /// caller) the first recorded.
    first_site: FxHashMap<LocalDefId, (Span, GenericArgsRef<'tcx>)>,
    propagating: Vec<Propagating<'tcx>>,
    /// Instantiations found and not yet expanded by `propagate`.
    queue: VecDeque<Instantiation<'tcx>>,
    /// `Drop::drop`, which drop glue calls: `None` only without `core`.
    drop_fn: Option<DefId>,
}

impl<'tcx> Census<'tcx> {
    /// Files a use made in `caller`'s body at `site`: kept for `propagate`
    /// while it still mentions `caller`'s parameters, recorded where it
    /// lands once it does not.
    fn file(&mut self, tcx: TyCtxt<'tcx>, caller: LocalDefId, used: Use<'tcx>, site: Span) {
        if used.has_non_region_param() {
            self.propagating.push(Propagating { caller, used, site });
        } else {
            self.record(tcx, used, site);
        }
    }

    /// Records a use with concrete arguments at `site` against every local
    /// generic fn it lands on, and queues each argument set not seen before.
    fn record(&mut self, tcx: TyCtxt<'tcx>, used: Use<'tcx>, site: Span) {
        match used {
            Use::Fn(callee, args) => self.land(tcx, callee, args, site),
            Use::Drop(dropped) => {
                let Some(drop_fn) = self.drop_fn else {
                    return;
                };
                for part in glue_drops(tcx, dropped) {
                    self.land(tcx, drop_fn, tcx.mk_args(&[part.into()]), site);
                }
            }
        }
    }

    fn land(&mut self, tcx: TyCtxt<'tcx>, callee: DefId, args: GenericArgsRef<'tcx>, site: Span) {
        if let Some(found) = resolve(tcx, callee, args)
            && self.insert(found, site)
        {
            self.queue.push_back(found);
        }
    }

    /// True when `args` is a new argument set for `item`.
    fn insert(&mut self, (item, args): Instantiation<'tcx>, site: Span) -> bool {
        let new = self.concrete.entry(item).or_default().insert(args);
        let first = self.first_site.entry(item).or_insert((site, args));
        if site.lo() < first.0.lo() {
            *first = (site, args);
        }
        new
    }
}

/// Where a use of `callee` with concrete `args` actually lands: the local
/// generic fn whose body is compiled for it, with the arguments that body
/// sees. A trait method call lands on the impl's method when the receiver
/// type picks a local impl; dispatch through `dyn`, closures' call shims and
/// anything foreign land nowhere this lint counts.
fn resolve<'tcx>(
    tcx: TyCtxt<'tcx>,
    callee: DefId,
    args: GenericArgsRef<'tcx>,
) -> Option<Instantiation<'tcx>> {
    // A foreign free fn or inherent method can only resolve to itself; a
    // foreign trait's item can still resolve to an impl in this crate.
    if !callee.is_local() && tcx.trait_of_assoc(callee).is_none() {
        return None;
    }
    let env = ty::TypingEnv::fully_monomorphized();
    let args = tcx
        .try_normalize_erasing_regions(env, ty::Unnormalized::new_wip(args))
        .ok()?;
    if args.has_non_region_param() {
        return None;
    }
    let instance = ty::Instance::try_resolve(tcx, env, callee, args).ok()??;
    let ty::InstanceKind::Item(item) = instance.def else {
        return None;
    };
    let item = item.as_local()?;
    if !generic_fn(tcx, item) {
        return None;
    }
    let args = tcx.erase_and_anonymize_regions(instance.args);
    // The sets recorded for `item` are later substituted into argument lists
    // written against `item`'s own generics, so they must line up with them.
    (args.len() == tcx.generics_of(item).count() && !args.has_non_region_param())
        .then_some((item, args))
}

/// `Drop::drop`, the method drop glue calls on each part of a value that
/// has a `Drop` impl.
fn drop_fn(tcx: TyCtxt<'_>) -> Option<DefId> {
    let drop_trait = tcx.lang_items().drop_trait()?;
    tcx.associated_items(drop_trait)
        .in_definition_order()
        .find(|item| item.is_fn())
        .map(|item| item.def_id)
}

/// The parts of a dropped value of concrete type `ty` whose `Drop` impl is
/// written in this crate, as the self types the glue calls `Drop::drop` at:
/// `ty` itself, then whatever its fields, elements, boxed contents, closure
/// captures and values held across a suspension point own, transitively. A
/// part behind a reference or raw pointer is not dropped, one inside a
/// `ManuallyDrop` or a `union` is not dropped by glue, and a `dyn` drops
/// through its vtable; what a foreign `Drop` impl drops by hand (`Vec`'s
/// elements) is dropped in that crate's code, which is not read. `ty` is
/// normalized first so that an `impl Trait` value is walked as the type it
/// hides; every part pushed after that comes out of a normalized type.
fn glue_drops<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Vec<Ty<'tcx>> {
    let env = ty::TypingEnv::fully_monomorphized();
    let mut local = Vec::new();
    let Ok(ty) = tcx.try_normalize_erasing_regions(env, ty::Unnormalized::new_wip(ty)) else {
        return local;
    };
    let mut seen: FxHashSet<Ty<'tcx>> = FxHashSet::default();
    let mut stack = vec![ty];
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) || !ty.needs_drop(tcx, env) {
            continue;
        }
        match *ty.kind() {
            ty::Adt(def, args) => {
                if def.is_manually_drop() {
                    continue;
                }
                if def.destructor(tcx).is_some_and(|d| d.did.is_local()) {
                    local.push(ty);
                }
                if def.is_union() {
                    continue;
                }
                if def.is_box() {
                    // The contents, dropped through the box's built-in deref
                    // before the box itself; its fields hold only a pointer.
                    stack.extend(args.types());
                }
                stack.extend(def.all_fields().filter_map(|field| {
                    tcx.try_normalize_erasing_regions(env, field.ty(tcx, args))
                        .ok()
                }));
            }
            ty::Array(elem, _) | ty::Slice(elem) | ty::Pat(elem, _) => stack.push(elem),
            ty::Tuple(tys) => stack.extend(tys),
            ty::Closure(_, args) => stack.extend(args.as_closure().upvar_tys()),
            ty::CoroutineClosure(_, args) => {
                stack.extend(args.as_coroutine_closure().upvar_tys());
            }
            ty::Coroutine(def, args) => {
                stack.extend(args.as_coroutine().upvar_tys());
                if let Some(layout) = tcx.mir_coroutine_witnesses(def) {
                    stack.extend(layout.field_tys.iter().filter_map(|saved| {
                        tcx.try_instantiate_and_normalize_erasing_regions(
                            args,
                            env,
                            ty::EarlyBinder::bind(saved.ty),
                        )
                        .ok()
                    }));
                }
            }
            _ => {}
        }
    }
    local
}

/// The generic fns expression `e` uses, each with the argument list typeck
/// inferred for it (the callee's whole list, `impl` parameters first) and
/// the span to cite. What `e` itself names comes first: a path names a fn
/// item through its `FnDef` type, which carries the arguments even where
/// typeck recorded none against the node (the `IntoIterator::into_iter` and
/// `Iterator::next` a `for` loop desugars to); a method call, and the
/// operators, indexing and explicit `*x` that resolve to a trait method,
/// record theirs against the node. Then the derefs typeck inserted on `e`'s
/// value to reach a method's receiver type, a field or a coercion's target:
/// those are adjustments on `e`, not expressions, and each overloaded step
/// is a call to `Deref::deref` or `DerefMut::deref_mut` on the type the step
/// starts from.
fn fn_uses<'tcx>(
    tcx: TyCtxt<'tcx>,
    typeck: &TypeckResults<'tcx>,
    e: &Expr<'_>,
    mut each: impl FnMut(DefId, GenericArgsRef<'tcx>, Span),
) {
    let Some(ty) = typeck.expr_ty_opt(e) else {
        return;
    };
    let named = match e.kind {
        ExprKind::Path(..) => match *ty.kind() {
            ty::FnDef(def, args) => Some((def, args)),
            _ => None,
        },
        _ => typeck
            .type_dependent_def_id(e.hir_id)
            .map(|def| (def, typeck.node_args(e.hir_id))),
    };
    // Constructors are `FnDef`s too, and an argument list that does not line
    // up with the callee's generics cannot be resolved or substituted.
    if let Some((def, args)) = named
        && matches!(tcx.def_kind(def), DefKind::Fn | DefKind::AssocFn)
        && args.len() == tcx.generics_of(def).count()
    {
        each(def, args, use_site(tcx, e));
    }
    let mut source = ty;
    for adjustment in typeck.expr_adjustments(e) {
        if let Adjust::Deref(DerefAdjustKind::Overloaded(deref)) = adjustment.kind {
            each(
                deref.method_call(tcx),
                tcx.mk_args(&[source.into()]),
                e.span,
            );
        }
        source = adjustment.target;
    }
}

/// The span to show for a use: the whole call when `e` is the callee path
/// of one, else the expression itself (a method call, a fn item passed as a
/// value).
fn use_site(tcx: TyCtxt<'_>, e: &Expr<'_>) -> Span {
    match tcx.parent_hir_node(e.hir_id) {
        Node::Expr(
            call @ Expr {
                kind: ExprKind::Call(callee, _),
                ..
            },
        ) if callee.hir_id == e.hir_id => call.span,
        _ => e.span,
    }
}

/// Every use of a generic fn in the crate, direct ones recorded as concrete
/// argument sets and the ones inside generic bodies kept for `propagate`.
fn census<'tcx>(tcx: TyCtxt<'tcx>) -> Census<'tcx> {
    let mut census = Census {
        drop_fn: drop_fn(tcx),
        ..Census::default()
    };
    for owner in tcx.hir_body_owners() {
        if !tcx.has_typeck_results(owner) {
            continue;
        }
        // A `const` or `static` initializer, an array length, a discriminant
        // or a `const {}` block is evaluated at compile time: what it calls is
        // interpreted, not compiled into the binary. A `const fn`'s body is
        // kept, because a `const fn` called at run time is compiled like any
        // other.
        if let Some(ConstContext::Const { .. } | ConstContext::Static(_)) =
            tcx.hir_body_const_context(owner)
        {
            continue;
        }
        // A closure is its own body owner but is typed with its enclosing
        // fn, whose parameters are the ones its argument lists mention.
        let caller = tcx.typeck_root_def_id_local(owner);
        let typeck = tcx.typeck(owner);
        let body = tcx.hir_body_owned_by(owner);
        for_each_expr_without_closures(body.value, |e| {
            fn_uses(tcx, typeck, e, |callee, args, site| {
                census.file(tcx, caller, Use::Fn(callee, args), site);
            });
            ControlFlow::<()>::Continue(())
        });
        // What the body drops has no HIR node; MIR has a terminator per
        // value dropped, whose place is typed as that value. Drops in unwind
        // cleanup count too: glue that only runs during a panic is compiled
        // all the same. The site cited is the dropped local's declaration.
        if let Some(mir) = mir_for(tcx, owner) {
            for data in mir.basic_blocks.iter() {
                if let Some(terminator) = &data.terminator
                    && let TerminatorKind::Drop { place, .. } = terminator.kind
                {
                    let dropped = place.ty(&mir.local_decls, tcx).ty;
                    let site = mir.local_decls[place.local].source_info.span;
                    census.file(tcx, caller, Use::Drop(dropped), site);
                }
            }
        }
    }
    propagate(tcx, &mut census);
    census
}

/// Follows uses inside generic bodies through their callers' concrete
/// argument sets to a fixpoint. A worklist of instantiations, seeded with
/// every set recorded directly: taking one off substitutes its arguments
/// into each use inside that fn's body, records where the use lands, and
/// queues the result when it is a set not seen before. Every instantiation
/// is expanded once, so the work done follows the number found, and what is
/// found is the same whichever order bodies were walked or items declared
/// in -- a chain of generic callers is followed to its end whether it was
/// written caller-first or callee-first.
///
/// The one thing that never reaches a fixpoint is polymorphic recursion
/// (`f::<T>` calling `f::<W<T>>`): every step is a new, deeper set. rustc
/// refuses to build that once instantiation passes the recursion limit, and
/// a derived use is dropped here by the same measure -- when its arguments
/// or its dropped type nest deeper than `nesting_limit` -- so the walk stops
/// on such a program
/// instead of spinning. The test is on the set itself, not on how many
/// steps led to it, so it cuts no chain short and drops nothing a program
/// that builds instantiates: those types all passed rustc's own limit.
fn propagate<'tcx>(tcx: TyCtxt<'tcx>, census: &mut Census<'tcx>) {
    let limit = nesting_limit(tcx);
    let propagating = std::mem::take(&mut census.propagating);
    let mut uses: FxHashMap<LocalDefId, Vec<&Propagating<'tcx>>> = FxHashMap::default();
    for edge in &propagating {
        uses.entry(edge.caller).or_default().push(edge);
    }
    let mut nesting = Nesting::default();
    while let Some((caller, caller_args)) = census.queue.pop_front() {
        let Some(edges) = uses.get(&caller) else {
            continue;
        };
        for edge in edges {
            let Some(used) = edge.used.instantiate(tcx, caller_args) else {
                continue;
            };
            if used.has_non_region_param() || used.nesting(&mut nesting) > limit {
                continue;
            }
            census.record(tcx, used, edge.site);
        }
    }
}

/// How deep an argument set derived in `propagate` may nest before it is
/// taken for polymorphic recursion and dropped: rustc's recursion limit (the
/// crate's `#![recursion_limit]`, 128 unless raised), the depth at which the
/// monomorphization collector gives up on the same program and the trait
/// solver on the same type. A crate whose real types nest anywhere near it
/// has had to raise the limit to compile at all, and the bound rises with
/// it.
fn nesting_limit(tcx: TyCtxt<'_>) -> usize {
    tcx.recursion_limit().0
}

/// The nesting depth of types and consts: `u8` is 1, `Vec<u8>` 2,
/// `&[Vec<u8>]` 4. Memoized on the interned node, because polymorphic
/// recursion can double a type's written size per step (`f::<T>` calling
/// `f::<(T, T)>`) while interning stores each distinct subtree once; the
/// memo keeps the walk proportional to the distinct subtrees.
#[derive(Default)]
struct Nesting<'tcx> {
    depth: FxHashMap<GenericArg<'tcx>, usize>,
}

impl<'tcx> Nesting<'tcx> {
    fn of_args(&mut self, args: GenericArgsRef<'tcx>) -> usize {
        args.iter().map(|arg| self.of(arg)).max().unwrap_or(0)
    }

    fn of(&mut self, arg: GenericArg<'tcx>) -> usize {
        if let Some(&depth) = self.depth.get(&arg) {
            return depth;
        }
        let mut children = Children(Vec::new());
        match arg.kind() {
            GenericArgKind::Type(ty) => ty.super_visit_with(&mut children),
            GenericArgKind::Const(ct) => ct.super_visit_with(&mut children),
            GenericArgKind::Lifetime(_) => return 0,
        }
        let below = ensure_sufficient_stack(|| {
            children
                .0
                .into_iter()
                .map(|child| self.of(child))
                .max()
                .unwrap_or(0)
        });
        self.depth.insert(arg, below + 1);
        below + 1
    }
}

/// Collects the types and consts directly inside one type or const, without
/// descending into them.
struct Children<'tcx>(Vec<GenericArg<'tcx>>);

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Children<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        self.0.push(ty.into());
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) {
        self.0.push(ct.into());
    }
}

// ── the sharper help ─────────────────────────────────────────────────────────
//
// One shape gets a second help line, because for it the general advice --
// move the stretch into an inner fn, keep the generic fn as the shim --
// undersells what the body shows. It is the fn whose every use of a generic
// argument is to turn it, once, on the way in, into a value of a concrete
// type -- `let src = bytes.as_ref();`, `let s: &str = x.foo();` -- and, when
// it took the argument by value, to drop it on the way out. Such a fn is
// generic only to spare its callers the conversion; it could take the
// converted value instead, and then there is nothing left to instantiate.
// That is only a suggestion where the signature is the fn's own: a free fn or
// an inherent method. A trait's provided method, or a method of a trait impl
// (`self` in a blanket `impl<T: HasName> Digest for T` is opaque exactly
// there), cannot trade `&self` for `&str`, so it gets the general help alone.
//
// The claim is about the whole body and is checked against the classifier's
// own test, item by item: every counted statement or terminator a parameter
// appears in must be (a) the conversion -- a call to a trait method with the
// argument, or a borrow of it, as its one operand, returning normally into a
// bare local whose type no parameter appears in; (b) the borrow or move that
// carries the argument to that operand (`_3 = &_1`, `_4 = &(*_3)`,
// `_3 = move _1`), which may chain; or (c) a `Drop` of the argument or of
// such a carrier. One dependent item that is none of these -- a second
// operand, a `size_of::<T>()`, a field of the argument read, the argument
// returned or stored, a constant built per instantiation -- and the line is
// withheld, since the fn then does not *only* convert. The argument has to be
// opaque: its type with the references peeled is the parameter itself
// (`bytes: B`, `text: &S`), because only then is every use of it an item the
// test sees as dependent; a `&Framed<T>` read for its `[u8; 4]` header is
// used, independently, for something no conversion explains. Every opaque
// argument must be converted at least once, or the sentence would pass over
// one in silence. And the conversion and its carriers must sit on the way
// in: in the straight run of blocks from the entry before the first branch,
// loop head or return, which every call executes exactly once -- so a caller
// handed the job calls the method exactly as often as the fn did. A
// conversion inside a loop or behind an
// `if`, one taking a second argument (`f()` on an `F: FnOnce()` is
// `call_once(f, ())`), one through an inherent or free fn, one yielding `()`,
// or one of a field, gets the general help alone. Missing a shape costs a
// hint; matching one wrongly would print a false sentence.

/// One conversion the fn makes of an opaque generic argument: `bytes` turned
/// into a `&[u8]` through `AsRef::as_ref`.
struct Conversion<'tcx> {
    /// The argument converted, as the body numbers it.
    arg: mir::Local,
    /// Trait and method, unqualified: `AsRef::as_ref`, `HasFoo::foo`.
    through: String,
    /// The type that comes out; no parameter appears in it.
    obtains: Ty<'tcx>,
}

/// The blocks every call of the fn executes exactly once, in order: from the
/// entry block along terminators with one normal successor -- a jump, a call
/// that returns, a drop, an assert that holds -- for as long as the next
/// block has no other way in. The first `SwitchInt`, loop head, diverging
/// call or return ends the run.
fn entry_run(body: &mir::Body<'_>) -> Vec<BasicBlock> {
    let predecessors = body.basic_blocks.predecessors();
    let mut run = Vec::new();
    // Built MIR never jumps back to its entry block (a `loop` at the top of a
    // fn gets a head block of its own); a body where something did is not one
    // this reasons about.
    if !predecessors[mir::START_BLOCK].is_empty() {
        return run;
    }
    let mut block = mir::START_BLOCK;
    // A block with exactly one predecessor, reached from a run that starts at
    // a block with none, cannot close a cycle; the length bound only makes
    // that visible.
    while run.len() <= body.basic_blocks.len() {
        run.push(block);
        let next = match body.basic_blocks[block].terminator().kind {
            TerminatorKind::Goto { target }
            | TerminatorKind::Drop { target, .. }
            | TerminatorKind::Assert { target, .. }
            | TerminatorKind::Call {
                target: Some(target),
                ..
            } => target,
            _ => break,
        };
        if predecessors[next].len() != 1 || body.basic_blocks[next].is_cleanup {
            break;
        }
        block = next;
    }
    run
}

/// The conversions `body` makes of its opaque generic arguments, when
/// converting (and dropping) them is everything it does with any parameter
/// and its signature is its own to change; empty otherwise. One pass over
/// the body with the classifier's test, made only for a fn about to be
/// reported.
fn only_conversions<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> Vec<Conversion<'tcx>> {
    // A trait's method, provided or implemented, has the trait's signature.
    let def = body.source.def_id();
    if tcx.trait_of_assoc(def).is_some() || tcx.trait_impl_of_assoc(def).is_some() {
        return Vec::new();
    }
    let decls = &body.local_decls;
    let opaque = |local: mir::Local| {
        body.local_kind(local) == mir::LocalKind::Arg
            && matches!(decls[local].ty.peel_refs().kind(), ty::Param(_))
    };
    if !body.args_iter().any(opaque) {
        return Vec::new();
    }
    // Carrier temp -> the argument it borrows or holds.
    let mut carriers: FxHashMap<mir::Local, mir::Local> = FxHashMap::default();
    // The argument a place stands for: an opaque argument or a carrier, bare
    // or dereferenced, with nothing else projected.
    let argument_behind =
        |carriers: &FxHashMap<mir::Local, mir::Local>, place: Place<'tcx>| -> Option<mir::Local> {
            if !place
                .projection
                .iter()
                .all(|elem| matches!(elem, mir::ProjectionElem::Deref))
            {
                return None;
            }
            if opaque(place.local) {
                Some(place.local)
            } else {
                carriers.get(&place.local).copied()
            }
        };
    let run = entry_run(body);
    let mut on_way_in: IndexVec<BasicBlock, bool> =
        IndexVec::from_elem_n(false, body.basic_blocks.len());
    for &block in &run {
        on_way_in[block] = true;
    }
    // The entry run first and in execution order, so every carrier is known
    // before the item that uses it; then the rest, where only drops may
    // mention a parameter.
    let rest = body
        .basic_blocks
        .indices()
        .filter(|&block| !on_way_in[block] && !body.basic_blocks[block].is_cleanup);
    let order: Vec<BasicBlock> = run.iter().copied().chain(rest).collect();
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let mut found: Vec<Conversion<'tcx>> = Vec::new();
    for block in order {
        let data = &body.basic_blocks[block];
        let early = on_way_in[block];
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            if !counted_statement(statement) || !places.dependent_statement(statement, at(index)) {
                continue;
            }
            // (b) a carrier: `_k = &_1`, `_k = &(*_j)`, `_k = move _1`, into
            // a temp (not the return place, not an argument overwritten).
            let carried = if early
                && let Some((target, rvalue)) = statement.kind.as_assign()
                && let Some(temp) = target.as_local()
                && body.local_kind(temp) == mir::LocalKind::Temp
                && let mir::Rvalue::Ref(_, mir::BorrowKind::Shared, place)
                | mir::Rvalue::CopyForDeref(place)
                | mir::Rvalue::Use(mir::Operand::Move(place) | mir::Operand::Copy(place), _) =
                    rvalue
            {
                argument_behind(&carriers, *place).map(|arg| (temp, arg))
            } else {
                None
            };
            let Some((temp, arg)) = carried else {
                return Vec::new();
            };
            carriers.insert(temp, arg);
        }
        let Some(terminator) = &data.terminator else {
            continue;
        };
        if !counted_terminator(terminator)
            || !places.dependent_terminator(terminator, at(data.statements.len()))
        {
            continue;
        }
        match &terminator.kind {
            // (c) the argument, or a temp it was moved into, going out of
            // scope -- anywhere in the body.
            TerminatorKind::Drop { place, .. }
                if place
                    .as_local()
                    .is_some_and(|local| opaque(local) || carriers.contains_key(&local)) => {}
            // (a) the conversion.
            TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(_),
                ..
            } if early => {
                let conversion = if let [operand] = &args[..]
                    && let Some(place) = operand.node.place()
                    && let Some(arg) = argument_behind(&carriers, place)
                    && let Some((callee, _)) = func.const_fn_def()
                    && let Some(trait_id) = tcx.trait_of_assoc(callee)
                    && let Some(obtained) = destination.as_local()
                    && !decls[obtained].ty.has_non_region_param()
                    && !decls[obtained].ty.is_unit()
                    && !decls[obtained].ty.is_never()
                {
                    Some(Conversion {
                        arg,
                        // `item_name` reads a symbol, not the printer, so it
                        // is safe to call before the finding is weighed.
                        through: format!("{}::{}", tcx.item_name(trait_id), tcx.item_name(callee)),
                        obtains: decls[obtained].ty,
                    })
                } else {
                    None
                };
                let Some(conversion) = conversion else {
                    return Vec::new();
                };
                found.push(conversion);
            }
            _ => return Vec::new(),
        }
    }
    // Every opaque argument converted at least once.
    if body
        .args_iter()
        .filter(|&arg| opaque(arg))
        .any(|arg| !found.iter().any(|c| c.arg == arg))
    {
        return Vec::new();
    }
    found
}

/// The name the source gives argument `arg` (`bytes`, `self`), or, for a
/// pattern in argument position that has none, its type.
fn arg_name(body: &mir::Body<'_>, arg: mir::Local) -> String {
    let named = body
        .var_debug_info
        .iter()
        .find_map(|info| match info.value {
            mir::VarDebugInfoContents::Place(place)
                if place.as_local() == Some(arg) && info.composite.is_none() =>
            {
                Some(info.name)
            }
            mir::VarDebugInfoContents::Place(_) | mir::VarDebugInfoContents::Const(_) => None,
        });
    match named {
        Some(name) => format!("`{name}`"),
        None => with_no_trimmed_paths!(format!("its `{}` argument", body.local_decls[arg].ty)),
    }
}

/// The second help line, for a non-empty `only_conversions`, less its
/// subject: "only uses `text` to obtain a `&str` through `AsRef::as_ref`; it
/// could take `&str` instead". The fn's name goes in front when the finding
/// is emitted (`report`), not here: this runs while the body is searched,
/// before the census has said whether the fn is reported at all, and
/// printing a def path asks rustc's trimmed-path printer, which insists a
/// diagnostic follow. Several conversions of one argument, and several
/// arguments, are listed in argument order with repeats folded; every type is
/// printed untrimmed, like the instantiation note's.
fn conversion_help(body: &mir::Body<'_>, conversions: &[Conversion<'_>]) -> Option<String> {
    let mut clauses: Vec<String> = Vec::new();
    // Distinct (type, method) pairs listed, and the one type to name when
    // there is exactly one of them.
    let mut listed = 0;
    let mut single: Option<String> = None;
    for arg in body.args_iter() {
        let mut ways: Vec<String> = Vec::new();
        for conversion in conversions.iter().filter(|c| c.arg == arg) {
            let ty = with_no_trimmed_paths!(format!("`{}`", conversion.obtains));
            let way = format!("a {ty} through `{}`", conversion.through);
            if !ways.contains(&way) {
                ways.push(way);
                listed += 1;
                single = Some(ty);
            }
        }
        if !ways.is_empty() {
            clauses.push(format!(
                "{} to obtain {}",
                arg_name(body, arg),
                join(&ways, "and")
            ));
        }
    }
    let instead = match (listed, single) {
        (1, Some(one)) => one,
        (0, _) | (_, None) => return None,
        _ => "those".to_owned(),
    };
    Some(format!(
        "only uses {}; it could take {instead} instead",
        join(&clauses, "and")
    ))
}

// ── where a region is in the source ──────────────────────────────────────────
//
// A region is a set of MIR blocks; the reader needs a place in the file. Every
// counted item carries the span of the source it was lowered from, so to a
// first approximation the region's place is the hull of its items' spans,
// first to last. Three things spoil the approximation, and the fixtures' MIR
// (dumped with `-Zdump-mir -Zmir-include-spans`) shows all three.
//
// An item lowered from a macro's expansion carries a position inside the
// macro's definition, wherever that is; what sits in the body is the
// invocation, so such a span is walked out to the innermost call site that
// lands in the body. A desugaring is not treated that way: the `into_iter`
// and `next` calls a `for` loop becomes carry the loop head's own position
// (in a marked context, but in place), a `?` its operand's, and walking those
// out would turn each into the whole `for` or `?` expression, so a desugared
// span is kept where it is. An item with no position in the body either way
// (nothing the fixtures produce, but a span is not obliged to have one) takes
// no part.
//
// Some items carry the span of a construct rather than of code. The `()` a
// block, a loop or an `else`-less `if` evaluates to is an assignment spanning
// the whole construct; a unit fn's `_0 = const ()` spans the whole body; the
// `x = a + f(b)` that consumes a call's result spans the call, which ended
// the block before. In the hull such a span stretches it over everything the
// construct encloses, region or not -- over the parameter-dependent prefix
// the region had to start after, say. So an item whose span contains the
// span of any counted item *outside* the region is left out of the hull: it
// names more than the region holds. The one exception is a span the region
// shares exactly with a parameter-free item outside, which is one source
// expression compiled to items on both sides of the region's edge -- the
// `for` head is both the `into_iter` before the loop and the `next` inside
// it -- and says nothing about which side the source is on. Shared with a
// dependent item it says plenty: `for v in S::make()` puts the dependent
// call, the `into_iter` and the `next` all at the head's span, and the head
// is `S`'s however many parameter-free items it also compiled to, so a
// dependent item is never excepted, and the region's items at its span go.
//
// And source order is not control-flow order. A `for` pattern is written
// before the loop head but bound inside the loop body, so the body of a loop
// over a generic iterator -- a region on its own, entered once per item --
// has the dependent `next` call sitting between its first item and the rest.
// What the span shown must never do is underline something that depends on
// a parameter as if it could move, so the dependent items outside the region
// cut the body into pieces, each remaining item falls into the piece its
// start is in, and the piece holding the most items is the stretch reported.
// Nothing dependent lies inside it: an item starting before a cut and
// reaching past it would contain the cut item and is already gone, and an
// item starting exactly at a cut goes with the piece before, which therefore
// ends short of the cut item's end. Parameter-free items outside the region
// can lie inside it -- where a source statement straddles the region's first
// or last block boundary its other half is underlined too -- and are
// the same code at every instantiation whichever side they are on.
//
// Checked against the dumps: `checksum`'s region (everything between
// `bytes.as_ref()` and `drop(bytes)`) comes out as `17u32` through
// `src.len()`, swallowing neither; a unit method's whole-body `_0 = ()` and a
// loop's `()` are dropped wherever they would reach over the prefix; a loop
// body under a generic iterator comes out as the body's own statements,
// without the pattern that precedes the head.
//
// When the best piece holds less than half of the region's placed items, one
// underline would misrepresent the region, and the site falls back to where
// the region starts -- its first remaining item by position -- under a note
// that says so: "the stretch, N statements starting here, ...".
//
// The site goes on the finding as a `span_note`, not a label on the primary
// span: the note's sentence (what the stretch uses and yields, what the call
// costs inside a loop) is long, and under its own `note:` heading it reads
// as a sentence, where at the end of a multi-line underline's closing
// bracket it would trail off the margin.

/// Where a region can be pointed at in the fn's source. Either way the span
/// is a secondary one: the finding's own span stays the fn's signature, which
/// is what the baseline keys on and what an `#[allow]` goes on.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RegionSite {
    /// One stretch of the body, from the region's first item in it to its
    /// last, holding at least half the region's items and nothing outside
    /// the region that depends on a parameter.
    Stretch(Span),
    /// No such stretch: the region's first item by position, for "the
    /// stretch, N statements starting here".
    Start(Span),
}

impl RegionSite {
    /// The span the note goes at.
    pub(crate) fn span(self) -> Span {
        match self {
            RegionSite::Stretch(span) | RegionSite::Start(span) => span,
        }
    }

    /// How the note at that span names the region, as the singular subject
    /// of its sentence: the span is the stretch itself, or only where its
    /// `size` counted statements start.
    pub(crate) fn subject(self, size: usize) -> String {
        match self {
            RegionSite::Stretch(_) => "this stretch".to_owned(),
            RegionSite::Start(_) => format!("the stretch, {size} statements starting here,"),
        }
    }

    /// Puts the note on a finding: `rest` is the predicate that finishes the
    /// sentence `subject` starts ("uses only `src: &[u8]` and yields the
    /// returned `u32`").
    pub(crate) fn note(self, diag: &mut Diag<'_, ()>, size: usize, rest: &str) {
        diag.span_note(self.span(), format!("{} {rest}", self.subject(size)));
    }
}

/// Where `span`, the source span of one MIR item, sits in the fn body as
/// written: a macro expansion is walked out to its invocation, a desugaring
/// stays where its source is, and the result is in the body's own context so
/// spans can be compared and joined by position. `None` when no call site on
/// the way out lands inside the body.
fn body_position(mut span: Span, body_span: Span) -> Option<Span> {
    loop {
        let in_macro = span.from_expansion()
            && matches!(span.ctxt().outer_expn_data().kind, ExpnKind::Macro(..));
        if !in_macro && body_span.contains(span) {
            return Some(span.with_ctxt(body_span.ctxt()));
        }
        span = span.parent_callsite()?;
    }
}

/// Where the region made of `blocks` of `def`'s `body` is in the source, by
/// the rules above; `None` when none of its counted items has a position in
/// the body, which leaves the finding nothing to point at but the fn.
pub(crate) fn region_site<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
) -> Option<RegionSite> {
    let body_span = tcx.hir_body_owned_by(def).value.span;
    // The region's own items need only their positions; everyone else's
    // need the dependence test too, put the way the classifier puts it.
    let mut inside: Vec<Span> = Vec::new();
    let mut outside: Vec<(Span, bool)> = Vec::new();
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup {
            continue;
        }
        if blocks.contains(block) {
            inside.extend(counted_spans(body, block).filter_map(|s| body_position(s, body_span)));
            continue;
        }
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            if counted_statement(statement)
                && let Some(span) = body_position(statement.source_info.span, body_span)
            {
                outside.push((span, places.dependent_statement(statement, at(index))));
            }
        }
        if let Some(terminator) = &data.terminator
            && counted_terminator(terminator)
            && let Some(span) = body_position(terminator.source_info.span, body_span)
        {
            let dependent = places.dependent_terminator(terminator, at(data.statements.len()));
            outside.push((span, dependent));
        }
    }
    let placed = inside.len();
    let by_position = |s: &Span| (s.lo(), std::cmp::Reverse(s.hi()));
    let first = *inside.iter().min_by_key(|s| by_position(s))?;

    // A span shared exactly with a parameter-free outside item is one
    // expression on both sides of the region's edge: no evidence either way.
    // Shared with a dependent one it is evidence -- the region cannot own
    // that expression -- so the dependent item stays, to drop the region's
    // items at that span from the hull and to cut there.
    let shared: FxHashSet<(BytePos, BytePos)> = inside.iter().map(|s| (s.lo(), s.hi())).collect();
    outside.retain(|(s, dependent)| *dependent || !shared.contains(&(s.lo(), s.hi())));

    // Drop the items that name a construct reaching outside the region: those
    // whose span contains an outside item's. With the outside sorted by start
    // and the least end from each index on precomputed, that is one binary
    // search per item.
    outside.sort_by_key(|(s, _)| s.lo());
    let mut least_end = vec![BytePos(u32::MAX); outside.len() + 1];
    for i in (0..outside.len()).rev() {
        least_end[i] = least_end[i + 1].min(outside[i].0.hi());
    }
    let names_more = |item: &Span| {
        let from = outside.partition_point(|(s, _)| s.lo() < item.lo());
        least_end[from] <= item.hi()
    };
    inside.retain(|item| !names_more(item));
    inside.sort_by_key(by_position);
    let Some(&first_kept) = inside.first() else {
        return Some(RegionSite::Start(first));
    };

    // Cut at every dependent outside item; an item starting exactly at a cut
    // belongs to the piece before it. One pass in position order builds the
    // pieces as (items held, start, end); the best-populated wins, the
    // earlier one on a tie.
    let cuts: Vec<BytePos> = outside
        .iter()
        .filter(|(_, dependent)| *dependent)
        .map(|(s, _)| s.lo())
        .collect();
    let piece_of = |item: &Span| cuts.partition_point(|&cut| cut < item.lo());
    let mut pieces: Vec<(usize, BytePos, BytePos)> = Vec::new();
    let mut open = None;
    for item in &inside {
        let piece = piece_of(item);
        match pieces.last_mut() {
            Some((held, _, hi)) if open == Some(piece) => {
                *held += 1;
                *hi = (*hi).max(item.hi());
            }
            _ => {
                pieces.push((1, item.lo(), item.hi()));
                open = Some(piece);
            }
        }
    }
    let (held, lo, hi) =
        pieces
            .into_iter()
            .fold((0, first_kept.lo(), first_kept.hi()), |best, piece| {
                if piece.0 > best.0 { piece } else { best }
            });
    debug_assert!(
        !outside
            .iter()
            .any(|(s, dependent)| *dependent && lo <= s.lo() && s.hi() <= hi),
        "a dependent item inside the region's span"
    );
    Some(if held * 2 >= placed {
        RegionSite::Stretch(body_span.with_lo(lo).with_hi(hi))
    } else {
        RegionSite::Start(first_kept)
    })
}

// ── the report ───────────────────────────────────────────────────────────────

/// The type and const parameters in scope for `def`, `impl` ones first, as
/// written.
fn param_names(tcx: TyCtxt<'_>, def: LocalDefId) -> Vec<String> {
    let generics = tcx.generics_of(def);
    (0..generics.count())
        .map(|i| generics.param_at(i, tcx))
        .filter(|p| !matches!(p.kind, GenericParamDefKind::Lifetime))
        .map(|p| format!("`{}`", p.name))
        .collect()
}

/// `T = u32, N = 4` for one concrete argument set of `def`.
fn render_args<'tcx>(tcx: TyCtxt<'tcx>, def: LocalDefId, args: GenericArgsRef<'tcx>) -> String {
    let generics = tcx.generics_of(def);
    let pairs: Vec<String> = args
        .iter()
        .enumerate()
        .filter_map(|(i, arg)| {
            let param = generics.param_at(i, tcx);
            (!matches!(param.kind, GenericParamDefKind::Lifetime))
                .then(|| with_no_trimmed_paths!(format!("{} = {arg}", param.name)))
        })
        .collect();
    pairs.join(", ")
}

/// The name the source gives a whole MIR local, when it is a variable the
/// user wrote: the debuginfo entry that maps a name to exactly this local --
/// no field of it, no fragment of a variable rustc split up. A name a
/// desugaring invented (`iter` for a `for` loop's iterator, `val` and
/// `residual` for a `?`) comes from an expansion and is not the user's, so
/// that local is described by its type alone. Arguments have entries like
/// any `let`. The rule is the one `unchecked_input_len` names its seeds by
/// (`user_name_of`, a method on that lint's own state).
fn local_name(body: &mir::Body<'_>, local: Local) -> Option<String> {
    body.var_debug_info
        .iter()
        .find_map(|info| match info.value {
            mir::VarDebugInfoContents::Place(place)
                if place.as_local() == Some(local)
                    && info.composite.is_none()
                    && !info.source_info.span.from_expansion() =>
            {
                Some(info.name.to_string())
            }
            mir::VarDebugInfoContents::Place(_) | mir::VarDebugInfoContents::Const(_) => None,
        })
}

/// Where the source puts the value of a temporary that crosses a stretch's
/// edge: the name it ends up under, read off the one statement on the other
/// side of the edge that consumes the temporary whole.
enum Landing {
    /// Copied or moved as it is into a variable the user named (the `n` of a
    /// `let n = { .. }`), or, for a reference, reborrowed into one at the
    /// same type (the `text` of `let text: &[u8] = &buf[a..b]`, which binds
    /// what the indexing call returned).
    Let(String),
    /// Moved into a named field, of a struct literal (`Settings { height:
    /// <it>, .. }`) or by assignment (`self.col = <it>`).
    Field(String),
}

/// The name of the field `place` ends at, when its last step is a field of a
/// struct, a union or an enum variant. A tuple's or a closure's fields have
/// only positions, which name nothing a reader wrote.
fn field_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    place: Place<'tcx>,
) -> Option<String> {
    let (base, mir::ProjectionElem::Field(field, _)) = place.as_ref().last_projection()? else {
        return None;
    };
    let base_ty = base.ty(body, tcx);
    let ty::Adt(def, _) = *base_ty.ty.kind() else {
        return None;
    };
    let variant = match base_ty.variant_index {
        Some(index) => def.variant(index),
        None if def.is_enum() => return None,
        None => def.non_enum_variant(),
    };
    Some(variant.fields.get(field)?.name.to_string())
}

/// Where the temporary `local` lands, judged from the blocks `among` (for a
/// value a stretch hands back, the blocks past its exit; for one it takes,
/// its own): the single statement there that uses the local, when that
/// statement copies or moves it whole (or reborrows it, type unchanged) into
/// a named variable, into a named field of an aggregate being built, or into
/// a named field by assignment.
/// Used twice, used in part (`_5.0` out of a checked-add pair), passed to a
/// call or folded into a larger expression, it lands nowhere this can name
/// and the caller falls back to quoting the expression it holds.
fn landing<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    among: &DenseBitSet<BasicBlock>,
    local: Local,
) -> Option<Landing> {
    let mut of = DenseBitSet::new_empty(body.local_decls.len());
    of.insert(local);
    let mut user: Option<&Statement<'tcx>> = None;
    for block in among.iter() {
        let data = &body.basic_blocks[block];
        if data.is_cleanup {
            continue;
        }
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            let mut uses = UsesAny {
                of: &of,
                found: false,
            };
            uses.visit_statement(statement, at(index));
            // A statement that writes the local is where it comes from, not
            // where it goes.
            let writes = match &statement.kind {
                StatementKind::Assign(assign) => assign.0.as_local() == Some(local),
                _ => false,
            };
            if uses.found && !writes {
                if user.is_some() {
                    return None;
                }
                user = Some(statement);
            }
        }
        if let Some(terminator) = &data.terminator {
            let mut uses = UsesAny {
                of: &of,
                found: false,
            };
            uses.visit_terminator(terminator, at(data.statements.len()));
            let writes = matches!(
                &terminator.kind,
                TerminatorKind::Call { destination, .. } if destination.local == local
            );
            if uses.found && !writes {
                return None;
            }
        }
    }
    let StatementKind::Assign(assign) = &user?.kind else {
        return None;
    };
    let (dest, rvalue) = &**assign;
    let whole = |operand: &Operand<'tcx>| match operand {
        Operand::Copy(place) | Operand::Move(place) => place.as_local() == Some(local),
        _ => false,
    };
    match rvalue {
        Rvalue::Use(operand, _) if whole(operand) => match dest.as_local() {
            Some(var) => local_name(body, var).map(Landing::Let),
            None => field_name(tcx, body, *dest).map(Landing::Field),
        },
        // `let text: &[u8] = &buf[a..b]` binds `text` by reborrowing the
        // reference the indexing call returned: the same value under the
        // user's name, when the reborrow keeps its type.
        Rvalue::Ref(_, _, place)
            if place.local == local
                && matches!(place.projection[..], [mir::ProjectionElem::Deref]) =>
        {
            let var = dest.as_local()?;
            (body.local_decls[var].ty == body.local_decls[local].ty)
                .then(|| local_name(body, var).map(Landing::Let))?
        }
        Rvalue::Aggregate(kind, operands) => {
            let mir::AggregateKind::Adt(adt, variant, _, _, None) = **kind else {
                return None;
            };
            let index = operands.iter().position(whole)?;
            let field = tcx
                .adt_def(adt)
                .variant(variant)
                .fields
                .get(rustc_abi::FieldIdx::from_usize(index))?;
            Some(Landing::Field(field.name.to_string()))
        }
        _ => None,
    }
}

/// One value crossing a stretch's edge, as the finding names it:
/// `` `acc: u32` `` for a variable; for a value the source never names, the
/// type and where it goes or comes from -- the fn's return slot is "the
/// returned `u32`"; a temporary the other side of the edge (`among`) moves
/// whole into something named is "the `u32` assigned to `total`" or "the
/// `u32` for field `height`" (`landing`), so that a struct literal's seven
/// initializers are seven fields and not seven temporaries; failing that, a
/// temporary holding an expression written in this body is "the `usize` from
/// `src.len()`" when that expression is short enough to quote, and anything
/// else (a desugaring's temporary, a long expression) is "a `usize`
/// temporary". Types are printed with full paths, as the instantiation note
/// prints its arguments, and for the same reason: the trimmed-path printer
/// insists a diagnostic follow it, and under the baseline this finding may
/// yet be swallowed. They are the body's own types, regions erased, so a
/// borrowed slice prints as `&[u8]`.
fn render_local<'tcx>(
    cx: &LateContext<'tcx>,
    body: &mir::Body<'tcx>,
    body_span: Span,
    among: &DenseBitSet<BasicBlock>,
    local: Local,
) -> String {
    let decl = &body.local_decls[local];
    let ty = with_no_trimmed_paths!(decl.ty.to_string());
    if let Some(name) = local_name(body, local) {
        return format!("`{name}: {ty}`");
    }
    if local == RETURN_PLACE {
        return format!("the returned `{ty}`");
    }
    match landing(cx.tcx, body, among, local) {
        Some(Landing::Let(name)) => return format!("the `{ty}` assigned to `{name}`"),
        Some(Landing::Field(name)) => return format!("the `{ty}` for field `{name}`"),
        None => {}
    }
    // A temporary's declaration span is the expression it holds; the return
    // slot's is the return type and an argument's its pattern, both outside
    // the body block, which is why containment is asked of the block.
    let span = decl.source_info.span;
    if !span.from_expansion()
        && body_span.contains(span)
        && let Some(text) = snippet_opt(cx, span)
        && !text.contains('\n')
        && text.len() <= 40
    {
        return format!("the `{ty}` from `{text}`");
    }
    format!("a `{ty}` temporary")
}

/// Locals of `def`'s `body` rendered for the finding, in the order given: a
/// region's live-ins or live-outs as `region_io` lists them, `among` the
/// blocks on the other side of the edge they cross (`render_local`). A `()`
/// local carries nothing and is left out (MIR's `return` reads `_0` even in
/// a fn that returns unit, so a `-> ()` stretch reaching it "yields" one);
/// two locals that render alike (a shadowed `let c`, both live) are said
/// once.
pub(crate) fn render_locals<'tcx>(
    cx: &LateContext<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    locals: &[Local],
    among: &DenseBitSet<BasicBlock>,
) -> Vec<String> {
    // The block, not the whole item: the signature's slots (an argument's
    // pattern, the return type) are not expressions to quote.
    let body_span = cx.tcx.hir_body_owned_by(def).value.span;
    let mut out: Vec<String> = Vec::new();
    for &local in locals {
        if body.local_decls[local].ty.is_unit() {
            continue;
        }
        let shown = render_local(cx, body, body_span, among, local);
        if !out.contains(&shown) {
            out.push(shown);
        }
    }
    out
}

/// The stretch a finding reports, reduced to what the diagnostic prints: the
/// region search's answer with a place in the source (`region_site`) and
/// names put to its locals (`render_locals` over `RegionIo`'s two lists).
/// Built only for a fn that is about to be reported, so the snippets and
/// type strings it holds are made a handful of times per crate.
pub(crate) struct Stretch {
    /// Where it is in the source; `None` when none of its items has a
    /// position there, and the fn's own span is all there is to point at.
    pub(crate) site: Option<RegionSite>,
    /// Counted statements and terminators inside it.
    pub(crate) size: usize,
    /// The values it reads that the code before it computed: the inner fn's
    /// parameters.
    pub(crate) uses: Vec<String>,
    /// The values it computes that the code after it reads: what the inner
    /// fn returns.
    pub(crate) yields: Vec<String>,
    /// Control can come back to its entry after leaving it, so the call that
    /// replaces it runs once per iteration rather than once per call of the
    /// fn.
    pub(crate) in_loop: bool,
    /// Further stretches in the same body, disjoint from this one, that
    /// would have qualified on their own.
    pub(crate) other_regions: usize,
}

/// The finding's sentence: what the fn is generic over, how many times this
/// crate compiles it, how much of its body is the one stretch every copy
/// repeats, and the three facts that make that stretch movable, so the
/// reader knows what was checked and what was not.
fn finding_message(
    tcx: TyCtxt<'_>,
    def: LocalDefId,
    sets: usize,
    size: usize,
    total: usize,
) -> String {
    let name = tcx.def_path_str(def);
    let params = param_names(tcx, def);
    let them = match &params[..] {
        [one] => one.clone(),
        _ => "them".to_owned(),
    };
    format!(
        "`{name}` is generic over {} and compiled {sets} times in this crate, once per distinct \
         set of arguments, but {} of the {total} counted statements in its body (as MIR, before \
         optimization) are one stretch that is the same in every copy: it never mentions {}, \
         control enters it once and leaves it once, and the values it uses and yields have types \
         free of {them}",
        join(&params, "and"),
        size,
        join(&params, "or"),
    )
}

/// The note at the stretch, after its subject ("this stretch ..."): what it
/// takes from the code before it and hands the code after it -- the inner
/// fn's parameters and results, by name and type -- then, when they apply,
/// the clauses that change what the fix costs or finds: a stretch inside a
/// loop is one call per iteration, and a body with more qualifying stretches
/// than the one shown says how many. "uses only `acc: u32`, `src: &[u8]` and
/// yields the returned `u32`; it sits inside a loop, so the call that
/// replaces it runs once per iteration".
fn stretch_clauses(stretch: &Stretch, min_statements: usize) -> String {
    let taken = match &stretch.uses[..] {
        [] => "uses nothing computed before it".to_owned(),
        list => format!("uses only {}", list.join(", ")),
    };
    let handed = match &stretch.yields[..] {
        [] => "yields nothing read after it".to_owned(),
        list => format!("yields {}", list.join(", ")),
    };
    let mut rest = format!("{taken} and {handed}");
    if stretch.in_loop {
        rest.push_str(
            "; it sits inside a loop, so the call that replaces it runs once per iteration",
        );
    }
    match stretch.other_regions {
        0 => {}
        1 => rest.push_str(&format!(
            "; the body has 1 more such stretch of at least {min_statements} statements"
        )),
        more => rest.push_str(&format!(
            "; the body has {more} more such stretches of at least {min_statements} statements"
        )),
    }
    rest
}

/// The help: the one fix this lint stands behind, and its whole price.
fn finding_help(tcx: TyCtxt<'_>, def: LocalDefId) -> String {
    format!(
        "move the stretch into a non-generic fn that takes the values it uses and returns what \
         it yields, and call that fn in its place; `{}` keeps its signature and the cost is one \
         call",
        tcx.def_path_str(def)
    )
}

/// What the search of one fn's body found, kept for the report: the stretch
/// it settled on, the body's counted size, and the second help line when the
/// body has the conversion shape. Everything in it was rendered while the
/// body was borrowed; none of it holds the body, so the borrow is released
/// before the census reads the same bodies.
struct Searched {
    def: LocalDefId,
    stretch: Stretch,
    /// Counted statements and terminators in the whole body.
    total: usize,
    /// `conversion_help`'s line, less the fn's name.
    sharper: Option<String>,
}

/// One fn to report: what the search found, with the census's numbers for
/// the fn.
struct Finding<'tcx> {
    searched: Searched,
    sets: usize,
    site: Option<(Span, GenericArgsRef<'tcx>)>,
}

/// Emits one finding, exactly once, at the fn's own span -- the span the
/// baseline files it under and findings are sorted by -- with the stretch,
/// the instantiation that proves the count and any `#[inline]` caveat as
/// notes, in that order. `emit_hir_then` rather than `emit_with_note` because
/// there are two spans to cite, and because it reads the lint level at the
/// fn's own node: `check_crate_post` runs with the crate root as the current
/// node, where an `#[allow]` on the fn would go unread.
fn report<'tcx>(cx: &LateContext<'tcx>, f: &Finding<'tcx>, min_statements: usize) {
    let tcx = cx.tcx;
    let Searched {
        def,
        stretch,
        total,
        sharper,
    } = &f.searched;
    let def = *def;
    let msg = finding_message(tcx, def, f.sets, stretch.size, *total);
    let site = f.site.map(|(site, args)| {
        (
            site.source_callsite(),
            format!(
                "one of the {} instantiations, with {}",
                f.sets,
                render_args(tcx, def, args)
            ),
        )
    });
    let rest = stretch_clauses(stretch, min_statements);
    let inline = inline_note(tcx, def);
    let help = finding_help(tcx, def);
    let sharper = sharper
        .as_ref()
        .map(|line| format!("`{}` {line}", tcx.def_path_str(def)));
    emit_hir_then(
        cx,
        GENERIC_BODY_NOT_GENERIC,
        tcx.local_def_id_to_hir_id(def),
        tcx.def_span(def),
        msg,
        |diag| {
            // Where the stretch has a place in the source the note goes at
            // it, worded for which kind of place it is (`RegionSite::note`);
            // with no place at all it is a plain note under the fn.
            match stretch.site {
                Some(site) => site.note(diag, stretch.size, &rest),
                None => {
                    diag.note(format!("the stretch {rest}"));
                }
            }
            if let Some((site, note)) = site {
                diag.span_note(site, note);
            }
            if let Some(inline) = inline {
                diag.note(inline);
            }
            diag.help(help);
            if let Some(sharper) = sharper {
                diag.help(sharper);
            }
        },
    );
}

impl<'tcx> LateLintPass<'tcx> for GenericBodyNotGeneric {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _decl: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if matches!(kind, FnKind::Closure)
            || span.from_expansion()
            || body.value.span.from_expansion()
            || builds_coroutine(body)
            || !eligible(cx.tcx, def_id)
            || !splittable(cx.tcx, def_id)
        {
            return;
        }
        self.candidates.push(def_id);
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let tcx = cx.tcx;
        // Search first: the census walks every body in the crate, and a crate
        // with no fn holding a stretch worth reporting should not pay for it.
        // Each candidate's body is borrowed only while it is searched -- the
        // census reads the same bodies through `mir_for` afterwards -- and
        // what a finding would print is rendered into the `Stretch` before
        // the borrow goes, so nothing carried past this point holds a body.
        // Per fn the order is cheapest refusal first: the classifier's count
        // of independent items turns away a body that cannot hold
        // `min_statements` of them however they are arranged, before any
        // flow graph is built; `best_region` does the rest.
        let searched: Vec<Searched> = std::mem::take(&mut self.candidates)
            .into_iter()
            .filter_map(|def| {
                let body = mir_for(tcx, def)?;
                let (facts, measure) = classify(tcx, &body);
                if measure.independent < self.min_statements {
                    return None;
                }
                let region = best_region(tcx, &body, &facts, self.min_statements)?;
                // What the stretch takes lands inside it; what it hands back
                // lands in the blocks past it (cleanup ones are skipped).
                let mut past = DenseBitSet::new_filled(body.basic_blocks.len());
                past.subtract(&region.blocks);
                let stretch = Stretch {
                    site: region_site(tcx, def, &body, &region.blocks),
                    size: region.size,
                    uses: render_locals(cx, def, &body, &region.params, &region.blocks),
                    yields: render_locals(cx, def, &body, &region.returns, &past),
                    in_loop: region.in_loop,
                    other_regions: region.other_regions,
                };
                let sharper = conversion_help(&body, &only_conversions(tcx, &body));
                Some(Searched {
                    def,
                    stretch,
                    total: measure.total,
                    sharper,
                })
            })
            .collect();
        if searched.is_empty() {
            return;
        }
        let census = census(tcx);
        let mut findings: Vec<Finding<'tcx>> = searched
            .into_iter()
            .filter_map(|searched| {
                let sets = census.concrete.get(&searched.def).map_or(0, |s| s.len());
                (sets >= self.min_instantiations).then(|| Finding {
                    site: census.first_site.get(&searched.def).copied(),
                    searched,
                    sets,
                })
            })
            .collect();
        // One finding per fn, in source order, each emitted once at the fn's
        // own span (`report`).
        findings.sort_by_key(|f| tcx.def_span(f.searched.def).lo());
        for f in &findings {
            report(cx, f, self.min_statements);
        }
    }
}
