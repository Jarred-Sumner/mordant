//! Finds generic functions where a long part of the body does not use its
//! type or const parameters. What is reported, what is not, and the two
//! settings are written once, on the lint declared below.
//!
//! The number of concrete argument sets each function is used with is
//! counted by this file's own walk over the crate, not by asking rustc.
//! Asking rustc would build the optimized MIR of every reachable body, and
//! building that discards the earlier body that every MIR lint in this
//! pack reads (see `mir_flow::mir_for`). The rustc query is
//! `collect_and_partition_mono_items`; the body it would discard is
//! `mir_drops_elaborated_and_const_checked`. A check build should not run
//! that query anyway, since it is part of code generation. The walk reads
//! the type checker's record of each call, fn-item use and inserted
//! `Deref`, reads dropped types off MIR `Drop` terminators (HIR has no node
//! for a drop), and follows generic callers until nothing new is found. It
//! finds fewer uses than rustc would, so the count is a minimum; the lint
//! doc says what it misses.

use std::collections::VecDeque;
use std::ops::ControlFlow;

use clippy_utils::source::snippet_opt;
use clippy_utils::visitors::for_each_expr_without_closures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::work_queue::WorkQueue;
use rustc_hir::attrs::InlineAttr;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{ClosureKind, ConstContext, Expr, ExprKind, LangItem, Node};
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    self, BasicBlock, Local, Location, Operand, Place, RETURN_PLACE, Rvalue, Statement,
    StatementKind, Terminator, TerminatorEdges, TerminatorKind,
};
use rustc_middle::ty::adjustment::{Adjust, DerefAdjustKind};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, GenericArgKind, GenericArgsRef, GenericParamDefKind, Ty, TyCtxt, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor, TypeckResults,
};
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::MaybeLiveLocals;
use rustc_span::{BytePos, ExpnKind, Span, Spanned, Symbol};

use crate::MordantConfig;
use crate::baseline::{emit_hir_then, join};
use crate::mir_flow::{FlowGraph, local_name, mir_for, reaching, reads_any};

rustc_session::declare_lint! {
    /// Flags a generic function whose body contains a long stretch of code
    /// that does not use its type or const parameters. rustc compiles the
    /// whole body again for every distinct set of concrete arguments the
    /// function is used with, so that stretch is compiled several times where
    /// once would do: moved into a non-generic inner function called from the
    /// same place, it is compiled once, the outer signature is unchanged, and
    /// the cost per copy is one call. The finding underlines the stretch,
    /// lists by name and type the values it reads from the code before it
    /// and leaves for the code after it -- the inner function's arguments
    /// and results -- says when it is inside a loop, since the call then
    /// runs once per iteration, and points at one call site with concrete
    /// arguments. If all the function does with a generic argument is
    /// convert it once on entry through a trait method (`x.as_ref()`) and
    /// drop it on exit, a second help line says the function could take the
    /// converted value instead. That line is omitted when the flagged
    /// function is itself a trait method or implements one, because its
    /// signature is set by the trait.
    ///
    /// Two settings apply. `generic-body-not-generic-min-instantiations`
    /// (default 2) is how many distinct argument sets this crate must use
    /// the function with. `generic-body-not-generic-min-statements` (default
    /// 24) is the size the stretch must reach, counted in statements of MIR,
    /// the compiler's form of the body before optimization, where one source
    /// line is usually several. Only statements that do something at run
    /// time count, and only hand-written ones count toward the minimum: what
    /// `write!` expands to adds to the printed size but cannot make a
    /// stretch reportable by itself, since there is no source to move.
    ///
    /// A stretch is taken in whole basic blocks, the straight-line pieces
    /// MIR divides a body into, each ending where control branches, calls
    /// or returns. It meets three conditions. Control enters it at one point
    /// and leaves it for one point, or continues to the return. Nothing in it
    /// names a parameter (`T::default()`, `size_of::<T>()`, `i < N`) or
    /// uses a value whose type involves one: dereferencing a `&T` or
    /// reading `self.inner.len` through `inner: Wrap<T>` depends on the
    /// parameters, reading a `[u8; 4]` field of a `Framed<T>` or indexing a
    /// `[u8; N]` does not, and a borrowed constant is judged by its
    /// expression (`&[1, 2, 3]` no, `&T::ZERO` yes). Every value it takes in
    /// or leaves behind is a whole local of a parameter-free type, so `self`
    /// on a `Framed<T>` cannot be an input even when only its header is
    /// read: write `let header = self.header;` and the stretch starts on the
    /// next line. Dependent code belongs before the stretch (`bytes.as_ref()`)
    /// or after it (the drop of `bytes`, a `T::from(acc)` on its result).
    ///
    /// How many argument sets a function is used with is counted from this
    /// crate's own code: direct calls and fn-item uses, followed through
    /// generic callers (if `load<T>` is called only from `run<T>` and `run`
    /// is called with `A` and `B`, `load` has two), plus calls the compiler
    /// inserts -- a `Deref` impl reached through `w.field`, `w.method()` or
    /// a coercion of `&w`, and a `Drop` impl run where a value owning one,
    /// directly or in a field, element, box or closure, goes out of scope.
    /// Uses evaluated at compile time (`const` and `static` initializers,
    /// array lengths, `const {}` blocks) are not counted. Calls through
    /// `dyn`, function pointers and other crates are not seen, including
    /// what `Vec`'s `Drop` and `mem::drop` do, so a function used only
    /// downstream is not reported and the printed count is a minimum.
    ///
    /// Some duplicated code is deliberately not reported: an independent
    /// stretch below the minimum; one with an early `return` or `?` in its
    /// middle, which is two exits; independent statements in the same basic
    /// block as a dependent one at either end of a stretch, because a block
    /// is taken whole or not at all; a body whose independent and dependent
    /// statements alternate (a `W: Write` written to throughout, a const
    /// flag tested every few lines), because sharing
    /// them would take a `dyn` call, a branch or a call per alternation
    /// rather than one call -- though a flag tested once only splits the
    /// body in two, and the larger side is reported if it qualifies; the
    /// inside of a branch a const parameter selects (`if FLAG { .. }`),
    /// because each copy keeps only its own branch; a stretch whose inputs
    /// would include a value that is a constant in each copy (`let width =
    /// width_of(TAG)`), because what is computed from it is evaluated at
    /// compile time now and would not be behind a `width: usize` argument;
    /// a stretch whose results would borrow from a local it creates (the
    /// reported stretch ends before that borrow); one that moves out, on one
    /// path only, a value whose drop does more than free memory, such as a
    /// lock guard. Functions marked `#[inline(always)]`, `#[track_caller]`,
    /// `#[target_feature]` or `#[naked]`, and `async` and `gen` functions,
    /// are never reported, because for each an added inner call changes
    /// cost or behaviour; a plain `#[inline]` function is reported with a
    /// note that the inner function must not repeat the attribute.
    ///
    /// Warns by default because every claim in a finding is checked against
    /// the code: the stretch is identical in every copy, moves out
    /// unchanged, and the edited program compiles. It cannot measure the
    /// bytes saved, which depend on inlining, optimization level, LTO and
    /// identical-code folding.
    pub GENERIC_BODY_NOT_GENERIC,
    Warn,
    "a generic fn with a stretch of body that is the same in every instantiation, compiled once per instantiation"
}

pub struct GenericBodyNotGeneric {
    min_statements: usize,
    min_instantiations: usize,
}

rustc_session::impl_lint_pass!(GenericBodyNotGeneric => [GENERIC_BODY_NOT_GENERIC]);

impl GenericBodyNotGeneric {
    pub fn new(config: &MordantConfig) -> Self {
        Self {
            min_statements: config.generic_body_not_generic_min_statements,
            min_instantiations: config.generic_body_not_generic_min_instantiations,
        }
    }
}

/// A fn item or method with a type or const parameter in scope (its own, or
/// the `impl`'s or trait's). Not a closure: its parameters are its parent's.
/// Lifetimes alone do not count: they are erased before codegen. Macro- and
/// derive-generated fns are included so `propagate` can pass their concrete
/// arguments on to the hand-written generic fns they call; they are never
/// reported themselves.
fn generic_fn(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
    tcx.def_kind(def).is_fn_like()
        && !tcx.is_closure_like(def.to_def_id())
        && tcx.generics_of(def).requires_monomorphization(tcx)
}

/// Whether moving part of the body into a non-generic inner fn is a valid
/// fix. These attributes rule it out:
/// - `#[inline(always)]` / `#[rustc_force_inline]`: the author asked for no
///   out-of-line call.
/// - `#[track_caller]`: panic locations in the moved code would change.
/// - `#[target_feature]`: the inner fn would be compiled without the features.
/// - `#[naked]`: the body is one asm block; no call can be added.
///
/// Plain `#[inline]` is a hint, not a demand: the fn is still reported, with
/// a note (`inline_note`). ABI, `const` and `unsafe` do not matter.
fn splittable(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
    let attrs = tcx.codegen_fn_attrs(def);
    !attrs.inline.always()
        && !attrs
            .flags
            .intersects(CodegenFnAttrFlags::TRACK_CALLER | CodegenFnAttrFlags::NAKED)
        && attrs.target_features.is_empty()
}

/// Whether the lint may report `def`: a generic fn or method (`generic_fn`)
/// that the fix is valid for (`splittable`), written by hand rather than by
/// a macro or derive, and not `async` or `gen`. An `async fn`'s own body
/// only builds the coroutine and returns it; the written code is in the
/// coroutine body, which `generic_fn` excludes as closure-like and which is
/// lowered to a state machine an inner fn cannot take part of.
fn candidate_fn(tcx: TyCtxt<'_>, def: LocalDefId) -> bool {
    if !generic_fn(tcx, def) || !splittable(tcx, def) || tcx.def_span(def).from_expansion() {
        return false;
    }
    let body = tcx.hir_body_owned_by(def).value;
    !body.span.from_expansion()
        && !matches!(body.kind, ExprKind::Closure(c) if matches!(c.kind, ClosureKind::Coroutine(_)))
}

/// The extra note for an `#[inline]` fn, `None` otherwise: if the inner fn
/// keeps the attribute, every codegen unit that calls it compiles its own
/// copy and nothing is shared.
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
// This part counts, for each basic block of a function, the statements and
// terminators that do something at run time, and how many of those depend on
// a type or const parameter. Storage markers, fake reads, plain jumps and the
// like do nothing at run time and are not counted. The counts (`BlockFacts`)
// are taken from the MIR that `mir_for` returns, which no optimization pass
// or inlining has changed, so they describe only the code written in this
// function. Blocks that only run while a panic unwinds repeat the normal
// path's drops, so they are recorded with zero counts and flagged, and the
// stretch search never uses them.
//
// A counted item is *dependent* when a type or const parameter appears in it
// (a cast's target type, a callee's generic arguments, a constant, the field
// type recorded on a projection) or in the type of a place it reads or
// writes. The type tested for a place is the place's own, not its base
// local's: `(*_1).header` with `_1: &Framed<T>` has type `[u8; 4]` and is
// independent, while `(*_1).inner.len` with `inner: Wrap<T>` has a `Field`
// projection whose recorded type is `Wrap<T>`, so it is dependent. Every
// other counted item is *independent*: the same code at every instantiation.
// Whether a sequence of independent blocks can actually be moved out of the
// function is for the stretch search to decide, and it tests whole locals, so
// `(*_1).header` still cannot start a stretch unless it is first copied into
// a local of its own.
//
// A borrow of a constant expression like `&[1, 2, 3]` or `&T::ZERO` is
// compiled to a separate small MIR body, called a *promoted*, and the
// statement that remains in the function (`_n = const promoted[k]`) refers to
// the function with its full generic argument list, so `has_non_region_param`
// is always true for it. For that one statement shape
// `promoted_load_names_param` checks the constant's type and the promoted
// body's locals, statements and terminators instead. A promoted load of any
// other shape (for example another body's promoted, copied in by inlining) is
// checked as a whole like every other statement, and so stays dependent. If
// the promoted body itself loads another of the function's constants, that
// inner load is checked as a whole too, so the borrow counts as dependent;
// this can only make a stretch smaller, never report one that is not there.

/// One basic block's counts. All zero for an unwind cleanup block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockFacts {
    pub(crate) counted: u32,
    pub(crate) dependent: u32,
    /// Counted items whose span the author wrote by hand (`hand_written`).
    /// A stretch's minimum size is measured in these, so that fifty statements
    /// from one `log!` line do not qualify on their own; the size a finding
    /// prints is `counted`.
    pub(crate) hand_written: u32,
}

/// Whether the item at `span` was written by hand: its syntax context is the
/// root, or a desugaring (a `for` loop's `next()`, a `?`, an `.await`) of
/// code that was. Anything a macro expanded to is not, but tokens the author
/// passed in keep their own context: in `log!("{}", compute(x))` the call
/// `compute(x)` is hand-written and the formatting around it is not. This
/// stops at the first macro and answers false; `body_position` uses the same
/// chain of expansions but continues through each macro to its call site.
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

/// Statements that do something at run time; storage markers, fake reads,
/// type ascriptions and coverage counters emit nothing.
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
/// asserts, inline asm. Plain jumps, the return and unwind edges are not.
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

/// Finds whether any place a statement or terminator reads or writes has a
/// type a parameter appears in. Calling `has_non_region_param` on the item
/// checks the types written in it, but not the declared type of a local,
/// which is stored in the body; that is why this needs the body. Only the
/// final projected type is tested, not the base local's (no `super_place`),
/// so `(*_1).header` with `_1: &Framed<T>` tests `[u8; 4]`, not `Framed<T>`.
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
    /// Two answers, kept apart for `const_only`: whether the statement itself
    /// names a parameter (a promoted load judged by `promoted_load_names_param`),
    /// and whether a place it touches has a type one appears in.
    fn statement_params(&mut self, statement: &Statement<'tcx>) -> (bool, bool) {
        self.found = false;
        self.visit_statement(statement, Location::START);
        let names_param = promoted_load_names_param(self.tcx, self.body, statement)
            .unwrap_or_else(|| statement.has_non_region_param());
        (names_param, self.found)
    }

    /// The same for a terminator.
    fn terminator_params(&mut self, terminator: &Terminator<'tcx>) -> (bool, bool) {
        self.found = false;
        self.visit_terminator(terminator, Location::START);
        (terminator.has_non_region_param(), self.found)
    }

    /// Whether a counted statement is dependent; the classifier and the help share it.
    fn dependent_statement(&mut self, statement: &Statement<'tcx>) -> bool {
        let (names_param, typed_place) = self.statement_params(statement);
        names_param || typed_place
    }

    /// The same for a counted terminator.
    fn dependent_terminator(&mut self, terminator: &Terminator<'tcx>) -> bool {
        let (names_param, typed_place) = self.terminator_params(terminator);
        names_param || typed_place
    }

    /// Whether a definition names a parameter only in a constant (`N`,
    /// `L::WIDTH`), a callee's generic arguments (`size_of::<T>()`), or a
    /// cast's or aggregate's type, while every place it touches has a
    /// parameter-free type: a value that is a constant in each copy, which
    /// is where `per_copy_consts` starts.
    fn const_only(&mut self, rhs: AssignedValue<'_, 'tcx>) -> bool {
        let (names_param, typed_place) = match rhs {
            AssignedValue::Value(statement, _) => self.statement_params(statement),
            AssignedValue::Call(terminator, ..) => self.terminator_params(terminator),
        };
        names_param && !typed_place
    }
}

/// If `statement` loads one of `body`'s own promoted constants
/// (`_n = const promoted[k]`), whether that constant names a type or const
/// parameter, judged by its type and the promoted body (see the note above).
/// `None` for any other statement.
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
    // `promoted_mir` runs borrowck if it has not run, then lowers the promoted
    // bodies to runtime MIR once, cached. It does not consume the body
    // `mir_for` reads.
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

/// Counts every basic block of `body` (indexed like `body.basic_blocks`) in
/// one pass, and returns the body's total of counted items with them.
pub(crate) fn classify<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
) -> (IndexVec<BasicBlock, BlockFacts>, usize) {
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let facts: IndexVec<BasicBlock, BlockFacts> = body
        .basic_blocks
        .iter()
        .map(|data| {
            if data.is_cleanup {
                return BlockFacts::default();
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
            for statement in &data.statements {
                if !counted_statement(statement) {
                    continue;
                }
                count(
                    places.dependent_statement(statement),
                    statement.source_info.span,
                );
            }
            if let Some(terminator) = &data.terminator
                && counted_terminator(terminator)
            {
                count(
                    places.dependent_terminator(terminator),
                    terminator.source_info.span,
                );
            }
            facts
        })
        .collect();
    let total = facts.iter().map(|block| block.counted as usize).sum();
    (facts, total)
}

// ── stretch inputs and outputs ───────────────────────────────────────────────
//
// A stretch is a set of whole basic blocks that control enters at one block,
// its entry. Every normal edge out of the stretch goes to one block, its
// exit, or the stretch contains the fn's `return` and has no exit. Moving
// the stretch into an inner fn replaces its blocks with one call,
// `let (r1, r2) = inner(p1, p2);`, whose return edge goes to the exit.
// `stretch_io` decides that call's params and results as follows and
// returns `None` when one of these checks refuses the stretch.
//
// 1. Params. A local is a param when the stretch may read it before writing
//    it, or when it is a result that the stretch does not write on every
//    path to the exit (on that path the value at the exit is the value the
//    local had at the entry). A local is live at a point when a later
//    statement may read the value it has there. One backward pass computing
//    that (liveness) over the stretch's blocks alone, with the results as the
//    set live on leaving the stretch, finds both kinds (`live_in`).
// 2. Results. A local is a result when the stretch may change it (assign,
//    call destination, mutable borrow, drop) and it is live in the whole
//    body at the exit's first statement (`LocalFacts::live_at`). When the
//    stretch contains the `return` the only result is `_0`. Both passes use
//    `live_in`, which follows normal edges only. rustc's `MaybeLiveLocals`
//    also follows unwind edges into cleanup blocks, which read locals that
//    are not the inner fn's: drop flags (hidden `bool`s that
//    `ElaborateDrops` adds to record whether a local still needs its drop)
//    and the flag-guarded drops of locals the stretch already moved out of.
//    The outer fn's cleanup blocks run whether unwinding starts inside the
//    call or after it, so a read that happens only there adds no param and
//    no result.
// 3. Types. The inner fn declares every other local the stretch names, so
//    none of those may have a type that mentions a type or const parameter
//    (the lint doc's condition that nothing in the stretch uses a value
//    whose type involves a parameter; `fit_blocks` checks it once per
//    block, so `stretch_io` does not check it again), and a param or result
//    needs a type that can be written in a fn signature (`nameable`). A
//    param may not be a local whose value is a compile-time constant in
//    each instantiation either, such as `size_of::<T>()` or `width_of(TAG)`
//    (`LocalFacts::per_copy_consts`): rustc evaluates what is computed from it
//    at compile time inside each copy, and could not if the value were
//    passed in as a run-time argument, so the moved code would be slower
//    than any copy.
// 4. Params passed by value. A param the stretch moves out of or drops is
//    passed by value, and the inner fn drops it. When such a param is still
//    live at the exit and the stretch does not reassign it (the stretch moved
//    it on some paths only, and the outer fn drops it after the exit on the
//    others), refuse if the body after the exit reads it (`read_past`) or if
//    its drop does more than free memory (`has_significant_drop`). A drop
//    flag is never a param or a result. One that the stretch sets may be
//    live after the exit only when every drop it guards is of such a param
//    (`LocalFacts::flag_guards`). One that the stretch tests and that code
//    before the entry set would let the stretch drop a value that was never
//    initialized, so a drop flag live at the entry refuses too.
// 5. Borrows held across the call. A param that the stretch writes, moves
//    out of or drops (or writes through, when it is a `Box` or `&mut`) is
//    passed by value or as `&mut`, so the call moves or mutably borrows it
//    at the entry, where the original fn used it only inside the stretch.
//    Refuse when a local live at the entry may hold a borrow of such a
//    param (E0505, E0502), or may hold an exclusive borrow of any param
//    (E0499, E0502). Such params are `StretchLocals::exclusive`, and the
//    locals that may hold a borrow are found by `borrow_holders`, run over
//    the blocks outside the stretch that reach the entry.
// 6. Addresses the inner fn owns. The inner fn owns the storage of every
//    local the stretch names except the params it takes by reference, so a
//    borrow the stretch takes of one must not be usable after the call. A
//    local whose `StorageLive` and `StorageDead` are all in the stretch
//    needs no check (`storage_within`). For the rest, refuse when a local
//    that may hold the address is a result, or the address is stored
//    through a pointer, or passed to a call together with an argument whose
//    type could store it (`may_take_address`), or built into one aggregate
//    with such a value. The locals that may hold the address are found by
//    `borrow_holders`, run over the stretch. rustc's `MaybeBorrowedLocals`
//    cannot be used here: it clears a borrow only at `StorageDead`, so a
//    `v.push(..)` in the stretch on a `v` read after the exit, the case this
//    lint exists for, would refuse every stretch containing it.
// 7. A result that borrows a param passed by reference. In the inner fn's
//    signature such a result borrows the whole param, where the original fn
//    borrowed one place inside it. Check each statement reachable from the
//    exit while the result, or a local it was copied into, is live, and
//    refuse at the first use of the param that the borrow forbids: any use
//    when the borrow is exclusive (the param is passed as `&mut`, or the
//    result may hold a `&mut`: E0503, E0499), and a write, move, drop or
//    `&mut` when it is shared (E0506, E0505, E0502). When the borrow is
//    exclusive, reaching the entry again with the result live refuses too,
//    since the call there borrows the param again (`result_borrows_param`).

/// Facts about the whole body's locals, computed once per fn and read once
/// per candidate stretch.
pub(crate) struct LocalFacts {
    /// Locals live on entry to each non-cleanup block, ignoring unwind edges.
    live: FxHashMap<BasicBlock, DenseBitSet<Local>>,
    /// The set `{_0}`: what a `return` reads.
    returned: DenseBitSet<Local>,
    storage: IndexVec<Local, Storage>,
    /// The non-cleanup blocks.
    normal: DenseBitSet<BasicBlock>,
    /// From the fn `per_copy_consts`. A stretch may not take one as a param.
    per_copy_consts: DenseBitSet<Local>,
    /// Locals that look like drop flags: a `bool` past the arguments that no
    /// user named and that is only ever assigned a constant. A temporary for
    /// an `if` on `matches!` or `&&` has the same shape and is included; that
    /// only costs a stretch the blocks setting it.
    drop_flags: DenseBitSet<Local>,
    /// Each `switchInt` on a drop flag, with the local dropped on its
    /// `otherwise` edge, or `None` when that edge is not a direct `Drop`.
    flag_guards: Vec<(Local, Option<Local>)>,
}

/// The non-cleanup blocks holding a local's storage markers, and whether any
/// of them is a `StorageDead`. Arguments and the return place have none.
#[derive(Default)]
struct Storage {
    blocks: Vec<BasicBlock>,
    dies: bool,
}

impl LocalFacts {
    pub(crate) fn new(body: &mir::Body<'_>, per_copy_consts: &DenseBitSet<Local>) -> Self {
        let mut storage = IndexVec::from_fn_n(|_| Storage::default(), body.local_decls.len());
        let mut normal = DenseBitSet::new_empty(body.basic_blocks.len());
        // `local_info` is cleared by this phase; user-named means has debuginfo.
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
        Self {
            live: live_in(body, &normal, &returned),
            returned,
            storage,
            normal,
            per_copy_consts: per_copy_consts.clone(),
            drop_flags,
            flag_guards,
        }
    }

    /// Locals live at the first statement of `block`, along normal edges.
    fn live_at(&self, block: BasicBlock) -> DenseBitSet<Local> {
        match self.live.get(&block) {
            Some(live) => live.clone(),
            None => DenseBitSet::new_empty(self.storage.len()),
        }
    }

    /// Whether the local has a `StorageDead` on a normal path and all its
    /// storage markers are inside `blocks`. Then no borrow of it is usable
    /// outside `blocks`, since a borrow never outlives `StorageDead`.
    fn storage_within(&self, local: Local, blocks: &DenseBitSet<BasicBlock>) -> bool {
        let storage = &self.storage[local];
        storage.dies && storage.blocks.iter().all(|&b| blocks.contains(b))
    }
}

/// Returns the locals whose value is a constant in each instantiation, and
/// the blocks that run or not depending on such a constant
/// (`blocks_under_const_branch`). `stretch_io` refuses a stretch that takes
/// one of these locals as a param: rustc evaluates what is computed from it
/// at compile time inside each copy, and could not if the value were passed
/// in as a run-time argument.
///
/// The set starts as every local assigned from something that names a type
/// or const parameter but uses no place whose type does, the destination
/// included: `size_of::<T>()`, `L::WIDTH * 2`, `width_of(TAG)`, but not
/// `[0u8; N]` stored into a `[u8; N]` (the test is `PlaceParams::const_only`).
/// Two steps then add to it in turn until neither adds a local, since each
/// can add to the other's input. `add_locals_computed_from` adds every
/// local computed from one already in the set.
/// `consts_assigned_under_const_branch` adds every local assigned a value
/// that reads no local inside a block that runs only under a branch on such
/// a constant (`blocks_under_const_branch`): `match TAG { 0 => 1, _ => 4 }`
/// lowers to one literal assignment per arm.
/// In those two steps only the assigned value is searched, never the
/// destination place (see `assigned_local`). The length of a `&[u8]` unsized
/// from a `[u8; N]` is not counted as a constant, so that a stretch taking it
/// still reports.
fn per_copy_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    flow: &FlowGraph,
) -> (DenseBitSet<Local>, DenseBitSet<BasicBlock>) {
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let mut consts = DenseBitSet::new_empty(body.local_decls.len());
    for (dest, rhs) in assignments(body, body.basic_blocks.indices()) {
        if let Some(local) = assigned_local(dest)
            && places.const_only(rhs)
        {
            consts.insert(local);
        }
    }
    loop {
        if !consts.is_empty() {
            add_locals_computed_from(
                body,
                || body.basic_blocks.indices(),
                &mut consts,
                |dest| assigned_local(dest).is_some(),
            );
        }
        let branch_blocks = blocks_under_const_branch(body, flow, &consts);
        if !consts_assigned_under_const_branch(body, &branch_blocks, &mut consts) {
            return (consts, branch_blocks);
        }
    }
}

/// Adds to `per_copy_consts` every local that a block in
/// `blocks_under_const_branch` assigns a value reading no local: a literal,
/// an aggregate of literals, a call on constant arguments. Returns whether it
/// added any. A value computed from a run-time
/// local stays run-time (`acc = acc.swap_bytes()` under `if WIDE`), so the
/// code after such a branch still reports. A `RuntimeChecks` operand is a
/// constant of the compiler session, not of the copy, so it is not added.
fn consts_assigned_under_const_branch(
    body: &mir::Body<'_>,
    blocks_under_const_branch: &DenseBitSet<BasicBlock>,
    per_copy_consts: &mut DenseBitSet<Local>,
) -> bool {
    let every = DenseBitSet::new_filled(body.local_decls.len());
    let mut grew = false;
    for (dest, rhs) in assignments(body, blocks_under_const_branch.iter()) {
        if let Some(local) = assigned_local(dest)
            && !per_copy_consts.contains(local)
            && !matches!(
                rhs,
                AssignedValue::Value(_, Rvalue::Use(Operand::RuntimeChecks(_), _))
            )
            && !rhs.reads_any(&every)
        {
            grew |= per_copy_consts.insert(local);
        }
    }
    grew
}

/// Adds to `set` every local in `blocks` assigned a value that reads a
/// local already in `set`, and repeats until a pass adds none. `accept` can
/// refuse a destination place. The value is an assignment's right-hand side
/// or a call's callee and arguments, never the destination place, so
/// `buf[width] = b` adds nothing.
fn add_locals_computed_from<'tcx, I: Iterator<Item = BasicBlock>>(
    body: &mir::Body<'tcx>,
    blocks: impl Fn() -> I,
    set: &mut DenseBitSet<Local>,
    accept: impl Fn(Place<'tcx>) -> bool,
) {
    // A pass only adds locals, so one that adds none is the last. Block
    // order roughly follows control flow, so one or two are usually enough.
    loop {
        let mut grew = false;
        for (dest, rhs) in assignments(body, blocks()) {
            if !set.contains(dest.local) && accept(dest) && rhs.reads_any(set) {
                grew |= set.insert(dest.local);
            }
        }
        if !grew {
            return;
        }
    }
}

/// The local a store to `place` sets: `place.local` when the store writes
/// the whole local or a field of it, `None` when it writes through a pointer
/// or into an array element (the array stays run-time data).
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

/// What an assigned value is computed from: an `Assign`'s right-hand side,
/// or a `Call`'s callee and arguments. Never the destination place.
#[derive(Clone, Copy)]
enum AssignedValue<'a, 'tcx> {
    Value(&'a Statement<'tcx>, &'a Rvalue<'tcx>),
    Call(
        &'a Terminator<'tcx>,
        &'a Operand<'tcx>,
        &'a [Spanned<Operand<'tcx>>],
    ),
}

impl<'tcx> AssignedValue<'_, 'tcx> {
    fn visit(self, visitor: &mut impl Visitor<'tcx>) {
        match self {
            AssignedValue::Value(_, rvalue) => visitor.visit_rvalue(rvalue, Location::START),
            AssignedValue::Call(_, func, args) => {
                visitor.visit_operand(func, Location::START);
                for arg in args {
                    visitor.visit_operand(&arg.node, Location::START);
                }
            }
        }
    }

    /// Whether the value uses a local in `of`.
    fn reads_any(self, of: &DenseBitSet<Local>) -> bool {
        reads_any(of, |uses| self.visit(uses))
    }
}

/// Every assignment and call result in the non-cleanup blocks among
/// `blocks`: the destination place and the `AssignedValue`.
fn assignments<'a, 'tcx>(
    body: &'a mir::Body<'tcx>,
    blocks: impl Iterator<Item = BasicBlock>,
) -> impl Iterator<Item = (Place<'tcx>, AssignedValue<'a, 'tcx>)> {
    blocks
        .map(|block| &body.basic_blocks[block])
        .filter(|data| !data.is_cleanup)
        .flat_map(|data| {
            let assigns = data.statements.iter().filter_map(|statement| {
                let StatementKind::Assign(assign) = &statement.kind else {
                    return None;
                };
                Some((assign.0, AssignedValue::Value(statement, &assign.1)))
            });
            let call = data.terminator.as_ref().and_then(|terminator| {
                let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } = &terminator.kind
                else {
                    return None;
                };
                Some((*destination, AssignedValue::Call(terminator, func, args)))
            });
            assigns.chain(call)
        })
}

/// The params and results a non-generic inner fn holding a stretch would
/// have, each in declaration order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StretchIo {
    /// Locals the stretch reads before writing, or returns without writing
    /// on every path.
    pub(crate) params: Vec<Local>,
    /// Locals the stretch may change that are still read after it along a
    /// normal edge; `_0` when the stretch holds the fn's `return`.
    pub(crate) returns: Vec<Local>,
}

/// The params and results of an inner fn holding the stretch `blocks`, or
/// `None` when the stretch cannot become one. `blocks` are whole non-cleanup
/// blocks that all passed `fit_blocks` (so no local they use has a type or
/// const parameter in its type), `entry` is the only block entered from
/// outside, and every normal edge out goes to `exit`; `exit == None` means
/// the stretch holds the `return`. Each reason to refuse is commented where
/// it is checked.
pub(crate) fn stretch_io<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    locals: &LocalFacts,
    blocks: &DenseBitSet<BasicBlock>,
    entry: BasicBlock,
    exit: Option<BasicBlock>,
) -> Option<StretchIo> {
    debug_assert!(blocks.contains(entry) && exit.is_none_or(|x| !blocks.contains(x)));
    let names = StretchLocals::of(tcx, body, blocks);
    let live_out = match exit {
        Some(exit) => locals.live_at(exit),
        None => {
            let mut returned = DenseBitSet::new_empty(body.local_decls.len());
            returned.insert(RETURN_PLACE);
            returned
        }
    };
    // A local the stretch moves out of and never reassigns, still live at
    // the exit: allowed only when all the body does with it after the exit
    // is a drop that just frees memory.
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
    // A drop flag that the stretch sets and the body tests after the exit
    // may guard only a local the stretch moved out of and did not reassign
    // (the loop above checked those).
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
    // Params: what is live at the entry when liveness is computed over the
    // stretch's blocks alone, with exactly `returns` live on leaving them.
    let params = live_in(body, blocks, &returns)
        .remove(&entry)
        .unwrap_or_else(|| DenseBitSet::new_empty(body.local_decls.len()));
    // Params and results need a type that can be written in a signature.
    if params
        .iter()
        .chain(returns.iter())
        .any(|local| !nameable(body.local_decls[local].ty))
    {
        return None;
    }
    // A drop flag live at the entry was last set before the stretch ran:
    // the stretch would drop, on some path, a local that may not have been
    // initialized.
    if params.iter().any(|local| locals.drop_flags.contains(local)) {
        return None;
    }
    // A param that is a constant in each copy is constant-folded there, and
    // could not be if it were passed in as an argument.
    if params
        .iter()
        .any(|local| locals.per_copy_consts.contains(local))
    {
        return None;
    }
    // A param passed by value or as `&mut` while a local live at the entry
    // may hold a borrow of it (or of data reached through it), or any param
    // while one may hold an exclusive borrow of it, is E0505/E0499/E0502 at
    // the call (`let name = &rec.name;`, then the stretch writes
    // `rec.count`, then `out.push(name)` after it). Addresses are followed
    // only through blocks outside the stretch from which the entry is
    // reachable; skipped when no local live at the entry has a type that
    // can hold a borrow.
    let mut live = locals.live_at(entry);
    live.union(&params);
    if live.iter().any(|local| can_hold_borrow(body, local)) {
        let mut outside = reaching(body, entry);
        outside.intersect(&locals.normal);
        outside.subtract(blocks);
        let live_holds = |set: &DenseBitSet<Local>| live.iter().any(|local| set.contains(local));
        let mut exclusive_params = params.clone();
        exclusive_params.intersect(&names.exclusive);
        if !exclusive_params.is_empty() {
            let holders = borrow_holders(tcx, body, &outside, &exclusive_params);
            if holders.escaped || live_holds(&holders.any) {
                return None;
            }
        }
        let mut shared = params.clone();
        shared.subtract(&names.exclusive);
        if !shared.is_empty() {
            let holders = borrow_holders(tcx, body, &outside, &shared);
            if holders.escaped_exclusive || live_holds(&holders.exclusive) {
                return None;
            }
        }
    }
    // The address of a local the inner fn owns or returns may not outlive
    // the call: every addressed local except params the stretch neither
    // moves nor returns. Storage contained in the stretch already
    // guarantees it.
    let mut owned_addressed = names.addressed.clone();
    for local in names.addressed.iter() {
        let by_ref =
            params.contains(local) && !names.moved.contains(local) && !returns.contains(local);
        if by_ref || locals.storage_within(local, blocks) {
            owned_addressed.remove(local);
        }
    }
    if !owned_addressed.is_empty() {
        let holders = borrow_holders(tcx, body, blocks, &owned_addressed);
        if holders.escaped || returns.iter().any(|local| holders.any.contains(local)) {
            return None;
        }
    }
    // A result that borrows from a param the stretch does not move borrows
    // the whole param while the result is live; refuse if the body uses the
    // param again in that span (`let chunk = &cur.data[a..b];` in the
    // stretch, then `cur.reads += 1; out.push(chunk)` after it, is E0503
    // once `chunk` is returned by `inner(cur)`).
    if let Some(exit) = exit {
        let past = PastExit {
            tcx,
            body,
            locals,
            stretch: blocks,
            entry,
            exit,
        };
        if past.result_borrows_param(&names.exclusive, &params, &returns) {
            return None;
        }
    }
    Some(StretchIo {
        params: params.iter().collect(),
        returns: returns.iter().collect(),
    })
}

/// What a stretch does to each local it names, from one walk of its blocks.
struct StretchLocals {
    /// Possibly changed by the stretch: assigned, a call or asm destination,
    /// mutably borrowed, or dropped. A write through a deref does not count.
    written: DenseBitSet<Local>,
    /// Moved out of or dropped in the stretch, whole or in part: passed by
    /// value.
    moved: DenseBitSet<Local>,
    /// Its own storage borrowed in the stretch (`&x`, `&mut x.field`,
    /// `&raw const x`), not something it points at (`&(*x).field`).
    addressed: DenseBitSet<Local>,
    /// Locals the call cannot take as `&`: `written`, `moved`, and each
    /// `Box` or `&mut` local that the stretch writes or mutably borrows
    /// through (`(*m).len = ..`), also via an `inserted_pointer_copy` of it.
    /// A `&` or raw pointer that the stretch writes through is only copied,
    /// so it is not here.
    exclusive: DenseBitSet<Local>,
    /// Locals of any type that the stretch writes or mutably borrows through.
    /// The owning ones (`owning`) are added to `exclusive` after the walk.
    through: DenseBitSet<Local>,
    /// Locals whose type owns what it points at (`owns_pointee`).
    owning: DenseBitSet<Local>,
}

impl StretchLocals {
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
        let mut names = StretchLocals {
            written: empty.clone(),
            moved: empty.clone(),
            addressed: empty.clone(),
            exclusive: empty.clone(),
            through: empty,
            owning,
        };
        let mut copies: Vec<(Local, Local)> = Vec::new();
        for block in blocks.iter() {
            let data = &body.basic_blocks[block];
            for statement in &data.statements {
                names.visit_statement(statement, Location::START);
                copies.extend(inserted_pointer_copy(tcx, body, statement));
            }
            if let Some(terminator) = &data.terminator
                && !matches!(terminator.kind, TerminatorKind::Return)
            {
                names.visit_terminator(terminator, Location::START);
            }
        }
        // A copy of a copy (`(*(*p).a).b`) needs a second pass; passes only add.
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

/// Whether the statement copies a pointer from one local into another such
/// that the borrow checker counts each use of the copy as a use of the
/// original (`copies_borrow`). Returns `(copy, original)`. Only MIR passes
/// insert such copies: `Derefer` (`_t = copy (*p).f`), `ElaborateBoxDerefs`
/// (`_t = copy b.0.0`).
fn inserted_pointer_copy<'tcx>(
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
    copies_borrow(tcx, body, *place).then_some((dest.local, place.local))
}

/// `root` plus each local in `blocks` that an `inserted_pointer_copy`
/// assigns a pointer out of `root` to, directly or through another one.
fn pointer_copies_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
    root: Local,
) -> DenseBitSet<Local> {
    let copies: Vec<(Local, Local)> = blocks
        .iter()
        .flat_map(|block| &body.basic_blocks[block].statements)
        .filter_map(|statement| inserted_pointer_copy(tcx, body, statement))
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

impl<'tcx> Visitor<'tcx> for StretchLocals {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _: Location) {
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
    }
}

/// Which locals are live at the start of each block in `blocks`. A local is
/// live where something later may read the value it has now. Only paths
/// through `blocks` count, and only along edges a run without a panic can
/// take: not unwind edges, and not the edge to the empty `unreachable` block
/// that every enum `match` has for its impossible case. Where such an edge
/// leaves `blocks`, or a block returns, exactly `boundary` is live. Reads and
/// writes are found as rustc's `MaybeLiveLocals` finds them, except that
/// `return` does not read `_0`: `boundary` says whether `_0` is live.
fn live_in(
    body: &mir::Body<'_>,
    blocks: &DenseBitSet<BasicBlock>,
    boundary: &DenseBitSet<Local>,
) -> FxHashMap<BasicBlock, DenseBitSet<Local>> {
    let empty = DenseBitSet::new_empty(body.local_decls.len());
    let preds = body.basic_blocks.predecessors();
    let mut start: FxHashMap<BasicBlock, DenseBitSet<Local>> =
        blocks.iter().map(|block| (block, empty.clone())).collect();
    // Queue later blocks first: backward liveness converges fastest that way.
    let mut queue: WorkQueue<BasicBlock> = WorkQueue::with_none(body.basic_blocks.len());
    let mut order: Vec<BasicBlock> = blocks.iter().collect();
    order.reverse();
    for block in order {
        queue.insert(block);
    }
    let mut state = empty;
    while let Some(block) = queue.pop() {
        block_live(body, block, &start, boundary, &mut state, |_, _| {});
        // States only gain locals: requeue predecessors only on a change.
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

/// Sets `state` to the locals live at the start of `block`, working backward
/// from its end by the rules on `live_in`. Live at the end: the union of
/// `live_at` over the successors, plus `boundary` if the block returns or a
/// successor is not in `live_at`. Cleanup successors and the empty
/// `unreachable` block add nothing (see `live_in`; the stretch search leaves
/// them out too). `each(i, state)` is called with the locals live just
/// before statement `i`; `i == statements.len()` means just before the
/// terminator.
fn block_live(
    body: &mir::Body<'_>,
    block: BasicBlock,
    live_at: &FxHashMap<BasicBlock, DenseBitSet<Local>>,
    boundary: &DenseBitSet<Local>,
    state: &mut DenseBitSet<Local>,
    mut each: impl FnMut(usize, &DenseBitSet<Local>),
) {
    let data = &body.basic_blocks[block];
    state.clear();
    if let Some(terminator) = &data.terminator {
        let mut leaves = matches!(terminator.kind, TerminatorKind::Return);
        for next in terminator.successors() {
            let data = &body.basic_blocks[next];
            if data.is_cleanup || (data.terminator.is_some() && data.is_empty_unreachable()) {
                continue;
            }
            match live_at.get(&next) {
                Some(live) => {
                    state.union(live);
                }
                None => leaves = true,
            }
        }
        if leaves {
            state.union(boundary);
        }
        if !matches!(terminator.kind, TerminatorKind::Return) {
            // A call or inline asm writes its destination only if it comes
            // back instead of panicking, and only that case is followed here:
            // apply the write, then the terminator's reads.
            if let TerminatorEdges::AssignOnReturn { return_, place, .. } = terminator.edges()
                && !return_.is_empty()
            {
                MaybeLiveLocals.apply_call_return_effect(state, block, place);
            }
            let at = Location {
                block,
                statement_index: data.statements.len(),
            };
            MaybeLiveLocals::transfer_function(state).visit_terminator(terminator, at);
        }
    }
    each(data.statements.len(), state);
    for (index, statement) in data.statements.iter().enumerate().rev() {
        let at = Location {
            block,
            statement_index: index,
        };
        MaybeLiveLocals::transfer_function(state).visit_statement(statement, at);
        each(index, state);
    }
}

/// Whether `from`, or any block reachable from it without panicking, reads
/// `local` other than to drop it. Its `Drop` terminator does not count, nor
/// does a `Discriminant` read of it (the first step of dropping an enum;
/// callers ask only about a partly moved local, and nothing else reads its
/// discriminant). Statement order is ignored: a read after a write counts.
fn read_past(body: &mir::Body<'_>, from: BasicBlock, local: Local) -> bool {
    let mut of = DenseBitSet::new_empty(body.local_decls.len());
    of.insert(local);
    let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut queue = VecDeque::from([from]);
    seen.insert(from);
    while let Some(block) = queue.pop_front() {
        let data = &body.basic_blocks[block];
        let found = reads_any(&of, |uses| {
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind
                    && let Rvalue::Discriminant(place) = &assign.1
                    && !place.is_indirect()
                    && place.local == local
                {
                    continue;
                }
                uses.visit_statement(statement, Location::START);
            }
            if let Some(terminator) = &data.terminator
                && !matches!(
                    &terminator.kind,
                    TerminatorKind::Drop { place, .. } if !place.is_indirect() && place.local == local
                )
            {
                uses.visit_terminator(terminator, Location::START);
            }
        });
        if found {
            return true;
        }
        if let Some(terminator) = &data.terminator {
            for next in terminator.successors() {
                if !body.basic_blocks[next].is_cleanup && seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
    }
    false
}

/// Whether a fn signature can name this type: no closure, coroutine, fn item
/// or opaque `impl Trait` anywhere in it. A local of such a type is fine if
/// only used inside the inner fn (inferred), not as its argument or return.
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

/// Whether a value of this type can hold a reference or pointer to a local:
/// a lifetime (`&T`, `Iter<'_, u8>`, a closure capturing `&x`, `dyn Trait`)
/// or a raw pointer type appears in the type or its type arguments. Struct
/// and enum fields are not inspected, so a type that keeps a raw pointer
/// only in a field and has no lifetime parameter (`Vec<u8>`,
/// `struct P(*mut u8)`) is false: it is assumed to own what it points to,
/// and storing a local's address in one takes `unsafe`, which this check
/// does not follow.
fn may_hold_address(ty: Ty<'_>) -> bool {
    ty.walk().any(|arg| match arg.kind() {
        GenericArgKind::Lifetime(_) => true,
        GenericArgKind::Type(ty) => ty.is_raw_ptr(),
        GenericArgKind::Const(_) => false,
    })
}

/// Whether a callee given a value of this type can, through it, store a
/// reference somewhere that outlives the call. True when it contains, at any
/// depth including struct and enum fields: a `&mut T` where `T` can hold an
/// address (`may_hold_address`); a raw pointer to such a `T`, even behind
/// `&`, since the value does not own the target and other copies usually
/// exist (the other handles of an `Rc<Cell<..>>` or `Arc<Mutex<..>>`);
/// behind `&`, a non-`Freeze` type (`Cell`, `Mutex`) around such a `T`; or a
/// type whose contents are not visible (trait object, alias that does not
/// normalize, coroutine, type parameter). What the callee owns (the value, a
/// `Box`'s contents) is dropped or returned, and a fn pointer holds nothing.
/// `fmt::Arguments` is exempt by name: it points at data `format_args!`
/// built in the caller's statement and no safe code writes through it. So
/// `&[u8]`, `usize`, `&Vec<&str>`, `fmt::Arguments` are false (the usual
/// arguments beside `&mut buf` in `write!(buf, ..)`); `&mut Vec<&u8>`,
/// `&Cell<Option<&T>>` are true.
fn may_take_address<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    // Each type to inspect, with whether data reached through it is still
    // writable: false after passing through a shared `&`.
    let mut pending = vec![(ty, true)];
    let mut seen: FxHashSet<(Ty<'tcx>, bool)> = FxHashSet::default();
    // Types nest finitely but instantiate without bound (`S<T>` holding a
    // `Box<S<(T, T)>>`); nothing real gets near the cap.
    let mut types_left = 4096u32;
    while let Some((ty, writable)) = pending.pop() {
        if !seen.insert((ty, writable)) {
            continue;
        }
        types_left -= 1;
        if types_left == 0 {
            return true;
        }
        match *ty.kind() {
            // Not owned; behind `&` the pointer is immutable, its target not.
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
                    // Owned like a field, but the container's `is_freeze`
                    // did not check the contents.
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
            // A trait object, a coroutine, a type parameter: contents unknown.
            _ => return true,
        }
    }
    false
}

/// Whether moving or dropping a local of this type ends every borrow taken
/// through it: `Box` and `&mut` yes; `&T` and raw pointers are `Copy`, and
/// the borrow checker records no borrow through one (`places_conflict`).
fn owns_pointee(ty: Ty<'_>) -> bool {
    !(ty.is_raw_ptr() || ty.ref_mutability() == Some(mir::Mutability::Not))
}

/// Whether the borrow checker records a borrow of `place` against
/// `place.local`: the local's own storage (`&x`, `&mut x.f`) or a place
/// reached through `Box` and `&mut` derefs only (`&**boxed`, `&mut *m`).
/// Moving the local meanwhile is E0505; not so after `&*r` on `r: &String`.
fn borrow_of_local<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>, place: Place<'tcx>) -> bool {
    place.iter_projections().all(|(base, elem)| {
        !matches!(elem, mir::ProjectionElem::Deref) || owns_pointee(base.ty(body, tcx).ty)
    })
}

/// Whether a plain read of `place` copies out a pointer whose uses the borrow
/// checker counts as uses of `place.local` (as `borrow_of_local`, for a read).
/// Source code cannot (a `&mut` is not `Copy`; a copied `&T` field borrows
/// nothing of its container), but two MIR passes can: `Derefer` copies the
/// `&mut` at `(*m).f` to reach `(*(*m).f).g`, and `ElaborateBoxDerefs`
/// copies a `Box`'s raw pointer or the whole `Box`.
fn copies_borrow<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>, place: Place<'tcx>) -> bool {
    if place.projection.is_empty() {
        return false;
    }
    if body.local_decls[place.local].ty.boxed_ty().is_some() {
        return true;
    }
    let read = place.ty(body, tcx).ty;
    place.is_indirect()
        && borrow_of_local(tcx, body, place)
        && (read.ref_mutability() == Some(mir::Mutability::Mut) || read.boxed_ty().is_some())
}

/// Whether `local`, by its type, could hold a tracked reference: it can hold
/// an address (`may_hold_address`), or it is a `Box` -- the `Box` temporary
/// `Derefer` copies from behind a reference points into the original.
fn can_hold_borrow(body: &mir::Body<'_>, local: Local) -> bool {
    let ty = body.local_decls[local].ty;
    may_hold_address(ty) || ty.boxed_ty().is_some()
}

/// What `borrow_holders` found about references into the locals `of`.
struct BorrowHolders {
    /// Locals that may hold a reference into one of `of`, or a value
    /// computed from one.
    any: DenseBitSet<Local>,
    /// The subset of `any` that comes from a `&mut` or `&raw mut` borrow.
    /// While one lives, even a read of the borrowed local is an error.
    exclusive: DenseBitSet<Local>,
    /// Some such reference may be held somewhere not in `any`: it was
    /// written through a pointer, passed to asm, a tail call or a yield, or
    /// passed to a call or put in a struct together with something that
    /// could store it (`v` in `v.push(&x)`, `Both { x: &x, v: &mut v }`).
    escaped: bool,
    /// The same, for a reference from an exclusive borrow.
    escaped_exclusive: bool,
}

impl BorrowHolders {
    fn escape(&mut self, exclusive: bool) {
        self.escaped = true;
        self.escaped_exclusive |= exclusive;
    }
}

/// Whether one operand may hold a reference that `borrow_holders` found (its
/// `bool`) while a different operand could store a reference (`storable`):
/// the code that receives both can then store the first in the second.
fn beside_storable<T>(
    mut operands: impl Iterator<Item = (bool, T)> + Clone,
    storable: impl Fn(T) -> bool,
) -> bool {
    let holding = operands.clone().filter(|&(holds, _)| holds).count();
    operands.any(|(holds, operand)| holding > usize::from(holds) && storable(operand))
}

/// Answers: does this piece of MIR produce a reference into one of the
/// locals in `of`, and if so, is the borrow exclusive? `ReadsBorrow::of`
/// visits one rvalue, operand, statement or terminator and returns both.
struct ReadsBorrow<'a, 'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    of: &'a DenseBitSet<Local>,
    found: &'a BorrowHolders,
    holds: bool,
    exclusive: bool,
}

impl<'a, 'mir, 'tcx> ReadsBorrow<'a, 'mir, 'tcx> {
    fn of(
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        of: &'a DenseBitSet<Local>,
        found: &'a BorrowHolders,
        visit: impl FnOnce(&mut Self),
    ) -> (bool, bool) {
        let mut visitor = ReadsBorrow {
            tcx,
            body,
            of,
            found,
            holds: false,
            exclusive: false,
        };
        visit(&mut visitor);
        (visitor.holds, visitor.exclusive)
    }
}

impl<'tcx> Visitor<'tcx> for ReadsBorrow<'_, '_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        use MutatingUseContext as M;
        use NonMutatingUseContext as N;
        let exclusive_borrow =
            matches!(context, PlaceContext::MutatingUse(M::Borrow | M::RawBorrow));
        if self.of.contains(place.local) {
            let (holds, exclusive) = match context {
                _ if exclusive_borrow => {
                    let of_local = borrow_of_local(self.tcx, self.body, *place);
                    (of_local, of_local)
                }
                PlaceContext::NonMutatingUse(N::SharedBorrow | N::FakeBorrow | N::RawBorrow) => {
                    (borrow_of_local(self.tcx, self.body, *place), false)
                }
                PlaceContext::NonMutatingUse(N::Copy | N::Move | N::Inspect) => {
                    let of_local = copies_borrow(self.tcx, self.body, *place);
                    let unique = place.ty(self.body, self.tcx).ty.ref_mutability()
                        == Some(mir::Mutability::Mut);
                    (of_local, of_local && unique)
                }
                _ => (false, false),
            };
            self.holds |= holds;
            self.exclusive |= exclusive;
        }
        // An exclusive reborrow through a local in `found` (`&mut *p`) is
        // exclusive even if that local was not.
        if exclusive_borrow && place.is_indirect() && self.found.any.contains(place.local) {
            self.holds = true;
            self.exclusive = true;
        }
        // `super_place` passes the base and any `Index` local to `visit_local`.
        self.super_place(place, context, location);
    }

    fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
        self.holds |= self.found.any.contains(local);
        self.exclusive |= self.found.exclusive.contains(local);
    }
}

/// Which locals in `blocks` may hold a reference into a local in `of`.
/// Found first: borrows and pointer copies that the borrow checker counts
/// against such a local (`borrow_of_local`, `copies_borrow`); then, until
/// nothing is added, each assignment or call result computed from a found
/// local whose type can hold an address (not a `usize` from `len()`).
/// Including a local that holds no such reference only makes one stretch
/// refuse; missing one that does would report a split that does not compile.
fn borrow_holders<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
    of: &DenseBitSet<Local>,
) -> BorrowHolders {
    let typing_env = body.typing_env(tcx);
    let can_hold = |place: &Place<'tcx>| !place.is_indirect() && can_hold_borrow(body, place.local);
    // Whether the code that receives `operand` could store a reference in
    // it (`may_take_address`). Not if it moves a whole local whose type shows
    // no address (a type parameter, say): nothing here reads that local again.
    let storable = |operand: &Operand<'tcx>| {
        let moved_away = matches!(operand, Operand::Move(place)
            if place.projection.is_empty()
                && !may_hold_address(body.local_decls[place.local].ty));
        !moved_away && may_take_address(tcx, typing_env, operand.ty(&body.local_decls, tcx))
    };
    let empty = DenseBitSet::new_empty(body.local_decls.len());
    let mut found = BorrowHolders {
        any: empty.clone(),
        exclusive: empty,
        escaped: false,
        escaped_exclusive: false,
    };
    // Each pass only adds locals, and finitely many qualify: a pass that
    // adds none is the fixpoint.
    loop {
        let mut grew = false;
        for block in blocks.iter() {
            let data = &body.basic_blocks[block];
            for statement in &data.statements {
                match &statement.kind {
                    StatementKind::Assign(assign) => {
                        let (dest, rvalue) = &**assign;
                        let (holds, exclusive) = ReadsBorrow::of(tcx, body, of, &found, |p| {
                            p.visit_rvalue(rvalue, Location::START);
                        });
                        // A struct, tuple or closure that holds both a found
                        // reference and something that could store one
                        // (`Both { v: &buf[..n], p: &mut parts }`) reaches a
                        // call as one argument, where the rule below does not
                        // check them together: count it as escaped when built
                        // (`built`) or when either is assigned into a field
                        // (`beside`).
                        let built = match rvalue {
                            Rvalue::Aggregate(_, operands) if holds && operands.len() > 1 => {
                                let by_operand: Vec<bool> = operands
                                    .iter()
                                    .map(|operand| {
                                        ReadsBorrow::of(tcx, body, of, &found, |p| {
                                            p.visit_operand(operand, Location::START);
                                        })
                                        .0
                                    })
                                    .collect();
                                beside_storable(by_operand.iter().copied().zip(operands), storable)
                            }
                            _ => false,
                        };
                        let beside = !dest.projection.is_empty()
                            && !dest.is_indirect()
                            && (holds || found.any.contains(dest.local))
                            && may_take_address(tcx, typing_env, body.local_decls[dest.local].ty);
                        if built || beside {
                            found.escape(exclusive || found.exclusive.contains(dest.local));
                        }
                        if !holds {
                            continue;
                        }
                        if dest.is_indirect() {
                            found.escape(exclusive);
                        } else if can_hold(dest) {
                            grew |= found.any.insert(dest.local);
                            if exclusive {
                                grew |= found.exclusive.insert(dest.local);
                            }
                        }
                    }
                    // `copy_nonoverlapping` writes through pointers: a found
                    // reference in it may be stored anywhere.
                    StatementKind::Intrinsic(_) => {
                        let (holds, exclusive) = ReadsBorrow::of(tcx, body, of, &found, |p| {
                            p.visit_statement(statement, Location::START);
                        });
                        if holds {
                            found.escape(exclusive);
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
                        ReadsBorrow::of(tcx, body, of, &found, |p| {
                            p.visit_operand(operand, Location::START);
                        })
                    };
                    let by_func = pick(func);
                    let by_arg: Vec<(bool, bool)> =
                        args.iter().map(|arg| pick(&arg.node)).collect();
                    let holds = by_arg.iter().any(|&(holds, _)| holds);
                    let exclusive = by_arg.iter().any(|&(_, exclusive)| exclusive);
                    if by_func.0 {
                        found.escape(by_func.1);
                    }
                    if !holds {
                        continue;
                    }
                    // With another argument to store the reference in, the
                    // callee may do so; otherwise it can only return it.
                    let arg_holds = by_arg.iter().map(|&(holds, _)| holds);
                    let can_store = beside_storable(arg_holds.zip(args), |arg| storable(&arg.node));
                    if can_store || destination.is_indirect() {
                        found.escape(exclusive);
                    }
                    if can_hold(destination) {
                        grew |= found.any.insert(destination.local);
                        if exclusive {
                            grew |= found.exclusive.insert(destination.local);
                        }
                    }
                }
                TerminatorKind::TailCall { .. }
                | TerminatorKind::InlineAsm { .. }
                | TerminatorKind::Yield { .. } => {
                    let (holds, exclusive) = ReadsBorrow::of(tcx, body, of, &found, |p| {
                        p.visit_terminator(terminator, Location::START);
                    });
                    if holds {
                        found.escape(exclusive);
                    }
                }
                // The rest only read operands or use no local; nothing stored.
                _ => {}
            }
        }
        if !grew {
            break;
        }
    }
    found
}

/// Whether a value of this type can hold a borrow the borrow checker
/// enforces: a lifetime appears in it. A raw pointer holds an address but
/// no such borrow; a `usize` read through a borrow holds neither.
fn has_lifetime(ty: Ty<'_>) -> bool {
    ty.walk()
        .any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(_)))
}

/// Whether a returned value of this type borrows a read-only param as
/// shared only: a `&T` with `T` holding no address, so no `&mut` can be
/// inside. Anything else (`&mut [u8]`, `Iter<'_, T>`) counts as exclusive.
fn shared_ref_only(ty: Ty<'_>) -> bool {
    matches!(*ty.kind(), ty::Ref(_, pointee, mir::Mutability::Not) if !may_hold_address(pointee))
}

/// One stretch and the body around it, for the last check in `stretch_io`:
/// whether the outer fn uses a param after the call returns while a
/// returned value that may borrow the param is still live.
struct PastExit<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    locals: &'a LocalFacts,
    /// The stretch's blocks.
    stretch: &'a DenseBitSet<BasicBlock>,
    /// The only block of the stretch that an edge from outside it enters, so
    /// the first block of the stretch on any path from `exit` back into it.
    /// Reaching it runs the call again, which borrows the params again.
    entry: BasicBlock,
    /// The block the call returns to.
    exit: BasicBlock,
}

/// Finds whether one statement or terminator uses `local` in a way that a
/// live borrow of `local` forbids. An exclusive borrow forbids every use. A
/// shared borrow forbids a write, move, drop or `&mut` of `local` or of a
/// place reached through it (`p.count += 1`, `consume(p)`). A use through a
/// pointer copied out of `local` (`through`) counts; storage markers do not.
struct ConflictingUse<'a> {
    local: Local,
    /// `local` plus the locals `pointer_copies_of` finds for it.
    through: &'a DenseBitSet<Local>,
    exclusive: bool,
    found: bool,
}

impl<'tcx> Visitor<'tcx> for ConflictingUse<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if place.local == self.local || place.is_indirect() && self.through.contains(place.local) {
            self.found |= match context {
                PlaceContext::NonUse(_) => false,
                PlaceContext::MutatingUse(_)
                | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => true,
                PlaceContext::NonMutatingUse(_) => self.exclusive,
            };
        }
        // `super_place` passes any `Index` local to `visit_local`, as a read.
        self.super_place(place, context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if local == self.local && self.exclusive && context.is_use() {
            self.found = true;
        }
    }
}

impl PastExit<'_, '_> {
    /// Whether the body after `exit` uses a param, in a way the borrow
    /// forbids (`ConflictingUse`), while a returned value that may borrow it
    /// is live.
    /// In the inner fn's signature such a value borrows the whole param, so
    /// until the value is dead the outer fn may not write, move, drop or
    /// `&mut` the param, and may not use it at all when the borrow is
    /// exclusive. Only locals in `returns` whose type has a lifetime can
    /// borrow (`has_lifetime`). For each param, `borrow_holders` over the
    /// stretch finds which of them may borrow it, and `used_while_borrowed`
    /// checks the uses after `exit`. The borrow is exclusive for a param in
    /// `exclusive_locals`
    /// (passed by value or `&mut`) or the returned type may hold a `&mut`
    /// (`shared_ref_only` is false). A returned param does not borrow itself.
    fn result_borrows_param(
        &self,
        exclusive_locals: &DenseBitSet<Local>,
        params: &DenseBitSet<Local>,
        returns: &DenseBitSet<Local>,
    ) -> bool {
        let body = self.body;
        let borrowing_results: Vec<Local> = returns
            .iter()
            .filter(|&local| has_lifetime(body.local_decls[local].ty))
            .collect();
        if borrowing_results.is_empty() {
            return false;
        }
        // The blocks after `exit` and whether they reach `entry`, computed
        // on first use; most stretches never need them.
        let mut past: Option<(DenseBitSet<BasicBlock>, bool)> = None;
        let mut one_param = DenseBitSet::new_empty(body.local_decls.len());
        for param in params.iter() {
            one_param.clear();
            one_param.insert(param);
            let holders = borrow_holders(self.tcx, body, self.stretch, &one_param);
            for &result in &borrowing_results {
                if result == param || !holders.any.contains(result) {
                    continue;
                }
                let exclusive = exclusive_locals.contains(param)
                    || !shared_ref_only(body.local_decls[result].ty);
                let (blocks, reenters) = past.get_or_insert_with(|| self.reach());
                if self.used_while_borrowed(blocks, *reenters, param, result, exclusive) {
                    return true;
                }
            }
        }
        false
    }

    /// The blocks control can be in after the call returns: `exit` itself
    /// and every block reachable from it without entering the stretch again,
    /// leaving out unwind cleanup blocks. Also whether some edge from those
    /// blocks enters the stretch, which can only happen at `entry`.
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
                if self.stretch.contains(next) {
                    reenters = true;
                } else if seen.insert(next) {
                    pending.push(next);
                }
            }
        }
        (seen, reenters)
    }

    /// Whether the body after `exit` uses `param` in a way the borrow forbids
    /// (`ConflictingUse`) while the borrow is live: while `result` or a local
    /// assigned from it is live (`locals_holding`), and anywhere after one of
    /// them is stored other than in a local (`stored_untracked_from`). When
    /// exclusive, reaching `entry` with it live counts: the call there
    /// borrows `param`.
    fn used_while_borrowed(
        &self,
        past: &DenseBitSet<BasicBlock>,
        reenters: bool,
        param: Local,
        result: Local,
        exclusive: bool,
    ) -> bool {
        let body = self.body;
        let holding = locals_holding(body, past, result);
        let (live_throughout, store_reenters) = self.stored_untracked_from(past, &holding);
        let through = pointer_copies_of(self.tcx, body, past, param);
        let holding_live_at = |block: BasicBlock| {
            self.locals
                .live
                .get(&block)
                .is_some_and(|live| holding.iter().any(|local| live.contains(local)))
        };
        if exclusive && (store_reenters || reenters && holding_live_at(self.entry)) {
            return true;
        }
        for block in past.iter() {
            let live_throughout = live_throughout.contains(block);
            // Not in `live_throughout` and the borrow is dead on entry, so
            // any read of `holding` below is of a new value.
            if !live_throughout && !holding_live_at(block) {
                continue;
            }
            let data = &body.basic_blocks[block];
            let live = (!live_throughout).then(|| live_before(body, self.locals, block, &holding));
            let holding_live_before = |index: usize| live.as_ref().is_none_or(|live| live[index]);
            let mut param_use = ConflictingUse {
                local: param,
                through: &through,
                exclusive,
                found: false,
            };
            for (index, statement) in data.statements.iter().enumerate() {
                if !holding_live_before(index) {
                    continue;
                }
                param_use.visit_statement(statement, Location::START);
                if param_use.found {
                    return true;
                }
            }
            if let Some(terminator) = &data.terminator
                && holding_live_before(data.statements.len())
            {
                param_use.visit_terminator(terminator, Location::START);
                if param_use.found {
                    return true;
                }
            }
        }
        false
    }

    /// Blocks of `past` where the borrow counts as live at every point: each
    /// block with a store `escapes` finds, and every block of `past` reachable
    /// from one. A path from one into the stretch leaves at `exit` again and
    /// so reaches all of `past`. Also returns whether such a path into the
    /// stretch exists.
    fn stored_untracked_from(
        &self,
        past: &DenseBitSet<BasicBlock>,
        holding: &DenseBitSet<Local>,
    ) -> (DenseBitSet<BasicBlock>, bool) {
        let body = self.body;
        let mut live_throughout = DenseBitSet::new_empty(body.basic_blocks.len());
        let mut pending: Vec<BasicBlock> = past
            .iter()
            .filter(|&block| escapes(self.tcx, body, block, holding))
            .collect();
        for &block in &pending {
            live_throughout.insert(block);
        }
        let mut reenters = false;
        while let Some(block) = pending.pop() {
            let Some(terminator) = &body.basic_blocks[block].terminator else {
                continue;
            };
            for mut next in terminator.successors() {
                if self.stretch.contains(next) {
                    reenters = true;
                    // Running the stretch again does not end a borrow stored
                    // after it; go on from `exit`.
                    next = self.exit;
                }
                if past.contains(next) && live_throughout.insert(next) {
                    pending.push(next);
                }
            }
        }
        (live_throughout, reenters)
    }
}

/// Whether `block` stores a borrow held by a local in `holding` somewhere
/// other than a local. Only an operand whose type has a lifetime holds it
/// (`has_lifetime`; not `chunk.len()`). The stores are: assigning it through
/// a pointer to a place that can hold a borrow (`*slot = chunk`); an
/// intrinsic, asm, yield or tail call reading it; calling it; and passing
/// it to a call whose destination is behind a pointer or whose other
/// arguments could store it (`parts.push(chunk)`, `may_take_address`).
fn escapes<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    block: BasicBlock,
    holding: &DenseBitSet<Local>,
) -> bool {
    let data = &body.basic_blocks[block];
    let holds = |operand: &Operand<'tcx>| {
        has_lifetime(operand.ty(&body.local_decls, tcx))
            && reads_any(holding, |uses| uses.visit_operand(operand, Location::START))
    };
    for statement in &data.statements {
        let escaped = match &statement.kind {
            StatementKind::Assign(assign) => {
                let (dest, rvalue) = &**assign;
                dest.is_indirect()
                    && has_lifetime(dest.ty(body, tcx).ty)
                    && reads_any(holding, |uses| uses.visit_rvalue(rvalue, Location::START))
            }
            StatementKind::Intrinsic(_) => reads_any(holding, |uses| {
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
            if holds(func) {
                return true;
            }
            let by_arg: Vec<bool> = args.iter().map(|arg| holds(&arg.node)).collect();
            if !by_arg.contains(&true) {
                return false;
            }
            if destination.is_indirect() && has_lifetime(destination.ty(body, tcx).ty) {
                return true;
            }
            let typing_env = body.typing_env(tcx);
            // Unlike `borrow_holders`, an argument moved in whole counts:
            // the callee may keep it.
            beside_storable(by_arg.iter().copied().zip(args), |arg| {
                may_take_address(tcx, typing_env, arg.node.ty(&body.local_decls, tcx))
            })
        }
        TerminatorKind::TailCall { .. }
        | TerminatorKind::InlineAsm { .. }
        | TerminatorKind::Yield { .. } => reads_any(holding, |uses| {
            uses.visit_terminator(terminator, Location::START)
        }),
        _ => false,
    }
}

/// The locals that keep the returned value's borrow live after the call:
/// `result` itself, plus every local in `blocks` that is assigned, or
/// receives a call result, from a local already in the set, when its type
/// can hold a borrow (`let rest = &chunk[1..]`, `let pair = (chunk, n)`;
/// not `chunk.len()`).
fn locals_holding(
    body: &mir::Body<'_>,
    blocks: &DenseBitSet<BasicBlock>,
    result: Local,
) -> DenseBitSet<Local> {
    let mut holding = DenseBitSet::new_empty(body.local_decls.len());
    holding.insert(result);
    add_locals_computed_from(
        body,
        || blocks.iter(),
        &mut holding,
        |dest| !dest.is_indirect() && has_lifetime(body.local_decls[dest.local].ty),
    );
    holding
}

/// For each statement of `block` and for its terminator, whether any local
/// in `holding` is live just before it, ignoring unwind edges. Successor live
/// sets come from `LocalFacts`, and a `return` reads `_0`.
fn live_before(
    body: &mir::Body<'_>,
    locals: &LocalFacts,
    block: BasicBlock,
    holding: &DenseBitSet<Local>,
) -> Vec<bool> {
    let mut state = DenseBitSet::new_empty(body.local_decls.len());
    let mut live = vec![false; body.basic_blocks[block].statements.len() + 1];
    block_live(
        body,
        block,
        &locals.live,
        &locals.returned,
        &mut state,
        |index, state| live[index] = holding.iter().any(|local| state.contains(local)),
    );
    live
}

// ── the stretch search ───────────────────────────────────────────────────────
//
// The search looks for a set of whole blocks (a stretch) that control
// enters at one block (its entry) and leaves toward one node (its exit),
// and that could become a fn of its own. It works on `FlowGraph`, which
// has only the edges taken when nothing panics; every `return` is an edge
// to one added node, EXIT, and the exit is a block or EXIT; a block that
// never returns has no successor. The stretch must pass these checks:
//
//   (1) every block of it is not unwind cleanup, nothing counted in it is
//       dependent, it is not left by a `become` (sent to EXIT like a `return`,
//       but it never assigns `_0` and needs the caller's own signature), and
//       no local it names has a type a parameter appears in (`fit_blocks`
//       checks this once per block);
//   (2) it has one entry: every predecessor of a member other than the
//       entry is a member, and the entry itself is the fn's entry block or
//       has a predecessor outside the stretch (`StretchBuilder::single_entry`);
//   (3) it has one exit: every successor of a member is a member or is the
//       exit. A `return` has successor EXIT, so it is allowed exactly when
//       the exit is EXIT. A diverging block has no successor and constrains
//       nothing (`StretchBuilder::single_exit`);
//   (4) every local it reads before writing has a parameter-free type, and
//   (5) so does every local it writes that is read after the exit. Both
//       follow from (1), since those locals are ones the stretch names.
//       `stretch_io` computes the two sets and refuses what a signature
//       cannot express (a constant of the instantiation as a param, a drop
//       flag as a param or result, a borrowed by-value param, an escaping
//       address: that section);
//   (6) items that are not counted (storage markers, plain jumps) emit no
//       code and stay with the outer fn's locals, so they need no check;
//   (7) its hand-written counted items (`BlockFacts::hand_written`) number
//       at least `min_statements` (`StretchBuilder::candidate`). The reported
//       size counts all of its counted items, macro-written ones included.
//
// **Why such a stretch is one call.** Build an inner fn whose parameters are
// the locals the stretch reads before writing and whose body is the stretch's
// blocks, entry first, with each edge to the exit made a `return` of the
// tuple of the locals in (5) (`_0` itself when the exit is EXIT). No block,
// param or result of it names a type or const parameter, so it is compiled
// once. In the outer fn replace the stretch by one block,
// `let (rets) = inner(params)`, with return edge to the exit. The stretch
// had one entry, so all control that entered it now enters the call; it had
// one exit, so all control that left it now returns from the call to the
// same block. Panics and divergence happen inside the inner fn, and
// unwinding drops its locals and then the outer fn's: the same set as
// before. The outer signature is unchanged, so callers are too. The cost is
// one call, run once per iteration when the exit can reach the entry again
// (`in_loop`).
//
// **How the candidates are enumerated.** When the exit is a block, having
// one exit puts it on every path from the entry to EXIT: the exit is a
// post-dominator of the entry. Given the entry and such an exit, one entry
// and one exit allow one stretch only: the blocks reachable from the entry
// without entering the exit. So an entry's candidate exits are its nearest
// post-dominator, then that block's nearest post-dominator, and so on to
// EXIT (its immediate post-dominator chain). The stretches for those exits
// nest, so one breadth-first search from the entry, paused at each
// candidate exit, yields all of them in the time of the largest. An unfit
// block ends the chain. The one-entry, one-exit and size checks can fail at
// one candidate exit and hold at a later one, and the first two are
// counters updated per edge, so checking a candidate reads only the exit's
// predecessors, not the whole stretch.
//
// Every fit block on a path to a `return` is tried as an entry, in reverse
// post-order (each block before its successors, back edges aside), except one
// that runs only under a branch on a constant of the instantiation
// (`blocks_under_const_branch`). The largest valid stretch is kept
// (hand-written size, then the earlier entry). Two filters avoid that work per
// entry. First, a stretch lies among the blocks reachable only through its
// entry (the entry's dominator subtree), cut at unfit blocks, so that subtree's
// hand-written total, summed once bottom-up, bounds the stretch's size: an
// entry whose bound is under `min_statements`, or not above the best so far, is
// skipped. Second, `stretch_io`, whose cost also grows with locals, runs only
// afterwards, on the passing candidates that would exceed the best: largest
// first, then bisecting down the chain if that is refused
// (`Search::largest_passing`), with at most `MAX_IO_CHECKS_PER_ROUND` runs
// summed over all entries of a round. So a fn costs one classification pass,
// two dominator computations, at most one breadth-first search per entry the
// size bound does not skip, and a bounded number of `stretch_io` runs.
// Body-wide liveness is built only once a candidate is fit, has one entry and
// one exit, and is large enough. A body with under `min_statements`
// hand-written items in its blocks with nothing dependent is rejected before
// the flow graph is built, and in its fit blocks after.
//
// `other_stretches` repeats the search with the found stretch's blocks
// excluded, up to `OTHER_STRETCHES_CAP` more times, excluding each result in
// turn: a `const FLAG: bool` tested once mid-body leaves a stretch on each
// side. Blocks are never split, so a statement that is not dependent is
// lost when it shares a block with one that is (the lint's doc says so).
// For the same reason a `count += 1` whose checked addition is computed in
// one block and whose sum is stored in the next keeps that whole block out
// of a stretch ending between the two: the inner fn would have to return
// the unnamed `(u32, bool)` pair, so `StretchBuilder::collect` skips that
// exit. The exits before and after it on the same chain are still
// candidates.

/// A set of whole basic blocks that passes every check in the note above,
/// so it could become a non-generic fn called once from where the blocks
/// were.
#[derive(Clone, Debug)]
pub(crate) struct Stretch {
    /// The one block control enters the stretch at.
    pub(crate) entry: BasicBlock,
    /// The one node every edge out of the stretch goes to; `None` when that
    /// node is EXIT (the stretch runs to the fn's `return`).
    pub(crate) exit: Option<BasicBlock>,
    /// The member blocks, indexed like `body.basic_blocks`.
    pub(crate) blocks: DenseBitSet<BasicBlock>,
    /// Counted statements and terminators in the members: the size a
    /// finding prints. At least `min_statements` of them are hand-written.
    pub(crate) size: usize,
    /// Hand-written items among `size`: what is compared to `min_statements`
    /// and what stretches are ranked by.
    written: usize,
    /// Locals read in the stretch before the stretch writes them, in
    /// declaration order: the inner fn's parameters (`stretch_io`).
    pub(crate) params: Vec<Local>,
    /// Locals the stretch may change that are read after it: the inner fn's
    /// return values. `_0` when `exit` is `None` and the stretch assigns it.
    pub(crate) returns: Vec<Local>,
    /// The exit can reach the entry again, so the call replacing the stretch
    /// runs once per iteration of an enclosing loop.
    pub(crate) in_loop: bool,
    /// How many further valid stretches, disjoint from this one and from each
    /// other, a repeated search found. At most `OTHER_STRETCHES_CAP`.
    pub(crate) other_stretches: usize,
}

/// How many further disjoint stretches `best_stretch` counts after the best
/// one. Each costs a full search round.
const OTHER_STRETCHES_CAP: usize = 3;

/// The most `stretch_io` calls one search round makes, summed over all its
/// entry blocks. Each call is a liveness fixpoint plus an address walk over
/// the body, so the cap is per round, not per entry: a body whose candidates
/// keep failing would otherwise cost a set of calls per entry block. A round
/// that reaches the cap keeps the best stretch found so far.
const MAX_IO_CHECKS_PER_ROUND: usize = 32;

/// The blocks a stretch may be made of: in the flow graph, nothing counted
/// in the block is dependent, no local it uses has a type with a parameter,
/// and it does not end in a tail call (which needs the caller's own
/// signature).
/// Blocks excluded by an earlier search round are tracked separately, in
/// `excluded`, so this is computed once per body.
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
        if !flow.contains(block) || facts[block].dependent != 0 {
            continue;
        }
        if let Some(terminator) = &data.terminator
            && matches!(terminator.kind, TerminatorKind::TailCall { .. })
        {
            continue;
        }
        let generic_use = reads_any(&generic, |uses| {
            for statement in &data.statements {
                uses.visit_statement(statement, Location::START);
            }
            if let Some(terminator) = &data.terminator
                && !matches!(terminator.kind, TerminatorKind::Return)
            {
                uses.visit_terminator(terminator, Location::START);
            }
        });
        if !generic_use {
            fit.insert(block);
        }
    }
    fit
}

/// The blocks that run only when a branch on a per-instantiation constant
/// goes one way. No stretch may be entered at one: after constant propagation
/// each copy keeps only the arm its own constant selects, so such a block is
/// not the same code in every copy. The branches looked for are `SwitchInt`s
/// whose discriminant is a constant naming a parameter (`if FLAG`) or reads a
/// `per_copy_consts` local (`match coding_of(TAG)`, `N > 4`); the blocks each
/// one decides are those control-dependent on it, directly or through further
/// branches, up to where its arms rejoin (`FlowGraph::decides`). A block that
/// post-dominates the branch runs either way and is not decided. The switch
/// block itself need not be dependent, so this reads the flow graph, not the
/// classifier, and repeats for branches inside decided blocks until nothing
/// changes.
fn blocks_under_const_branch(
    body: &mir::Body<'_>,
    flow: &FlowGraph,
    per_copy_consts: &DenseBitSet<Local>,
) -> DenseBitSet<BasicBlock> {
    // A value that is a constant in each copy: a constant that names a
    // parameter, or a place that reads a `per_copy_consts` local (the local
    // switched on, or one indexing it). A `RuntimeChecks` operand asks the
    // session, not the instantiation, and is the same in every copy.
    let per_copy = |discr: &Operand<'_>| match discr {
        Operand::Constant(constant) => constant.const_.has_non_region_param(),
        Operand::Copy(place) | Operand::Move(place) => reads_any(per_copy_consts, |uses| {
            uses.visit_place(
                place,
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect),
                Location::START,
            );
        }),
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
    // Each branch is expanded once: the first time it is found to switch on
    // a per-copy constant or to run under such a branch. What it decides is
    // marked, and any block among that with a choice of successors is a
    // branch whose every arm is decided too.
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
/// stretch with that block as entry: the hand-written item total of the fit
/// blocks in its dominator subtree, with each path down the subtree cut at
/// its first unfit block. (A stretch lies inside its entry's dominator
/// subtree because it has one entry.) One bottom-up pass over the dominator
/// tree. Unwind cleanup blocks are in the tree but never fit, so they and
/// everything below them add nothing.
fn stretch_size_bounds(
    facts: &IndexSlice<BasicBlock, BlockFacts>,
    body: &mir::Body<'_>,
    fit: &DenseBitSet<BasicBlock>,
    n: usize,
) -> IndexVec<BasicBlock, usize> {
    let mut children: IndexVec<BasicBlock, Vec<BasicBlock>> = IndexVec::from_elem_n(Vec::new(), n);
    let dominators = body.basic_blocks.dominators();
    for block in body.basic_blocks.indices() {
        if let Some(parent) = dominators.immediate_dominator(block) {
            children[parent].push(block);
        }
    }
    let mut bound: IndexVec<BasicBlock, usize> = IndexVec::from_elem_n(0, n);
    // Post-order over the dominator tree without recursion: a block is
    // summed once all its children have been.
    let mut stack: Vec<(BasicBlock, bool)> = vec![(mir::START_BLOCK, false)];
    while let Some((block, summed_children)) = stack.pop() {
        if summed_children {
            if fit.contains(block) {
                bound[block] = facts[block].hand_written as usize
                    + children[block].iter().map(|&c| bound[c]).sum::<usize>();
            }
        } else {
            stack.push((block, true));
            stack.extend(children[block].iter().map(|&c| (c, false)));
        }
    }
    bound
}

/// One candidate stretch for an entry: the first `members` blocks collected
/// from the entry, with exit `exit`. All are fit, the stretch has one entry
/// and one exit, and it is large enough; `stretch_io` has not run on it.
struct Candidate {
    exit: BasicBlock,
    members: usize,
    /// Counted items in the members: what the finding prints.
    size: usize,
    /// Hand-written items among them: what is compared to `min_statements`
    /// and what candidates are ranked by.
    written: usize,
}

/// Collects the candidate stretches for one entry block by breadth-first
/// search: the blocks added so far, and two edge counts that tell whether
/// the stretch has one entry and one exit. Reused across entries after a
/// `reset`.
struct StretchBuilder<'a> {
    flow: &'a FlowGraph,
    facts: &'a IndexSlice<BasicBlock, BlockFacts>,
    fit: &'a DenseBitSet<BasicBlock>,
    in_stretch: DenseBitSet<BasicBlock>,
    /// The stretch's blocks in the order they were added, entry first. Each
    /// `Candidate` is a prefix of it.
    members: Vec<BasicBlock>,
    /// Counted items over the members (the reported size), and the
    /// hand-written ones among them (what is compared to `min_statements`,
    /// so a stretch made of one macro expansion fails).
    size: usize,
    written: usize,
    /// Edges from outside the stretch into a member other than the entry.
    entries: usize,
    /// Edges from a member to a node outside the stretch (EXIT included).
    exits: usize,
    queue: VecDeque<BasicBlock>,
}

impl<'a> StretchBuilder<'a> {
    fn new(
        flow: &'a FlowGraph,
        facts: &'a IndexSlice<BasicBlock, BlockFacts>,
        fit: &'a DenseBitSet<BasicBlock>,
        n: usize,
    ) -> Self {
        StretchBuilder {
            flow,
            facts,
            fit,
            // Sized `n + 1` so EXIT has a bit, which is never set.
            in_stretch: DenseBitSet::new_empty(n + 1),
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
            self.in_stretch.remove(block);
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

    /// Adds a fit block and updates `entries` and `exits` for each of its
    /// edges; a self-edge changes neither.
    fn add(&mut self, block: BasicBlock) {
        debug_assert!(self.fit.contains(block) && !self.in_stretch.contains(block));
        let is_entry = self.members.is_empty();
        self.in_stretch.insert(block);
        self.members.push(block);
        self.size += self.facts[block].counted as usize;
        self.written += self.facts[block].hand_written as usize;
        for &pred in self.flow.preds(block) {
            if pred == block {
                continue;
            }
            if self.in_stretch.contains(pred) {
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
            if self.in_stretch.contains(succ) {
                // Edges into the entry were never counted in `entries`.
                if succ != entry {
                    self.entries -= 1;
                }
            } else {
                self.exits += 1;
            }
        }
        self.queue.push_back(block);
    }

    /// Adds every fit block reachable from the queue without entering
    /// `stop_at` or EXIT. Returns `false` on meeting an unfit or excluded
    /// block: that block would be in this candidate and in every larger one
    /// from this entry, so the caller stops.
    fn add_avoiding(&mut self, stop_at: BasicBlock, excluded: &DenseBitSet<BasicBlock>) -> bool {
        let exit = self.flow.exit();
        while let Some(block) = self.queue.pop_front() {
            for &succ in self.flow.succs(block) {
                if succ == stop_at || succ == exit || self.in_stretch.contains(succ) {
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

    /// Whether the stretch has one entry: no edge from outside it enters a
    /// member other than the entry. The clause about the entry itself can
    /// only fail for an unreachable entry, which is never tried.
    fn single_entry(&self) -> bool {
        let entry = self.entry();
        self.entries == 0
            && (entry == mir::START_BLOCK
                || self
                    .flow
                    .preds(entry)
                    .iter()
                    .any(|&p| !self.in_stretch.contains(p)))
    }

    /// Whether the stretch has one exit: every edge out of it goes to
    /// `exit`. True exactly
    /// when the count of edges out equals the count of `exit`'s predecessors
    /// inside the stretch. A `return` inside the stretch is an edge to EXIT,
    /// so it fails this for any other `exit`.
    fn single_exit(&self, exit: BasicBlock) -> bool {
        let exit_preds_inside = self
            .flow
            .preds(exit)
            .iter()
            .filter(|&&p| self.in_stretch.contains(p))
            .count();
        self.exits == exit_preds_inside
    }

    /// The candidate made of the members collected so far, if it has one
    /// entry, one exit, and at least `min_statements` hand-written items.
    /// The search never makes `exit` a member; that is
    /// checked anyway because `single_exit` is only correct when it is not.
    fn candidate(&self, exit: BasicBlock, min_statements: usize) -> Option<Candidate> {
        (self.written >= min_statements
            && !self.in_stretch.contains(exit)
            && self.single_entry()
            && self.single_exit(exit))
        .then_some(Candidate {
            exit,
            members: self.members.len(),
            size: self.size,
            written: self.written,
        })
    }

    /// Tries each block on `entry`'s post-dominator chain as the exit and
    /// returns every candidate that passed the structural rules, smallest
    /// first (sizes never decrease along the chain). An exit whose
    /// predecessor in the stretch ends in the overflow check of a `count +=
    /// 1` is left out (`ends_in_overflow_check`): the sum is stored in the
    /// exit, so the inner fn would have to return the unnamed `(sum,
    /// overflowed)` pair. The candidate for the next exit on the chain, if
    /// there is one, contains that store.
    fn collect(
        &mut self,
        body: &mir::Body<'_>,
        entry: BasicBlock,
        excluded: &DenseBitSet<BasicBlock>,
        min_statements: usize,
    ) -> Vec<Candidate> {
        self.reset();
        self.add(entry);
        let mut passed = Vec::new();
        let mut previous: Option<BasicBlock> = None;
        let chain: Vec<BasicBlock> = self.flow.ipdom_chain(entry).collect();
        for exit in chain {
            // The previous exit post-dominates the entry, so it is reachable
            // and belongs to this larger candidate.
            if let Some(previous) = previous
                && !self.in_stretch.contains(previous)
            {
                if !self.fit.contains(previous) || excluded.contains(previous) {
                    break;
                }
                self.add(previous);
            }
            if !self.add_avoiding(exit, excluded) {
                break;
            }
            let returns_pair =
                self.flow.preds(exit).iter().any(|&pred| {
                    self.in_stretch.contains(pred) && ends_in_overflow_check(body, pred)
                });
            if !returns_pair {
                passed.extend(self.candidate(exit, min_statements));
            }
            previous = Some(exit);
        }
        passed
    }

    /// The first `members` blocks added, as a set.
    fn blocks(&self, members: usize) -> DenseBitSet<BasicBlock> {
        let mut blocks = DenseBitSet::new_empty(self.facts.len());
        for &block in &self.members[..members] {
            blocks.insert(block);
        }
        blocks
    }
}

/// Whether `block` computes checked arithmetic into a local the source
/// never named and ends by asserting that it did not overflow: the `(sum,
/// overflowed)` pair of a `count += 1`. The block after the assert reads the
/// sum out of that pair, so the pair is live on the edge between them.
fn ends_in_overflow_check(body: &mir::Body<'_>, block: BasicBlock) -> bool {
    let data = &body.basic_blocks[block];
    let TerminatorKind::Assert { cond, msg, .. } = &data.terminator().kind else {
        return false;
    };
    let Some(pair) = cond.place().map(|tested| tested.local) else {
        return false;
    };
    matches!(**msg, mir::AssertKind::Overflow(..))
        && local_name(body, pair).is_none()
        && data.statements.iter().any(|statement| {
            matches!(
                &statement.kind,
                StatementKind::Assign(assign)
                    if assign.0.as_local() == Some(pair)
                        && matches!(assign.1, Rvalue::BinaryOp(op, _) if op.is_overflowing())
            )
        })
}

/// Per-fn state shared by every search round. `locals` is the body-wide
/// liveness `stretch_io` needs, built on first use and kept.
struct Search<'a, 'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    flow: &'a FlowGraph,
    builder: StretchBuilder<'a>,
    /// Per block, `stretch_size_bounds`.
    size_bound: IndexVec<BasicBlock, usize>,
    /// Passed on to `LocalFacts`.
    per_copy_consts: DenseBitSet<Local>,
    /// Blocks never allowed as a stretch's entry.
    blocks_under_const_branch: DenseBitSet<BasicBlock>,
    locals: Option<LocalFacts>,
    min_statements: usize,
}

impl Search<'_, '_, '_> {
    /// Runs one search over blocks that are fit and not `excluded` and
    /// returns the best stretch that passed every check, ranked by
    /// hand-written size, then by earlier entry in reverse post-order. It
    /// can miss the largest when the binary search in `largest_passing`
    /// skipped it or the round reached `MAX_IO_CHECKS_PER_ROUND`, but never
    /// returns an invalid one.
    fn round(&mut self, excluded: &DenseBitSet<BasicBlock>) -> Option<Stretch> {
        let mut best: Option<Stretch> = None;
        let mut io_checks_left = MAX_IO_CHECKS_PER_ROUND;
        for &entry in self.body.basic_blocks.reverse_postorder() {
            if io_checks_left == 0 {
                break;
            }
            // Skip blocks that cannot be an entry: unfit, excluded, under a
            // constant branch (not entered in every copy), or on no path to a `return` (a
            // stretch needs an exit, EXIT at the least).
            if !self.builder.fit.contains(entry)
                || excluded.contains(entry)
                || self.blocks_under_const_branch.contains(entry)
                || !self.flow.can_return(entry)
            {
                continue;
            }
            let bound = self.size_bound[entry];
            if bound < self.min_statements || best.as_ref().is_some_and(|b| bound <= b.written) {
                continue;
            }
            let passed = self
                .builder
                .collect(self.body, entry, excluded, self.min_statements);
            // Sizes never decrease along the chain, so the candidates larger
            // than the best so far are a tail of the list.
            let best_written = best.as_ref().map_or(0, |b| b.written);
            let from = passed.partition_point(|c| c.written <= best_written);
            if let Some(found) = self.largest_passing(entry, &passed[from..], &mut io_checks_left) {
                best = Some(found);
            }
        }
        best
    }

    /// Calls `stretch_io` on one entry's `candidates` and returns the largest
    /// that passes, if any. `candidates` are sorted by size ascending and all
    /// larger than the best so far; the largest is tried first, and when it
    /// fails a binary search over the rest finds the largest that passes.
    /// (That assumes failures nest as the candidates do, which they mostly
    /// do; when they do not, the result is still a stretch that passed every
    /// rule.) Each `stretch_io` call decrements `io_checks_left`; at zero it
    /// returns what it has. `in_loop` and `other_stretches` are left for
    /// `best_stretch` to set.
    fn largest_passing(
        &mut self,
        entry: BasicBlock,
        candidates: &[Candidate],
        io_checks_left: &mut usize,
    ) -> Option<Stretch> {
        let mut found = None;
        let (mut low, mut high) = (0, candidates.len());
        let mut pick = high.checked_sub(1)?;
        while low < high && *io_checks_left > 0 {
            *io_checks_left -= 1;
            let candidate = &candidates[pick];
            let blocks = self.builder.blocks(candidate.members);
            let exit = (candidate.exit != self.flow.exit()).then_some(candidate.exit);
            let locals = self
                .locals
                .get_or_insert_with(|| LocalFacts::new(self.body, &self.per_copy_consts));
            // The drop flags, the per-copy constants, the signature types,
            // and the two address checks.
            match stretch_io(self.tcx, self.body, locals, &blocks, entry, exit) {
                Some(io) => {
                    found = Some(Stretch {
                        entry,
                        exit,
                        blocks,
                        size: candidate.size,
                        written: candidate.written,
                        params: io.params,
                        returns: io.returns,
                        in_loop: false,
                        other_stretches: 0,
                    });
                    low = pick + 1;
                }
                None => high = pick,
            }
            pick = low + (high - low) / 2;
        }
        found
    }
}

/// The largest set of whole blocks in `body` that passes every check in the
/// note above with at least `min_statements` hand-written counted items, or
/// `None`. `facts` is `classify`'s per-block result for the same body.
pub(crate) fn best_stretch<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    facts: &IndexSlice<BasicBlock, BlockFacts>,
    min_statements: usize,
) -> Option<Stretch> {
    let n = body.basic_blocks.len();
    // The cheapest check first: not enough hand-written items among the
    // blocks with nothing dependent, however they are arranged. A body
    // whose only such blocks are one logging macro's expansion stops here,
    // before a flow graph is built for it.
    let total: usize = facts
        .iter()
        .filter(|f| f.dependent == 0)
        .map(|f| f.hand_written as usize)
        .sum();
    if total < min_statements {
        return None;
    }
    let flow = FlowGraph::new(body);
    let fit = fit_blocks(body, facts, &flow);
    let fit_total: usize = fit.iter().map(|b| facts[b].hand_written as usize).sum();
    if fit_total < min_statements {
        return None;
    }
    let (consts, branch_blocks) = per_copy_consts(tcx, body, &flow);
    let mut search = Search {
        tcx,
        body,
        flow: &flow,
        builder: StretchBuilder::new(&flow, facts, &fit, n),
        size_bound: stretch_size_bounds(facts, body, &fit, n),
        per_copy_consts: consts,
        blocks_under_const_branch: branch_blocks,
        locals: None,
        min_statements,
    };
    let mut excluded = DenseBitSet::new_empty(n);
    let mut stretch = search.round(&excluded)?;
    excluded.union(&stretch.blocks);
    // Further stretches: exclude the best and search again. `size_bound` was
    // summed over blocks now excluded, so it is still a valid upper bound.
    while stretch.other_stretches < OTHER_STRETCHES_CAP
        && let Some(other) = search.round(&excluded)
    {
        stretch.other_stretches += 1;
        excluded.union(&other.blocks);
    }
    // An exit block with a path back to the entry puts the replacing call
    // inside a loop of the outer fn. No path leads back from a `return`.
    stretch.in_loop = stretch
        .exit
        .is_some_and(|exit| reaching(body, stretch.entry).contains(exit));
    Some(stretch)
}

// ── counting instantiations ─────────────────────────────────────────────────

/// A use of a generic fn whose arguments still mention the enclosing fn's
/// generic parameters, so it differs for each concrete argument list of
/// `caller`. A drop of a value of type `T` is the use `drop_glue::<T>`, the
/// fn a MIR `Drop` terminator compiles to.
struct Propagating<'tcx> {
    /// The fn whose parameters those are: for a use in a closure body, the
    /// fn around it (the closure's typeck root).
    caller: LocalDefId,
    /// The fn called or named, with the generic arguments the type checker
    /// inferred. It may still be a trait item rather than the impl's method.
    callee: DefId,
    args: GenericArgsRef<'tcx>,
    site: Span,
}

/// A local generic fn and one concrete argument list its body is compiled
/// for: normalized, regions erased, no parameters left.
type Instantiation<'tcx> = (LocalDefId, GenericArgsRef<'tcx>);

/// Counts, for each local generic fn, the distinct concrete argument lists
/// this crate uses it with.
#[derive(Default)]
struct InstantiationCounts<'tcx> {
    /// Each fn's argument lists, both levels in source order so that
    /// `propagate` and the note do not depend on hash order.
    concrete: FxIndexMap<LocalDefId, FxIndexSet<GenericArgsRef<'tcx>>>,
    /// The use the note cites for each fn: the one with the lowest span,
    /// and on a tie the first recorded.
    first_site: FxHashMap<LocalDefId, (Span, GenericArgsRef<'tcx>)>,
    propagating: Vec<Propagating<'tcx>>,
    /// Instantiations `propagate` has not expanded yet.
    queue: VecDeque<Instantiation<'tcx>>,
    /// The `Drop::drop` method; `None` only in a crate without `core`.
    drop_fn: Option<DefId>,
}

impl<'tcx> InstantiationCounts<'tcx> {
    /// Adds a use of `callee` found in `caller`'s body. One that still
    /// mentions `caller`'s parameters waits for `propagate`; others are
    /// recorded now.
    fn add_use(
        &mut self,
        tcx: TyCtxt<'tcx>,
        caller: LocalDefId,
        callee: DefId,
        args: GenericArgsRef<'tcx>,
        site: Span,
    ) {
        if args.has_non_region_param() {
            self.propagating.push(Propagating {
                caller,
                callee,
                args,
                site,
            });
        } else {
            self.record(tcx, callee, args, site);
        }
    }

    /// Records a use with concrete arguments against each local generic fn
    /// it instantiates, and queues each argument list not seen before.
    /// `drop_glue::<T>` is the code rustc generates to drop a `T`, which
    /// calls `Drop::drop` on each part of `T` with a `Drop` impl.
    fn record(&mut self, tcx: TyCtxt<'tcx>, callee: DefId, args: GenericArgsRef<'tcx>, site: Span) {
        if !tcx.is_lang_item(callee, LangItem::DropGlue) {
            self.count(tcx, callee, args, site);
        } else if let Some(drop_fn) = self.drop_fn {
            for part in glue_drops(tcx, args.type_at(0)) {
                self.count(tcx, drop_fn, tcx.mk_args(&[part.into()]), site);
            }
        }
    }

    /// Counts one use of `callee` with concrete `args` against the local
    /// generic fn compiled for it, if any (`resolve`), and queues each new
    /// argument list for `propagate`.
    fn count(&mut self, tcx: TyCtxt<'tcx>, callee: DefId, args: GenericArgsRef<'tcx>, site: Span) {
        let Some((item, args)) = resolve(tcx, callee, args) else {
            return;
        };
        let lists = self.concrete.entry(item).or_default();
        if lists.insert(args) && lists.len() <= MAX_EXPANDED {
            self.queue.push_back((item, args));
        }
        let first = self.first_site.entry(item).or_insert((site, args));
        if site.lo() < first.0.lo() {
            *first = (site, args);
        }
    }
}

/// The local generic fn whose body is compiled for a use of `callee` with
/// concrete `args`, and the arguments that body sees. A trait method call
/// resolves to the impl's method when the impl is local; `dyn` dispatch,
/// closure call shims and foreign fns give `None`.
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
    // `propagate` substitutes these args for `item`'s own generic parameters,
    // so there must be exactly one per parameter.
    (args.len() == tcx.generics_of(item).count() && !args.has_non_region_param())
        .then_some((item, args))
}

/// The `Drop::drop` method.
fn drop_fn(tcx: TyCtxt<'_>) -> Option<DefId> {
    let drop_trait = tcx.lang_items().drop_trait()?;
    tcx.associated_items(drop_trait)
        .in_definition_order()
        .find(|item| item.is_fn())
        .map(|item| item.def_id)
}

/// The types inside a value of concrete type `ty` whose `Drop` impl is in
/// this crate and runs when the value is dropped: `ty` itself, its fields,
/// elements, `Box` contents, closure captures and values a coroutine holds
/// across a suspension point, transitively. Excluded: anything behind a
/// reference or raw pointer, inside `ManuallyDrop` or a `union`, or behind
/// `dyn` (its type is unknown here), and what a foreign `Drop` impl drops
/// by hand (`Vec`'s elements), since that crate's code is not read.
fn glue_drops<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Vec<Ty<'tcx>> {
    let env = ty::TypingEnv::fully_monomorphized();
    let mut local = Vec::new();
    // Normalized so that an `impl Trait` is walked as its hidden type.
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
                    // The box's fields are just a pointer; the contents it
                    // drops are its type argument.
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

/// Calls `each` for every fn `e` uses, with the generic arguments the type
/// checker inferred and the span to cite. A path's `FnDef` type carries
/// its arguments even where the type checker recorded none on the node (a
/// `for` loop's desugared `into_iter` and `next`); a method call,
/// overloaded operator, index or `*x` records them on the node. Each
/// overloaded auto-deref on `e` adds a `Deref::deref` or
/// `DerefMut::deref_mut` call with no expression.
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
    // Skip constructors, which are `FnDef`s too, and any argument list
    // without exactly one entry per generic parameter: `resolve` and
    // `propagate` cannot substitute it.
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

/// The span to cite: the whole call when `e` is its callee, else `e`.
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

/// Collects every use of a generic fn in the crate, then runs `propagate`.
fn count_instantiations<'tcx>(tcx: TyCtxt<'tcx>) -> InstantiationCounts<'tcx> {
    let mut counts = InstantiationCounts {
        drop_fn: drop_fn(tcx),
        ..InstantiationCounts::default()
    };
    for owner in tcx.hir_body_owners() {
        if !tcx.has_typeck_results(owner) {
            continue;
        }
        // Skip `const` and `static` initializers, array lengths, enum
        // discriminants and `const {}` blocks: they are evaluated at compile
        // time, so what they call is not compiled into the binary. A
        // `const fn` body stays, since it may also be called at run time.
        if let Some(ConstContext::Const { .. } | ConstContext::Static(_)) =
            tcx.hir_body_const_context(owner)
        {
            continue;
        }
        // A closure has its own body but shares its enclosing fn's typeck
        // results and generic parameters.
        let caller = tcx.typeck_root_def_id_local(owner);
        let typeck = tcx.typeck(owner);
        let body = tcx.hir_body_owned_by(owner);
        for_each_expr_without_closures(body.value, |e| {
            fn_uses(tcx, typeck, e, |callee, args, site| {
                counts.add_use(tcx, caller, callee, args, site);
            });
            ControlFlow::<()>::Continue(())
        });
        // Drops have no HIR expression, so read them from MIR: one `Drop`
        // terminator per value dropped, unwind cleanup included, since that
        // code is compiled too. Each is a use of `drop_glue::<T>`. The
        // span cited is the local's declaration.
        if let Some(mir) = mir_for(tcx, owner)
            && let Some(drop_glue) = tcx.lang_items().drop_glue_fn()
        {
            for data in mir.basic_blocks.iter() {
                if let Some(terminator) = &data.terminator
                    && let TerminatorKind::Drop { place, .. } = terminator.kind
                {
                    let dropped = place.ty(&mir.local_decls, tcx).ty;
                    let site = mir.local_decls[place.local].source_info.span;
                    counts.add_use(tcx, caller, drop_glue, tcx.mk_args(&[dropped.into()]), site);
                }
            }
        }
    }
    propagate(tcx, &mut counts);
    counts
}

/// The most argument lists of one fn that `propagate` expands; lists past it
/// are still counted. No crate that builds is expected to come near it. It
/// exists for a fn that calls itself through two wrapper types (`f::<T>`
/// calling both `f::<A<T>>` and `f::<B<T>>`): that program does not build,
/// but the lint still runs on it, and the number of lists doubles with each
/// level of nesting, so the depth test in `propagate` alone would allow
/// 2^128 of them.
const MAX_EXPANDED: usize = 1 << 16;

/// Finds the instantiations made inside generic bodies. Each queued
/// (fn, concrete args) pair is taken once: the args are substituted into
/// every use in that fn's body that mentioned its parameters, the result is
/// recorded, and any argument list not seen before is queued in turn. The
/// result does not depend on the order bodies were visited in.
///
/// A fn that calls itself with a deeper type each time (`f::<T>` calling
/// `f::<W<T>>`) never stops producing new lists. rustc refuses to build such
/// a program once instantiation passes the recursion limit, and a derived
/// use is dropped here when its types nest deeper than that limit. The test
/// is on the type, not on how many steps produced it, so nothing a program
/// that builds instantiates is dropped.
fn propagate<'tcx>(tcx: TyCtxt<'tcx>, counts: &mut InstantiationCounts<'tcx>) {
    let limit = tcx.recursion_limit().0;
    let propagating = std::mem::take(&mut counts.propagating);
    let mut uses: FxHashMap<LocalDefId, Vec<&Propagating<'tcx>>> = FxHashMap::default();
    for edge in &propagating {
        uses.entry(edge.caller).or_default().push(edge);
    }
    let env = ty::TypingEnv::fully_monomorphized();
    let mut nesting = Nesting::default();
    while let Some((caller, caller_args)) = counts.queue.pop_front() {
        let Some(edges) = uses.get(&caller) else {
            continue;
        };
        for edge in edges {
            // The edge's args with `caller`'s parameters replaced by `caller_args`.
            let Ok(args) = tcx.try_instantiate_and_normalize_erasing_regions(
                caller_args,
                env,
                ty::EarlyBinder::bind(edge.args),
            ) else {
                continue;
            };
            if args.has_non_region_param() || nesting.of(args) > limit {
                continue;
            }
            counts.record(tcx, edge.callee, args, edge.site);
        }
    }
}

/// Measures how deeply types nest: `u8` is 1, `Vec<u8>` 2, `&[Vec<u8>]` 4.
/// Each distinct type is measured once and remembered, because `f::<T>`
/// calling `f::<(T, T)>` doubles the written type at every step while adding
/// only one distinct type to it.
#[derive(Default)]
struct Nesting<'tcx> {
    /// The deepest nesting seen among the types visited so far, at the level
    /// `visit_ty` is currently inside.
    depth: usize,
    measured: FxHashMap<Ty<'tcx>, usize>,
}

impl<'tcx> Nesting<'tcx> {
    fn of(&mut self, args: GenericArgsRef<'tcx>) -> usize {
        self.depth = 0;
        args.visit_with(self);
        self.depth
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Nesting<'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        let depth = match self.measured.get(&ty) {
            Some(&depth) => depth,
            None => {
                let outer = std::mem::take(&mut self.depth);
                ensure_sufficient_stack(|| ty.super_visit_with(self));
                let depth = std::mem::replace(&mut self.depth, outer) + 1;
                self.measured.insert(ty, depth);
                depth
            }
        };
        self.depth = self.depth.max(depth);
    }
}

// ── the extra help line ──────────────────────────────────────────────────────
//
// A reported fn gets a second help line when all it does with its generic
// arguments is convert each into a value of a concrete type (`bytes.as_ref()`,
// `x.foo()`) and, if taken by value, drop it: such a fn could take the
// converted values and not be generic. Every counted statement or terminator
// that mentions a parameter must be (a) a trait-method call whose one operand
// is the argument or a borrow of it, returning into a local whose type has no
// parameter and is not `()` or `!`; (b) a borrow or move of the argument into
// a temp that (a) or another (b) uses; or (c) a drop of the argument or of
// such a temp. (a) and (b) must be in `entry_run`, the blocks every call
// executes exactly once, so a caller doing the conversion itself calls the
// method as often as the fn did. Only arguments whose type, references
// removed, is a bare parameter count; there must be one, and each must be
// converted at least once. Trait methods, provided or in an impl, cannot
// change their signature and never get the line. Anything else withholds it:
// a case not matched only omits the line; a case matched wrongly would print
// a false claim.

/// One conversion of a generic argument: `bytes.as_ref()` giving a `&[u8]`.
struct Conversion<'tcx> {
    arg: mir::Local,
    /// `Trait::method`, unqualified: `AsRef::as_ref`, `HasFoo::foo`.
    through: String,
    /// The result type; no parameter appears in it.
    obtains: Ty<'tcx>,
}

/// The blocks every call of the fn executes exactly once, in order: from the
/// entry block, following terminators with one normal successor while the
/// next block has exactly one predecessor. The first `SwitchInt`, loop head,
/// diverging call or return ends it.
fn entry_run(body: &mir::Body<'_>) -> Vec<BasicBlock> {
    let predecessors = body.basic_blocks.predecessors();
    let mut run = Vec::new();
    // Built MIR never jumps back to its entry block (a `loop` at the top of a
    // fn gets a head block of its own).
    if !predecessors[mir::START_BLOCK].is_empty() {
        return run;
    }
    let mut block = mir::START_BLOCK;
    // Single-predecessor blocks reached from a block with none cannot form a
    // cycle; the length bound only makes that visible.
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

/// The conversions `body` makes of its generic arguments when the conditions
/// in the note above all hold; empty otherwise. One pass over the body with
/// the classifier's test, made only for a fn about to be reported.
fn only_conversions<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> Vec<Conversion<'tcx>> {
    // A trait's method, provided or implemented, has the trait's signature.
    let def = body.source.def_id();
    if tcx.trait_of_assoc(def).is_some() || tcx.trait_impl_of_assoc(def).is_some() {
        return Vec::new();
    }
    let decls = &body.local_decls;
    // An argument whose type, references removed, is a bare parameter.
    let bare_param_arg = |local: mir::Local| {
        body.local_kind(local) == mir::LocalKind::Arg
            && matches!(decls[local].ty.peel_refs().kind(), ty::Param(_))
    };
    if !body.args_iter().any(bare_param_arg) {
        return Vec::new();
    }
    // Temp -> the argument it borrows or was moved from.
    let mut arg_of_temp: FxHashMap<mir::Local, mir::Local> = FxHashMap::default();
    // The argument a place refers to: such an argument or a temp in
    // `arg_of_temp`, as is or dereferenced, with nothing else projected.
    let arg_of_place =
        |temps: &FxHashMap<mir::Local, mir::Local>, place: Place<'tcx>| -> Option<mir::Local> {
            if !place
                .projection
                .iter()
                .all(|elem| matches!(elem, mir::ProjectionElem::Deref))
            {
                return None;
            }
            if bare_param_arg(place.local) {
                Some(place.local)
            } else {
                temps.get(&place.local).copied()
            }
        };
    let run = entry_run(body);
    let mut in_entry_run: IndexVec<BasicBlock, bool> =
        IndexVec::from_elem_n(false, body.basic_blocks.len());
    for &block in &run {
        in_entry_run[block] = true;
    }
    // The entry run first, in execution order, so each temp is recorded
    // before its use; then the other blocks, where only drops may mention a
    // parameter.
    let rest = body
        .basic_blocks
        .indices()
        .filter(|&block| !in_entry_run[block] && !body.basic_blocks[block].is_cleanup);
    let order: Vec<BasicBlock> = run.iter().copied().chain(rest).collect();
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let mut found: Vec<Conversion<'tcx>> = Vec::new();
    for block in order {
        let data = &body.basic_blocks[block];
        let early = in_entry_run[block];
        for statement in &data.statements {
            if !counted_statement(statement) || !places.dependent_statement(statement) {
                continue;
            }
            // (b) `_k = &_1`, `_k = &(*_j)`, `_k = move _1`, into a temp (not
            // the return place, not an argument overwritten).
            if early
                && let Some((target, rvalue)) = statement.kind.as_assign()
                && let Some(temp) = target.as_local()
                && body.local_kind(temp) == mir::LocalKind::Temp
                && let mir::Rvalue::Ref(_, mir::BorrowKind::Shared, place)
                | mir::Rvalue::CopyForDeref(place)
                | mir::Rvalue::Use(mir::Operand::Move(place) | mir::Operand::Copy(place), _) =
                    rvalue
                && let Some(arg) = arg_of_place(&arg_of_temp, *place)
            {
                arg_of_temp.insert(temp, arg);
            } else {
                return Vec::new();
            }
        }
        let Some(terminator) = &data.terminator else {
            continue;
        };
        if !counted_terminator(terminator) || !places.dependent_terminator(terminator) {
            continue;
        }
        match &terminator.kind {
            // (c) a drop of the argument or of a temp it was moved into,
            // anywhere in the body.
            TerminatorKind::Drop { place, .. }
                if place
                    .as_local()
                    .is_some_and(|l| bare_param_arg(l) || arg_of_temp.contains_key(&l)) => {}
            // (a) the conversion.
            TerminatorKind::Call {
                func,
                args,
                destination,
                target: Some(_),
                ..
            } if early => {
                if let [operand] = &args[..]
                    && let Some(place) = operand.node.place()
                    && let Some(arg) = arg_of_place(&arg_of_temp, place)
                    && let Some((callee, _)) = func.const_fn_def()
                    && let Some(trait_id) = tcx.trait_of_assoc(callee)
                    && let Some(obtained) = destination.as_local()
                    && !decls[obtained].ty.has_non_region_param()
                    && !decls[obtained].ty.is_unit()
                    && !decls[obtained].ty.is_never()
                {
                    found.push(Conversion {
                        arg,
                        // `item_name` reads a symbol, not the printer, so it
                        // is safe to call before it is known whether the
                        // finding is reported.
                        through: format!("{}::{}", tcx.item_name(trait_id), tcx.item_name(callee)),
                        obtains: decls[obtained].ty,
                    });
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Every bare-parameter argument converted at least once.
    if body
        .args_iter()
        .filter(|&arg| bare_param_arg(arg))
        .any(|arg| !found.iter().any(|c| c.arg == arg))
    {
        return Vec::new();
    }
    found
}

/// The argument's source name in backticks (`bytes`, `self`), or "its `<type>`
/// argument" for a pattern argument that has no name.
fn arg_name(body: &mir::Body<'_>, arg: mir::Local) -> String {
    match local_name(body, arg) {
        Some(name) => format!("`{name}`"),
        None => with_no_trimmed_paths!(format!("its `{}` argument", body.local_decls[arg].ty)),
    }
}

/// The extra help line without its subject: "only uses `text` to obtain a
/// `&str` through `AsRef::as_ref`; it could take `&str` instead". `report`
/// prepends the fn's name: this runs before the fn is known to be reported,
/// and printing a def path uses rustc's trimmed-path printer, which then
/// requires that a diagnostic be emitted. Conversions are listed in argument
/// order with repeats folded and every type printed untrimmed.
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

// ── where a stretch is in the source ─────────────────────────────────────────
//
// The finding needs one place in the source to point at, but a stretch is a
// set of MIR blocks. Each counted MIR item has the span of the source it was
// lowered from, so the first guess is the smallest span covering all of the
// stretch's items. Three corrections are applied first. To see the spans a
// correction acts on, dump a fixture's MIR with `-Zdump-mir
// -Zmir-include-spans`.
//
// 1. An item lowered from a macro expansion has a span inside the macro's
//    definition. It is replaced by the span of the innermost macro call that
//    is inside the fn body. Desugarings (`for`, `?`) are left alone: their
//    spans are already in place, and their call site is the whole `for` or
//    `?` expression. An item with no span inside the body is ignored.
//
// 2. Some items span a whole construct rather than one statement: the `()` a
//    block, loop or `else`-less `if` evaluates to; a unit fn's `_0 = ()`,
//    which spans the whole body; the `x = a + f(b)` that reads a call's
//    result, which spans the call in the previous block. Such a span can
//    cover code outside the stretch, so an item is dropped when its span
//    contains the span of any counted item outside the stretch. One
//    exception: a span equal to that of a parameter-free outside item. That
//    is one source expression compiled to items on both sides of the stretch
//    boundary (a `for` head is both the `into_iter` before the loop and the
//    `next` inside it) and does not say which side the source is on. A span
//    equal to a parameter-dependent outside item's is not excepted: that
//    expression is compiled partly to parameter-dependent code, so it cannot
//    all be inside the stretch, and the stretch's items with that span are
//    dropped.
//
// 3. Source order is not control-flow order. A `for` pattern is written
//    before the loop head but bound inside the loop body, so a stretch that is
//    the body of a loop over a generic iterator has the dependent `next` call
//    between its first item and the rest. The span shown must never cover
//    code that depends on a parameter. So the remaining items are split into
//    segments at the start of every dependent outside item, each item goes to
//    the segment its start is in (an item starting exactly at a split point
//    goes to the segment before), and the segment with the most items is
//    reported. No dependent item lies inside it: an item that starts at or
//    before a split point and ends at or after the dependent item's end
//    contains it, and was dropped in step 2.
//    Parameter-free outside items may lie inside it; they are the same code
//    at every instantiation.
//
// If the best segment holds fewer than half of the stretch's items that have a
// span, one underline would misrepresent the stretch. The site is then the
// stretch's first remaining item, and the note says "the stretch, N statements
// starting here, ...".
//
// The site is a `span_note` on the finding, not a label on its primary span:
// the note's sentence is long and reads better under its own `note:` heading
// than after a multi-line underline.

/// The source span a stretch is pointed at with. It is a secondary span: the
/// finding's own span stays the fn's signature, which the baseline file
/// matches findings by and which an `#[allow]` goes on.
#[derive(Clone, Copy, Debug)]
pub(crate) enum StretchSite {
    /// From the stretch's first item to its last within one segment that
    /// holds at least half the stretch's items and no parameter-dependent
    /// item from outside the stretch.
    Whole(Span),
    /// No such segment: the stretch's first item by position, for "the
    /// stretch, N statements starting here".
    Start(Span),
}

/// Moves `span`, one MIR item's, to its innermost macro call site inside
/// `body_span` (a desugaring stays in place) and into `body_span`'s context,
/// so results compare by position. `None` if no call site is in the body.
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

/// The source site of the stretch made of `blocks` of `def`'s `body`, by the
/// rules above; `None` when no counted item of it has a span in the body.
pub(crate) fn stretch_site<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
) -> Option<StretchSite> {
    let body_span = tcx.hir_body_owned_by(def).value.span;
    // Spans of the stretch's items, and (span, parameter-dependent) of the rest.
    let mut inside: Vec<Span> = Vec::new();
    let mut outside: Vec<(Span, bool)> = Vec::new();
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    for (block, data) in body.basic_blocks.iter_enumerated() {
        // A cleanup block was recorded empty; it yields nothing here either.
        if data.is_cleanup {
            continue;
        }
        let in_stretch = blocks.contains(block);
        for statement in &data.statements {
            if counted_statement(statement)
                && let Some(span) = body_position(statement.source_info.span, body_span)
            {
                if in_stretch {
                    inside.push(span);
                } else {
                    outside.push((span, places.dependent_statement(statement)));
                }
            }
        }
        if let Some(terminator) = &data.terminator
            && counted_terminator(terminator)
            && let Some(span) = body_position(terminator.source_info.span, body_span)
        {
            if in_stretch {
                inside.push(span);
            } else {
                outside.push((span, places.dependent_terminator(terminator)));
            }
        }
    }
    let placed = inside.len();
    let by_position = |s: &Span| (s.lo(), std::cmp::Reverse(s.hi()));
    let first = *inside.iter().min_by_key(|s| by_position(s))?;

    // Step 2's exception: remove parameter-free outside items whose span
    // equals a stretch item's. Dependent ones stay, for steps 2 and 3.
    let shared: FxHashSet<(BytePos, BytePos)> = inside.iter().map(|s| (s.lo(), s.hi())).collect();
    outside.retain(|(s, dependent)| *dependent || !shared.contains(&(s.lo(), s.hi())));

    // Step 2: drop items whose span contains an outside item's span. With
    // `outside` sorted by start and the least end from each index on
    // precomputed, that is one binary search per item.
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
        return Some(StretchSite::Start(first));
    };

    // Step 3: split at every dependent outside item's start; an item exactly
    // at a split point goes to the segment before. One pass in position
    // order builds segments as (item count, start, end); the segment with
    // the most items is chosen, the earlier one on a tie.
    let split_points: Vec<BytePos> = outside
        .iter()
        .filter(|(_, dependent)| *dependent)
        .map(|(s, _)| s.lo())
        .collect();
    let segment_of = |item: &Span| split_points.partition_point(|&at| at < item.lo());
    let mut segments: Vec<(usize, BytePos, BytePos)> = Vec::new();
    let mut open = None;
    for item in &inside {
        let segment = segment_of(item);
        match segments.last_mut() {
            Some((held, _, hi)) if open == Some(segment) => {
                *held += 1;
                *hi = (*hi).max(item.hi());
            }
            _ => {
                segments.push((1, item.lo(), item.hi()));
                open = Some(segment);
            }
        }
    }
    let (held, lo, hi) =
        segments
            .into_iter()
            .fold((0, first_kept.lo(), first_kept.hi()), |best, segment| {
                if segment.0 > best.0 { segment } else { best }
            });
    debug_assert!(
        !outside
            .iter()
            .any(|(s, dependent)| *dependent && lo <= s.lo() && s.hi() <= hi),
        "a dependent item inside the stretch's span"
    );
    Some(if held * 2 >= placed {
        StretchSite::Whole(body_span.with_lo(lo).with_hi(hi))
    } else {
        StretchSite::Start(first_kept)
    })
}

// ── the report ───────────────────────────────────────────────────────────────

/// The type and const parameter names of `def`, `impl` ones first, each in backticks.
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

/// The named variable or field a temporary is assigned to, whole.
enum AssignedTo {
    /// `let n = <it>`, or `let text: &[u8] = &*<it>` at the same type.
    Let(Symbol),
    /// `Settings { height: <it>, .. }` or `self.col = <it>`.
    Field(Symbol),
}

/// The name of the last field in `place` (`height` for `_3.height`); `None`
/// for tuple and closure fields, which have only positions.
fn field_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    place: Place<'tcx>,
) -> Option<Symbol> {
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
    Some(variant.fields.get(field)?.name)
}

/// Finds the one statement in the blocks `among` that reads `local`, and
/// returns the named variable or field it assigns the whole value to. `None`
/// when the local is read twice, read in part (`_5.0`), or read by a call.
fn assigned_to<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    among: &DenseBitSet<BasicBlock>,
    local: Local,
) -> Option<AssignedTo> {
    let mut of = DenseBitSet::new_empty(body.local_decls.len());
    of.insert(local);
    let mut user: Option<&Statement<'tcx>> = None;
    for block in among.iter() {
        let data = &body.basic_blocks[block];
        if data.is_cleanup {
            continue;
        }
        for statement in &data.statements {
            let found = reads_any(&of, |uses| uses.visit_statement(statement, Location::START));
            // The statement that assigns the local is not a read of it.
            let writes = match &statement.kind {
                StatementKind::Assign(assign) => assign.0.as_local() == Some(local),
                _ => false,
            };
            if found && !writes {
                if user.is_some() {
                    return None;
                }
                user = Some(statement);
            }
        }
        if let Some(terminator) = &data.terminator {
            let found = reads_any(&of, |uses| {
                uses.visit_terminator(terminator, Location::START)
            });
            let writes = matches!(
                &terminator.kind,
                TerminatorKind::Call { destination, .. } if destination.local == local
            );
            if found && !writes {
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
            Some(var) => local_name(body, var).map(AssignedTo::Let),
            None => field_name(tcx, body, *dest).map(AssignedTo::Field),
        },
        // `let text: &[u8] = &buf[a..b]`: a reborrow of `*local` at the same type.
        Rvalue::Ref(_, _, place)
            if place.local == local
                && matches!(place.projection[..], [mir::ProjectionElem::Deref]) =>
        {
            let var = dest.as_local()?;
            (body.local_decls[var].ty == body.local_decls[local].ty)
                .then(|| local_name(body, var).map(AssignedTo::Let))?
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
            Some(AssignedTo::Field(field.name))
        }
        _ => None,
    }
}

/// One local as the finding names it: `` `acc: u32` ``, "the returned `u32`",
/// "the `u32` assigned to `total`", "the `u32` for field `height`", "the
/// `usize` from `src.len()`" (a short expression in this body), else "a
/// `usize` temporary". Types go through `with_no_trimmed_paths!` because the
/// trimmed printer requires that a diagnostic follow, and this one may still
/// be filtered out by the baseline.
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
    match assigned_to(cx.tcx, body, among, local) {
        Some(AssignedTo::Let(name)) => return format!("the `{ty}` assigned to `{name}`"),
        Some(AssignedTo::Field(name)) => return format!("the `{ty}` for field `{name}`"),
        None => {}
    }
    // A temporary's span is its expression. An argument's or the return
    // slot's span is outside the body block, so `contains` excludes them.
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

/// Describes each local in `locals` for the finding, in order. A `()` local
/// is skipped: it carries no value, yet MIR's `return` still reads the
/// return slot in a fn returning unit, so a stretch reaching that `return`
/// would otherwise list a `()`. Two locals that describe alike (a shadowed
/// `let c`, both live) are listed once.
pub(crate) fn render_locals<'tcx>(
    cx: &LateContext<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    locals: &[Local],
    among: &DenseBitSet<BasicBlock>,
) -> Vec<String> {
    // The body block's span, so argument patterns and the return type are excluded.
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

/// The main message: "`f` is generic over `T` and compiled 3 times in this
/// crate, ..., but 12 of the 20 counted statements in its body ... are one
/// stretch that is the same in every copy: ...".
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

/// The note text after "this stretch": "uses only `acc: u32`, `src: &[u8]`
/// and yields the returned `u32`; it is inside a loop, so the call that
/// replaces it runs once per iteration".
fn stretch_clauses(stretch: &Searched, min_statements: usize) -> String {
    let uses = match &stretch.uses[..] {
        [] => "uses nothing computed before it".to_owned(),
        list => format!("uses only {}", list.join(", ")),
    };
    let yields = match &stretch.yields[..] {
        [] => "yields nothing read after it".to_owned(),
        list => format!("yields {}", list.join(", ")),
    };
    let mut rest = format!("{uses} and {yields}");
    if stretch.in_loop {
        rest.push_str(
            "; it is inside a loop, so the call that replaces it runs once per iteration",
        );
    }
    match stretch.other_stretches {
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

/// The search result for one fn, and what the diagnostic prints about its
/// reported stretch. Holds rendered strings only, not the MIR body, so the
/// body's borrow ends before `count_instantiations` reads it again.
struct Searched {
    def: LocalDefId,
    /// Where the stretch is in the source; `None` when no item in it has a
    /// span there.
    site: Option<StretchSite>,
    /// Statements and terminators inside the stretch.
    size: usize,
    /// Values it reads that earlier code computed (the new fn's parameters).
    uses: Vec<String>,
    /// Values it computes that later code reads (the new fn's results).
    yields: Vec<String>,
    /// It is inside a loop, so the replacement call runs once per iteration.
    in_loop: bool,
    /// Other disjoint stretches in the same body that also qualify.
    other_stretches: usize,
    /// Counted statements and terminators in the whole body.
    total: usize,
    /// `conversion_help`'s line, less the fn's name.
    signature_help: Option<String>,
}

/// Emits the diagnostic for one fn at the fn's own span, with notes for the
/// stretch, one instantiation, and any `#[inline]` caveat. `emit_hir_then`
/// with the fn's HirId so an `#[allow]` on the fn is honoured; the current
/// node during `check_crate_post` is the crate root.
fn report<'tcx>(
    cx: &LateContext<'tcx>,
    stretch: &Searched,
    sets: usize,
    site: Option<(Span, GenericArgsRef<'tcx>)>,
    min_statements: usize,
) {
    let tcx = cx.tcx;
    let def = stretch.def;
    let msg = finding_message(tcx, def, sets, stretch.size, stretch.total);
    let site = site.map(|(site, args)| {
        (
            site.source_callsite(),
            format!(
                "one of the {sets} instantiations, with {}",
                render_args(tcx, def, args)
            ),
        )
    });
    let rest = stretch_clauses(stretch, min_statements);
    let inline = inline_note(tcx, def);
    // The help line: the fix and its cost.
    let help = format!(
        "move the stretch into a non-generic fn that takes the values it uses and returns what \
         it yields, and call that fn in its place; `{}` keeps its signature and the cost is one \
         call",
        tcx.def_path_str(def)
    );
    let signature_help = stretch
        .signature_help
        .as_ref()
        .map(|line| format!("`{}` {line}", tcx.def_path_str(def)));
    emit_hir_then(
        cx,
        GENERIC_BODY_NOT_GENERIC,
        tcx.local_def_id_to_hir_id(def),
        tcx.def_span(def),
        msg,
        |diag| {
            // The note goes at the stretch's source span if it has one;
            // `rest` finishes the sentence ("uses only `src: &[u8]` and
            // yields the returned `u32`").
            match stretch.site {
                Some(StretchSite::Whole(span)) => {
                    diag.span_note(span, format!("this stretch {rest}"));
                }
                Some(StretchSite::Start(span)) => {
                    let size = stretch.size;
                    diag.span_note(
                        span,
                        format!("the stretch, {size} statements starting here, {rest}"),
                    );
                }
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
            if let Some(signature_help) = signature_help {
                diag.help(signature_help);
            }
        },
    );
}

impl<'tcx> LateLintPass<'tcx> for GenericBodyNotGeneric {
    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let tcx = cx.tcx;
        // Search the candidates first, cheapest check first, and count
        // instantiations (which reads every body in the crate) only if one of
        // them has something to report.
        let mut searched: Vec<Searched> = tcx
            .hir_body_owners()
            .filter(|&def| candidate_fn(tcx, def))
            .filter_map(|def| {
                let body = mir_for(tcx, def)?;
                let (facts, total) = classify(tcx, &body);
                let stretch = best_stretch(tcx, &body, &facts, self.min_statements)?;
                // `uses` are read inside the stretch; `yields` in the blocks after it.
                let mut past = DenseBitSet::new_filled(body.basic_blocks.len());
                past.subtract(&stretch.blocks);
                Some(Searched {
                    def,
                    site: stretch_site(tcx, def, &body, &stretch.blocks),
                    size: stretch.size,
                    uses: render_locals(cx, def, &body, &stretch.params, &stretch.blocks),
                    yields: render_locals(cx, def, &body, &stretch.returns, &past),
                    in_loop: stretch.in_loop,
                    other_stretches: stretch.other_stretches,
                    total,
                    signature_help: conversion_help(&body, &only_conversions(tcx, &body)),
                })
            })
            .collect();
        if searched.is_empty() {
            return;
        }
        let counts = count_instantiations(tcx);
        // Report in source order.
        searched.sort_by_key(|s| tcx.def_span(s.def).lo());
        for searched in &searched {
            let sets = counts.concrete.get(&searched.def).map_or(0, |s| s.len());
            if sets >= self.min_instantiations {
                let site = counts.first_site.get(&searched.def).copied();
                report(cx, searched, sets, site, self.min_statements);
            }
        }
    }
}
