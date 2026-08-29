//! `generic_body_not_generic`: a generic function whose body is mostly the
//! same whatever its type and const parameters are, and which this crate
//! stamps out at several concrete argument sets anyway.
//!
//! **Why it is opt-in.** The two numbers a finding states are read straight
//! off the code and are exact: how many of the body's MIR statements do not
//! mention a parameter or a value whose type does, and how many distinct
//! concrete argument sets this crate instantiates the function with. What
//! the finding is *for* -- bytes duplicated in the binary -- is not something
//! source can prove: it depends on inlining, the opt level, LTO and
//! identical-code folding, on instantiations made in other crates, and on
//! whether each copy survives as a symbol at all. And the remedy, a thin
//! generic shim over a non-generic inner function, is a judgment per
//! function. That is the profile of the pack's other surveys (`bool_cluster`,
//! `parallel_params`): off until `generic-body-not-generic-enabled = true`,
//! run once over a size-sensitive crate, read the list.
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
//! are worth measuring; the MIR measure, the census and the report all run
//! at the end with `'tcx` locals.

use std::collections::VecDeque;
use std::ops::ControlFlow;

use clippy_utils::visitors::for_each_expr_without_closures;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, ConstContext, Expr, ExprKind, FnDecl, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    self, Location, Place, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::adjustment::{Adjust, DerefAdjustKind};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, GenericArg, GenericArgKind, GenericArgsRef, GenericParamDefKind, Ty, TyCtxt,
    TypeSuperVisitable, TypeVisitableExt, TypeVisitor, TypeckResults,
};
use rustc_span::Span;

use crate::MordantConfig;
use crate::baseline::{emit, emit_with_note, join};
use crate::mir_flow::mir_for;

rustc_session::declare_lint! {
    /// Flags a generic function (type or const parameters, its own or its
    /// `impl`'s) whose body, read as pre-optimization MIR, is mostly
    /// independent of those parameters -- at least
    /// `generic-body-not-generic-min-statements` statements (default 24) and
    /// `generic-body-not-generic-min-share-percent` percent of the counted
    /// body (default 50) mention no parameter and no value whose type does --
    /// and which this crate instantiates with at least
    /// `generic-body-not-generic-min-instantiations` (default 2) distinct
    /// concrete argument sets. rustc compiles the whole body once per set, so
    /// the independent part is emitted that many times over. The usual fix is
    /// to keep the generic function as a thin shim that converts its
    /// arguments (`path.as_ref()`, `&mut f as &mut dyn FnMut(..)`) and calls a
    /// non-generic inner function holding the body.
    ///
    /// A statement is counted when it does something at run time: storage
    /// markers, plain jumps, returns and unwinding edges are not. It is
    /// dependent when it names a parameter anywhere -- a call to `T::default`,
    /// a `size_of::<T>()`, a cast to `[u8; N]` -- or reads or writes a place
    /// whose type, or the type of a field it goes through, involves one. A
    /// place's type is the type of what it names, not of the variable it
    /// starts from: moving a `T`, dereferencing a `&T`, dropping a `Vec<T>`
    /// and reading `self.inner.len` through an `inner: Wrap<T>` are
    /// dependent, while reading a `[u8; 4]` field of `self` where `Self` is
    /// `Framed<T>`, indexing a `[u8; N]` and arithmetic on a `u32` local are
    /// not. The independent count errs low in one way: a statement that
    /// borrows a constant expression (`&[1, 2, 3]`, the literal pieces of a
    /// `format!`) counts as dependent whatever its value, because rustc files
    /// such constants under the generic function with all its parameters.
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
    /// Runs only with `generic-body-not-generic-enabled = true` in
    /// `dylint.toml`: the counts are exact, but what they cost in the binary
    /// depends on inlining, opt level, LTO and symbol folding, which the
    /// source does not show, and a body that is cheap to duplicate is not
    /// worth the indirection.
    pub GENERIC_BODY_NOT_GENERIC,
    Warn,
    "a generic fn whose body barely depends on its parameters, compiled once per instantiation"
}

pub struct GenericBodyNotGeneric {
    min_statements: usize,
    min_share_percent: usize,
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
            min_share_percent: config.generic_body_not_generic_min_share_percent,
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

// ── the body measure ─────────────────────────────────────────────────────────
//
// The body read is the one `mir_for` serves: the pre-optimization MIR, before
// inlining has copied callees in and before any pass has folded anything
// away, so what is counted is what the fn itself says. Every statement and
// terminator in a block that is not unwind cleanup counts once, except the
// ones that are bookkeeping rather than code. Cleanup blocks are the drops
// run while a panic unwinds; they mirror the normal path's drops and would
// count every one of them twice.
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
// instantiation, which is the part a non-generic inner fn could hold once.
//
// The test errs toward dependent in one known way: a promoted constant (what
// a borrow of a constant expression becomes -- `&[1, 2, 3]`, `&"lit"`, every
// `format_args!` pieces array) is a `Const::Unevaluated` carrying the fn's
// own identity arguments, so the statement that loads it mentions every
// parameter whether or not its value does. That under-counts independence,
// never over-counts it, and is the rule as stated; exempting promoteds would
// mean classifying each constant by hand instead of asking the whole item
// for its flags, which is the one check that cannot miss a type.

/// The counted statements and terminators of one body: how many there are,
/// and how many of those no type or const parameter appears in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BodyMeasure {
    pub(crate) independent: usize,
    pub(crate) total: usize,
}

impl BodyMeasure {
    /// The body is mostly not about its parameters: at least
    /// `min_statements` independent items, and those at least
    /// `min_share_percent` of everything counted.
    pub(crate) fn mostly_independent(
        self,
        min_statements: usize,
        min_share_percent: usize,
    ) -> bool {
        self.independent >= min_statements
            && self.independent * 100 >= self.total * min_share_percent
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

fn measure<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> BodyMeasure {
    let mut measure = BodyMeasure {
        independent: 0,
        total: 0,
    };
    let mut count = |dependent: bool| {
        measure.total += 1;
        if !dependent {
            measure.independent += 1;
        }
    };
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup {
            continue;
        }
        let at = |statement_index| Location {
            block,
            statement_index,
        };
        for (index, statement) in data.statements.iter().enumerate() {
            if !counted_statement(statement) {
                continue;
            }
            places.found = false;
            places.visit_statement(statement, at(index));
            count(statement.has_non_region_param() || places.found);
        }
        if let Some(terminator) = &data.terminator
            && counted_terminator(terminator)
        {
            places.found = false;
            places.visit_terminator(terminator, at(data.statements.len()));
            count(terminator.has_non_region_param() || places.found);
        }
    }
    measure
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

struct Finding<'tcx> {
    def: LocalDefId,
    measure: BodyMeasure,
    sets: usize,
    site: Option<(Span, GenericArgsRef<'tcx>)>,
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
            || !eligible(cx.tcx, def_id)
        {
            return;
        }
        self.candidates.push(def_id);
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let tcx = cx.tcx;
        // Measure first: the census walks every body in the crate, and a
        // crate with no body worth reporting should not pay for it.
        let measured: Vec<(LocalDefId, BodyMeasure)> = std::mem::take(&mut self.candidates)
            .into_iter()
            .filter_map(|def| {
                let m = measure(tcx, &*mir_for(tcx, def)?);
                m.mostly_independent(self.min_statements, self.min_share_percent)
                    .then_some((def, m))
            })
            .collect();
        if measured.is_empty() {
            return;
        }
        let census = census(tcx);
        let mut findings: Vec<Finding<'tcx>> = measured
            .into_iter()
            .filter_map(|(def, measure)| {
                let sets = census.concrete.get(&def).map_or(0, |s| s.len());
                (sets >= self.min_instantiations).then(|| Finding {
                    def,
                    measure,
                    sets,
                    site: census.first_site.get(&def).copied(),
                })
            })
            .collect();
        findings.sort_by_key(|f| tcx.def_span(f.def).lo());
        for f in findings {
            let name = tcx.def_path_str(f.def);
            let params = param_names(tcx, f.def);
            let (them, one_of_them) = match params.len() {
                1 => ("it", "it"),
                _ => ("them", "one of them"),
            };
            let msg = format!(
                "`{name}` is generic over {}, but {} of the {} statements in its body (as MIR, \
                 before optimization) do not depend on {them}, and this crate instantiates it \
                 with {} distinct sets of arguments, so that part is compiled {} times over",
                join(&params, "and"),
                f.measure.independent,
                f.measure.total,
                f.sets,
                f.sets,
            );
            let help = format!(
                "move the part that does not depend on {one_of_them} into a non-generic inner \
                 fn and keep `{name}` as a thin generic shim that converts its arguments and \
                 calls it, so the body is compiled once"
            );
            let span = tcx.def_span(f.def);
            match f.site {
                Some((site, args)) => emit_with_note(
                    cx,
                    GENERIC_BODY_NOT_GENERIC,
                    span,
                    msg,
                    site.source_callsite(),
                    format!(
                        "one of the {} instantiations, with {}",
                        f.sets,
                        render_args(tcx, f.def, args)
                    ),
                    help,
                ),
                None => emit(cx, GENERIC_BODY_NOT_GENERIC, span, msg, help),
            }
        }
    }
}
