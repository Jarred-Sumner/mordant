//! The new fn's parameters and result, refused unless the call still borrow-checks.

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Expr, ExprKind, HirId, Node, PatKind, QPath, StmtKind};
use rustc_index::bit_set::DenseBitSet;
use rustc_lint::LateContext;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor as MirVisitor};
use rustc_middle::mir::{
    self, BasicBlock, CastKind, InlineAsmOperand, Local, Location, Operand, Place, PlaceTy,
    ProjectionElem, RETURN_PLACE, Rvalue, StatementKind, TerminatorKind, VarDebugInfoContents,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, FnSig, GenericArgKind, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::symbol::kw;
use rustc_span::{BytePos, Span, Symbol};

use super::print::{anonymous, printed_ty};
use super::run::Run;
use crate::generic_body_not_generic::borrows::borrow_holders;
use crate::generic_body_not_generic::region::SharedPart;
use crate::generic_body_not_generic::region_io::{
    LocalUses, live_at_start, partly_moved, pointer_copies_of, read_after,
};
use crate::mir_flow::reaching;

/// `inputs` as the run first names them, with `mut` needs. `Lets`: `let`s read later, likewise.
pub(super) struct CallSignature {
    pub(super) inputs: Vec<(Symbol, bool, String)>,
    pub(super) result: CallResult,
}

pub(super) enum CallResult {
    Unit,
    Lets {
        names: Vec<(Symbol, bool)>,
        tys: Vec<String>,
    },
    Tail(Option<String>),
}

/// The user variable `local` holds: its name and the span of its binding pattern.
pub(super) fn user_var(body: &mir::Body<'_>, local: Local) -> Option<(Symbol, Span)> {
    body.var_debug_info
        .iter()
        .find_map(|info| match info.value {
            VarDebugInfoContents::Place(place)
                if place.as_local() == Some(local)
                    && info.composite.is_none()
                    && !info.source_info.span.from_expansion() =>
            {
                Some((info.name, info.source_info.span))
            }
            _ => None,
        })
}

/// Whether `blocks` assign to `local` or borrow it as `&mut`, directly or through a `Box`.
fn needs_mut<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
    local: Local,
) -> bool {
    struct Mutated<'a, 'tcx> {
        body: &'a mir::Body<'tcx>,
        root: Local,
        held: DenseBitSet<Local>,
        found: bool,
    }
    impl<'tcx> MirVisitor<'tcx> for Mutated<'_, 'tcx> {
        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _: Location) {
            let local = place.local;
            let own = if place.is_indirect() {
                local != self.root
                    && self.held.contains(local)
                    && !self.body.local_decls[local].ty.is_ref()
            } else {
                local == self.root
            };
            self.found |= own
                && matches!(context, PlaceContext::MutatingUse(by) if by != MutatingUseContext::Drop);
        }
    }

    let mut mutated = Mutated {
        body,
        root: local,
        held: pointer_copies_of(tcx, body, blocks, local),
        found: false,
    };
    for block in blocks.iter() {
        let data = &body.basic_blocks[block];
        if !data.is_cleanup {
            mutated.visit_basic_block_data(block, data);
        }
    }
    mutated.found
}

/// Whether every value `local: &T` (no lifetime in `T`) can hold borrows a local's memory or comes
/// from such a reference or an `anonymous` parameter, so an elided parameter lifetime suffices.
fn borrows_a_local<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    declared: FnSig<'tcx>,
    local: Local,
    seen: &mut DenseBitSet<Local>,
) -> bool {
    if !seen.insert(local) {
        return true;
    }
    let ty::Ref(_, inner, _) = *body.local_decls[local].ty.kind() else {
        return false;
    };
    if inner.walk().any(|arg| arg.as_region().is_some()) {
        return false;
    }
    if (1..=body.arg_count).contains(&local.as_usize()) {
        return matches!(
            declared.inputs().get(local.as_usize() - 1).map(|ty| ty.kind()),
            Some(&ty::Ref(region, _, _)) if anonymous(tcx, region)
        );
    }
    let from = |operand: &Operand<'tcx>, seen: &mut _| match operand {
        Operand::Copy(place) | Operand::Move(place) => place
            .as_local()
            .is_some_and(|source| borrows_a_local(tcx, body, declared, source, seen)),
        Operand::Constant(_) | Operand::RuntimeChecks(_) => false,
    };
    for data in body.basic_blocks.iter() {
        for statement in &data.statements {
            let StatementKind::Assign(assign) = &statement.kind else {
                continue;
            };
            let (place, rvalue) = &**assign;
            if place.local != local {
                continue;
            }
            let ok = place.projection.is_empty()
                && match rvalue {
                    Rvalue::Use(operand, _)
                    | Rvalue::Cast(
                        CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                        operand,
                        _,
                    ) => from(operand, seen),
                    Rvalue::CopyForDeref(source) => source
                        .as_local()
                        .is_some_and(|source| borrows_a_local(tcx, body, declared, source, seen)),
                    // A `Box` owns its memory. A reference local counts as what it borrows.
                    Rvalue::Ref(_, _, source) => {
                        let mut place_ty = PlaceTy::from_ty(body.local_decls[source.local].ty);
                        source.projection.iter().enumerate().all(|(depth, elem)| {
                            let owned = match elem {
                                ProjectionElem::Deref => match *place_ty.ty.kind() {
                                    ty::Adt(adt, _) => adt.is_box(),
                                    ty::Ref(..) if depth == 0 => {
                                        borrows_a_local(tcx, body, declared, source.local, seen)
                                    }
                                    _ => false,
                                },
                                _ => true,
                            };
                            place_ty = place_ty.projection_ty(tcx, elem);
                            owned
                        })
                    }
                    _ => false,
                };
            if !ok {
                return false;
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
            } if destination.local == local => {
                let Some(&ty::FnDef(callee, generic_args)) = func.constant().map(|c| c.ty().kind())
                else {
                    return false;
                };
                let sig = tcx
                    .fn_sig(callee)
                    .instantiate(tcx, generic_args)
                    .skip_normalization();
                let tied = matches!(*sig.output().skip_binder().kind(),
                    ty::Ref(region, _, _) if matches!(region.kind(), ty::ReBound(..)));
                if !destination.projection.is_empty()
                    || !tied
                    || sig.c_variadic()
                    || sig.inputs().skip_binder().len() != args.len()
                {
                    return false;
                }
                for (input, arg) in sig.inputs().skip_binder().iter().zip(args.iter()) {
                    if input.has_bound_regions() && !from(&arg.node, seen) {
                        return false;
                    }
                }
            }
            TerminatorKind::InlineAsm { operands, .. } => {
                let writes = |op: &InlineAsmOperand<'tcx>| {
                    matches!(op, InlineAsmOperand::Out { place: Some(place), .. }
                        | InlineAsmOperand::InOut { out_place: Some(place), .. }
                        if place.local == local)
                };
                if operands.iter().any(writes) {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

/// The new fn's parameters and result, or `None` unless each is a plainly named variable of a
/// printable type, and passing it as itself (not `&x`) borrow-checks and moves no drop that matters.
pub(super) fn call_signature<'tcx>(
    cx: &LateContext<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    part: &SharedPart,
    run: &Run<'tcx>,
) -> Option<CallSignature> {
    struct FirstUses<'tcx> {
        tcx: TyCtxt<'tcx>,
        run: Span,
        at: FxHashMap<HirId, BytePos>,
    }
    impl<'tcx> Visitor<'tcx> for FirstUses<'tcx> {
        type NestedFilter = rustc_middle::hir::nested_filter::OnlyBodies;
        fn maybe_tcx(&mut self) -> TyCtxt<'tcx> {
            self.tcx
        }
        fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
            if let ExprKind::Path(QPath::Resolved(None, path)) = expr.kind
                && let Res::Local(id) = path.res
                && self.run.contains(expr.span)
            {
                let at = self.at.entry(id).or_insert(expr.span.lo());
                *at = (*at).min(expr.span.lo());
            }
            intravisit::walk_expr(self, expr);
        }
    }

    let tcx = cx.tcx;
    let home = tcx.parent_module_from_def_id(def);
    let typing_env = body.typing_env(tcx);
    let lifetimes = |ty: Ty<'tcx>| {
        ty.walk()
            .filter(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(_)))
            .count()
    };
    let result_ty = |ty: Ty<'tcx>| {
        let region = ty.has_erased_regions() || ty.has_free_regions() || ty.has_bound_regions();
        if ty.is_never() || region {
            None
        } else {
            printed_ty(tcx, home, ty, false)
        }
    };

    let mut uses = FirstUses {
        tcx,
        run: run.span,
        at: FxHashMap::default(),
    };
    uses.visit_body(tcx.hir_body_owned_by(def));
    let declared = tcx.fn_sig(def).instantiate_identity().skip_normalization();
    let declared = tcx.liberate_late_bound_regions(def.to_def_id(), declared);
    let part_uses = LocalUses::of(tcx, body, &part.blocks);
    let mut outside = reaching(body, part.entry);
    outside.subtract(&part.blocks);
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup {
            outside.remove(block);
        }
    }
    let mut live = live_at_start(body, part.entry);
    for &local in &part.params {
        live.insert(local);
    }
    let mut inputs: Vec<(BytePos, Symbol, bool, String)> = Vec::new();
    let mut region_inputs: Vec<Local> = Vec::new();
    let (mut elided, mut unrelated) = (0, false);
    let (mut local_ref, mut ref_assigned) = (false, false);
    for &local in &part.params {
        let (name, span) = user_var(body, local)?;
        let ty = body.local_decls[local].ty;
        let (_, &first) = uses
            .at
            .iter()
            .find(|&(&id, _)| tcx.hir_span(id) == span && tcx.hir_name(id) == name)?;
        let copied = tcx.type_is_copy_modulo_regions(typing_env, ty);
        let caller_keeps = copied || matches!(ty.kind(), ty::Ref(..));
        if name == kw::SelfLower
            || span.hi() > run.span.lo()
            || inputs.iter().any(|input| input.1 == name)
            || (!caller_keeps && part.exit.is_some_and(|exit| read_after(body, exit, local)))
        {
            return None;
        }
        // Moved or reborrowed exclusively: E0505/E0502 if a borrow of it is still held then.
        if !copied {
            let mut of = DenseBitSet::new_empty(body.local_decls.len());
            of.insert(local);
            let holders = borrow_holders(tcx, body, &outside, &of);
            if holders.escaped || live.iter().any(|held| holders.any.contains(held)) {
                return None;
            }
        }
        // Moved into the new fn: its drop would run early, and all of it must still be there.
        if !caller_keeps
            && (ty.has_significant_drop(tcx, typing_env) || partly_moved(body, &outside, local))
        {
            return None;
        }
        // Two elided lifetimes are unrelated in the new signature: safe only on plain `&T`s.
        let regions = lifetimes(ty);
        if regions != 0 {
            region_inputs.push(local);
        }
        elided += regions;
        unrelated |= regions != 0
            && !matches!(*ty.kind(), ty::Ref(_, inner, by) if by.is_not() && lifetimes(inner) == 0);
        // A parameter's declared lifetimes print. A local's inferred ones need `borrows_a_local`.
        let is_param = (1..=body.arg_count).contains(&local.as_usize());
        let shown = if regions != 0 && is_param {
            *declared.inputs().get(local.as_usize() - 1)?
        } else {
            ty
        };
        if regions != 0 && !is_param {
            local_ref = true;
            let seen = &mut DenseBitSet::new_empty(body.local_decls.len());
            if !borrows_a_local(tcx, body, declared, local, seen) {
                return None;
            }
        }
        let as_mut = needs_mut(tcx, body, &part.blocks, local);
        ref_assigned |= regions != 0 && as_mut;
        inputs.push((first, name, as_mut, printed_ty(tcx, home, shown, true)?));
    }
    // A run that assigns to a reference it reads could store one argument's borrow in another.
    if (elided >= 2 && unrelated) || (local_ref && ref_assigned) {
        return None;
    }
    // A borrow of memory now in the new fn must not reach an argument with a lifetime.
    if !region_inputs.is_empty() {
        let holders = borrow_holders(tcx, body, &part.blocks, &part_uses.addressed);
        if holders.escaped
            || region_inputs
                .iter()
                .any(|&local| holders.any.contains(local))
        {
            return None;
        }
    }
    inputs.sort_by_key(|input| input.0);
    let inputs = inputs
        .into_iter()
        .map(|(_, name, as_mut, ty)| (name, as_mut, ty))
        .collect();

    let result = if run.with_tail {
        let ty = tcx.typeck(def).expr_ty_adjusted(run.block.expr?);
        let holder = match tcx.parent_hir_node(run.block.hir_id) {
            Node::Expr(value) => match tcx.parent_hir_node(value.hir_id) {
                Node::LetStmt(let_)
                    if matches!(let_.pat.kind, PatKind::Binding(..))
                        && let_.init.is_some_and(|init| init.hir_id == value.hir_id) =>
                {
                    Some(let_.pat.span)
                }
                _ => None,
            },
            _ => None,
        };
        let takes_value = |local| {
            local == RETURN_PLACE
                || user_var(body, local).is_some_and(|(_, span)| Some(span) == holder)
        };
        if part
            .returns
            .iter()
            .any(|&local| !takes_value(local) || body.local_decls[local].ty != ty)
        {
            return None;
        }
        CallResult::Tail(if ty.is_unit() {
            None
        } else {
            Some(result_ty(ty)?)
        })
    } else {
        let mut lets: Vec<(Symbol, Span)> = Vec::new();
        for stmt in &run.block.stmts[run.stmts.clone()] {
            if let StmtKind::Let(let_) = stmt.kind {
                let_.pat
                    .each_binding(|_, _, span, ident| lets.push((ident.name, span)));
            }
        }
        let mut outside = DenseBitSet::new_filled(body.basic_blocks.len());
        outside.subtract(&part.blocks);
        let mut read_later: Vec<(BytePos, Symbol, bool, String)> = Vec::new();
        for &local in &part.returns {
            let ty = body.local_decls[local].ty;
            if local == RETURN_PLACE && ty.is_unit() {
                continue;
            }
            let (name, span) = user_var(body, local)?;
            // Returned whole, so no field of it may be moved out in the run.
            if !lets.contains(&(name, span))
                || lets.iter().filter(|l| l.0 == name).count() != 1
                || partly_moved(body, &part.blocks, local)
            {
                return None;
            }
            let as_mut = needs_mut(tcx, body, &outside, local);
            read_later.push((span.lo(), name, as_mut, result_ty(ty)?));
        }
        read_later.sort_by_key(|out| out.0);
        if read_later.is_empty() {
            CallResult::Unit
        } else {
            CallResult::Lets {
                names: read_later.iter().map(|out| (out.1, out.2)).collect(),
                tys: read_later.into_iter().map(|out| out.3).collect(),
            }
        }
    };
    Some(CallSignature { inputs, result })
}
